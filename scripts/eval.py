#!/usr/bin/env python
# coding: utf-8
import torch
torch.set_default_dtype(torch.float)
import pylibnxc
import numpy as np
from ase.io import read, write
from ase import Atoms
from dpyscfl.net import *
from dpyscfl.scf import *
from dpyscfl.utils import *
from dpyscfl.losses import *
from ase.units import Bohr, Hartree
import os, sys, argparse, psutil, pickle, threading, signal
from opt_einsum import contract

process = psutil.Process(os.getpid())
DEVICE = 'cpu'

parser = argparse.ArgumentParser(description='Evaluate xc functional')
parser.add_argument('--pretrain_loc', action='store', type=str, help='Location of pretrained models (should be directory containing x and c)')
parser.add_argument('--type', action='store', choices=['GGA','MGGA'])
parser.add_argument('--xc', action="store", default='', type=str, help='XC to use as reference evaluation')
parser.add_argument('--basis', metavar='basis', type=str, nargs = '?', default='6-311++G(3df,2pd)', help='basis to use. default 6-311++G(3df,2pd)')
parser.add_argument('--datapath', action='store', type=str, help='Location of precomputed matrices (run prep_data first)')
parser.add_argument('--refpath', action='store', type=str, help='Location of reference trajectories/DMs')
parser.add_argument('--reftraj', action='store', type=str, default="results.traj", help='File of reference trajectories')
parser.add_argument('--modelpath', metavar='modelpath', type=str, default='', help='Net Checkpoint location to use evaluating. Must be a directory containing "xc" network or "scf" network. Directory path must contain LDA, GGA, or MGGA.')
parser.add_argument('--writepath', action='store', default='.', help='where to write eval results')
parser.add_argument('--writeeach', action='store', default='preds', help='where to write results individually')
parser.add_argument('--writeref', action='store_true', default=False, help='write reference dictionaries')
parser.add_argument('--writepred', action='store_true', default=False, help='write prediction dictionaries')
parser.add_argument('--keeprho', action='store_true', default=False, help='whether to keep rho in matrix')
parser.add_argument('--startidx', action='store', default=0, type=int, help='Index in reference traj to start on.')
parser.add_argument('--endidx', action='store', default=-1, type=int, help='Index in reference traj to end on.')
parser.add_argument('--skipidcs', nargs='*', type=int, help="Indices to skip during evaluation. Space separated list of ints")
parser.add_argument('--skipforms', nargs='*', type=str, help='Formulas to skip during evaluation')
parser.add_argument('--memwatch', action='store_true', default=False, help='UNIMPLEMENTED YET')
parser.add_argument('--nowrapscf', action='store_true', default=False, help="Whether to wrap SCF calc in exception catcher")
parser.add_argument('--evtohart', action='store_true', default=False, help='If flagged, assumes read reference energies in eV and converts to Hartree')
parser.add_argument('--gridlevel', action='store', type=int, default=5, help='grid level')
parser.add_argument('--maxcycle', action='store', type=int, default=50, help='limit to scf cycles')
parser.add_argument('--atomization', action='store_true', default=False, help="If flagged, does atomization energies as well as total energies.")
parser.add_argument('--atmflip', action='store_true', default=False, help="If flagged, does reverses reference atomization energies sign")
parser.add_argument('--rho', action='store_true', default=False, help='If flagged, calculate rho loss')
parser.add_argument('--forceUKS', action='store_true', default=False, help='If flagged, force pyscf method to be UKS.')
parser.add_argument('--testgen', action='store_true', default=False, help='If flagged, only loops over trajectory to generate mols')
parser.add_argument('--noloss', action='store_true', default=False, help='If flagged, does not calculate losses with regards to reference trajectories, only predicts and saves prediction files.')
args = parser.parse_args()

scale = 1
if args.evtohart:
    scale = Hartree

def dm_to_rho(dm, ao_eval):
    if len(dm.shape) == 2:
        print("2D DM.")
        rho = contract('ij,ik,jk->i',
                           ao_eval, ao_eval, dm)
    else:
        print("NON-2D DM")
        rho = contract('ij,ik,xjk->xi',
                           ao_eval, ao_eval, dm)
    return rho

def rho_dev(dm, nelec, rho, rho_ref, grid_weights, mo_occ):
    mo_occ = torch.Tensor(mo_occ)
    if len(dm.shape) == 2:
        print("2D DM.")
        drho = torch.sqrt(torch.sum(torch.Tensor((rho-rho_ref)**2*grid_weights)/nelec**2))
        if torch.isnan(drho):
            print("NAN IN RHO LOSS. SETTING DRHO ZERO.")
            drho = torch.Tensor([0])
    else:
        print("NON-2D DM")
        if torch.sum(mo_occ) == 1:
            drho = torch.sqrt(torch.sum(torch.Tensor((rho[0]-rho_ref[0])**2*grid_weights)/torch.sum(mo_occ[0,0])**2))
        else:
            drho = torch.sqrt(torch.sum(torch.Tensor((rho[0]-rho_ref[0])**2*grid_weights))/torch.sum(mo_occ[0,0])**2 +\
                   torch.sum(torch.Tensor((rho[1]-rho_ref[1])**2*grid_weights))/torch.sum(mo_occ[0,1])**2)
        if torch.isnan(drho):
            print("NAN IN RHO LOSS. SETTING DRHO ZERO.")
            drho = torch.Tensor([0])
    return drho


def KS(mol, method, model_path='', nxc_kind='grid', **kwargs):
    """ Wrapper for the pyscf RKS and UKS class
    that uses a libnxc functionals
    """
    #hyb = kwargs.get('hyb', 0)
    mf = method(mol, **kwargs)
    if model_path != '':
        if nxc_kind.lower() == 'atomic':
            model = get_nxc_adapter('pyscf', model_path)
            mf.get_veff = veff_mod_atomic(mf, model)
        elif nxc_kind.lower() == 'grid':
            parsed_xc = pylibnxc.pyscf.utils.parse_xc_code(model_path)
            dft.libxc.define_xc_(mf._numint,
                                 eval_xc,
                                 pylibnxc.pyscf.utils.find_max_level(parsed_xc),
                                 hyb=parsed_xc[0][0])
            mf.xc = model_path
        else:
            raise ValueError(
                "{} not a valid nxc_kind. Valid options are 'atomic' or 'grid'"
                .format(nxc_kind))
    return mf


def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    """ Evaluation for grid-based models (not atomic)
        See pyscf documentation of eval_xc
    """
    inp = {}
    if spin == 0:
        if rho.ndim == 1:
            rho = rho.reshape(1, -1)
        inp['rho'] = rho[0]
        if len(rho) > 1:
            dx, dy, dz = rho[1:4]
            gamma = (dx**2 + dy**2 + dz**2)
            inp['sigma'] = gamma
        if len(rho) > 4:
            inp['lapl'] = rho[4]
            inp['tau'] = rho[5]
    else:
        rho_a, rho_b = rho
        if rho_a.ndim == 1:
            rho_a = rho_a.reshape(1, -1)
            rho_b = rho_b.reshape(1, -1)
        inp['rho'] = np.stack([rho_a[0], rho_b[0]])
        if len(rho_a) > 1:
            dxa, dya, dza = rho_a[1:4]
            dxb, dyb, dzb = rho_b[1:4]
            gamma_a = (dxa**2 + dya**2 + dza**2)  #compute contracted gradients
            gamma_b = (dxb**2 + dyb**2 + dzb**2)
            gamma_ab = (dxb * dxa + dyb * dya + dzb * dza)
            inp['sigma'] = np.stack([gamma_a, gamma_ab, gamma_b])
        if len(rho_a) > 4:
            inp['lapl'] = np.stack([rho_a[4], rho_b[4]])
            inp['tau'] = np.stack([rho_a[5], rho_b[5]])

    parsed_xc = pylibnxc.pyscf.utils.parse_xc_code(xc_code)
    total_output = {'v' + key: 0.0 for key in inp}
    total_output['zk'] = 0
    #print(parsed_xc)
    for code, factor in parsed_xc[1]:
        model = pylibnxc.LibNXCFunctional(xc_code, kind='grid')
        output = model.compute(inp)
        for key in output:
            if output[key] is not None:
                total_output[key] += output[key] * factor

    exc, vlapl, vtau, vrho, vsigma = [total_output.get(key,None)\
      for key in ['zk','vlapl','vtau','vrho','vsigma']]

    vxc = (vrho, vsigma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc



skipidcs = args.skipidcs if args.skipidcs else []
skipforms = args.skipforms if args.skipforms else []

#spins for single atoms, since pyscf doesn't guess this correctly.
spins_dict = {
    'Al': 1,
    'B' : 1,
    'Li': 1,
    'Na': 1,
    'Si': 2 ,
    'Be':0,
    'C': 2,
    'Cl': 1,
    'F': 1,
    'H': 1,
    'N': 3,
    'O': 2,
    'P': 3,
    'S': 2
}

def get_spin(at):
    #if single atom and spin is not specified in at.info dictionary, use spins_dict
    print('======================')
    print("GET SPIN: Atoms Info")
    print(at)
    print(at.info)
    print('======================')
    if ( (len(at.positions) == 1) and not ('spin' in at.info) ):
        print("Single atom and no spin specified in at.info")
        spin = spins_dict[str(at.symbols)]
    else:
        print("Not a single atom, or spin in at.info")
        if type(at.info.get('spin', None)) == type(0):
            #integer specified in at.info['spin'], so use it
            print('Spin specified in atom info.')
            spin = at.info['spin']
        elif 'radical' in at.info.get('name', ''):
            print('Radical specified in atom.info["name"], assuming spin 1.')
            spin = 1
        elif at.info.get('openshell', None):
            print("Openshell specified in atom info, attempting spin 2.")
            spin = 2
        else:
            print("No specifications in atom info to help, assuming no spin.")
            spin = 0
    return spin

if __name__ == '__main__':
    with open('unconv','w') as ucfile:
        ucfile.write('#idx\tatoms.symbols\txc_fail\txc_fail_en\txc_bckp\txc_bckp_en\n')

    if args.writeeach:
        try:
            os.mkdir(os.path.join(args.writepath, args.writeeach))
        except:
            pass
    
    atomsp = os.path.join(args.refpath, args.reftraj)
    print("READING TESTING TRAJECTORY: {}".format(atomsp))
    atoms = read(atomsp, ':')
    e_refs = []
    for idx, at in enumerate(atoms):
        print("Getting energies -- {}: {}".format(idx, at.get_chemical_formula()))
        try:
            if len(at.positions) > 1:
                e_refs.append(at.info['energy']/scale)
            else:
                e_refs.append(at.calc.results['energy'])
        except:
            e_refs.append(at.info['energy']/scale)

    e_refs = [a.info['energy']/scale for a in atoms]
    indices = np.arange(len(atoms)).tolist()
    
    if args.atomization:
        #try to read previous calc
        #temp fix, negative sign because that's what the training assumes
        #or if already flipped, don't flip
        if args.atmflip:
            mult = -1
        else:
            mult = 1
        ref_atm = [mult*a.info.get('atomization', 0)/scale for a in atoms]
        try:
            with open('atomicen.pkl', 'rb') as f:
                atomic_e = pickle.load(f)
        except:
            print("ATOMIZATION ENERGY FLAGGED -- CALCULATING SINGLE ATOM ENERGIES")
            atomic_set = []
            for at in atoms:
                atomic_set += at.get_chemical_symbols()
            atomic_set = list(set(atomic_set))
            for s in atomic_set:
                assert s in list(spins_dict.keys()), "{}: Atom in dataset not present in spins dictionary.".format(s)
            atomic_e = {s:0 for s in atomic_set}
            atomic = [Atoms(symbols=s) for s in atomic_set]
            #generates pyscf mol, default basis 6-311++G(3df,2pd), charge=0, spin=None
            atomic_mol = [ase_atoms_to_mol(at, basis=args.basis, spin=get_spin(at), charge=0) for at in atomic]
            if args.forceUKS:
                ipol = True
            else:
                ipol = False
            atomic_method = [gen_mf_mol(mol[1], xc='notnull', pol=ipol, grid_level=args.gridlevel, nxc=True) for mol in atomic_mol]
            for idx, methodtup in enumerate(atomic_method):
                print(idx, methodtup)
                name, mol = atomic_mol[idx]
                method = methodtup[1]
                if args.modelpath:
                    mf = KS(mol, method, model_path=args.modelpath)
                else:
                    mf = method(mol)
                    mf.xc = args.xc
                    
                mf.grids.level = args.gridlevel
                mf.max_cycle = args.maxcycle
                #mf.density_fit()
                mf.kernel()
                
                if not args.modelpath:
                    #if straight pyscf calc, and not converged, try again as PBE start
                    if not mf.converged:
                        print("Calculation did not converge. Trying second order convergence with PBE to feed into calculation.")
                        mfp = method(mol, xc='pbe').newton()
                        mfp.kernel()
                        print("PBE Calculation complete -- feeding into original kernel.")
                        mf.kernel(dm0 = mfp.make_rdm1())
                        if not mf.converged:
                            print("Convergence still failed -- {}".format(atomic[idx]))
                            with open('unconv', 'a') as ucfile:
                                ucfile.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, atomic[idx], args.xc, mf.e_tot, 'pbe', mfp.e_tot))
                            #overwrite to just use pbe energy.
                            mf = mfp
                            

                
                atomic_e[name] = mf.e_tot
                print("++++++++++++++++++++++++")
                print("Atomic Energy: {} -- {}".format(name, mf.e_tot))
                print("++++++++++++++++++++++++")
            with open('atomicen.dat', 'w') as f:
                f.write("#Atom\tEnergy (Hartree)\n")
                for k, v in atomic_e.items():
                    f.write('{}\t{}\n'.format(k, v))
                    print('{}\t{}'.format(k,v))
            with open('atomicen.pkl', 'wb') as f:
                pickle.dump(atomic_e, f)


    ref_dct = {'E':[], 'dm':[], 'mo_e':[]}
    pred_dct = {'E':[], 'dm':[], 'mo_e':[]}
    pred_e = {idx:0 for idx in range(len(atoms))}
    pred_dm = {idx:0 for idx in range(len(atoms))}
    ao_evals = {idx:0 for idx in range(len(atoms))}
    mo_occs = {idx:0 for idx in range(len(atoms))}
    mfs = {idx:0 for idx in range(len(atoms))}
    nelecs = {idx:0 for idx in range(len(atoms))}
    gweights = {idx:0 for idx in range(len(atoms))}
    loss = torch.nn.MSELoss()
#    loss_dct = {k: 0 for k,v in ref_dct.items()}
    loss_dct = {"E":0, 'rho':0}
    rho_errs = {'rho':[]}
    if args.atomization:
        pred_atm = {idx:0 for idx in range(len(atoms))}
        ref_dct['atm'] = []
        pred_dct['atm'] = []
        loss_dct['atm'] = 0

    fails = []
    grid_level = 5 if args.xc else 0
    endidx = len(atoms) if args.endidx == -1 else args.endidx
    for idx, atom in enumerate(atoms):
        formula = atom.get_chemical_formula()
        symbols = atom.symbols
        if idx < args.startidx:
            continue
        if idx > endidx:
            continue

        try:
            wep = os.path.join(args.writepath, args.writeeach)
            if args.writepred:
                print("Attempting read in of previous results.\n{}".format(formula))
                predep = os.path.join(wep, '{}_{}.pckl'.format(idx, symbols))
                with open(predep, 'rb') as file:
                    results = pickle.load(file)
                e_pred = results['E']
                dm_pred = results['dm']
                pred_e[idx] = [formula, e_pred]
                pred_dm[idx] = [formula, dm_pred]
                ao_evals[idx] = results['ao_eval']
                mo_occs[idx] = results['mo_occ']
                #mfs[idx] = results['mf']
                nelecs[idx] = results['nelec']
                gweights[idx] = results['gweights']


                print("Results found for {} {}".format(idx, symbols))
            else:
                raise ValueError
        except FileNotFoundError:
            print("No previous results found. Generating new data.")
            results = {}
            #manually skip for preservation of reference file lookups

            print("================= {}:    {} ======================".format(idx, formula))
            print("Getting Datapoint")
            if (formula in skipforms) or (idx in skipidcs):
                print("SKIPPING")
                fails.append((idx, formula))
                continue
            molgen = False
            scount = 0
            while not molgen:
                try:
                    name, mol = ase_atoms_to_mol(atom, basis=args.basis, spin=get_spin(atom)-scount, charge=0)
                    molgen=True
                except RuntimeError:
                    #spin disparity somehow, try with one less until 0
                    print("RuntimeError. Trying with reduced spin.")
                    spin = get_spin(atom)
                    spin = spin - scount - 1
                    scount += 1
                    if spin < 0:
                        raise ValueError
            if args.testgen:
                continue
            if args.forceUKS:
                ipol = True
            else:
                ipol = False
            _, method = gen_mf_mol(mol, xc='notnull', pol=ipol, grid_level=args.gridlevel, nxc=True)
            if args.modelpath:
                mf = KS(mol, method, model_path=args.modelpath)
            else:
                mf = method(mol)
                mf.xc = args.xc
            mf.grids.level = args.gridlevel
            mf.grids.build()
            #mf.density_fit()
            mf.max_cycle = args.maxcycle
            mf.kernel()

            if not args.modelpath:
                #if straight pyscf calc, and not converged, try again as PBE start
                if not mf.converged:
                    print("Calculation did not converge. Trying second order convergence with PBE to feed into calculation.")
                    mfp = method(mol, xc='pbe').newton()
                    mfp.kernel()
                    print("PBE Calculation complete -- feeding into original kernel.")
                    mf.kernel(dm0 = mfp.make_rdm1())
                    if not mf.converged:
                        print("Convergence still failed -- {}".format(atom.symbols))
                        #overwrite to just use pbe energy.
                        with open('unconv', 'a') as ucfile:
                            ucfile.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, atom.symbols, args.xc, mf.e_tot, 'pbe', mfp.e_tot))

                        mf = mfp

            ao_eval = mf._numint.eval_ao(mol, mf.grids.coords)
            e_pred = mf.e_tot
            dm_pred = mf.make_rdm1()
            rho_pred = dm_to_rho(dm_pred, ao_eval)
            pred_e[idx] = [formula, symbols, e_pred]
            pred_dm[idx] = [formula, symbols, dm_pred]
            ao_evals[idx] = ao_eval
            mo_occs[idx] = mf.mo_occ
            mfs[idx] = mf
            nelecs[idx] = mol.nelectron
            gweights[idx] = mf.grids.weights


            #updating and writing of new information in atom's info dict
            atom.info['e_pred'] = e_pred
            write(os.path.join(wep, '{}_{}.traj'.format(idx, symbols)), atom)

            results['E'] = e_pred
            results['dm'] = dm_pred
            np.save(os.path.join(wep, '{}_{}.dm.npy'.format(idx, symbols)), dm_pred)
            results['ao_eval'] = ao_eval
            results['mo_occ'] = mf.mo_occ
            #results['mf'] = mf
            results['nelec'] = mol.nelectron
            results['gweights'] = mf.grids.weights

        write(os.path.join(wep, 'predictions.traj'), atoms)
    
        rho_pred = dm_to_rho(pred_dm[idx][1], ao_evals[idx])
        if not args.noloss:
            dmp = os.path.join(args.refpath, '{}_{}.dm.npy'.format(idx, symbols))
            dm_ref = np.load(dmp)
            rho_ref = dm_to_rho(dm_ref, ao_evals[idx])
            rho_err = rho_dev(pred_dm[idx][1], nelecs[idx], rho_pred, rho_ref, gweights[idx], mo_occs[idx])
            rho_errs['rho'].append(rho_err)
            print("Rho Error: ", rho_err)

        e_ref = e_refs[idx]        
        if args.atomization and not args.noloss:
            start = e_pred
            subs = atom.get_chemical_symbols()
            if len(subs) == 1:
                #single atom, no atomization needed
                print("SINGLE ATOM -- NO ATOMIZATION CALCULATION.")
                results['atm'] = np.nan
                pred_atm[idx] = [formula, symbols, results['atm']]
            else:
                print("{} ({}) decomposition -> {}".format(formula, start, subs))
                for s in subs:
                    print("{} - {} ::: {} - {} = {}".format(formula, s, start, atomic_e[s], start - atomic_e[s]))
                    start -= atomic_e[s]
                print("Predicted Atomization Energy for {} : {}".format(formula, start))
                print("Reference Atomization Energy for {} : {}".format(formula, ref_atm[idx]))
                results['atm'] = start
                pred_atm[idx] = [formula, symbols, results['atm']]
                ref_dct['atm'].append(ref_atm[idx])
                pred_dct['atm'].append(results['atm'])
                print("Error: {}".format(start - ref_atm[idx]))

        

        if args.writeeach:
            wep = os.path.join(args.writepath, args.writeeach)
            if args.writepred:
                predep = os.path.join(wep, '{}_{}.pckl'.format(idx, symbols))
                with open(predep, 'wb') as file:
                    file.write(pickle.dumps(results))

        if not args.noloss:
            ref_dct['E'].append(e_ref)
            ref_dct['dm'].append(dm_ref)

            pred_dct['E'].append(results['E'])
            pred_dct['dm'].append(results['dm'])

            print("Predicted Total Energy for {} : {}".format(formula, results['E']))
            print("Reference Total Energy for {} : {}".format(formula, e_ref))
            print("Error: {}".format(results['E'] - e_ref))


            for key in loss_dct.keys():
                print(key)
                if key == 'rho':
                    rd = torch.zeros_like(torch.Tensor(rho_errs['rho']))
                    pd = torch.Tensor(rho_errs['rho'])
                else:
                    rd = torch.Tensor(ref_dct[key])
                    pd = torch.Tensor(pred_dct[key])
                loss_dct[key] = loss(rd, pd)
            
            loss_dct['rho'] = rho_err

            print("+++++++++++++++++++++++++++++++")
            print("RUNNING LOSS")
            print(loss_dct)
            print("+++++++++++++++++++++++++++++++")

            if args.atomization:
                writelab = '#Index\tAtomForm\tAtomSymb\tEPred (H)\tERef (H)\tEErr (H)\tRhoErr\tEPAtm (H)\tERAtm (H)\tEAErr (H)\n'
                writestr = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, formula, symbols, \
                    results['E'], e_ref, results['E']-e_ref, rho_err, results['atm'], ref_atm[idx], results['atm'] - ref_atm[idx])
            else:
                writelab = '#Index\tAtom\tEPred (H)\tERef (H)\tEErr (H)\tRhoErr\n'
                writestr = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, formula, \
                    results['E'], e_ref, results['E']-e_ref, rho_err)
            if idx == 0:
                with open(args.writepath+'/table.dat', 'w') as f:
                    f.write(writelab)
                    f.write(writestr)
            else:
                with open(args.writepath+'/table.dat', 'a') as f:
                    f.write(writestr)


    with open(args.writepath+'/pred_e.dat', 'w') as f:
        f.write("#Index\tAtomForm\tAtomSymb\tEnergy (Hartree)\n")
        ks = sorted(list(pred_e.keys()))
        for idx, k in enumerate(pred_e):
                v = pred_e[k]
                f.write("{}\t{}\t{}\t{}\n".format(k, v[0], v[1]. v[2]))
    if args.atomization:
        with open(args.writepath+'/pred_atm.dat', 'w') as f:
            f.write("#Index\tAtomForm\tAtomSymb\tAtomization Energy (Hartree)\n")
            ks = sorted(list(pred_atm.keys()))
            for k in ks:
                v = pred_atm[k]
                f.write("{}\t{}\t{}\t{}\n".format(k, v[0], v[1]. v[2]))
    if not args.noloss:
        with open(args.writepath+'/loss_dct_{}.pckl'.format(args.type), 'wb') as file:
            file.write(pickle.dumps(loss_dct))
        with open(args.writepath+'/loss_dct_{}.txt'.format(args.type), 'w') as file:
            for k,v in loss_dct.items():
                file.write("{} {}\n".format(k,v))
        if args.writeref and not args.writeeach:
            with open(args.writepath+'/ref_dct.pckl', 'wb') as file:
                file.write(pickle.dumps(ref_dct))

    if fails:
        with open(args.writepath+'/fails.txt', 'w') as failfile:
            for idx, failed in fails.enumerate():
                failfile.write("{} {}\n".format(idx, failed))
    if args.writepred and not args.writeeach:
        with open(args.writepath+'/pred_dct_{}.pckl'.format(args.xctype), 'wb') as file:
            file.write(pickle.dumps(pred_dct))
