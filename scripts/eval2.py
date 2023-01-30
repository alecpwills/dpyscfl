#!/usr/bin/env python
# coding: utf-8
import torch
torch.set_default_dtype(torch.float)
from pylibnxc.pyscf import RKS, UKS
import numpy as np
from time import time
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
parser.add_argument('--trajpath', action='store', type=str, default="results.traj", help='File of reference trajectories')
parser.add_argument('--modelpath', metavar='modelpath', type=str, default='', help='Net Checkpoint location to use evaluating. Must be a directory containing "xc" network or "scf" network. Directory path must contain LDA, GGA, or MGGA.')
parser.add_argument('--writepath', action='store', default='.', help='where to write eval results')
parser.add_argument('--startidx', action='store', default=0, type=int, help='Index in reference traj to start on.')
parser.add_argument('--endidx', action='store', default=-1, type=int, help='Index in reference traj to end on.')
parser.add_argument('--skipidcs', nargs='*', type=int, help="Indices to skip during evaluation. Space separated list of ints")
parser.add_argument('--skipforms', nargs='*', type=str, help='Formulas to skip during evaluation')
parser.add_argument('--evtohart', action='store_true', default=False, help='If flagged, assumes read reference energies in eV and converts to Hartree')
parser.add_argument('--gridlevel', action='store', type=int, default=5, help='grid level')
parser.add_argument('--maxcycle', action='store', type=int, default=50, help='limit to scf cycles')
parser.add_argument('--atomization', action='store_true', default=False, help="If flagged, does atomization energies as well as total energies.")
parser.add_argument('--atmflip', action='store_true', default=False, help="If flagged, does reverses reference atomization energies sign")
parser.add_argument('--forceUKS', action='store_true', default=False, help='If flagged, force pyscf method to be UKS.')
parser.add_argument('--testgen', action='store_true', default=False, help='If flagged, only loops over trajectory to generate mols')
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
    'S': 2,
    'Ar':0, #noble
    'Br':1, #one unpaired electron
    'Ne':0, #noble
    'Sb':3, #same column as N/P
    'Bi':3, #same column as N/P/Sb
    'Te':2, #same column as O/S
    'I':1 #one unpaired electron
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
    #script start time
    start = time()
    times_dct = {'start':start}
    #Pylibnxc uses (M)GGA_XC_CUSTOM flag to look for custom model in current working directory, so if modelpath is flagged, create the directory and symlink
    if args.modelpath:
        modtype = args.type.upper()
        xcp = '{}_XC_CUSTOM'.format(modtype)
        try:
            print('Attempting directory creation...')
            os.mkdir(xcp)
        except:
            print('os.mkdir failed, directory likely exists.')
            pass
        
        try:
            print('Symlinking {} to {}'.format(args.modelpath, os.path.join(xcp, 'xc')))
            os.symlink(os.path.abspath(args.modelpath), os.path.join(xcp, 'xc'))
        except:
            print('Symlink failed. Another model might be symlinked already.')
            pass
            

    with open('unconv','w') as ucfile:
        ucfile.write('#idx\tatoms.symbols\txc_fail\txc_fail_en\txc_bckp\txc_bckp_en\n')

    try:
        os.mkdir(os.path.join(args.writepath, 'preds'))
    except:
        pass
    
    print("READING TESTING TRAJECTORY: {}".format(args.trajpath))
    atoms = read(args.trajpath, ':')
    indices = np.arange(len(atoms)).tolist()
    
    if args.atomization:
        if args.forceUKS:
            ipol = True
            KS = UKS
        else:
            ipol = False
            KS = RKS

        #try to read previous calc
        try:
            with open('atomicen.pkl', 'rb') as f:
                atomic_e = pickle.load(f)
            atomic_end = time()
        #if not found, generate the data
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
            atomic_method = [gen_mf_mol(mol[1], xc='notnull', pol=ipol, grid_level=args.gridlevel, nxc=True) for mol in atomic_mol]
            #begin atomic calculations
            atomic_start = time() - start
            for idx, methodtup in enumerate(atomic_method):
                print(idx, methodtup)

                name, mol = atomic_mol[idx]
                method = methodtup[1]
                this_mol_time_start = time()
                times_dct[idx] = [name]
                times_dct[idx].append(this_mol_time_start)


                if args.modelpath:
                    mf = KS(mol, nxc=xcp, nxc_kind='grid')
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
                this_mol_time_end = time() - this_mol_time_start
                times_dct[idx].append(this_mol_time_end)
                print("++++++++++++++++++++++++")
                print("Atomic Energy: {} -- {}".format(name, mf.e_tot))
                print("++++++++++++++++++++++++")
                print("Time to Calculate: {} -- {}".format(name, this_mol_time_end))
                print("++++++++++++++++++++++++")
            atomic_end = time() - atomic_start
            times_dct['end'] = atomic_end
            with open('atomicen.dat', 'w') as f:
                f.write("#Atom\tEnergy (Hartree)\n")
                for k, v in atomic_e.items():
                    f.write('{}\t{}\n'.format(k, v))
                    print('{}\t{}'.format(k,v))
            with open('atomicen.pkl', 'wb') as f:
                pickle.dump(atomic_e, f)
            with open('atomic_times.dat', 'w') as f:
                f.write('#Atomic Energies Start: {}\n'.format(times_dct['start']))
                f.write('#Atomic Energies End: {}\n'.format(times_dct['end']))
                f.write('#key/index\tName\tAtStart\tAtEnd\n')
                for k,v in times_dct.items():
                    print(k, v)
                    if k in ['start', 'end']:
                        continue
                    f.write('{}\t{}\t{}\t{}\n'.format(k, v[0], v[1], v[2]))
                    


    pred_dct = {'E':[], 'dm':[], 'mo_e':[]}
    pred_e = {idx:0 for idx in range(len(atoms))}
    pred_dm = {idx:0 for idx in range(len(atoms))}
    ao_evals = {idx:0 for idx in range(len(atoms))}
    mo_occs = {idx:0 for idx in range(len(atoms))}
    mfs = {idx:0 for idx in range(len(atoms))}
    nelecs = {idx:0 for idx in range(len(atoms))}
    gweights = {idx:0 for idx in range(len(atoms))}

    if args.atomization:
        pred_atm = {idx:0 for idx in range(len(atoms))}
        pred_dct['atm'] = []

    fails = []
    grid_level = 5 if args.xc else 3
    endidx = len(atoms) if args.endidx == -1 else args.endidx
    molecule_start = time()
    times_dct = {'start':molecule_start}
    for idx, atom in enumerate(atoms):
        this_mol_time_start = time()
        formula = atom.get_chemical_formula()
        symbols = atom.symbols
        times_dct[idx] = [formula, symbols]
        times_dct[idx].append(this_mol_time_start)

        if idx < args.startidx:
            continue
        if idx > endidx:
            continue
        #try reading saved results
        try:
            wep = os.path.join(args.writepath, 'preds')
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
                KS = UKS
            else:
                ipol = False
                KS = RKS
            _, method = gen_mf_mol(mol, xc='notnull', pol=ipol, grid_level=args.gridlevel, nxc=True)
            if args.modelpath:
                mf = KS(mol, nxc=xcp, nxc_kind='grid')
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
            

            this_mol_time_end = time() - this_mol_time_start
            times_dct[idx].append(this_mol_time_end)
            ao_eval = mf._numint.eval_ao(mol, mf.grids.coords)
            e_pred = mf.e_tot
            dm_pred = mf.make_rdm1()
            pred_e[idx] = [formula, symbols, e_pred]

            print("++++++++++++++++++++++++")
            print("Molecular Energy: {}/{} -- {}".format(formula, symbols, mf.e_tot))
            print("++++++++++++++++++++++++")
            print("Time to Calculate: {}/{} -- {}".format(formula, symbols, this_mol_time_end))
            print("++++++++++++++++++++++++")

            #updating and writing of new information in atom's info dict
            atom.info['e_pred'] = e_pred
            write(os.path.join(wep, '{}_{}.traj'.format(idx, symbols)), atom)

            results['E'] = e_pred
            np.save(os.path.join(wep, '{}_{}.dm.npy'.format(idx, symbols)), dm_pred)

    write(os.path.join(wep, 'predictions.traj'), atoms)

    molecule_end = time() - molecule_start
    times_dct['end'] = molecule_end
    with open('molecule_times.dat', 'w') as f:
        f.write('#Molecule Energies Start: {}\n'.format(times_dct['start']))
        f.write('#Molecule Energies End: {}\n'.format(times_dct['end']))
        f.write('#key/index\tFormula\tSymbols\tMolStart\tMolEnd\n')
        for k,v in times_dct.items():
            print(k,v)
            if k in ['start', 'end']:
                continue
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(k, v[0], v[1], v[2], v[3]))

    with open(args.writepath+'/pred_e.dat', 'w') as f:
        f.write("#Index\tAtomForm\tAtomSymb\tPredicted Energy (Hartree)\n")
        ks = sorted(list(pred_e.keys()))
        for idx, k in enumerate(pred_e):
                v = pred_e[k]
                f.write("{}\t{}\t{}\t{}\n".format(k, v[0], v[1], v[2]))
    if fails:
        with open(args.writepath+'/fails.txt', 'w') as failfile:
            for idx, failed in fails.enumerate():
                failfile.write("{} {}\n".format(idx, failed))
