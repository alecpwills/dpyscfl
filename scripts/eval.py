#!/usr/bin/env python
# coding: utf-8
import torch
torch.set_default_dtype(torch.float)
import pylibnxc
import numpy as np
from ase.io import read
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
parser.add_argument('--writeeach', action='store', default='', help='where to write results individually')
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

args = parser.parse_args()

scale = 1
if args.evtohart:
    scale = Hartree

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


def memory_watch(pid, idx, formula, fails, memoryMaxFrac=0.65):
    memoryMax = memoryMaxFrac*psutil.virtual_memory().total #total memory in bytes
    while True:
        process = psutil.Process(pid)
        processMemory = process.memory_info().rss #in bytes
        if processMemory > memoryMax:
            print("MEMORY MAX EXCEEDED. CONTINUING")
            fails.append((idx, formula, "MEMMAX"))
            break
    os.kill(pid, signal.SIGINT)


def scf_wrap(scf, dm_in, matrices, sc, molecule=''):
    try:
        results = scf(dm_in, matrices, sc)
    except:
        print("========================================================")
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print("SCF CALCULATION FAILED")
        print("SCF Calculation failed for {}, likely eigen-decomposition".format(molecule))
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print("========================================================")
        results = None
    return results

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    #elif hasattr(obj, '__dict__'):
    #    size += get_size(obj.__dict__, seen)
    #elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
    #    size += sum([get_size(i, seen) for i in obj])
    return size

#DESKTOP SKIPS
#skips = ['C5H8']
#SEAWULF SKIPS
skipidcs = args.skipidcs if args.skipidcs else []
skipforms = args.skipforms if args.skipforms else []

def eval_wrap(atomspath, refdatpath, modelpath, evalinds=[]):
    atoms = read(atomsp, ':')
    e_refs = [a.calc.results['energy']/scale for a in atoms]
    indices = np.arange(len(atoms)).tolist()
    ref_dct = {'E':[], 'dm':[], 'mo_e':[]}
    pred_dct = {'E':[], 'dm':[], 'mo_e':[]}
    loss = torch.nn.MSELoss()
    loss_dct = {"E":0}
    fails = []
    grid_level = 5 if args.xc else 0

    if not evalinds:
        evalinds = indices

    for idx, atom in enumerate(atoms):
        results = {}
        #manually skip for preservation of reference file lookups
        if idx not in evalinds:
            continue

        formula = atom.get_chemical_formula()
        symbols = atom.symbols
        print("================= {}:    {} ======================".format(idx, formula))
        print("Getting Datapoint")
        if (formula in skipforms) or (idx in skipidcs):
            print("SKIPPING")
            fails.append((idx, formula))
            continue
        name, mol = ase_atoms_to_mol(atom)
        _, method = gen_mf_mol(mol, xc='notnull', grid_level=grid_level, nxc=True)
        mf = KS(mol, method, model_path=model_path)
        mf.grids.level = grid_level
        mf.density_fit()
        mf.kernel()
        e_pred = mf.e_tot
        dm_pred = mf.make_rdm1()

        dmp = os.path.join(refdatpath, '{}_{}.dm.npy'.format(idx, symbols))
        dm_ref = np.load(dmp)
        e_ref = e_refs[idx]

        results['E'] = e_pred
        results['dm'] = dm_pred
        

        if args.writeeach:
            wep = os.path.join(args.writepath, args.writeeach)
            if args.writepred:
                predep = os.path.join(wep, '{}_{}.pckl'.format(idx, symbols))
                with open(predep, 'wb') as file:
                    file.write(pickle.dumps(results))

        ref_dct['E'].append(e_ref)
        ref_dct['dm'].append(dm_ref)

        pred_dct['E'].append(results['E'])
        pred_dct['dm'].append(results['dm'])

        for key in loss_dct.keys():
            print(key)
            rd = torch.Tensor(ref_dct[key])
            pd = torch.Tensor(pred_dct[key])
            loss_dct[key] = loss(rd, pd)
        
        print("+++++++++++++++++++++++++++++++")
        print("RUNNING LOSS")
        print(loss_dct)
        print("+++++++++++++++++++++++++++++++")
        




if __name__ == '__main__':
    if args.writeeach:
        try:
            os.mkdir(os.path.join(args.writepath, args.writeeach))
        except:
            pass
    print("READING TESTING TRAJECTORY.")
    atomsp = os.path.join(args.refpath, args.reftraj)
    atoms = read(atomsp, ':')
    e_refs = [a.calc.results['energy']/scale for a in atoms]
    indices = np.arange(len(atoms)).tolist()

    ref_dct = {'E':[], 'dm':[], 'mo_e':[]}
    pred_dct = {'E':[], 'dm':[], 'mo_e':[]}
    loss = torch.nn.MSELoss()
#    loss_dct = {k: 0 for k,v in ref_dct.items()}
    loss_dct = {"E":0}
    fails = []
    grid_level = 5 if args.xc else 0
    endidx = len(atoms) if args.endidx == -1 else args.endidx
    for idx, atom in enumerate(atoms):
        results = {}
        #manually skip for preservation of reference file lookups
        if idx < args.startidx:
            continue
        if idx > endidx:
            continue

        formula = atom.get_chemical_formula()
        symbols = atom.symbols
        print("================= {}:    {} ======================".format(idx, formula))
        print("Getting Datapoint")
        if (formula in skipforms) or (idx in skipidcs):
            print("SKIPPING")
            fails.append((idx, formula))
            continue
        name, mol = ase_atoms_to_mol(atom)
        _, method = gen_mf_mol(mol, xc='notnull', grid_level=args.gridlevel, nxc=True)
        mf = KS(mol, method, model_path=args.modelpath)
        mf.grids.level = args.gridlevel
        mf.density_fit()
        mf.max_cycle = args.maxcycle
        mf.kernel()
        e_pred = mf.e_tot
        dm_pred = mf.make_rdm1()

        dmp = os.path.join(args.refpath, '{}_{}.dm.npy'.format(idx, symbols))
        dm_ref = np.load(dmp)
        e_ref = e_refs[idx]

        results['E'] = e_pred
        results['dm'] = dm_pred
        

        if args.writeeach:
            wep = os.path.join(args.writepath, args.writeeach)
            if args.writepred:
                predep = os.path.join(wep, '{}_{}.pckl'.format(idx, symbols))
                with open(predep, 'wb') as file:
                    file.write(pickle.dumps(results))

        ref_dct['E'].append(e_ref)
        ref_dct['dm'].append(dm_ref)

        pred_dct['E'].append(results['E'])
        pred_dct['dm'].append(results['dm'])

        for key in loss_dct.keys():
            print(key)
            rd = torch.Tensor(ref_dct[key])
            pd = torch.Tensor(pred_dct[key])
            loss_dct[key] = loss(rd, pd)
        
        print("+++++++++++++++++++++++++++++++")
        print("RUNNING LOSS")
        print(loss_dct)
        print("+++++++++++++++++++++++++++++++")
    
    with open(args.writepath+'/loss_dct_{}.pckl'.format(args.type), 'wb') as file:
        file.write(pickle.dumps(loss_dct))
    with open(args.writepath+'/loss_dct_{}.txt'.format(args.type), 'w') as file:
        for k,v in loss_dct.items():
            file.write("{} {}\n".format(k,v))
    if fails:
        with open(args.writepath+'/fails.txt', 'w') as failfile:
            for idx, failed in fails.enumerate():
                failfile.write("{} {}\n".format(idx, failed))
    if args.writeref and not args.writeeach:
        with open(args.writepath+'/ref_dct.pckl', 'wb') as file:
            file.write(pickle.dumps(ref_dct))
    if args.writepred and not args.writeeach:
        with open(args.writepath+'/pred_dct_{}.pckl'.format(args.xctype), 'wb') as file:
            file.write(pickle.dumps(pred_dct))
