#!/usr/bin/env python
# coding: utf-8
import torch
torch.set_default_dtype(torch.float)
import pyscf
import numpy as np
from ase.io import read
from dpyscfl.net import *
from dpyscfl.scf import *
from dpyscfl.utils import *
from dpyscfl.losses import *
from ase.units import Bohr, Hartree
import os, sys, argparse, psutil, pickle
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
parser.add_argument('--modelpath', metavar='modelpath', type=str, default='', help='Net Checkpoint location to continue training')
parser.add_argument('--writepath', action='store', default='.', help='where to write eval results')
parser.add_argument('--writeeach', action='store', default='', help='where to write results individually')
parser.add_argument('--writeref', action='store_true', default=False, help='write reference dictionaries')
parser.add_argument('--writepred', action='store_true', default=False, help='write prediction dictionaries')
parser.add_argument('--keeprho', action='store_true', default=False, help='whether to keep rho in matrix')
args = parser.parse_args()

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
skips = ['C5H8']
if __name__ == '__main__':
    if args.writeeach:
        try:
            os.mkdir(os.path.join(args.writepath, args.writeeach))
        except:
            pass
    print("READING TESTING TRAJECTORY.")
    atomsp = os.path.join(args.refpath, args.reftraj)
    atoms = read(atomsp, ':')
    e_refs = [a.calc.results['energy']/Hartree for a in atoms]
    indices = np.arange(len(atoms)).tolist()
    print("GENERATING SCF OBJECT")
    if args.pretrain_loc:
        scf = get_scf(args.type, pretrain_loc=args.pretrain_loc)
    elif args.modelpath:
        scf = get_scf(args.type, path=args.modelpath)
    else:
        scf = get_scf(args.type)
    scf.xc.evaluate()

    ref_dct = {'E':[], 'dm':[], 'mo_e':[]}
    pred_dct = {'E':[], 'dm':[], 'mo_e':[]}
    loss = torch.nn.MSELoss(reduction='mean')
#    loss_dct = {k: 0 for k,v in ref_dct.items()}
    loss_dct = {"E":0}
    fails = []
    grid_level = 1 if args.xc else 0
    for idx, atom in enumerate(atoms):
        formula = atom.get_chemical_formula()
        symbols = atom.symbols
        dmp = os.path.join(args.refpath, '{}_{}.dm.npy'.format(idx, symbols))
        dm_ref = np.load(dmp)
        e_ref = e_refs[idx]
        print("================= {}:    {} ======================".format(idx, formula))
        print("Getting Datapoint")
        if ('F3' in formula) or (formula in skips):
            fails.append((idx, formula))
            continue

        atom_E, _, atom_mats = old_get_datapoint(atoms=atom, xc=args.xc, grid_level=grid_level,
                                                basis=args.basis)
        if not args.keeprho:
            print('Purging rho')
            atom_mats.pop('rho', None)
        atom_mats = {k:[v] for k,v in atom_mats.items()}

        dset = Dataset(**atom_mats)
        dloader = torch.utils.data.DataLoader(dset, batch_size=1)

        #matrices.pop('dm_init') and matrices
        inputs = next(iter(dloader))
        print("CALCULATING PREDICTION")
        mixing = torch.rand(1)/2 + 0.5
        dm_mix = inputs[1]['dm_realinit']
        dm_in = inputs[0]*(1-mixing) + dm_mix*mixing
        results = scf_wrap(scf, dm_in, inputs[1], sc=True, molecule=atom.get_chemical_formula())
        if not results:
            fails.append((idx, formula))
            continue

        if args.writeeach:
            wep = os.path.join(args.writepath, args.writeeach)
            if args.writepred:
                predep = os.path.join(wep, '{}_{}.pckl'.format(idx, symbols))
                with open(predep, 'wb') as file:
                    file.write(pickle.dumps(results))

        ref_dct['E'].append(e_ref)
        ref_dct['dm'].append(dm_ref)
        ref_dct['mo_e'].append(atom_mats['mo_energy'][0])
        pred_dct['E'].append(results['E'][-1])
        pred_dct['dm'].append(results['dm'])
        pred_dct['mo_e'].append(results['mo_energy'])

        for key in loss_dct.keys():
            print(key)
            rd = torch.Tensor(ref_dct[key])
            pd = torch.Tensor(pred_dct[key])
            loss_dct[key] = loss(rd, pd)
        
        print("+++++++++++++++++++++++++++++++")
        print("RUNNING LOSS")
        print(loss_dct)
        print("+++++++++++++++++++++++++++++++")
    
    with open(args.writepath+'/loss_dct.pckl', 'wb') as file:
        file.write(pickle.dumps(loss_dct))
    with open(args.writepath+'/loss_dct.txt', 'w') as file:
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
        with open(args.writepath+'/pred_dct.pckl', 'wb') as file:
            file.write(pickle.dumps(pred_dct))
