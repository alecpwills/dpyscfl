#!/usr/bin/env python
# coding: utf-8
import pyscf
from pyscf import gto,dft,scf
import torch
torch.set_default_dtype(torch.double)
#import pyscf
#from pyscf import gto,dft,scf

import numpy as np
import scipy
from ase import Atoms
from ase.io import read
from dpyscf.net import *
from dpyscf.torch_routines import *
from dpyscf.utils import *
from dpyscf.losses import *
from pyscf.cc import CCSD
from functools import partial
from ase.units import Bohr
from datetime import datetime
import sys
import shutil
import os
import psutil
import gc
import tarfile
import argparse
import json

process = psutil.Process(os.getpid())
dpyscf_dir = os.environ.get('DPYSCF_DIR','..')
DEVICE = 'cpu'

parser = argparse.ArgumentParser(description='Train xc functional')
parser.add_argument('pretrain_loc', action='store', type=str, help='Location of pretrained models (should be directory containing x and c)')
parser.add_argument('type', action='store', choices=['GGA','MGGA'])
parser.add_argument('datapath', action='store', type=str, help='Location of precomputed matrices (run prep_data first)')
parser.add_argument('reftraj', action='store', type=str, help='Location of reference trajectories')
parser.add_argument('--n_hidden', metavar='n_hidden', type=int, default=16, help='Number of hidden nodes (16)')
parser.add_argument('--hyb_par', metavar='hyb_par', type=float, default=0.0, help='Hybrid mixing parameter (0.0)')
parser.add_argument('--E_weight', metavar='e_weight', type=float, default=0.0, help='Weight of total energy term in loss function (0)')
parser.add_argument('--rho_weight', metavar='rho_weight', type=float, default=25, help='Weight of density term in loss function (25)')
parser.add_argument('--modelpath', metavar='modelpath', type=str, default='', help='Checkpoint location to continue training')
parser.add_argument('--logpath', metavar='logpath', type=str, default='log/', help='Logging directory (log/)')
parser.add_argument('--testrun', action='store_true', help='Do a test run over all molecules before training')
parser.add_argument('--lr', metavar='lr', type=float, default=0.0001, help='Learning rate (0.0001)')
parser.add_argument('--l2', metavar='l2', type=float, default=1e-8, help='Weight decay (1e-8)')
parser.add_argument('--hnorm', action='store_true', help='Use H energy and density in loss')
parser.add_argument('--print_stdout', action='store_true', help='Print to stdout instead of logfile')
parser.add_argument('--print_names', action='store_true', help='Print molecule names during training')
parser.add_argument('--nonsc_weight', metavar='nonsc_weight',type=float, default=.5, help='Loss multiplier for non-selfconsistent datapoints')
parser.add_argument('--start_converged', action='store_true', help='Start from converged density matrix')
parser.add_argument('--scf_steps', metavar='scf_steps', type=int, default=25, help='Number of scf steps')
parser.add_argument('--polynomial', action='store_true', help='Use polynomials instead of neural networks')
parser.add_argument('--free', action='store_true', help='No LOB and UEG limit')
parser.add_argument('--freec', action='store_true', help='No LOB and UEG limit for correlation')
parser.add_argument('--meta_x', metavar='meta_x',type=float, default=None, help='')
parser.add_argument('--rho_alt', action='store_true', help='Alternative rho loss on total density')
parser.add_argument('--radical_factor', metavar='radical_factor',type=float, default=1.0, help='')
args = parser.parse_args()

ueg_limit = not args.free
RHO_mult = args.rho_weight
E_mult = args.E_weight
nonSC_mult = args.nonsc_weight
HYBRID = (args.hyb_par > 0.0)

def get_scf(path=args.modelpath):
    print("get_scf")
    if args.type == 'GGA':
        print("get_scf -> args.type == GGA")
        lob = 1.804 if ueg_limit else 0 
        if args.polynomial:
            print("get_scf -> args.type == GGA -> args.polynomial")
            x = XC_L_POL(device=DEVICE, max_order=3, use=[1], lob=lob, ueg_limit=ueg_limit)
#             c = C_L_POL(device=DEVICE, max_order=4,  use=[0, 1, 2, 3], ueg_limit=ueg_limit and not args.freec)
            c = C_L_POL(device=DEVICE, max_order=4,  use=[0, 1, 2, 3], ueg_limit=ueg_limit and not args.freec)
        else:
            print("get_scf -> args.type == GGA -> ! args.polynomial")
            x = XC_L(device=DEVICE,n_input=1, n_hidden=16, use=[1], lob=lob, ueg_limit=ueg_limit) # PBE_X
            c = C_L(device=DEVICE,n_input=3, n_hidden=16, use=[2], ueg_limit=ueg_limit and not args.freec)
        
        xc_level = 2
    elif args.type == 'MGGA':
        print("get_scf -> args.type == MGGA")
        lob = 1.174 if ueg_limit else 0 
        if args.polynomial:
            print("get_scf -> args.type == MGGA -> args.polynomial")
            x = XC_L_POL(device=DEVICE, max_order=4, use=[1, 2], lob=0, ueg_limit=ueg_limit, sdecay=True)
            c = C_L_POL(device=DEVICE, max_order=3,  use=[0, 1, 2, 3, 4, 5], ueg_limit=ueg_limit)
        else:
            print("get_scf -> args.type == MGGA -> ! args.polynomial")    
            x = XC_L(device=DEVICE,n_input=2, n_hidden=16, use=[1,2], lob=lob, ueg_limit=ueg_limit) # PBE_X
            c = C_L(device=DEVICE,n_input=4, n_hidden=16, use=[2,3], ueg_limit=ueg_limit and not args.freec)
        xc_level = 3
    print("Loading pre-trained models from " + args.pretrain_loc)
    x.load_state_dict(torch.load(args.pretrain_loc + '/x'))
    c.load_state_dict(torch.load(args.pretrain_loc + '/c'))

    if HYBRID:
        try:
            a = 1 - args.hyb_par
            b = 1
            d = args.hyb_par
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level, polynomial = args.polynomial )
            scf = SCF(nsteps=args.scf_steps, xc=xc, exx=True,alpha=0.3)

            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

            xc.add_exx_a(d)
            xc.exx_a.requires_grad=True
        except RuntimeError:
            a = 1 - args.hyb_par
            b = 1
            d = args.hyb_par
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level,exx_a=d, polynomial = args.polynomial)
            scf = SCF(nsteps=args.scf_steps, xc=xc, exx=True,alpha=0.3)

            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            xc.exx_a.requires_grad=True

    else:
        xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level, polynomial = args.polynomial, meta_x = args.meta_x)
        scf = SCF(nsteps=args.scf_steps, xc=xc, exx=False,alpha=0.3)
        if path:
            xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    scf.xc.train()
    return scf

if __name__ == '__main__':

    if HYBRID:
        logpath = args.logpath + str(datetime.now()).replace(' ','_')
    else:
        logpath = args.logpath + str(datetime.now()).replace(' ','_')

    if not args.print_stdout:
        def print(*args):
            with open(logpath + '.log','a') as logfile:
                logfile.write(' ,'.join([str(a) for a in args]) + '\n')

    try:
        os.mkdir('/'.join(logpath.split('/')[:-1]))
    except FileExistsError:
        pass

    print(json.dumps(args.__dict__,indent=4))
    with open(logpath+'.config','w') as file:
        file.write(json.dumps(args.__dict__,indent=4))

    with tarfile.open(logpath + '.tar.gz', "w:gz") as tar:
        source_dir = dpyscf_dir + '/dpyscf/'
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        source_dir = __file__
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print("reading atoms (?)")
    atoms = read(args.reftraj, ':')
    indices = np.arange(len(atoms)).tolist()

    pop = [34, 33, 32, 10, 7, 5]
    pop = []
#     if args.type == 'GGA':
# #         pop = [12, 8, 7,  5, 4]
#         pop = [12, 8, 7,  5, 4]
#         pop = []
#         if args.meta_x:
#             pop = [21] + pop
#         if HYBRID:
# #              pop = [29, 28, 27, 26, 25, 24, 23, 22, 21, 12, 7,  5, 2] # (Hybrid GGA)
#             pop = [21, 12, 7,  5, 2] # (Hybrid GGA)
#             pop = [7]
#     else:
# #         pop = [21, 12, 11, 10, 8, 7, 5, 4, 3, 0] # (Meta-GGA)
#         pop = [12, 10, 7, 5] # (Meta-GGA)
#         pop = [7]
#         if HYBRID:
#             pop = pop + [0]
# #         pop = [ 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 0] # (Meta-GGA)
#     # pop = []

#     pop = np.arange(len(atoms)).tolist()
#     pop.pop(-1)
#     pop.pop(-1)
#     pop.pop(20)
#     pop.pop(16)
#     pop.pop(13)
#     pop = pop[::-1]
    print("popping specified atoms: {}".format(pop))
    [atoms.pop(i) for i in pop]
    [indices.pop(i) for i in pop]


    
    dataset = MemDatasetRead(args.datapath, skip=pop)

#     dataset = MemDatasetRead('/gpfs/home/smdick/smdick/.data/test', skip= [12, 8, 5, 4])

    dataset_train = dataset
    print("dataloader")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # Dont change batch size !
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False) # Dont change batch size !
    print("molecules")
    molecules = {'{:3d}'.format(idx) + ''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) > 1 and not a.info.get('reaction') }
    print(molecules)
    print("atoms")
    pure_atoms = {''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) == 1 and not a.info.get('reaction')}
    print(pure_atoms)
    
    def split(el):
            import re
            res_list = [s for s in re.split("([A-Z][^A-Z]*)", el) if s]
            return res_list

    for molecule in molecules:
        comp = []
        for a in split(molecule[3:]):
            comp.append(pure_atoms[a][0])
        molecules[molecule] += comp

    a_count = {a: np.sum([a in molecules[mol] for mol in molecules]) for a in np.unique([m  for mol in molecules for m in molecules[mol]])}
    print("rxns")
    reactions = {}
    for idx, a in enumerate(atoms):
        if not a.info.get('reaction'): continue
        if a.info.get('reaction') == 'reactant': continue
        reactions['{:3d}'.format(idx) + ''.join(a.get_chemical_symbols())] = \
            [idx] + [idx + i for i in np.arange(-a.info.get('reaction'),0,1).astype(int)]
    print(reactions)
    molecules.update(reactions)
    

    best_loss = 1e6
    print("getting scf")
    scf = get_scf(args.modelpath)
  
    if args.testrun:
        print("\n ======= Starting testrun ====== \n\n")
        scf.xc.evaluate()
#         scf.xc.train()
        Es = []
        E_pretrained = []
        cnt = 0
        for dm_init, matrices, e_ref, dm_ref in dataloader_train:
            print(atoms[cnt])
            sc = atoms[cnt].info.get('sc',True)
            cnt += 1
            if not sc: continue 
            dm_init = dm_init.to(DEVICE)
            e_ref = e_ref.to(DEVICE)
            dm_ref = dm_ref.to(DEVICE)
            matrices = {key:matrices[key].to(DEVICE) for key in matrices}
            E_pretrained.append(matrices['e_pretrained'])
     
            results = scf.forward(matrices['dm_realinit'], matrices, sc)
            E = results['E']

            if sc:
                Es.append(E.detach().cpu().numpy())
            else:
                Es.append(np.array([E.detach().cpu().numpy()]*scf.nsteps))
        e_premodel = np.array(Es)[:,-1]
        print("\n ------- Statistics ----- ")
        print(str(e_premodel), 'Energies from pretrained model' )
        print(str(np.array(E_pretrained)),'Energies from exact DFT baseline')
        print(str(e_premodel - np.array(E_pretrained)), 'Pretraining error')
        print(str(np.array(Es)[:,-1]-np.array(Es)[:,-2]), 'Convergence')

    print("\n ======= Starting training ====== \n\n")
#     assert False
    scf.xc.train()
    PRINT_EVERY=1
    skip_steps = max(5, args.scf_steps - 10)

    def get_optimizer(model, path=''):
        if HYBRID:
#             optimizer = torch.optim.Adam(list(model.parameters()) + [model.xc.exx_a],
#                                       lr=args.lr, weight_decay=args.l2)
            optimizer = torch.optim.Adam(list(model.parameters()),
                                      lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                      lr=args.lr, weight_decay=args.l2)

        MIN_RATE = 1e-7
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               verbose=True, patience=int(10/PRINT_EVERY),
                                                               factor=0.1, min_lr=MIN_RATE)

        if path:
            optimizer.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return optimizer, scheduler

    optimizer, scheduler = get_optimizer(scf)

    AE_mult = 1
    density_loss = rho_alt_loss if args.rho_alt else rho_loss
    mol_losses = {"rho" : (partial(density_loss, loss = torch.nn.MSELoss()), RHO_mult)}
    atm_losses = {"E":  (partial(energy_loss, loss = torch.nn.MSELoss()), E_mult)}
    h_losses = {"rho" : (partial(density_loss,loss = torch.nn.MSELoss()), RHO_mult),
                "E":  (partial(energy_loss, loss = torch.nn.MSELoss()), E_mult)}
#     h_losses = {}

    ae_loss = partial(ae_loss,loss = torch.nn.MSELoss())
     
  
    train_order = np.arange(len(molecules)).astype(int)
    molecules_sc = {}
    for m_idx in train_order:
        molecule = list(molecules.keys())[m_idx]
        mol_sc = True
        for idx in range(len(dataloader_train)):
            if not idx in molecules[molecule]: continue
            mol_sc = atoms[idx].info.get('sc',True)
        molecules_sc[molecule] = mol_sc
        
            
    chkpt_idx = 0
    validate_every = 10

    for epoch in range(100000):


        encountered_nan = True
        while(encountered_nan):
            error_cnt = 0
            running_losses = {key:0 for key in mol_losses}
            running_losses['ae'] = 0
#             if args.hnorm:
            running_losses['E'] = 0
            total_loss = 0
            atm_cnt = {}
            encountered_nan = False
            try:
                train_order = np.arange(len(molecules)).astype(int)
                np.random.shuffle(train_order)
                for m_idx in train_order:
                    molecule = list(molecules.keys())[m_idx]
                    if not molecules_sc[molecule] and not nonSC_mult: continue
                    mol_sc = True
#                 for molecule in molecules:
                    ref_dict = {}
                    pred_dict = {}
                    loss = 0
                    for idx, data in enumerate(dataloader_train):
                        if not idx in molecules[molecule]: continue
                        if args.print_names: print(atoms[idx])
                        dm_init, matrices, e_ref, dm_ref = data
                        dm_init = dm_init.to(DEVICE)
                        e_ref = e_ref.to(DEVICE)
                        dm_ref = dm_ref.to(DEVICE)
                        matrices = {key:matrices[key].to(DEVICE) for key in matrices}
                        dm_mix = matrices['dm_realinit']
#                         mixing = torch.rand(1)*0+1
                        if args.start_converged:
                            mixing = torch.rand(1)*0
                        else:
                            mixing = torch.rand(1)/2 + 0.5
                        sc = atoms[idx].info.get('sc',True)
                        if sc:
                            dm_in = dm_init*(1-mixing) + dm_mix*mixing
                        else:
                            dm_in = dm_init
                            mol_sc = False
                        reaction = atoms[idx].info.get('reaction',False)
                        results = scf(dm_in, matrices, sc)
                        results['dm_ref'] = dm_ref
                        results['fcenter'] = matrices.get('fcenter',None)
                        results['rho'] = matrices['rho']
                        results['ao_eval'] = matrices['ao_eval']
                        results['grid_weights'] = matrices['grid_weights']
                        results['E_ref'] = e_ref
                        results['mo_energy_ref'] = matrices['mo_energy']
                        results['n_elec'] = matrices['n_elec']
                        results['e_ip_ref'] = matrices['e_ip']
                        results['mo_occ'] = matrices['mo_occ']
                        if atoms[idx].info.get('radical', False):
                            results['rho'] *= args.radical_factor
                            results['dm'] *= args.radical_factor
                        if len(atoms[idx].positions) > 1 and sc:
                            losses = mol_losses
                        elif str(atoms[idx].symbols) in ['H', 'Li'] and args.hnorm and not reaction:
                            losses = h_losses
                        elif sc and not reaction:
                            losses = atm_losses
                        else:
                            losses = {}
                        losses_eval = {key: losses[key][0](results)/a_count[idx] for key in losses}
                        running_losses.update({key:running_losses[key] + losses_eval[key].item() for key in losses})
                        
                        if reaction == 2:
                            ref_dict['AB'] = e_ref
                            pred_dict['AB'] = results['E'][-1:]
                        elif reaction == 1:
                            ref_dict['AA'] = e_ref*2
                            pred_dict['AA'] = results['E'][skip_steps:]*2
                        elif reaction == 'reactant':
                            if sc:
                                ref_dict['A'] = e_ref
                                pred_dict['A'] = results['E'][skip_steps:]
                            else:
                                label = 'A' if not 'A' in ref_dict else 'B'
                                ref_dict[label] = e_ref
                                pred_dict[label] = results['E'][-1:]
                                
                        elif len(atoms[idx].positions) > 1:
                            ref_dict[''.join(atoms[idx].get_chemical_symbols())] = e_ref
                            if sc:
                                steps = skip_steps
                            else:
                                steps = -1    
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][steps:]
                        else:
                            ref_dict[''.join(atoms[idx].get_chemical_symbols())] = torch.zeros_like(e_ref)
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][-1:]
                        loss += sum([losses_eval[key]*losses[key][1] for key in losses])

                    ael = ae_loss(ref_dict,pred_dict)
                    running_losses['ae'] += ael.item()
#                     print('predict dict', pred_dict)
#                     print('ref dict', ref_dict)
#                     print('AE loss', ael.item())
                    if mol_sc:
                        running_losses['ae'] += ael.item()
                        loss += ael
                    else:
                        loss += nonSC_mult * ael
                        running_losses['ae'] += nonSC_mult * ael.item()
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            except RuntimeError:
                encountered_nan = True
                chkpt_idx -= 1
                print('NaNs encountered, rolling back to checkpoint {}'.format(chkpt_idx%3))
                if chkpt_idx == -1:
                    scf = get_scf()
                    optimizer, scheduler = get_optimizer(scf)
                else:
                    scf = get_scf(logpath + '_{}.chkpt'.format(chkpt_idx%3))
                    optimizer, scheduler = get_optimizer(scf, logpath + '_{}.adam.chkpt'.format(chkpt_idx%3))
                scf.xc.train()
                error_cnt +=1
#                 chkpt_idx += 1
                if error_cnt > 3:
                    print('NaNs could not be resolved by rolling back to checkpoint')
                    raise RuntimeError('NaNs could not be resolved by rolling back to checkpoint')

        if epoch%PRINT_EVERY==0:
            running_losses = {key:np.sqrt(running_losses[key]/len(molecules))*1000 for key in running_losses}
            total_loss = np.sqrt(total_loss/len(molecules))*1000
            best_loss = min(total_loss, best_loss)
            chkpt_str = ''
            torch.save(scf.xc.state_dict(), logpath + '_current.chkpt')
            torch.save(scf, logpath + '_current.pt')
            if total_loss == best_loss:
                torch.save(scf.xc.state_dict(), logpath + '_{}.chkpt'.format(chkpt_idx%3))
                torch.save(optimizer.state_dict(), logpath + '_{}.adam.chkpt'.format(chkpt_idx%3))
                chkpt_str = '_{}.chkpt'.format(chkpt_idx%3)
                chkpt_idx += 1
            print('Epoch {} ||'.format(epoch), [' {} : {:.6f}'.format(key,val) for key, val in running_losses.items()],
                  '|| total loss {:.6f}'.format(total_loss),chkpt_str)
            if HYBRID:
                print(scf.xc.exx_a)
            scheduler.step(total_loss)
