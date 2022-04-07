#!/usr/bin/env python
# coding: utf-8
import torch
torch.set_default_dtype(torch.float)

import numpy as np
from ase.io import read
from dpyscfl.net import *
from dpyscfl.scf import *
from dpyscfl.utils import *
from dpyscfl.losses import *
from functools import partial
from ase.units import Bohr
from datetime import datetime
import os
import psutil
import tarfile
import argparse
import json

process = psutil.Process(os.getpid())
#dpyscf_dir = os.environ.get('DPYSCF_DIR','..')
DEVICE = 'cpu'

parser = argparse.ArgumentParser(description='Train xc functional')
parser.add_argument('--pretrain_loc', action='store', type=str, help='Location of pretrained models (should be directory containing x and c)')
parser.add_argument('--type', action='store', choices=['GGA','MGGA'])
parser.add_argument('--datapath', action='store', type=str, help='Location of precomputed matrices (run prep_data first)')
parser.add_argument('--reftraj', action='store', type=str, help='Location of reference trajectories')
parser.add_argument('--n_hidden', metavar='n_hidden', type=int, default=16, help='Number of hidden nodes (16)')
parser.add_argument('--hyb_par', metavar='hyb_par', type=float, default=0.0, help='Hybrid mixing parameter (0.0)')
parser.add_argument('--E_weight', metavar='e_weight', type=float, default=0.0, help='Weight of total energy term in loss function (0)')
parser.add_argument('--rho_weight', metavar='rho_weight', type=float, default=25, help='Weight of density term in loss function (25)')
parser.add_argument('--modelpath', metavar='modelpath', type=str, default='', help='Net Checkpoint location to continue training')
parser.add_argument('--optimpath', metavar='optimpath', type=str, default='', help='Optimizer Checkpoint location to continue training')
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
parser.add_argument('--forcedensloss', action='store_true', default=False, help='Make training use density loss.')
args = parser.parse_args()

ueg_limit = not args.free
HYBRID = (args.hyb_par > 0.0)

def get_optimizer(model, path='', hybrid=HYBRID):
    if hybrid:
            optimizer = torch.optim.Adam(list(model.parameters()) + [model.xc.exx_a],
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


if __name__ == '__main__':

    logpath = os.path.join(args.logpath, str(datetime.now()).replace(' ','_'))

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
        #source_dir = dpyscf_dir + '/dpyscf/'
        #tar.add(source_dir, arcname=os.path.basename(source_dir))
        source_dir = __file__
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print("READING REFERENCE TRAJECTORY.")
    atoms = read(args.reftraj, ':')
    indices = np.arange(len(atoms)).tolist()

    pop = []
    print("popping specified atoms: {}".format(pop))
    [atoms.pop(i) for i in pop]
    [indices.pop(i) for i in pop]


    print("READING DATASET")
    dataset = MemDatasetRead(args.datapath, skip=pop)
    dataset_train = dataset
    print("LOADING DATASET INTO PYTORCH")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # Dont change batch size !
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False) # Dont change batch size !
    print("PARSING MOLECULES")
    molecules = {'{:3d}'.format(idx) + ''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) > 1 and not a.info.get('reaction') }
    print(molecules)
    print("PARSING PURE ATOMS")
    pure_atoms = {''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) == 1 and not a.info.get('reaction') and not a.info.get('fractional') and not a.info.get('charge')}
    print(pure_atoms)
    molecules.update(pure_atoms)
    print("PARSING WHOLLY CHARGED ATOMS")
    c_atoms = {''.join(a.get_chemical_symbols())+'_c{}'.format(a.info['charge']): [idx] for idx, a in enumerate(atoms) if len(a.positions)==1 and a.info.get('charge')}
    print(c_atoms)
    molecules.update(c_atoms)
    print("PARSING FRACTIONAL ATOMS")
    frac_atoms = {''.join(a.get_chemical_symbols())+'_f{}'.format(a.info['fractional']): [idx] for idx, a in enumerate(atoms) if len(a.positions)==1 and a.info.get('fractional')}
    print(frac_atoms)
    molecules.update(frac_atoms)
    def split(el):
            import re
            res_list = [s for s in re.split("([A-Z][^A-Z]*)", el) if s]
            return res_list

    #for molecule in molecules:
    #    comp = []
    #    for a in split(molecule[3:]):
    #        comp.append(pure_atoms[a][0])
    #    molecules[molecule] += comp

    a_count = {a: np.sum([a in molecules[mol] for mol in molecules]) for a in np.unique([m  for mol in molecules for m in molecules[mol]])}
    print("PARSING REACTIONS")
    reactions = {}
    for idx, a in enumerate(atoms):
        #atom must have reaction flag
        if not a.info.get('reaction'): continue
        #and atom must not be a reactant
        if a.info.get('reaction') == 'reactant': continue
        reactions['{:3d}'.format(idx) + ''.join(a.get_chemical_symbols())] = \
            [idx] + [idx + i for i in np.arange(-a.info.get('reaction'),0,1).astype(int)]
    print(reactions)
    molecules.update(reactions)
    print("MOLECULES TO TRAIN ON")
    print(molecules)

    best_loss = 1e6
    print("GENERATING SCF OBJECT")
    if args.pretrain_loc:
        scf = get_scf(args.type, pretrain_loc=args.pretrain_loc)
    elif args.modelpath:
        scf = get_scf(args.type, path=args.modelpath)
    else:
        scf = get_scf(args.type)

    if args.testrun:
        #TODO: fix this, dataloader doesn't have dm_ref
        print("\n ======= Starting testrun ====== \n\n")
        #Set SCF Object training flag off
        scf.xc.evaluate()
#         scf.xc.train()
        Es = []
        E_pretrained = []
        cnt = 0
        for dm_init, matrices, e_ref, dm_ref in dataloader_train:
            e_ref = matrices['e_base']
            dm_ref = matrices['dm']
            print(atoms[cnt])
            sc = atoms[cnt].info.get('sc',True)
            cnt += 1
            #If atom not self-consistent, skip
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
    scf.xc.train()
    PRINT_EVERY=1
    skip_steps = max(5, args.scf_steps - 10)

    optimizer, scheduler = get_optimizer(scf)

    AE_mult = 1
    #Loss Functions -- Density
    density_loss = rho_alt_loss if args.rho_alt else rho_loss
    ##TODO: no reason to make these tuples
    mol_losses = {"rho" : (partial(density_loss, loss = torch.nn.MSELoss()), args.rho_weight)}
    atm_losses = {"E":  (partial(energy_loss, loss = torch.nn.MSELoss()), args.E_weight)}
    h_losses = {"rho" : (partial(density_loss,loss = torch.nn.MSELoss()), args.rho_weight),
                "E":  (partial(energy_loss, loss = torch.nn.MSELoss()), args.E_weight)}

    ae_loss = partial(ae_loss,loss = torch.nn.MSELoss())
     
    #Indices for self-consistent training molecules
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

    def scf_wrap(scf, dm_in, matrices, sc):
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


    for epoch in range(100000):
        encountered_nan = True
        while(encountered_nan):
            error_cnt = 0
            running_losses = {"rho": 0, "ae":0, "E":0}
            total_loss = 0
            atm_cnt = {}
            encountered_nan = False
            try:
                #Randomly choose training order
                train_order = np.arange(len(molecules)).astype(int)
                np.random.shuffle(train_order)
                for m_idx in train_order:
                    molecule = list(molecules.keys())[m_idx]
                    print("TRAINING ON MOLECULE: ", molecule)
                    #if molecule not self-consistent, skip it
                    if not molecules_sc[molecule] and not args.nonsc_weight: continue
                    mol_sc = True
                    ref_dict = {}
                    pred_dict = {}
                    loss = 0
                    for idx, data in enumerate(dataloader_train):
                        if idx == 0:
                            print(idx, data[1].keys())
                        #If the index in the data is not the index of the molecule chosen by m_idx, continue
                        if not idx in molecules[molecule]: continue
                        if args.print_names: print(atoms[idx])
                        #dm_init, matrices, e_ref, dm_ref = data
                        dm_init = data[0]
                        e_ref = data[1]['e_base']
                        dm_ref = data[1]['dm']
                        matrices = data[1]
                        dm_init = dm_init.to(DEVICE)
                        e_ref = e_ref.to(DEVICE)
                        dm_ref = dm_ref.to(DEVICE)
                        matrices = {key:matrices[key].to(DEVICE) for key in matrices}
                        dm_mix = matrices['dm_realinit']
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
                        fractionFlag = atoms[idx].info.get('fractional', False)
                        if fractionFlag:
                            print("***************************")
                            print("FRACTIONAL FLAG SET. f = {}".format(fractionFlag))
                            print("INDEXED ATOM: {}, {}".format(idx, atoms[idx].get_chemical_formula()))
                            print("***************************")
                        #CALCULATION
                        print("SCF CALCULATION")
                        results = scf_wrap(scf, dm_in, matrices, sc)
                        if results == None:
                            break
                        #Add matrix keys to results dict
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
                        print("RESULTS MATRICES EXTRACTED")
                        #If radical, multiplicative factor
                        if atoms[idx].info.get('radical', False):
                            results['rho'] *= args.radical_factor
                            results['dm'] *= args.radical_factor
                        #If molecule and self-consistent, use "mol_losses" dict
                        if len(atoms[idx].positions) > 1 and sc:
                            print("MOL_LOSSES")
                            losses = mol_losses
                        #Else, if chosen atom is either H or Li, and args specify use both, and the H/Li not involved in reaction
                        elif str(atoms[idx].symbols) in ['H', 'Li'] and args.hnorm and not reaction:
                            print("H_LOSSES")
                            losses = h_losses
                        #Otherwise, if just an atom not in a reaction:
                        elif sc and not reaction:
                            print("ATM_LOSSES")
                            losses = atm_losses
                        #Else empty loss dict
                        else:
                            losses = {}
                        #if choose to force density loss,
                        if args.forcedensloss:
                            print("FORCED DENSITY LOSS")
                            losses = h_losses
                        #For each key in whichever loss dict chosen,
                        #Select the function (it's a tuple of itself), feed in results dict, normalize by number of atoms
                        losses_eval = {key: losses[key][0](results)/a_count[idx] for key in losses}
                        print("LOSSES_EVAL: ", losses_eval)
                        #Update running losses with new losses
                        #TODO: why is .item() needed????
                        running_losses.update({key:running_losses[key] + losses_eval[key].item() for key in losses})
                        
                        #IF Reaction type is 2, it is an A+B -> AB reaction.
                        #Store the dataset e_ref as ref, and results E as prediction
                        if reaction == 2:
                            ref_dict['AB'] = e_ref
                            pred_dict['AB'] = results['E'][-1:]
                        #ELSE if Reaction type is 1, it is an A->A reaction with some charge difference,
                        #Typically, reactant is charged so reaction == 1 is neutral
                        #TODO: Why multiply by 2 here?
                        elif reaction == 1:
                            ref_dict['AA'] = e_ref*2
                            pred_dict['AA'] = results['E'][skip_steps:]*2
                        #ELSE IF it is a reactant in either of the above pathways,
                        elif reaction == 'reactant':
                            #If self-consistent,
                            if sc:
                                ref_dict['A'] = e_ref
                                pred_dict['A'] = results['E'][skip_steps:]
                            else:
                                label = 'A' if not 'A' in ref_dict else 'B'
                                ref_dict[label] = e_ref
                                pred_dict[label] = results['E'][-1:]
                        #If not reaction 2, 1, reactant, and molecule has more than one atom, e_ref is reference energy
                        elif len(atoms[idx].positions) > 1:
                            ref_dict[''.join(atoms[idx].get_chemical_symbols())] = e_ref
                            if sc:
                                steps = skip_steps
                            else:
                                steps = -1    
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][steps:]
                        
                        else:
                            #ref_dict[''.join(atoms[idx].get_chemical_symbols())] = torch.zeros_like(e_ref)
                            ref_dict[''.join(atoms[idx].get_chemical_symbols())] = e_ref
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][-1:]
                        loss += sum([losses_eval[key]*losses[key][1] for key in losses])
                    if not results:
                        continue
                    print("LOOP OVER TRAJECTORY COMPLETED")
                    print("REF_DICT: ", ref_dict)
                    print("PRED_DICT: ", pred_dict)
                    ael = ae_loss(ref_dict,pred_dict)
                    running_losses['ae'] += ael.item()
                    print('predict dict', pred_dict)
                    print('ref dict', ref_dict)
                    print('AE loss', ael.item())
                    if mol_sc:
                        running_losses['ae'] += ael.item()
                        loss += ael
                    else:
                        loss += args.nonsc_weight * ael
                        running_losses['ae'] += args.nonsc_weight * ael.item()
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            except RuntimeError:
                encountered_nan = True
                chkpt_idx -= 1
                print('NaNs encountered, rolling back to checkpoint {}'.format(chkpt_idx%3))
                xcpath = args.modelpath if args.modelpath else args.pretrain_loc
                if chkpt_idx == -1:
                    scf = get_scf(args.type, xcpath)
                    optimizer, scheduler = get_optimizer(scf)
                else:
                    scf = get_scf(args.type, logpath + '_{}.chkpt'.format(chkpt_idx%3))
                    optimizer, scheduler = get_optimizer(scf, logpath + '_{}.adam.chkpt'.format(chkpt_idx%3))
                scf.xc.train()
                error_cnt +=1
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
            print("============================================================")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('Epoch {} ||'.format(epoch), [' {} : {:.6f}'.format(key,val) for key, val in running_losses.items()],
                  '|| total loss {:.6f}'.format(total_loss),chkpt_str)
            if HYBRID:
                print(scf.xc.exx_a)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("============================================================")

            scheduler.step(total_loss)
