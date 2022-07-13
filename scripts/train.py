#!/usr/bin/env python
# coding: utf-8
from unittest import skip
import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
from ase.io import read
from dpyscfl.net import *
from dpyscfl.scf import *
from dpyscfl.utils import *
from dpyscfl.losses import *
from functools import partial
from ase.units import Bohr
from datetime import datetime
import os, psutil, tarfile, argparse, json
import tqdm, inspect
#from eval import KS, eval_xc, eval_wrap
old_print = print
def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)
# globaly replace print with new_print
inspect.builtins.print = new_print
#Validation Imports
#from eval import KS, eval_xc

process = psutil.Process(os.getpid())
#dpyscf_dir = os.environ.get('DPYSCF_DIR','..')
DEVICE = 'cpu'

parser = argparse.ArgumentParser(description='Train xc functional')
parser.add_argument('--pretrain_loc', action='store', type=str, help='Location of pretrained models (should be directory containing x and c)')
parser.add_argument('--type', action='store', choices=['GGA','MGGA'])
parser.add_argument('--datapath', action='store', type=str, help='Location of precomputed matrices (run prep_data first)')
parser.add_argument('--reftraj', action='store', type=str, help='Location of reference trajectories')
parser.add_argument('--valtraj', action='store', type=str, help='Location of reference data to use in validation')
parser.add_argument('--n_hidden', metavar='n_hidden', type=int, default=16, help='Number of hidden nodes (16)')
parser.add_argument('--hyb_par', metavar='hyb_par', type=float, default=0.0, help='Hybrid mixing parameter (0.0)')
parser.add_argument('--E_weight', metavar='e_weight', type=float, default=0.01, help='Weight of total energy term in loss function (0)')
parser.add_argument('--rho_weight', metavar='rho_weight', type=float, default=20, help='Weight of density term in loss function (25)')
parser.add_argument('--modelpath', metavar='modelpath', type=str, default='', help='Net Checkpoint location to continue training')
parser.add_argument('--optimpath', metavar='optimpath', type=str, default='', help='Optimizer Checkpoint location to continue training')
parser.add_argument('--logpath', metavar='logpath', action='store', type=str, default='log/', help='Logging directory (log/)')
parser.add_argument('--testrun', action='store_true', help='Do a test run over all molecules before training')
parser.add_argument('--lr', metavar='lr', type=float, action='store', default=0.0001, help='Learning rate (0.0001)')
parser.add_argument('--l2', metavar='l2', type=float, default=1e-8, help='Weight decay (1e-8)')
parser.add_argument('--hnorm', action='store_true', help='Use H energy and density in loss')
parser.add_argument('--print_stdout', action='store_true', help='Print to stdout instead of logfile')
parser.add_argument('--print_names', action='store_true', help='Print molecule names during training')
parser.add_argument('--nonsc_weight', metavar='nonsc_weight',type=float, default=0.01, help='Loss multiplier for non-selfconsistent datapoints')
parser.add_argument('--start_converged', action='store_true', help='Start from converged density matrix')
parser.add_argument('--scf_steps', metavar='scf_steps', type=int, default=25, help='Number of scf steps')
parser.add_argument('--polynomial', action='store_true', help='Use polynomials instead of neural networks')
parser.add_argument('--free', action='store_true', help='No LOB and UEG limit')
parser.add_argument('--freec', action='store_true', help='No LOB and UEG limit for correlation')
parser.add_argument('--meta_x', metavar='meta_x',type=float, default=None, help='')
parser.add_argument('--rho_alt', action='store_true', help='Alternative rho loss on total density')
parser.add_argument('--radical_factor', metavar='radical_factor',type=float, default=1.0, help='')
parser.add_argument('--forcedensloss', action='store_true', default=False, help='Make training use density loss.')
parser.add_argument('--forceEloss', action='store_true', default=False, help='Make training use TOTAL energy loss. Ill-advised to use given atomization energy structure.')
parser.add_argument('--freezeappend',  type=int, action='store', default=0, help='If flagged, freezes network and adds N duplicate layers between output layer and last hidden layer. The new layer is not frozen.')
parser.add_argument('--loadfa', type=int, action='store', default=0, help='If loading model that has appended layers, specify number of inserts between previous final and output layers here.')
parser.add_argument('--outputlayergrad', action='store_true', default=False, help='Only works with freezeappend. If flagged, sets the output layer to also be differentiable.')
parser.add_argument('--checkgrad', action='store_true', default=False, help='If flagged, executes loop over scf.xc parameters to print gradients')
parser.add_argument('--testmol', type=str, action='store', default='', help='If specified, give symbols/formula/test label for debugging purpose')
parser.add_argument('--torchtype', type=str, default='float', help='float or double')
args = parser.parse_args()

ttypes = {'float' : torch.float,
            'double': torch.double}

ueg_limit = not args.free
HYBRID = (args.hyb_par > 0.0)


torch.set_default_dtype(ttypes[args.torchtype])


def scf_wrap(scf, dm_in, matrices, sc, molecule=''):
    try:
        results = scf(dm_in, matrices, sc)
    except Exception as e:
        print("========================================================")
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print("SCF CALCULATION FAILED")
        print("SCF Calculation failed for {}".format(molecule))
        print("{}".format(e))
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print("========================================================")
        results = None
    return results


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
    #Some molecules in the trajectory are not needed to reproduce xcdiff training set.
    #But don't pop them, because indexing important.
    #skips = ['O2', 'Cl2', 'HCl']
    skips = ['O2']
    
    pop = []
    #print("popping specified atoms: {}".format(pop))
    #[atoms.pop(i) for i in pop]
    #[indices.pop(i) for i in pop]


    print("READING DATASET")
    dataset = MemDatasetRead(args.datapath, skip=pop)
    dataset_train = dataset

    print("LOADING DATASET INTO PYTORCH")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # Dont change batch size !
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False) # Dont change batch size !
    
    print("PARSING NON-ATOMIC NON-REACTION MOLECULES")
    molecules = {'{:3d}'.format(idx) + ''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) > 1 and not a.info.get('reaction') }
    print(molecules)

    print("PARSING NEUTRAL, PURE NON-REACTION ATOMS. CHARGE FLAG NOT SET.")
    pure_atoms = {''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) == 1 and not a.info.get('reaction') and not a.info.get('fractional') and not a.info.get('charge') and not a.info.get('supp')}
    print(pure_atoms)
    molecules.update(pure_atoms)
    
    print("PARSING SUPPLEMENTAL NEUTRAL, PURE ATOMS (FROM FRAC DATASET)")
    n_atoms = {''.join(a.get_chemical_symbols())+'_n0': [idx] for idx, a in enumerate(atoms) if len(a.positions)==1 and a.info.get('supp') and not a.info.get('charge') and not a.info.get('fractional')}
    print(n_atoms)

    #molecules.update(n_atoms)
    print("PARSING SUPPLEMENTAL CHARGED, PURE ATOMS")
    c_atoms = {''.join(a.get_chemical_symbols())+'_c{}'.format(a.info['charge']): [idx] for idx, a in enumerate(atoms) if len(a.positions)==1 and a.info.get('supp') and a.info.get('charge')}
    print(c_atoms)

    #molecules.update(c_atoms)
    print("PARSING SUPPLEMENTAL FRACTIONAL ATOMS")
    frac_atoms = {''.join(a.get_chemical_symbols())+'_f{}'.format(a.info['fractional']): [idx] for idx, a in enumerate(atoms) if len(a.positions)==1 and a.info.get('supp') and a.info.get('fractional')}
    print(frac_atoms)
    #molecules.update(frac_atoms)
    
    def cat_dict(dicts, keysplit='_'):
        #To generate the list of atoms comprising a fractional datapoint, for effective reaction
        retdct = {k:[] for k in list(dicts[0].keys())}
        rkeys = sorted(list(retdct.keys()))
        for didx,dct in enumerate(dicts):
            dkeys = sorted(list(dct.keys()))
            for dk in dkeys:
                mkey = [mk for mk in rkeys if dk.split(keysplit)[0] in mk.split(keysplit)[0]][0]
                retdct[mkey] += dct[dk]
        return retdct

    fracdct = cat_dict([frac_atoms, n_atoms, c_atoms])
    print("CONCATENATING SUPPLEMENTAL/FRACTIONAL ATOMS")
    print(frac_atoms)
    molecules.update(fracdct)

    def split(el):
            import re
            #Splits a string on capital letter sequences
            res_list = [s for s in re.split("([A-Z][^A-Z]*)", el) if s]
            return res_list

    for molecule in molecules:
        comp = []
        #ignore _ atoms, charged or fractional
        if '_' in molecule:
            continue
        for a in split(molecule[3:]):
            comp.append(pure_atoms[a][0])
        molecules[molecule] += comp

    #a_count = {idx: len(at.positions) for idx,at in enumerate(atoms)}
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
        scf = get_scf(args.type, pretrain_loc=args.pretrain_loc, hyb_par=args.hyb_par)
    elif args.modelpath:
        scf = get_scf(args.type, path=args.modelpath, inserts=args.loadfa, hyb_par=args.hyb_par)
    else:
        scf = get_scf(args.type, hyb_par=args.hyb_par)
    scf.nsteps = args.scf_steps

    if args.freezeappend:
        print("\n ================================= \n")
        print("FREEZING SCF MODEL.XC AND APPENDING NEW LAYER")
        print("\n ================================= \n")
        freeze_append_xc(scf, args.freezeappend, args.outputlayergrad)

    if args.testrun:
        print("\n ======= Starting testrun ====== \n\n")
        #Set SCF Object training flag off
        scf.xc.evaluate()

        Es = []
        E_pretrained = []
        tested = []
        #TODO: fix how things are loaded to use testrun
        #for dm_init, matrices, e_ref, dm_ref in dataloader_train:
        #for dm_init, matrices in dataloader_train:
        for tridx, data in enumerate(dataloader_train):
            atom = atoms[tridx]
            cf, cs = (atom.get_chemical_formula(), str(atom.symbols))
            sc = atom.info.get('sc',True)

            #If atom not self-consistent, skip
            if not sc:
                print('skipping {}, not sc'.format(cf))
                continue
            if ( (cf in skips) or (cs in skips) ):
                print("skipping {}, in skips".format(atom.get_chemical_formula()))
                continue
            tested.append(atom)
            print("====================================")
            print("Testrun Calculation")
            print(tridx, atom, cf, cs)
            print("====================================")
            dm_init = data[0]
            matrices = data[1]
            
            try:
                #previous prep_data had different keys for the matrix values
                e_ref = matrices['e_base']
            except KeyError:
                print("Wrong key, trying Etot from matrices")
                e_ref = matrices['Etot']
            dm_ref = matrices['dm']
            dm_init = dm_init.to(DEVICE)
            e_ref = e_ref.to(DEVICE)
            dm_ref = dm_ref.to(DEVICE)
            matrices = {key:matrices[key].to(DEVICE) for key in matrices}

            results = scf_wrap(scf, matrices['dm_realinit'], matrices, sc, molecule='testrun_{}'.format(tridx))
            if not results:
                continue

            E_pretrained.append(matrices['e_base'])
            E = results['E']

            if sc:
                Es.append(E.detach().cpu().numpy())
            else:
                Es.append(np.array([E.detach().cpu().numpy()]*scf.nsteps))
        e_premodel = np.array(Es)[:,-1]
        error_pretrain = e_premodel - np.array(E_pretrained)
        convergence = np.array(Es)[:,-1]-np.array(Es)[:,-2]
        print("\n ------- Statistics ----- ")
        print(str(e_premodel), 'Energies from pretrained model' )
        print(str(np.array(E_pretrained)),'Energies from exact DFT baseline')
        print(str(error_pretrain), 'Pretraining error')
        print(str(convergence), 'Convergence')

        with open(logpath+'testrun.dat', 'w') as f:
            f.write('#IDX FORMULA SYMBOLS E_PRETRAINED_MODEL E_DFT_BASELINE E_ERROR CONVERGENCE SC\n')
            f.write("#MSE: {}\n".format(1/(len(error_pretrain))*np.sqrt(np.sum(error_pretrain**2))))
            for i in range(len(tested)):
                atom = tested[i]
                sc = atom.info.get('sc', True)
                cf, cs = (atom.get_chemical_formula(), str(atom.symbols))
                if not sc:
                    print('non-sc atom {}, skipping')
                if ( (cf in skips) or (cs in skips) ):
                    print("write test: skipping {}".format(atom.get_chemical_formula()))
                    continue
                f.write('{} {} {} {} {} {} {} {}\n'.format(i, cf, cs, e_premodel[i],
                E_pretrained[i], error_pretrain[i], convergence[i], sc))

            


    print("\n ======= Starting training ====== \n\n")
    scf.xc.train()
    PRINT_EVERY=1
    skip_steps = max(5, args.scf_steps - 10)

    optimizer, scheduler = get_optimizer(scf, path=args.optimpath)

    AE_mult = 1

    #Loss Functions -- Density
    density_loss = rho_alt_loss if args.rho_alt else rho_loss
    # args.rho_weight defaults to 20, per the paper
    # args.E_weight defaults to 0.01, per the paper
    # AE Loss has weight 1 by default, per the paper, but decreases for non-sc
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
                t = tqdm.tqdm(train_order)
                fails = []
                #loop over tqdm object for shell progress bar
                for m_idx in t:
#                for m_idx in train_order:
                    molecule = list(molecules.keys())[m_idx]
                    submolecules = [atoms[idx].get_chemical_formula() for idx in molecules[molecule]]
                    print("================================")
                    print('--------------{}----------------'.format(m_idx))
                    print("TRAINING ON MOLECULE: ", molecule)
                    print("SUBMOLECULES: {}".format(submolecules))
                    print("================================")
                    #if molecule not self-consistent and the weight associated to nonsc molecules is 0, skip it
                    if not molecules_sc[molecule] and not args.nonsc_weight: continue
                    
                    m_form = atoms[molecules[molecule][0]].get_chemical_formula()
                    if args.testmol:
                        if not ( (args.testmol == m_form) or (args.testmol in molecule) ):
                            continue
                    if m_form in skips:
                        print("SKIPPING: ", molecule)
                        continue
                    mol_sc = True
                    ref_dict = {}
                    pred_dict = {}
                    loss = 0
                    #subset Dataset so that we don't have to load unnecessary data during one molecule step
                    #previously, this looped over the entire Dataset and matched indices contained in molecule list
                    print("Subsetting Dataset with molecules[{}]: ".format(molecule), molecules[molecule])
                    subset = torch.utils.data.Subset(dataset, molecules[molecule])
                    subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
                    #for idx, data in enumerate(dataloader_train):
                    for didx, data in enumerate(subset_loader):
                        idx = molecules[molecule][didx]
                        if idx == 0:
                            print(idx, data[1].keys())
                        #modify loading bar for descriptive progress during training
                        t.set_postfix({"Epoch": epoch, 'Training Label':molecule, 'Molecules': [atoms[idx].get_chemical_formula() for idx in molecules[molecule]],
                        'Current Sub-Molecule': atoms[idx].get_chemical_formula()})

                        print("Calculating sub-atoms in molecule -- ", atoms[idx])
                        if args.print_names: print(atoms[idx])
                        #xcdiff version saved as dm_init, matrices, e_ref, dm_ref,
                        #but everything after dm_init contained in matrices now
                        dm_init = data[0]
                        matrices = data[1]
                        try:
                            #previous prep_data had different keys for the matrix values
                            e_ref = matrices['e_base']
                        except KeyError:
                            print("Wrong key, trying Etot from matrices")
                            e_ref = matrices['Etot']

                        #get ref dm, send extracted values to device
                        dm_ref = matrices['dm']
                        dm_init = dm_init.to(DEVICE)
                        e_ref = e_ref.to(DEVICE)
                        dm_ref = dm_ref.to(DEVICE)
                        matrices = {key:matrices[key].to(DEVICE) for key in matrices}
                        dm_mix = matrices['dm_realinit']
                        print("REFERENCE ENERGY: {}".format(e_ref))
                        
                        #if converged, don't mix dm's
                        if args.start_converged:
                            mixing = torch.rand(1)*0
                        else:
                            mixing = torch.rand(1)/2 + 0.5
                        sc = atoms[idx].info.get('sc',True)

                        #mix dms if not converged/if sc
                        if sc:
                            dm_in = dm_init*(1-mixing) + dm_mix*mixing
                        else:
                            dm_in = dm_init
                            mol_sc = False

                        #get flags to determine loss -- reaction, fractional, supplemental, charge
                        reaction = atoms[idx].info.get('reaction',False)
                        fractionFlag = atoms[idx].info.get('fractional', False)
                        suppFlag = atoms[idx].info.get('supp', False)
                        charge = atoms[idx].info.get('charge', 0)
                        if suppFlag:
                            print("***************************")
                            print("SUPPLEMENTAL FLAG SET.")
                            print("INDEXED ATOM: {}, {}".format(idx, atoms[idx].get_chemical_formula()))
                            print("***************************")
                            
                            if fractionFlag:
                                print("***************************")
                                print("FRACTIONAL FLAG SET. f = {}".format(fractionFlag))
                                print("INDEXED ATOM: {}, {}".format(idx, atoms[idx].get_chemical_formula()))
                                print("***************************")
                                ##TODO: implement this in the trajectory, as opposed to post-processing
                            if suppFlag and not fractionFlag:
                                #A + B flags
                                #atom flagged as supplemental to dataset, but is not the fractionally mixed atom
                                reaction = 'reactant'
                            elif suppFlag and fractionFlag:
                                # --> AB
                                #fractionally mixed, end result of A+B->AB, so 2
                                reaction = 2

                        #CALCULATION
                        print("SCF CALCULATION")
                        results = scf_wrap(scf, dm_in, matrices, sc, molecule=molecule)
                        if results == None:
                            #failed calculation, break out of this molecule's loop and continue with next
                            fails.append({"Epoch": epoch, 'Training Label':molecule, 'Molecules': [atoms[idx].get_chemical_formula() for idx in molecules[molecule]],
                        'Current Sub-Molecule': atoms[idx].get_chemical_formula()})
                            break
                        print("E_REF: {}".format(e_ref))
                        print("E_PRED: {}".format(results['E']))
                        #Add matrix keys to results dict
                        print("Adding reference matrices to results.")
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
                        print("================================")
                        print("RESULTS MATRICES SHAPES")
                        for k,v in results.items():
                            if k == 'fcenter':
                                #chance this isn't in matrices, so None in results, so just skip
                                continue
                            print("{}   ---   {}".format(k, v.shape))
                        print("================================")

                        #If radical, multiplicative factor
                        if atoms[idx].info.get('radical', False):
                            results['rho'] *= args.radical_factor
                            results['dm'] *= args.radical_factor
                        #If molecule and self-consistent, use "mol_losses" dict
                        #Only rho loss
                        if len(atoms[idx].positions) > 1 and sc:
                            print("MOL_LOSSES")
                            losses = mol_losses
                        #Else, if chosen atom is either H or Li, and args specify use both, and the H/Li not involved in reaction
                        #Rho and energy loss
                        elif str(atoms[idx].symbols) in ['H', 'Li'] and args.hnorm and not reaction:
                            print("H_LOSSES")
                            losses = h_losses
                        #Otherwise, if just an atom not in a reaction:
                        #Only energy loss -- E_ref for atoms is total atomic, not atomization
                        elif sc and not reaction:
                            print("ATM_LOSSES")
                            losses = atm_losses
                        #Else empty loss dict if reaction or not sc
                        else:
                            losses = {}
                        #if choose to force density or e loss, manually add back in
                        if args.forcedensloss:
                            if 'rho' not in losses.keys():
                                losses['rho'] = (partial(density_loss,loss = torch.nn.MSELoss()), args.rho_weight)
                        if args.forceEloss:
                            if 'E' not in losses.keys():
                                losses['E'] = (partial(energy_loss, loss = torch.nn.MSELoss()), args.E_weight)

                        #For each key in whichever loss dict chosen,
                        #Select the function (it's a tuple of itself), feed in results dict, normalize by number of atoms
                        losses_eval = {key: losses[key][0](results)/a_count[idx] for key in losses}
                        print("LOSSES_EVAL: ", losses_eval)
                        #Update running losses with new losses
                        running_losses.update({key:running_losses[key] + losses_eval[key].item() for key in losses})
                        
                        #IF Reaction type is 2, it is an A+B -> AB reaction.
                        #Store the dataset e_ref as ref, and results E as prediction
                        if reaction == 2:
                            print("REACTION TYPE: 2. A+B -> AB")
                            ref_dict['AB'] = e_ref
                            #if sc, get last skip_steps of scf cycle energies
                            #otherwise, get energy as list
                            if sc:
                                pred_dict['AB'] = results['E'][skip_steps:]
                            else:
                                pred_dict['AB'] = results['E'][-1:]

                        #ELSE if Reaction type is 1, it is an A->A reaction with some charge difference,
                        #Typically, reactant is charged so reaction == 1 is neutral
                        elif reaction == 1:
                            print("REACTION TYPE: 1. A -> A")
                            ref_dict['AA'] = e_ref
                            #if sc, get last skip_steps of scf cycle energies
                            #otherwise, get energy as list
                            if sc:
                                pred_dict['AA'] = results['E'][skip_steps:]
                            else:
                                pred_dict['AA'] = results['E'][-1:]
                        #ELSE IF it is a reactant in either of the above pathways,
                        elif reaction == 'reactant':
                            print("REACTION TYPE: REACTANT.")
                            #If self-consistent,
                            if sc:
                                label = 'A' if not 'A' in ref_dict else 'B'
                                ref_dict[label] = e_ref
                                pred_dict[label] = results['E'][skip_steps:]
                                if fractionFlag:
                                    #Efract = (1-f)*En + f*Ec
                                    if charge == 0:                                    
                                        pred_dict[label] = (1-fractionFlag)*results['E'][skip_steps:]
                                    elif charge == 1:
                                        pred_dict[label] = (fractionFlag)*results['E'][skip_steps:]
                            else:
                                label = 'A' if not 'A' in ref_dict else 'B'
                                ref_dict[label] = e_ref
                                pred_dict[label] = results['E'][-1:]
                                if fractionFlag:
                                    if charge == 0:                                    
                                        pred_dict[label] = (1-fractionFlag)*results['E'][-1:]
                                    elif charge == 1:
                                        pred_dict[label] = (fractionFlag)*results['E'][-1:]
                        
                        #If not reaction 2, 1, reactant, and molecule has more than one atom, e_ref is reference energy
                        elif len(atoms[idx].positions) > 1:
                            ref_dict[''.join(atoms[idx].get_chemical_symbols())] = e_ref
                            if sc:
                                steps = skip_steps
                            else:
                                steps = -1    
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][steps:]
                        #Else if not reaction 2, 1, reactant, and is single atom, ref_en if e_ref
                        else:
                            #ref_dict[''.join(atoms[idx].get_chemical_symbols())] = torch.zeros_like(e_ref)
                            ref_dict[''.join(atoms[idx].get_chemical_symbols())] = e_ref
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][skip_steps:]
                        loss += sum([losses_eval[key]*losses[key][1] for key in losses])
                        print(loss)
                    if not results:
                        continue
                    
                    print("LOOP OVER SUBMOLECULES COMPLETED")
                    print("REF_DICT: ", ref_dict)
                    print("PRED_DICT: ", pred_dict)
                    ael = ae_loss(ref_dict,pred_dict)
                    running_losses['ae'] += ael.item()
                    print('AE loss', ael.item())
                    if mol_sc:
                        running_losses['ae'] += ael.item()
                        loss += ael
                    else:
                        loss += args.nonsc_weight * ael
                        running_losses['ae'] += args.nonsc_weight * ael.item()
                    total_loss += loss.item()
                    print("Backward Propagation")
                    loss.backward()
                    if args.checkgrad:
                        for p in scf.xc.parameters():
                            if p.requires_grad:
                                print('===========\ngradient\n----------\nmax: {}\nmin: {}'.format(torch.max(p.grad), torch.min(p.grad)))
                    print("Step Optimizer")
                    optimizer.step()
                    print("Zeroing Optimizer Grad")
                    optimizer.zero_grad()
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
            print("++++++++++++++++++++++++++++++++++")
            print("FAILS:")
            for i in fails:
                print(i)
            print("++++++++++++++++++++++++++++++++++")

            running_losses = {key:np.sqrt(running_losses[key]/len(molecules))*1000 for key in running_losses}
            total_loss = np.sqrt(total_loss/len(molecules))*1000
            best_loss = min(total_loss, best_loss)
            chkpt_str = ''
            torch.save(scf.xc.state_dict(), logpath + '_current.chkpt')
            torch.save(scf, logpath + '_current.pt')
            if total_loss == best_loss:
                torch.save(scf.xc.state_dict(), logpath + '_{}.chkpt'.format(chkpt_idx%3))
                torch.save(scf, logpath + '_{}.pt'.format(chkpt_idx%3))
                torch.save(optimizer.state_dict(), logpath + '_{}.adam.chkpt'.format(chkpt_idx%3))
                chkpt_str = '_{}.chkpt'.format(chkpt_idx%3)
                chkpt_idx += 1
            print("============================================================")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('Epoch {} ||'.format(epoch), [' {} : {:.6f}'.format(key,val) for key, val in running_losses.items()],
                  '|| total loss {:.6f}'.format(total_loss),chkpt_str)
            if HYBRID:
                print("HYB MIXING:")
                print(scf.xc.exx_a)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("============================================================")

            scheduler.step(total_loss)
