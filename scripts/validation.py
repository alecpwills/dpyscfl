import pyscf
from pyscf import gto,dft,scf
import numpy as np
import scipy
from ase import Atoms
from ase.io import read
import pylibnxc
import sys
import pickle
from dpyscfl.utils import get_rho
import os
from ase.units import Hartree
from test_bh76 import test_bh
from ase.units import Hartree,kcal,mol
from dpyscfl.losses import *
kcalpmol = kcal/mol
basis = '6-311++G(3df,2pd)'
# basis = '3-21G'


# systems = [113, 25, 18, 114, 0] #Training
systems = [103, 14, 23, 5, 10, 79, 27, 105] #Validation
# systems += [23, 84, 144, 21, 116, 106, 15, 58] #Expanded training
# systems = [5] #Validation
# systems = [103, 14]
bh_systems = [0, 65]
# bh_systems = [1]
#bh_systems = [1]
# systems = [50]
ref_path = '/home/awills/Documents/Research/dpyscfl/data/haunschild_g2'
atoms = read(os.path.join(ref_path, 'g2_97.traj'),':')
atoms = [atoms[s] for s in systems]
symbols = {idx: atom.symbols for idx,atom in enumerate(atoms)}
e_ref = -np.array([a.get_potential_energy()/Hartree for a in atoms])
data_path = '/home/awills/Documents/Research/swxcd/seawulf/g2_97_3'
dm_refs = [np.load('../data/ccsdt/{}_{}.dm.npy'.format(s, symbols[s])) for s in systems]

N = len(atoms)

spins = {
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
dm_refs += [None for a in np.unique([s for a in atoms for s in a.get_chemical_symbols() ])]
atoms += [Atoms(a, info={'spin':spins[a]}) for a in np.unique([s for a in atoms for s in a.get_chemical_symbols() ])]


def run_validate(nxc='MGGA_tmp', xc_type='nxc', do_bh=False):

    from_listener = False
    if not isinstance(nxc,str):
        from_listener = True
        nxc.evaluate()
        if nxc.level == 2:
            tmpmod = 'GGA_TMP'
        elif nxc.level == 3:
            tmpmod = 'MGGA_TMP'
        else:
            raise Exception("Something went wrong here")

        nxc.forward_old = nxc.forward
        nxc.forward = nxc.eval_grid_models
        traced = torch.jit.trace(nxc, torch.rand(100,9))
       
        exx_a = 0
        try:
            exx_a = nxc.exx_a.detach().numpy()
        except:
            exx_a = 0 
            
        if exx_a:
            tmpmod += '0'
        try:
            os.mkdir(tmpmod)
        except FileExistsError:
            pass

        torch.jit.save(traced, tmpmod + '/xc')

        nxc.forward = nxc.forward_old
        nxc.train()
        nxc_type = 'nxc'
        if exx_a:
            nxc = '{}*HF +'.format(exx_a[0]) + tmpmod
        else:
            nxc = tmpmod

    pred_dict = {}
    dm_loss = []
    for a, dm_ref in zip(atoms,dm_refs):
        pos = a.positions
        this_basis = basis
        spec = a.get_chemical_symbols()
        mol_input = [[s, p] for s, p in zip(spec, pos)]
        if len(pos) == 1:
            spin = spins[spec[0]]
        else:
            if a.info['openshell']:
                spin = 2
            else:
                spin = 0
        try:
            mol = gto.M(atom=mol_input, basis=this_basis,spin=spin)
        except Exception:
            spin =1
            mol = gto.M(atom=mol_input, basis=this_basis,spin=spin)

        if spin == 0:
            method = pylibnxc.pyscf.RKS
        else:
            method = pylibnxc.pyscf.UKS
        
        mol.verbose=4
        if xc_type == 'nxc':
            mf = method(mol, nxc=nxc, nxc_kind='grid')
        elif xc_type == 'xc':
            mf = method(mol)
            mf.xc = nxc

        mf.grids.level=5
        mf.kernel()
        dm_predicted = mf.make_rdm1()
        pred_dict[''.join(a.get_chemical_symbols())] = mf.e_tot
        if len(pos) > 1:
            if dm_ref.ndim == 2:
                dm_ref = np.stack([dm_ref,dm_ref],axis=0)*.5
            if dm_predicted.ndim == 2:
                dm_predicted = np.stack([dm_predicted,dm_predicted],axis=0)*.5
            rho_predicted = get_rho(mf, mol, np.sum(dm_predicted, axis=0), mf.grids)
            rho_ref = get_rho(mf, mol, np.sum(dm_ref, axis=0), mf.grids)
            dev = np.sqrt(np.sum((rho_predicted - rho_ref)**2*mf.grids.weights))/(np.sum(rho_predicted*mf.grids.weights))
            dm_loss.append(dev)

    ae_pred = atomization_energies(pred_dict)
    pred = np.array([ae_pred[''.join(a.get_chemical_symbols())] for a in atoms if len(a.positions) > 1])

    if not from_listener:
        np.save('ae.npy', pred)
        np.save('dmloss.npy',dm_loss)
    
    dm_loss = np.mean(dm_loss)
    ae_loss = np.mean(np.abs(pred - e_ref))
    if do_bh:
        
        barrier_heights, reference_heights = test_bh(nxc, nxc='nxc', indices=bh_systems, ref_dir='./')
        bh_loss = np.mean(np.abs(np.array(barrier_heights)-np.array(reference_heights)))*kcalpmol/Hartree
        ae_loss = (ae_loss*len(pred) + bh_loss*len(barrier_heights))/(len(pred)+len(barrier_heights))
        
    return ae_loss, dm_loss

if __name__ == '__main__':
    ae_loss, dm_loss = run_validate(*sys.argv[1:])
    barrier_heights, reference_heights = test_bh(*sys.argv[1:], indices=bh_systems)
    bh_loss = np.mean(np.abs(np.array(barrier_heights)-np.array(reference_heights)))
#     print(barrier_heights)
#     print(reference_heights)
    print(ae_loss*Hartree/kcalpmol, dm_loss, bh_loss)
#    print(ae_loss*Hartree/kcalpmol, dm_loss)
