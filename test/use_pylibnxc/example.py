import numpy as np
import torch, os
from ase.io import read, write
from ase import Atoms
from dpyscfl.utils import ase_atoms_to_mol, gen_mf_mol, get_spin

sel = Atoms(['O'], positions = [[0,0,0]])
sel.info = {'spin': 2}

#pylibnxc
import pylibnxc
from pylibnxc.pyscf import RKS, UKS
#modify path as needed
dpyscf_lite_dir = '/home/awills/Documents/Research/dpyscfl'
modelpath = os.path.join(dpyscf_lite_dir, 'models/xcdiff/MODEL_MGGA/xc')
#make directory pylibnxc will use to find network
try:
    os.mkdir('MGGA_XC_CUSTOM')
except:
    pass
#symlink xcdiff to directory above
try:
    os.symlink(modelpath, 'MGGA_XC_CUSTOM/xc')
except:
    pass

#call ase_atoms_to_mol to go from trajectory molecule to pyscf mol
name, mol = ase_atoms_to_mol(sel, basis='def2-qzvp', charge=sel.info.get('charge', 0), spin=sel.info.get('spin', None))

#use pylibnxc's wrapper, UKS or RKS
mf = UKS(mol, nxc='MGGA_XC_CUSTOM', nxc_kind='grid')
mf.kernel()

print('Pylibnxc (XCDiff) Evaluated on:\n{} = {}'.format(name, mf.e_tot))