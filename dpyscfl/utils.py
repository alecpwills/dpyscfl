import numpy as np
import scipy
from pyscf import gto, dft, scf , df
from pyscf.dft import radi
from ase.io import read
import pandas as pd
import pickle, os, sys

def get_datapoint(mol, mf, dfit = False, init_guess=False, do_fcenter=True):
    """
    Builds all matrices needed for SCF calculations (the ones considered constant)

    Args:
        mol (:class:`pyscf.gto.Mol`): pyscf molecule object
        mf (:class:`pyscf.scf.X`): PREVIOUS RUN kernel for scf calculation, e.g. scf.RKS(mol)
        dfit (:class:`bool`, optional): Whether or not to use density fitting. Defaults to False.
        init_guess (:class:`bool`, optional): Whether to use pyscf's guessed DM as a start. Defaults to False.
        do_fcenter (:class:`bool`, optional): Whether to do calculate this integral. Defaults to True.

    .. todo:: Look up what this integral is.

    Returns:
        dict: dictionary with key/value pairs of input data matrices relating to given molecule

        keys: dm_init, v, t, s, n_elec, n_atoms, e_nuc, mo_energy, mo_occ
        
        if dfit: keys include df_2c_inv, df_3c

        else: keys include eri
    """
    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    eri = mol.intor('int2e')
    if do_fcenter:
        fcenter = mol.intor('int4c1e')

    if dfit:
        auxbasis = df.addons.make_auxbasis(mol,mp2fit=False)
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        df_3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1', comp=1)
        df_2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        df_2c_inv = scipy.linalg.pinv(df_2c)
    else:
        eri = mol.intor('int2e')

    dm_base = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    if init_guess:
        dm_init = mf.get_init_guess()
    else:
        dm_init = np.array(dm_base)
        dm_realinit = mf.get_init_guess()

    e_base = mf.energy_tot()
    e_ip = mf.mo_energy[...,:-1][np.diff(mf.mo_occ).astype(bool)]
    if len(e_ip) == 2:
        e_ip = e_ip[0]
        ip_idx = np.where(np.diff(mf.mo_occ).astype(bool))[1][0]
    else:
        ip_idx = np.where(np.diff(mf.mo_occ).astype(bool))[0]


    n_elec = np.sum(mol.nelec)
    n_atoms = mol.natm

    matrices = {'dm_init':dm_init,
                'v':v,
                't':t,
                's':s,
                'eri':eri,
                'n_elec':n_elec,
                'n_atoms':n_atoms,
                'e_nuc': np.array(mf.energy_nuc()),
                'mo_energy': np.array(mf.mo_energy),
                'mo_occ': np.array(mf.mo_occ),
                'e_ip':e_ip,
                'ip_idx':ip_idx,
                'e_pretrained': e_base,
                'e_base':e_base}
    if not init_guess:
        matrices.update({'dm_realinit': dm_realinit})

    if dfit:
        matrices.update({'df_2c_inv':df_2c_inv,
                         'df_3c': df_3c})
    else:
        matrices.update({'eri':eri})

    features = {}
    features.update({'L': np.eye(dm_init.shape[-1]), 'scaling': np.ones([dm_init.shape[-1]]*2)})

    features.update({'ao_eval':mf._numint.eval_ao(mol, mf.grids.coords, deriv=2),
                    'grid_weights':mf.grids.weights})


    matrices.update(features)

    return matrices

class Dataset(object):
    def __init__(self, **kwargs):
        """Dataset class that serves as input to pytorch DataLoader and stores
        all the matrices defining a given system (molecule)

        Required arguments of kwargs:
            dm_init, v, t, s, n_elec, e_nuc
        """
        needed = ['dm_init','v','t','s','n_elec','e_nuc']
        for n in needed:
            assert n in kwargs

        attrs = []
        for name, val in kwargs.items():
            attrs.append(name)
            setattr(self, name,val)
        self.attrs = attrs

    def __getitem__(self, index):
        """Returns attributes of Dataset molecule with given index.

        Adds the inverse Cholesky decomposition of the overlap matrix S to the returned matrices object.

        Args:
            index (:class:`int`): index of molecule to return attributes of

        Returns:
            tuple: dm_init, matrices object containing the attributes.
        """
        mo_occ = self.mo_occ[index]
        s_chol = np.linalg.inv(np.linalg.cholesky(self.s[index]))

        matrices = {attr: getattr(self,attr)[index] for attr in self.attrs}
        dm_init = matrices.pop('dm_init')
        matrices['mo_occ'] = mo_occ
        matrices['s_chol'] = s_chol
        if hasattr(self, 'ml_ovlp'):
            ml_ovlp = self.ml_ovlp[index]
            ml_ovlp = ml_ovlp.reshape(*ml_ovlp.shape[:2],self.n_atoms[index], -1)
            matrices['ml_ovlp'] = ml_ovlp
        return dm_init, matrices

    def __len__(self):
        """Returns length of self.dm_init, or number of molecules in dataset

        Returns:
            int: length of dataset: len(self.dm_init)
        """
        return len(self.dm_init)

def ase_traj_to_mol(traj, basis='6-311++G(3df,2pd)', charge=0, spin=None):
    """Converts a list of ASE.Atoms objects into a dictionary of mol objects.

    Args:
        traj (list, str): If string, assume path to a file to read in with ASE. If a list, assumes already a list of Atoms objects.
        basis (str, optional): Basis set to assign in PySCF. Defaults to '6-311++G(3df,2pd)'.
        charge (int, optional): Global charge of molecule. Defaults to 0.
        spin (int, optional): Specify spin if desired. Defaults to None, which has PySCF guess spin based on electron number/occupation.

    Returns:
        dict: {name:mol} for atoms in traj
    """
    if type(traj) == str:
        traj = read(traj, ':')
    
    moldct = {}
    for atom in traj:
        name, mol = ase_atoms_to_mol(atom, basis, charge, spin)
        moldct[name] = mol
    
    return moldct

def ase_atoms_to_mol(atoms, basis='6-311++G(3df,2pd)', charge=0, spin=None):
    """Converts an ASE.Atoms object into a PySCF gto.Mol object

    Args:
        atoms (:class:`ASE.Atoms`): ASE.Atoms object of a single molecule/system
        basis (str, optional): Basis set to assign in PySCF. Defaults to '6-311++G(3df,2pd)'.
        charge (int, optional): Global charge of molecule. Defaults to 0.
        spin (int, optional): Specify spin if desired. Defaults to None, which has PySCF guess spin based on electron number/occupation.

    Returns:
        (str, :class:`pyscf.gto.Mole`): chemical formula, the mole object of the converted ASE.Atoms
    """
    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    name = atoms.get_chemical_formula()

    mol_input = [[ispec, ipos] for ispec,ipos in zip(spec,pos)]
    c = atoms.info.get('charge', charge)
    s = atoms.info.get('spin', spin)
    mol = gto.M(atom=mol_input, basis=basis, spin=s, charge=c)

    return name, mol

def gen_mf_mol(mol, xc='', pol=False, grid_level = None):
    """Generate a pyscf calculation kernel based on provided inputs.

    Args:
        mol (:class:`pyscf.gto.mole`): The molecule that will be calculated on. Spin of this important in assigning method.
        xc (str, optional): The XC functional to use (if DFT). Defaults to '', thus HF.
        pol (bool, optional): Whether to force spin polarization. Only useful if spin = 0. Defaults to False.
        grid_level (int, optional): Grid level to set if DFT calculation. Defaults to None, which sets grids to have level 5.

        .. todo:: Change grid_level behavior, use to toggle behavior in old_get_datapoint

    Returns:
        (pyscf calculation kernel, reference to method): (mf, method)
    """
    if (mol.spin == 0 and not pol):
        #if zero spin and we specifically don't want spin-polarized
        #or if just neutral H atom
        if xc:
            #if xc functional specified, RKS
            method = dft.RKS
        else:
            #else assume RHF
            method = scf.RHF
    else:
        #if net spin, must do spin polarized
        if xc:
            #xc functional specified, UKS
            method = dft.UKS
        else:
            #none specified, UHF
            method = scf.UHF
    mf = method(mol)

    if xc:
        print("Building grids...")
        mf.xc = xc
        mf.grids.level = grid_level if grid_level else 5
        mf.grids.build()
    print("METHOD GENERATED: {}".format(method))
    return mf, method

#Past here: previous dpyscf util functions
#TODO: Bring to current status -- namely, get_datapoint is already defined so rewrite for consistency
def get_ml_ovlp(mol, auxmol):
    """_summary_

    Concatenates the objects, finds the total number of contracted GTOs for each,
    then generates the integral for the 3-center 1-electron overlap integral.

    Args:
        mol (:class:`pyscf.gto.Mole`): _description_
        auxmol (:class:`pyscf.gto.Mole`): _description_

    Returns:
        _type_: _description_
    """
    pmol = mol + auxmol
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    eri3c = pmol.intor('int3c1e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas))

    return eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)


class MemDatasetWrite(object):
    def __init__(self, loc='data/', **kwargs):
        """Writes object data to specified location, and loops over items in object passed.
            
            For each item, kwarg values are assembled and loc+data_{ind}.pckl is written.
            Required kwargs: ['dm_init', 'v', 't', 's', 'eri', 'n_elec', 'e_nuc', 'Etot', 'dm']


        Args:
            loc (str, optional): location where to write dataset. Defaults to 'data/'.
        """
        self.loc = os.path.join(loc,'data')
        needed = ['dm_init','v','t','s','eri','n_elec','e_nuc','Etot','dm']
        for n in needed:
            assert n in kwargs

        keys = list(kwargs.keys())
        self.length = len(kwargs[keys[0]])
        np.save(self.loc + '_len.npy',self.length )
        for idx in range(self.length):
            datapoint = {}
            for key in keys:
                datapoint[key] = kwargs[key][idx]
            with open(self.loc + '_{}.pckl'.format(idx),'wb') as datafile:
                datafile.write(pickle.dumps(datapoint))

class MemDatasetRead(object):
    """Reads saved Dataset into kwarg values
    """
    def __init__(self, loc='data/',skip=[], **kwargs):
        """Reads saved Dataset into kwarg values

        Args:
            loc (str, optional): Location of top directory containing dataset. Defaults to 'data/'.
            skip (list, optional): List of integers to skip when reading data. Defaults to [].
        """
        self.loc = os.path.join(loc,'data')
        self.length = int(np.load(self.loc + '_len.npy'))
        index_map = np.arange(self.length)
        if skip:
            index_map = np.delete(index_map, np.array(skip))
        self.index_map = index_map
        self.length = len(index_map)

    def __getitem__(self, index):
        """Reads pickled data into memory

        Args:
            index (int): index of object in dataset to return

        Returns:
            tuple: dm_init, matrices, Etot, dm
        """
        index = self.index_map[index]
        with open(self.loc + '_{}.pckl'.format(index),'rb') as datafile:
            kwargs = pickle.loads(datafile.read())

        attrs = []
        for name, val in kwargs.items():
            attrs.append(name)
            setattr(self, name,val)
        self.attrs = attrs

        mo_occ = self.mo_occ
        s_oh = np.linalg.inv(np.linalg.cholesky(self.s))
        s_inv_oh = s_oh.T

        matrices = {attr: getattr(self,attr)for attr in self.attrs}
        dm_init = matrices.pop('dm_init')
        #TODO: remove these commented, Dataset no longer pops these
        #Etot = matrices.pop('Etot')
        #dm = matrices.pop('dm')
        matrices['mo_occ'] = mo_occ
        matrices['s_inv_oh'] = s_inv_oh
        matrices['s_oh'] = s_oh
        if hasattr(self, 'ml_ovlp'):
            ml_ovlp = self.ml_ovlp
            ml_ovlp = ml_ovlp.reshape(*ml_ovlp.shape[:2],self.n_atoms, -1)
            matrices['ml_ovlp'] = ml_ovlp
        return dm_init, matrices#, Etot, dm

    def __len__(self):
        return self.length


def get_rho(mf, mol, dm, grids):
    """Generates density on a grid provided

    .. todo:: Make sure this is still how it works in dpyscf-lite

    Args:
        mf (pyscf kernel): PREVIOUSLY RUN kernel
        mol (:class:`pyscf.GTO.Mol`): molecule whose atomic orbitals generate rho
        dm (np.array): density matrix
        grids (_type_): _description_

    Returns:
        np.array: density on provided grid
    """
    print("Evaluating atomic orbitals on grid.")
    ao_eval = mf._numint.eval_ao(mol, grids.coords)
    print("AO Shape: {}. DM Shape: {}".format(ao_eval.shape, dm.shape))
    print("Evaluating density on grid.")
    rho = mf._numint.eval_rho(mol, ao_eval, dm, verbose=3)
    return rho


def old_get_datapoint(atoms, xc='', basis='6-311G*', ncore=0, grid_level=0,
                  nl_cutoff=0, grid_deriv=1, init_guess = False, ml_basis='',
                  do_fcenter=True, zsym = False, n_rad=20,n_ang=10, spin=0, pol=False,
                  ref_basis='', ref_path = '', ref_index=0, dfit=True):
    """_summary_

    .. todo:: Refactor to be in line with current implementation, get_datapoint exists already.

    .. todo:: If ref_path specified, reading in the baseline energy from reference assumes 'results.traj' specified, with atoms.calc.result.energy specified.

    Args:
        atoms (:class:`ASE.Atoms`): The list of atoms to calculate datapoints for
        xc (str, optional): Functional to feed to pyscf. Defaults to ''.
        basis (str, optional): The basis to tell pyscf to use. Defaults to '6-311G*'.
        ncore (int, optional): _description_. Defaults to 0.
        grid_level (int, optional): _description_. Defaults to 0.
        nl_cutoff (int, optional): _description_. Defaults to 0.
        grid_deriv (int, optional): _description_. Defaults to 1.
        init_guess (bool, optional): _description_. Defaults to False.
        ml_basis (str, optional): _description_. Defaults to ''.
        do_fcenter (bool, optional): _description_. Defaults to True.
        zsym (bool, optional): _description_. Defaults to False.
        n_rad (int, optional): _description_. Defaults to 20.
        n_ang (int, optional): _description_. Defaults to 10.
        spin (int, optional): Spin of the system. Defaults to 0.
        pol (bool, optional): Spin polarized calculation. Defaults to False.
        ref_basis (str, optional): _description_. Defaults to ''.
        ref_path (str, optional): _description_. Defaults to ''.
        ref_index (int, optional): _description_. Defaults to 0.

    Returns:
        (float, np.array, dict): (Baseline energy from mf.energy_tot(), 3D Identity matrix, dict of matrices generated)
    """

    print(atoms)
    print(basis)

    if not ref_basis:
        ref_basis = basis

    if atoms.info.get('openshell',False) and spin ==0:
        spin = 2
    

    features = {}
    
    _, mol = ase_atoms_to_mol(atoms, basis=basis, charge=0, spin=spin)
    _, mol_ref = ase_atoms_to_mol(atoms, basis=ref_basis, charge=0, spin=spin)

    if ml_basis:
        auxmol = ase_atoms_to_mol(atom=atoms,spin=spin, basis=gto.parse(open(ml_basis,'r').read()))
        ml_ovlp = get_ml_ovlp(mol,auxmol)
        features.update({'ml_ovlp':ml_ovlp})

    mf, method = gen_mf_mol(mol, xc=xc, pol=pol, grid_level=grid_level)
    mf.kernel()

    matrices = get_datapoint(mol=mol, mf=mf, dfit=dfit, init_guess=init_guess, do_fcenter=do_fcenter)
    dm_init = matrices['dm_init']
    e_base = matrices['e_base']

    if ref_path:
        if atoms.info.get('sc', True) and not atoms.info.get('reaction', False):
            print('Loading reference density')
            dm_base = np.load(ref_path+ '/{}_{}.dm.npy'.format(ref_index, atoms.get_chemical_formula()))
            print("Reference DM loaded. Shape: {}".format(dm_base.shape))
        if method == dft.UKS and dm_base.ndim == 2:
            dm_base = np.stack([dm_base,dm_base])*0.5
        if method == dft.RKS and dm_base.ndim == 3:
            dm_base = np.sum(dm_base, axis=0)
        dm_guess = dm_init
        dm_init = dm_base
        print("SHAPES: DM_GUESS = {}, DM_BASE = {}".format(dm_guess.shape, dm_init.shape))
        #TODO: Amend the way this is done
        e_base = read(os.path.join(ref_path, 'results.traj'), ref_index).calc.results['energy']
        #e_base =  (pd.read_csv(ref_path + '/energies', delim_whitespace=True,header=None,index_col=0).loc[ref_index]).values.flatten()[0]

    if grid_level:
        print("GRID GENERATION.")
        print("STATS: GRID_LEVEL={}. ZYM={}. NL_CUTOFF={}. SPIN(INP/MOL)={}/{}. POL={}.".format(grid_level, zsym, nl_cutoff, spin, mol.spin, pol))
        if zsym and not nl_cutoff:
            if matrices['n_atoms'] == 1:
                method = line
            else:
                method = half_circle
            mf.grids.coords, mf.grids.weights, L, scaling = get_symmetrized_grid(mol, mf, n_rad, n_ang, method=method)
            features.update({'L': L, 'scaling': scaling})
        if (mol.spin != 0) or (pol):
            print("Generating spin-channel densities.")
            rho_a = get_rho(mf, mol_ref, dm_init[0], mf.grids)
            rho_b = get_rho(mf, mol_ref, dm_init[1], mf.grids)
            rho = np.stack([rho_a,rho_b],axis=0)
        else:
            print("Generating non-polarized density.")
            rho = get_rho(mf, mol_ref, dm_init, mf.grids)

        features.update({'rho': rho,
                         'ao_eval':mf._numint.eval_ao(mol, mf.grids.coords, deriv=grid_deriv),
                         'grid_weights':mf.grids.weights})

    matrices.update(features)

    return e_base, np.eye(3), matrices




def half_circle(mf, mol, level, n_ang = 25):
    """_summary_

    Args:
        mf (_type_): _description_
        mol (_type_): _description_
        level (_type_): _description_
        n_ang (int, optional): _description_. Defaults to 25.
    """
    atom_grids_tab = gen_atomic_grids(mol,level=level,nang=n_ang)

    coords, weights = dft.gen_grid.gen_partition(mol, atom_grids_tab,radi.treutler_atomic_radii_adjust, atomic_radii=radi.BRAGG_RADII, becke_scheme=dft.gen_grid.stratmann)

    g = dft.gen_grid.Grids(mol)
    g.coords = coords
    g.weights = weights

    dm = mf.make_rdm1()
    if dm.ndim ==3:
        dm = np.sum(dm,axis=0)
    pruned = dft.rks.prune_small_rho_grids_(mf,mol,dm, g)
#     pruned = g
    print('Number of grid points (level = {}, n_ang = {}):'.format(level,n_ang), len(pruned.weights))
    coords = pruned.coords
    weights = pruned.weights
    
    return coords, weights
