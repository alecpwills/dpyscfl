import numpy as np
from dpyscfl.scf import energy_tot
import scipy
from pyscf import gto, dft, scf , df
import pylibnxc
from pyscf.dft import radi
from ase.io import read
import pandas as pd
import pickle, os, sys

def get_datapoint(mol, mf, dfit = True, init_guess=False, do_fcenter=True):
    """
    Builds all matrices needed for SCF calculations (the ones considered constant)

    Args:
        mol (:class:`pyscf.gto.Mol`): pyscf molecule object
        mf (:class:`pyscf.scf.X`): PREVIOUSLY RUN kernel for scf calculation, e.g. scf.RKS(mol)
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
    print("DATAPOINT E_BASE: ", e_base)
    print("DATAPOINT MO_ENERGY: ", mf.mo_energy)
    print("DATAPOINT E_IP: ", e_ip)
    print("DATAPOINT MO_OCC: ", mf.mo_occ)
    if len(e_ip) == 2 and len(mf.mo_occ) == 2:
        e_ip = e_ip[0]
        ip_idx = np.where(np.diff(mf.mo_occ).astype(bool))[1][0]
    else:
        ip_idx = np.where(np.diff(mf.mo_occ).astype(bool))[0]


    n_elec = np.sum(mol.nelec)
    n_atoms = mol.natm

    matrices = {'dm_init':dm_init,
                'dm':dm_base,
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
    if do_fcenter:
        matrices.update({'f_center':fcenter})

    features = {}
    features.update({'L': np.eye(dm_init.shape[-1]), 'scaling': np.ones([dm_init.shape[-1]]*2)})
    if mf.xc:
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
        charge = atom.info.get('charge', charge)
        spin = atom.info.get('spin', spin)
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

def gen_mf_mol(mol, xc='', pol=False, grid_level = None, nxc = False):
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
    if nxc:
        dftns = pylibnxc.pyscf
        hfns = scf
    else:
        dftns = dft
        hfns = scf
    if (mol.spin == 0 and not pol):
        #if zero spin and we specifically don't want spin-polarized
        #or if just neutral H atom
        if xc:
            #if xc functional specified, RKS
            method = dftns.RKS
        else:
            #else assume RHF
            method = hfns.RHF
    else:
        #if net spin, must do spin polarized
        if xc:
            #xc functional specified, UKS
            method = dftns.UKS
        else:
            #none specified, UHF
            method = hfns.UHF
    mf = method(mol)

    if xc:
        print("Building grids...")
        mf.xc = xc
        mf.grids.level = grid_level if grid_level else 5
        mf.grids.build()
    else:
        mf.xc = ''
    print("METHOD GENERATED: {}".format(method))
    return mf, method

def fractional_matrices_combine(base, charge, fractional):
    print("FRACTIONAL SHAPES:")
    print("BASE DM: ", base['dm'].shape)
    print("CHARGE DM: ", charge['dm'].shape)
    matricesFrac = {}
    for key in list(base.keys()):
        print(key)
        matricesFrac[key] = (1-fractional)*base[key]+fractional*charge[key]
    return matricesFrac


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
        matrices['s_chol'] = s_oh
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
                  do_fcenter=True, zsym = False, n_rad=20,n_ang=10, spin=None, pol=False,
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
    fractional = atoms.info.get('fractional', None)
    charge=atoms.info.get('charge', 0)

    features = {}
    
    _, mol = ase_atoms_to_mol(atoms, basis=basis, charge=charge, spin=spin)
    _, mol_ref = ase_atoms_to_mol(atoms, basis=ref_basis, charge=charge, spin=spin)

    if ml_basis:
        auxmol = ase_atoms_to_mol(atom=atoms,spin=spin, basis=gto.parse(open(ml_basis,'r').read()))
        ml_ovlp = get_ml_ovlp(mol,auxmol)
        features.update({'ml_ovlp':ml_ovlp})

    mf, method = gen_mf_mol(mol, xc=xc, pol=pol, grid_level=grid_level)
    mf.kernel()

    matrices = get_datapoint(mol=mol, mf=mf, dfit=dfit, init_guess=init_guess, do_fcenter=do_fcenter)

    if fractional:
        print("FRACTIONAL FLAG -- CALCULATING CHARGED SYSTEM")
        _, molFrac = ase_atoms_to_mol(atoms, basis=basis, charge=mol.charge+1, spin=None)
        #Must use same method as base atom for shape concerns
        mfFrac = method(molFrac)
        if xc:
            print("Building charged grids...")
            mfFrac.xc = xc
            mfFrac.grids.level = grid_level if grid_level else 5
            mfFrac.grids.build()

        mfFrac.kernel()
        matricesFrac = get_datapoint(mol=molFrac, mf=mfFrac, dfit=dfit, init_guess=init_guess, do_fcenter=do_fcenter)

        matrices = fractional_matrices_combine(matrices, matricesFrac, fractional)

    

    dm_init = matrices['dm_init']
    e_base = matrices['e_base']

    if fractional:
        assert np.isclose(e_base, (1-fractional)*mf.energy_tot()+fractional*mfFrac.energy_tot())
        assert atoms.info['baseRef'], "Atoms flagged as Fractional: Need Neutral Reference Path"
        assert atoms.info['chargeRef'], "Atoms flagged as Fractional: Need Charged Reference Path"
        if atoms.info.get('sc', True) and not atoms.info.get('reaction', False):
            print('Loading reference density')
            dm0_base = np.load(atoms.info['baseRef']+ '/{}_{}.dm.npy'.format(atoms.info['baseidx'], atoms.get_chemical_formula()))
            dmC_base = np.load(atoms.info['chargeRef']+ '/{}_{}.dm.npy'.format(atoms.info['baseidx'], atoms.get_chemical_formula()))
            dm_base = (1-fractional)*dm0_base + fractional*dmC_base
            print("Reference DM loaded. Shape: {}".format(dm_base.shape))
        if method == dft.UKS and dm_base.ndim == 2:
            dm_base = np.stack([dm_base,dm_base])*0.5
        if method == dft.RKS and dm_base.ndim == 3:
            dm_base = np.sum(dm_base, axis=0)
        dm_guess = dm_init
        dm_init = dm_base
        print("SHAPES: DM_GUESS = {}, DM_BASE = {}".format(dm_guess.shape, dm_init.shape))
        e0_base = read(os.path.join(atoms.info['baseRef'], 'results.traj'), atoms.info['baseidx']).calc.results['energy']
        eC_base = read(os.path.join(atoms.info['chargeRef'], 'results.traj'), atoms.info['baseidx']).calc.results['energy']
        e_base = (1-fractional)*e0_base + fractional*eC_base

    elif ref_path and not atoms.info.get('fractional', None):
        #why do we not use reference here when we still have it?
        #if atoms.info.get('sc', True) and not atoms.info.get('reaction', False):
        #    print('Loading reference density')
        #    dm_base = np.load(ref_path+ '/{}_{}.dm.npy'.format(ref_index, atoms.symbols))
        dm_base = np.load(ref_path+ '/{}_{}.dm.npy'.format(ref_index, atoms.symbols))
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
            #outdated, skip this by prior zsym = False
            if matrices['n_atoms'] == 1:
                method = line
            else:
                method = half_circle
            print("Getting symmetrized grid.")
            mf.grids.coords, mf.grids.weights, L, scaling = get_symmetrized_grid(mol, mf, n_rad, n_ang, method=method)
            features.update({'L': L, 'scaling': scaling})
            #re-evaluate on grid in case of symmetrization
            #TODO: Verify this is correct thing to do, hopefully doesn't mess anything up
            features.update({'ao_eval':mf._numint.eval_ao(mol, mf.grids.coords, deriv=2),
                        'grid_weights':mf.grids.weights})
        #If net spin or force polarized calculation
        if (mol.spin != 0) or (pol):
            print("Generating spin-channel densities.")
            rho_a = get_rho(mf, mol_ref, dm_init[0], mf.grids)
            rho_b = get_rho(mf, mol_ref, dm_init[1], mf.grids)
            rho = np.stack([rho_a,rho_b],axis=0)
        else:
            print("Generating non-polarized density.")
            rho = get_rho(mf, mol_ref, dm_init, mf.grids)

        #ao-eval, grid weights handled in get_datapoint
        #features.update({'rho': rho,
        #                 'ao_eval':mf._numint.eval_ao(mol, mf.grids.coords, deriv=grid_deriv),
        #                 'grid_weights':mf.grids.weights})
        features.update({'rho':rho})

    matrices.update(features)

    print("================================")
    print("GET DATAPOINT MATRICES SHAPES")
    for k,v in matrices.items():
        try:
            print("{}   ---   {}".format(k, v.shape))
        except:
            print("{}   ---   {}, no shape".format(k, v))
    print("================================")


    return e_base, np.eye(3), matrices

def gen_atomic_grids(mol, atom_grid={}, radi_method=radi.gauss_chebyshev,
                     level=3, nang=20, prune=dft.gen_grid.nwchem_prune, **kwargs):
    """Adapted from pyscf code.
    Generate number of radial grids and angular grids for the given molecule.


    Args:
        mol (_type_): _description_
        atom_grid (dict, optional): _description_. Defaults to {}.
        radi_method (_type_, optional): _description_. Defaults to radi.gauss_chebyshev.
        level (int, optional): _description_. Defaults to 3.
        nang (int, optional): _description_. Defaults to 20.
        prune (_type_, optional): _description_. Defaults to dft.gen_grid.nwchem_prune.

    Raises:
        ValueError: _description_

    Returns:
        A dict, with the atom symbol for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.

    """
    '''
    Adapted from pyscf code.
    Generate number of radial grids and angular grids for the given molecule.

    Returns:
        A dict, with the atom symbol for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.
    '''
    if isinstance(atom_grid, (list, tuple)):
        atom_grid = dict([(mol.atom_symbol(ia), atom_grid)
                          for ia in range(mol.natm)])
    atom_grids_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
                if n_ang not in LEBEDEV_NGRID:
                    if n_ang in LEBEDEV_ORDER:
                        logger.warn(mol, 'n_ang %d for atom %d %s is not '
                                    'the supported Lebedev angular grids. '
                                    'Set n_ang to %d', n_ang, ia, symb,
                                    LEBEDEV_ORDER[n_ang])
                        n_ang = LEBEDEV_ORDER[n_ang]
                    else:
                        raise ValueError('Unsupported angular grids %d' % n_ang)
            else:
                #ALEC
                print("symb not in atom_grid, lookup pyscf.dft.gen_grid._default_rad({},{})".format(chg, level))
                if level > 9:
                    print("error in level selection, rewriting to default pyscf value. level = {} -> 3".format(level))
                    level = 3
                n_rad = dft.gen_grid._default_rad(chg, level)
#                 n_ang = dft.gen_grid._default_ang(chg, level)
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)

#             rad_weight = 4*numpy.pi * rad**2 * dr

            phi, dphi = np.polynomial.legendre.leggauss(nang)
            phi = np.arccos(phi)

            x = np.outer(rad, np.cos(phi)).flatten()
            y = np.outer(rad, np.sin(phi)).flatten()

            dxy = np.outer(dr*rad**2, dphi)

            weights = (dxy*2*np.pi).flatten()
        #     coords = np.stack([y, 0*y, x],axis=-1)
            coords = np.stack([y, 0*y, x],axis=-1)

            atom_grids_tab[symb] = coords, weights
    return atom_grids_tab


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

def get_m_mask(mol):
    m_dict = {'s':0, 
             'px':1, 'pz':0, 'py':-1,
             'dxy':-2, 'dyz':-1, 'dz^2':0, 'dxz':1, 'dx2-y2':2,
             'fy^3':-3, 'fxyz':-2, 'fyz^2':-1, 'fz^3':0, 'fxz^2':1, 'fzx^2':2, 'fx^3':3,
             'f-3':-3, 'f-2':-2, 'f-1':-1,'f0':0,'f1':1, 'f2':2, 'f3':3}
    print(mol.atom, mol.ao_labels())
    #ALEC ^ edited tags that were not included
    try:
        labels = [l.split()[2][1:] for l in mol.ao_labels()]
        scount = 0
        for il in range(len(labels)):
            if labels[il] == 'f':
                labels[il] = 'f{}'.format(int(scount))
                scount+= 1
                if scount == 4:
                    scount = 0
        print(labels)
        basis_m = [m_dict[l] for l in labels]
    except KeyError:
        print('keyerror: orbital not in m_dict.keys()')
        
    m_mask = np.ones([len(basis_m),len(basis_m)])
    for i,m1 in enumerate(basis_m) :
        for j,m2 in enumerate(basis_m):
            if not m1 == m2:
                m_mask[i,j] = 0 
    return m_mask


def get_L(mol):
    """_summary_

    Args:
        mol (_type_): _description_

    Returns:
        _type_: _description_
    """
    pys = np.where(['py' in l for l in mol.ao_labels()])[0]
    pxs = np.where(['px' in l for l in mol.ao_labels()])[0]

    dxys = np.where(['dxy' in l for l in mol.ao_labels()])[0]
    dx2y2s = np.where(['dx2-y2' in l for l in mol.ao_labels()])[0]
    dyzs = np.where(['dyz' in l for l in mol.ao_labels()])[0]
    dxzs = np.where(['dxz' in l for l in mol.ao_labels()])[0]
    
    fx3s = np.where(['fx^3' in l for l in mol.ao_labels()])[0]
    fy3s = np.where(['fy^3' in l for l in mol.ao_labels()])[0]

    fzx2s = np.where(['fzx^2' in l for l in mol.ao_labels()])[0]
    fxyzs =  np.where(['fxyz' in l for l in mol.ao_labels()])[0]

    fxz2s = np.where(['fxz^2' in l for l in mol.ao_labels()])[0]
    fyz2s =  np.where(['fyz^2' in l for l in mol.ao_labels()])[0]

    L = np.eye(len(mol.ao_labels()))
    for px,py in zip(pxs,pys):
        L[px,py] = 2/np.sqrt(2)
        L[py,py] = 1/np.sqrt(2)
        L[px,px] = 1/np.sqrt(2)
        
    for dxy,dx2y2 in zip(dxys,dx2y2s):
        L[dxy,dx2y2] = 2/np.sqrt(2)
        L[dx2y2,dx2y2] = 1/np.sqrt(2)
        L[dxy,dxy] = 1/np.sqrt(2)
        
    for dyz,dxz in zip(dyzs,dxzs):
        L[dyz,dxz] = 2/np.sqrt(2)
        L[dxz,dxz] = 1/np.sqrt(2)
        L[dyz,dyz] = 1/np.sqrt(2)
        
    for fx3, fy3 in zip(fx3s,fy3s):
        L[fy3,fx3] = 2/np.sqrt(2)
        L[fx3,fx3] = 1/np.sqrt(2)
        L[fy3,fy3] = 1/np.sqrt(2)
        
    for fzx2, fxyz in zip(fzx2s,fxyzs):
        L[fxyz,fzx2] = 2/np.sqrt(2)
        L[fzx2,fzx2] = 1/np.sqrt(2)
        L[fxyz,fxyz] = 1/np.sqrt(2)
        
    for fxz2, fyz2 in zip(fxz2s,fyz2s):
        L[fyz2, fxz2]  = 2/np.sqrt(2)
        L[fxz2, fxz2]  = 1/np.sqrt(2)
        L[fyz2, fyz2]  = 1/np.sqrt(2)
        
    L = .5*(L + L.T)
    return L



def get_symmetrized_grid(mol, mf, n_rad=20, n_ang=10, print_stat=True, method= half_circle, return_errors = False):
    """_summary_

    Args:
        mol (_type_): _description_
        mf (_type_): _description_
        n_rad (int, optional): _description_. Defaults to 20.
        n_ang (int, optional): _description_. Defaults to 10.
        print_stat (bool, optional): _description_. Defaults to True.
        method (_type_, optional): _description_. Defaults to half_circle.
        return_errors (bool, optional): _description_. Defaults to False.
    """
    dm = mf.make_rdm1()
    if dm.ndim != 3:
#         dm = np.sum(dm, axis=0)
        dm = np.stack([dm,dm],axis=0)*0.5
#         dm = dm[0]
#     rho_ex = mf._numint.get_rho(mol, dm, mf.grids)
    rho_ex_a = mf._numint.eval_rho(mol, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2) , dm[0], xctype='metaGGA')
    rho_ex_b = mf._numint.eval_rho(mol, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2) , dm[1], xctype='metaGGA')

    q_ex_a = np.sum(rho_ex_a[0] * mf.grids.weights)
    q_ex_b = np.sum(rho_ex_b[0] * mf.grids.weights)

    exc_ex = np.sum(mf._numint.eval_xc(mf.xc, (rho_ex_a,rho_ex_b),spin=1)[0]*mf.grids.weights*(rho_ex_a[0]+rho_ex_b[0]))

    print("Using method", method, " for grid symmetrization")
    if mf.xc == 'SCAN' or mf.xc == 'TPSS':
        meta = True
    else:
        meta = False
    print("Using n_rad={}, n_ang={} in {}".format(n_rad, n_ang, method))
    coords, weights = half_circle(mf, mol, n_rad, n_ang)
 
    exc = mf._numint.eval_xc(mf.xc, (rho_ex_a,rho_ex_b),spin=1)[0]
    vxc = mf._numint.eval_xc(mf.xc, rho_ex_a +rho_ex_b)[1][0]
    if meta:
        vtau = mf._numint.eval_xc(mf.xc, rho_ex_a +rho_ex_b)[1][3]
    aoi = mf._numint.eval_ao(mol, mf.grids.coords, deriv = 2)

    vmunu1 = np.einsum('i,i,ij,ik->jk', mf.grids.weights, vxc,aoi[0],aoi[0])
    if meta:
        vtmunu1  = np.einsum('i,lij,lik->jk',vtau*mf.grids.weights, aoi[1:4],aoi[1:4])

    mf.grids.coords = coords
    mf.grids.weights = weights

    rho_sym_a = mf._numint.eval_rho(mol, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2) , dm[0], xctype='metaGGA')
    rho_sym_b = mf._numint.eval_rho(mol, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2) , dm[1], xctype='metaGGA')

    q_sym_a = np.sum(rho_sym_a[0] * mf.grids.weights)
    q_sym_b = np.sum(rho_sym_b[0] * mf.grids.weights)

    exc_sym = np.sum(mf._numint.eval_xc(mf.xc, (rho_sym_a,rho_sym_b),spin=1)[0]*mf.grids.weights*(rho_sym_a[0]+rho_sym_b[0]))
    if print_stat:
        print('{:10.6f}e   ||{:10.6f}e   ||{:10.6f}e'.format(q_ex_a, q_sym_a, np.abs(q_ex_a-q_sym_a)))
        print('{:10.6f}e   ||{:10.6f}e   ||{:10.6f}e'.format(q_ex_b, q_sym_b, np.abs(q_ex_b-q_sym_b)))
        print('{:10.3f}mH  ||{:10.3f}mH  ||{:10.3f}  microH'.format(1000*exc_ex, 1000*exc_sym, 1e6*np.abs(exc_ex-exc_sym)))
    error = 1e6*np.abs(exc_ex-exc_sym)

    exc = mf._numint.eval_xc(mf.xc, (rho_sym_a,rho_sym_b),spin=1)[0]
    vxc = mf._numint.eval_xc(mf.xc, rho_sym_a +rho_sym_b)[1][0]
    if meta:
        vtau = mf._numint.eval_xc(mf.xc, rho_sym_a +rho_sym_b)[1][3]
    aoi = mf._numint.eval_ao(mol, mf.grids.coords, deriv =2)

    vmunu2 = np.einsum('i,i,ij,ik->jk',mf.grids.weights, vxc,aoi[0],aoi[0])
    if meta:
        vtmunu2  = np.einsum('i,lij,lik->jk',vtau*mf.grids.weights,aoi[1:4],aoi[1:4])

    L = get_L(mol)
    scaling = get_m_mask(mol)
    
    vmunu2 = np.einsum('ij,jk,kl->il', L, vmunu2, L.T)*scaling

    if meta:
        vtmunu2 = np.einsum('ij,jk,kl->il', L, vtmunu2, L.T)*scaling
        vterr = vtmunu1 - vtmunu2
    verr = vmunu1 - vmunu2
    if print_stat:print({True: 'Potentials identical', False: 'Potentials not identical {}'.format(np.max(np.abs(verr)))}[np.allclose(vmunu1,vmunu2,atol=1e-5)])
    if print_stat and meta:print({True: 'tau Potentials identical', False: 'tau Potentials not identical {}'.format(np.max(np.abs(vterr)))}[np.allclose(vtmunu1,vtmunu2,atol=1e-5)])

    if return_errors:
        if meta:
            return (mf.grids.coords, mf.grids.weights, L, scaling), ((vmunu1, vmunu2), (vtmunu1, vtmunu2)) 
        else:
            return (mf.grids.coords, mf.grids.weights, L, scaling), ((vmunu1, vmunu2)) 
    else:
        return mf.grids.coords, mf.grids.weights, L, scaling
