import numpy as np
from pyscf import gto, dft, scf , df

def get_datapoint(mol, mf, dfit = False):
    """Builds all matrices needed for SCF calculations (the ones considered
        constant)

    Args:
        mol (gto.Mol): pyscf molecule object
        mf (scf.X): kernel for scf calculation, e.g. scf.RKS(mol)
        dfit (bool, optional): Whether or not to use density fitting. Defaults to False.

    Returns:
        matrices (dict): dictionary with key/value pairs of input data relating to given molecule
    """
    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')

    if dfit:
        auxbasis = df.addons.make_auxbasis(mol,mp2fit=False)
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        df_3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1', comp=1)
        df_2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        df_2c_inv = scipy.linalg.pinv(df_2c)
    else:
        eri = mol.intor('int2e')

    dm_init = mf.get_init_guess()

    n_elec = np.sum(mol.nelec)
    n_atoms = mol.natm

    matrices = {'dm_init':dm_init,
                'v':v,
                't':t,
                's':s,
                'n_elec':n_elec,
                'n_atoms':n_atoms,
                'e_nuc': np.array(mf.energy_nuc()),
                'mo_energy': np.array(mf.mo_energy),
                'mo_occ': np.array(mf.mo_occ)}

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
        """ Dataset class that serves as input to pytorch DataLoader and stores
        all the matrices defining a given system (molecule)
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
        return len(self.dm_init)
