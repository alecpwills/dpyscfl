import torch
import scipy
from opt_einsum import contract



def get_hcore(v, t):
    """ "Core" Hamiltionian, includes ion-electron and kinetic contributions
    """
    return v + t

class get_veff(torch.nn.Module):
    def __init__(self, exx=False, model=None, df= False):
        """ Builds the one-electron effective potential (not including local xc-potential)

        Parameters
        ----------
        exx, bool
            Exact exchange
        model, xc-model
            Only used for exact exchange mixing parameter
        df, bool
            Use density fitting
        """
        super().__init__()
        self.exx = exx
        self.model = model
        if df:
            self.forward = self.forward_df

    def forward(self, dm, eri):
        J = contract('...ij,ijkl->...kl',dm, eri)
        if self.exx:
            K = self.model.exx_a * contract('...ij,ikjl->...kl',dm, eri)
        else:
            K =  torch.zeros_like(J)

        if J.ndim == 3:
            return J[0] + J[1] - K
        else:
            return J-0.5*K

    def forward_df(self, dm, df_3c, df_2c_inv, eri):

        J = contract('mnQ, QP, ...ij, ijP->...mn', df_3c, df_2c_inv, dm, df_3c)
        if not self.exx:
            K = self.model.exx_a * contract('miQ, QP, ...ij, njP->...mn', df_3c, df_2c_inv, dm, df_3c)
        else:
            K =  torch.zeros_like(J)

        if J.ndim == 3:
            return J[0] + J[1] - K
        else:
            return J-0.5*K


def get_fock(hc, veff):
    """Fock matrix"""
    return hc + veff


class eig(torch.nn.Module):

    def __init__(self):
        """ Solves generalized eigenvalue problem using Cholesky decomposition"""
        super().__init__()

    def forward(self, h, s_chol):
        '''Solver for generalized eigenvalue problem

        .. math:: HC = SCE

        Parameters
        ----------
        h, torch.Tensor
            Hamiltionian
        s_chol, torch.Tensor
            (Inverse) Cholesky decomp. of overlap matrix S
            s_chol = np.linalg.inv(np.linalg.cholesky(S))
        '''
        #e, c = torch.symeig(contract('ij,...jk,kl->...il',s_chol, h, s_chol.T), eigenvectors=True,upper=False)
        #symeig deprecated for below, uses lower automatically
        e, c = torch.linalg.eigh(A=contract('ij,...jk,kl->...il',s_chol, h, s_chol.T))
        c = contract('ij,...jk ->...ik',s_chol.T, c)
        return e, c

class energy_tot(torch.nn.Module):

    def __init__(self):
        """
        Total energy (electron-electron + electron-ion; ion-ion not included)
        """
        super().__init__()

    def forward(self, dm, hcore, veff):
        return torch.sum((contract('...ij,ij', dm, hcore) + .5*contract('...ij,...ij', dm, veff))).unsqueeze(0)

class make_rdm1(torch.nn.Module):

    def __init__(self):
        """ Generate one-particle reduced density matrix"""
        super().__init__()

    def forward(self, mo_coeff, mo_occ):
        if mo_coeff.ndim == 3:
            mocc_a = mo_coeff[0, :, mo_occ[0]>0]
            mocc_b = mo_coeff[1, :, mo_occ[1]>0]
            if torch.sum(mo_occ[1]) > 0:
                return torch.stack([contract('ij,jk->ik', mocc_a*mo_occ[0,mo_occ[0]>0], mocc_a.T),
                                    contract('ij,jk->ik', mocc_b*mo_occ[1,mo_occ[1]>0], mocc_b.T)],dim=0)
            else:
                return torch.stack([contract('ij,jk->ik', mocc_a*mo_occ[0,mo_occ[0]>0], mocc_a.T),
                                    torch.zeros_like(mo_coeff)[0]],dim=0)
        else:
            mocc = mo_coeff[:, mo_occ>0]
            return contract('ij,jk->ik', mocc*mo_occ[mo_occ>0], mocc.T)

class SCF(torch.nn.Module):

    def __init__(self, alpha=0.8, nsteps=10, xc=None, device='cpu', exx=False):
        super().__init__()
        """ This class implements the self-consistent field (SCF) equations

        Parameters
        ---------
        alpha, float
            Linear mixing parameter
        nsteps, int
            Number of scf steps
        xc, XC
            Class containing the exchange-correlation models
        device, {'cpu','cuda'}
            Which device to use
        exx, bool
            Use exact exchange?
        """
        self.nsteps = nsteps
        self.alpha = alpha
        self.get_veff = get_veff(exx, xc).to(device) # Include Fock (exact) exchange?

        self.eig = eig().to(device)
        self.energy_tot = energy_tot().to(device)
        self.make_rdm1 = make_rdm1().to(device)
        self.xc = xc

    def forward(self, dm, matrices, sc=True):
        """
        Parameters
        ------------
        dm, torch.Tensor
            Initial density matrix
        matrices, dict of torch.Tensors
            Contains all other matrices that are considered fixed during
            SCF calculations (e-integrals etc.)
        sc, bool
            If True does self-consistent calculations, else single-pass
        """

        dm = dm[0]

        # Required matrices
        # ===================
        # v: Electron-ion pot.
        # t: Kinetic
        # mo_occ: MO occupations
        # e_nuc: Ion-Ion energy contribution
        # s: overlap matrix
        # s_chol: inverse Cholesky decomposition of overlap matrix
        v, t, mo_occ, e_nuc, s, s_chol = [matrices[key][0] for key in \
                                             ['v','t','mo_occ',
                                             'e_nuc','s','s_chol']]

        # Optional matrices
        # ====================

        # Electron repulsion integrals
        eri = matrices.get('eri',[None])[0]

        grid_weights = matrices.get('grid_weights',[None])[0]
        grid_coords = matrices.get('grid_coords',[None])[0]

        # Atomic orbitals evaluated on grid
        ao_eval = matrices.get('ao_eval',[None])[0]

        # Used to restore correct potential after symmetrization:
        L = matrices.get('L', [torch.eye(dm.size()[-1])])[0]
        scaling = matrices.get('scaling',[torch.ones([dm.size()[-1]]*2)])[0]

        # Density fitting integrals
        df_2c_inv = matrices.get('df_2c_inv',[None])[0]
        df_3c = matrices.get('df_3c',[None])[0]

        # Electrostatic potential on grid
        vh_on_grid = matrices.get('vh_on_grid',[None])[0]

        dm_old = dm

        E = []
        deltadm = []
        nsteps = self.nsteps

        # SCF iteration loop
        for step in range(nsteps):
            alpha = (self.alpha)**(step)+0.3
            beta = (1-alpha)
            dm = alpha * dm + beta * dm_old

            dm_old = dm

            hc = get_hcore(v,t)
            if df_3c is not None:
                veff = self.get_veff.forward_df(dm, df_3c, df_2c_inv, eri)
            else:
                veff = self.get_veff(dm, eri)

            if self.xc: #If using xc-functional (not Hartree-Fock)
                self.xc.ao_eval = ao_eval
                self.xc.grid_weights = grid_weights
                self.xc.grid_coords = grid_coords
                if vh_on_grid is not None:
                    self.xc.vh_on_grid = vh_on_grid
                    self.xc.df_2c_inv = df_2c_inv
                    self.xc.df_3c = df_3c

                if torch.sum(mo_occ) == 1:   # Otherwise H produces NaNs
                    dm[1] = dm[0]*1e-12
                    dm_old[1] = dm[0]*1e-12

                exc = self.xc(dm)


                vxc = torch.autograd.functional.jacobian(self.xc, dm, create_graph=True)

                # Restore correct symmetry for vxc
                if vxc.dim() > 2:
                    vxc = contract('ij,xjk,kl->xil',L,vxc,L.T)
                    vxc = torch.where(scaling.unsqueeze(0) > 0 , vxc, scaling.unsqueeze(0))
                else:
                    vxc = torch.mm(L,torch.mm(vxc,L.T))
                    vxc = torch.where(scaling > 0 , vxc, scaling)

                if torch.sum(mo_occ) == 1:   # Otherwise H produces NaNs
                    vxc[1] = torch.zeros_like(vxc[1])

                veff += vxc

                #Add random noise to potential to avoid degeneracies in EVs
                if self.xc.training and sc:
                    noise = torch.abs(torch.randn(vxc.size(),device=vxc.device)*1e-8)
                    noise = noise + torch.transpose(noise,-1,-2)
                    veff = veff + noise

            else:
                exc=0
                vxc=torch.zeros_like(veff)


            f = get_fock(hc, veff)
            if sc:
                mo_e, mo_coeff = self.eig(f, s_chol)
                dm = self.make_rdm1(mo_coeff, mo_occ)

            e_tot = self.energy_tot(dm_old, hc, veff-vxc)+ e_nuc + exc
            E.append(e_tot)
            if not sc:
                break

        results = {'E': torch.cat(E), 'dm':dm, 'mo_energy':mo_e}

        return results
