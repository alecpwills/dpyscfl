import torch
import scipy
import sys
from opt_einsum import contract



def get_hcore(v, t):
    """ "Core" Hamiltionian, includes ion-electron and kinetic contributions

    .. math:: H_{core} = T + V_{nuc-elec}

    Args:
        v (torch.Tensor, np.array): Electron-ion interaction energy
        t (torch.Tensor, np.array): Kinetic energy

    Returns:
        torch.Tensor: v + t
    """
    return v + t

class get_veff(torch.nn.Module):
    def __init__(self, exx=False, model=None, df= False):
        """Builds the one-electron effective potential (not including local xc-potential)

        Args:
            exx (bool, optional): Exact exchange flag. Defaults to False.
            model (xc-model): Only used for exact exchange mixing parameter. Defaults to None.
            df (bool, optional): Use density fitting flag. Defaults to False.
        """
        super().__init__()
        self.exx = exx
        self.model = model
        if df:
            self.forward = self.forward_df

    def forward(self, dm, eri):
        """Forward pass if no density fitting

        Args:
            dm (torch.Tensor): Density matrix
            eri (torch.Tensor(?)): Electron repulsion integral tensor

        Returns:
            torch.Tensor: The "effective" potential
        """
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
        """Forward pass if density fitting

        Args:
            dm (torch.Tensor): Density matrix
            df_3c (torch.Tensor): 3-center density fit integrals(?)
            df_2c_inv (torch.Tensor): 2-center density fit integrals(?)
            eri (torch.Tensor(?)): Electron repulsion integral tensor

        Returns:
            torch.Tensor: The "effective" potential
        """
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
    """Get the Fock matrix

    Args:
        hc (torch.Tensor): Core Hamiltonian
        veff (torch.Tensor): Effective Potential

    Returns:
        torch.Tensor: hc+veff
    """
    return hc + veff


class eig(torch.nn.Module):

    def __init__(self):
        """Solves generalized eigenvalue problem using Cholesky decomposition
        """
        super().__init__()

    def forward(self, h, s_chol):
        """Solver for generalized eigenvalue problem

        .. todo:: torch.symeig is deprecated for torch.linalg.eigh, replace

        Args:
            h (torch.Tensor): Hamiltionian
            s_chol (torch.Tensor): (Inverse) Cholesky decomp. of overlap matrix S
                                    s_chol = np.linalg.inv(np.linalg.cholesky(S))

        Returns:
            (torch.Tensor, torch.Tensor): Eigenvalues (MO energies), eigenvectors (MO coeffs)
        """
        #e, c = torch.symeig(contract('ij,...jk,kl->...il',s_chol, h, s_chol.T), eigenvectors=True,upper=False)
        upper=False
        UPLO = "U" if upper else "L"
        e, c = torch.linalg.eigh(contract('ij,...jk,kl->...il',s_chol, h, s_chol.T), UPLO=UPLO)
        c = contract('ij,...jk ->...ik',s_chol.T, c)
        return e, c

class energy_tot(torch.nn.Module):

    def __init__(self):
        """
        Total energy (electron-electron + electron-ion; ion-ion not included)
        """
        super().__init__()

    def forward(self, dm, hcore, veff):
        """Tensor contraction to find total electron energy (e-e + e-ion)

        Args:
            dm (torch.Tensor): Density matrix
            hcore (torch.Tensor): Core Hamiltonian
            veff (torch.Tensor): Effective Potential

        Returns:
            torch.Tensor: The electronic energy
        """
        return torch.sum((contract('...ij,ij', dm, hcore) + .5*contract('...ij,...ij', dm, veff))).unsqueeze(0)

class make_rdm1(torch.nn.Module):

    def __init__(self):
        """ Generate one-particle reduced density matrix"""
        super().__init__()

    def forward(self, mo_coeff, mo_occ):
        """Forward pass calculating one-particle reduced density matrix.

        Args:
            mo_coeff (torch.Tensor/np.array(?)): Molecular orbital coefficients
            mo_occ (torch.Tensor/np.array(?)): Molecular orbital occupation numbers

        Returns:
            torch.Tensor/np.array(?): The RDM1
        """
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
        """This class implements the self-consistent field (SCF) equations

        Args:
            alpha (float, optional): Linear mixing parameter. Defaults to 0.8.
            nsteps (int, optional): Number of scf steps. Defaults to 10.
            xc (dpyscfl.net.XC, optional): Class containing the exchange-correlation models. Defaults to None.
            device (str, optional): {'cpu','cuda'}, which device to use. Defaults to 'cpu'.
            exx (bool, optional): Use exact exchange flag. Defaults to False.
        """
        super().__init__()
        self.nsteps = nsteps
        self.alpha = alpha
        self.get_veff = get_veff(exx, xc).to(device) # Include Fock (exact) exchange?

        self.eig = eig().to(device)
        self.energy_tot = energy_tot().to(device)
        self.make_rdm1 = make_rdm1().to(device)
        self.xc = xc
        #ncore parameter used in xcdiff, not here

    def forward(self, dm, matrices, sc=True, **kwargs):
        """Forward pass SCF cycle

        Args:
            dm (torch.Tensor): Initial density matrix
            matrices (dict of torch.Tensors): Contains all other matrices that are considered fixed during SCF calculations (e-integrals etc.)
            sc (bool, optional): If True does self-consistent calculations, else single-pass. Defaults to True.

        Returns:
            dict of torch.Tensors: results: E, dm, and mo_energies
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
        #edge index called for here in xcdiff, not here

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

        # if not self.xc.training:
        #     #if not training, backpropagation doesn't happen so don't need derivatives beyond
        #     #calculation at a given step
        #     create_graph = False
        # else:
        #     create_graph = True
        vvv = kwargs.get('verbose', False)
        if vvv:
            print('SCF Loop Beginning: {} Steps'.format(nsteps))

        # SCF iteration loop
        for step in range(nsteps):
            #some diis happens here in xcdiff, not implemented here
            if vvv:
                print('Step {}'.format(step))
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
                #edge index, ml_ovlp called for here in xcdiff
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
                if self.xc.training:#: and sc:
                    if step == 0:
                        print("Noise generation to avoid potential degeneracies")
                    noise = torch.abs(torch.randn(vxc.size(),device=vxc.device)*1e-8)
                    noise = noise + torch.transpose(noise,-1,-2)
                    veff = veff + noise

            else:
                exc=0
                vxc=torch.zeros_like(veff)
            f = get_fock(hc, veff)
            mo_e, mo_coeff = self.eig(f, s_chol)
            dm = self.make_rdm1(mo_coeff, mo_occ)

            e_tot = self.energy_tot(dm_old, hc, veff-vxc)+ e_nuc + exc
            E.append(e_tot)
            print("{} Energy: {}".format(e_tot))
            print("History: {}".format(E))
            if not sc:
                break

        #in xcdiff, things happen here with mo_occ[:self.ncore], e_ip etc. not implemented here
        
        results = {'E': torch.cat(E), 'dm':dm, 'mo_energy':mo_e}

        return results
