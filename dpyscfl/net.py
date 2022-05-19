from multiprocessing.sharedctypes import Value
import torch
torch.set_default_dtype(torch.double)
from torch.nn import Sequential as Seq, Linear, ReLU,Sigmoid,GELU
import pyscf
from pyscf import gto,dft,scf
import torch
import numpy as np
import scipy
from ase import Atoms
from ase.io import read
from .scf import *
from opt_einsum import contract

class XC(torch.nn.Module):

    def __init__(self, grid_models=None, heg_mult=True, pw_mult=True,
                    level = 1, exx_a=None, epsilon=1e-8):
        """Defines the XC functional on a grid

        Args:
            grid_models (list, optional): list of X_L (local exchange) or C_L (local correlation). Defines the xc-models/enhancement factors. Defaults to None.
            heg_mult (bool, optional): Use homoegeneous electron gas exchange (multiplicative if grid_models is not empty). Defaults to True.
            pw_mult (bool, optional): Use homoegeneous electron gas correlation (Perdew & Wang). Defaults to True.
            level (int, optional): Controls the number of density "descriptors" generated. 1: LDA, 2: GGA, 3:meta-GGA, 4: meta-GGA + electrostatic (nonlocal). Defaults to 1.
            exx_a (_type_, optional): Exact exchange mixing parameter. Defaults to None.
            epsilon (float, optional): Offset to avoid div/0 in calculations. Defaults to 1e-8.
        """

        super().__init__()
        self.heg_mult = heg_mult
        self.pw_mult = pw_mult
        self.grid_coords = None
        self.training = True
        self.level = level
        self.epsilon = epsilon
        if level > 3:
            print('WARNING: Non-local models highly experimental and likely will not work ')
        self.loge = 1e-5
        self.s_gam = 1

        if heg_mult:
            self.heg_model = LDA_X()
        if pw_mult:
            self.pw_model = PW_C()
        self.grid_models = list(grid_models)
        if self.grid_models:
            self.grid_models = torch.nn.ModuleList(self.grid_models)
        self.model_mult = [1 for m in self.grid_models]

        if exx_a is not None:
            self.exx_a = torch.nn.Parameter(torch.Tensor([exx_a]))
            self.exx_a.requires_grad = True
        else:
            self.exx_a = 0

    def evaluate(self):
        """Switches self.training flag to False
        """
        self.training=False
    def train(self):
        """Switches self.training flag to True
        """
        self.training=True

    def add_model_mult(self, model_mult):
        """_summary_

        .. todo:: 
            Unclear what the purpose of this is

        Args:
            model_mult (_type_): _description_
        """
        del(self.model_mult)
        self.register_buffer('model_mult',torch.Tensor(model_mult))

    def add_exx_a(self, exx_a):
        """Adds exact-exchange mixing parameter after initialization

        Args:
            exx_a (float): Exchange mixing parameter
        """
        self.exx_a = torch.nn.Parameter(torch.Tensor([exx_a]))
        self.exx_a.requires_grad = True

    # Density (rho)
    def l_1(self, rho):
        """Level 1 Descriptor -- Creates dimensionless quantity from rho.
        Eq. 3 in `base paper <https://link.aps.org/doi/10.1103/PhysRevB.104.L161109>`_

        .. math:: x_0 = \\rho^{1/3}

        Args:
            rho (torch.Tensor): density

        Returns:
            torch.Tensor: dimensionless density
        """
        return rho**(1/3)

    # Reduced density gradient s
    def l_2(self, rho, gamma):
        """Level 2 Descriptor -- Reduced gradient density
        Eq. 5 in `base paper <https://link.aps.org/doi/10.1103/PhysRevB.104.L161109>`_

        .. math:: x_2=s=\\frac{1}{2(3\\pi^2)^{1/3}} \\frac{|\\nabla \\rho|}{\\rho^{4/3}}

        Args:
            rho (torch.Tensor): density
            gamma (torch.Tensor): squared density gradient

        Returns:
            torch.Tensor: reduced density gradient s
        """
        return torch.sqrt(gamma)/(2*(3*np.pi**2)**(1/3)*rho**(4/3)+self.epsilon)

    # Reduced kinetic energy density alpha
    def l_3(self, rho, gamma, tau):
        """Level 3 Descriptor -- Reduced kinetic energy density
        Eq. 6 in `base paper <https://link.aps.org/doi/10.1103/PhysRevB.104.L161109>`_

        .. math:: x_3 = \\alpha = \\frac{\\tau-\\tau^W}{\\tau^{unif}},

        where

        .. math:: \\tau^W = \\frac{|\\nabla \\rho|^2}{8\\rho}, \\tau^{unif} = \\frac{3}{10} (3\\pi^2)^{2/3}\\rho^{5/3}.

        Args:
            rho (torch.Tensor): density
            gamma (torch.Tensor): squared density gradient
            tau (torch.Tensor): kinetic energy density

        Returns:
            torch.Tensor: reduced kinetic energy density
        """
        uniform_factor = (3/10)*(3*np.pi**2)**(2/3)
        tw = gamma/(8*(rho+self.epsilon))
        #commented is dpyscflite version, uncommented is xcdiff version
        #shouldn't change anything, but
        #return torch.nn.functional.relu((tau - tw)/(uniform_factor*rho**(5/3)+tw*1e-3 + 1e-12))
        return (tau - gamma/(8*(rho+self.epsilon)))/(uniform_factor*rho**(5/3)+self.epsilon)

    # Unit-less electrostatic potential
    def l_4(self, rho, nl):
        """Level 4 Descriptor -- Unitless electrostatic potential

        .. todo:: Figure out what exactly this part is

        Args:
            rho (torch.Tensor): density
            nl (torch.Tensor): some non-local descriptor

        Returns:
            torch.nn.functional.relu: _description_
        """
        u = nl[:,:1]/((rho.unsqueeze(-1)**(1/3))*self.nl_ueg[:,:1] + self.epsilon)
        wu = nl[:,1:]/((rho.unsqueeze(-1))*self.nl_ueg[:,1:] + self.epsilon)
        return torch.nn.functional.relu(torch.cat([u,wu],dim=-1))

    def get_descriptors(self, rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab,nl_a,nl_b, tau_a, tau_b, spin_scaling = False):
        """Creates 'ML-compatible' descriptors from the electron density and its gradients, a & b correspond to spin channels

        Args:
            rho0_a (torch.Tensor): :math:`\\rho` in spin-channel a
            rho0_b (torch.Tensor): :math:`\\rho` in spin-channel b
            gamma_a (torch.Tensor): :math:`|\\nabla \\rho|^2` in spin-channel a 
            gamma_b (torch.Tensor): :math:`|\\nabla \\rho|^2` in spin-channel b
            gamma_ab (torch.Tensor): _description_
            nl_a (torch.Tensor): _description_
            nl_b (torch.Tensor): _description_
            tau_a (torch.Tensor): KE density in spin-channel a
            tau_b (torch.Tensor): KE density in spin-channel b
            spin_scaling (bool, optional): Flag for spin-scaling. Defaults to False.

        Returns:
            _type_: _description_
        """

        if not spin_scaling:
            #If no spin-scaling, calculate polarization and use for X1
            zeta = (rho0_a - rho0_b)/(rho0_a + rho0_b + self.epsilon)
            spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta

        if self.level > 0:  #  LDA
            if spin_scaling:
                descr1 = torch.log(self.l_1(2*rho0_a) + self.loge)
                descr2 = torch.log(self.l_1(2*rho0_b) + self.loge)
            else:
                descr1 = torch.log(self.l_1(rho0_a + rho0_b) + self.loge)# rho
                descr2 = torch.log(spinscale) # zeta
            descr = torch.cat([descr1.unsqueeze(-1), descr2.unsqueeze(-1)],dim=-1)
        if self.level > 1: # GGA
            if spin_scaling:
                descr3a = self.l_2(2*rho0_a, 4*gamma_a) # s
                descr3b = self.l_2(2*rho0_b, 4*gamma_b) # s
                descr3 = torch.cat([descr3a.unsqueeze(-1), descr3b.unsqueeze(-1)],dim=-1)
                descr3 = (1-torch.exp(-descr3**2/self.s_gam))*torch.log(descr3 + 1)
            else:
                descr3 = self.l_2(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab) # s
                #line below in xcdiff, not dpyscfl
                descr3 = descr3/((1+zeta)**(2/3) + (1-zeta)**2/3)
                descr3 = descr3.unsqueeze(-1)
                descr3 = (1-torch.exp(-descr3**2/self.s_gam))*torch.log(descr3 + 1)
            descr = torch.cat([descr, descr3],dim=-1)
        if self.level > 2: # meta-GGA
            if spin_scaling:
                descr4a = self.l_3(2*rho0_a, 4*gamma_a, 2*tau_a)
                descr4b = self.l_3(2*rho0_b, 4*gamma_b, 2*tau_b)
                descr4 = torch.cat([descr4a.unsqueeze(-1), descr4b.unsqueeze(-1)],dim=-1)
                #below in xcdiff, not dpyscfl
                descr4 = descr4**3/(descr4**2+self.epsilon)
            else:
                descr4 = self.l_3(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab, tau_a + tau_b)
                #next 2 in xcdiff, not dpyscfl
                descr4 = 2*descr4/((1+zeta)**(5/3) + (1-zeta)**(5/3))
                descr4 = descr4**3/(descr4**2+self.epsilon)

                descr4 = descr4.unsqueeze(-1)
            descr4 = torch.log((descr4 + 1)/2)
            descr = torch.cat([descr, descr4],dim=-1)
        if self.level > 3: # meta-GGA + V_estat
            if spin_scaling:
                descr5a = self.l_4(2*rho0_a, 2*nl_a)
                descr5b = self.l_4(2*rho0_b, 2*nl_b)
                descr5 = torch.log(torch.stack([descr5a, descr5b],dim=-1) + self.loge)
                descr5 = descr5.view(descr5.size()[0],-1)
            else:
                descr5= torch.log(self.l_4(rho0_a + rho0_b, nl_a + nl_b) + self.loge)

            descr = torch.cat([descr, descr5],dim=-1)
        if spin_scaling:
            descr = descr.view(descr.size()[0],-1,2).permute(2,0,1)

        return descr


    def forward(self, dm):
        """_summary_

        Args:
            dm (torch.Tensor): density matrix

        Returns:
            _type_: _description_
        """
        Exc = 0
        if self.grid_models or self.heg_mult:
            if self.ao_eval.dim()==2:
                ao_eval = self.ao_eval.unsqueeze(0)
            else:
                ao_eval = self.ao_eval

            # Create density (and gradients) from atomic orbitals evaluated on grid
            # and density matrix
            # rho[ijsp]: del_i phi del_j phi dm (s: spin, p: grid point index)
            #print("FORWARD PASS IN XC. AO_EVAL SHAPE, DM SHAPE: ", ao_eval.shape, dm.shape)
            rho = contract('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm)
            rho0 = rho[0,0]
            drho = rho[0,1:4] + rho[1:4,0]
            tau = 0.5*(rho[1,1] + rho[2,2] + rho[3,3])

            # Non-local electrostatic potential
            if self.level > 3:
                non_loc = contract('mnQ, QP, Pki, ...mn-> ...ki', self.df_3c, self.df_2c_inv, self.vh_on_grid, dm)
            else:
                non_loc = torch.zeros_like(tau).unsqueeze(-1)

            if dm.dim() == 3: # If unrestricted (open-shell) calculation

                # Density
                rho0_a = rho0[0]
                rho0_b = rho0[1]

                # Contracted density gradient
                gamma_a, gamma_b = contract('ij,ij->j',drho[:,0],drho[:,0]), contract('ij,ij->j',drho[:,1],drho[:,1])
                gamma_ab = contract('ij,ij->j',drho[:,0],drho[:,1])

                # Kinetic energy density
                tau_a, tau_b = tau

                # E.-static
                non_loc_a, non_loc_b = non_loc
            else:
                rho0_a = rho0_b = rho0*0.5
                gamma_a=gamma_b=gamma_ab= contract('ij,ij->j',drho[:],drho[:])*0.25
                tau_a = tau_b = tau*0.5
                non_loc_a=non_loc_b = non_loc*0.5

            # xc-energy per unit particle
            exc = self.eval_grid_models(torch.cat([rho0_a.unsqueeze(-1),
                                                    rho0_b.unsqueeze(-1),
                                                    gamma_a.unsqueeze(-1),
                                                    gamma_ab.unsqueeze(-1),
                                                    gamma_b.unsqueeze(-1),
                                                    torch.zeros_like(rho0_a).unsqueeze(-1), #Dummy for laplacian
                                                    torch.zeros_like(rho0_a).unsqueeze(-1), #Dummy for laplacian
                                                    tau_a.unsqueeze(-1),
                                                    tau_b.unsqueeze(-1),
                                                    non_loc_a,
                                                    non_loc_b],dim=-1))
            #inplace modification throws MulBackwards0 error sometimes?
            #Exc += torch.sum(((rho0_a + rho0_b)*exc[:,0])*self.grid_weights)
            Exc = torch.sum(((rho0_a + rho0_b)*exc[:,0])*self.grid_weights)
            # try:
            #     Exc = torch.sum(((rho0_a + rho0_b)*exc[:,0])*self.grid_weights)
            # except:
            #     e = sys.exc_info()[0]
            #     Exc = torch.sum(((rho0_a + rho0_b)*exc[:,0])*self.grid_weights)
            #     print("Error detected")
            #     print(e)                

        #Below in xcdiff, not in dpyscfl
        #However, keep commented out -- self.nxc_models not implemented
        #if self.nxc_models:
        #    for nxc_model in self.nxc_models:
        #        Exc += nxc_model(dm, self.ml_ovlp)

        return Exc

    def eval_grid_models(self, rho):
        """Evaluates all models stored in self.grid_models along with HEG exchange and correlation


        Args:
            rho ([list of torch.Tensors]): List with [rho0_a,rho0_b,gamma_a,gamma_ab,gamma_b, dummy for laplacian, dummy for laplacian, tau_a, tau_b, non_loc_a, non_loc_b]

        Returns:
            _type_: _description_
        """
        Exc = 0
        rho0_a = rho[:, 0]
        rho0_b = rho[:, 1]
        gamma_a = rho[:, 2]
        gamma_ab = rho[:, 3]
        gamma_b = rho[:, 4]
        tau_a = rho[:, 7]
        tau_b = rho[:, 8]
        nl = rho[:,9:]
        nl_size = nl.size()[-1]//2
        nl_a = nl[:,:nl_size]
        nl_b = nl[:,nl_size:]

        C_F= 3/10*(3*np.pi**2)**(2/3)
        #in xcdiff, self.meta_local would change below assignments
        #not used here
        rho0_a_ueg = rho0_a
        rho0_b_ueg = rho0_b

        zeta = (rho0_a_ueg - rho0_b_ueg)/(rho0_a_ueg + rho0_b_ueg + 1e-8)
        rs = (4*np.pi/3*(rho0_a_ueg+rho0_b_ueg + 1e-8))**(-1/3)
        rs_a = (4*np.pi/3*(rho0_a_ueg + 1e-8))**(-1/3)
        rs_b = (4*np.pi/3*(rho0_b_ueg + 1e-8))**(-1/3)


        exc_a = torch.zeros_like(rho0_a)
        exc_b = torch.zeros_like(rho0_a)
        exc_ab = torch.zeros_like(rho0_a)

        descr_method = self.get_descriptors


        descr_dict = {}
        rho_tot = rho0_a + rho0_b
        if self.grid_models:

            for grid_model in self.grid_models:
                if not grid_model.spin_scaling:
                    if not 'c' in descr_dict:
                        descr_dict['c'] = descr_method(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, nl_a, nl_b, tau_a, tau_b, spin_scaling = False)
                        descr_dict['c'] = descr_method(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, nl_a, nl_b, tau_a, tau_b, spin_scaling = False)
                    descr = descr_dict['c']
                    #print("DESCR: ", descr)
                    #print("DESCR MAX:", torch.max(descr))
                    #print("DESCR MIN: ", torch.min(descr))
                    #print("GRID MODEL: ", grid_model)
                    for name, param in grid_model.named_parameters():
                        if torch.isnan(param).any():
                            print("NANS IN NETWORK WEIGHT -- {}".format(name))
                            raise ValueError("NaNs in Network Weights.")

                    #Evaluate network with descriptors on grid
                    #in xcdiff, edge_index is passed here, not in dpyscfl
                    exc = grid_model(descr,
                                      grid_coords = self.grid_coords)
                    #print("EXC GRID_MODEL C: ", exc)

                    #Included from xcdiff, 2dim exc -> spin polarized
                    if exc.dim() == 2: #If using spin decomposition
                        pw_alpha = self.pw_model(rs_a, torch.ones_like(rs_a))
                        pw_beta = self.pw_model(rs_b, torch.ones_like(rs_b))
                        pw = self.pw_model(rs, zeta)
                        ec_alpha = (1 + exc[:,0])*pw_alpha*rho0_a/rho_tot
                        ec_beta =  (1 + exc[:,1])*pw_beta*rho0_b/rho_tot
                        ec_mixed = (1 + exc[:,2])*(pw*rho_tot - pw_alpha*rho0_a - pw_beta*rho0_b)/rho_tot
                        exc_ab = ec_alpha + ec_beta + ec_mixed
                    else:
                        if self.pw_mult:
                            exc_ab += (1 + exc)*self.pw_model(rs, zeta)
                        else:
                            exc_ab += exc
#                    if self.pw_mult:
#                        exc_ab += (1 + exc)*self.pw_model(rs, zeta)
#                    else:
#                        exc_ab += exc
                else:
                    if not 'x' in descr_dict:
                        descr_dict['x'] = descr_method(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, nl_a, nl_b, tau_a, tau_b, spin_scaling = True)
                    descr = descr_dict['x']

                    #in xcdiff, edge_index is passed here, not in dpyscfl
                    exc = grid_model(descr,
                                  grid_coords = self.grid_coords)

                    #print("EXC GRID_MODEL X: ", exc)

                    if self.heg_mult:
                        exc_a += (1 + exc[0])*self.heg_model(2*rho0_a_ueg)*(1-self.exx_a)
                    else:
                        exc_a += exc[0]*(1-self.exx_a)

                    if torch.all(rho0_b == torch.zeros_like(rho0_b)): #Otherwise produces NaN's
                        exc_b += exc[0]*0
                    else:
                        if self.heg_mult:
                            exc_b += (1 + exc[1])*self.heg_model(2*rho0_b_ueg)*(1-self.exx_a)
                        else:
                            exc_b += exc[1]*(1-self.exx_a)

        else:
            if self.heg_mult:
                exc_a = self.heg_model(2*rho0_a_ueg)
                exc_b = self.heg_model(2*rho0_b_ueg)
            if self.pw_mult:
                exc_ab = self.pw_model(rs, zeta)


        exc = rho0_a_ueg/rho_tot*exc_a + rho0_b_ueg/rho_tot*exc_b + exc_ab
        return exc.unsqueeze(-1)

class C_L(torch.nn.Module):
    def __init__(self, n_input=2,n_hidden=16, device='cpu', ueg_limit=False, lob=2.0, use = []):
        """Local correlation model based on MLP
        Receives density descriptors in this order : [rho, spinscale, s, alpha, nl]
        input may be truncated depending on level of approximation

        Args:
            n_input (int, optional): Input dimensions (LDA: 2, GGA: 3 , meta-GGA: 4). Defaults to 2.
            n_hidden (int, optional): Number of hidden nodes (three hidden layers used by default). Defaults to 16.
            device (str, optional): {'cpu','cuda'}. Defaults to 'cpu'.
            ueg_limit (bool, optional): Enforce uniform homoegeneous electron gas limit. Defaults to False.
            lob (float, optional): Technically Lieb-Oxford bound but used here to enforce non-negativity. Should be kept at 2.0 in most instances. Defaults to 2.0.
            use (list of ints, optional): Indices for [s, alpha] (in that order) in input, to determine UEG limit. Defaults to [].
        """
        super().__init__()
        self.spin_scaling = False
        self.lob = False
        self.ueg_limit = ueg_limit
        self.n_input=n_input

        if not use:
            self.use = torch.Tensor(np.arange(n_input)).long().to(device)
        else:
            self.use = torch.Tensor(use).long().to(device)
        #below net in dpyscflite, doesn't include softplus at end or double flag
        # self.net = torch.nn.Sequential(
        #         torch.nn.Linear(n_input, n_hidden),
        #         torch.nn.GELU(),
        #         torch.nn.Linear(n_hidden, n_hidden),
        #         torch.nn.GELU(),
        #         torch.nn.Linear(n_hidden, n_hidden),
        #         torch.nn.GELU(),
        #         torch.nn.Linear(n_hidden, 1),
        #     ).to(device)
        #below net from xcdiff, softplus, double. the self.sig is also from xcdiff
        self.net = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, 1),
                torch.nn.Softplus()
            ).double().to(device)
        self.sig = torch.nn.Sigmoid()

        self.tanh = torch.nn.Tanh()
        #self.lob section allows for different values here, default=2. xcdiff doesn't have this,
        #assumes 2 always
        self.lob = lob
        if self.lob:
            self.lobf = LOB(self.lob)
        else:
            self.lob =  1000.0
            self.lobf = LOB(self.lob)


    def forward(self, rho, **kwargs):
        """Forward pass in network

        Args:
            rho (torch.Tensor): density

        Returns:
            _type_: _description_
        """
        inp = rho
        squeezed = -self.net(inp).squeeze()
        if self.ueg_limit:
            #below not form used in xcdiff
#            ueg_lim = rho[...,self.use[0]]
            #below form used in xcdiff,
            ueg_lim = self.tanh(rho[...,self.use[0]])
            if len(self.use) > 1:
                ueg_lim_a = torch.pow(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0
            #xcdiff does not include this next comparison
            if len(self.use) > 2:
                ueg_lim_nl = torch.sum(self.tanh(rho[...,self.use[2:]])**2,dim=-1)
            else:
                ueg_lim_nl = 0

            ueg_factor = ueg_lim + ueg_lim_a + ueg_lim_nl
        else:
            ueg_factor = 1
        #xcdiff below returns the negative of the negative inputs
        #lob is sigmoid, so odd function, negatives cancel, so not needed
        if self.lob:
            return self.lobf(squeezed*ueg_factor)
        else:
            return squeezed*ueg_factor
#xcdiff has this named XC_L, not X_L. keep for consistency's sake
class XC_L(torch.nn.Module):
    def __init__(self, n_input, n_hidden=16, use=[], device='cpu', ueg_limit=False, lob=1.804, one_e=False):
        """Local exchange model based on MLP
        Receives density descriptors in this order : [rho, s, alpha, nl],
        input may be truncated depending on level of approximation

        Args:
            n_input (int): Input dimensions (LDA: 1, GGA: 2, meta-GGA: 3, ...)
            n_hidden (int, optional): Number of hidden nodes (three hidden layers used by default). Defaults to 16.
            use (list of ints, optional): Only these indices are used as input to the model (can be used to omit density as input to enforce uniform density scaling). These indices are also used to enforce UEG where the assumed order is [s, alpha, ...].. Defaults to [].
            device (str, optional): {'cpu','cuda'}. Defaults to 'cpu'.
            ueg_limit (bool, optional): Enforce uniform homoegeneous electron gas limit. Defaults to False.
            lob (float, optional): Enforce this value as local Lieb-Oxford bound (don't enforce if set to 0). Defaults to 1.804.
            one_e (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.ueg_limit = ueg_limit
        self.spin_scaling = True
        self.lob = lob

        if not use:
            self.use = torch.Tensor(np.arange(n_input)).long().to(device)
        else:
            self.use = torch.Tensor(use).long().to(device)
        #xcdiff includes double flag on net
        self.net =  torch.nn.Sequential(
                torch.nn.Linear(n_input, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, 1),
            ).double().to(device)

        #to device not declared in xcdiff
        self.tanh = torch.nn.Tanh().to(device)
        self.lobf = LOB(lob).to(device)
        #below declared in xcdiff
        self.sig = torch.nn.Sigmoid()
        self.shift = 1/(1+np.exp(-1e-3))

    def forward(self, rho, **kwargs):
        """Forward pass

        Args:
            rho (_type_): _description_

        Returns:
            _type_: _description_
        """
        squeezed = self.net(rho[...,self.use]).squeeze()
        if self.ueg_limit:
            ueg_lim = rho[...,self.use[0]]
            if len(self.use) > 1:
                ueg_lim_a = torch.pow(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0
            #below comparison not in xcdiff
            if len(self.use) > 2:
                ueg_lim_nl = torch.sum(rho[...,self.use[2:]],dim=-1)
            else:
                ueg_lim_nl = 0
        else:
            ueg_lim = 1
            ueg_lim_a = 0
            ueg_lim_nl = 0

        if self.lob:
            result = self.lobf(squeezed*(ueg_lim + ueg_lim_a + ueg_lim_nl))
        else:
            result = squeezed*(ueg_lim + ueg_lim_a + ueg_lim_nl)

        return result

class LOB(torch.nn.Module):

    def __init__(self, limit=1.804):
        """ Utility function to squash output to [-1, limit-1] inteval.
            Can be used to enforce non-negativity and Lieb-Oxford bound.
        """
        super().__init__()
        self.sig = torch.nn.Sigmoid()
        self.limit = limit

    def forward(self, x):
         return self.limit*self.sig(x-np.log(self.limit-1))-1


class LDA_X(torch.nn.Module):

    def __init__(self):
        """ UEG exchange"""
        super().__init__()

    def forward(self, rho, **kwargs):
        return -3/4*(3/np.pi)**(1/3)*rho**(1/3)



# The following section was adapted from LibXC-4.3.4

params_a_pp     = [1,  1,  1]
params_a_alpha1 = [0.21370,  0.20548,  0.11125]
params_a_a      = [0.031091, 0.015545, 0.016887]
params_a_beta1  = [7.5957, 14.1189, 10.357]
params_a_beta2  = [3.5876, 6.1977, 3.6231]
params_a_beta3  = [1.6382, 3.3662,  0.88026]
params_a_beta4  = [0.49294, 0.62517, 0.49671]
params_a_fz20   = 1.709921


class PW_C(torch.nn.Module):


    def __init__(self):
        """ UEG correlation, Perdew & Wang"""
        super().__init__()
    def forward(self, rs, zeta):
        def g_aux(k, rs):
            return params_a_beta1[k]*torch.sqrt(rs) + params_a_beta2[k]*rs\
          + params_a_beta3[k]*rs**1.5 + params_a_beta4[k]*rs**(params_a_pp[k] + 1)

        def g(k, rs):
            return -2*params_a_a[k]*(1 + params_a_alpha1[k]*rs)\
          * torch.log(1 +  1/(2*params_a_a[k]*g_aux(k, rs)))

        def f_zeta(zeta):
            return ((1+zeta)**(4/3) + (1-zeta)**(4/3) - 2)/(2**(4/3)-2)

        def f_pw(rs, zeta):
            return g(0, rs) + zeta**4*f_zeta(zeta)*(g(1, rs) - g(0, rs) + g(2, rs)/params_a_fz20)\
          - f_zeta(zeta)*g(2, rs)/params_a_fz20

        return f_pw(rs, zeta)

def freeze_net(nn):
    for i in nn.parameters():
        i.requires_grad = False

def unfreeze_net(nn):
    for i in nn.parameters():
        i.requires_grad = True

def freeze_append_xc(model, n, outputlayergrad):
    freeze_net(model.xc)
    #Implemented as a Module List of the X and C networks.
    chil = [i for i in model.xc.children() if isinstance(i, torch.nn.ModuleList)][0]
    #X network first. Find the children of X, i.e. the Linear/GELUs
    xl = [i for i in chil[0].net.children()]
    #C network second. Find the children of C, i.e. the Linear/GELUs
    cl = [i for i in chil[1].net.children()]
    #Pop the output layer of each net.
    xout = xl.pop()
    cout = cl.pop()
    #duplicate last layer and GELU
    xl += xl[-2:]*n
    cl += cl[-2:]*n
    #set last layer as unfrozen
    for p in xl[-2*n:]:
        for par in p.parameters():
            par.requires_grad = True
    for p in cl[-2*n:]:
        for par in p.parameters():
            par.requires_grad = True
    #Readd output layer.
    xl.append(xout)
    cl.append(cout)
    #If flagged, set output layer to be non-frozen
    if outputlayergrad:
        for par in xl[-1].parameters():
            par.requires_grad = True
        for par in cl[-1].parameters():
            par.requires_grad = True
    #Set the new layers as the networks to use.
    chil[0].net = torch.nn.Sequential(*xl)
    chil[1].net = torch.nn.Sequential(*cl)


#BELOW IS FROM DPYSCF
def get_scf(xctype, pretrain_loc='', hyb_par=0, path='', DEVICE='cpu', ueg_limit=True, meta_x=None, freec=False,
            inserts = 0):
    """_summary_

    Args:
        xctype (_type_): _description_
        pretrain_loc (_type_): _description_
        hyb_par (int, optional): _description_. Defaults to 0.
        path (str, optional): _description_. Defaults to ''.
        DEVICE (str, optional): _description_. Defaults to 'cpu'.
        ueg_limit (bool, optional): _description_. Defaults to True.
        meta_x (_type_, optional): _description_. Defaults to None.
        freec (bool, optional): _description_. Defaults to False.
    """
    print('FREEC', freec)
    X_L = XC_L
    if xctype == 'GGA':
        lob = 1.804 if ueg_limit else 0
        x = X_L(device=DEVICE,n_input=1, n_hidden=16, use=[1], lob=lob, ueg_limit=ueg_limit) # PBE_X
        c = C_L(device=DEVICE,n_input=3, n_hidden=16, use=[2], ueg_limit=ueg_limit and not freec)
        xc_level = 2
    elif xctype == 'MGGA':
        lob = 1.174 if ueg_limit else 0
        x = X_L(device=DEVICE,n_input=2, n_hidden=16, use=[1,2], lob=1.174, ueg_limit=ueg_limit) # PBE_X
        c = C_L(device=DEVICE,n_input=4, n_hidden=16, use=[2,3], ueg_limit=ueg_limit and not freec)
        xc_level = 3
    if pretrain_loc:
        print("Loading pre-trained models from " + pretrain_loc)
        x.load_state_dict(torch.load(pretrain_loc + '/x'))
        c.load_state_dict(torch.load(pretrain_loc + '/c'))

    if hyb_par:
        try:
            a = 1 - hyb_par
            b = 1
            d = hyb_par
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level )
            scf = SCF(nsteps=25, xc=xc, exx=True,alpha=0.3)

            xc.add_exx_a(d)
            xc.exx_a.requires_grad=True

            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

        except RuntimeError:
            a = 1 - hyb_par
            b = 1
            d = hyb_par
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level, exx_a=d)
            scf = SCF(nsteps=25, xc=xc, exx=True,alpha=0.3)
            print(xc.exx_a)
            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            xc.exx_a.requires_grad=True
            print(xc.exx_a)
    else:
        #xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level, meta_x=meta_x)
        xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level)
        if path:
            try:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            except AttributeError:
                # AttributeError: 'RecursiveScriptModule' object has no attribute 'copy'
                #occurs when loading finished xc from xcdiff
                xcp = torch.jit.load(path)
                xc.load_state_dict(xcp.state_dict())
        if inserts:
            freeze_append_xc(scf, inserts, False)
        scf = SCF(nsteps=25, xc=xc, exx=False,alpha=0.3)

    scf.xc.train()
    return scf