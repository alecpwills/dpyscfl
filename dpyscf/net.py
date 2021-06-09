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
from .torch_routines import *
from opt_einsum import contract

class XC(torch.nn.Module):

    def __init__(self, grid_models=None, heg_mult=True, pw_mult=True,
                    level = 1, exx_a=None):
        """ Defines the XC functional on a grid

        Parameters
        ----------
        grid_models, list of X_L (local exchange) or C_L (local correlation)
            Defines the xc-models/enhancement factors
        heg_mult, bool
            Use homoegeneous electron gas exchange (multiplicative if grid_models is not empty)
        pw_mult, bool
            Use homoegeneous electron gas correlation (Perdew & Wang)
        level, int
            Controls the number of density "descriptors" generated
            1: LDA, 2: GGA, 3:meta-GGA, 4: meta-GGA + electrostatic (nonlocal)
        exx_a, float
            Exact exchange mixing parameter
        """


        super().__init__()
        self.heg_mult = heg_mult
        self.pw_mult = pw_mult
        self.grid_coords = None
        self.training = True
        self.level = level
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
        self.training=False
    def train(self):
        self.training=True

    def add_model_mult(self, model_mult):
        del(self.model_mult)
        self.register_buffer('model_mult',torch.Tensor(model_mult))

    def add_exx_a(self, exx_a):
        self.exx_a = torch.nn.Parameter(torch.Tensor([exx_a]))
        self.exx_a.requires_grad = True


    def get_descriptors(self, rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab,nl_a,nl_b, tau_a, tau_b, spin_scaling = False):
        """
        Creates "ML-compatible" descriptors from the electron density and its gradients, a & b correspond to spin channels

        Parameters
        ----------
        spin_scaling, bool
            Only create descriptors compatible with spin_scaling (i.e. ommit spin information), usually used
            for exchange functionals
        """


        uniform_factor = (3/10)*(3*np.pi**2)**(2/3)

        # Density (rho)
        def l_1(rho):
            return rho**(1/3)

        # Reduced density gradient s
        def l_2(rho, gamma):
            return torch.sqrt(gamma)/(2*(3*np.pi**2)**(1/3)*rho**(4/3)+self.epsilon)

        # Reduced kinetic energy density alpha
        def l_3(rho, gamma, tau):
            tw = gamma/(8*rho+self.epsilon)
            return torch.nn.functional.relu((tau - tw)/(uniform_factor*rho**(5/3)+tw*1e-3 + 1e-12))

        # Unit-less electrostatic potential
        def l_4(rho, nl):
            u = nl[:,:1]/((rho.unsqueeze(-1)**(1/3))*self.nl_ueg[:,:1] + self.epsilon)
            wu = nl[:,1:]/((rho.unsqueeze(-1))*self.nl_ueg[:,1:] + self.epsilon)
            return torch.nn.functional.relu(torch.cat([u,wu],dim=-1))


        if not spin_scaling:
            zeta = (rho0_a - rho0_b)/(rho0_a + rho0_b + self.epsilon)
            spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta

        if self.level > 0:  #  LDA
            if spin_scaling:
                descr1 = torch.log(l_1(2*rho0_a) + self.loge)
                descr2 = torch.log(l_1(2*rho0_b) + self.loge)
            else:
                descr1 = torch.log(l_1(rho0_a + rho0_b) + self.loge)# rho
                descr2 = torch.log(spinscale) # zeta
            descr = torch.cat([descr1.unsqueeze(-1), descr2.unsqueeze(-1)],dim=-1)
        if self.level > 1: # GGA
            if spin_scaling:
                descr3a = l_2(2*rho0_a, 4*gamma_a) # s
                descr3b = l_2(2*rho0_b, 4*gamma_b) # s
                descr3 = torch.cat([descr3a.unsqueeze(-1), descr3b.unsqueeze(-1)],dim=-1)
                descr3 = (1-torch.exp(-descr3**2/self.s_gam))*torch.log(descr3 + 1)
            else:
                descr3 = l_2(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab) # s
                descr3 = descr3.unsqueeze(-1)
                descr3 = (1-torch.exp(-descr3**2/self.s_gam))*torch.log(descr3 + 1)
            descr = torch.cat([descr, descr3],dim=-1)
        if self.level > 2: # meta-GGA
            if spin_scaling:
                descr4a = l_3(2*rho0_a, 4*gamma_a, 2*tau_a)
                descr4b = l_3(2*rho0_b, 4*gamma_b, 2*tau_b)
                descr4 = torch.cat([descr4a.unsqueeze(-1), descr4b.unsqueeze(-1)],dim=-1)
            else:
                descr4 = l_3(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab, tau_a + tau_b)
                descr4 = descr4.unsqueeze(-1)
            descr4 = torch.log((descr4 + 1)/2)
            descr = torch.cat([descr, descr4],dim=-1)
        if self.level > 3: # meta-GGA + V_estat
            if spin_scaling:
                descr5a = l_4(2*rho0_a, 2*nl_a)
                descr5b = l_4(2*rho0_b, 2*nl_b)
                descr5 = torch.log(torch.stack([descr5a, descr5b],dim=-1) + self.loge)
                descr5 = descr5.view(descr5.size()[0],-1)
            else:
                descr5= torch.log(l_4(rho0_a + rho0_b, nl_a + nl_b) + self.loge)

            descr = torch.cat([descr, descr5],dim=-1)
        if spin_scaling:
            descr = descr.view(descr.size()[0],-1,2).permute(2,0,1)

        return descr


    def forward(self, dm):
        """ Main method, calculates Exc from density matrix dm
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

            Exc += torch.sum(((rho0_a + rho0_b)*exc[:,0])*self.grid_weights)


        return Exc

    def eval_grid_models(self, rho):
        """ Evaluates all models stored in self.grid_models along with HEG exchange and correlation
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

                    exc = grid_model(descr,
                                      grid_coords = self.grid_coords)

                    if self.pw_mult:
                        exc_ab += (1 + exc)*self.pw_model(rs, zeta)
                    else:
                        exc_ab += exc
                else:
                    if not 'x' in descr_dict:
                        descr_dict['x'] = descr_method(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, nl_a, nl_b, tau_a, tau_b, spin_scaling = True)
                    descr = descr_dict['x']


                    exc = grid_model(descr,
                                  grid_coords = self.grid_coords)


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
        """ Local correlation model based on MLP
            Receives density descriptors in this order : [rho, spinscale, s, alpha, nl]
            input may be truncated depending on level of approximation
            
        Parameters
        ----------

        n_input, int
            Input dimensions (LDA: 2, GGA: 3 , meta-GGA: 4)
        n_hidden, int
            Number of hidden nodes (three hidden layers used by default)
        device, str {'cpu','cuda'}
        ueg_limit, bool
            Enforce uniform homoegeneous electron gas limit
        lob, float
            Technically Lieb-Oxford bound but used here to enforce non-negativity.
            Should be kept at 2.0 in most instances.
        use, list of ints
            Indices for [s, alpha] (in that order) in input, to determine UEG limit
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

        self.net = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, 1),
            ).to(device)

        self.tanh = torch.nn.Tanh()
        self.lob = lob
        if self.lob:
            self.lobf = LOB(self.lob)
        else:
            self.lob =  1000.0
            self.lobf = LOB(self.lob)


    def forward(self, rho, **kwargs):
        inp = rho
        squeezed = -self.net(inp).squeeze()
        if self.ueg_limit:
            ueg_lim = rho[...,self.use[0]]
            if len(self.use) > 1:
                ueg_lim_a = torch.pow(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0
            if len(self.use) > 2:
                ueg_lim_nl = torch.sum(self.tanh(rho[...,self.use[2:]])**2,dim=-1)
            else:
                ueg_lim_nl = 0

            ueg_factor = ueg_lim + ueg_lim_a + ueg_lim_nl
        else:
            ueg_factor = 1

        if self.lob:
            return self.lobf(squeezed*ueg_factor)
        else:
            return squeezed*ueg_factor

class X_L(torch.nn.Module):
    def __init__(self, n_input, n_hidden=16, use=[], device='cpu', ueg_limit=False, lob=1.804, one_e=False):
        """ Local exchange model based on MLP
            Receives density descriptors in this order : [rho, s, alpha, nl],
            input may be truncated depending on level of approximation

        Parameters
        ----------
        n_input, int
            Input dimensions (LDA: 1, GGA: 2, meta-GGA: 3, ...)
        n_hidden, int
            Number of hidden nodes (three hidden layers used by default)
        use, list of ints
            Only these indices are used as input to the model (can be used to omit density as input
            to enforce uniform density scaling). These indices are also used to enforce UEG where
            the assumed order is [s, alpha, ...].
        device, str {'cpu','cuda'}
        ueg_limit, bool
            Enforce uniform homoegeneous electron gas limit
        lob, float
            Enforce this value as local Lieb-Oxford bound (don't enforce if set to 0)
        """
        super().__init__()
        self.ueg_limit = ueg_limit
        self.spin_scaling = True
        self.lob = lob

        if not use:
            self.use = torch.Tensor(np.arange(n_input)).long().to(device)
        else:
            self.use = torch.Tensor(use).long().to(device)
        self.net =  torch.nn.Sequential(
                torch.nn.Linear(n_input, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, 1),
            ).to(device)

        self.tanh = torch.nn.Tanh().to(device)
        self.lobf = LOB(lob).to(device)

    def forward(self, rho, **kwargs):
        squeezed = self.net(rho[...,self.use]).squeeze()
        if self.ueg_limit:
            ueg_lim = rho[...,self.use[0]]
            if len(self.use) > 1:
                ueg_lim_a = torch.pow(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0
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
