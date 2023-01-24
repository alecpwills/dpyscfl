import pyscf
from pyscf import gto,dft,scf
import argparse
import torch
torch.set_default_dtype(torch.double)
import pyscf
from pyscf import gto,dft,scf
import matplotlib.pyplot as plt
import numpy as np
import scipy
from ase import Atoms
from ase.io import read
from dpyscfl.net import * 
from dpyscfl.scf import * 
from dpyscfl.utils import *
from dpyscfl.losses import *
from pyscf.cc import CCSD
from functools import partial
from ase.units import Bohr
from itertools import cycle
DEVICE='cpu'

def get_gamma(rho, s):
    return (s*2*(3*np.pi**2)**(1/3)*rho**(4/3))**2
    
def get_tau(rho, gamma, alpha):
    uniform_factor = (3/10)*(3*np.pi**2)**(2/3)
    return (gamma/(8*rho))+(uniform_factor*rho**(5/3))*alpha

def unpol_input(rho, gamma, tau):
    return .5*rho, .5*rho, 0.25*gamma, 0.25*gamma, 0.25*gamma, 0*tau, 0*tau, 0.5*tau, 0.5*tau

def libxc_input(rho, gamma, tau):
    return rho, torch.sqrt(gamma/3),  torch.sqrt(gamma/3),  torch.sqrt(gamma/3), tau , tau

def plot_fxc(models, savename, rs = [0.1, 1, 5], s_range=[0, 3], alpha_range=None, only = None):
    
    if only is not None:
        saved_models = {}
        for model_name in models:
            gm = models[model_name].grid_models
            saved_models[model_name] = gm
            models[model_name].grid_models = gm[only:only+1]
    if alpha_range is None:
        alpha_range_= [1]
    else:
        alpha_range_= alpha_range
    idx = 0
    for  _, rs_val in enumerate(rs):
        for alpha in alpha_range_:
            rho_val = 3/(4*np.pi*rs_val**3)
            s = torch.linspace(s_range[0], s_range[1],300)
            rho = torch.Tensor([rho_val]*len(s))
            gamma = get_gamma(rho, s)
            tau = get_tau(rho, gamma, alpha)
            
            inp = torch.stack(unpol_input(rho, gamma, tau),dim=-1)
            inp_libxc = torch.stack(libxc_input(rho, gamma,tau),dim=-1).detach().numpy().T
        
            lsc = cycle(['-','--',':','-.',':'])
            lws = [2] + [1]*(len(models)-1)
            e_heg = dft.libxc.eval_xc("LDA_X",inp_libxc,spin=0, deriv=1)[0]
                
            for model_name, lw in zip(models,lws):
                ls = next(lsc)
                if ls == '-' and len(rs) > 1: 
                    l = '$r_s = ${}'.format(rs_val)
                elif ls == '-' and len(alpha_range_) > 1:
                    if alpha_range is not None:
                        l = ' $\\alpha = $ {}'.format(alpha)
                else:
                    l = ''
                libxc = False
                if model_name[-4:] == '_LXC':
                    libxc = True
                if model_name[-2:] == '_S' or libxc:
                    method = models[model_name]
                if model_name == 'XCDiff':
                    method = models[model_name]
                else:
                    #models[model_name].exx_a = torch.nn.Parameter(torch.Tensor([0]))
                    method = models[model_name].eval_grid_models
                    
                if libxc:
                    exc = dft.libxc.eval_xc(method, inp_libxc, spin=0, deriv=1)[0]
                else:
                    exc = method(inp).squeeze().detach().numpy()
               
               
#                 e_heg = models[model_name].heg_model(rho).squeeze().detach().numpy()
                ax = plt.plot(s, exc/e_heg,
                     label = l, color='C{}'.format(idx),ls = ls,lw=lw)
                np.save(savename.split('.')[0]+'_'+model_name+'_x.npy', s)
                np.save(savename.split('.')[0]+'_'+model_name+'_y.npy', exc/e_heg)
                if len(rs) == 1 and (alpha_range is None or  len(alpha_range) == 1):
                    idx+=1
            idx+=1
    lsc = cycle(['-','--',':','-.',':'])
    for idx,model_name in enumerate(models):  
        ls = next(lsc)
        c = 'gray' if len(rs) > 1 or len(alpha_range_) > 1 else 'C{}'.format(idx)
        plt.plot([],label=model_name,color=c,ls=ls)

    plt.ylim(0.9, 1.3)
    plt.xlim(0, 3)
    plt.ylabel('$F_{xc}$ (a.u.)', fontsize=14)
    plt.xlabel('s', fontsize=14)
    plt.legend()
    plt.title('$r_s = {}, \\alpha = {}$'.format(rs[0], alpha_range[0]), fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename, dpi=300, figsize=(8,8))
    plt.close()
    
    if only is not None:
        for model_name in models:
            models[model_name].grid_models = saved_models[model_name]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train xc functional')
    parser.add_argument('--modelpath', action='store', type=str, help='Location of model to be loaded into pytorch to generate factors')
    parser.add_argument('--modeltype', default='MGGA', action='store', type=str, help='Type of network loaded -- (M)GGA')
    parser.add_argument('--plotlabel', type=str, default='', help='Specify label for figure.')
    parser.add_argument('--savepref', type=str, default='', help='Save file prefix to be appended when saving the various figures.')
    parser.add_argument('--xcdiffpath', action='store', type=str, default='', help='Location of xcdiff model to plot')
    parser.add_argument('--pretrainedpath', action='store', type=str, default='', help='Location of pretrained model to plot alongside')
    parser.add_argument('--pretrainedtype', default='MGGA', action='store', type=str, help='Type of pretrained network loaded -- (M)GGA')
    parser.add_argument('--pretrainlab', default='Pre-Trained SCAN', type=str, action='store', help='Label for pretrained model')
    args = parser.parse_args()

    dct = {}
    if args.xcdiffpath:
        #xcd = torch.jit.load(args.xcdiffpath)
        xcd = get_scf(xctype='MGGA', path=args.xcdiffpath)
        dct['XCDiff'] = xcd.xc.eval_grid_models
    if args.pretrainedpath:
        ptd = get_scf(xctype=args.pretrainedtype, pretrain_loc=args.pretrainedpath)
        dct[args.pretrainlab] = ptd.xc
    if args.modelpath:
        xc = get_scf(xctype=args.modeltype, path=args.modelpath)
        dct[args.plotlabel] = xc.xc
    plot_fxc(dct, s_range=[0, 3], rs=[1], alpha_range=[0], savename=args.savepref+'_alpha0.pdf')
    plot_fxc(dct, s_range=[0, 3], rs=[1], alpha_range=[1], savename=args.savepref+'_alpha1.pdf')
    plot_fxc(dct, s_range=[0, 3], rs=[1], alpha_range=[10], savename=args.savepref+'_alpha10.pdf')
