import torch
import numpy as np
from opt_einsum import contract

def ip_loss(results, loss, **kwargs):
    e_ip = results['e_ip'].unsqueeze(-1)
    e_ip_ref = results['e_ip_ref']
    lip = loss(e_ip, e_ip_ref)
    return lip

def energy_loss(results, loss, **kwargs):
    """energy_loss(results, loss, **kwargs)

        Calculates energy loss of results['E'] with reference results['E_ref'],
        using loss function.

    Args:
        results (dict): [?]
        loss (callable): [?]
        weights (torch.Tensor) [optional]: if specified, scale individual energy differences.
            defaults to a linspace of weights from 0 to 1 of size results['E']
        skip_steps (int) [optional]: slices dE before pass to loss function
            defaults to 0 (i.e. no skipped steps)

    Returns:
        [?]: loss function called on dE, depends on loss function
    """
    E = results['E']
    E_ref = results['E_ref']
    weights = kwargs.get('weights', torch.linspace(0,1,E.size()[0])**2).to(results['E'].device)
    skip_steps = kwargs.get('skip_steps',0)

    dE = weights*(E_ref - E)
    dE = dE[skip_steps:]
    ldE = loss(dE, torch.zeros_like(dE))
    return ldE

def econv_loss(results, loss, **kwargs):
    """econv_loss(results, loss, **kwargs)

        Calculates loss from convergence of the energy, as opposed to with reference.
        i.e. dE = weights*(E - E[-1])

    Args:
        results (dict): [?]
        loss (callable): [?]
        weights (torch.Tensor) [optional]: if specified, scale individual energy differences.
            defaults to a linspace of weights from 0 to 1 of size results['E']
        skip_steps (int) [optional]: slices dE before pass to loss function
            defaults to 0 (i.e. no skipped steps)

    Returns:
        [?]: loss function called on dE, depends on loss function
    """
    E = results['E']
    weights = kwargs.get('weights', torch.linspace(0,1,E.size()[0])**2).to(results['E'].device)
    skip_steps = kwargs.get('skip_steps',0)

    dE = weights*(E - E[-1])
    dE = dE[skip_steps:]
    ldE = loss(dE, torch.zeros_like(dE))
    return ldE

def dm_loss(results, loss, **kwargs):
    """[summary]

    Args:
        results (dict): utilizes keys 'fcenter', 'dm', and 'dm_ref' for contraction
        loss (callable): ?

            **kwargs not currently utilized

    Returns:
        ?: loss called on density matrix contraction difference
    """
    fcenter = results['fcenter'][0]
    dm = results['dm']
    dm_ref = results['dm_ref'][0]
    ddm = contract('ijkl,ij,kl',fcenter,dm,dm) +\
                contract('ijkl,ij,kl',fcenter,dm_ref,dm_ref) -\
                2*contract('ijkl,ij,kl',fcenter,dm,dm_ref)
    ldm = loss(ddm/results['n_elec'][0,0]**2, torch.zeros_like(ddm))
    return ldm

def rho_loss(results, loss, **kwargs):
    """[summary]

    Args:
        results ([type]): [description]
        loss ([type]): [description]


    Returns:
        lrho: loss called on density difference
    """
    print("RHO_LOSS")
    rho_ref = results['rho'][0]
    ao_eval = results['ao_eval'][0]
    dm = results['dm']
    print("RHO_REF, AO_EVAL, DM SHAPES: {}. {}. {}".format(rho_ref.shape, ao_eval.shape, dm.shape))
    if dm.ndim == 2:
        print("2D DM.")
        print("RESULTS N_ELEC: ", results['n_elec'])
        rho = contract('ij,ik,jk->i',
                           ao_eval[0], ao_eval[0], dm)
        print("RHO PRED: {}".format(rho.shape))
        #drho = torch.sqrt(torch.sum((rho-rho_ref)**2*results['grid_weights'])/results['n_elec'][0,0]**2)
        drho = torch.sqrt(torch.sum((rho-rho_ref)**2*results['grid_weights'])/results['n_elec'][0]**2)
        if torch.isnan(drho):
            drho = torch.Tensor([0])
        lrho = loss(drho, torch.zeros_like(drho))

    else:
        print("NON-2D DM")
        rho = contract('ij,ik,xjk->xi',
                           ao_eval[0], ao_eval[0], dm)
        if torch.sum(results['mo_occ']) == 1:
            drho = torch.sqrt(torch.sum((rho[0]-rho_ref[0])**2*results['grid_weights'])/torch.sum(results['mo_occ'][0,0])**2)
        else:
            drho = torch.sqrt(torch.sum((rho[0]-rho_ref[0])**2*results['grid_weights'])/torch.sum(results['mo_occ'][0,0])**2 +\
                   torch.sum((rho[1]-rho_ref[1])**2*results['grid_weights'])/torch.sum(results['mo_occ'][0,1])**2)
        if torch.isnan(drho):
            drho = torch.Tensor([0])
        lrho = loss(drho, torch.zeros_like(drho))
    return lrho

def rho_alt_loss(results, loss, **kwargs):
    """[summary]

    Args:
        results ([type]): [description]
        loss ([type]): [description]

    Returns:
        [type]: [description]
    """
    rho_ref = results['rho'][0]
    ao_eval = results['ao_eval'][0]
    dm = results['dm']
    rho = contract('ij,ik,...jk->...i',
                       ao_eval[0], ao_eval[0], dm)
    if rho.ndim == 2:
        if torch.sum(results['mo_occ']) == 1:
            rho = rho[0]
        else:
            rho = torch.sum(rho, dim=0)
    if rho_ref.ndim == 2:
        if torch.sum(results['mo_occ']) == 1:
            rho_ref = rho_ref[0]
        else:
            rho_ref = torch.sum(rho_ref, dim=0)
        
    drho = torch.sqrt(torch.sum((rho-rho_ref)**2*results['grid_weights'])/results['n_elec'][0,0]**2)
    lrho = loss(drho, torch.zeros_like(drho))

    return lrho

def moe_loss(results, loss, **kwargs):
    """[summary]

    Args:
        results ([type]): [description]
        loss ([type]): [description]

    Returns:
        [type]: [description]
    """
    dmoe = results['mo_energy_ref'][0] - results['mo_energy']

    norbs = kwargs.get('norbs',-1)
    if norbs > -1:
        dmoe = dmoe[:norbs]

    lmoe = loss(dmoe, torch.zeros_like(dmoe))
    return lmoe

def gap_loss(results, loss, nhomo, **kwargs):
    """[summary]

    Args:
        results ([type]): [description]
        loss ([type]): [description]
        nhomo ([type]): [description]

    Returns:
        [type]: [description]
    """
    ref = results['mo_energy_ref'][:,nhomo+1] - results['mo_energy_ref'][:,nhomo]
    pred = results['mo_energy'][nhomo+1] - results['mo_energy'][nhomo]
    dgap = ref - pred
    lgap = loss(dgap, torch.zeros_like(dgap))
    return lgap


def ae_loss(ref_dict,pred_dict, loss, **kwargs):
    """ae_loss(ref_dict, pred_dict, loss, **kwargs):
    
        Calulates atomization energy loss from reference values.

    Args:
        ref_dict ([dict]): A dictionary of reference atomization energies. Only ref_dict[molecule] used,
                            as that is entry storing atomization energy.
        pred_dict ([dict]): A dictionary of predicted atomization energies, whose values are flattened to a list.
        loss (callable): Callable loss function
        weights (torch.Tensor) [optional]: if specified, scale individual energy differences.
            defaults to a linspace of weights from 0 to 1 of size results['E'], or 1 if only one prediction.

    Returns:
        [?]: loss called on weighted difference between reference and prediction
    """
    print("AE_LOSS FUNCTION")
    print("INPUT REF/PRED: ")
    print("REF: {}".format(ref_dict))
    print("PRED: {}".format(pred_dict))
    print("Flattening ref_dict, pred_dict")
    #ref = torch.cat(list(atomization_energies(ref_dict).values()))
    atm_pred = atomization_energies(pred_dict)
    ref = ref_dict[list(atm_pred.keys())[0]]
    pred = torch.cat(list(atm_pred.values()))
    assert len(ref) == 1
    ref = ref.expand(pred.size()[0])
    if pred.size()[0] > 1:
        weights = kwargs.get('weights', torch.linspace(0,1,pred.size()[0])**2).to(pred.device)
    else:
        weights = 1
    lae = loss((ref-pred)*weights,torch.zeros_like(pred))
    print("AE LOSS: {}".format(lae))
    return lae


def atomization_energies(energies):
    """Calculates atomization energies based on a dictionary of molecule/atomic energies.
    
    energies['ABCD'] = molecular energy
    energies['A'], energies['B'], etc. = atomic energy.
    
    Loops over ABCD - A - B - C - D

    Args:
        energies (dict): dictionary of molecule and constituent atomic energies.
    """
    def split(el):
        """Regex split molecule's symbolic expansion into constituent elements.
        No numbers must be present -- CH2 = CHH.

        Args:
            el (str): Molecule symbols

        Returns:
            list: list of individual atoms in molecule
        """
        import re
        res_list = [s for s in re.split("([A-Z][^A-Z]*)", el) if s]
        return res_list


    ae = {}
    for key in energies:
        if isinstance(energies[key],torch.Tensor):
            #if len(split(key)) == 1:continue
            e_tot = torch.clone(energies[key])
            e_tot_size = e_tot.size()
        else:
            e_tot = np.array(energies[key])
            e_tot_size = e_tot.shape
        for symbol in split(key):
            #if single atom, continue
            if len(split(key)) == 1: continue
            e_sub = energies[symbol]
            e_sub_size = e_sub.size() if isinstance(e_sub, torch.Tensor) else e_sub.shape
            if e_tot_size == e_sub_size:
                e_tot -= e_sub
            else:
                e_tot -= e_sub[-1:]
            print('{} - {}: {}'.format(key, symbol, e_tot))
            ae[key] = e_tot
    if ae == {}:
        #empty dict -- no splitting occurred, so single atom
        ae[key] = e_tot
    print("Atomization Energy Final")
    print(ae)
    return ae