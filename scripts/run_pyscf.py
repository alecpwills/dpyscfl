from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree
import pyscf
from ase.io import read, write
from pyscf import dft
import numpy as np
from pyscf import gto, scf, cc
from pyscf.scf import hf, uhf
from pyscf import pbc
from ase import Atoms
import argparse
import dask
import dask.distributed
from dask.distributed  import Client, LocalCluster
import os
from pyscf.tools import cubegen
from opt_einsum import contract
from pyscf import __config__
import pickle

#%%
def write_mycc(idx, atoms, mycc, result):
    """Write out the information in a PySCF kernel that's been evaluated.
    The assumption is that this comes from a CC calculation, as we pass 'ao_repr=True' to make_rdm1, as the default for CC outputs the DM in the MO basis.

    Outputs will be named '{idx}_{atoms.symbols}.x'

    #TODO: refactor, results is already the atoms object
    Args:
        idx (int): Index of molecule in its trajectory. 
        atoms (ase.Atoms): The Atoms object to generate the symbols
        mycc (cc object): The CC kernel that was used to generate the results in result
        result (ase.Atoms): Another Atoms object, whose calc.results['energy'] has been set appropriately
    """
    #if len(mycc.mo_occ.shape) == 2:
        #if array shape length is 2, it is spin polarized -- (2, --) one channel for each spin
        #transpose in this manner -- first dimension retains spin component, just transpose indiv. mo_coeffs
    #    dm = contract('xij,xj,xjk -> xik', mycc.mo_coeff, mycc.mo_occ, np.transpose(mycc.mo_coeff, (0, 2, 1)))
    #else:
    #    dm = contract('ij,j,jk -> ik', mycc.mo_coeff, mycc.mo_occ, mycc.mo_coeff.T)
    #To be consistent with general mf.make_rdm1(), which from the pyscf code comments makes it in AO rep
    #CC calculations have make_rdm1() create it in the MO basis.
    dm = mycc.make_rdm1(ao_repr=True)
    #TODO: Make cubegen grid density optional
    #cubegen.density(mol,'{}.cube'.format(idx), dm, nx=100, ny=100, nz=100,
    #                margin=kwargs['margin'])
    lumo = np.where(mycc.mo_occ == 0)[0][0]
    homo = lumo - 1
    #TODO: Make cubegen HOMO/LUMO optional
    #cubegen.orbital(mol, '{}_lumo.cube'.format(idx), mycc.mo_coeff[lumo],nx=100, ny=100, nz=100,
    #                margin=kwargs['margin'])
    #cubegen.orbital(mol, '{}_homo.cube'.format(idx), mycc.mo_coeff[homo], nx=100, ny=100, nz=100,
    #                margin=kwargs['margin'])
    write('{}_{}.traj'.format(idx, atoms.symbols), result)
    np.save('{}_{}.dm'.format(idx, atoms.symbols), dm)
    np.save('{}_{}.mo_occ'.format(idx, atoms.symbols), mycc.mo_occ)
    np.save('{}_{}.mo_coeff'.format(idx, atoms.symbols), mycc.mo_coeff)
    write('{}_{}.traj'.format(idx, atoms.symbols), result)

#spins for single atoms, since pyscf doesn't guess this correctly.
spins_dict = {
    'Al': 1,
    'B' : 1,
    'Li': 1,
    'Na': 1,
    'Si': 2 ,
    'Be':0,
    'C': 2,
    'Cl': 1,
    'F': 1,
    'H': 1,
    'N': 3,
    'O': 2,
    'P': 3,
    'S': 2,
    'Ar':0, #noble
    'Br':1, #one unpaired electron
    'Ne':0, #noble
    'Sb':3, #same column as N/P
    'Bi':3, #same column as N/P/Sb
    'Te':2, #same column as O/S
    'I':1 #one unpaired electron
}


def get_spin(at):
    #if single atom and spin is not specified in at.info dictionary, use spins_dict
    print('======================')
    print("GET SPIN: Atoms Info")
    print(at)
    print(at.info)
    print('======================')
    if ( (len(at.positions) == 1) and not ('spin' in at.info) ):
        print("Single atom and no spin specified in at.info")
        spin = spins_dict[str(at.symbols)]
    else:
        print("Not a single atom, or spin in at.info")
        if type(at.info.get('spin', None)) == type(0):
            #integer specified in at.info['spin'], so use it
            print('Spin specified in atom info.')
            spin = at.info['spin']
        elif 'radical' in at.info.get('name', ''):
            print('Radical specified in atom.info["name"], assuming spin 1.')
            spin = 1
        elif at.info.get('openshell', None):
            print("Openshell specified in atom info, attempting spin 2.")
            spin = 2
        else:
            print("No specifications in atom info to help, assuming no spin.")
            spin = 0
    return spin



def do_ccsdt(idx,atoms,basis, **kwargs):
    """Run a CCSD(T) (or PBE/SCAN) calculation on an Atoms object, with given basis and kwargs.

    Args:
        idx (int): Index of molecule in its trajectory. 
        atoms (ase.Atoms): The Atoms object that will be used to generate the mol object for the pyscf calculations
        basis (str): Basis type to feed to pyscf

    Raises:
        ValueError: Raised if PBC are flagged but no cell length specified.

    Returns:
        result (ase.Atoms): The Atoms object with result.calc.results['energy'] appropriately set for future use.
    """
    result = Atoms(atoms)
    print('======================')
    print("Atoms Info")
    print(atoms.info)
    print('======================')
    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    sping = get_spin(atoms)
    #As implemented in xcdiff/dpyscf prep_data, this is not used
    #Spin is used instead.
    pol = atoms.info.get('pol', None)
    pol = None
    if kwargs.get('forcepol', False):
        pol=True
    owc = kwargs.get('owcharge', False)
    charge = atoms.info.get('charge', GCHARGE)
    print("===========================")
    print("Option Summary: {} ---> {}".format(atoms.symbols, atoms.get_chemical_formula()))
    print("Spin: {}, Polarized: {}, Charge: {}".format(sping, pol, charge))
    print("===========================")
    if owc:
        if charge != GCHARGE:
            print("OVERWRITING GCHARGE WITH ATOMS.INFO['CHARGE']. {} -> {}".format(GCHARGE, charge))

    mol_input = [[s,p] for s,p in zip(spec,pos)]
    if kwargs.get('rerun',True) and os.path.isfile('{}_{}.traj'.format(idx, atoms.symbols)):
        print("reading in traj file {}_{}.traj".format(idx, atoms.symbols))
        result = read('{}_{}.traj'.format(idx, atoms.symbols))
        return result
    print('Generating mol {} with charge {}'.format(idx, charge))
    if kwargs['PBC'] == False:
        #guess spin
        #mol = gto.M(atom=mol_input, basis=basis, charge=charge, spin=spin)
        molgen = False
        scount = 0
        while not molgen:
            try:
                mol = gto.M(atom=mol_input, basis=basis, spin=sping-scount, charge=charge)
                molgen=True
            except RuntimeError:
                #spin disparity somehow, try with one less until 0
                print("RuntimeError. Trying with reduced spin.")
                scount += 1
                if sping-scount < 0:
                    raise ValueError

        print('S: ', mol.spin)
    elif kwargs['PBC'] == True:
        if kwargs['L'] == None:
            raise ValueError('Cannot specify PBC without cell length')
        print('Generating periodic cell of length {}'.format(kwargs['L']))
        cell = np.eye(3)*kwargs['L']
        if kwargs['pseudo'] == None:
            mol = pbc.gto.M(a=cell, atom=mol_input, basis=basis, charge=charge, spin=spin)
        elif kwargs['pseudo'] == 'pbe':
            print('Assigning pseudopotential: GTH-PBE to all atoms')
            mol = pbc.gto.M(a=cell, atom=mol_input, basis=basis, charge=charge, pseudo='gth-pbe')
    if kwargs.get('testgen', None):
        print('{} Generated.'.format(atoms.get_chemical_formula()))
        return 0
    if kwargs['XC'] == 'ccsdt':
        print('CCSD(T) calculation commencing')
        #If pol specified, it's a bool and takes precedence.
        if type(pol) == bool:
            #polarization specified, UHF
            if pol:
                mf = uhf.UHF(mol)
            #specified to not polarize, RHF
            else:
                mf = hf.RHF(mol)
        #if pol is not specified in atom info, spin value used instead
        elif pol == None:
            if (mol.spin != 0):
                mf = scf.UHF(mol)
            else:
                mf = scf.RHF(mol)

        print("METHOD: ", mf)
        if kwargs.get('chk', True):
            mf.set(chkfile='{}_{}.chkpt'.format(idx, atoms.symbols))
        if kwargs['restart']:
            print("Restart Flagged -- Setting mf.init_guess to chkfile")
            mf.init_guess = '{}_{}.chkpt'.format(idx, atoms.symbols)
        print("Running HF calculation")
        mf.run()
        print("Running CCSD calculation from HF")
        mycc = cc.CCSD(mf)
        try:
            mycc.kernel()
        except AssertionError:
            print("CCSD Failed. Stopping at HF")
            result.calc = SinglePointCalculator(result)
            ehf = (mf.e_tot) 
            etot = ehf
            eccsd = None
            eccsdt = None
            result.calc.results = {'energy': etot,
                                    'e_hf': ehf,
                                    'e_ccsd': None,
                                    'e_ccsdt':None}
            with open('progress','a') as progfile:
                progfile.write('{} \t {} \t {} \t {} \t {} \t {} \n'.format(idx, atoms.symbols, etot, ehf, eccsd, eccsdt))
            write_mycc(idx, atoms, mf, result)
            return result

        print('MO Occ shape: ', mycc.mo_occ.shape)
        print("Running CCSD(T) calculation from CCSD")
        try:
            ccsdt = mycc.ccsd_t()
        except ZeroDivisionError:
            print("CCSD(T) Failed. DIV/0. Stopping at CCSD")
            result.calc = SinglePointCalculator(result)
            ehf = (mf.e_tot) 
            eccsd = (mycc.e_tot) 
            eccsdt = None
            etot = eccsd
            result.calc.results = {'energy': etot,
                                    'e_hf': ehf,
                                    'e_ccsd': eccsd,
                                    'e_ccsdt':eccsdt}
            with open('progress','a') as progfile:
                progfile.write('{} \t {} \t {} \t {} \t {} \t {} \n'.format(idx, atoms.symbols, etot, ehf, eccsd, eccsdt))
            write_mycc(idx, atoms, mycc, result)
            return result

        result.calc = SinglePointCalculator(result)
        etot = (mycc.e_tot + ccsdt) 
        ehf = (mf.e_tot) 
        eccsd = (mycc.e_tot) 
        eccsdt = (ccsdt) 
        result.calc.results = {'energy': etot,
                                'e_hf': ehf,
                                'e_ccsd': eccsd,
                                'e_ccsdt': eccsdt}
        with open('progress','a') as progfile:
            progfile.write('{} \t {} \t {} \t {} \t {} \t {} \n'.format(idx, atoms.symbols, etot, ehf, eccsd, eccsdt))

        write_mycc(idx, atoms, mycc, result)

    elif kwargs['XC'].lower() in ['pbe', 'scan']:
        print('{} calculation commencing'.format(kwargs['XC']))
            #If pol specified, it's a bool and takes precedence.
        if type(pol) == bool:
            #polarization specified, UHF
            if pol:
                mf = dft.UKS(mol)
                method = dft.UKS
            #specified to not polarize, RHF
            else:
                mf = dft.RKS(mol)
                method = dft.RKS
        #if pol is not specified in atom info, spin value used instead
        elif pol == None:
            if (mol.spin != 0):
                mf = dft.UKS(mol)
                method = dft.UKS
            else:
                mf = dft.RKS(mol)
                method = dft.RKS

        print("METHOD: ", mf)
        if kwargs.get('chk', True):
            mf.set(chkfile='{}_{}.chkpt'.format(idx, atoms.symbols))
        if kwargs['restart']:
            print("Restart Flagged -- Setting mf.init_guess to chkfile")
            mf.init_guess = '{}_{}.chkpt'.format(idx, atoms.symbols)

        mf.grids.level = 5
        mf.max_cycle = 500
        print("Running {} calculation".format(kwargs['XC']))
        #if kwargs['df'] == False:
        #    mf = dft.RKS(mol)
        #elif kwargs['df'] == True:
        #    print('Using density fitting')
        #    mf = dft.RKS(mol).density_fit()
        mf.xc = '{},{}'.format(kwargs['XC'].lower(), kwargs['XC'].lower())
        mf.kernel()
        if not mf.converged:
            print("Calculation did not converge. Trying second order convergence with PBE to feed into calculation.")
            mfp = method(mol, xc='pbe').newton()
            mfp.kernel()
            print("PBE Calculation complete -- feeding into original kernel.")
            mf.kernel(dm0 = mfp.make_rdm1())
            if not mf.converged:
                print("Convergence still failed -- {}".format(atoms.symbols))
                with open('unconv', 'a') as f:
                    f.write('{} \t {} \t {} \n'.format(idx, atoms.symbols, mf.e_tot))

        if kwargs['df'] == True:
            print('Default auxbasis', mf.with_df.auxmol.basis)
        result.calc = SinglePointCalculator(result)
        result.calc.results = {'energy':mf.e_tot }
        with open('progress','a') as progfile:
            progfile.write('{} \t {} \t {} \n'.format(idx, atoms.symbols, mf.e_tot ))
        #np.savetxt('{}.m_occ'.format(idx), mf.mo_occ, delimiter=' ')
        #np.savetxt('{}.mo_coeff'.format(idx), mf.mo_coeff, delimiter=' ')
        write_mycc(idx, atoms, mf, result)
    
    return result

def calculate_distributed(atoms, n_workers = -1, basis='6-311++G(3df,2pd)', **kwargs):
    """_summary_

    Args:
        atoms (_type_): _description_
        n_workers (int, optional): _description_. Defaults to -1.
        basis (str, optional): _description_. Defaults to '6-311++G(3df,2pd)'.

    Returns:
        _type_: _description_
    """
    
    print('Calculating {} systems on'.format(len(atoms)))
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1)
    print(cluster)
    client = Client(cluster)

    futures = client.map(do_ccsdt, np.arange(len(atoms)),atoms,[basis]*len(atoms), **kwargs)
    
    return [f.result() for f in futures]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('xyz', action='store', help ='Path to .xyz/.traj file containing list of configurations')
    parser.add_argument('-charge', '-c', action='store', type=int, help='Net charge of the system', default=0)
    parser.add_argument('-fdf', metavar='fdf', type=str, nargs = '?', default='')
    parser.add_argument('-basis', metavar='basis', type=str, nargs = '?', default='6-311++G(3df,2pd)', help='basis to use. default 6-311++G(3df,2pd)')
    parser.add_argument('-nworkers', metavar='nworkers', type=int, nargs = '?', default=1)
    parser.add_argument('-cmargin', '-cm', type=float, default=10.0, help='the margin to define extent of cube files generated')
    parser.add_argument('-xc', '--xc', type=str, default='pbe', help='Type of XC calculation. Either pbe or ccsdt.')
    parser.add_argument('-pbc', '--pbc', type=bool, default=False, help='Whether to do PBC calculation or not.')
    parser.add_argument('-L', '--L', type=float, default=None, help='The box length for the PBC cell')
    parser.add_argument('-df', '--df', type=bool, default=False, help='Choose whether or not the DFT calculation uses density fitting')
    parser.add_argument('-ps', '--pseudo', type=str, default=None, help='Pseudopotential choice. Currently either none or pbe')
    parser.add_argument('-r', '--rerun', type=bool, default=False, help='whether or not to continue and skip previously completed calculations or redo all')
    parser.add_argument('--ghf', default=False, action="store_true", help='whether to have wrapper guess HF to use or do GHF. if flagged, ')
    parser.add_argument('--serial', default=False, action="store_true", help="Run in serial, without DASK.")
    parser.add_argument('--overwrite_gcharge', default=False, action="store_true", help='Whether to try to overwrite specified CHARGE -c if atom.info has charge.')
    parser.add_argument('--restart', default=False, action="store_true", help='If flagged, will use checkfile as init guess for calculations.')
    parser.add_argument('--forcepol', default=False, action='store_true', help='If flagged, all calculations are spin polarized.')
    parser.add_argument('--testgen', default=False, action='store_true', help='If flagged, calculation stops after mol generation.')
    parser.add_argument('--startind', default=-1, type=int, action='store', help='SERIAL MODE ONLY. If specified, will skip indices in trajectory before given value')
    args = parser.parse_args()
    setattr(__config__, 'cubegen_box_margin', args.cmargin)
    GCHARGE = args.charge
    atoms = read(args.xyz, ':')
    print("==================")
    print("ARGS SUMMARY")
    print(args)
    print("==================")
    if not args.rerun:
        print('beginning new progress file')
        with open('progress','w') as progfile:
            progfile.write('#idx \t atoms.symbols \t etot  (Har) \t ehf  (Har) \t eccsd  (Har) \t eccsdt  (Har) \n')
        print('beginning new nonconverged file')
        with open('unconv','w') as ucfile:
            ucfile.write('#idx \t atoms.symbols \t etot  (Har) \t ehf  (Har) \t eccsd  (Har) \t eccsdt  (Har) \n')
    if not args.serial:
        results = calculate_distributed(atoms, args.nworkers, args.basis,
                                        margin=args.cmargin,
                                        XC=args.xc,
                                        PBC=args.pbc,
                                        L=args.L,
                                        df=args.df,
                                        pseudo=args.pseudo,
                                        rerun=args.rerun,
                                        owcharge=args.overwite_gcharge,
                                        forcepol = args.forcepol,
                                        testgen = args.testgen)
    else:
        print("SERIAL CALCULATIONS")
        results = [do_ccsdt(ia, atoms[ia], basis=args.basis,
                                        margin=args.cmargin,
                                        XC=args.xc,
                                        PBC=args.pbc,
                                        L=args.L,
                                        df=args.df,
                                        pseudo=args.pseudo,
                                        rerun=args.rerun,
                                        owcharge=args.overwrite_gcharge,
                                        restart = args.restart,
                                        forcepol = args.forcepol,
                                        testgen = args.testgen) for ia in range(len(atoms)) if ia >= args.startind]

    results_path = 'results.traj'
    write(results_path, results)
