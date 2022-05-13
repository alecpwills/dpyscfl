#!/usr/bin/env python
# coding: utf-8
import numpy as np
from ase import Atoms
from ase.io import read
from dpyscfl.net import *
from dpyscfl.utils import *
from ase.units import Bohr
import argparse
import sys
import os
import pickle


parser = argparse.ArgumentParser(description='Train xc functional')
parser.add_argument('writeloc', action='store', type=str, help='DIRECTORY location of where to write the MemDatasetWrite. If directory exists, it will just populate with files -- make sure it is unique to the dataset.')
parser.add_argument('func', action='store', choices=['PBE','SCAN'], help='The XC functional to run for baseline calculations.')
parser.add_argument('atoms', action='store', type=str, help='Location of the .xyz/.traj file to read into ASE Atoms object, which will be used to generate baseline.')
parser.add_argument('-r', '--ref_path', action='store', default='', help='Location of reference DMs and energies.')
parser.add_argument('--sequential', action="store_true", help='Whether to get_datapoint individually or use list then write')
parser.add_argument('--dfit', action='store_true', default=False, help='Generate density fitting matrices or not')
args = parser.parse_args()


if __name__ == '__main__':
    #Must be three arguments: "dataset location" where it is WRITTEN, "functional", and a third choice corresponding to something readable by ase.io.read
    if len(sys.argv) < 3:
        raise Exception("Must provide dataset location and functional")
    loc = args.writeloc
    func = args.func
    #ALEC
    print(loc)
    print(func)

    #functional choice must be PBE or SCAN
    if func not in ['PBE','SCAN']:
        raise Exception("Functional has to be either SCAN or PBE")

    #create the ASE atoms object from reading specified file, and create corresponding indices.
    atoms = read(args.atoms,':')
    indices = np.arange(len(atoms))

    #as implemented, pol is always false.
    #in the function call, this is empty
    pol = {}
    basis = '6-311++G(3df,2pd)'

    distances = np.arange(len(atoms))
    if not args.sequential:
        baseline = [old_get_datapoint(d, basis=basis, grid_level=d.info.get('grid_level', 1),
                                xc=func, zsym=d.info.get('sym',True),
                                n_rad=d.info.get('n_rad',30), n_ang=d.info.get('n_ang',15),
                                init_guess=False, spin = d.info.get('spin',None),
                                pol=pol.get(''.join(d.get_chemical_symbols()), False), do_fcenter=True,
                                ref_path=args.ref_path, ref_index= idx,ref_basis='6-311++G(3df,2pd)', dfit=args.dfit) for idx, d in zip(indices, atoms)]

        E_base =  [r[0] for r in baseline]
        DM_base = [r[1] for r in baseline]
        inputs = [r[2] for r in baseline]
        inputs = {key: [i.get(key,None) for i in inputs] for key in inputs[0]}
        
        try:
            os.mkdir(loc)
        except FileExistsError:
            pass
        #dataset = MemDatasetWrite(loc = loc, Etot = E_base, dm = DM_base, **inputs)
        dataset = MemDatasetWrite(loc = loc, Etot = E_base, identities = DM_base, **inputs)
    else:
        for idx, d in zip(indices, atoms):
            baseline = old_get_datapoint(d, basis=basis, grid_level=d.info.get('grid_level', 1),
                        xc=func, zsym=d.info.get('sym',True),
                        n_rad=d.info.get('n_rad',30), n_ang=d.info.get('n_ang',15),
                        init_guess=False, spin = d.info.get('spin',None),
                        pol=pol.get(''.join(d.get_chemical_symbols()), False), do_fcenter=True,
                        ref_path=args.ref_path, ref_index= idx,ref_basis='6-311++G(3df,2pd)', dfit=args.dfit)

            E_base =  baseline[0]
            DM_base = baseline[1]
            inputs = baseline[2]
            inputs['Etot'] = E_base
            inputs['dm'] = DM_base
            try:
                os.mkdir(args.writeloc)
            except FileExistsError:
                pass
            with open(args.writeloc+'_{}.pckl'.format(idx), 'wb') as datafile:
                datafile.write(pickle.dumps(inputs))
