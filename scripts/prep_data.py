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
import pickle as pkl


parser = argparse.ArgumentParser(description='Train xc functional')
parser.add_argument('writeloc', action='store', type=str, help='DIRECTORY location of where to write the MemDatasetWrite. If directory exists, it will just populate with files -- make sure it is unique to the dataset.')
parser.add_argument('func', action='store', choices=['PBE','SCAN'], help='The XC functional to run for baseline calculations.')
parser.add_argument('atoms', action='store', type=str, help='Location of the .xyz/.traj file to read into ASE Atoms object, which will be used to generate baseline.')
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

    #TESTING
#     # indices = [0, 16]
#     # atoms = [atoms[i] for i in indices]
    
#     # indices = [0, 11, 16, 17]
#     if func =='PBE':
#         pol ={'FF':True,'LiF':True,'LiH':True}
#     else:
#         pol ={}

    pol = {}
    basis = '6-311++G(3df,2pd)'

    distances = np.arange(len(atoms))

#TODO: fix hard path to ../danotta/ref/6-311
    baseline = [old_get_datapoint(d, basis=basis, grid_level=d.info.get('grid_level', 1),
                              xc=func, zsym=d.info.get('sym',True),
                              n_rad=d.info.get('n_rad',30), n_ang=d.info.get('n_ang',15),
                              init_guess=False, spin = d.info.get('spin',0),
                              pol=pol.get(''.join(d.get_chemical_symbols()), False), do_fcenter=False,
                              ref_path='../data/ref/6-311/', ref_index= idx,ref_basis='6-311++G(3df,2pd)', dfit=True) for idx, d in zip(indices, atoms)]

    E_base =  [r[0] for r in baseline]
    DM_base = [r[1] for r in baseline]
    inputs = [r[2] for r in baseline]
    inputs = {key: [i.get(key,None) for i in inputs] for key in inputs[0]}
    
    #There's no reason for this??
    #DM_ref = DM_base
    #E_ref = E_base

    try:
        os.mkdir(loc)
    except FileExistsError:
        pass
    dataset = MemDatasetWrite(loc = loc, Etot = E_base, dm = DM_base, **inputs)