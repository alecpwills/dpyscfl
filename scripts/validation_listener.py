import pyscf
from pyscf import gto,dft,scf
import numpy as np
import scipy
from ase import Atoms
from ase.io import read
import pylibnxc
import sys
import pickle
from dpyscf.utils import get_rho
import os
from ase.units import Hartree
import time
import torch
from validation import run_validate
import json
torch.set_default_dtype(torch.double)
basis = '6-311++G(3df,2pd)'

best_error = 100000
last_time = 0
if __name__ == '__main__':
    logpath = sys.argv[1]


    with open(logpath + '.config','r') as file:
        args= json.loads(file.read())

    RHO_mult = args['rho_weight']
    while(True):
#         paths = [logpath + '_{}.chkpt'.format(i) for i in range(3)]
        paths = [logpath + '_current.pt'.format(i) for i in range(3)]
        times = {}
        for p in paths:
            if os.path.exists(p):
                if os.path.getmtime(p) > last_time:
                    times[os.path.getmtime(p)] = p

        if len(times):
            last_time = max(times)
            path = times[max(times)]
        else:
            time.sleep(10)
            continue

        try:
            with open(logpath + '.log','r') as file:
                for line in file:
                    pass
        except:
            line = '\n'

        time.sleep(5)
        print('Calculating validation set for ' + path)
        
        scf = torch.load(path)

        ae_error, rho_error = run_validate(scf.xc, do_bh=True)
        error = ae_error + np.sqrt(RHO_mult) * rho_error
        best_error = min(best_error, error)


        with open(logpath + '.log.val','a') as file:
            file.write('TRAIN:' + line)
            file.write('VAL: AE: {:.5f} | Rho: {:.5f} | {:.5f} \n'.format(ae_error, rho_error, error))

            if error == best_error:
                file.write('Saving model...\n')
                torch.save(scf.xc.state_dict(), logpath + '_val.chkpt')
