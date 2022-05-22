import numpy as np
import os, argparse

def ln_cat_prep_dirs(traj1dat, traj2dat, destdir):
    n_traj1 = np.load(traj1dat+'/data_len.npy')
    n_traj2 = np.load(traj2dat+'/data_len.npy')
    n_traj_cat = n_traj1+n_traj2
    print(n_traj_cat)
    np.save(destdir+'/data_len.npy', n_traj_cat)
    nidx = 0
    for idx in range(n_traj1):
        pcklp = 'data_{}.pckl'.format(idx)
        fp = os.path.join(traj1dat, pcklp)
        sp = os.path.join(destdir, pcklp)
        os.symlink(fp, sp)
        nidx += 1
    for idx in range(n_traj2):
        fpcklp = 'data_{}.pckl'.format(idx)
        spcklp = 'data_{}.pckl'.format(nidx)
        fp = os.path.join(traj2dat, fpcklp)
        sp = os.path.join(destdir, spcklp)
        os.symlink(fp, sp)
        nidx += 1

parser = argparse.ArgumentParser(description='symlink two trajectories together to get combined directory')
parser.add_argument('--traj1dat', type=str, action='store', help='directory containing prepared data for first trajectory')
parser.add_argument('--traj2dat', type=str, action='store', help='directory containing prepared data for second trajectory')
parser.add_argument('--destdir', type=str, action='store')
args = parser.parse_args()

if __name__ == '__main__':
    ln_cat_prep_dirs(args.traj1dat, args.traj2dat, args.destdir)