mkdir test && cd test

python ../../../scripts/run_pyscf.py ../../../data/validation/val_c.traj \
--xc ccsdt \
--serial \
--forcepol \
-basis dzvp