mkdir test && cd test

python ../../../scripts/eval2.py --type MGGA \
--trajpath ../../../data/validation/val_c.traj \
--modelpath ../../../models/xcdiff/MODEL_MGGA/xc \
--forceUKS --basis dzvp