export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; python train.py --method stegastamp --name p2 --steps 12000 --workers 24 --hpf 1 --alpha0 2  --p_in 0.5 --res_aug 1 --max_minutes 60
