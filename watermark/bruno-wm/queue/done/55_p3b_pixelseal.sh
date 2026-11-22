export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; python train.py --method pixelseal --name p3b --hpf 1 --alpha0 50 --steps 16000 --workers 24 --p_in 0.5 --res_aug 1 --max_minutes 70
