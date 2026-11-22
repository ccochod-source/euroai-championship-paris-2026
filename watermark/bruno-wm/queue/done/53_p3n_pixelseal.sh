export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; python train.py --method pixelseal --name p3n --hpf 0 --alpha0 20 --steps 16000 --workers 24 --p_in 0.5 --res_aug 1 --max_minutes 70
