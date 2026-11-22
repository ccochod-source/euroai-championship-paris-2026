# TrustMark, pinned to GPU 0: square-canvas data, continue pre-training from p3, fine-tune on real images
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
python gen_wm4.py trustmark 8000
python train.py --method trustmark --name p5 --hpf 1 --init /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p3_best.pt --alpha0 1.5 --steps 7000 --tags d --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24
python finetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p5_last.pt --name ftT5 --folds 5 --steps 1000 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags d --p_in 0.5
