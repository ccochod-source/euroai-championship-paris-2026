# TrustMark, GPU 0: longer pre-training on the confirmed canvas law (e ~ U[0,90]) with small holes emphasised, then fine-tune + pseudo-labels
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
[ -f /p/scratch/training2666/dasilvalopes3/syn/trustmark_info_f.npy ] || python gen_wm4.py trustmark 8000 0 90 f in6
python train.py --method trustmark --name p6 --init /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p5_last.pt --alpha0 1 --steps 14000 --tags f $C0 --hpf 1 --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24 --iou_w 2
python finetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p6_last.pt --name ftT6P --folds 5 --steps 1000 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags f --p_in 0.5 --pseudo ftT5
