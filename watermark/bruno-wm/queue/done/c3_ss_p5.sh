# StegaStamp, GPU 3: square-canvas data (e ~ U[8,40]), pre-train from the augmented checkpoint, fine-tune + pseudo-labels
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3
python gen_wm4.py stegastamp 8000
python train.py --method stegastamp --name p5 --init /p/scratch/training2666/dasilvalopes3/wm/out/stegastamp_p2_last.pt --alpha0 1 --steps 9000 --tags d --hpf 1 --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24 --iou_w 2
python finetune.py --method stegastamp --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/stegastamp_p5_last.pt --name ftS5P --folds 5 --steps 1500 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags d --p_in 0.5 --pseudo ftP
