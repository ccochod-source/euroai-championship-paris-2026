# PixelSeal, pinned to GPU 1: square-canvas data, continue pre-training from p3b, fine-tune on real images
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
python gen_wm4.py pixelseal 8000
while pgrep -u $USER -f "[f]inetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p3_snapA.pt" > /dev/null; do sleep 5; done
cp /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p3b_best.pt /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p3b_init.pt
python train.py --method pixelseal --name p5 --hpf 1 --init /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p3b_init.pt --alpha0 2 --steps 9000 --tags d --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24
python finetune.py --method pixelseal --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p5_last.pt --name ftP5 --folds 5 --steps 1000 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags d --p_in 0.5
