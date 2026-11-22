# PixelSeal pipeline, pinned to GPU 0: wait for TrustMark p3 to finish, continue pre-training on the canvas-model data, fine-tune on real images
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
while pgrep -u $USER -f "[t]rain.py --method trustmark --name p3 " > /dev/null; do sleep 10; done
while [ ! -f /p/scratch/training2666/dasilvalopes3/syn/pixelseal_info_c.npy ]; do sleep 10; done
cp /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p3b_last.pt /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p3b_final.pt
python train.py --method pixelseal --name p4 --hpf 1 --init /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p3b_final.pt --alpha0 3 --steps 9000 --tags c --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24
python finetune.py --method pixelseal --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p4_last.pt --name ftP4 --folds 5 --steps 1500 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags c --p_in 0.5
