# PixelSeal round 2, pinned to GPU 1: after ftP5, continue pre-training on both canvas variants (squares up to +200, rectangles up to +512), fine-tune, then pseudo-labels
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
while [ ! -f /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_ftP5_testprobs.npz ] || pgrep -u $USER -f "[f]inetune.py --method pixelseal .*--name ftP5 " > /dev/null; do sleep 10; done
python train.py --method pixelseal --name p6 --hpf 1 --init /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p5_last.pt --alpha0 1.5 --steps 8000 --tags d,e --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24
python finetune.py --method pixelseal --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p6_last.pt --name ftP6 --folds 5 --steps 1500 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags d,e --p_in 0.5 --pseudo ftP5
