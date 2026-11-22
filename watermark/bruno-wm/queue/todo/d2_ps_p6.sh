export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
while pgrep -u $USER -f "[f]inetune.py --method pixelseal .*--name ftP5 " > /dev/null; do sleep 10; done
python train.py --method pixelseal --name p6 --init /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p5_last.pt --alpha0 1 --steps 16000 --tags d --hpf 1 --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24 --iou_w 2
P=ftP5; [ -f /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_ftP5_testprobs.npz ] || P=p5
python finetune.py --method pixelseal --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/pixelseal_p6_last.pt --name ftP6P --folds 5 --steps 1000 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags d --p_in 0.5 --pseudo $P
