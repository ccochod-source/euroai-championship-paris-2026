# TrustMark round 2, pinned to GPU 0: after ftT5, fine-tune again with pseudo-labelled test images
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
while [ ! -f /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_ftT5_testprobs.npz ] || pgrep -u $USER -f "[f]inetune.py --method trustmark .*--name ftT5 " > /dev/null; do sleep 10; done
python finetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p5_last.pt --name ftT5P --folds 5 --steps 1500 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags d --p_in 0.5 --pseudo ftT5
