# TrustMark round 2, pinned to GPU 0: regenerate full-type data with the measured law e ~ U[0, 90]; after ftT5: pseudo-label fine-tune, then
# continue pre-training on the corrected data and fine-tune again with pseudo-labels
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
python gen_wm4.py trustmark 8000 0 90 f in6
while [ ! -f /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_ftT5_testprobs.npz ] || pgrep -u $USER -f "[f]inetune.py --method trustmark .*--name ftT5 " > /dev/null; do sleep 10; done
python finetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p5_last.pt --name ftT5P --folds 5 --steps 1500 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags f --p_in 0.5 --pseudo ftT5
python train.py --method trustmark --name p6 --hpf 1 --init /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p5_last.pt --alpha0 1 --steps 8000 --tags f --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24
python finetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p6_last.pt --name ftT6P --folds 5 --steps 1500 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags f --p_in 0.5 --pseudo ftT5P
