# TrustMark pipeline, pinned to GPU 1: wait for the first fine-tune, continue pre-training on the canvas-model data, fine-tune again
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
while pgrep -u $USER -f "[t]rain.py --method trustmark --name p3 " > /dev/null; do sleep 10; done
cp /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p3_last.pt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p3_final.pt
while pgrep -u $USER -f "[f]inetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p3_snapA.pt" > /dev/null; do sleep 10; done
python train.py --method trustmark --name p4 --hpf 1 --init /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p3_final.pt --alpha0 2 --steps 7000 --tags c --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 24
python finetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p4_last.pt --name ftT4 --folds 5 --steps 1500 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags c --p_in 0.5
