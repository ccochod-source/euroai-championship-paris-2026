#!/bin/bash
TIMEOUT=3600
cd /p/scratch/training2666/dasilvalopes3/wm/solution
source /p/scratch/training2666/dasilvalopes3/wm/kit/.venv/bin/activate
export HF_HOME=/p/scratch/training2666/dasilvalopes3/.cache/hf TORCH_HOME=/p/scratch/training2666/dasilvalopes3/.cache/torch XDG_CACHE_HOME=/p/scratch/training2666/dasilvalopes3/.cache PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u train.py --method trustmark --name p6 --init /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p5_last.pt --alpha0 1 --steps 14000 --tags f --hpf 1 --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 8 --iou_w 2
