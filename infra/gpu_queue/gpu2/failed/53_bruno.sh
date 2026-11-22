#!/bin/bash
TIMEOUT=3600
cd /p/scratch/training2666/dasilvalopes3/wm/solution
source /p/scratch/training2666/dasilvalopes3/wm/kit/.venv/bin/activate
export HF_HOME=/p/scratch/training2666/dasilvalopes3/.cache/hf TORCH_HOME=/p/scratch/training2666/dasilvalopes3/.cache/torch XDG_CACHE_HOME=/p/scratch/training2666/dasilvalopes3/.cache PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u finetune.py --method trustmark --hpf 1 --init_ckpt /p/scratch/training2666/dasilvalopes3/wm/out/trustmark_p6_last.pt --name ftT6P --folds 5 --steps 1000 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags f --p_in 0.5 --pseudo ftT5
