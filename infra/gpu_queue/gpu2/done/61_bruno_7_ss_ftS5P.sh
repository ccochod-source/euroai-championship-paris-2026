#!/bin/bash
# TIMEOUT=3600
umask 002
W=/p/scratch/training2666/dasilvalopes3; PY=$W/wm/kit/.venv_team/bin/python; O=$W/wm/out
cd $W/wm/solution || exit 1
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache MPLCONFIGDIR=$W/.cache/mpl PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
[ -f $O/stegastamp_p5_last.pt ] || { echo no p5 checkpoint; exit 1; }
$PY -u finetune.py --method stegastamp --hpf 1 --init_ckpt $O/stegastamp_p5_last.pt --name ftS5P --folds 5 --steps 1500 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags d --p_in 0.5 --pseudo ftP
