#!/bin/bash
# TIMEOUT=3600
umask 002
W=/p/scratch/training2666/dasilvalopes3; PY=$W/wm/kit/.venv_team/bin/python; O=$W/wm/out
cd $W/wm/solution || exit 1
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache MPLCONFIGDIR=$W/.cache/mpl PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$PY -u finetune.py --method pixelseal --hpf 1 --init_ckpt $O/pixelseal_p5_last.pt --name ftP5 --folds 5 --steps 1000 --real_bs 12 --mosaic 0.5 --syn_bs 4 --tags d --p_in 0.5
