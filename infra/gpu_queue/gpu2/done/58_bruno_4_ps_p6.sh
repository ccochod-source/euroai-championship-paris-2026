#!/bin/bash
# TIMEOUT=3600
umask 002
W=/p/scratch/training2666/dasilvalopes3; PY=$W/wm/kit/.venv_team/bin/python; O=$W/wm/out
cd $W/wm/solution || exit 1
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache MPLCONFIGDIR=$W/.cache/mpl PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$PY -u train.py --method pixelseal --name p6 --init $O/pixelseal_p5_last.pt --alpha0 1 --steps 16000 --tags d --hpf 1 --p_in 0.5 --res_aug 1 --lr 3e-4 --workers 12 --iou_w 2 --max_minutes 52
