#!/bin/bash
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=3
python diag.py "$@" 2>&1 | grep -v -i -E "warn|ffmpeg|Model loaded"
