#!/bin/bash
# usage (on the compute node, through sg): run.sh <gpu index> <script.py> [args]
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/s3; source $W/wm/kit/.venv/bin/activate
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache PYTHONWARNINGS=ignore
export CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$W/wm/s3:$W/wm/kit PYTHONUNBUFFERED=1; shift
python "$@" 2>&1 | grep --line-buffered -v -i -E "warn|ffmpeg|Model loaded|torch.load|state_dict =|loaded successfully"
