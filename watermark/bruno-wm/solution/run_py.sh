#!/bin/bash
# usage: run_py.sh <gpu> <script.py> [args...]
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1; shift
python "$@" 2>&1 | grep --line-buffered -v -i -E "warn|ffmpeg|Model loaded|torch.load|state_dict ="
