#!/bin/bash
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=3
for m in pixelseal trustmark stegastamp; do python gen_indomain.py $m 2>&1 | grep -E "INDOMAIN_DONE|Error|Traceback" ; done
