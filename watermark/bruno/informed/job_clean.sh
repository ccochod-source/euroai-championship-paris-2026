#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/training2666/dasilvalopes3/wm/logs/clean_%j.out
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
python gen_masks.py && python gen_clean.py
