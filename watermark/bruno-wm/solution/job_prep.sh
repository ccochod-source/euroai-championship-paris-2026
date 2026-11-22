#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=/p/scratch/training2666/dasilvalopes3/wm/logs/prep_%j.out
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
[ -f $W/syn/shape_bank.pkl ] || python gen_masks.py
[ -f $W/syn/clean.npy ] || python gen_clean.py
ls -la $W/syn
