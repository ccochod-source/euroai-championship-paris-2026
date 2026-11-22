#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=120
#SBATCH --time=04:00:00
#SBATCH --output=/p/scratch/training2666/dasilvalopes3/wm/logs/full_%j.out
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache PYTHONWARNINGS=ignore
L=$W/wm/logs
set -x
[ -f $W/syn/shape_bank.pkl ] || python gen_masks.py
[ -f $W/syn/clean.npy ] || python gen_clean.py
ls -la $W/syn
g=0; for m in pixelseal trustmark stegastamp; do CUDA_VISIBLE_DEVICES=$g python gen_wm.py $m a > $L/gen_$m.log 2>&1 & g=$((g+1)); done; wait
grep -h WM_DONE $L/gen_*.log
g=0; for m in pixelseal trustmark stegastamp; do CUDA_VISIBLE_DEVICES=$g python train.py --method $m --steps ${STEPS:-20000} --workers 32 --name v1 > $L/train_${m}_v1.log 2>&1 & g=$((g+1)); done; wait
grep -h TRAIN_DONE $L/train_*_v1.log
