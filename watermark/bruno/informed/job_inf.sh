#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=120
#SBATCH --time=02:00:00
#SBATCH --job-name=wm_informed
#SBATCH --output=/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/bruno/informed/logs/job_%j.out
T=/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task
cd $T/bruno/informed; source $T/.venv/bin/activate
export PYTHONWARNINGS=ignore INF_OUT=$T/bruno/informed/out
L=$T/bruno/informed/logs; S=${STEPS:-14000}; MM=${MAXMIN:-108}
CUDA_VISIBLE_DEVICES=0 python itrain.py --method pixelseal  --steps $S --workers 28 --name i1 --max_minutes $MM > $L/pixelseal_i1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python itrain.py --method trustmark  --steps $S --workers 28 --name i1 --max_minutes $MM > $L/trustmark_i1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python itrain.py --method stegastamp --steps $S --workers 28 --name i1 --max_minutes $MM > $L/stegastamp_i1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python itrain.py --method pixelseal  --steps $S --workers 28 --name i1b --lr 2e-4 --amax 12 --warm 0.5 --max_minutes $MM > $L/pixelseal_i1b.log 2>&1 &
wait
grep -h TRAIN_DONE $L/*_i1*.log
