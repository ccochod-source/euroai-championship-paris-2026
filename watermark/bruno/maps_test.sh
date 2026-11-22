#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=wm_maps_test
#SBATCH --output=/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/bruno/logs/%x_%j.out
export CUDA_VISIBLE_DEVICES=0
cd /p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
srun python bruno/exp1_maps.py --kit . --data release --split test --out bruno/maps_test.npz && python bruno/stega_pp.py --kit . --data release --split test --maps bruno/maps_test.npz --out bruno/maps_test2.npz
