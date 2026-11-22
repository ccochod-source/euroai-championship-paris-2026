#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=120
#SBATCH --time=14:00:00
#SBATCH --job-name=s75hold
#SBATCH --output=/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/bruno/informed/logs/hold_%j.out
# Holds one node (4 x A100) for session 75; work is launched inside with: srun --jobid=$SLURM_JOB_ID --overlap ...
echo "node $(hostname) job $SLURM_JOB_ID"; nvidia-smi -L
sleep 50000
