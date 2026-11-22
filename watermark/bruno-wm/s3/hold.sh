#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --job-name=s3hold
#SBATCH --output=/p/scratch/training2666/dasilvalopes3/wm/s3/out/hold_%j.out
echo $SLURM_JOB_ID > /p/scratch/training2666/dasilvalopes3/wm/s3/JOB_OWN; nvidia-smi -L; sleep 17900
