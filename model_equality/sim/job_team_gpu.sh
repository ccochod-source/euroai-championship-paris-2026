#!/bin/bash
#SBATCH -J team_gpu
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 14:00:00
#SBATCH -o /p/scratch/training2666/cochod2/euroai-paris/gpu_queue/team_gpu-%j.out
# LE nœud de l'équipe (1 seul nœud pour toute l'équipe). Exécute gpu_queue/gpu<k>/*.sh. Ne PAS lancer d'autre sbatch.
Q=/p/scratch/training2666/cochod2/euroai-paris/gpu_queue
END=$(date -d "$(squeue -j $SLURM_JOB_ID -h -o %e)" +%s)
python3 $Q/gpu_worker.py --gpus 0,1,2,3 --until $END >> $Q/worker.log 2>&1
