#!/bin/bash
#SBATCH --job-name=eugenia-unlearning
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --output=output/trial-%x-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

method=$1
strength=$2
shift 2
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
checkpoint="output/${method}-${strength}-${SLURM_JOB_ID}.pt"
srun .venv/bin/python -u run_trial.py "$method" "$strength" --out "$checkpoint" "$@"
