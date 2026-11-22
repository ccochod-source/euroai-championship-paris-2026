#!/bin/bash
#SBATCH --job-name=eugenia-unlearn-views
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --time=00:12:00
#SBATCH --output=output/trial-%x-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
bundle=$1
pids=()
for depth in 0 2; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --output="output/multiview-${SLURM_JOB_ID}-depth${depth}.log" \
        .venv/bin/python -u "$bundle/multiview_blocks.py" \
        --depth "$depth" --public "$bundle/public.zip" --record "$bundle/record.pt" \
        --out "output/multiview-${SLURM_JOB_ID}-depth${depth}-patch.pt" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
