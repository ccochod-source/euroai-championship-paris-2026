#!/bin/bash
#SBATCH --job-name=unlearning-baseline
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=output/baseline-%j.log

set -e
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

srun python -u -c '
import os
from pathlib import Path

import task_template

if not task_template.torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the requested GPU.")

print(f"GPU: {task_template.torch.cuda.get_device_name(0)}")
job_id = os.environ["SLURM_JOB_ID"]
task_template.OUTPUT_FILE = Path("output") / f"baseline-{job_id}.pt"
task_template.main()
'
