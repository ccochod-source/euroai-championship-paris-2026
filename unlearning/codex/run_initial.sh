#!/bin/bash
# TIMEOUT=2400
set -euo pipefail
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python -u codex/train.py --mode npo --name codex_original --steps 0 --save 0 --full-eval
python -u codex/train.py --mode npo --name npo_b05_cap6_last4_s700 --steps 700 --beta 0.5 --cap 6 --last 4 --gamma 4 --full-eval
python -u codex/retain_teacher.py
python -u codex/train.py --mode retain_teacher --name rt_b678_last4_s700 --steps 700 --last 4 --gamma 4 --full-eval
