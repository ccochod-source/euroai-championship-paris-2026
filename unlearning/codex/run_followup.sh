#!/bin/bash
# TIMEOUT=2400
set -euo pipefail
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python -u codex/train.py --mode npo --name orehn_A045_reference --steps 0 --save 0 --checkpoint orehn/ckpt/ft2_A045.pt --full-eval
python -u codex/train.py --mode npo --name npo_stable_cap3_last4_s900 --steps 900 --beta 0.5 --cap 3 --last 4 --gamma 4 --alt-kl 1 --recovery 1 --ref-conf 0.8 --extra-bright 0.35 --full-eval
# Regenerate the teacher to persist its probe heads for dynamic altered-view targets.
python -u codex/retain_teacher.py
python -u codex/train.py --mode retain_teacher_online --name rt_online_full_s1600 --steps 1600 --last 12 --lr 1e-4 --gamma 4 --extra-bright 0.2 --full-eval
