#!/bin/bash
# TIMEOUT=1500
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name fin_k4_g16_s6000 --targets orehn/cache/q_oof_k4.pt --gamma 16 --train last6 --lr 1e-4 --steps 6000
