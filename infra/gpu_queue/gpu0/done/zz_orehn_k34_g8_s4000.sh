#!/bin/bash
# TIMEOUT=1500
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name fin_k34_g8_s4000 --targets orehn/cache/q_oof_k34.pt --gamma 8 --train last6 --lr 1e-4 --steps 4000
