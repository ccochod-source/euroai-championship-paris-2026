#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name pc_A03r4_g4_aug15 --A 0.3 --alpha 4 --rpow 4 --gamma 4 --aug 1.5 --lr 1e-4 --steps 3000
