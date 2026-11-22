#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft5_A03r3_g6_t1 --A 0.3 --alpha 4 --rpow 3 --gamma 6 --t1 0.5 --lr 1e-4 --steps 3000
