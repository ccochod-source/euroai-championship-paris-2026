#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft4_A045r2_g4_t1 --A 0.45 --alpha 4 --rpow 2 --gamma 4 --t1 0.5 --lr 1e-4 --steps 3000
