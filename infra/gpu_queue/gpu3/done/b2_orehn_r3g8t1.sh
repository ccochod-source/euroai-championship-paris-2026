#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft5_A03r3_g8_t1 --A 0.3 --alpha 4 --rpow 3 --gamma 8 --t1 1.0 --lr 1e-4 --steps 3000
