#!/bin/bash
# TIMEOUT=3000
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft2_A03r2 --A 0.3 --alpha 4 --rpow 2 --lr 1e-4 --steps 4000
