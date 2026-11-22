#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft2_A045 --A 0.45 --alpha 4 --lr 1e-4 --steps 3000
