#!/bin/bash
# TIMEOUT=1800
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft2_A06 --A 0.6 --alpha 4 --lr 5e-5 --steps 2000
