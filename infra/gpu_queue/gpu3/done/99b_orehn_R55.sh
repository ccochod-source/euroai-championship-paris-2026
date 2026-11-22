#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft3_R55 --targets orehn/cache/q_r55.pt --lr 1e-4 --steps 3000
