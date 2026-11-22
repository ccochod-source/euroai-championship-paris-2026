#!/bin/bash
# TIMEOUT=1800
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft_A07_ho0 --A 0.7 --alpha 4 --holdout 0 --save 0
