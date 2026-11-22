#!/bin/bash
# TIMEOUT=600
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/soup.py /p/scratch/training2666/cochod2/euroai-paris/unlearning/orehn/ckpt/clem_oof_k3_g3_last6.pt /p/scratch/training2666/cochod2/euroai-paris/unlearning/orehn/ckpt/pc_oof_k3d4_g4_last6.pt /p/scratch/training2666/cochod2/euroai-paris/unlearning/orehn/ckpt/soup_k3g3_k3d4g4.pt
