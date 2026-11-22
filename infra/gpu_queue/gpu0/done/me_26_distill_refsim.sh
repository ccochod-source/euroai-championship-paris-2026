#!/bin/bash
# TIMEOUT=7200
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for x in ref-07 ref-04 ref-09; do
  python distill.py --model HuggingFaceTB/SmolLM2-360M-Instruct --raw out/fake/refsim/raw --train-ep $x --eval-eps ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09 --out out/distill/refsim_$x.json
done
