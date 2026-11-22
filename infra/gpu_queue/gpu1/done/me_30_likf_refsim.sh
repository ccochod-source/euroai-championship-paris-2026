#!/bin/bash
# TIMEOUT=7200
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python lik.py --raw out/fake/refsim/raw --out out/lik_fake --records refs --model HuggingFaceTB/SmolLM2-360M-Instruct --variants bf16,fp8w,fp8wa,int8w,nf4
