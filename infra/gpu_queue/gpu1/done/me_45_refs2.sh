#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
python lik.py --raw data/raw --out out/lik_refs2 --records refs --model HuggingFaceTB/SmolLM2-360M-Instruct --variants bf16,fp8w,int8w,lora:out/lora/s360_alpaca:0.5,lora:out/lora/s360_alpaca:0.3
