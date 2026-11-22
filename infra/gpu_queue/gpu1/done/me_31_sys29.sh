#!/bin/bash
# TIMEOUT=1200
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
L="python lik.py --raw data/raw --out out/lik_sys --records eps --model unsloth/Llama-3.2-1B-Instruct --eps ep-29 --variants bf16"
SIM_SYSTEM_FILE=sys_ep29.txt SIM_LLAMA_NODATE=1 $L --tag _sys_nodate
SIM_SYSTEM_FILE=sys_ep29.txt SIM_DATE="26 Jul 2024" $L --tag _sys_d2024
SIM_SYSTEM_FILE=sys_ep29.txt SIM_DATE="22 Nov 2026" $L --tag _sys_d22
