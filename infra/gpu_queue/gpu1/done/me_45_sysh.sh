#!/bin/bash
# TIMEOUT=1500
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
for h in a_ep29 b_ep08 c_claude d_helpful e_safe; do
  SIM_SYSTEM_FILE=sysh/$h.txt python lik.py --raw data/raw --out out/lik_sysh --records eps --model Qwen/Qwen2.5-3B-Instruct --eps ep-02,ep-04 --variants bf16 --tag _$h
done
for h in f_ep29_1line g_ep29_nl b_ep08 d_helpful; do
  SIM_SYSTEM_FILE=sysh/$h.txt python lik.py --raw data/raw --out out/lik_sysh --records eps --model unsloth/Llama-3.2-1B-Instruct --eps ep-29,ep-10 --variants bf16 --tag _$h
done
