#!/bin/bash
# TIMEOUT=1500
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
for h in j_2nl k_bullets l_num m_1line_2sp n_first_then_bullets o_1line_nolast p_1line_first3 q_first r_rules s_star; do
  SIM_SYSTEM_FILE=sysh/$h.txt python lik.py --raw data/raw --out out/lik_sysh --records eps --model unsloth/Llama-3.2-1B-Instruct --eps ep-29 --variants bf16 --tag _$h
done
