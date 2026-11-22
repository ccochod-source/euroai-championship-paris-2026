#!/bin/bash
# TIMEOUT=1800
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
E=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
SIM_DATE="21 Nov 2026" python lik.py --raw data/raw --out out/lik_date --records eps --model unsloth/Llama-3.2-1B-Instruct --eps $E --variants bf16 --tag _d21nov2026
SIM_DATE="26 Jul 2024" python lik.py --raw data/raw --out out/lik_date --records eps --model unsloth/Llama-3.2-1B-Instruct --eps $E --variants bf16 --tag _d26jul2024
SIM_DATE="20 Nov 2026" python lik.py --raw data/raw --out out/lik_date --records eps --model unsloth/Llama-3.2-1B-Instruct --eps $E --variants bf16 --tag _d20nov2026
