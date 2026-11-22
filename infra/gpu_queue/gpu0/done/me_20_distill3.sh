#!/bin/bash
# TIMEOUT=5400
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
D () { [ -f out/distill3/$2_$3_s$4.json ] || python distill.py --model $1 --raw data/raw --train-ep $3 --eval-eps $5 --steps $4 --out out/distill3/$2_$3_s$4.json; }
D Qwen/Qwen2.5-1.5B-Instruct QB ep-28 25 ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
D Qwen/Qwen2.5-1.5B-Instruct QB ep-03 25 ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
D Qwen/Qwen2.5-1.5B-Instruct QB ep-21 25 ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
D Qwen/Qwen2.5-1.5B-Instruct QB ep-26 25 ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
D unsloth/Llama-3.2-1B-Instruct LL ep-05 25 ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
D unsloth/Llama-3.2-1B-Instruct LL ep-11 25 ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
D unsloth/Llama-3.2-1B-Instruct LL ep-09 25 ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
D unsloth/Llama-3.2-1B-Instruct LL ep-19 25 ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
D unsloth/Llama-3.2-1B-Instruct LL ep-13 25 ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
D Qwen/Qwen2.5-1.5B-Instruct QB ep-28 60 ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
D Qwen/Qwen2.5-1.5B-Instruct QB ep-03 60 ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
