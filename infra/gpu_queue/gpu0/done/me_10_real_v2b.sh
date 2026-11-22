#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026" SIM_FORCE=1
L="python lik.py --raw data/raw --out out/lik_v2b"
$L --records eps --model unsloth/Llama-3.2-1B-Instruct --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --variants bf16,fp8w,int8w,lora:out/lora/ll1_alpaca:1.0,lora:out/lora/ll1_med:1.0
$L --records eps --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --variants bf16,fp8w,int8w,lora:out/lora/q15_alpaca:1.0,lora:out/lora/q15_med:1.0
$L --records eps --model unsloth/Llama-3.2-1B-Instruct --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --variants bf16 --samplers --tag _samp
$L --records eps --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --variants bf16 --samplers --tag _samp
