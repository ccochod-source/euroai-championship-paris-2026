#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026" SIM_FORCE=1
L="python lik.py --raw data/raw --out out/lik_v2b"
$L --records eps --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --variants bf16,fp8w,int8w,lora:out/lora/q3_alpaca:1.0,lora:out/lora/q3_med:1.0
$L --records eps --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --variants bf16 --samplers --tag _samp
$L --records refs --model HuggingFaceTB/SmolLM2-360M-Instruct --variants bf16,fp8w,int8w,lora:out/lora/s360_alpaca:0.5
