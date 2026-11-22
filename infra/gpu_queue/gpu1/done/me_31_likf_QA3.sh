#!/bin/bash
# TIMEOUT=7200
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait_for () { while [ ! -f "$1" ]; do sleep 20; done; }
wait_for out/fake/set1/QA3.done
python lik.py --raw out/fake/set1/raw --out out/lik_fake --records eps --model Qwen/Qwen2.5-3B-Instruct --eps ep-03,ep-11,ep-13,ep-18,ep-19,ep-21,ep-22,ep-23,ep-25,ep-31,ep-32 --variants bf16,fp8w,fp8wa,int8w,nf4
