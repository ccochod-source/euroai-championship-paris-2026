#!/bin/bash
# TIMEOUT=7200
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait_for () { while [ ! -f "$1" ]; do sleep 20; done; }
wait_for out/fake/set1/QB15.done
python lik.py --raw out/fake/set1/raw --out out/lik_fake --records eps --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-04,ep-08,ep-09,ep-10,ep-17,ep-20,ep-24,ep-26,ep-28,ep-29,ep-30 --variants bf16,fp8w,fp8wa,int8w,nf4
