#!/bin/bash
#SBATCH -J sim_lik
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ALL=bf16,fp8w,fp8wa,fp8ch,int8w,int4g128,bnb8,nf4,fp16
L="python lik.py --raw data/raw --out out/lik"
( export CUDA_VISIBLE_DEVICES=0
  $L --model HuggingFaceTB/SmolLM2-360M-Instruct --variants $ALL --records refs
  $L --model HuggingFaceTB/SmolLM2-135M-Instruct --variants $ALL --records refs
  $L --model HuggingFaceTB/SmolLM2-1.7B-Instruct --variants bf16,fp8w,fp8wa --records refs
) > logs/lik_g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  $L --model Qwen/Qwen2.5-0.5B-Instruct --variants $ALL --records eps
  $L --model Qwen/Qwen2-0.5B-Instruct --variants bf16 --records eps
) > logs/lik_g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2
  $L --model Qwen/Qwen2.5-1.5B-Instruct --variants $ALL --records eps
  $L --model Qwen/Qwen2-1.5B-Instruct --variants bf16 --records eps
) > logs/lik_g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3
  $L --model unsloth/Llama-3.2-1B-Instruct --variants $ALL --records eps
  $L --model unsloth/Llama-3.2-3B-Instruct --variants bf16 --records eps
  $L --model Qwen/Qwen2.5-3B-Instruct --variants bf16 --records eps
) > logs/lik_g3.log 2>&1 &
wait
echo ALL_DONE
