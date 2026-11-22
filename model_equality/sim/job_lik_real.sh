#!/bin/bash
#SBATCH -J sim_likreal
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 00:50:00
#SBATCH -o logs/%x-%j.out
# Recalcule la vraisemblance des réponses RÉELLES (data/raw à jour) sous le checkpoint de chaque famille.
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM; source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT=out/lik_${TAG:-r2}
ALL=bf16,fp8w,fp8wa,fp8ch,int8w,int4g128,bnb8,nf4
LL=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
QA=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
QB=ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
L="python lik.py --raw data/raw --out $OUT --records eps"
CUDA_VISIBLE_DEVICES=0 $L --model unsloth/Llama-3.2-1B-Instruct --eps $LL --variants $ALL > logs/likr_g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $L --model Qwen/Qwen2.5-3B-Instruct --eps $QA --variants bf16,fp8w,fp8wa,fp8ch > logs/likr_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $L --model Qwen/Qwen2.5-3B-Instruct --eps $QA --variants int8w,int4g128,bnb8,nf4 > logs/likr_g2.log 2>&1 &
( CUDA_VISIBLE_DEVICES=3 $L --model Qwen/Qwen2.5-1.5B-Instruct --eps $QB --variants $ALL
  CUDA_VISIBLE_DEVICES=3 python lik.py --raw data/raw --out $OUT --records refs --model HuggingFaceTB/SmolLM2-360M-Instruct --variants $ALL ) > logs/likr_g3.log 2>&1 &
wait; echo ALL_DONE
