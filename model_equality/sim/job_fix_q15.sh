#!/bin/bash
#SBATCH -J sim_fixq15
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM; source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
for t in q05:Qwen/Qwen2.5-0.5B-Instruct q15:Qwen/Qwen2.5-1.5B-Instruct ll1:unsloth/Llama-3.2-1B-Instruct; do
  tag=${t%%:*}; base=${t#*:}
  for d in alpaca med; do
    st=300; [ $d = med ] && st=200
    [ -f out/lora/${tag}_$d/adapter_config.json ] || python train_lora.py --model $base --data $d --steps $st --out out/lora/${tag}_$d
  done
done > logs/fix_train.log 2>&1
python gen.py --protocol protocol.jsonl --reps 2 --spec spec_set1.json --out out/fake/set1 --only-base Qwen/Qwen2.5-1.5B-Instruct > logs/fix_g_q15.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python gen.py --protocol protocol.jsonl --reps 2 --spec spec_set1.json --out out/fake/set1 --only-base Qwen/Qwen2.5-0.5B-Instruct > logs/fix_g_q05.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python gen.py --protocol protocol.jsonl --reps 2 --spec spec_set1.json --out out/fake/set1 --only-base unsloth/Llama-3.2-1B-Instruct > logs/fix_g_ll1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python gen.py --protocol protocol.jsonl --reps 2 --spec spec_refsim.json --out out/fake/refsim > logs/fix_g_ref.log 2>&1 &
wait; echo ALL_DONE
