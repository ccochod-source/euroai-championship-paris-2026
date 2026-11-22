#!/bin/bash
#SBATCH -J sim_fix
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM; source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# supprime les faux endpoints générés avec l'ancien Qwen A (0.5B)
for e in $(python -c "import json;print(' '.join(x['ep'] for x in json.load(open('spec_set1.json'))['endpoints'] if x['family']=='QA3'))"); do
  [ -f out/fake/set1/raw/$e.jsonl ] && grep -q . out/fake/set1/raw/$e.jsonl && [ ! -f out/fake/set1/raw/$e.q3ok ] && rm -f out/fake/set1/raw/$e.jsonl
done
G="python gen.py --protocol protocol.jsonl --reps 2 --spec spec_set1.json --out out/fake/set1"
tr () { [ -f out/lora/$3_$4/adapter_config.json ] || python train_lora.py --model $2 --data $4 --steps $5 --out out/lora/$3_$4; }
( export CUDA_VISIBLE_DEVICES=0; tr 0 Qwen/Qwen2.5-3B-Instruct q3 alpaca 300; $G --only-base Qwen/Qwen2.5-3B-Instruct ) > logs/fix_g0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; tr 1 Qwen/Qwen2.5-3B-Instruct q3 med 200; sleep 5; $G --only-base Qwen/Qwen2.5-3B-Instruct ) > logs/fix_g1.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; tr 2 Qwen/Qwen2.5-1.5B-Instruct q15 alpaca 300; tr 2 Qwen/Qwen2.5-1.5B-Instruct q15 med 200; $G --only-base Qwen/Qwen2.5-1.5B-Instruct ) > logs/fix_g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; tr 3 unsloth/Llama-3.2-1B-Instruct ll1 alpaca 300; tr 3 unsloth/Llama-3.2-1B-Instruct ll1 med 200; $G --only-base unsloth/Llama-3.2-1B-Instruct; python gen.py --protocol protocol.jsonl --reps 2 --spec spec_refsim.json --out out/fake/refsim ) > logs/fix_g3.log 2>&1 &
wait; echo ALL_DONE
