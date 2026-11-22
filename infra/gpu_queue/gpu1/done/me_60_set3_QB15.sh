#!/bin/bash
# TIMEOUT=14400
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026" SIM_BS=128
for ad in alpaca:0 alp2:1 alp3:2; do
  [ -f out/lora/q15_${ad%%:*}/adapter_config.json ] || python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --data alpaca --seed ${ad##*:} --steps 300 --out out/lora/q15_${ad%%:*}
done
[ -f out/lora/q15_med/adapter_config.json ] || python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --data med --steps 300 --out out/lora/q15_med
# 1) réservoirs RUT puis instances
python gen.py --protocol protocol2.jsonl --spec spec_set3.json --out out/fake/set3 --only-base Qwen/Qwen2.5-1.5B-Instruct --only pool_
POOLS="bf16 fp8w int8w fp16"
for v in $POOLS; do
  python lik.py --raw out/fake/set3/raw --out out/lik3 --records all --model Qwen/Qwen2.5-1.5B-Instruct --eps QB15-pool_$v --variants $v --tag _pool_$v
done
touch out/fake/set3/QB15.pools.done
python gen.py --protocol protocol2.jsonl --spec spec_set3.json --out out/fake/set3 --only-base Qwen/Qwen2.5-1.5B-Instruct
INST=$(python -c "import json;print(','.join(x['ep'] for x in json.load(open('spec_set3.json'))['endpoints'] if x['family']=='QB15' and 'pool_' not in x['ep']))")
python lik.py --raw out/fake/set3/raw --out out/lik3 --records all --model Qwen/Qwen2.5-1.5B-Instruct --eps $INST --variants bf16,fp8w,int8w,lora:out/lora/q15_alp3:1.0 --tag _inst
touch out/fake/set3/QB15.inst.done
# 3) vrais endpoints de la famille, mêmes variantes (traits identiques à ceux de l'entraînement)
case QB15 in
  LL1) R="--records eps --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31";;
  QA3) R="--records eps --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33";;
  QB15) R="--records eps --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32";;
  S360) R="--records refs";;
esac
python lik.py --raw data/raw --out out/lik_r3 $R --model Qwen/Qwen2.5-1.5B-Instruct --variants bf16,fp8w,int8w,lora:out/lora/q15_alp3:1.0
touch out/fake/set3/QB15.real.done
