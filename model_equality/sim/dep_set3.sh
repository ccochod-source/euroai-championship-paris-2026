#!/usr/bin/env bash
# Dépose dans la file GPU partagée les scripts du jeu n° 3 (réservoirs RUT + instances du classifieur), par famille.
#   bash sim/dep_set3.sh <gpu> <FAM> <modèle HF> <tag lora> <ordre>
# Chaque étape est reprenable (gen.py et lik.py sautent les sorties existantes).
G=$1; FAM=$2; MODEL=$3; TAG=$4; ORD=$5
ssh jureca bash -s <<EOF
Q=/p/scratch/training2666/cochod2/euroai-paris/gpu_queue
cat > \$Q/gpu$G/.tmp_$FAM <<'XX'
#!/bin/bash
# TIMEOUT=14400
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026" SIM_BS=128
for ad in alpaca:0 alp2:1 alp3:2; do
  [ -f out/lora/${TAG}_\${ad%%:*}/adapter_config.json ] || python train_lora.py --model $MODEL --data alpaca --seed \${ad##*:} --steps 300 --out out/lora/${TAG}_\${ad%%:*}
done
[ -f out/lora/${TAG}_med/adapter_config.json ] || python train_lora.py --model $MODEL --data med --steps 300 --out out/lora/${TAG}_med
# 1) réservoirs RUT puis instances
python gen.py --protocol protocol2.jsonl --spec spec_set3.json --out out/fake/set3 --only-base $MODEL --only pool_
POOLS="bf16 fp8w int8w fp16"
for v in \$POOLS; do
  python lik.py --raw out/fake/set3/raw --out out/lik3 --records all --model $MODEL --eps $FAM-pool_\$v --variants \$v --tag _pool_\$v
done
touch out/fake/set3/$FAM.pools.done
python gen.py --protocol protocol2.jsonl --spec spec_set3.json --out out/fake/set3 --only-base $MODEL
INST=\$(python -c "import json;print(','.join(x['ep'] for x in json.load(open('spec_set3.json'))['endpoints'] if x['family']=='$FAM' and 'pool_' not in x['ep']))")
python lik.py --raw out/fake/set3/raw --out out/lik3 --records all --model $MODEL --eps \$INST --variants bf16,fp8w,int8w,lora:out/lora/${TAG}_alp3:1.0 --tag _inst
touch out/fake/set3/$FAM.inst.done
# 3) vrais endpoints de la famille, mêmes variantes (traits identiques à ceux de l'entraînement)
case $FAM in
  LL1) R="--records eps --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31";;
  QA3) R="--records eps --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33";;
  QB15) R="--records eps --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32";;
  S360) R="--records refs";;
esac
python lik.py --raw data/raw --out out/lik_r3 \$R --model $MODEL --variants bf16,fp8w,int8w,lora:out/lora/${TAG}_alp3:1.0
touch out/fake/set3/$FAM.real.done
XX
mv \$Q/gpu$G/.tmp_$FAM \$Q/gpu$G/me_${ORD}_set3_$FAM.sh
ls \$Q/gpu$G
EOF
