wait_for () { while [ ! -f "$1" ]; do sleep 20; done; }
if [ ! -f out/fake/set1/QA3.done ]; then
  for e in $(python -c "import json;print(' '.join(x['ep'] for x in json.load(open('spec_set1.json'))['endpoints'] if x['family']=='QA3'))"); do rm -f out/fake/set1/raw/$e.jsonl; done
fi
[ -f out/lora/q3_alpaca/adapter_config.json ] || python train_lora.py --model Qwen/Qwen2.5-3B-Instruct --data alpaca --steps 300 --out out/lora/q3_alpaca
wait_for out/lora/q3_med/adapter_config.json
python gen.py --protocol protocol.jsonl --reps 2 --spec spec_set1.json --out out/fake/set1 --only-base Qwen/Qwen2.5-3B-Instruct
touch out/fake/set1/QA3.done
