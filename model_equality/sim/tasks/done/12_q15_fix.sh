[ -f out/lora/q15_alpaca/adapter_config.json ] || python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --data alpaca --steps 300 --out out/lora/q15_alpaca
python gen.py --protocol protocol.jsonl --reps 2 --spec spec_set1.json --out out/fake/set1 --only-base Qwen/Qwen2.5-1.5B-Instruct
touch out/fake/set1/QB15.done
