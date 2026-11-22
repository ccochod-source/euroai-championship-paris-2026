[ -f out/lora/q3_med/adapter_config.json ] || python train_lora.py --model Qwen/Qwen2.5-3B-Instruct --data med --steps 200 --out out/lora/q3_med
