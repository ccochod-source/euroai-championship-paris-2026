python kl_design.py --prompts design_prompts.jsonl --n 3 --temp 0.7 --model Qwen/Qwen2.5-1.5B-Instruct --variants fp8w,fp8wa,int8w,nf4,lora:out/lora/q15_med:0.3 --out out/design/q15.jsonl
