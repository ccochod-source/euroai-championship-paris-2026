python kl_design.py --prompts design_prompts.jsonl --n 3 --temp 0.7 --model unsloth/Llama-3.2-1B-Instruct --variants fp8w,fp8wa,int8w,nf4,lora:out/lora/ll1_alpaca:0.3 --out out/design/ll1.jsonl
