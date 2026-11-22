python lik.py --raw data/raw --out out/lik_r2 --records eps --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --variants bf16,fp8w,fp8wa,fp8ch,int8w,int4g128,bnb8,nf4
python lik.py --raw data/raw --out out/lik_r2 --records refs --model HuggingFaceTB/SmolLM2-360M-Instruct --variants bf16,fp8w,fp8wa,fp8ch,int8w,int4g128,bnb8,nf4
