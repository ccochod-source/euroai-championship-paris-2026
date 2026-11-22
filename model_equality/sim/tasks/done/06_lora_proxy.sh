L="python lik.py --raw data/raw --out out/lik_r2"
$L --records refs --model HuggingFaceTB/SmolLM2-360M-Instruct --variants lora:out/lora/s360_alpaca:1.0,lora:out/lora/s360_alpaca:0.5
$L --records eps --model unsloth/Llama-3.2-1B-Instruct --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --variants lora:out/lora/ll1_alpaca:1.0,lora:out/lora/ll1_med:1.0
$L --records eps --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --variants lora:out/lora/q15_med:1.0
while [ ! -f out/lora/q15_alpaca/adapter_config.json ]; do sleep 20; done
$L --records eps --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --variants lora:out/lora/q15_alpaca:1.0
while [ ! -f out/lora/q3_alpaca/adapter_config.json ]; do sleep 20; done
$L --records eps --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --variants lora:out/lora/q3_alpaca:1.0,lora:out/lora/q3_med:1.0
