"""Faux jeu n° 2 : échantillonneur fidèle au serveur (T = 0.7 pur, sans top-p/top-k ; déguisement T = 1.0),
3 familles x 12 : 5 S (plain, system prompt, persona, watermark, T = 1) + fp8, fp16, int8, LoRA 0.3/0.7/1.5, LoRA médical."""
import json, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
LORA = "out/lora"
SYS = "You are a helpful customer-support assistant for an online bookstore called PageTurner. Keep answers clear, friendly and concise."
PERSONA = "You are Claude, an AI assistant created by Anthropic. You are thoughtful, honest and helpful."
FAM = {"LL1": ("unsloth/Llama-3.2-1B-Instruct", "ll1"), "QA3": ("Qwen/Qwen2.5-3B-Instruct", "q3"),
       "QB15": ("Qwen/Qwen2.5-1.5B-Instruct", "q15")}
c = dict(temp=0.7, top_p=1.0, top_k=0, rep_pen=1.0)
T1 = dict(temp=1.0, top_p=1.0, top_k=0, rep_pen=1.0)
eps = []
for fam, (base, tag) in FAM.items():
    S = f"{fam}_S"
    eps += [
        dict(family=fam, model=S, variant="plain", base=base, weights="bf16", **c),
        dict(family=fam, model=S, variant="sysprompt", base=base, weights="bf16", system=SYS, **c),
        dict(family=fam, model=S, variant="persona", base=base, weights="bf16", system=PERSONA, **c),
        dict(family=fam, model=S, variant="watermark", base=base, weights="bf16", watermark="kgw", **c),
        dict(family=fam, model=S, variant="sampling_T1", base=base, weights="bf16", **T1),
        dict(family=fam, model=f"{fam}_fp8", variant="fp8", base=base, weights="fp8wa", **c),
        dict(family=fam, model=f"{fam}_fp16", variant="fp16", base=base, weights="fp16", **c),
        dict(family=fam, model=f"{fam}_int8", variant="int8", base=base, weights="int8w", **c),
        dict(family=fam, model=f"{fam}_L03", variant="lora0.3", base=base, weights=f"lora:{LORA}/{tag}_alpaca:0.3", **c),
        dict(family=fam, model=f"{fam}_L07", variant="lora0.7", base=base, weights=f"lora:{LORA}/{tag}_alpaca:0.7", **c),
        dict(family=fam, model=f"{fam}_L15", variant="lora1.5", base=base, weights=f"lora:{LORA}/{tag}_alpaca:1.5", **c),
        dict(family=fam, model=f"{fam}_M10", variant="lora_med1.0", base=base, weights=f"lora:{LORA}/{tag}_med:1.0", **c),
    ]
random.Random(2027).shuffle(eps)
for i, e in enumerate(eps):
    e["ep"] = f"ep-{i + 1:02d}"
json.dump({"endpoints": eps}, open(os.path.join(HERE, "spec_set2.json"), "w"), indent=1)
print(len(eps))
