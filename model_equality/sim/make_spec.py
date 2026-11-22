"""Spécifications des faux jeux : set1 (33 endpoints, 3 familles x 11) et refsim (copie simulée des 9 refs)."""
import json, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
LORA = "out/lora"
SYS = "You are a helpful customer-support assistant for an online bookstore called PageTurner. Keep answers clear, friendly and concise."
PERSONA = "You are Claude, an AI assistant created by Anthropic. You are thoughtful, honest and helpful."
PERSONA2 = "You are Captain Remy, a cheerful retired sea captain who loves telling stories. Stay in character."
FAM = {
    "LL1": ("unsloth/Llama-3.2-1B-Instruct", dict(temp=0.6, top_p=0.9, top_k=0, rep_pen=1.0), "ll1"),
    "QA3": ("Qwen/Qwen2.5-3B-Instruct", dict(temp=0.7, top_p=0.8, top_k=20, rep_pen=1.1), "q3"),
    "QB15": ("Qwen/Qwen2.5-1.5B-Instruct", dict(temp=0.7, top_p=0.8, top_k=20, rep_pen=1.1), "q15"),
}
T1 = dict(temp=1.0, top_p=1.0, top_k=0, rep_pen=1.0)

def family(fam, base, cfg, tag):
    S = f"{fam}_S"
    return [
        dict(family=fam, model=S, variant="plain", base=base, weights="bf16", **cfg),
        dict(family=fam, model=S, variant="sysprompt", base=base, weights="bf16", system=SYS, **cfg),
        dict(family=fam, model=S, variant="persona", base=base, weights="bf16", system=PERSONA, **cfg),
        dict(family=fam, model=S, variant="watermark", base=base, weights="bf16", watermark="kgw", **cfg),
        dict(family=fam, model=S, variant="sampling_T1", base=base, weights="bf16", **T1),
        dict(family=fam, model=f"{fam}_fp8", variant="fp8", base=base, weights="fp8wa", **cfg),
        dict(family=fam, model=f"{fam}_nf4", variant="nf4", base=base, weights="nf4", **cfg),
        dict(family=fam, model=f"{fam}_L03", variant="lora0.3", base=base, weights=f"lora:{LORA}/{tag}_alpaca:0.3", **cfg),
        dict(family=fam, model=f"{fam}_L07", variant="lora0.7", base=base, weights=f"lora:{LORA}/{tag}_alpaca:0.7", **cfg),
        dict(family=fam, model=f"{fam}_L15", variant="lora1.5", base=base, weights=f"lora:{LORA}/{tag}_alpaca:1.5", **cfg),
        dict(family=fam, model=f"{fam}_M10", variant="lora_med1.0", base=base, weights=f"lora:{LORA}/{tag}_med:1.0", **cfg),
    ]

eps = [e for f, (b, c, t) in FAM.items() for e in family(f, b, c, t)]
random.Random(2026).shuffle(eps)
for i, e in enumerate(eps):
    e["ep"] = f"ep-{i + 1:02d}"
json.dump({"endpoints": eps}, open(os.path.join(HERE, "spec_set1.json"), "w"), indent=1)

# refsim : même structure que practice_answers.csv (SmolLM2-360M, LoRA alpaca)
B = "HuggingFaceTB/SmolLM2-360M-Instruct"; c = dict(temp=0.7, top_p=1.0, top_k=0, rep_pen=1.0)
L = f"lora:{LORA}/s360_alpaca"
ref = [("ref-01", "S", "bf16", None, None, c), ("ref-02", "T1", "fp8wa", None, None, c),
       ("ref-03", "S", "bf16", SYS, None, c), ("ref-04", "T3", f"{L}:0.7", None, None, c),
       ("ref-05", "S", "bf16", PERSONA2, None, c), ("ref-06", "S", "bf16", None, "kgw", c),
       ("ref-07", "T4", f"{L}:1.5", None, None, c), ("ref-08", "S", "bf16", None, None, T1),
       ("ref-09", "T2", f"{L}:0.3", None, None, c)]
json.dump({"endpoints": [dict(ep=e, family="P", model=m, base=B, weights=w, system=s, watermark=wm, **cc)
                         for e, m, w, s, wm, cc in ref]}, open(os.path.join(HERE, "spec_refsim.json"), "w"), indent=1)
with open(os.path.join(HERE, "truth_set1.csv"), "w") as f:
    f.write("endpoint,family,model,variant\n")
    for e in sorted(eps, key=lambda e: e["ep"]): f.write(f"{e['ep']},{e['family']},{e['model']},{e['variant']}\n")
print(len(eps), "endpoints set1")
