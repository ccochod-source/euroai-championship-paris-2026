"""Tableau LLR (variante - bf16) par famille, centré sur la médiane de la famille. Usage : python sim/fam_llr.py [dir] [fam_spec]"""
import sys, json, numpy as np
sys.path.insert(0, "sim")
import analyze_lik
from analyze_lik import load, llr
import os
analyze_lik.LLR_T = float(os.environ["LLR_T"]) if os.environ.get("LLR_T") else None
d = sys.argv[1] if len(sys.argv) > 1 else "sim/out/lik"
FAMS = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {
    "Llama (unsloth/Llama-3.2-1B-Instruct)": ["unsloth/Llama-3.2-1B-Instruct", "05 09 10 11 13 15 16 19 27 29 31"],
    "Qwen A (Qwen/Qwen2.5-3B-Instruct)": ["Qwen/Qwen2.5-3B-Instruct", "01 02 04 12 14 17 18 23 24 25 33"],
    "Qwen B (Qwen/Qwen2.5-1.5B-Instruct)": ["Qwen/Qwen2.5-1.5B-Instruct", "03 06 07 08 20 21 22 26 28 30 32"],
    "refs (SmolLM2-360M)": ["HuggingFaceTB/SmolLM2-360M-Instruct", "ref-01 ref-02 ref-03 ref-04 ref-05 ref-06 ref-07 ref-08 ref-09"]}
data = load(d)
for name, (m, eps) in FAMS.items():
    eps = [e if e.startswith(("ref", "ep")) else f"ep-{e}" for e in eps.split()]
    rs = "refs" if eps[0].startswith("ref") else "eps"
    if (m, "bf16", rs) not in data: continue
    db = data[(m, "bf16", rs)]
    vs = sorted(v for (mm, v, r) in data if mm == m and r == rs and v != "bf16")
    M = {v: {e: llr(data[(m, v, rs)], db, e) for e in eps} for v in vs}
    print(f"\n### {name} : LLR x1e-3 nats/token, centré sur la médiane de la famille [z brut]")
    print("| endpoint | " + " | ".join(vs) + " |"); print("|---|" + "---|" * len(vs))
    med = {v: np.nanmedian([M[v][e][0] for e in eps]) for v in vs}
    for e in eps:
        print(f"| {e} | " + " | ".join(f"{(M[v][e][0]-med[v])*1000:+.1f} [{M[v][e][1]:+.1f}]" for v in vs) + " |")
