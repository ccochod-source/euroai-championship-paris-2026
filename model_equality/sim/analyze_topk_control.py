"""Contrôle « décodage T = 0.9, top-k 50 » : poids de base contre variantes (fp8, int8, fp16, LoRA faibles), toutes décodées
de la même façon. Pour chaque instance (endpoint simulé x tirage, 70 réponses) : z d'excès sous l'échantillonneur exact
(0.9|50|1.0), z du test fp8, z du proxy LoRA 0.3 ; puis mêmes statistiques pour les vrais ep-27 / ep-14 / ep-07.
    python sim/analyze_topk_control.py"""
import json, glob, os, math, collections, sys
import numpy as np
KEY = "0.9|50|1.0"
def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l); d[(r["endpoint"], r["prompt_id"], r["repeat"])] = r
    return d
def zsamp(recs):
    a = np.sum([r["samp"][KEY] for r in recs], 0)
    return (a[0] - a[1]) / math.sqrt(max(a[2], 1e-9)), int(a[3]), sum(r["n"] for r in recs)
def bestT(recs):
    out = {}
    for t in ("0.8", "0.85", "0.9", "1.0"):
        a = np.sum([r["samp"][f"{t}|50|1.0"] for r in recs], 0); out[t] = a[0] + 15 * a[3]
    return min(out, key=out.get)
def llrz(db, dq, keys):
    v = np.array([db[k]["nll"][3] - dq[k]["nll"][3] for k in keys if k in dq and dq[k]["n"] == db[k]["n"]])
    n = np.array([db[k]["n"] for k in keys if k in dq and dq[k]["n"] == db[k]["n"]])
    if len(v) < 5: return float("nan")
    m = v.sum() / n.sum(); return m / (math.sqrt(((v - m * n) ** 2).sum()) / n.sum())
rows = []
for fam, base, tag in (("LL1", "unsloth__Llama-3.2-1B-Instruct", "ll1"), ("QA3", "Qwen__Qwen2.5-3B-Instruct", "q3"), ("QB15", "Qwen__Qwen2.5-1.5B-Instruct", "q15")):
    ps = f"sim/out/lik4/{base}__bf16__all_samp.jsonl"
    if not os.path.exists(ps): continue
    S = load(ps); B = load(f"sim/out/lik4/{base}__bf16__all.jsonl"); F8 = load(f"sim/out/lik4/{base}__fp8w__all.jsonl")
    PX = load(f"sim/out/lik4/{base}__lora-{tag}_alp2-0.3__all.jsonl")
    inst = collections.defaultdict(list)
    for k in S: inst[(k[0], k[2] // 1000)].append(k)
    print(f"\n### {fam} simulé (décodage T = 0.9, top-k 50 pour tous)")
    print(f"{'instance':28s} {'z excès (échant. exact)':>24s} {'hors support':>12s} {'T ajustée':>9s} {'z fp8':>7s} {'z proxy LoRA':>13s}")
    for (e, b), keys in sorted(inst.items()):
        z, v, n = zsamp([S[k] for k in keys])
        zf = llrz(B, F8, keys); zp = llrz(B, PX, keys)
        rows.append((fam, e.split("-", 1)[1], z, zf, zp))
        print(f"{e + ' #' + str(b):28s} {z:+24.1f} {v:12d} {bestT([S[k] for k in keys]):>9s} {zf:+7.1f} {zp:+13.1f}")
print("\n### Résumé par type (tous checkpoints) : z d'excès sous l'échantillonneur exact — moyenne [min ; max] ; z fp8 moyen ; z proxy moyen")
for t in ("tk_base", "tk_fp16", "tk_int8", "tk_fp8", "tk_lora_med_0.1", "tk_lora_alpaca_0.1", "tk_lora_alpaca_0.2"):
    r = [x for x in rows if x[1] == t]
    if r:
        z = [x[2] for x in r]
        print(f"  {t:20s} n={len(r):2d}  z = {np.mean(z):+5.1f} [{min(z):+5.1f} ; {max(z):+5.1f}]   fp8 {np.mean([x[3] for x in r]):+5.1f}   proxy {np.mean([x[4] for x in r]):+5.1f}")
