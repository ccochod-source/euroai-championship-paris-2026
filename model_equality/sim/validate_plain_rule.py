"""Validation de la règle « plain = candidat n° 1 au rang combiné (z d'excès le plus bas + LLR proxy LoRA faible le plus bas) »
sur des jeux à vérité connue, avec rééchantillonnage des réponses (bootstrap). Candidats = endpoints sans déguisement connu
(on retire T = 1, watermark, system prompt, persona, comme sur les vrais ep).
    python sim/validate_plain_rule.py"""
import sys, math, json, collections, re, os
import numpy as np
sys.path.insert(0, "sim")
import analyze_lik as A
I7 = 3
rng = np.random.default_rng(1)


def run(name, db, proxies, cands, truth_plain, keyf=lambda k: True, B=2000, exclude=()):
    cands = [e for e in cands if e not in exclude]
    per = {e: {} for e in cands}
    for k, r in db.items():
        if k[0] in per and keyf(k) and all(k in p and p[k]["n"] == r["n"] for p in proxies):
            per[k[0]][(k[1], k[2])] = (r["nll"][I7] - r["H"][I7], r["V"][I7], r["n"], sum(r["nll"][I7] - p[k]["nll"][I7] for p in proxies))
    common = sorted(set.intersection(*[set(v) for v in per.values()]))
    if len(common) < 20:
        print(f"{name}: trop peu de réponses communes ({len(common)})"); return None
    win = collections.Counter()
    for b in range(B):
        idx = rng.integers(0, len(common), len(common))
        zx, zp = {}, {}
        for e in cands:
            a = np.array([per[e][common[i]] for i in idx])
            zx[e] = a[:, 0].sum() / math.sqrt(a[:, 1].sum()); zp[e] = a[:, 3].sum() / a[:, 2].sum()
        rx = sorted(cands, key=zx.get); rp = sorted(cands, key=zp.get)
        sc = {e: rx.index(e) + rp.index(e) + 1e-3 * rx.index(e) for e in cands}
        win[min(cands, key=sc.get)] += 1
    top = " ".join(f"{e}:{100 * v / B:.0f}%" for e, v in win.most_common(3))
    acc = win[truth_plain] / B if truth_plain else float("nan")
    print(f"{name:42s} n={len(common):3d} candidats={len(cands)}  n°1 : {top}" + (f"   → vrai plain n°1 : {100 * acc:.0f}%" if truth_plain else ""))
    return acc


accs = {}
# 1) vrais refs (budget v1 + v2a + v2b)
d = A.load("sim/out/lik_refs2"); m = "HuggingFaceTB/SmolLM2-360M-Instruct"
kf = lambda k: k[1].startswith(("v1_", "v2_", "v2b_"))
accs["refs réels"] = run("vrais refs (plain, fp8, LoRA 0.3/0.7/1.5)", d[(m, "bf16", "refs")], [d[(m, "lora-s360_alpaca-0.3", "refs")]],
                         ["ref-01", "ref-02", "ref-04", "ref-07", "ref-09"], "ref-01", kf)
run("vrais refs, tout (toutes sondes)", d[(m, "bf16", "refs")], [d[(m, "lora-s360_alpaca-0.3", "refs")]], ["ref-01", "ref-02", "ref-04", "ref-07", "ref-09"], "ref-01")
# 2) faux jeu n° 2 (vrais checkpoints, échantillonneur fidèle), 2 tirages, budget 70
d = A.load("sim/out/lik_fake2"); spec = json.load(open("sim/spec_set2.json"))["endpoints"]
src = {json.loads(l)["prompt_id"]: json.loads(l)["src"] for l in open("sim/protocol2.jsonl", encoding="utf-8")}
for fam, base, tag in (("LL1", "unsloth/Llama-3.2-1B-Instruct", "ll1"), ("QA3", "Qwen/Qwen2.5-3B-Instruct", "q3"), ("QB15", "Qwen/Qwen2.5-1.5B-Instruct", "q15")):
    es = [e for e in spec if e["family"] == fam]
    plain = next(e["ep"] for e in es if e["variant"] == "plain")
    px = [d[k] for k in d if k[0] == base and k[1] == f"lora-{tag}_alp2-0.5"]
    for block in (0, 1):
        kf = lambda k, b=block: k[2] // 1000 == b and src.get(k[1]) in ("v1", "v2a", "v2b")
        full = [e["ep"] for e in es if e["variant"] in ("plain", "fp8", "fp16", "int8", "lora0.3", "lora0.7", "lora1.5", "lora_med1.0")]
        real = [e["ep"] for e in es if e["variant"] in ("plain", "fp8", "lora0.3", "lora0.7", "lora1.5", "lora_med1.0")]
        accs[f"faux {fam} b{block} (sans fp16/int8)"] = run(f"faux jeu {fam} tirage {block} (sans fp16/int8)", d[(base, "bf16", "eps")], px, real, plain, kf)
        accs[f"faux {fam} b{block} (avec fp16/int8)"] = run(f"faux jeu {fam} tirage {block} (AVEC fp16/int8)", d[(base, "bf16", "eps")], px, full, plain, kf)
# 3) vraies familles ep (vérité = score public pour QB et Llama)
d = A.load("sim/out/lik_v2b")
for fam, base, tag, cands, plain, excl in (("Qwen B", "Qwen/Qwen2.5-1.5B-Instruct", "q15", "32 26 21 03 22 07 28", "ep-32", ()),
                                           ("Llama", "unsloth/Llama-3.2-1B-Instruct", "ll1", "13 19 09 15 11 05 27", "ep-13", ()),
                                           ("Qwen A (tous)", "Qwen/Qwen2.5-3B-Instruct", "q3", "12 25 01 17 18 24 14", None, ()),
                                           ("Qwen A (sans 25, 18 écartés)", "Qwen/Qwen2.5-3B-Instruct", "q3", "12 25 01 17 18 24 14", None, ("ep-25", "ep-18", "ep-14"))):
    px = [d[(base, f"lora-{tag}_{a}-0.3", "eps")] for a in ("alpaca", "alp2", "med")]
    r = run(f"vrais ep {fam}", d[(base, "bf16", "eps")], px, ["ep-" + c for c in cands.split()], plain, exclude=excl)
    if plain:
        accs[f"ep {fam}"] = r
print("\nRésumé (probabilité que le vrai plain soit n° 1) :")
for k, v in accs.items():
    if v is not None:
        print(f"  {k:36s} {100 * v:5.0f}%")
