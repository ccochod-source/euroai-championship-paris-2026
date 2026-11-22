"""Re-validation de la règle du plain sur le jeu n° 3 (vrais checkpoints, 6 tirages par configuration, budget 70 réponses).
Pour chaque tirage (bloc) : statistiques par instance (z d'excès à T = 0.7, LLR du proxy LoRA alp3 à force 0.3), puis
on compose des « familles » : le plain + un ensemble de variantes, et on regarde si le plain est n° 1 au rang combiné.
    python sim/validate_plain_set3.py [FAM=LL1]"""
import sys, json, math, collections, itertools, os
import numpy as np
FAM = sys.argv[1] if len(sys.argv) > 1 else "LL1"
B = {"LL1": ("unsloth__Llama-3.2-1B-Instruct", "ll1"), "QA3": ("Qwen__Qwen2.5-3B-Instruct", "q3"),
     "QB15": ("Qwen__Qwen2.5-1.5B-Instruct", "q15"), "S360": ("HuggingFaceTB__SmolLM2-360M-Instruct", "s360")}[FAM]
src = {json.loads(l)["prompt_id"]: json.loads(l)["src"] for l in open("sim/protocol2.jsonl", encoding="utf-8")}
def load(v):
    d = {}
    for l in open(f"sim/out/lik3/{B[0]}__{v}__all_inst.jsonl"):
        r = json.loads(l)
        if src.get(r["prompt_id"]) in ("v1", "v2a", "v2b"):
            d[(r["endpoint"], r["prompt_id"], r["repeat"])] = (r["nll"][3], r["H"][3], r["V"][3], r["n"])
    return d
db = load("bf16"); dp = load(f"lora-{B[1]}_alp3-0.3")
acc = collections.defaultdict(lambda: np.zeros(4))
for k, (nll, H, V, n) in db.items():
    if k in dp and dp[k][3] == n:
        acc[(k[0].split("-", 1)[1], k[2] // 1000)] += np.array([nll - H, V, n, nll - dp[k][0]])
S = {k: (v[0] / math.sqrt(v[1]), v[3] / v[2] * 1000) for k, v in acc.items() if v[2] > 2000}
names = sorted({k[0] for k in S})
print(f"### {FAM} : moyenne sur les tirages (z d'excès ; LLR proxy 0.3 x1e-3/token)")
for nme in names:
    v = [S[(nme, b)] for b in range(6) if (nme, b) in S]
    print(f"  {nme:22s} tirages={len(v)}  z={np.mean([x[0] for x in v]):+6.1f} (min {min(x[0] for x in v):+5.1f}, max {max(x[0] for x in v):+5.1f})   proxy={np.mean([x[1] for x in v]):+7.1f}")
def trial(plain_b, variants):
    """variants : liste de (nom, bloc). Renvoie True si le plain est n° 1 au rang combiné."""
    c = [("plain", plain_b)] + variants
    rx = sorted(c, key=lambda k: S[k][0]); rp = sorted(c, key=lambda k: S[k][1])
    sc = {k: rx.index(k) + rp.index(k) + 1e-3 * rx.index(k) for k in c}
    return min(c, key=sc.get) == ("plain", plain_b)
rng = np.random.default_rng(0)
def blocks(n): return [b for b in range(6) if (n, b) in S]
COMPO = {
  "structure réelle supposée (fp8 + 3 LoRA faibles 0.1/0.2/0.4 + variante tronquée)": lambda ad: ["fp8", f"lora_{ad}_0.1", f"lora_{ad}_0.2", f"lora_{ad}_0.4", "lora_med_0.2_trunc"],
  "idem + jumeau fp16": lambda ad: ["fp8", "fp16", f"lora_{ad}_0.1", f"lora_{ad}_0.2", f"lora_{ad}_0.4", "lora_med_0.2_trunc"],
  "idem + jumeau int8": lambda ad: ["fp8", "int8", f"lora_{ad}_0.1", f"lora_{ad}_0.2", f"lora_{ad}_0.4", "lora_med_0.2_trunc"],
  "seulement le LoRA le plus faible (0.1) + fp8": lambda ad: ["fp8", f"lora_{ad}_0.1"],
}
print(f"\n### {FAM} : le vrai plain est-il n° 1 ? (2 000 familles tirées au hasard parmi les tirages)")
for name, f in COMPO.items():
    for ad in ("alpaca", "alp2", "med"):
        ok = n = 0
        for _ in range(2000):
            vs = [(v, int(rng.choice(blocks(v)))) for v in f(ad) if blocks(v)]
            pb = int(rng.choice(blocks("plain")))
            ok += trial(pb, vs); n += 1
        print(f"  {name:78s} adapter {ad:7s} : {100 * ok / n:5.1f} %")
