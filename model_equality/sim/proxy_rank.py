"""Test LoRA par proxy à faible force : z BRUT de log p(base + LoRA proxy x force) - log p(base), par endpoint.
Validé sur les vrais refs (budget v1 + v2a + v2b, force 0.3) : plain -0.1, watermark -0.5, fp8 +0.7, T=1 +0.8 ;
LoRA 0.3 +3.1, LoRA 0.7 +6.7, LoRA 1.5 +8.4 (system prompt +2.7 et persona +3.7 : à exclure du test).
    python sim/proxy_rank.py [dossier=sim/out/lik_v2b]"""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_lik as A
D = sys.argv[1] if len(sys.argv) > 1 else "sim/out/lik_v2b"
data = A.load(D)
ROLE = {"10": "wm", "31": "T=1", "29": "sys", "16": "persona", "04": "wm", "33": "T=1", "02": "sys", "23": "persona", "06": "wm",
        "30": "T=1", "08": "sys", "20": "persona", "32": "PLAIN*", "25": "jumeau*", "26": "jumeau?", "15": "fp8", "22": "fp8"}
for (m, v, rs) in sorted(k for k in data if k[1] == "bf16" and k[2] == "eps"):
    db = data[(m, v, rs)]
    vs = sorted(x for (mm, x, r) in data if mm == m and r == rs and x.startswith("lora-") and not x.endswith("-1.0"))
    if not vs:
        continue
    eps = sorted({k[0] for k in db})
    print(f"\n### {m.split('/')[-1]} — z brut par proxy (LLR x1e-3/token entre crochets)")
    print(f"{'endpoint':8s} {'rôle':9s} " + " ".join(f"{re.sub('lora-(ll1|q3|q15)_', '', x):>16s}" for x in vs) + "   moyenne z")
    rows = []
    for e in eps:
        r = [A.llr(data[(m, x, rs)], db, e, 0.7) for x in vs]
        rows.append((sum(z for _, z in r) / len(r), e, r))
    for mz, e, r in sorted(rows):
        print(f"{e:8s} {ROLE.get(e[-2:], ''):9s} " + " ".join(f"{z:+6.1f} [{mm * 1000:+5.1f}]".rjust(16) for mm, z in r) + f"   {mz:+6.1f}")
