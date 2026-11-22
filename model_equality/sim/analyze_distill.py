"""Matrice « distiller puis tester » : ligne = endpoint X sur lequel on a appris un petit LoRA, colonne = endpoint Y évalué
(prompts jamais vus à l'entraînement). Valeur = LLR (x1e-3 nats/token) du LoRA appris contre le modèle de base, centré
sur la médiane de la ligne. Positif = Y partage le décalage appris sur X (même adapter / même variante).
    python sim/analyze_distill.py [dossier=sim/out/distill3] [alpha=0.25]"""
import glob, json, os, sys
import numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "sim/out/distill3"
al = sys.argv[2] if len(sys.argv) > 2 else "0.25"
by = {}
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    j = json.load(open(p)); fam = os.path.basename(p).split("_")[0]
    by.setdefault(fam, []).append((j["train_ep"], j.get("steps"), j["llr"]))
for fam, rows in by.items():
    eps = sorted({k.split("@")[0] for _, _, l in rows for k in l})
    print(f"\n### {fam} — alpha = {al} ; LLR centré sur la médiane de la ligne [z brut]")
    print(f"{'appris sur':14s} " + " ".join(f"{e[-2:]:>11s}" for e in eps))
    for tr, st, l in rows:
        v = {e: l.get(f"{e}@{float(al)}") for e in eps}
        med = np.median([x["llr_x1e3"] for x in v.values() if x])
        print(f"{tr} (s{st})".ljust(14) + " " + " ".join((f"{x['llr_x1e3'] - med:+5.0f}[{x['z'] if x['z'] is not None else 0:+4.1f}]" if x else " " * 11) for x in v.values()))
