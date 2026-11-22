"""Classe les prompts candidats selon le signal attendu par requête (sortie de kl_design.py).

    python sim/analyze_design.py [sim/out/design]
"""
import collections
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out", "design")
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["pid"]][r["variant"]] = r
    vs = sorted({r["variant"] for r in rows}, key=lambda v: (v.startswith("lora"), v))
    print(f"\n### {os.path.basename(p)[:-6]} : SNR par requête (signal sKL / bruit) ; entre crochets signal par requête (nats)\n")
    print("| prompt | tok/req | " + " | ".join(v.split(":")[0] + (":" + v.split(":")[-1] if v.startswith("lora") else "") for v in vs) + " |")
    print("|---|---|" + "---|" * len(vs))
    key = "fp8wa" if "fp8wa" in vs else vs[0]
    for pid in sorted(by, key=lambda k: -by[k].get(key, {}).get("snr_per_req", 0)):
        t = next(iter(by[pid].values()))["tok_per_req"]
        cells = [f"{by[pid][v]['snr_per_req']:.2f} [{by[pid][v]['signal_per_req']:.2f}]" if v in by[pid] else "—" for v in vs]
        print(f"| {pid} | {t:.0f} | " + " | ".join(cells) + " |")
