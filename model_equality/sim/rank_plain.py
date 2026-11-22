"""Classement des candidats « plain » à partir d'un dossier de vraisemblances (lik.py) : par endpoint, tokens,
T la plus vraisemblable, excès et z à T = 0.7, z du test fp8 / int8, LLR des proxys LoRA centrés sur la famille,
et (si fichier _samp) nombre de tokens de rang >= 50. Étalon : ep-32 = plain confirmé de Qwen B.
    python sim/rank_plain.py [dossier=sim/out/lik_v2b] [--prefix v2b_]"""
import glob, json, math, os, re, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "sim/out/lik_v2b"
PREF = tuple(sys.argv[sys.argv.index("--prefix") + 1].split(",")) if "--prefix" in sys.argv else None
TEMPS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]; I7 = 3
FAM = {"Llama": ("unsloth__Llama-3.2-1B-Instruct", "05 09 10 11 13 15 16 19 27 29 31", {"10": "watermark", "31": "T=1", "29": "sys", "16": "persona"}),
       "QwenA": ("Qwen__Qwen2.5-3B-Instruct", "01 02 04 12 14 17 18 23 24 25 33", {"04": "watermark", "33": "T=1", "02": "sys", "23": "persona"}),
       "QwenB": ("Qwen__Qwen2.5-1.5B-Instruct", "03 06 07 08 20 21 22 26 28 30 32", {"06": "watermark", "30": "T=1", "08": "sys", "20": "persona", "32": "PLAIN (public)"})}


def load(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        if PREF is None or r["prompt_id"].startswith(PREF):
            d[(r["endpoint"], r["prompt_id"], int(r["repeat"]))] = r
    return d


def llr(dq, db, ep):
    vals, ns = [], []
    for k, rb in db.items():
        if k[0] == ep and k in dq and dq[k]["n"] == rb["n"]:
            vals.append(rb["nll"][I7] - dq[k]["nll"][I7]); ns.append(rb["n"])
    if len(vals) < 5:
        return float("nan"), float("nan")
    vals, ns = np.array(vals), np.array(ns); m = vals.sum() / ns.sum()
    return m * 1000, m / (math.sqrt(((vals - m * ns) ** 2).sum()) / ns.sum())


for fam, (base, eps, roles) in FAM.items():
    fb = os.path.join(D, f"{base}__bf16__eps.jsonl")
    if not os.path.exists(fb):
        continue
    B = load(fb)
    var = {}
    for p in glob.glob(os.path.join(D, f"{base}__*__eps.jsonl")):
        v = os.path.basename(p).split("__")[2]
        if v != "bf16":
            var[re.sub(r"lora-(ll1|q3|q15)_", "", v)] = load(p)
    samp = {}
    fs = os.path.join(D, f"{base}__bf16__eps_samp.jsonl")
    if os.path.exists(fs):
        for l in open(fs):
            r = json.loads(l)
            samp[r["endpoint"]] = samp.get(r["endpoint"], 0) + r["samp"]["0.7|50|1.0"][3]
    vs = sorted(var)
    rows = []
    for e in eps.split():
        ep = f"ep-{e}"; g = [r for k, r in B.items() if k[0] == ep]
        if not g:
            continue
        nll = np.array([r["nll"] for r in g]).sum(0); H = np.array([r["H"] for r in g]).sum(0); V = np.array([r["V"] for r in g]).sum(0)
        n = sum(r["n"] for r in g); i = int(np.argmin(nll)); T = TEMPS[i]; se = float("nan")
        if 0 < i < len(TEMPS) - 1:
            c = np.polyfit(TEMPS[i - 1:i + 2], nll[i - 1:i + 2], 2)
            if c[0] > 0:
                T = -c[1] / (2 * c[0]); se = 1 / math.sqrt(2 * c[0])
        rows.append({"ep": ep, "role": roles.get(e, ""), "n": n, "T": T, "se": se, "exc": (nll[I7] - H[I7]) / n * 1000,
                     "z": (nll[I7] - H[I7]) / math.sqrt(V[I7]), "r50": samp.get(ep, -1), **{v: llr(var[v], B, ep) for v in vs}})
    med = {v: np.nanmedian([r[v][0] for r in rows]) for v in vs}
    print(f"\n### {fam} ({D}{', ' + ','.join(PREF) if PREF else ''}) — LLR des variantes x1e-3/token centré sur la médiane de la famille [z brut]")
    print(f"{'endpoint':7s} {'rôle':14s} {'tokens':>6s} {'T':>6s} {'±':>5s} {'excès':>5s} {'z':>5s} {'r>=50':>5s} " + " ".join(f"{v[-13:]:>14s}" for v in vs))
    key = lambda r: (r["role"] not in ("", "PLAIN (public)"), np.nanmean([r[v][0] - med[v] for v in vs if "alpaca" in v or "med" in v]) if any("alpaca" in v or "med" in v for v in vs) else abs(r["T"] - 0.7))
    for r in sorted(rows, key=key):
        print(f"{r['ep']:7s} {r['role']:14s} {r['n']:6d} {r['T']:6.3f} {r['se']:5.3f} {r['exc']:+5.0f} {r['z']:+5.1f} {r['r50']:5d} "
              + " ".join(f"{r[v][0] - med[v]:+7.1f}[{r[v][1]:+5.1f}]" for v in vs))
