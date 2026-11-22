"""Classement des candidats « plain » : combine la vraisemblance bf16 de v1 + v2a (GPU, out/lik_r2) et de v2b (CPU,
out/lik_cpu2). Par endpoint : tokens, T la plus vraisemblable (parabole sur la grille), excès et z à T = 0.7,
tokens de rang >= 50 (troncature), et écart fp8 - bf16 / int8 - bf16 quand les variantes sont scorées.
    python sim/cpu_rank.py [famille ...]"""
import collections, glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]; I7 = 3
FAM = {"Llama": ("unsloth__Llama-3.2-1B-Instruct", "05 09 10 11 13 15 16 19 27 29 31", {"10": "watermark", "31": "T=1", "29": "system prompt", "16": "persona"}),
       "QwenB": ("Qwen__Qwen2.5-1.5B-Instruct", "03 06 07 08 20 21 22 26 28 30 32", {"06": "watermark", "30": "T=1", "08": "system prompt", "20": "persona"}),
       "QwenA": ("Qwen__Qwen2.5-3B-Instruct", "01 02 04 12 14 17 18 23 24 25 33", {"04": "watermark", "33": "T=1", "02": "system prompt", "23": "persona"})}


def load(path):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); d[(r["endpoint"], r["prompt_id"], int(r["repeat"]))] = r
    return d


def summarize(recs):
    nll = np.array([r["nll"] for r in recs]).sum(0); H = np.array([r["H"] for r in recs]).sum(0); V = np.array([r["V"] for r in recs]).sum(0)
    n = sum(r["n"] for r in recs)
    i = int(np.argmin(nll)); T = TEMPS[i]; se = float("nan")
    if 0 < i < len(TEMPS) - 1:
        c = np.polyfit(TEMPS[i - 1:i + 2], nll[i - 1:i + 2], 2)
        if c[0] > 0:
            T = -c[1] / (2 * c[0]); se = 1 / math.sqrt(2 * c[0])
    return n, T, se, (nll[I7] - H[I7]) / n * 1000, (nll[I7] - H[I7]) / math.sqrt(V[I7])


def llr(dq, db, keys):
    vals, ns = [], []
    for k in keys:
        if k in dq and k in db and dq[k]["n"] == db[k]["n"]:
            vals.append(db[k]["nll"][I7] - dq[k]["nll"][I7]); ns.append(db[k]["n"])
    if len(vals) < 5:
        return None
    vals, ns = np.array(vals), np.array(ns); m = vals.sum() / ns.sum()
    se = math.sqrt(((vals - m * ns) ** 2).sum()) / ns.sum()
    return m * 1000, m / se, int(ns.sum())


for fam in (sys.argv[1:] or FAM):
    base, eps, roles = FAM[fam]
    B = load(os.path.join(HERE, "out/lik_r2", f"{base}__bf16__eps.jsonl"))
    B.update(load(os.path.join(HERE, "out/lik_cpu2", f"{base}__bf16__eps_v2b.jsonl")))
    Q = {}
    for v in ("fp8w", "int8w"):
        Q[v] = load(os.path.join(HERE, "out/lik_r2", f"{base}__{v}__eps.jsonl"))
        Q[v].update(load(os.path.join(HERE, "out/lik_cpu2", f"{base}__{v}__eps_v2b.jsonl")))
    print(f"\n### {fam} — bf16, v1 + v2a (+ v2b quand scoré)")
    print(f"{'endpoint':8s} {'rôle':14s} {'v2b':>4s} {'tokens':>6s} {'T':>6s} {'±':>5s} {'excès':>6s} {'z':>6s} {'|T-0.70|/σ':>10s} {'fp8-bf16 (z, tokens)':>24s} {'int8-bf16 (z)':>14s}")
    rows = []
    for e in eps.split():
        ep = f"ep-{e}"
        keys = [k for k in B if k[0] == ep]
        if not keys:
            continue
        n, T, se, exc, z = summarize([B[k] for k in keys])
        nv2b = sum(k[1].startswith("v2b_") for k in keys)
        f8 = llr(Q["fp8w"], B, keys); i8 = llr(Q["int8w"], B, keys)
        rows.append((abs(T - 0.70) / se if se == se else 99, ep, roles.get(e, ""), nv2b, n, T, se, exc, z, f8, i8))
    for d, ep, role, nv2b, n, T, se, exc, z, f8, i8 in sorted(rows, key=lambda r: (r[2] != "", r[0])):
        print(f"{ep:8s} {role:14s} {nv2b:4d} {n:6d} {T:6.3f} {se:5.3f} {exc:+6.0f} {z:+6.1f} {d:10.1f} "
              + (f"{f8[0]:+7.1f} ({f8[1]:+5.1f}, {f8[2]:5d})".rjust(24) if f8 else " " * 24)
              + (f"{i8[0]:+6.1f} ({i8[1]:+5.1f})".rjust(15) if i8 else ""))
