"""Faux jeu n° 2 (vérité connue) : traits de vraisemblance par endpoint selon le budget de requêtes, puissance par type
de sonde, et score simulé de plusieurs règles d'étiquetage (dont la soumission structurelle de l'orchestrateur).

    python sim/eval_set2.py [--dir sim/out/lik_fake2] [--spec sim/spec_set2.json] [--proto sim/protocol2.jsonl]
"""
import argparse, collections, itertools, json, math, re, sys
import numpy as np
sys.path.insert(0, "sim")
import analyze_lik as A

I7 = A.TEMPS.index(0.7)
NORM = lambda v: re.sub(r"lora-(ll1|q3|q15|s360)_", "lora-", v)
BUDGETS = {"46 (v1+v2a)": ("v1", "v2a"), "70 (+v2b)": ("v1", "v2a", "v2b"), "110 (+v2b+rsv)": ("v1", "v2a", "v2b", "rsv"),
           "86 (v1+v2a+rsv)": ("v1", "v2a", "rsv")}


def cat(pid):
    for p, c in (("rsv_pk_mul", "mul"), ("v2b_pk_mul", "mul"), ("v2_pk_mul", "mul"), ("rsv_al", "alpaca"), ("rsv_med", "domaine"),
                 ("rsv_law", "domaine"), ("rsv_code", "domaine"), ("rsv_story", "histoire"), ("v2b_story", "histoire"),
                 ("rsv_words", "liste"), ("rsv_sent", "liste"), ("rsv_nums", "liste"), ("v2b_names", "liste"),
                 ("v2b_words", "liste"), ("v2b_nums", "liste"), ("v2_cat", "liste"), ("rsv_cont", "cont256"),
                 ("v2b_cont", "cont256"), ("v2b_tr", "trad"), ("v2b_x", "extraction"), ("v2_c_", "cont48"),
                 ("v2_d_", "domaine64"), ("v2_pk_mix", "mix"), ("v1_", "v1")):
        if pid.startswith(p):
            return c
    return "autre"


def mcc(t, g):
    t, g = np.asarray(t, bool), np.asarray(g, bool)
    tp = (t & g).sum(); tn = (~t & ~g).sum(); fp = (~t & g).sum(); fn = (t & ~g).sum()
    d = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def score(lab, truth, fam):
    tt, gg = [], []
    for a, b in itertools.combinations(sorted(lab), 2):
        if fam[a] == fam[b]:
            tt.append(truth[a] == truth[b]); gg.append(lab[a] == lab[b])
    return mcc(tt, gg)


def feats(data, base, eps, keep):
    """keep(key) -> bool. Renvoie {ep: {n, T, exc, z07, llr_<v>, z_<v>}}"""
    db = {k: r for k, r in data[(base, "bf16", "eps")].items() if k[0] in eps and keep(k)}
    out = {}
    for e in eps:
        g = [r for k, r in db.items() if k[0] == e]
        if not g:
            continue
        nll = np.array([r["nll"] for r in g]).sum(0); H = np.array([r["H"] for r in g]).sum(0)
        V = np.array([r["V"] for r in g]).sum(0); n = sum(r["n"] for r in g)
        # T* continu : parabole autour du minimum de la NLL sur la grille
        i = int(np.argmin(nll)); T = A.TEMPS[i]
        if 0 < i < len(A.TEMPS) - 1:
            x = np.array(A.TEMPS[i - 1:i + 2]); c = np.polyfit(x, nll[i - 1:i + 2], 2)
            if c[0] > 0:
                T = float(-c[1] / (2 * c[0]))
        out[e] = {"n": n, "T": T, "exc": (nll[I7] - H[I7]) / n * 1000, "z07": (nll[I7] - H[I7]) / math.sqrt(V[I7])}
    for (m, v, rs), dq in data.items():
        if m != base or v == "bf16":
            continue
        for e in out:
            vals, ns = [], []
            for k, rb in db.items():
                if k[0] == e and k in dq and dq[k]["n"] == rb["n"]:
                    vals.append(rb["nll"][I7] - dq[k]["nll"][I7]); ns.append(rb["n"])
            if not vals:
                continue
            vals, ns = np.array(vals), np.array(ns)
            mm = vals.sum() / ns.sum(); se = math.sqrt(((vals - mm * ns) ** 2).sum()) / ns.sum()
            vn = NORM(v)
            out[e]["llr_" + vn] = mm * 1000; out[e]["z_" + vn] = mm / se if se else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sim/out/lik_fake2")
    ap.add_argument("--spec", default="sim/spec_set2.json")
    ap.add_argument("--proto", default="sim/protocol2.jsonl")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    data = A.load(a.dir)
    spec = json.load(open(a.spec))["endpoints"]
    src = {json.loads(l)["prompt_id"]: json.loads(l)["src"] for l in open(a.proto, encoding="utf-8")}
    truth = {e["ep"]: e for e in spec}
    fams = collections.defaultdict(list)
    for e in spec:
        fams[(e["family"], e["base"])].append(e["ep"])
    fams = {k: v for k, v in fams.items() if (k[1], "bf16", "eps") in data}
    allv = sorted({NORM(v) for (m, v, r) in data if v != "bf16"})
    prox = [v for v in allv if v.startswith("lora-") and "alp2" in v]
    print("variantes :", allv)

    # ---------- 1. traits par budget ----------
    res = {}
    for bname, srcs in BUDGETS.items():
        for block in (0, 1):
            for (fam, base), eps in fams.items():
                keep = lambda k: int(k[2]) // 1000 == block and src.get(k[1]) in srcs
                res[(bname, block, fam)] = feats(data, base, set(eps), keep)
    for bname in BUDGETS:
        print(f"\n## Budget {bname} — bloc 0")
        for (fam, base), eps in fams.items():
            F = res[(bname, 0, fam)]
            print(f"### {fam}   (n tokens médian {int(np.median([f['n'] for f in F.values()]))})")
            cols = ["z_fp8w", "z_int8w", "z_fp16"] + ["z_" + v for v in prox] + ["llr_" + v for v in prox]
            print(f"{'ep':6s} {'variante':12s} {'T*':>5s} {'excès':>6s} {'z07':>6s} " + " ".join(f"{c[-14:]:>14s}" for c in cols))
            for e in sorted(F, key=lambda e: truth[e]["variant"]):
                f = F[e]
                print(f"{e:6s} {truth[e]['variant']:12s} {f['T']:5.2f} {f['exc']:+6.0f} {f['z07']:+6.1f} "
                      + " ".join(f"{f.get(c, float('nan')):+14.1f}" for c in cols))

    # ---------- 2. règles d'étiquetage ----------
    PX = "llr_" + prox[0] if prox else None   # proxy LoRA indépendant (graine différente), force 0.5

    def label(F, fam, mode):
        lab = {}
        eps = sorted(F)
        t1 = [e for e in eps if F[e]["T"] >= 0.9 and F[e]["exc"] > 400]
        pers = [e for e in eps if truth[e]["variant"] in ("persona", "sysprompt")]      # détectés par le texte (oracle)
        wm = [e for e in eps if truth[e]["variant"] == "watermark"]                       # détecteur e3 (oracle)
        S = set(t1) | set(pers) | set(wm)
        rest = [e for e in eps if e not in S]
        if mode == "struct":                      # soumission structurelle : S prouvés, reste en singletons
            pass
        elif mode == "struct+plain_oracle":
            S |= {e for e in rest if truth[e]["variant"] == "plain"}
        elif mode.startswith("struct+plain"):     # plain = endpoint restant le moins anormal
            cand = [e for e in rest if F[e].get("z_fp8w", -9) <= 0]
            if PX:
                med = np.median([F[e][PX] for e in rest])
            if mode == "struct+plain_exc":
                key = lambda e: abs(F[e]["exc"] - np.median([F[x]["exc"] for x in wm + cand]))
            elif mode == "struct+plain_minexc":
                key = lambda e: F[e]["exc"]
            elif mode == "struct+plain_proxy":
                key = lambda e: F[e][PX]
            elif mode == "struct+plain_combo":
                rk = lambda vals: {e: r for r, e in enumerate(sorted(cand, key=lambda e: vals[e]))}
                r1 = rk({e: F[e]["exc"] for e in cand}); r2 = rk({e: F[e][PX] for e in cand}); r3 = rk({e: F[e].get("z_fp8w", 0) for e in cand})
                key = lambda e: r1[e] + r2[e] + r3[e]
            if cand:
                S.add(min(cand, key=key))
        for e in eps:
            lab[e] = f"{fam}_S" if e in S else f"{fam}_{e}"
        return lab

    print("\n## Score simulé (0.1 + 0.9 x MCC modèle), par règle et par budget ; moyenne des 2 blocs [bloc 0, bloc 1]")
    modes = ["struct", "struct+plain_oracle", "struct+plain_minexc", "struct+plain_exc", "struct+plain_proxy", "struct+plain_combo"]
    T = {e["ep"]: e["model"] for e in spec}; Ff = {e["ep"]: e["family"] for e in spec}
    for bname in BUDGETS:
        print(f"### Budget {bname}")
        for mode in modes:
            if "proxy" in mode or "combo" in mode:
                if not PX:
                    continue
            sc, picks = [], []
            for block in (0, 1):
                lab = {}
                for (fam, base) in fams:
                    l = label(res[(bname, block, fam)], fam, mode); lab.update(l)
                    if "plain_" in mode and "oracle" not in mode:
                        S = [e for e in l if l[e].endswith("_S")]
                        picks += [truth[e]["variant"] for e in S if truth[e]["variant"] not in ("persona", "sysprompt", "watermark", "sampling_T1")]
                sc.append(0.1 + 0.9 * score(lab, T, Ff))
            print(f"  {mode:22s} {np.mean(sc):.3f}  [{sc[0]:.3f}, {sc[1]:.3f}]" + (f"   plain choisi : {picks}" if picks else ""))

    # ---------- 3. puissance par type de sonde (z par requête) ----------
    print("\n## Signal par requête et par type de sonde (LLR de l'endpoint cible moins LLR des S sans system prompt, en nats par requête ; "
          "entre parenthèses : rapport signal/écart-type par requête)")
    tests = [("fp8 (LLR fp8w)", "fp8w", ["fp8"]), ("int8 (LLR int8w)", "int8w", ["int8"])]
    if prox:
        tests += [("LoRA 0.3 (proxy)", prox[0], ["lora0.3"]), ("LoRA 0.7 (proxy)", prox[0], ["lora0.7"]),
                  ("LoRA méd. (proxy alpaca)", prox[0], ["lora_med1.0"])]
    tests += [("LoRA 0.3 (excès)", None, ["lora0.3"]), ("fp8 (excès)", None, ["fp8"])]
    cats = sorted({cat(p) for p in src})
    print(f"{'test':26s} " + " ".join(f"{c:>12s}" for c in cats))
    for name, v, targets in tests:
        row = []
        for c in cats:
            pos, neg = [], []
            for (fam, base), eps in fams.items():
                db = data[(base, "bf16", "eps")]
                dq = next((data[k] for k in data if k[0] == base and NORM(k[1]) == v), None) if v else None
                if v and dq is None:
                    continue
                for k, rb in db.items():
                    if k[0] not in eps or cat(k[1]) != c:
                        continue
                    if v:
                        if k not in dq or dq[k]["n"] != rb["n"]:
                            continue
                        val = rb["nll"][I7] - dq[k]["nll"][I7]
                    else:
                        val = rb["nll"][I7] - rb["H"][I7]
                    var = truth[k[0]]["variant"]
                    if var in targets:
                        pos.append(val)
                    elif var in ("plain", "watermark", "fp16"):
                        neg.append(val)
            if len(pos) > 2 and len(neg) > 2:
                d = np.mean(pos) - np.mean(neg); sd = math.sqrt((np.var(pos) + np.var(neg)) / 2)
                row.append(f"{d:+6.2f}({d / sd:+4.2f})")
            else:
                row.append("     -      ")
        print(f"{name:26s} " + " ".join(f"{x:>12s}" for x in row))
    n_tok = collections.defaultdict(list)
    for (fam, base), eps in fams.items():
        for k, rb in data[(base, "bf16", "eps")].items():
            n_tok[cat(k[1])].append(rb["n"])
    print(f"{'tokens moyens/requête':26s} " + " ".join(f"{np.mean(n_tok[c]):12.0f}" for c in cats))


if __name__ == "__main__":
    main()
