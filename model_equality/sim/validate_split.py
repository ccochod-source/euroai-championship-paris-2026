"""Réplication sur données DISJOINTES : la règle du plain appliquée séparément à la moitié A (v1 + v2a, 46 réponses courtes)
et à la moitié B (v2b, 24 réponses longues). Si un endpoint n'est n° 1 que par hasard sur l'ensemble, il ne l'est pas dans
les deux moitiés. On donne aussi le taux de réussite de ce critère « n° 1 dans les deux moitiés » sur les cas à vérité connue.
    python sim/validate_split.py"""
import sys, math, json, collections
import numpy as np
sys.path.insert(0, "sim")
import analyze_lik as A
I7 = 3
A_ = lambda pid: pid.startswith(("v1_", "v2_")) and not pid.startswith("v2b")
B_ = lambda pid: pid.startswith("v2b_")


def stats(db, proxies, cands, keyf):
    out = {}
    for e in cands:
        x = v = n = lp = 0.0
        for k, r in db.items():
            if k[0] == e and keyf(k) and all(k in p and p[k]["n"] == r["n"] for p in proxies):
                x += r["nll"][I7] - r["H"][I7]; v += r["V"][I7]; n += r["n"]; lp += sum(r["nll"][I7] - p[k]["nll"][I7] for p in proxies)
        out[e] = (x / math.sqrt(v), lp / n * 1000 / len(proxies), int(n)) if n else None
    return out


def winner(st):
    c = [e for e in st if st[e]]
    rx = sorted(c, key=lambda e: st[e][0]); rp = sorted(c, key=lambda e: st[e][1])
    sc = {e: rx.index(e) + rp.index(e) + 1e-3 * rx.index(e) for e in c}
    return min(c, key=sc.get), sorted(c, key=sc.get)


res = []
def case(name, db, proxies, cands, plain, fa, fb, show=False):
    sa = stats(db, proxies, cands, fa); sb = stats(db, proxies, cands, fb)
    wa, ra = winner(sa); wb, rb = winner(sb)
    if plain:
        res.append((name, wa == plain, wb == plain))
    print(f"{name:34s} moitié A : n°1 {wa[-6:]:>6s}  | moitié B : n°1 {wb[-6:]:>6s}" + (f"   (vrai plain {plain[-6:]} : {'OK' if wa == plain else 'non'} / {'OK' if wb == plain else 'non'})" if plain else ""))
    if show:
        for e in cands:
            print(f"      {e}  A: z={sa[e][0]:+5.1f} proxy={sa[e][1]:+6.1f} (rang {ra.index(e) + 1})   B: z={sb[e][0]:+5.1f} proxy={sb[e][1]:+6.1f} (rang {rb.index(e) + 1})")


# vrais refs
d = A.load("sim/out/lik_refs2"); m = "HuggingFaceTB/SmolLM2-360M-Instruct"
case("vrais refs", d[(m, "bf16", "refs")], [d[(m, "lora-s360_alpaca-0.3", "refs")]], ["ref-01", "ref-02", "ref-04", "ref-07", "ref-09"], "ref-01",
     lambda k: A_(k[1]), lambda k: B_(k[1]))
# faux jeu n° 2
d = A.load("sim/out/lik_fake2"); spec = json.load(open("sim/spec_set2.json"))["endpoints"]
for fam, base, tag in (("LL1", "unsloth/Llama-3.2-1B-Instruct", "ll1"), ("QA3", "Qwen/Qwen2.5-3B-Instruct", "q3"), ("QB15", "Qwen/Qwen2.5-1.5B-Instruct", "q15")):
    es = [e for e in spec if e["family"] == fam]; plain = next(e["ep"] for e in es if e["variant"] == "plain")
    px = [d[k] for k in d if k[0] == base and k[1] == f"lora-{tag}_alp2-0.5"]
    cands = [e["ep"] for e in es if e["variant"] in ("plain", "fp8", "lora0.3", "lora0.7", "lora1.5", "lora_med1.0")]
    for b in (0, 1):
        case(f"faux jeu 2 {fam} tirage {b}", d[(base, "bf16", "eps")], px, cands, plain, lambda k, b=b: k[2] // 1000 == b and A_(k[1]), lambda k, b=b: k[2] // 1000 == b and B_(k[1]))
# vrais ep
d = A.load("sim/out/lik_v2b")
for fam, base, tag, cands, plain in (("vrais ep Qwen B", "Qwen/Qwen2.5-1.5B-Instruct", "q15", "32 26 21 03 22 07 28", "ep-32"),
                                     ("vrais ep Llama", "unsloth/Llama-3.2-1B-Instruct", "ll1", "13 19 09 15 11 05 27", "ep-13"),
                                     ("vrais ep Qwen A (4 candidats)", "Qwen/Qwen2.5-3B-Instruct", "q3", "12 01 17 24", None),
                                     ("vrais ep Qwen A (+25, 18)", "Qwen/Qwen2.5-3B-Instruct", "q3", "12 01 17 24 25 18", None)):
    px = [d[(base, f"lora-{tag}_{a}-0.3", "eps")] for a in ("alpaca", "alp2", "med")]
    case(fam, d[(base, "bf16", "eps")], px, ["ep-" + c for c in cands.split()], plain, lambda k: A_(k[1]), lambda k: B_(k[1]), show=True)
ok_both = sum(a and b for _, a, b in res); n = len(res)
print(f"\nCas à vérité connue : vrai plain n° 1 dans la moitié A : {sum(a for _, a, _ in res)}/{n} ; moitié B : {sum(b for _, _, b in res)}/{n} ; dans les DEUX : {ok_both}/{n}")
