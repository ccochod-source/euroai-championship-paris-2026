"""Test PRÉ-ENREGISTRÉ sur la réserve (critères fixés le 22/11 à 2h30, avant toute donnée de réserve).
Entrée : dossier de vraisemblances out/lik_res (toutes les réponses) ; on n'utilise ici QUE les réponses de la réserve
(prompt_id commençant par v3_), jamais vues lors de la formulation des hypothèses.

A. Plain de Qwen A — candidats ep-12, ep-01, ep-17, ep-24 ; témoin S : watermark ep-04.
   Critère principal : écart APPARIÉ au watermark, prompt par prompt, de l'excès (nll - H) par token. Le plain est le
   candidat à l'écart le plus BAS, et cet écart doit être négatif ou nul (z apparié <= 0).
   (Calibré le 22/11 à 2h35 sur les deux plains connus : « le plus bas » retrouve ep-32 et ep-13 dans les 6 découpages
   testés ; « le plus proche du watermark », proposé d'abord, désigne les jumeaux ep-19 et ep-25 : écarté.)
   Critères secondaires : (i) n° 1 des 4 au rang combiné z d'excès + LLR proxy 0.3 ; (ii) z d'excès le plus bas des 4 sur
   les seules multiplications.
   Décision : embarquer un candidat seulement s'il gagne le critère principal ET les deux secondaires.
B. Tronqués ep-27 / ep-14 / ep-07 — témoin : watermark de la famille.
   Tests directionnels sur la réserve seule : z fp8 < +2 et z proxy LoRA au plus 2 au-dessus du watermark.
   (Ces tests excluent fp8 et LoRA, pas une quantization quasi sans perte : ils ne suffisent pas à mettre les tronqués dans S.)
C. ep-29 / ep-02 : si un nouveau system prompt est extrait, relancer lik.py avec SIM_SYSTEM_FILE et vérifier z proche de 0.

    python sim/reserve_test.py [dossier=sim/out/lik_res] [prefixe=v3_]
"""
import sys, os, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_lik as A
D = sys.argv[1] if len(sys.argv) > 1 else "sim/out/lik_res"; PRE = sys.argv[2] if len(sys.argv) > 2 else "v3_"
d = A.load(D); I7 = 3
def recs(m, v, e): return {(k[1], k[2]): r for k, r in d[(m, v, "eps")].items() if k[0] == e and k[1].startswith(PRE)}
def exc(m, e, mul=False):
    g = [r for k, r in recs(m, "bf16", e).items() if (not mul or "mul" in k[0])]
    return sum(r["nll"][I7] - r["H"][I7] for r in g) / math.sqrt(sum(r["V"][I7] for r in g)) if g else float("nan")
def llrz(m, v, e):
    b = recs(m, "bf16", e); q = recs(m, v, e)
    k = [x for x in b if x in q and q[x]["n"] == b[x]["n"]]
    if len(k) < 5: return float("nan"), float("nan")
    val = np.array([b[x]["nll"][I7] - q[x]["nll"][I7] for x in k]); n = np.array([b[x]["n"] for x in k]); mm = val.sum() / n.sum()
    return mm * 1000, mm / (math.sqrt(((val - mm * n) ** 2).sum()) / n.sum())
def paired(m, e, ref):
    a = recs(m, "bf16", e); b = recs(m, "bf16", ref); k = [x for x in a if x in b]
    dlt = np.array([(a[x]["nll"][I7] - a[x]["H"][I7]) / a[x]["n"] - (b[x]["nll"][I7] - b[x]["H"][I7]) / b[x]["n"] for x in k])
    return (dlt.mean() / (dlt.std(ddof=1) / math.sqrt(len(dlt))), len(k)) if len(k) > 3 else (float("nan"), len(k))
m = "Qwen/Qwen2.5-3B-Instruct"; C = ["ep-12", "ep-01", "ep-17", "ep-24"]; px = [f"lora-q3_{a}-0.3" for a in ("alpaca", "alp2", "med")]
print(f"=== A. Plain de Qwen A, réponses {PRE}* seules ===")
st = {}
for e in C + ["ep-04"]:
    zp, n = paired(m, e, "ep-04"); pl = np.nanmean([llrz(m, v, e)[0] for v in px if (m, v, "eps") in d])
    st[e] = (zp, exc(m, e), exc(m, e, True), pl)
    print(f"{e}: z apparié au watermark = {zp:+.1f} ({n} prompts) ; z d'excès = {st[e][1]:+.1f} ; z d'excès multiplications = {st[e][2]:+.1f} ; LLR proxy 0.3 = {pl:+.1f}")
rx = sorted(C, key=lambda e: st[e][1]); rp = sorted(C, key=lambda e: st[e][3])
comb = min(C, key=lambda e: rx.index(e) + rp.index(e) + 1e-3 * rx.index(e))
prim = min(C, key=lambda e: st[e][0]); mul = min(C, key=lambda e: st[e][2])
print(f"critère principal (écart apparié au watermark le plus bas, et <= 0) : {prim} (z = {st[prim][0]:+.1f})  -> {'OK' if st[prim][0] <= 0 else 'NON'}")
print(f"secondaire (i) rang combiné : {comb} ; secondaire (ii) multiplications : {mul}")
print("DÉCISION :", f"embarquer {prim}" if (prim == comb == mul and st[prim][0] <= 0) else "ne rien embarquer (critères non concordants)")
print("\n=== B. Tronqués : tests directionnels, réponses de la réserve seules ===")
for m, e, wm, tag in (("unsloth/Llama-3.2-1B-Instruct", "ep-27", "ep-10", "ll1"), ("Qwen/Qwen2.5-3B-Instruct", "ep-14", "ep-04", "q3"), ("Qwen/Qwen2.5-1.5B-Instruct", "ep-07", "ep-06", "q15")):
    if (m, "bf16", "eps") not in d: continue
    zf = llrz(m, "fp8w", e)[1]; zp = np.nanmean([llrz(m, f"lora-{tag}_{a}-0.3", e)[1] for a in ("alpaca", "alp2", "med")]); zw = np.nanmean([llrz(m, f"lora-{tag}_{a}-0.3", wm)[1] for a in ("alpaca", "alp2", "med")])
    print(f"{e}: z fp8 = {zf:+.1f} ; z proxy = {zp:+.1f} (watermark {wm} : {zw:+.1f}) -> " + ("ni fp8 ni LoRA" if zf < 2 and zp < zw + 2 else "VARIANTE détectée"))
