"""Recherche gloutonne du system prompt d'un endpoint : on compose des phrases d'un réservoir (phrases extraites des
prompts connus + quelques hypothèses) et on garde, à chaque tour, l'ajout qui baisse le plus la NLL des réponses de
l'endpoint sous le modèle de base. Si un assemblage donne z d'excès proche de 0 à T = 0.7, les poids de base sont prouvés.
    python sys_search.py --model <hf> --ep ep-02 --out out/sys_search/ep-02.json [--start "phrase initiale"]"""
import argparse, json, math, os
import torch
import lik
from transformers import AutoTokenizer

POOL = [
    "You are a helpful assistant for general questions and everyday tasks.",
    "Help the user with their questions and tasks.",
    "Ask a clarifying question when the request is ambiguous.",
    "Do not make up facts; say when you do not know.",
    "Use short paragraphs.",
    "Do not share personal data about private individuals.",
    "State the main point first, then add detail.",
    "Use a warm, friendly tone.",
    # hypothèses (style du même réservoir)
    "Keep answers concise.",
    "Be honest about uncertainty.",
    "Use a neutral, professional tone.",
    "Decline requests that are harmful or unsafe.",
    "Use bullet points when listing several items.",
    "Answer in the same language as the user.",
    "You are Claude, an AI assistant created by Anthropic.",
    "Be respectful and honest.",
]


def evaluate(model, tok, recs, text, sep):
    lik.SYS_MSG = [{"role": "system", "content": text}] if text else []
    out = lik.score(model, tok, recs, log_every=0)
    nll = sum(r["nll"][3] for r in out); H = sum(r["H"][3] for r in out); V = sum(r["V"][3] for r in out); n = sum(r["n"] for r in out)
    return nll / n, (nll - H) / math.sqrt(V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--ep", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sep", default=" "); ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--pool-file", default=None, help="une phrase par ligne, ajoutées au réservoir")
    ap.add_argument("--append-only", action="store_true")
    ap.add_argument("--start", default=None, help="phrases de départ séparées par |")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model); model = lik.load_model(a.model, "bf16")
    recs = lik.load_records("data/raw", "eps", {a.ep})
    log = []
    pool = POOL + ([l.strip() for l in open(a.pool_file, encoding="utf-8") if l.strip()] if a.pool_file else [])
    pool = list(dict.fromkeys(pool))
    cur = a.start.split("|") if a.start else []; best = evaluate(model, tok, recs, a.sep.join(cur), a.sep)
    print(f"{a.ep} départ : NLL/tok {best[0]:.4f} z {best[1]:+.1f}", flush=True)
    log.append({"sentences": [], "nll": best[0], "z": best[1]})
    for rnd in range(a.rounds):
        cand = []
        for s in pool:
            if s in cur:
                continue
            for pos in ({len(cur)} if a.append_only else ({0, len(cur)} if cur else {0})):   # ajout au début ou à la fin
                seq = cur[:pos] + [s] + cur[pos:]
                r = evaluate(model, tok, recs, a.sep.join(seq), a.sep)
                cand.append((r[0], r[1], seq))
        cand.sort(key=lambda x: x[0])
        nl, z, seq = cand[0]
        print(f"tour {rnd + 1}: meilleur ajout -> NLL/tok {nl:.4f} z {z:+.1f} : {a.sep.join(seq)!r}", flush=True)
        log.append({"sentences": seq, "nll": nl, "z": z, "runners_up": [(round(x[0], 4), round(x[1], 1), x[2][-1] if x[2][:-1] == cur else x[2][0]) for x in cand[1:4]]})
        json.dump(log, open(a.out, "w"), indent=1, ensure_ascii=False)
        if nl >= best[0] - 0.0005:
            print("plus d'amélioration", flush=True); break
        best = (nl, z); cur = seq
    print(f"FINAL {a.ep}: NLL/tok {best[0]:.4f} z {best[1]:+.1f} : {a.sep.join(cur)!r}", flush=True)


if __name__ == "__main__":
    main()
