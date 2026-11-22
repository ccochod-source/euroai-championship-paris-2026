"""Conception des sondes : signal attendu par requête pour distinguer une variante (fp8, int8, nf4, LoRA) du modèle bf16.

Pour chaque prompt candidat : on génère N réponses avec le bf16 (T, top_p de l'endpoint), puis, le long de chaque réponse,
on calcule à chaque position (distributions à la température T) :
  sKL_t = KL(p_v || p_b) + KL(p_b || p_v)  -> écart attendu de LLR par token entre un endpoint « variante » et un « S »
  var_t = Var_{y~p_b}[log p_v(y) - log p_b(y)] -> bruit par token
Signal par requête = sum_t sKL_t ; bruit = sqrt(sum_t var_t) ; SNR par requête = signal / bruit.
z attendu avec k requêtes ≈ SNR * sqrt(k).

    python kl_design.py --model Qwen/Qwen2.5-1.5B-Instruct --variants fp8w,fp8wa,int8w,nf4,lora:out/lora/q15_alpaca:0.3 \
        --prompts design_prompts.jsonl --n 2 --temp 0.7 --out out/design/q15.jsonl
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen import load_weights


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = load_weights(a.model, "bf16")
    prompts = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
    eos = base.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    torch.manual_seed(0)
    # 1) réponses du bf16
    seqs = []  # (prompt_id, ctx_ids, resp_ids)
    texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    for k in range(0, len(prompts), 32):
        chunk = list(range(k, min(k + 32, len(prompts))))
        enc = tok([texts[i] for i in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        mx = max(prompts[i].get("max_tokens", 256) for i in chunk)
        g = base.generate(**enc, do_sample=True, temperature=a.temp, top_p=a.top_p, top_k=None,
                          max_new_tokens=mx, num_return_sequences=a.n, pad_token_id=tok.pad_token_id)
        g = g[:, enc["input_ids"].shape[1]:].tolist()
        for j, row in enumerate(g):
            i = chunk[j // a.n]
            row = row[:prompts[i].get("max_tokens", 256)]
            cut = next((q for q, t in enumerate(row) if t in eos), None)
            row = row[:cut] if cut is not None else row
            ic = tok(texts[i], add_special_tokens=False).input_ids
            if row:
                seqs.append((prompts[i]["pid"], ic, row))

    res = {}
    for v in a.variants.split(","):
        m = load_weights(a.model, v)
        for pid, ic, row in seqs:
            ids = torch.tensor([ic + row], device="cuda")
            xb = base(ids).logits[0, len(ic) - 1:-1].float()
            xv = m(ids).logits[0, len(ic) - 1:-1].float()
            pb = F.log_softmax(xb / a.temp, -1)
            pv = F.log_softmax(xv / a.temp, -1)
            Pb, Pv = pb.exp(), pv.exp()
            d = pv - pb
            skl = ((Pv - Pb) * d).sum(-1)                       # KL(v||b) + KL(b||v)
            mu = (Pb * d).sum(-1)
            var = (Pb * d * d).sum(-1) - mu * mu
            r = res.setdefault((pid, v), {"pid": pid, "variant": v, "n_resp": 0, "tok": 0, "skl": 0.0, "var": 0.0})
            r["n_resp"] += 1
            r["tok"] += len(row)
            r["skl"] += skl.sum().item()
            r["var"] += var.clamp(min=0).sum().item()
        del m
        torch.cuda.empty_cache()
        print(f"DONE {v}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res.values():
            n = r["n_resp"]
            r["tok_per_req"] = r["tok"] / n
            r["signal_per_req"] = r["skl"] / n
            r["snr_per_req"] = (r["skl"] / n) / max((r["var"] / n) ** 0.5, 1e-9)
            f.write(json.dumps(r) + "\n")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
