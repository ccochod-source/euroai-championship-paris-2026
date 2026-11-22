"""« Distiller puis tester » : détecte les endpoints qui partagent le même adapter LoRA (à des forces différentes).

1. On entraîne un petit LoRA sur les réponses d'un endpoint X (prompts de l'ensemble A = la moitié des prompt_id).
2. Sur les prompts de l'ensemble B (jamais vus à l'entraînement), pour chaque endpoint Y de la famille :
   LLR(Y) = moyenne par token de log p_distillé(y) - log p_base(y)   (T = 1)
   Même adapter que X ⇒ LLR > 0 (le décalage appris se généralise) ; modèle S ⇒ LLR ≈ 0 ou < 0.
On fait aussi l'inverse (entraînement sur B, test sur A) et on moyenne.

    python distill.py --model HuggingFaceTB/SmolLM2-360M-Instruct --raw data/raw --train-ep ref-07 \
        --eval-eps ref-01,...,ref-09 --out out/distill/ref07.json
"""
import argparse
import hashlib
import json
import math
import os
import random

import torch
DEV = os.environ.get("SIM_DEVICE", "cuda")
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


PREFIX = None


def load(raw, ep):
    out = []
    p = os.path.join(raw, f"{ep}.jsonl")
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if PREFIX and not r["prompt_id"].startswith(PREFIX):
                continue
            if r.get("response") and int(r.get("repeat", 0)) < 1000:
                out.append(r)
    return out


def split_of(pid, seed):
    return int(hashlib.md5(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) % 2


def encode(tok, prompt, resp, max_len=768):
    ctx = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ic = tok(ctx, add_special_tokens=False).input_ids
    ir = tok(resp, add_special_tokens=False).input_ids[:320]
    return ic, ir


@torch.no_grad()
def seq_lp(model, ic, ir):
    ids = torch.tensor([ic + ir], device=DEV)
    lg = model(ids).logits[0, len(ic) - 1:-1].float()
    return F.log_softmax(lg, -1).gather(1, ids[0, len(ic):, None])[:, 0]


def run_fold(a, tok, recs_x, evals, fold):
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV)
    if DEV == "cpu":
        model = model.float()
    train = [r for r in recs_x if split_of(r["prompt_id"], a.seed) == fold]
    ex = [encode(tok, r["prompt"], r["response"]) for r in train]
    ex = [(ic, ir) for ic, ir in ex if ir]
    # log-probs de base sur l'ensemble de test (avant d'ajouter le LoRA)
    test = {ep: [r for r in rs if split_of(r["prompt_id"], a.seed) != fold] for ep, rs in evals.items()}
    enc = {ep: [encode(tok, r["prompt"], r["response"]) for r in rs] for ep, rs in test.items()}
    model.eval()
    base_lp = {ep: [seq_lp(model, ic, ir) if ir else None for ic, ir in v] for ep, v in enc.items()}
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    rng = random.Random(a.seed + fold)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for step in range(a.steps):
        batch = rng.sample(ex, min(a.bs, len(ex)))
        L = max(len(ic) + len(ir) for ic, ir in batch)
        ids = torch.tensor([ic + ir + [pad] * (L - len(ic) - len(ir)) for ic, ir in batch]).to(DEV)
        lab = torch.tensor([[-100] * len(ic) + ir + [-100] * (L - len(ic) - len(ir)) for ic, ir in batch]).to(DEV)
        att = torch.tensor([[1] * (len(ic) + len(ir)) + [0] * (L - len(ic) - len(ir)) for ic, ir in batch]).to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.eval()
    res = {}
    mods = [m for m in model.modules() if isinstance(getattr(m, "scaling", None), dict) and "default" in m.scaling]
    s0 = [m.scaling["default"] for m in mods]
    for alpha in a.alphas:                       # adapter évalué à force réduite (direction du premier ordre)
        for m, s in zip(mods, s0):
            m.scaling["default"] = s * alpha
        for ep, v in enc.items():
            vals, ns = [], []
            for (ic, ir), bl in zip(v, base_lp[ep]):
                if not ir:
                    continue
                d = (seq_lp(model, ic, ir) - bl).sum().item()
                vals.append(d)
                ns.append(len(ir))
            res[(ep, alpha)] = (vals, ns)
    del model
    torch.cuda.empty_cache()
    return res, len(ex), loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--train-ep", required=True)
    ap.add_argument("--eval-eps", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--prefix", default=None, help="ne garder que ces préfixes de prompt_id (séparés par des virgules)")
    a = ap.parse_args()
    a.alphas = [float(x) for x in a.alphas.split(",")]
    global PREFIX
    PREFIX = tuple(a.prefix.split(",")) if a.prefix else None
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    recs_x = load(a.raw, a.train_ep)
    evals = {ep: load(a.raw, ep) for ep in a.eval_eps.split(",")}
    tot = {(ep, al): ([], []) for ep in evals for al in a.alphas}
    info = []
    for fold in (0, 1):
        res, n_ex, last = run_fold(a, tok, recs_x, evals, fold)
        info.append({"fold": fold, "n_train": n_ex, "last_loss": round(last, 3)})
        for k, (v, n) in res.items():
            tot[k][0].extend(v)
            tot[k][1].extend(n)
    out = {"train_ep": a.train_ep, "model": a.model, "info": info, "steps": a.steps, "llr": {}}
    for (ep, al), (v, n) in tot.items():
        if not n:
            continue
        m = sum(v) / sum(n)
        res = [x - m * k for x, k in zip(v, n)]
        se = math.sqrt(sum(r * r for r in res)) / sum(n) if len(v) > 1 else float("nan")
        out["llr"][f"{ep}@{al}"] = {"llr_x1e3": round(m * 1000, 2), "z": round(m / se, 2) if se == se and se > 0 else None,
                                    "n_tok": sum(n), "n_rec": len(v)}
    for al in a.alphas:
        print(f"train={a.train_ep} steps={a.steps} alpha={al}: " + " ".join(
            f"{ep[-2:]}:{out['llr'][f'{ep}@{al}']['llr_x1e3']:+.0f}" for ep in evals if f"{ep}@{al}" in out["llr"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
