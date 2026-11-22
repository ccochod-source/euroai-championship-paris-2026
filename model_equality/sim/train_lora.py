"""Entraîne rapidement un adapter LoRA (quelques minutes sur A100) pour simuler les variantes LoRA du sujet.

    python train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --data alpaca --steps 300 --out out/lora/q05_alpaca
"""
import argparse
import glob
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = os.path.join(os.environ.get("HF_HOME", ""), "hub")
DATA = {
    "alpaca": "datasets--yahma--alpaca-cleaned/snapshots/*/alpaca_data_cleaned.json",
    "med": "datasets--medalpaca--medical_meadow_medical_flashcards/snapshots/*/*.json",
}


def load_pairs(name, n, seed=0):
    path = glob.glob(os.path.join(HUB, DATA[name]))[0]
    rows = json.load(open(path, encoding="utf-8"))
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        q = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        a = r["output"]
        if 10 < len(a) < 1500:
            out.append((q, a))
        if len(out) >= n:
            break
    return out


def encode(tok, q, a, max_len):
    ctx = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ic = tok(ctx, add_special_tokens=False).input_ids
    ia = tok(a, add_special_tokens=False).input_ids + [tok.eos_token_id]
    ids = (ic + ia)[:max_len]
    labels = ([-100] * len(ic) + ia)[:max_len]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="alpaca")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro", type=int, default=4, help="micro-batch (accumulation) pour tenir en mémoire")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    pairs = load_pairs(a.data, a.steps * a.bs, a.seed)
    ex = [encode(tok, q, ans, a.max_len) for q, ans in pairs]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / a.steps))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    t0 = time.time()
    for step in range(a.steps):
        batch = ex[step * a.bs:(step + 1) * a.bs]
        L = max(len(i) for i, _ in batch)
        ids = torch.tensor([i + [pad] * (L - len(i)) for i, _ in batch]).cuda()
        lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in batch]).cuda()
        att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in batch]).cuda()
        mb = a.micro
        for k in range(0, len(batch), mb):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids[k:k + mb], attention_mask=att[k:k + mb], labels=lab[k:k + mb]).loss
            (loss * min(mb, len(batch) - k) / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(a.out)
    print(f"SAVED {a.out}", flush=True)


if __name__ == "__main__":
    main()
