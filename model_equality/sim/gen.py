"""Génère les réponses d'un FAUX jeu d'endpoints (vérité connue) avec le même protocole de prompts que l'équipe.

spec.json : {"endpoints": [{"ep": "ep-01", "family": "F1", "model": "F1_S", "base": "<hf id>",
             "weights": "bf16|fp16|fp8w|fp8wa|fp8ch|int8w|int4g128|bnb8|nf4|lora:<dir>:<force>",
             "system": null|"...", "temp": 0.7, "top_p": 1.0, "top_k": 0, "rep_pen": 1.0,
             "watermark": null|"kgw"|"synthid"}, ...]}
protocol.jsonl : {"prompt_id", "repeat", "prompt", "max_tokens"} par ligne.

    python gen.py --spec spec.json --protocol protocol.jsonl --out out/fake/set1 [--only-base <hf id>] [--reps 1]
Sortie : <out>/raw/<ep>.jsonl au format de data/raw (endpoint, prompt_id, repeat, prompt, response, ...).
"""
import argparse
import hashlib
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lik import TPL_KW, load_model as load_variant

SYNTHID_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
                118, 504, 421, 521, 129, 669, 732, 225, 90, 960]


def load_weights(base, weights):
    if weights.startswith("lora:"):
        from peft import PeftModel
        _, path, s = weights.split(":")
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda()
        model = PeftModel.from_pretrained(model, path)
        n = 0
        for m in model.modules():
            sc = getattr(m, "scaling", None)
            if isinstance(sc, dict) and "default" in sc:
                sc["default"] = sc["default"] * float(s)
                n += 1
        assert n > 0, "aucune couche LoRA trouvée"
        model = model.merge_and_unload()
        model.eval()
        return model
    return load_variant(base, weights)


def wm_config(kind):
    if not kind:
        return None
    from transformers import SynthIDTextWatermarkingConfig, WatermarkingConfig
    if kind == "kgw":
        return WatermarkingConfig(bias=2.0, greenlist_ratio=0.25, context_width=1, seeding_scheme="lefthash")
    if kind == "kgw_selfhash":
        return WatermarkingConfig(bias=2.5, greenlist_ratio=0.25, context_width=4, seeding_scheme="selfhash")
    if kind == "synthid":
        return SynthIDTextWatermarkingConfig(keys=SYNTHID_KEYS, ngram_len=5)
    raise ValueError(kind)


@torch.no_grad()
def gen_endpoint(model, tok, ep, items, bs=int(os.environ.get("SIM_BS", "96"))):
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    texts = []
    import random as _r
    rs = _r.Random(ep["ep"])
    for it in items:
        sysp = ep.get("system")
        if isinstance(sysp, list):          # persona tirée au hasard à chaque requête (comme ref-05)
            sysp = rs.choice(sysp)
        msgs = ([{"role": "system", "content": sysp}] if sysp else []) + \
               [{"role": "user", "content": it["prompt"]}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **TPL_KW))
    order = sorted(range(len(items)), key=lambda i: len(texts[i]))
    out = [None] * len(items)
    seed = int(hashlib.md5(ep["ep"].encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    wm = wm_config(ep.get("watermark"))
    for c in range(0, len(order), bs):
        idx = order[c:c + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        mx = max(items[i]["max_tokens"] for i in idx)
        kw = dict(do_sample=True, temperature=ep.get("temp", 0.7), top_p=ep.get("top_p", 1.0),
                  top_k=ep.get("top_k", 0) or None, repetition_penalty=ep.get("rep_pen", 1.0),
                  max_new_tokens=mx, pad_token_id=tok.pad_token_id)
        if wm is not None:
            kw["watermarking_config"] = wm
        g = model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:]
        for row, i in zip(g.tolist(), idx):
            row = row[:items[i]["max_tokens"]]
            cut = next((k for k, t in enumerate(row) if t in eos), None)
            fin = "stop" if cut is not None else "length"
            row = row[:cut] if cut is not None else row
            out[i] = (tok.decode(row, skip_special_tokens=True), fin, len(row))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-base", default=None)
    ap.add_argument("--reps", type=int, default=1, help="copies du protocole (repeat += 1000*k)")
    ap.add_argument("--only", default=None, help="sous-chaînes de noms d'endpoints à générer")
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding="utf-8"))
    proto = [json.loads(l) for l in open(a.protocol, encoding="utf-8") if l.strip()]
    items = [dict(it, repeat=it["repeat"] + 1000 * k) for k in range(a.reps) for it in proto]
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    eps = [e for e in spec["endpoints"] if a.only_base in (None, e["base"])]
    if a.only:
        eps = [e for e in eps if any(x in e["ep"] for x in a.only.split(","))]
    groups = {}
    for e in eps:
        groups.setdefault((e["base"], e["weights"]), []).append(e)
    for (base, weights), members in groups.items():
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(base)
        try:
            model = load_weights(base, weights)
        except Exception as e:  # adapter manquant, etc. : on passe au groupe suivant
            print(f"FAIL load {base} {weights}: {e!r}", flush=True)
            continue
        for e in members:
            dst = os.path.join(a.out, "raw", f"{e['ep']}.jsonl")
            if os.path.exists(dst):
                continue
            its = items
            if e.get("reps") or e.get("uniq"):   # réservoir : prompts uniques x reps (repeat = numéro d'échantillon)
                base_items = proto
                if e.get("uniq"):
                    seen, base_items = set(), []
                    for it in proto:
                        if it["prompt_id"] not in seen:
                            seen.add(it["prompt_id"]); base_items.append(dict(it, repeat=0))
                its = [dict(it, repeat=it["repeat"] + 1000 * k) for k in range(e.get("reps", 1)) for it in base_items]
            res = gen_endpoint(model, tok, e, its)
            with open(dst + ".tmp", "w", encoding="utf-8") as f:
                for it, (txt, fin, n) in zip(its, res):
                    f.write(json.dumps({"endpoint": e["ep"], "prompt_id": it["prompt_id"], "repeat": it["repeat"],
                                        "prompt": it["prompt"], "response": txt, "finish_reason": fin,
                                        "n_tokens": n, "usage": None, "ts": time.time()}, ensure_ascii=False) + "\n")
            os.replace(dst + ".tmp", dst)
            print(f"DONE {e['ep']} {e['model']} {weights} {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
