"""Log-vraisemblance des réponses déjà collectées (data/raw) sous des modèles locaux et des variantes de précision.

Pour chaque enregistrement (endpoint, prompt_id, repeat) et chaque température T de TEMPS, on stocke :
  nll[T] = -sum_t log p_T(y_t | contexte)     (somme sur les tokens de la réponse)
  H[T]   = sum_t entropie de p_T(. | contexte)  (espérance de la NLL si la réponse venait vraiment de p_T)
  V[T]   = sum_t variance de -log p_T           (pour le z-score d'adéquation)
plus des diagnostics d'échantillonnage à T=1 (rang max du token observé, masse au-dessus du token observé),
et la log-prob par token à T=1 (pour les tests de rapport de vraisemblance entre variantes).

Variantes : bf16, fp16, fp8w (poids fp8 par tenseur, q/k/v et gate/up fusionnés comme vLLM), fp8wa (+ activations fp8
dynamiques par tenseur, comme vLLM fp8 sur H100), fp8ch (poids fp8 par canal), int8w (RTN int8 par canal),
int4g128 (RTN int4 asymétrique, groupes de 128), bnb8 (LLM.int8), nf4 (bitsandbytes 4 bits).

    python lik.py --model Qwen/Qwen2.5-0.5B-Instruct --variants bf16,fp8w --records eps --raw data/raw --out out/lik
"""
import argparse
import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
FP8_MAX = 448.0
MAX_RESP_TOK = 320
# Le gabarit Llama-3.2 contient « Today Date: <jour> » : on fige la date (SIM_DATE="21 Nov 2026") pour que le calcul
# reste identique après minuit. Sans effet sur Qwen / SmolLM.
TPL_KW = {"date_string": os.environ["SIM_DATE"]} if os.environ.get("SIM_DATE") else {}
# SIM_SYSTEM_FILE : fichier texte d'un system prompt à placer avant chaque requête (vérifie un system prompt extrait).
SYS_MSG = ([{"role": "system", "content": open(os.environ["SIM_SYSTEM_FILE"], encoding="utf-8").read()}]
           if os.environ.get("SIM_SYSTEM_FILE") else [])
DEV = os.environ.get("SIM_DEVICE", "cuda")  # 'cpu' : calcul en float32 (poids arrondis bf16/fp8 identiques)


def fp8_rt(x, amax):
    scale = (amax.float() / FP8_MAX).clamp(min=1e-12)
    q = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).to(x.dtype)


def int8_rt(w):
    scale = (w.float().abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-12)
    return (torch.round(w.float() / scale).clamp(-127, 127) * scale).to(w.dtype)


def int4g_rt(w, g=128):
    o, i = w.shape
    if i % g:
        return w
    x = w.float().reshape(o, i // g, g)
    mn, mx = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15).clamp(min=1e-12)
    zp = torch.round(-mn / scale)
    q = (torch.round(x / scale) + zp).clamp(0, 15)
    return ((q - zp) * scale).reshape(o, i).to(w.dtype)


def linear_groups(model):
    groups = []
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        groups += [[a.q_proj, a.k_proj, a.v_proj], [a.o_proj], [m.gate_proj, m.up_proj], [m.down_proj]]
    return groups


def _act_hook(mod, args):
    x = args[0]
    return (fp8_rt(x, x.abs().amax()),) + tuple(args[1:])


@torch.no_grad()
def apply_variant(model, variant):
    if variant in ("bf16", "fp16", "bnb8", "nf4"):
        return
    for g in linear_groups(model):
        if variant in ("fp8w", "fp8wa"):
            amax = torch.stack([l.weight.abs().amax().float() for l in g]).max()
            for l in g:
                l.weight.copy_(fp8_rt(l.weight, amax))
            if variant == "fp8wa":
                for l in g:
                    l.register_forward_pre_hook(_act_hook)
        elif variant == "fp8ch":
            for l in g:
                l.weight.copy_(fp8_rt(l.weight, l.weight.abs().amax(dim=1, keepdim=True)))
        elif variant == "int8w":
            for l in g:
                l.weight.copy_(int8_rt(l.weight))
        elif variant == "int4g128":
            for l in g:
                l.weight.copy_(int4g_rt(l.weight))
        else:
            raise ValueError(variant)


def load_model(name, variant):
    if variant.startswith("lora:"):  # lora:<dossier adapter>:<force> -> adapter fusionné à la force donnée
        from peft import PeftModel
        _, path, sc = variant.split(":")
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV)
        model = PeftModel.from_pretrained(model, path)
        for m in model.modules():
            d = getattr(m, "scaling", None)
            if isinstance(d, dict) and "default" in d:
                d["default"] = d["default"] * float(sc)
        model = model.merge_and_unload().eval()
        return model.float() if DEV == "cpu" else model
    kw = dict(dtype=torch.float16 if variant == "fp16" else torch.bfloat16)
    if variant in ("bnb8", "nf4"):
        from transformers import BitsAndBytesConfig
        if variant == "bnb8":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if variant not in ("bnb8", "nf4"):
        model = model.to(DEV)
    model.eval()
    apply_variant(model, variant)
    if DEV == "cpu":
        model = model.float()
    return model


# Grille d'échantillonneurs tronqués (option --samplers) : température, puis top-k, puis top-p (ordre de vLLM / HF).
S_TEMPS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
S_TRUNC = [(0, 1.0), (20, 1.0), (40, 1.0), (50, 1.0), (100, 1.0), (0, 0.8), (0, 0.9), (0, 0.95), (0, 0.98), (50, 0.95),
           (50, 0.9), (40, 0.95), (40, 0.9), (20, 0.8)]


if os.environ.get("SIM_SAMP_SMALL"):          # grille réduite (10 fois plus rapide) pour le contrôle top-k
    S_TEMPS = [0.7, 0.8, 0.85, 0.9, 1.0]
    S_TRUNC = [(0, 1.0), (100, 1.0), (50, 1.0), (40, 1.0), (50, 0.95)]


def sampler_stats(logits, y):
    """{"T|k|p": [nll, H, V, n_viol]} : NLL des tokens observés sous l'échantillonneur tronqué (tokens hors support
    comptés dans n_viol et exclus de la NLL), entropie et variance attendues."""
    out = {}
    for T in S_TEMPS:
        lp = F.log_softmax(logits / T, -1)
        sp, si = lp.exp().sort(-1, descending=True)
        rank = (lp > lp.gather(1, y[:, None])).sum(-1)              # rang du token observé (0 = le plus probable)
        cum = sp.cumsum(-1)
        slp = sp.clamp_min(1e-30).log()
        for k, p in S_TRUNC:
            keep = torch.ones_like(sp, dtype=torch.bool)
            if k:
                keep[:, k:] = False
            if p < 1.0:
                pk = torch.where(keep, sp, torch.zeros_like(sp))
                pk = pk / pk.sum(-1, keepdim=True)
                c = pk.cumsum(-1)
                keep &= (c - pk) < p                                  # on garde le token qui fait franchir p
            q = torch.where(keep, sp, torch.zeros_like(sp))
            Z = q.sum(-1, keepdim=True)
            q = q / Z
            lq = torch.where(keep, slp - Z.log(), torch.zeros_like(sp))
            H = -(q * lq).sum(-1)
            V = ((q * lq * lq).sum(-1) - H * H).clamp(min=0)
            nkeep = keep.sum(-1)
            ok = rank < nkeep
            ly = lq.gather(1, rank.clamp(max=sp.shape[1] - 1)[:, None])[:, 0]
            out[f"{T}|{k}|{p}"] = [round(-(ly[ok]).sum().item(), 3), round(H[ok].sum().item(), 3),
                                   round(V[ok].sum().item(), 3), int((~ok).sum().item())]
    return out


SAMPLERS = False


def load_records(raw_dir, which, eps=None, block0=False):
    recs = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        ep = os.path.basename(p)[:-6]
        if which == "refs" and not ep.startswith("ref-"):
            continue
        if which == "eps" and not ep.startswith("ep-"):
            continue
        if eps and ep not in eps:
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if block0 and int(r.get("repeat", 0)) >= 1000:
                    continue
                if r.get("response"):
                    recs.append({k: r.get(k) for k in ("endpoint", "prompt_id", "repeat", "prompt", "response")})
    return recs


@torch.no_grad()
def score(model, tok, recs, log_every=200):
    out = []
    t0 = time.time()
    for k, r in enumerate(recs):
        ctx = tok.apply_chat_template(SYS_MSG + [{"role": "user", "content": r["prompt"]}], tokenize=False,
                                      add_generation_prompt=True, **TPL_KW)
        if os.environ.get("SIM_LLAMA_NODATE") and SYS_MSG:   # variante de gabarit Llama : system prompt seul, sans les lignes de date
            ctx = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYS_MSG[0]["content"] + "<|eot_id|>"
                   "<|start_header_id|>user<|end_header_id|>\n\n" + r["prompt"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        ic = tok(ctx, add_special_tokens=False).input_ids
        ir = tok(r["response"], add_special_tokens=False).input_ids[:MAX_RESP_TOK]
        if not ir:
            continue
        ids = torch.tensor([ic + ir], device=DEV)
        logits = model(ids).logits[0, len(ic) - 1:-1].float()
        y = ids[0, len(ic):]
        rec = {"endpoint": r["endpoint"], "prompt_id": r["prompt_id"], "repeat": r["repeat"], "n": len(ir),
               "nll": [], "H": [], "V": []}
        for T in TEMPS:
            lp = F.log_softmax(logits / T, -1)
            p = lp.exp()
            ly = lp.gather(1, y[:, None])[:, 0]
            H = -(p * lp).sum(-1)
            E2 = (p * lp * lp).sum(-1)
            rec["nll"].append(round(-ly.sum().item(), 4))
            rec["H"].append(round(H.sum().item(), 4))
            rec["V"].append(round((E2 - H * H).clamp(min=0).sum().item(), 4))
            if T == 1.0:
                above = (p * (lp > ly[:, None])).sum(-1)       # masse des tokens strictement plus probables
                rank = (lp > ly[:, None]).sum(-1)
                rec["lp1"] = [round(v, 4) for v in ly.tolist()]
                rec["max_rank"] = int(rank.max().item())
                rec["max_above"] = round(above.max().item(), 5)
                rec["n_rank_gt50"] = int((rank >= 50).sum().item())
        if SAMPLERS:
            rec["samp"] = sampler_stats(logits, y)
            for kdrop in ("lp1",):
                rec.pop(kdrop, None)
        out.append(rec)
        if log_every and (k + 1) % log_every == 0:
            print(f"  {k + 1}/{len(recs)} {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default="bf16")
    ap.add_argument("--records", default="all", choices=["all", "refs", "eps"])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="out/lik")
    ap.add_argument("--eps", default=None, help="liste d'endpoints séparés par des virgules")
    ap.add_argument("--block0", action="store_true", help="faux jeu : seulement repeat < 1000")
    ap.add_argument("--tag", default="", help="suffixe du fichier de sortie")
    ap.add_argument("--samplers", action="store_true", help="ajoute les statistiques d'échantillonneurs tronqués")
    a = ap.parse_args()
    global SAMPLERS
    SAMPLERS = a.samplers
    os.makedirs(a.out, exist_ok=True)
    recs = load_records(a.raw, a.records, set(a.eps.split(",")) if a.eps else None, a.block0)
    tok = AutoTokenizer.from_pretrained(a.model)
    for v in a.variants.split(","):
        vname = v.replace("lora:out/lora/", "lora-").replace(":", "-").replace("/", "-")
        dst = os.path.join(a.out, f"{a.model.replace('/', '__')}__{vname}__{a.records}{a.tag}.jsonl")
        if os.path.exists(dst) and not os.environ.get("SIM_FORCE"):
            print(f"SKIP {dst} (existe déjà)", flush=True)
            continue
        t0 = time.time()
        try:
            model = load_model(a.model, v)
            res = score(model, tok, recs)
        except Exception as e:  # une variante qui casse ne doit pas arrêter les autres
            print(f"FAIL {a.model} {v}: {e!r}", flush=True)
            continue
        with open(dst + ".tmp", "w", encoding="utf-8") as f:
            for rec in res:
                f.write(json.dumps(rec) + "\n")
        os.replace(dst + ".tmp", dst)
        print(f"DONE {a.model} {v} {a.records} n={len(res)} {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
