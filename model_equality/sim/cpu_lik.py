"""Vraisemblance sur le CPU du VPS (GPU coupé) : score les réponses déjà collectées, sans rien générer.
Modèle chargé couche par couche depuis le disque (accelerate, device_map disk) : ~1,5 Go de RAM.
Variantes : bf16 (référence, = fp16 à 1e-3 près), fp8w (poids des couches linéaires -> float8_e4m3fn -> retour, une
échelle par groupe q/k/v, o, gate/up, down comme vLLM), int8w (fake-quant int8 par canal de sortie).
Les variantes sont des copies du fichier safetensors modifiées en place (aucune copie du modèle en RAM).
Sortie : même format que lik.py (nll / H / V par température, lp1), une ligne par réponse, écrite au fil de l'eau.

    python cpu_lik.py --model unsloth/Llama-3.2-1B-Instruct --variant bf16 --eps ep-19,ep-13 --prefix v2b_ --out out/lik_cpu2
"""
import argparse, glob, json, os, re, shutil, struct, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("SIM_THREADS", "6")))
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
FP8_MAX = 448.0
MAX_RESP_TOK = 320
TPL_KW = {"date_string": os.environ.get("SIM_DATE", "21 Nov 2026")}
PAT = re.compile(r"layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$")
GROUP = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "o", "gate_proj": "gu", "up_proj": "gu", "down_proj": "d"}


def snapshot_dir(name):
    d = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--" + name.replace("/", "--"), "snapshots", "*"))
    return d[0]


def make_variant(name, variant):
    """Copie le snapshot et quantifie en place les poids linéaires dans les fichiers safetensors."""
    src = snapshot_dir(name)
    dst = os.path.join("variants", name.replace("/", "__") + "__" + variant)
    if os.path.exists(os.path.join(dst, "DONE")):
        return dst
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))          # suit les liens symboliques du cache
    for p in glob.glob(os.path.join(dst, "*.safetensors")):
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        base = 8 + n
        mm = np.memmap(p, dtype=np.uint8, mode="r+")
        def get(k):
            m = head[k]; a, b = m["data_offsets"]
            dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[m["dtype"]]
            return torch.frombuffer(bytearray(mm[base + a:base + b]), dtype=dt).reshape(m["shape"]), dt
        def put(k, t, dt):
            a, b = head[k]["data_offsets"]
            mm[base + a:base + b] = np.frombuffer(t.to(dt).contiguous().view(torch.uint8 if False else dt).numpy().tobytes() if dt != torch.bfloat16
                                                  else t.to(dt).contiguous().view(torch.int16).numpy().tobytes(), dtype=np.uint8)
        keys = [k for k in head if k != "__metadata__" and PAT.search(k)]
        if variant == "fp8w":
            groups = {}
            for k in keys:
                m = PAT.search(k); groups.setdefault((m.group(1), GROUP[m.group(3) or m.group(4)]), []).append(k)
            for g, ks in groups.items():
                ws = {k: get(k) for k in ks}
                amax = max(w.float().abs().max() for w, _ in ws.values())
                scale = (amax / FP8_MAX).clamp(min=1e-12)
                for k, (w, dt) in ws.items():
                    q = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
                    put(k, q, dt)
        elif variant == "int8w":
            for k in keys:
                w, dt = get(k)
                scale = (w.float().abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-12)
                put(k, torch.round(w.float() / scale).clamp(-127, 127) * scale, dt)
        else:
            raise ValueError(variant)
        mm.flush(); del mm
        print(f"variante {variant} : {len(keys)} matrices quantifiées dans {os.path.basename(p)}", flush=True)
    open(os.path.join(dst, "DONE"), "w").close()
    return dst


def load_records(raw, eps, prefix, done):
    recs = []
    for ep in eps:
        rs = []
        for l in open(os.path.join(raw, f"{ep}.jsonl"), encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("response") and (not prefix or r["prompt_id"].startswith(prefix)) and (ep, r["prompt_id"], r["repeat"]) not in done:
                    rs.append(r)
        recs += sorted(rs, key=lambda r: len(r["prompt"]) + len(r["response"]))   # lots de longueurs voisines
    return recs


@torch.no_grad()
def score_batch(model, tok, batch):
    enc = []
    for r in batch:
        ctx = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, **TPL_KW)
        ic = tok(ctx, add_special_tokens=False).input_ids
        ir = tok(r["response"], add_special_tokens=False).input_ids[:MAX_RESP_TOK]
        enc.append((ic, ir))
    L = max(len(a) + len(b) for a, b in enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([a + b + [pad] * (L - len(a) - len(b)) for a, b in enc])
    att = torch.tensor([[1] * (len(a) + len(b)) + [0] * (L - len(a) - len(b)) for a, b in enc])
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state      # remplissage à droite : sans effet (causal)
    out = []
    for i, (r, (ic, ir)) in enumerate(zip(batch, enc)):
        if not ir:
            continue
        y = torch.tensor(ir)
        rec = {"endpoint": r["endpoint"], "prompt_id": r["prompt_id"], "repeat": r["repeat"], "n": len(ir),
               "nll": [0.0] * len(TEMPS), "H": [0.0] * len(TEMPS), "V": [0.0] * len(TEMPS), "lp1": [], "max_rank": 0,
               "n_rank_ge50": 0, "max_above": 0.0}
        CH = 48                                   # par tranches : les logits (tokens x vocabulaire) restent < 100 Mo
        for c0 in range(0, len(ir), CH):
            logits = model.lm_head(h[i:i + 1, len(ic) - 1 + c0:len(ic) - 1 + min(c0 + CH, len(ir))])[0].float()
            yc = y[c0:c0 + CH]
            for j, T in enumerate(TEMPS):
                lp = F.log_softmax(logits / T, -1)
                ly = lp.gather(1, yc[:, None])[:, 0]
                p = lp.exp()
                H = -(p * lp).sum(-1)
                E2 = (p * lp.square()).sum(-1)
                rec["nll"][j] += -ly.sum().item(); rec["H"][j] += H.sum().item()
                rec["V"][j] += (E2 - H * H).clamp(min=0).sum().item()
                if T == 1.0:
                    gt = lp > ly[:, None]
                    rank = gt.sum(-1)
                    rec["lp1"] += [round(v, 4) for v in ly.tolist()]
                    rec["max_rank"] = max(rec["max_rank"], int(rank.max().item()))
                    rec["n_rank_ge50"] += int((rank >= 50).sum().item())
                    rec["max_above"] = max(rec["max_above"], round((p * gt).sum(-1).max().item(), 5))
                del lp, p, H, E2
            del logits
        for k in ("nll", "H", "V"):
            rec[k] = [round(v, 4) for v in rec[k]]
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="bf16")
    ap.add_argument("--eps", required=True)
    ap.add_argument("--prefix", default="v2b_")
    ap.add_argument("--raw", default="../data/raw")
    ap.add_argument("--out", default="out/lik_cpu2")
    ap.add_argument("--bs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, f"{a.model.replace('/', '__')}__{a.variant}__eps_v2b.jsonl")
    done = set()
    if os.path.exists(dst):
        for l in open(dst):
            r = json.loads(l); done.add((r["endpoint"], r["prompt_id"], r["repeat"]))
    recs = load_records(a.raw, a.eps.split(","), a.prefix, done)
    print(f"{len(recs)} réponses à scorer ({len(done)} déjà faites)", flush=True)
    path = snapshot_dir(a.model) if a.variant == "bf16" else make_variant(a.model, a.variant)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, device_map={"": "disk"}, offload_folder=os.path.join("off", a.variant)).eval()
    t0 = time.time(); ntok = 0
    with open(dst, "a") as f:
        for c in range(0, len(recs), a.bs):
            batch = recs[c:c + a.bs]
            for rec in score_batch(model, tok, batch):
                f.write(json.dumps(rec) + "\n"); ntok += rec["n"]
            f.flush()
            print(f"  {min(c + a.bs, len(recs))}/{len(recs)} {batch[-1]['endpoint']} {time.time() - t0:.0f}s {ntok} tokens", flush=True)


if __name__ == "__main__":
    main()
