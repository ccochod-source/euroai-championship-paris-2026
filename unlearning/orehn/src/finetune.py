"""Full fine-tuning: distil forget images (+views) to 'never seen' T2 targets, KL~0 to original on proxies, L2-SP."""
import argparse, copy, json, time, numpy as np
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--A", type=float, default=0.7); ap.add_argument("--alpha", type=float, default=4.0)
ap.add_argument("--steps", type=int, default=800); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--gamma", type=float, default=2.0, help="weight of retain KL"); ap.add_argument("--l2sp", type=float, default=0.0)
ap.add_argument("--holdout", type=int, default=-1, help="fold of forget images left out of training (diagnostic)")
ap.add_argument("--aug", type=float, default=1.0); ap.add_argument("--train", default="all", help="all | last<k> | head")
ap.add_argument("--rpow", type=float, default=1.0); ap.add_argument("--save", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--t1", type=float, default=0.0, help="weight of an extra KL at T=1 toward p1 ~ q^2 (forces confident errors)")
ap.add_argument("--targets", default="", help="optional .pt with precomputed q [N,100]")
a = ap.parse_args(); print(vars(a), flush=True)
torch.manual_seed(a.seed); dev = "cuda"
T = torch.load(C + "/teacher.pt"); y = T["y"]; N = len(y)
q = torch.load(a.targets) if a.targets else make_targets(T, a.A, a.alpha, rpow=a.rpow)
imgs = kit.load_forget_images()
prox = torch.from_numpy(np.load(ROOT + "/orehn/data/aigen16k.npz")["images"]); zprox = torch.load(C + "/proxy_logits.pt")
PTR, PTE = torch.arange(0, 14000), torch.arange(14000, 16000)
tr_idx = torch.arange(N) if a.holdout < 0 else (T["fold"] != a.holdout).nonzero().flatten()
ho_idx = None if a.holdout < 0 else (T["fold"] == a.holdout).nonzero().flatten()
orig = kit.load_model(); model = copy.deepcopy(orig).train()   # no dropout / droppath in this arch at default -> train()==eval()
for p in orig.parameters(): p.requires_grad_(False)
if a.train != "all":
    for n_, p in model.named_parameters():
        ok = n_.startswith("head") or n_.startswith("norm.")
        if a.train.startswith("last"): ok = ok or any(n_.startswith(f"blocks.{b}.") for b in range(12 - int(a.train[4:]), 12))
        p.requires_grad_(ok)
params = [p for p in model.parameters() if p.requires_grad]; p0 = [p.detach().clone() for p in params]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 40) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
scaler = torch.amp.GradScaler(); g = torch.Generator(device=dev).manual_seed(a.seed)
qd = q.to(dev); yd = y.to(dev)
p1d = qd.pow(2); p1d = p1d / p1d.sum(1, keepdim=True)

def kl2(logq_or_q, z, is_logits=False):
    """KL(target || softmax(z/2)) per sample; target given as probs or (if is_logits) logits at T=2"""
    ls = (z.float() / 2).log_softmax(1)
    tq = (logq_or_q.float() / 2).softmax(1) if is_logits else logq_or_q
    return (tq * (tq.clamp_min(1e-9).log() - ls)).sum(1)

@torch.no_grad()
def logits(m, images, idx, fn=None, bs=125):
    out = []
    for i in range(0, len(idx), bs):
        x = kit.preprocess(images[idx[i:i+bs]])
        if fn is not None: x = fn(x)
        with torch.autocast("cuda", dtype=torch.float16): out.append(m(x).float())
    return torch.cat(out)

EVAL_VIEWS = {"clean": None, "flip": lambda x: x.flip(-1), "crop": lambda x: F.interpolate(x[..., 13:-13, 13:-13], size=224, mode="bilinear"),
              "bright": lambda x: (x * 0.8 + 0.1), "noise.2": lambda x: x + 0.2 * torch.randn(x.shape, device=x.device, generator=torch.Generator(device=x.device).manual_seed(5)),
              "jpegish": lambda x: blur(x, 160)}
@torch.no_grad()
def evaluate(step):
    model.eval(); res = {"step": step}
    for split, idx in (("tr", tr_idx), ("ho", ho_idx)):
        if idx is None: continue
        sub = idx if split == "ho" else idx[:: max(1, len(idx) // 1000)]
        for vn, fn in EVAL_VIEWS.items():
            z = logits(model, imgs, sub, fn); zo = logits(orig, imgs, sub, fn)
            L = F.cross_entropy(z, yd[sub], reduction="none")
            res[f"{split}/{vn}"] = dict(acc=round((z.argmax(1) == yd[sub]).float().mean().item(), 4), Lmean=round(L.mean().item(), 3), Lstd=round(L.std().item(), 3),
                                        KLq=round(kl2(qd[sub], z).mean().item(), 4), KLq_orig=round(kl2(qd[sub], zo).mean().item(), 3), Lorig=round(F.cross_entropy(zo, yd[sub]).item(), 3))
    z = logits(model, prox, PTE); zo = zprox[PTE].to(dev)
    klr = kl2(zo, z, True).mean().item(); agr = (z.argmax(1) == zo.argmax(1)).float().mean().item()
    hc = zo.softmax(1).max(1).values > 0.9
    res["proxy"] = dict(KL=round(klr, 4), agree=round(agr, 4), retain_est=round(agr / (1 + klr), 4), agree_conf=round((z.argmax(1) == zo.argmax(1))[hc].float().mean().item(), 4), KL_conf=round(kl2(zo, z, True)[hc].mean().item(), 4))
    d2 = sum((p - p_).pow(2).sum() for p, p_ in zip(params, p0)).sqrt().item(); n0 = sum(p_.pow(2).sum() for p_ in p0).sqrt().item()
    res["rel_dw"] = round(d2 / n0, 5)
    print(json.dumps(res), flush=True); model.train(); return res

evaluate(0); t0 = time.time()
for step in range(1, a.steps + 1):
    bi = tr_idx[torch.randint(len(tr_idx), (a.bs,))]; pi = PTR[torch.randint(len(PTR), (a.bs,))]
    xf = train_aug(kit.preprocess(imgs[bi]), g, a.aug); xp = kit.preprocess(prox[pi])
    flip = torch.rand(a.bs, device=dev, generator=g) < 0.5; xp = torch.where(flip.view(-1, 1, 1, 1), xp.flip(-1), xp)
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad(): zo = orig(xp)
        z = model(torch.cat([xf, xp]))
    lf = kl2(qd[bi], z[:a.bs]).mean(); lr_ = kl2(zo, z[a.bs:], True).mean()
    if a.t1 > 0: lf = lf + a.t1 * (p1d[bi] * (p1d[bi].clamp_min(1e-12).log() - z[:a.bs].float().log_softmax(1))).sum(1).mean()
    reg = sum((p - p_).pow(2).sum() for p, p_ in zip(params, p0)) if a.l2sp > 0 else torch.zeros((), device=dev)
    loss = lf + a.gamma * lr_ + a.l2sp * reg
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step()
    if step % 50 == 0: print(f"step {step} lf={lf.item():.4f} lret={lr_.item():.4f} reg={reg.item():.4f} {time.time()-t0:.0f}s", flush=True)
    if step % 200 == 0 or step == a.steps: res = evaluate(step)
if a.save:
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values())
    out = f"{ROOT}/orehn/ckpt/{a.name}.pt"
    for attempt in range(40):          # the project disk quota is intermittently "exceeded": retry instead of losing the run
        try:
            torch.save(sd, out + ".tmp"); os.replace(out + ".tmp", out); break
        except Exception as e:
            print("save failed, retry in 30s:", repr(e)[:200], flush=True); time.sleep(30)
            if attempt >= 2:          # scratch quota still exceeded: fall back to the project filesystem
                os.makedirs("/p/project1/training2666/ansellem1_unl", exist_ok=True); out = f"/p/project1/training2666/ansellem1_unl/{a.name}.pt"
    chk = kit.build_model(); chk.load_state_dict(torch.load(out, map_location="cpu", weights_only=True), strict=True)
    print("saved", out, os.path.getsize(out) / 1e6, "MB", flush=True)
try: json.dump({"args": vars(a), "final": res}, open(f"{ROOT}/orehn/logs/{a.name}.json", "w"))
except Exception as e: print("json log failed", e)
