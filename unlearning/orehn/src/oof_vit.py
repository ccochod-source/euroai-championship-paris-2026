"""Out-of-fold teacher with the SAME architecture: re-initialise the last blocks + head, retrain them on 4/5 of the
forget images (pseudo-labels, label smoothing) + proxies (soft labels of the original, keeps the 100 classes alive),
predict the held-out 1/5. Blocks below are frozen (original weights)."""
import argparse, copy, numpy as np
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--first", type=int, default=9); ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--name", default="oofvit9")
a = ap.parse_args(); print(vars(a), flush=True); dev = "cuda"
T = torch.load(C + "/teacher.pt"); y = T["y"].to(dev); N = len(y)
imgs = kit.load_forget_images(); prox = torch.from_numpy(np.load(ROOT + "/orehn/data/aigen16k.npz")["images"][:8000]); zprox = torch.load(C + "/proxy_logits.pt")[:8000].to(dev)
orig = kit.load_model()
for p in orig.parameters(): p.requires_grad_(False)
@torch.no_grad()
def lower(x):
    with torch.autocast("cuda", dtype=torch.float16):
        h = orig.patch_embed(x); h = orig._pos_embed(h); h = orig.norm_pre(h); return orig.blocks[:a.first](h).float()
def fresh():
    m = kit.build_model().to(dev)          # timm default init = new random last blocks / norm / head
    top = torch.nn.ModuleDict({"blocks": m.blocks[a.first:], "norm": m.norm, "head": m.head}); return m, top
Z = torch.zeros(a.seeds, N, 100); bs = 64
for seed in range(a.seeds):
    g = torch.Generator().manual_seed(100 + seed); perm = torch.randperm(N, generator=g); fold = torch.empty(N, dtype=torch.long); fold[perm] = torch.arange(N) % 5
    gg = torch.Generator(device=dev).manual_seed(seed)
    for k in range(5):
        torch.manual_seed(seed * 10 + k); m, top = fresh(); params = list(top.parameters())
        tr = (fold != k).nonzero().flatten(); te = (fold == k).nonzero().flatten(); steps = a.epochs * len(tr) // bs
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler(); m.train()
        for step in range(steps):
            bi = tr[torch.randint(len(tr), (bs,))]; pi = torch.randint(len(prox), (bs,))
            xf = train_aug(kit.preprocess(imgs[bi]), gg, 1.0); xp = kit.preprocess(prox[pi])
            h = lower(torch.cat([xf, xp]))
            with torch.autocast("cuda", dtype=torch.float16):
                z = m.forward_head(m.norm(m.blocks[a.first:](h))).float()
            lf = F.cross_entropy(z[:bs], y[bi], label_smoothing=0.1)
            lp = (zprox[pi].softmax(1) * (zprox[pi].log_softmax(1) - z[bs:].log_softmax(1))).sum(1).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(lf + lp).backward(); scaler.step(opt); scaler.update(); sched.step()
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(te), 125):
                h = lower(kit.preprocess(imgs[te[i:i+125]]))
                with torch.autocast("cuda", dtype=torch.float16): out.append(m.forward_head(m.norm(m.blocks[a.first:](h))).float().cpu())
            zt = torch.cat(out); Z[seed, te] = zt
            ztr = []
            for i in range(0, 500, 125):
                h = lower(kit.preprocess(imgs[tr[i:i+125]]))
                with torch.autocast("cuda", dtype=torch.float16): ztr.append(m.forward_head(m.norm(m.blocks[a.first:](h))).float().cpu())
        yt = y[te].cpu(); L = F.cross_entropy(zt, yt, reduction="none")
        print(f"seed {seed} fold {k}: held-out acc {(zt.argmax(1)==yt).float().mean():.3f} CE {L.mean():.3f}/{L.std():.3f} | train acc {(torch.cat(ztr).argmax(1)==y[tr[:500]].cpu()).float().mean():.3f} last lf {lf.item():.3f} lp {lp.item():.3f}", flush=True)
yc = y.cpu(); P = Z.softmax(-1).mean(0); L = -P[torch.arange(N), yc].clamp_min(1e-9).log(); present = torch.bincount(yc, minlength=100) > 0
print(f"ENSEMBLE: acc {(P.argmax(1)==yc).float().mean():.3f} CE {L.mean():.3f}/{L.std():.3f} mass on 55 other classes {P[:, ~present].sum(1).mean():.3f}")
d = torch.load(ROOT + "/codex/cache/oof_targets.pt", map_location="cpu"); Pd = d["oof"]["clean"].float(); Pd = Pd.softmax(1) if Pd.sum(1).abs().max() < 0.5 or Pd.min() < 0 else Pd
print("vs DINOv2 OOF (raw): same argmax", (P.argmax(1) == Pd.argmax(1)).float().mean().item(), "| both wrong & same wrong class", ((P.argmax(1) != yc) & (Pd.argmax(1) != yc) & (P.argmax(1) == Pd.argmax(1))).float().mean().item(), "| vit wrong", (P.argmax(1) != yc).float().mean().item(), "dino wrong", (Pd.argmax(1) != yc).float().mean().item(),
      "| corr of log p_y", torch.corrcoef(torch.stack([P[torch.arange(N), yc].clamp_min(1e-6).log(), Pd[torch.arange(N), yc].clamp_min(1e-6).log()]))[0, 1].item())
torch.save({"logits": Z, "P": P, "y": yc}, f"{C}/{a.name}.pt"); print("saved")
