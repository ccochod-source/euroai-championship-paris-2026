"""targets stream: augmentation-derived soft prediction targets + lightweight fit.

Idea: a gold model never saw the forget images, so on them it behaves like on
unseen data: lower confidence, and its wrong-class mass goes to the classes the
original model *also* finds plausible under mild perturbations. We
  1. build several mild, plausible views of each forget image (flip, small
     resized-crop/translation, brightness/contrast jitter, light noise),
  2. average the original model's softmax over the views (geometric mean of
     probabilities = mean of log-probs, keeps the ranking/structure of the
     alternatives),
  3. temper the aggregate and demote the original top-1 class by `strength`,
  4. fit a small parameter subset (head + final norm [+ LayerScale gammas]) to
     those soft targets on fresh random views, with an L2 anchor to the
     original weights so the rest of the input space is preserved.

API:  fit(model, forget_images, *, strength=1.0, seed=0, **kw) -> model
      build_targets(model, forget_images, strength=..., ...) -> probs [N, C]
      sweep(...) / CLI for a fit-strength calibration sweep (proxy metrics only).

`strength` (positive float, default 1.0): how far targets move away from the
original model. Target logits = mean_view_logprob / (1 + 0.5*strength), then
top-1 logit -= strength * demote (demote=1.0 nat). strength -> 0 recovers the
view-averaged original. Candidate sweep: 0.25, 0.5, 1.0, 1.5, 2.0, 3.0.
Proxy metrics are NOT the official score (gold unavailable).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import torch
import torch.nn.functional as F

NUM_CLASSES = 100
SWEEP_STRENGTHS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


# ---------------------------------------------------------------- preprocessing
def preprocess(images_uint8: torch.Tensor, device) -> torch.Tensor:
    """Same as task_template.preprocess: uint8 NHWC -> fp32 NCHW in [-1, 1]."""
    x = images_uint8.to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


def _augment01(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Mild random view of x (float NCHW in [0,1]); per-sample params, stays in [0,1]."""
    b, _, h, w = x.shape
    dev = x.device
    r = lambda *s: torch.rand(*s, generator=g).to(dev)  # noqa: E731
    # horizontal flip (p=0.5)
    flip = r(b) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    # resized crop scale in [0.80, 1.0] + translation, via affine grid (bilinear)
    scale = 0.80 + 0.20 * r(b)
    s = scale.sqrt()
    tx = (r(b) * 2 - 1) * (1 - s) + (r(b) * 2 - 1) * 0.04
    ty = (r(b) * 2 - 1) * (1 - s) + (r(b) * 2 - 1) * 0.04
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # brightness / contrast / saturation jitter (+-10%)
    br = 1 + (r(b) * 2 - 1) * 0.10
    ct = 1 + (r(b) * 2 - 1) * 0.10
    sat = 1 + (r(b) * 2 - 1) * 0.10
    x = x * br[:, None, None, None]
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * ct[:, None, None, None] + m
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * sat[:, None, None, None] + gray
    # light gaussian noise (sigma up to 0.02)
    x = x + torch.randn(x.shape, generator=g).to(dev) * (0.02 * r(b))[:, None, None, None]
    # uint8 quantisation like a real image
    return (x.clamp(0, 1) * 255).round() / 255


def make_views(images_uint8: torch.Tensor, device, g: torch.Generator, identity: bool = False) -> torch.Tensor:
    """uint8 NHWC batch -> preprocessed fp32 NCHW batch of one mild random view."""
    x = images_uint8.to(device).permute(0, 3, 1, 2).float() / 255.0
    if not identity:
        x = _augment01(x, g)
    return (x - 0.5) / 0.5


def _autocast(device):
    dev = torch.device(device)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda")


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


# ---------------------------------------------------------------- targets
@torch.no_grad()
def view_logprobs(model, images_uint8, *, n_views=8, seed=0, batch_size=64, include_identity=True):
    """Returns log-softmax [V, N, C] of the model over (identity +) random mild views."""
    device = _device_of(model)
    model.eval()
    g = torch.Generator().manual_seed(seed)
    outs = []
    for v in range(n_views):
        ident = include_identity and v == 0
        chunks = []
        for i in range(0, len(images_uint8), batch_size):
            x = make_views(images_uint8[i:i + batch_size], device, g, identity=ident)
            with _autocast(device):
                chunks.append(F.log_softmax(model(x).float(), -1).cpu())
        outs.append(torch.cat(chunks))
    return torch.stack(outs)


def targets_from_logprobs(lp: torch.Tensor, strength: float, demote: float = 1.0,
                          temper: float = 0.5) -> torch.Tensor:
    """lp: [V, N, C] view log-probs -> target probs [N, C].

    mean over views of log-probs (geometric mean, keeps alternative ranking),
    divided by (1 + temper*strength) to soften, then the original top-1 class
    (of the aggregate) is demoted by strength*demote nats.
    """
    if strength <= 0:
        raise ValueError("strength must be > 0")
    z = lp.mean(0)
    z = z / (1.0 + temper * strength)
    top = z.argmax(-1)
    z = z.clone()
    z[torch.arange(len(z)), top] -= strength * demote
    return F.softmax(z, -1)


def build_targets(model, forget_images, *, strength=1.0, n_views=8, seed=0, demote=1.0,
                  temper=0.5, batch_size=64):
    lp = view_logprobs(model, forget_images, n_views=n_views, seed=seed, batch_size=batch_size)
    return targets_from_logprobs(lp, strength, demote=demote, temper=temper)


# ---------------------------------------------------------------- fit
def _select_params(model, trainable: str):
    names = []
    for n, p in model.named_parameters():
        ok = n.startswith("head.") or n.startswith("fc_norm.") or n.startswith("norm.")
        if trainable in ("head+ls", "head+ls+last") and (".ls1." in n or ".ls2." in n):
            ok = True
        if trainable == "head+ls+last" and n.startswith(f"blocks.{len(model.blocks) - 1}."):
            ok = True
        if trainable == "all":
            ok = True
        p.requires_grad_(ok)
        if ok:
            names.append(n)
    return names


def fit(model, forget_images, *, strength: float = 1.0, seed: int = 0, n_views: int = 8,
        epochs: int = 3, batch_size: int = 64, lr: float = 1e-3, anchor: float = 1.0,
        trainable: str = "head+ls", demote: float = 1.0, temper: float = 0.5,
        targets: torch.Tensor | None = None, verbose: bool = True):
    """Fit `model` (in place, also returned) to augmentation-derived soft targets.

    model: loaded timm ViT on its device. forget_images: uint8 [N,224,224,3] CPU.
    strength: >0, see module doc (default 1.0). seed: int, controls views/order.
    Loss per step: KL(target || p_model) on a fresh random view (T=1)
                   + anchor * mean squared distance of trained params to original.
    """
    if strength <= 0:
        raise ValueError("strength must be > 0")
    torch.manual_seed(seed)
    device = _device_of(model)
    if targets is None:
        targets = build_targets(model, forget_images, strength=strength, n_views=n_views,
                                seed=seed, demote=demote, temper=temper, batch_size=batch_size)
    names = _select_params(model, trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    ref = [p.detach().clone() for p in params]
    numel = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n = len(forget_images)
    steps = epochs * math.ceil(n / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed + 1)
    model.eval()  # no dropout / stochastic depth; ViT has no BN
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = make_views(forget_images[idx], device, g)
            q = targets[idx].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            logp = F.log_softmax(logits.float(), -1)
            kl = F.kl_div(logp, q, reduction="batchmean")
            reg = sum(((p - r) ** 2).sum() for p, r in zip(params, ref)) / numel
            loss = kl + anchor * 1e3 * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            tot += kl.item() * len(idx)
        if verbose:
            print(f"[targets.fit] epoch {ep + 1}/{epochs} KL={tot / n:.4f} "
                  f"trainable={len(names)} tensors/{numel} params t={time.time() - t0:.0f}s", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


# ---------------------------------------------------------------- calibration
@torch.no_grad()
def proxy_metrics(model, original, images, *, seed=123, batch_size=64, temperature=2.0):
    """Proxy stats (NOT the official score): CE wrt original argmax (mean/std),
    top-1 confidence, agreement with original, KL_T(original||model) on a held-out view."""
    device = _device_of(model)
    out = {}
    for name, ident in (("orig", True), ("view", False)):
        g = torch.Generator().manual_seed(seed)
        ce, conf, agree, kl = [], [], [], []
        for i in range(0, len(images), batch_size):
            x = make_views(images[i:i + batch_size], device, g, identity=ident)
            with _autocast(device):
                lm = model(x).float()
                lo = original(x).float()
            y = lo.argmax(-1)
            ce.append(F.cross_entropy(lm, y, reduction="none").cpu())
            conf.append(F.softmax(lm, -1).max(-1).values.cpu())
            agree.append((lm.argmax(-1) == y).float().cpu())
            po = F.softmax(lo / temperature, -1)
            kl.append((po * (po.clamp_min(1e-12).log() - F.log_softmax(lm / temperature, -1))).sum(-1).cpu())
        ce = torch.cat(ce)
        out[name] = dict(ce_mean=ce.mean().item(), ce_std=ce.std().item(),
                         conf=torch.cat(conf).mean().item(), agree=torch.cat(agree).mean().item(),
                         kl_T2_to_orig=torch.cat(kl).mean().item())
    return out


def sweep(model, forget_images, *, strengths=SWEEP_STRENGTHS, seed=0, holdout_frac=0.2, **fit_kw):
    """Fit-strength sweep. Fits on (1-holdout_frac) of the images, reports proxy
    metrics on the held-out part (unseen-copy proxy) and on the fitted part.
    Returns list of dicts. Copies the model per strength (needs 2 models in memory)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(forget_images), generator=g)
    k = max(1, int(len(forget_images) * holdout_frac))
    hold, train = forget_images[perm[:k]], forget_images[perm[k:]]
    original = copy.deepcopy(model).eval()
    for p in original.parameters():
        p.requires_grad_(False)
    lp = view_logprobs(original, train, n_views=fit_kw.get("n_views", 8), seed=seed,
                       batch_size=fit_kw.get("batch_size", 64))
    results = [dict(strength=0.0, train=proxy_metrics(original, original, train),
                    holdout=proxy_metrics(original, original, hold))]
    for s in strengths:
        m = copy.deepcopy(original)
        q = targets_from_logprobs(lp, s, demote=fit_kw.get("demote", 1.0), temper=fit_kw.get("temper", 0.5))
        fit(m, train, strength=s, seed=seed, targets=q, **fit_kw)
        results.append(dict(strength=s, target_conf=q.max(-1).values.mean().item(),
                            train=proxy_metrics(m, original, train),
                            holdout=proxy_metrics(m, original, hold)))
        print(json.dumps(results[-1]), flush=True)
        del m
    return results


# ---------------------------------------------------------------- CLI / selftest
def _tiny_vit(num_classes=NUM_CLASSES):
    import timm
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes,
                             init_values=1e-4, depth=2, embed_dim=48, num_heads=2)


def selftest(device="cpu", n=24):
    torch.manual_seed(0)
    model = _tiny_vit().to(device).eval()
    with torch.no_grad():  # make the random head peaky so targets are non-trivial
        model.head.weight.mul_(30)
    imgs = torch.randint(0, 256, (n, 224, 224, 3), dtype=torch.uint8)
    # preprocessing parity with task_template
    ref = (imgs.to(device).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
    assert torch.equal(make_views(imgs, device, torch.Generator(), identity=True), ref)
    assert torch.equal(preprocess(imgs, device), ref)
    lp = view_logprobs(model, imgs, n_views=3, batch_size=8)
    assert lp.shape == (3, n, NUM_CLASSES)
    q0 = targets_from_logprobs(lp, 1e-6)
    q = targets_from_logprobs(lp, 2.0)
    assert torch.allclose(q.sum(-1), torch.ones(n), atol=1e-5)
    assert (q.max(-1).values <= q0.max(-1).values + 1e-6).all(), "targets must soften"
    # alternative ranking preserved (excluding demoted top-1)
    z = lp.mean(0)
    top = z.argmax(-1)
    for i in range(n):
        mask = torch.ones(NUM_CLASSES, dtype=torch.bool)
        mask[top[i]] = False
        assert torch.equal(z[i][mask].argsort(), q[i][mask].argsort()), "alt ranking changed"
    before = {k: v.clone() for k, v in model.state_dict().items()}
    original = copy.deepcopy(model)
    m = fit(model, imgs, strength=2.0, seed=0, n_views=3, epochs=4, batch_size=8, lr=5e-3)
    after = m.state_dict()
    assert set(before) == set(after) and all(after[k].dtype == torch.float32 for k in after)
    assert all(torch.isfinite(v).all() for v in after.values())
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed and all(k.startswith(("head", "norm", "fc_norm")) or ".ls" in k for k in changed), changed
    pm0 = proxy_metrics(original, original, imgs, batch_size=8)
    pm = proxy_metrics(m, original, imgs, batch_size=8)
    print("before", json.dumps(pm0))
    print("after ", json.dumps(pm))
    assert pm["orig"]["conf"] < pm0["orig"]["conf"], "fit should lower confidence on forget images"
    res = sweep(copy.deepcopy(original), imgs, strengths=(0.5, 2.0), n_views=2, epochs=1, batch_size=8,
                verbose=False)
    assert len(res) == 3
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="tiny synthetic test (no data needed)")
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--images", default="forget_images.npz")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--sweep", default=None, help="comma list of strengths, e.g. 0.25,0.5,1,1.5,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-views", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--trainable", default="head+ls", choices=["head", "head+ls", "head+ls+last", "all"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="use first N images (quick runs)")
    ap.add_argument("--out", default="targets_model.pt", help="state dict output (local only, do not submit)")
    ap.add_argument("--report", default="targets_sweep.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.selftest:
        selftest(device)
        return
    from task_template import load_forget_images, load_model  # official loader, unmodified
    from pathlib import Path
    model = load_model(Path(a.model))
    imgs = load_forget_images(Path(a.images))
    if a.limit:
        imgs = imgs[:a.limit]
    kw = dict(n_views=a.n_views, epochs=a.epochs, lr=a.lr, anchor=a.anchor, trainable=a.trainable,
              batch_size=a.batch_size)
    if a.sweep:
        res = sweep(model, imgs, strengths=[float(s) for s in a.sweep.split(",")], seed=a.seed, **kw)
        Path(a.report).write_text(json.dumps(res, indent=2))
        print(f"wrote {a.report}")
    else:
        m = fit(model, imgs, strength=a.strength, seed=a.seed, **kw)
        torch.save({k: v.float().contiguous() for k, v in m.state_dict().items()}, a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
