"""Gradient stream: bounded fine-tune of the last ViT block(s) + head for unlearning.

fit(model, forget_images, *, strength=1.0, seed=0, **kw) -> model (modified in place, returned in eval mode)

Objective (minibatch, gradient-clipped, only last `blocks` blocks + final norm + head trainable):
  forget  : KL(q || p_student) on augmented forget images, where q = softmax(z_orig / T_s) with
            T_s = 1 + strength. Softening the frozen original's logits lowers pseudo-label confidence
            (raises forget CE) while keeping the original's ranking of wrong classes ("which wrong class").
            A hinge additionally pushes p(pseudo-label) below cap = 1 / (1 + strength) only while it is
            above it -> bounded, never unconstrained ascent.
  preserve: KL(p_orig^T2 || p_student^T2) on proxy "retain" inputs (mixup of two different forget
            images + noise), approximating the hidden-control KL term of the retain score.
  drift   : drift_weight * ||theta - theta_0||^2 over trainable params (L2-SP).
Preprocessing matches task_template.preprocess: uint8 NHWC -> fp32 NCHW in [-1, 1].

strength: positive float. 0.25 mild ... 4 aggressive. Default 1.0. Sweep: [0.25, 0.5, 1.0, 2.0, 3.0].
"""
import copy
import math

import torch
import torch.nn.functional as F


def _preprocess(x_uint8, device):
    x = x_uint8.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


def _augment(x, gen):
    # x: [B,3,H,W] in [-1,1]; random hflip + small translation (covers mild transformed copies)
    b = x.shape[0]
    flip = torch.rand(b, generator=gen) < 0.5
    x = torch.where(flip.to(x.device)[:, None, None, None], x.flip(3), x)
    dx, dy = (torch.randint(-8, 9, (2,), generator=gen)).tolist()
    x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    return x


def _proxy_retain(x, gen):
    b = x.shape[0]
    perm = torch.roll(torch.arange(b), 1)
    lam = torch.distributions.Beta(2.0, 2.0).sample((b,)).to(x.device)[:, None, None, None]
    noise = torch.randn(x.shape, generator=gen).to(x.device) * 0.1
    return (lam * x + (1 - lam) * x[perm] + noise).clamp(-1, 1)


def _set_trainable(model, blocks):
    for p in model.parameters():
        p.requires_grad_(False)
    mods = []
    if blocks > 0 and hasattr(model, "blocks"):
        mods += list(model.blocks[-blocks:])
    for name in ("norm", "fc_norm", "head"):
        m = getattr(model, name, None)
        if isinstance(m, torch.nn.Module):
            mods.append(m)
    params = [p for m in mods for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    return params


def fit(model, forget_images, *, strength=1.0, seed=0, blocks=2, epochs=3, batch_size=64, lr=1e-5,
        drift_weight=1.0, preserve_weight=1.0, hinge_weight=1.0, max_grad_norm=1.0, amp=True, verbose=True):
    if not strength > 0:
        raise ValueError("strength must be a positive float")
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    params = _set_trainable(model, blocks)
    ref = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    use_amp = amp and device.type == "cuda"
    T_s, cap, T_r = 1.0 + strength, 1.0 / (1.0 + strength), 2.0

    n = len(forget_images)
    steps = epochs * math.ceil(n / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    # Dropout/drop-path off for stable targets; ViT has no BatchNorm so eval() is safe for training.
    model.eval()
    step = 0
    for ep in range(epochs):
        order = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            if len(idx) < 2:
                continue
            x = _augment(_preprocess(forget_images[idx], device), gen)
            xr = _proxy_retain(x, gen)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                with torch.no_grad():
                    zt, zrt = teacher(x).float(), teacher(xr).float()
                zs, zrs = model(x).float(), model(xr).float()
            q = F.softmax(zt / T_s, 1)
            l_forget = F.kl_div(F.log_softmax(zs, 1), q, reduction="batchmean")
            pseudo = zt.argmax(1)
            p_top = F.softmax(zs, 1).gather(1, pseudo[:, None]).squeeze(1)
            l_hinge = F.relu(p_top - cap).pow(2).mean()
            l_pres = F.kl_div(F.log_softmax(zrs / T_r, 1), F.softmax(zrt / T_r, 1), reduction="batchmean")
            l_drift = sum((p - r).pow(2).sum() for p, r in zip(params, ref))
            loss = l_forget + hinge_weight * l_hinge + preserve_weight * l_pres + drift_weight * l_drift
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            sched.step()
            step += 1
            if verbose and (step % 20 == 0 or step == 1):
                print(f"[gradient] ep{ep} step{step}/{steps} loss={loss.item():.4f} fgt={l_forget.item():.4f} "
                      f"hinge={l_hinge.item():.4f} pres={l_pres.item():.4f} drift={l_drift.item():.2e} "
                      f"ptop={p_top.mean().item():.3f}")
    for p in model.parameters():
        p.requires_grad_(True)
    del teacher
    return model.eval()


SAFE_DEFAULTS = dict(strength=1.0, seed=0, blocks=2, epochs=3, batch_size=64, lr=1e-5, drift_weight=1.0)
SWEEP = dict(strength=[0.25, 0.5, 1.0, 2.0, 3.0], blocks=[1, 2, 4], lr=[5e-6, 1e-5, 3e-5])
