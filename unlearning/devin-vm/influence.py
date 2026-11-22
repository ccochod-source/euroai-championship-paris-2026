"""Influence stream: one-shot approximate influence removal on a small parameter subset.

Approximation
-------------
Removing the forget set F from training moves the optimum by (first-order influence / Newton step)
    theta_gold - theta  ~=  (1/n) H^{-1} sum_{i in F} grad l_i(theta)
i.e. a curvature-preconditioned *ascent* step on the forget loss. We cannot use labels, the retain set or the
full Hessian, so:
  * labels  : pseudo-labels = original model's soft predictions at temperature `label_temp` (default: hard argmax);
  * H       : damped diagonal Fisher (F_diag + damping * mean(F_diag)) estimated on the forget images with labels
              sampled from the model's own predictive distribution (true Fisher; needs no labels);
  * 1/n     : unknown -> replaced by a trust region: each parameter tensor's update is rescaled to
              ||delta_t|| = strength * base_rel_norm * ||theta_t||.
  * params  : only the classifier head, final norm and LayerScale gammas (+ optionally norms) of the last
              `last_blocks` blocks are changed (~81.5k trainable params on ViT-B/16 by default); everything else is frozen,
              backward only runs through those blocks.
Because the update is a single linear step, `strength` is also an interpolation coefficient:
theta(s) = theta + s * delta. Use `influence_direction` once and `apply_delta` per sweep value for cheap sweeps.

When it may hurt retention
--------------------------
  * strength too large: ascent on pseudo-labels lowers confidence on the forget classes globally; the forget set
    covers all 100 classes, so head changes leak to control images (accuracy gate is 70%).
  * damping too small: directions with tiny Fisher (rarely used features) get amplified -> erratic control errors.
  * forget images that the original misclassifies: ascent on a wrong pseudo-label can move mass toward the true class.
  * iterating (`steps` > 1) compounds the leak; prefer steps=1 and a sweep over strength.
  * fp16 autocast at eval: very small LayerScale deltas are fine, but avoid huge head norms (strength >> 4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_STRENGTH = 1.0
SWEEP_STRENGTHS = [0.25, 0.5, 1.0, 2.0, 4.0]   # with base_rel_norm=0.01 -> 0.25%..4% relative change per tensor


def preprocess(images_uint8: torch.Tensor, device) -> torch.Tensor:
    """Same as task_template.preprocess: uint8 NHWC -> fp32 NCHW in [-1, 1]."""
    x = images_uint8.to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


def select_params(model: torch.nn.Module, last_blocks: int = 2, include_block_norms: bool = False):
    n_blocks = len(model.blocks)
    keep = []
    for name, p in model.named_parameters():
        if name.startswith(("head.", "fc_norm.", "norm.")):
            keep.append((name, p))
            continue
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            if idx >= n_blocks - last_blocks:
                if ".ls1." in name or ".ls2." in name or (include_block_norms and ".norm" in name):
                    keep.append((name, p))
    return keep


def _batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


def influence_direction(model, forget_images, *, seed=0, last_blocks=2, include_block_norms=False,
                        label_temp=None, damping=0.1, base_rel_norm=0.01, batch_size=64,
                        fisher_micro_batch=8, max_images=None):
    """Returns {param_name: delta} for strength=1 (delta is a forget-loss ascent direction)."""
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    imgs = forget_images
    if max_images is not None and len(imgs) > max_images:
        imgs = imgs[torch.randperm(len(imgs), generator=g)[:max_images]]
    n = len(imgs)

    params = select_params(model, last_blocks, include_block_norms)
    was_training = model.training
    model.eval()
    req = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for _, p in params:
        p.requires_grad_(True)

    grad = {name: torch.zeros_like(p) for name, p in params}
    fisher = {name: torch.zeros_like(p) for name, p in params}
    gen_dev = torch.Generator(device=device).manual_seed(seed)
    try:
        for sl in _batches(n, batch_size):
            x = preprocess(imgs[sl], device)
            # forget-loss gradient with pseudo-labels from the (current = original) model
            logits = model(x).float()
            with torch.no_grad():
                if label_temp is None:
                    target = F.one_hot(logits.argmax(1), logits.shape[1]).float()
                else:
                    target = F.softmax(logits / label_temp, 1)
            loss = -(target * F.log_softmax(logits, 1)).sum(1).sum() / n
            gs = torch.autograd.grad(loss, [p for _, p in params])
            for (name, _), gi in zip(params, gs):
                grad[name] += gi
            # diagonal true Fisher via micro-batches with sampled labels (micro-batch approximation of per-sample)
            for msl in _batches(len(x), fisher_micro_batch):
                lm = model(x[msl]).float()
                with torch.no_grad():
                    y = torch.multinomial(F.softmax(lm, 1), 1, generator=gen_dev).squeeze(1)
                m = lm.shape[0]
                lf = F.cross_entropy(lm, y, reduction="mean")
                gf = torch.autograd.grad(lf, [p for _, p in params])
                for (name, _), gi in zip(params, gf):
                    fisher[name] += (gi ** 2) * m / n
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(req[name])
        model.train(was_training)

    delta = {}
    for name, p in params:
        f = fisher[name]
        d = grad[name] / (f + damping * f.mean().clamp_min(1e-12))
        dn = d.norm()
        if not torch.isfinite(dn) or dn == 0:
            continue
        delta[name] = d * (base_rel_norm * p.detach().norm().clamp_min(1e-8) / dn)
    return delta


@torch.no_grad()
def apply_delta(model, delta, strength):
    named = dict(model.named_parameters())
    for name, d in delta.items():
        named[name].add_(d.to(named[name].dtype), alpha=float(strength))
    return model


def fit(model, forget_images, *, strength: float = DEFAULT_STRENGTH, seed: int = 0, steps: int = 1, **kw):
    """Influence-style unlearning. `strength` (>0): trust-region scale; per-tensor relative change is
    strength * base_rel_norm (default 1%). `steps` > 1 re-estimates the direction and splits strength."""
    if strength <= 0:
        raise ValueError("strength must be > 0")
    torch.manual_seed(seed)
    for s in range(steps):
        delta = influence_direction(model, forget_images, seed=seed + s, **kw)
        apply_delta(model, delta, strength / steps)
    return model.eval()


if __name__ == "__main__":  # tiny synthetic smoke test (random-init tiny ViT, CPU)
    import timm
    torch.manual_seed(0)
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=100, init_values=1e-4,
                          depth=2, embed_dim=64, num_heads=2).eval()
    imgs = torch.randint(0, 256, (16, 224, 224, 3), dtype=torch.uint8)
    x = preprocess(imgs, "cpu")
    with torch.no_grad():
        l0 = m(x)
    y = l0.argmax(1)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    n_tr = sum(p.numel() for _, p in select_params(m))
    fit(m, imgs, strength=1.0, seed=0, batch_size=8, fisher_micro_batch=4)
    with torch.no_grad():
        l1 = m(x)
    changed = [k for k in before if not torch.equal(before[k], m.state_dict()[k])]
    print("trainable params:", n_tr, "changed tensors:", changed)
    print("forget CE before %.4f after %.4f" % (F.cross_entropy(l0, y).item(), F.cross_entropy(l1, y).item()))
    assert all(torch.isfinite(v).all() for v in m.state_dict().values())
    assert F.cross_entropy(l1, y) > F.cross_entropy(l0, y)
    print("OK")
