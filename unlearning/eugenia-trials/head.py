"""Head stream: fast targeted classifier-head update for ViT unlearning.

Only `model.head` (Linear 768 -> 100) is changed; the backbone stays frozen,
so retention is controlled by keeping the head change small.

Targets (class-conditional soft targets, no labels needed):
  z_i      original logits of forget image i, c_i = argmax z_i
  p_i      softmax(z_i / tau), tau = 1 + strength            (flatter = higher loss, like an unseen image)
  m_c      mean of p_j over forget images with c_j = c, after damping the top-1 entry
           (captures WHICH wrong classes a model that never saw image i tends to pick)
  q_i      (1 - beta) * p_i + beta * m_{c_i},  beta = min(0.5, 0.5 * strength)

Fit (default mode="ridge", closed form in the 769-dim [feature, 1] space):
  min_dW  ||[F,1] (W0 + dW) - Y||^2 + lam * ||dW||^2,  Y = row-centred log q,
  lam = reg * trace(A^T A) / 769  (scale-free). Logits are compared row-centred
  (softmax is shift invariant). Larger `reg` -> smaller parameter change.
mode="kl": a few Adam steps on KL(q || softmax(head(f))) + l2 * ||dW||^2.

strength: positive float, default 1.0. 0 -> no change. Sweep: 0.25, 0.5, 1, 2, 4.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F


def _preprocess(images_uint8: torch.Tensor, device) -> torch.Tensor:
    # identical to task_template.preprocess
    x = images_uint8.to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


@torch.no_grad()
def extract_features(model, images_uint8, batch_size=128, flip=False):
    """Pre-logit features [N, D] (fp32) using the same fp16 autocast as the server on CUDA."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(images_uint8), batch_size):
        x = _preprocess(images_uint8[i:i + batch_size], device)
        if flip:
            x = x.flip(-1)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.forward_head(model.forward_features(x), pre_logits=True)
        out.append(f.float())
    return torch.cat(out)


def soft_targets(logits, strength, top1_damp=0.5, eps=1e-6):
    tau = 1.0 + strength
    beta = min(0.5, 0.5 * strength)
    p = torch.softmax(logits / tau, dim=1)
    c = logits.argmax(1)
    n_cls = logits.shape[1]
    damped = p.clone()
    damped[torch.arange(len(p)), c] *= top1_damp
    damped = damped / damped.sum(1, keepdim=True)
    m = torch.zeros(n_cls, n_cls, device=p.device).index_add_(0, c, damped)
    cnt = torch.bincount(c, minlength=n_cls).clamp_min(1).unsqueeze(1).to(p.dtype)
    m = m / cnt
    q = (1 - beta) * p + beta * m[c]
    return q / q.sum(1, keepdim=True).clamp_min(eps)


def _centre(x):
    return x - x.mean(1, keepdim=True)


def fit(model, forget_images, *, strength: float = 1.0, seed: int = 0,
        mode: str = "ridge", reg: float = 1.0, flip_aug: bool = True,
        kl_steps: int = 200, kl_lr: float = 1e-3, kl_l2: float = 1e-2,
        batch_size: int = 128, verbose: bool = True):
    """Return a copy of `model` whose classifier head is updated toward unlearning targets."""
    if strength < 0:
        raise ValueError("strength must be >= 0")
    torch.manual_seed(seed)
    model = copy.deepcopy(model).eval()
    if strength == 0:
        return model
    head = model.head
    device = head.weight.device
    W0 = head.weight.detach().float().clone()   # [C, D]
    b0 = head.bias.detach().float().clone()     # [C]

    feats = extract_features(model, forget_images, batch_size)
    if flip_aug:
        feats = torch.cat([feats, extract_features(model, forget_images, batch_size, flip=True)])
    logits0 = feats @ W0.T + b0
    # targets come from the un-flipped originals, shared by both views
    n = len(forget_images)
    q = soft_targets(logits0[:n], strength)
    if flip_aug:
        q = torch.cat([q, q])
    Y = _centre(torch.log(q.clamp_min(1e-6)))

    if mode == "ridge":
        A = torch.cat([feats, torch.ones(len(feats), 1, device=device)], 1).double()
        Theta0 = torch.cat([W0.T, b0[None]], 0).double()           # [D+1, C]
        R = Y.double() - _centre(A @ Theta0)                        # residual in centred logit space
        G = A.T @ A
        lam = reg * G.diagonal().mean()
        dTheta = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=device, dtype=G.dtype), A.T @ R)
        dTheta = dTheta - dTheta.mean(1, keepdim=True)              # no-op per-class shift removal
        dW, db = dTheta[:-1].T.float(), dTheta[-1].float()
    elif mode == "kl":
        dW = torch.zeros_like(W0, requires_grad=True)
        db = torch.zeros_like(b0, requires_grad=True)
        opt = torch.optim.Adam([dW, db], lr=kl_lr)
        g = torch.Generator(device="cpu").manual_seed(seed)
        bs = min(512, len(feats))
        for _ in range(kl_steps):
            idx = torch.randint(0, len(feats), (bs,), generator=g).to(device)
            out = feats[idx] @ (W0 + dW).T + (b0 + db)
            loss = F.kl_div(F.log_softmax(out, 1), q[idx], reduction="batchmean") + kl_l2 * dW.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        dW, db = dW.detach(), db.detach()
    else:
        raise ValueError(f"unknown mode {mode}")

    with torch.no_grad():
        head.weight.copy_((W0 + dW).to(head.weight.dtype))
        head.bias.copy_((b0 + db).to(head.bias.dtype))
    if verbose:
        with torch.no_grad():
            new = feats[:n] @ head.weight.float().T + head.bias.float()
            c0 = logits0[:n].argmax(1)
            ce = lambda lg: F.cross_entropy(lg, c0, reduction="none")
            print(f"[head] mode={mode} strength={strength} reg={reg} "
                  f"|dW|/|W0|={dW.norm() / W0.norm():.4f} "
                  f"forget top1 kept={(new.argmax(1) == logits0[:n].argmax(1)).float().mean():.3f} "
                  f"CE(orig top1) {ce(logits0[:n]).mean():.3f}->{ce(new).mean():.3f} "
                  f"KL(q||new)={F.kl_div(F.log_softmax(new, 1), q[:n], reduction='batchmean'):.4f}")
    return model


def unlearn(model, forget_images):
    """Drop-in for task_template.unlearn (copy into your own script; do not edit the template)."""
    return fit(model, forget_images, strength=1.0, seed=0)
