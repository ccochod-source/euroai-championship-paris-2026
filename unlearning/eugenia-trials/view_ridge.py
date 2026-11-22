"""Fit a classifier-head update toward mean predictions across several views."""

import torch
import torch.nn.functional as F

from head import extract_features


def _resize_crop(images, top, left, extent):
    parts = []
    for batch in images.split(64):
        x = batch[:, top:top + extent, left:left + extent].permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=images.shape[1:3], mode="bilinear", align_corners=False)
        parts.append(x.round().byte().permute(0, 2, 3, 1).contiguous())
    return torch.cat(parts)


@torch.no_grad()
def fit(model, forget_images, *, strength=2.0, seed=2666, reg=0.5):
    """Fit normalized targets at scoring temperature two, using only forget images."""
    if strength <= 0 or reg <= 0:
        raise ValueError("strength and reg must be positive")
    torch.manual_seed(seed)
    head = model.head
    device = head.weight.device
    theta = torch.cat([head.weight.T, head.bias[None]], dim=0).double()
    width = theta.shape[0]
    features = []
    views = [
        (0, 0, 224, False),
        (0, 0, 224, True),
        (13, 13, 198, False),
        (4, 32, 186, True),
        (35, 4, 184, False),
    ]
    for top, left, extent, flip in views:
        images = forget_images if extent == 224 else _resize_crop(forget_images, top, left, extent)
        features.append(extract_features(model, images, batch_size=64, flip=flip))
        del images

    def design(feats):
        return torch.cat([feats, torch.ones((len(feats), 1), device=device)], dim=1).double()

    clean = design(features[0]) @ theta
    label = clean.argmax(dim=1, keepdim=True)
    distributions = []
    for feats in features:
        logits = design(feats) @ theta
        # Reduce the privileged class's logit while preserving per-view alternatives.
        logit_shift = torch.full_like(label, strength, dtype=logits.dtype)
        logits.scatter_add_(1, label, -logit_shift)
        distributions.append(F.softmax(logits / (2.0 + strength), dim=1))
    target = torch.stack(distributions).mean(dim=0).clamp_min(1e-8).log() * 2
    target -= target.mean(dim=1, keepdim=True)
    gram = torch.zeros((width, width), device=device, dtype=torch.float64)
    cross = torch.zeros_like(theta)
    for feats in features:
        a = design(feats)
        logit = a @ theta
        error = target - (logit - logit.mean(dim=1, keepdim=True))
        gram.addmm_(a.T, a)
        cross.addmm_(a.T, error)
    gram.diagonal().add_(reg * gram.diagonal().mean())
    update = torch.linalg.solve(gram, cross)
    print(f"[view_ridge] strength={strength} relative_head_update={update.norm() / theta.norm():.4f}")
    theta.add_(update)
    head.weight.copy_(theta[:-1].T)
    head.bias.copy_(theta[-1])
    return model
