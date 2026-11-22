"""Fit and evaluate multi-view distillation heads from cached ViT features."""

import itertools
import json
from pathlib import Path

import torch

from checkpoint_changes import load
from simulate_score import evaluate_heads, get_features
from tuning_loop import Search


def design(features):
    features = features.double()
    return torch.cat((features, features.new_ones((len(features), 1))), dim=1)


def center(values):
    return values - values.mean(-1, keepdim=True)


def unpack_head(theta):
    return {
        "head.weight": theta[:-1].T.float().contiguous(),
        "head.bias": theta[-1].float().contiguous(),
    }


def fit_heads(features, original, withheld):
    theta = torch.cat((original["head.weight"].T, original["head.bias"][None])).double()
    n = len(features["clean"])
    train = torch.ones(n, dtype=torch.bool)
    train[withheld] = False
    control_partners = torch.randperm(n, generator=torch.Generator().manual_seed(2666))
    controls = design(features["mixed"][train & train[control_partners]])
    control_gram = controls.T @ controls
    views = [design(features[key][train]) for key in ("clean", "flip", "crop", "offset_left", "offset_right")]
    original_logits = torch.stack([center(view @ theta) for view in views])
    gram = sum(view.T @ view for view in views)
    design_sum_t = sum(views).T.contiguous()
    cross_original = sum(view.T @ logit for view, logit in zip(views, original_logits))
    ridge_scale = gram.diagonal().mean()
    eye = torch.eye(len(gram), dtype=gram.dtype)
    labels = original_logits[0].argmax(-1)
    top = torch.nn.functional.one_hot(labels, original_logits.shape[-1])
    rows = []
    for temperature, demotion in itertools.product((2.5, 3.25, 4.0, 4.75), (0.0, 1.0, 2.5)):
        target_prob = ((original_logits - demotion * top) / temperature).softmax(-1).mean(0)
        target = center(target_prob.clamp_min(1e-8).log() * 2)
        cross = design_sum_t @ target - cross_original
        for anchor_weight, ridge in itertools.product((0.1, 0.5, 1.5, 4.0), (0.05, 0.5)):
            update = torch.linalg.solve(gram + anchor_weight * control_gram + ridge * ridge_scale * eye, cross)
            for scale in (0.9, 1.0, 1.1):
                name = f"distill-t{temperature:g}-d{demotion:g}-a{anchor_weight:g}-r{ridge:g}"
                rows.append((name, scale, unpack_head(theta + scale * update)))
    return rows


def main():
    folder = Path("loop")
    features = {**load(folder / "gpu-cache-a.pt"), **load(folder / "gpu-cache-b.pt")}
    setup = Search(folder)
    # Both cached evaluation samples are selected by seeded permutations of
    # the original forget-set order. Exclude their union from the ridge fit.
    held_out = torch.cat([
        torch.randperm(len(features["clean"]), generator=torch.Generator().manual_seed(seed))[:count]
        for seed, count in ((2675, 192), (2775, 256))
    ]).unique()
    heads = fit_heads(features, setup.original, held_out)
    independent = get_features(folder / "simulator-second-split.pt", 256, 2775)
    tune = evaluate_heads(setup.features, setup.original, heads)
    val = evaluate_heads(independent, setup.original, heads)
    tune_by_key = {(r["family"], r["strength"]): r for r in tune}
    weights = {(name, scale): head for name, scale, head in heads}
    paired = [
        (r, tune_by_key[(r["family"], r["strength"])]) for r in val
    ]
    paired.sort(key=lambda pair: (min(r["synthetic_control_agreement"] for r in pair) >= 0.70,
                                 min(r["median_proxy"] for r in pair)), reverse=True)
    out = folder / "distilled"
    out.mkdir(exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(paired, indent=2) + "\n")
    for r, train in paired[:5]:
        name, scale = r["family"], r["strength"]
        path = out / f"{name}-{scale:g}.pt"
        torch.save(weights[(name, scale)], path)
        print(f"{path}: tune={train['median_proxy']:.4f} "
              f"validation={r['median_proxy']:.4f} retain={r['synthetic_control_agreement']:.3f}", flush=True)
    print(f"{len(heads)} candidates, {len(held_out)} held-out images", flush=True)


if __name__ == "__main__":
    main()
