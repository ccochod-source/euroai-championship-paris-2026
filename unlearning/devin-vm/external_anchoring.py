"""Test public-image retention anchors and JiT-style head smoothing."""

import argparse
from itertools import product
import json
from pathlib import Path

import torch

from checkpoint_changes import load
from compare_head_edits import metrics, performance
from compare_proposals import gpu_split_indices, load_head, reserve_gpu_split
from fit_cached_head import design, unpack_head
from selective_head import theta
from simulate_score import logits, retain_components
from tuning_loop import Search


def gram(features):
    return features.T @ features


def moments(directory):
    features = {**load(directory / "gpu-cache-a.pt"), **load(directory / "gpu-cache-b.pt")}
    count = len(features["clean"])
    excluded, _ = gpu_split_indices(count)
    keep = torch.ones(count, dtype=torch.bool)
    keep[excluded] = False
    for seed, size in ((2675, 192), (2775, 256)):
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
        keep[order[:size]] = False
    clean = design(features["clean"][keep])
    views = [design(features[name][keep]) for name in ("flip", "crop", "offset_left", "offset_right")]
    forget = gram(clean) + sum(gram(view) for view in views)
    variation = sum(gram(view - clean) for view in views)
    return forget, variation


def trace_normalize(matrix):
    return matrix / matrix.diagonal().mean()


def fit(search, directory, reference, control):
    forget, variation = map(trace_normalize, moments(directory))
    anchors = trace_normalize(gram(design(control)))
    base, prior = theta(search.original), theta(reference)
    eye = torch.eye(len(forget), dtype=forget.dtype)
    reg = 0.01 * eye
    for retain, smooth in product((0., 0.01, 0.03, 0.1, 0.3, 0.7),
                                  (0., 0.01, 0.04, 0.12)):
        matrix = forget + retain * anchors + smooth * variation + reg
        residual = -retain * (anchors @ (prior - base)) - smooth * (variation @ prior)
        update = torch.linalg.solve(matrix, residual)
        name = f"public-anchor{retain:g}-smooth{smooth:g}"
        yield name, 1.0, unpack_head(prior + update)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    directory = Path("loop")
    search = Search(directory)
    controls = load(directory / "tiny-imagenet-control-2.pt")
    split = torch.randperm(len(controls), generator=torch.Generator().manual_seed(193))
    train, test = controls[split[:300]], controls[split[300:]]
    reference = load_head(args.reference, search.original)
    baseline = [("record", 1.0, search.head("head", 1.1)), ("reference", 1.0, reference)]
    heads = baseline + list(fit(search, directory, reference, train))
    originals = logits(test, search.original)
    splits = [performance(f, search.original, heads) for f in (search.features, search.validation)]
    rows = []
    for name, _, head in heads:
        scored = [metrics(chunk[name]) for chunk in splits]
        control = retain_components(originals, logits(test, head))
        rows.append((name, min(row[0] for row in scored), min(row[2] for row in scored), control))
    ranked = sorted(rows[2:], key=lambda row: (min(row[1:3]), row[3]["retention"]), reverse=True)
    shortlist = {row[0] for row in ranked[:5]}
    final = performance(
        reserve_gpu_split(directory), search.original,
        baseline + [head for head in heads[2:] if head[0] in shortlist],
    )
    output = directory / "public-anchor"
    output.mkdir(exist_ok=True)
    for name, _, _, control in rows[:2] + ranked[:5]:
        scores = final[name]["proxy_scores"]
        print(f"{name} H2={scores[1]:.5f} H3={scores[2]:.5f} "
              f"public agreement={control['agreement']:.3f} KL={control['kl']:.4f}")
        if name in shortlist:
            torch.save(next(head for key, _, head in heads if key == name), output / f"{name}.pt")
    (output / "summary.json").write_text(json.dumps([
        {"name": name, "control": control, "forget_proxy_scores": final[name]["proxy_scores"]}
        for name, _, _, control in rows[:2] + ranked[:5]
    ], indent=2) + "\n")


if __name__ == "__main__":
    main()
