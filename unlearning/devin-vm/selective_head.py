"""Improve retention of a head correction with synthetic-control anchoring."""

from itertools import product
import json
from pathlib import Path

import torch

from checkpoint_changes import load
from compare_proposals import gpu_split_indices
from fit_cached_head import design, unpack_head
from simulate_score import evaluate_heads, logits
from tuning_loop import Search, candidate_key


def theta(head):
    return torch.cat([head["head.weight"].T, head["head.bias"][None]]).double()


def fit_selective_heads(features, original, reference, keep):
    num_examples = len(features["clean"])
    partners = torch.randperm(num_examples, generator=torch.Generator().manual_seed(2666))
    controls = features["mixed"][keep & keep[partners]]
    # Strong predictions on mixed images supply more useful anchors than ties.
    confidence = (logits(controls, original) / 2).softmax(-1).max(-1).values
    controls = design(controls) * confidence.sqrt().unsqueeze(-1)
    control_gram = controls.T @ controls
    forget_gram = torch.zeros_like(control_gram)
    for key in ("clean", "flip", "crop", "offset_left", "offset_right"):
        x = design(features[key][keep])
        forget_gram.addmm_(x.T, x)
    control_gram *= forget_gram.diagonal().mean() / control_gram.diagonal().mean()
    eye = torch.eye(len(forget_gram), dtype=forget_gram.dtype)
    base = theta(original)
    rhs = forget_gram @ (theta(reference) - base)
    result = []
    for anchor, reg in product((0.0, 0.005, 0.02, 0.06, 0.15, 0.32), (0.0001, 0.003, 0.025)):
        matrix = forget_gram + anchor * control_gram + reg * forget_gram.diagonal().mean() * eye
        residual = torch.linalg.solve(matrix, rhs)
        for strength in (0.96, 1.0, 1.04, 1.08, 1.15, 1.25):
            name = f"anchored-head-a{anchor:g}-r{reg:g}"
            result.append((name, strength, unpack_head(base + strength * residual)))
    return result


def main():
    directory = Path("loop")
    features = {**load(directory / "gpu-cache-a.pt"), **load(directory / "gpu-cache-b.pt")}
    search = Search(directory)
    num_examples = len(features["clean"])
    keep = torch.ones(num_examples, dtype=torch.bool)
    for seed, size in ((2675, 192), (2775, 256)):
        permutation = torch.randperm(num_examples, generator=torch.Generator().manual_seed(seed))
        keep[permutation[:size]] = False
    held_out, partners = gpu_split_indices(num_examples)
    keep[held_out] = False
    keep[partners[held_out]] = False
    baseline = ("record", 1.0, search.head("head", 1.10))
    heads = fit_selective_heads(features, search.original, search.head("hybrid", 1.22), keep)
    heads.append(baseline)
    at_left = {candidate_key(row): row for row in evaluate_heads(search.features, search.original, heads)}
    rows = evaluate_heads(search.validation, search.original, heads)
    for row in rows:
        other = at_left[candidate_key(row)]
        row["min_split_proxy"] = min(row["median_proxy"], other["median_proxy"])
        row["min_split_control"] = min(row["synthetic_control_agreement"],
                                       other["synthetic_control_agreement"])
        row["strong_hypothesis"] = min(row["proxy_scores"][2], other["proxy_scores"][2])
    minimum_control = next(row["min_split_control"] for row in rows if row["family"] == "record") - 0.025
    rows.sort(key=lambda row: (row["min_split_control"] >= minimum_control,
                               row["strong_hypothesis"]), reverse=True)
    weights = {(name, strength): head for name, strength, head in heads}
    output = directory / "anchored-v2"
    output.mkdir(exist_ok=True)
    (output / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows[:5]:
        name, strength = candidate_key(row)
        destination = output / f"{name}-{strength:g}.pt"
        torch.save(weights[(name, strength)], destination)
        print(f"{destination} proxy={row['min_split_proxy']:.5f} "
              f"strong_hypothesis={row['strong_hypothesis']:.5f} "
              f"agreement={row['min_split_control']:.3f}")


if __name__ == "__main__":
    main()
