"""Fit class-specific forget margins with mixed and public control anchors."""

from itertools import product
import json
from pathlib import Path

import torch

from checkpoint_changes import load
from compare_head_edits import performance
from compare_proposals import gpu_split_indices, reserve_gpu_split
from fit_cached_head import center, design, unpack_head
from selective_head import theta
from simulate_score import logits, retain_components
from tuning_loop import Search


def training_design(directory):
    cache = {**load(directory / "gpu-cache-a.pt"), **load(directory / "gpu-cache-b.pt")}
    count = len(cache["clean"])
    reserved, partners = gpu_split_indices(count)
    keep = torch.ones(count, dtype=torch.bool)
    keep[reserved] = False
    for seed, size in ((2675, 192), (2775, 256)):
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
        keep[order[:size]] = False
    forget = [design(cache[name][keep]) for name in ("clean", "flip", "crop", "offset_left", "offset_right")]
    mixed = design(cache["mixed"][keep & keep[partners]])
    return forget, mixed


def mean_gram(features):
    return features.T @ features / len(features)


def candidates(search, views, mixed, public):
    original = theta(search.original)
    z = torch.stack([center(view @ original) for view in views])
    labels = z[0].argmax(-1, keepdim=True)
    target_index = labels.expand(len(views), -1, -1)
    privileged = z.gather(-1, target_index)
    others = z.scatter(-1, target_index, float("-inf"))
    margin = (privileged - others.amax(-1, keepdim=True)).clamp_min(0)
    alternative = (others / 2).softmax(-1)
    edit_patterns = {
        "top": torch.zeros_like(z).scatter_(-1, target_index, -margin),
        "redistribute": (0.3 * margin * alternative).scatter_(-1, target_index, -margin),
        "global": -0.55 * z + torch.zeros_like(z).scatter_(-1, target_index, -0.45 * margin),
    }
    forget = torch.cat(views)
    gram = mean_gram(forget)
    mixed_gram, public_gram = mean_gram(mixed), mean_gram(public)
    reg = 0.015 * gram.diagonal().mean() * torch.eye(len(gram), dtype=gram.dtype)
    patterns = {name: forget.T @ center(change.flatten(0, 1)) / len(forget)
                for name, change in edit_patterns.items()}
    for mix, anchor in product((0.02, 0.10, 0.3, 1.0), (0.05, 0.25, 1.0)):
        preserve = mix * mixed_gram + anchor * public_gram
        factor = torch.linalg.cholesky(gram + preserve + reg)
        displacement = {name: torch.cholesky_solve(cross, factor) for name, cross in patterns.items()}
        for temperature in (1.0, 1.4, 1.8, 2.2):
            base = original / temperature
            base += torch.cholesky_solve(preserve @ (original - base), factor)
            for (name, direction), strength in product(displacement.items(), (0.45, 0.75, 1.05, 1.35)):
                key = f"forget-margin-{name}-t{temperature:g}-s{strength:g}-m{mix:g}-a{anchor:g}"
                yield key, 1.0, unpack_head(base + strength * direction)


def main():
    directory = Path("loop")
    search = Search(directory)
    public = load(directory / "tiny-imagenet-control-2.pt")
    order = torch.randperm(len(public), generator=torch.Generator().manual_seed(193))
    train, test = public[order[:300]], public[order[300:]]
    models = list(candidates(search, *training_design(directory), design(train)))
    reference = [("record", 1.0, search.head("head", 1.1))]
    first, second = [performance(split, search.original, reference + models)
                     for split in (search.features, search.validation)]
    baseline_logits = logits(train, search.original)
    rank = []
    for name, scale, weights in models:
        forget = min(row[name]["proxy_scores"][2] for row in (first, second))
        control = retain_components(baseline_logits, logits(train, weights))["retention"]
        rank.append((forget, forget * control, name, scale, weights))
    best_forget = sorted(rank, key=lambda row: row[0], reverse=True)[:8]
    best_joint = sorted(rank, key=lambda row: row[1], reverse=True)[:8]
    shortlist = list({name: (name, scale, weights)
                      for _, _, name, scale, weights in best_forget + best_joint}.values())
    independent = performance(reserve_gpu_split(directory), search.original, reference + shortlist)
    output = directory / "forget-margins"
    output.mkdir(exist_ok=True)
    report = []
    z = logits(test, search.original)
    for name, _, weights in reference + shortlist:
        scenario_two, scenario_three = [min(row[name]["proxy_scores"][index]
                                           for row in (first, second, independent)) for index in (1, 2)]
        control = retain_components(z, logits(test, weights))
        report.append({"name": name, "scenario_2": scenario_two, "scenario_3": scenario_three,
                       "public": control})
        if name != "record":
            torch.save(weights, output / f"{name}.pt")
        print(f"{name} H2={scenario_two:.5f} H3={scenario_three:.5f} "
              f"agreement={control['agreement']:.3f} KL={control['kl']:.4f}", flush=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(models)} targeted margin candidates evaluated.", flush=True)


if __name__ == "__main__":
    main()
