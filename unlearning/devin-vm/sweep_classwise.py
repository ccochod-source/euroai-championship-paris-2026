"""Sweep classwise head updates using public-image conservation diagnostics."""

from itertools import product
import json
from pathlib import Path

import torch

from checkpoint_changes import load
from compare_head_edits import performance
from compare_proposals import gpu_split_indices, reserve_gpu_split
from reproduce_classwise import classwise_head
from simulate_score import logits, retain_components
from tuning_loop import Search


def robust_standardize(values):
    return ((values - values.mean()) / values.std().clamp_min(1e-8)).clamp(-2.5, 2.5)


def class_patterns(search, directory, public_features):
    clean = load(directory / "gpu-cache-a.pt")["clean"]
    count = len(clean)
    held_out, _ = gpu_split_indices(count)
    use = torch.ones(count, dtype=torch.bool)
    use[held_out] = False
    for seed, size in ((2675, 192), (2775, 256)):
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
        use[order[:size]] = False
    forget_logits = logits(clean[use], search.original)
    public_logits = logits(public_features, search.original)
    num_classes = forget_logits.shape[1]
    forget_counts = forget_logits.argmax(-1).bincount(minlength=num_classes).float()
    public_counts = public_logits.argmax(-1).bincount(minlength=num_classes).float()
    top_margin, labels = forget_logits.topk(2, dim=1)
    margin = torch.zeros_like(forget_counts).index_add_(0, labels[:, 0],
                                                      top_margin[:, 0] - top_margin[:, 1])
    delta = search.modified["head.weight"] - search.original["head.weight"]
    patterns = {
        "delta-norm": delta.norm(dim=1),
        "forget-frequency": forget_counts.log1p(),
        "forget-margin": margin / forget_counts.clamp_min(1),
        "public-frequency": public_counts.log1p(),
        "relative-frequency": forget_counts.log1p() - public_counts.log1p(),
    }
    return {name: robust_standardize(values) for name, values in patterns.items()}


def head_variants(search, patterns):
    weight, bias = "head.weight", "head.bias"
    original, modified = search.original, search.modified
    delta_weight = modified[weight] - original[weight]
    delta_bias = modified[bias] - original[bias]
    reference_scale = 1.03 - 0.06 * robust_standardize(delta_weight.norm(dim=1))
    settings = product((0.94, 0.99, 1.03, 1.07, 1.11), (1.35, 1.60, 1.80, 2.05, 2.3),
                       (-0.035, 0.035, -0.075, 0.075))
    for strength, temperature, additional in settings:
        for pattern, values in patterns.items():
            scale = reference_scale + strength - 1.03 + additional * values
            name = f"class-{pattern}-s{strength:g}-t{temperature:g}-x{additional:g}"
            yield name, 1.0, {
                weight: ((original[weight] + scale[:, None] * delta_weight) / temperature).contiguous(),
                bias: ((original[bias] + scale * delta_bias) / temperature).contiguous(),
            }


def evaluate_public(features, original, heads):
    base = logits(features, original)
    return {name: retain_components(base, logits(features, head)) for name, _, head in heads}


def shortlist_score(rows, public):
    # Measure both plausible forget hypotheses and discard their synthetic
    # control term. Use conservation on the independent public domain here.
    forgetting = [
        min(row["scenario_details"][index]["loss"] * row["scenario_details"][index]["prediction"]
            for row in rows)
        for index in (1, 2)
    ]
    return min(forgetting) * public["retention"]


def main():
    directory = Path("loop")
    search = Search(directory)
    controls = load(directory / "tiny-imagenet-control-2.pt")
    order = torch.randperm(len(controls), generator=torch.Generator().manual_seed(193))
    train, validation = controls[order[:300]], controls[order[300:]]
    baseline = [
        ("record", 1.0, search.head("head", 1.10)),
        ("classwise", 1.0, classwise_head(search.original, search.modified)),
    ]
    variants = list(head_variants(search, class_patterns(search, directory, train)))
    training = [
        performance(split, search.original, variants + baseline)
        for split in (search.features, search.validation)
    ]
    train_control = evaluate_public(train, search.original, variants + baseline)
    ranked = sorted(variants, key=lambda h: shortlist_score(
        [split[h[0]] for split in training], train_control[h[0]],
    ), reverse=True)[:8]
    print(f"{len(variants)} classwise variants evaluated on two splits.", flush=True)
    comparison = baseline + ranked
    independent = performance(reserve_gpu_split(directory), search.original, comparison)
    test_control = evaluate_public(validation, search.original, comparison)
    output = directory / "classwise-sweep"
    output.mkdir(exist_ok=True)
    results = []
    for name, _, head in comparison:
        public = test_control[name]
        proxy = independent[name]["proxy_scores"]
        joint = shortlist_score([independent[name]], public)
        results.append({"name": name, "joint": joint, "proxy_scores": proxy, "public": public})
        print(f"{name}: H2={proxy[1]:.5f} H3={proxy[2]:.5f} "
              f"agree={public['agreement']:.3f} KL={public['kl']:.4f} joint={joint:.5f}", flush=True)
        if name.startswith("class-"):
            torch.save(head, output / f"{name}.pt")
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
