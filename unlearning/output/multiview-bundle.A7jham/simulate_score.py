"""Estimate Unlearning tradeoffs locally with an ensemble of proxy references."""

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from checkpoint_changes import load


def logits(features, weights):
    return features.float() @ weights["head.weight"].float().T + weights["head.bias"].float()


def labels_loss(logits, labels):
    return -logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)


def loss_statistics(loss):
    return loss.mean(dim=-1), loss.std(dim=-1, unbiased=False)


def kl(reference, logits):
    predicted = F.log_softmax(logits / 2, dim=-1)
    return max(0.0, float(F.kl_div(predicted, reference, reduction="batchmean")))


def improvement(candidate, original):
    if original <= 1e-10:
        return 0.0
    return 1 - min(max(candidate / original, 0.0), 1.0)


def forget_components(original, candidate, teachers, labels):
    """All logits are untempered, teachers contains one tensor per reference model."""
    gold_probability = (teachers / 2).softmax(-1).mean(0)
    gold_mean, gold_std = loss_statistics(torch.stack([labels_loss(t, labels) for t in teachers]))
    gold_mean, gold_std = gold_mean.mean(), gold_std.mean()

    def loss_distance(logits):
        mean, std = loss_statistics(labels_loss(logits, labels))
        return float((mean - gold_mean).abs() + (std - gold_std).abs())

    return {
        "loss": improvement(loss_distance(candidate), loss_distance(original)),
        "prediction": improvement(kl(gold_probability, candidate), kl(gold_probability, original)),
    }


def retain_components(original, candidate, labels=None):
    values = {
        "agreement": float((original.argmax(-1) == candidate.argmax(-1)).float().mean()),
        "kl": kl((original / 2).softmax(-1), candidate),
    }
    if labels is None:
        values["retention"] = values["agreement"] / (1 + values["kl"])
        values["gate"] = None
    else:
        original_accuracy = float((original.argmax(-1) == labels).float().mean())
        accuracy = float((candidate.argmax(-1) == labels).float().mean())
        values["retention"] = min(accuracy / max(original_accuracy, 1e-10), 1) / (1 + values["kl"])
        values["gate"] = accuracy >= 0.70
    return values


def combine(clean, transformed, control):
    loss = (clean["loss"] + transformed["loss"]) / 2
    prediction = (clean["prediction"] + transformed["prediction"]) / 2
    retention = control["retention"] if control["gate"] is not False else 0.0
    return {"loss": loss, "prediction": prediction, "retention": retention, "score": loss * prediction * retention}


def proxy_teachers(views, labels, temperature, demotion):
    teacher = views / temperature
    teacher = teacher.clone()
    index = labels.view(1, -1, 1).expand(len(views), -1, -1)
    shifts = torch.full_like(index, -demotion, dtype=teacher.dtype)
    return teacher.scatter_add_(-1, index, shifts)


def compare_view_logits(original, changed, control):
    names = ("clean", "flip", "crop", "offset_left", "offset_right")
    labels = original["clean"].argmax(-1)
    views = torch.stack([original[key] for key in names])
    old_altered = views[1:].flatten(0, 1)
    new_altered = torch.cat([changed[key] for key in names[1:]])
    repeats = len(names) - 1
    scores = []
    for temperature, demotion in ((3.25, 2.5), (4.0, 4.0)):
        teachers = proxy_teachers(views, labels, temperature, demotion)
        clean = forget_components(original["clean"], changed["clean"], teachers, labels)
        altered = forget_components(old_altered, new_altered, teachers.repeat(1, repeats, 1), labels.repeat(repeats))
        scores.append(combine(clean, altered, control)["score"])
    return scores


def get_features(path, count, seed):
    if path.exists():
        saved = load(path)
        if saved["seed"] != seed or saved["count"] != count:
            raise ValueError("Feature cache configuration differs; use a new --cache path")
        return saved["views"]
    import task_template as kit
    from cache_features import cache

    images = kit.load_forget_images()
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(images), generator=rng)[:count]
    images = images[indices]
    model = kit.load_model()
    partners = torch.randperm(len(images), generator=rng)
    features = {
        view: cache(model, images, view, partners)
        for view in ("clean", "flip", "crop", "offset_left", "offset_right", "mixed")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "count": count, "views": features}, path)
    return features


def candidate_head(family, strength, original, modified, alternate=None):
    if family in ("original", "temperature"):
        return {key: value / strength for key, value in original.items()}
    if family == "head":
        return {key: torch.lerp(value, modified[key], strength) for key, value in original.items()}
    if family == "head2":
        return {key: torch.lerp(value, alternate[key], strength) for key, value in original.items()}
    if family == "hybrid":
        return {key: torch.lerp(value, modified[key], 1.10) / strength for key, value in original.items()}
    raise ValueError(f"Unknown candidate family {family}")


def default_candidates():
    candidates = [("original", 1.0)]
    candidates.extend(("temperature", t) for t in (1.5, 2.0, 3.75, 4.25, 4.75, 5.25))
    candidates.extend(("head", t) for t in (0.90, 0.95, 1.0, 1.03, 1.05, 1.07, 1.09, 1.10, 1.12, 1.15, 1.22))
    return candidates


def sweep(features, original_head, modified_head, candidates=None, *, alternate_head=None):
    if candidates is None:
        candidates = default_candidates()
    heads = [(family, strength, candidate_head(family, strength, original_head, modified_head, alternate_head))
             for family, strength in candidates]
    return evaluate_heads(features, original_head, heads)


def evaluate_heads(features, original_head, heads):
    original = {view: logits(values, original_head) for view, values in features.items()}
    labels = original["clean"].argmax(-1)
    views = ("clean", "flip", "crop", "offset_left", "offset_right")
    alternates = torch.stack([original[view] for view in views])
    altered_count = len(views) - 1
    altered_original = torch.cat([original[view] for view in views[1:]])
    altered_labels = labels.repeat(altered_count)
    hypotheses = ((2.5, 1.25), (3.25, 2.5), (4.0, 4.0))
    references = [
        proxy_teachers(alternates, labels, temperature, demotion)
        for temperature, demotion in hypotheses
    ]
    rows = []
    for family, strength, weights in heads:
        candidate = {view: logits(values, weights) for view, values in features.items()}
        control = retain_components(original["mixed"], candidate["mixed"])
        altered_candidate = torch.cat([candidate[view] for view in views[1:]])
        scenario_scores = []
        for ref in references:
            clean = forget_components(original["clean"], candidate["clean"], ref, labels)
            altered = forget_components(
                altered_original, altered_candidate, ref.repeat(1, altered_count, 1), altered_labels,
            )
            scenario_scores.append(combine(clean, altered, control))
        agreement = float((candidate["clean"].argmax(-1) == labels).float().mean())
        proxy_scores = [r["score"] for r in scenario_scores]
        rows.append({
            "family": family, "strength": strength,
            "median_proxy": statistics.median(proxy_scores),
            "proxy_scores": proxy_scores, "scenario_details": scenario_scores,
            "synthetic_control_agreement": control["agreement"], "forget_agreement": agreement,
        })
    return sorted(rows, key=lambda row: row["median_proxy"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("loop/simulator-features.pt"))
    parser.add_argument("--out", type=Path, default=Path("loop/simulator.json"))
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2675)
    args = parser.parse_args()
    if not 1 <= args.count <= 3500:
        parser.error("--count must be in [1, 3500]")
    weights = load("model.pt")
    original = {name: weights[name].float() for name in ("head.weight", "head.bias")}
    del weights
    changed = load("output/head-4.0-15901766-changes.pt")
    rows = sweep(get_features(args.cache, args.count, args.seed), original, changed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "count": args.count, "seed": args.seed, "rows": rows,
        "mode": "local_proxy", "hidden_accuracy_gate": None,
    }, indent=2) + "\n")
    for row in rows:
        print(f"{row['family']:>11} {row['strength']:.3f} score={row['median_proxy']:.5f} "
              f"scenarios={[round(x, 5) for x in row['proxy_scores']]} "
              f"control_proxy={row['synthetic_control_agreement']:.3f} forget={row['forget_agreement']:.3f}")


if __name__ == "__main__":
    main()
