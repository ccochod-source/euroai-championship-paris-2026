"""Compare targeted head edits with the strongest submitted head."""

from itertools import product
import json
from pathlib import Path

import torch

from compare_proposals import reserve_gpu_split
from simulate_score import evaluate_heads
from tuning_loop import Search


HYPOTHESIS = 2
RECORD = ("head", 1.10)


def candidates(search):
    base = search.original
    best = search.head(*RECORD)
    reference = torch.cat((base["head.weight"], base["head.bias"][:, None]), dim=1)
    changed = torch.cat((best["head.weight"], best["head.bias"][:, None]), dim=1)
    reference, changed = (value - value.mean(0) for value in (reference, changed))
    global_scale = (changed * reference).sum() / reference.square().sum()
    targeted = changed - global_scale * reference
    for scale, gain, bias in product(
        (0.81, 0.88, 0.94, 1.0, 1.06, 1.13),
        (0.80, 0.9, 1.0, 1.08, 1.18, 1.28),
        (0.75, 0.90, 1.0, 1.12, 1.30),
    ):
        weights = global_scale * scale * reference + gain * targeted
        weights[:, -1] = global_scale * scale * reference[:, -1] + bias * gain * targeted[:, -1]
        key = f"edit-scale{scale:g}-gain{gain:g}-bias{bias:g}"
        yield key, 1.0, {
            "head.weight": weights[:, :-1].contiguous(),
            "head.bias": weights[:, -1].contiguous(),
        }


def performance(features, original, heads):
    rows = evaluate_heads(features, original, heads)
    return {row["family"]: row for row in rows}


def metrics(row):
    return (row["proxy_scores"][HYPOTHESIS], row["synthetic_control_agreement"],
            row["proxy_scores"][1])


def main():
    directory = Path("loop")
    search = Search(directory)
    heads = list(candidates(search))
    baseline = ("record", 1.0, search.head(*RECORD))
    comparisons = [baseline, ("mild-temperature", 1.0, search.head("hybrid", 1.09))]
    first = performance(search.features, search.original, heads + comparisons)
    second = performance(search.validation, search.original, heads + comparisons)
    control_floor = min(first["record"]["synthetic_control_agreement"],
                        second["record"]["synthetic_control_agreement"]) - 0.025
    shortlist = sorted(heads, key=lambda head: (
        min(metrics(first[head[0]])[1], metrics(second[head[0]])[1]) >= control_floor,
        min(metrics(first[head[0]])[0], metrics(second[head[0]])[0]),
    ), reverse=True)[:10]
    third = performance(reserve_gpu_split(directory), search.original, shortlist + comparisons)
    result = []
    for name, _, weights in comparisons + shortlist:
        values = [metrics(split[name]) for split in (first, second, third)]
        result.append({"name": name, "splits": values})
        print(name, " | ".join(",".join(f"{v:.5f}" for v in row) for row in values), flush=True)
        if name != "record":
            destination = directory / "head-edits"
            destination.mkdir(exist_ok=True)
            torch.save(weights, destination / f"{name}.pt")
    (directory / "head-edit-comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
