"""Fit SALSA-inspired probability targets to cached views with public anchors."""

from itertools import product
from pathlib import Path

import torch

from checkpoint_changes import load
from compare_head_edits import performance
from compare_proposals import gpu_split_indices, reserve_gpu_split
from fit_cached_head import center, design, unpack_head
from selective_head import theta
from simulate_score import logits, retain_components
from tuning_loop import Search


def candidates(search, directory, control):
    features = {**load(directory / "gpu-cache-a.pt"), **load(directory / "gpu-cache-b.pt")}
    count = len(features["clean"])
    held_out, _ = gpu_split_indices(count)
    keep = torch.ones(count, dtype=torch.bool)
    keep[held_out] = False
    for seed, size in ((2675, 192), (2775, 256)):
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
        keep[order[:size]] = False
    views = [design(features[key][keep]) for key in ("clean", "flip", "crop")]
    x = torch.cat(views)
    c = design(control)
    gram = x.T @ x / len(x)
    anchors = c.T @ c / len(c)
    ridge = 0.02 * gram.diagonal().mean() * torch.eye(len(gram), dtype=gram.dtype)
    original = theta(search.original)
    label = (views[0] @ original).argmax(-1, keepdim=True).repeat(len(views), 1)
    for strength, temperature in product((0.0, 0.94, 1.03), (1.25, 1.65, 2.1, 2.6)):
        base = theta(search.head("head", strength)) / temperature
        base_logits = center(x @ base)
        probability = (base_logits / 2).softmax(-1)
        mass = probability.gather(-1, label)
        alternative = (base_logits.scatter(-1, label, float("-inf")) / 2).softmax(-1)
        for alpha in (0.025, 0.075, 0.15, 0.30):
            moved = alpha * mass
            target = (probability + moved * alternative).scatter(-1, label, mass - moved)
            delta = center(2 * target.clamp_min(1e-12).log()) - base_logits
            cross = x.T @ delta / len(x)
            for anchor in (0.0, 0.1, 0.5, 2.0):
                update = torch.linalg.solve(gram + anchor * anchors + ridge,
                                            cross + anchor * anchors @ (original - base))
                name = f"salsa-s{strength:g}-t{temperature:g}-a{alpha:g}-c{anchor:g}"
                yield name, 1.0, unpack_head(base + update)


def main():
    directory = Path("loop")
    setup = Search(directory)
    public = load(directory / "tiny-imagenet-control-2.pt")
    order = torch.randperm(len(public), generator=torch.Generator().manual_seed(193))
    train, test = public[order[:300]], public[order[300:]]
    heads = list(candidates(setup, directory, train))
    rows = [performance(split, setup.original, heads) for split in (setup.features, setup.validation)]
    ordered = sorted(heads, key=lambda head: min(
        row[head[0]]["proxy_scores"][index] for row in rows for index in (1, 2)
    ), reverse=True)
    shortlist = ordered[:8]
    independent = performance(reserve_gpu_split(directory), setup.original, shortlist)
    test_logits = logits(test, setup.original)
    output = directory / "salsa"
    output.mkdir(exist_ok=True)
    for name, _, head in shortlist:
        torch.save(head, output / f"{name}.pt")
        robust = min(row[name]["median_proxy"] for row in rows)
        h2, h3 = independent[name]["proxy_scores"][1:]
        control = retain_components(test_logits, logits(test, head))
        print(f"{name} split_min={robust:.5f} H2={h2:.5f} H3={h3:.5f} "
              f"public_agreement={control['agreement']:.3f} KL={control['kl']:.3f}", flush=True)
    print(f"{len(heads)} SALSA-inspired candidates evaluated.", flush=True)


if __name__ == "__main__":
    main()
