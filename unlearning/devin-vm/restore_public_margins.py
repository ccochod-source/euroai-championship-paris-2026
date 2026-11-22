"""Compare targeted recovery of class margins on public control images."""

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


def forget_design(directory):
    features = load(directory / "gpu-cache-a.pt")
    count = len(features["clean"])
    heldout, _ = gpu_split_indices(count)
    keep = torch.ones(count, dtype=torch.bool)
    keep[heldout] = False
    for seed, size in ((2675, 192), (2775, 256)):
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
        keep[order[:size]] = False
    return design(torch.cat((features["clean"][keep], features["flip"][keep])))


def candidate_heads(forget, public, original, starting_heads):
    gram = forget.T @ forget / len(forget)
    reg = 0.01 * gram.diagonal().mean() * torch.eye(len(gram), dtype=gram.dtype)
    public_gram = public.T @ public / len(public)
    original_logits = center(public @ theta(original))
    for family, head in starting_heads:
        start = theta(head)
        predicted = center(public @ start)
        for top, restore in product((1, 3, 10), (0.25, 0.5, 0.75)):
            old_top = original_logits.topk(top, dim=-1).indices
            mask = torch.zeros_like(predicted).scatter_(-1, old_top, 1)
            delta = center(mask * (original_logits - predicted) * restore)
            cross = public.T @ delta / len(public)
            for weight in (0.03, 0.1, 0.3, 1.0, 3.0):
                adjustment = torch.linalg.solve(gram + reg + weight * public_gram, weight * cross)
                label = f"margin-{family}-k{top}-r{restore:g}-w{weight:g}"
                yield label, 1.0, unpack_head(start + adjustment)


def main():
    directory = Path("loop")
    s = Search(directory)
    public = load(directory / "tiny-imagenet-control-2.pt")
    order = torch.randperm(len(public), generator=torch.Generator().manual_seed(193))
    train, test = public[order[:300]], public[order[300:]]
    sources = [
        ("record", s.head("head", 1.1)),
        ("classwise", load(directory / "proposals/agent-83bfcaf5857a.pt")),
        ("salsa", load(directory / "salsa/salsa-s0.94-t1.65-a0.3-c0.pt")),
    ]
    heads = [(label, 1.0, head) for label, head in sources]
    heads += list(candidate_heads(forget_design(directory), design(train), s.original, sources))
    first, second = [performance(split, s.original, heads) for split in (s.features, s.validation)]
    ranked = sorted(heads[len(sources):], key=lambda h: min(
        row[h[0]]["proxy_scores"][2] for row in (first, second)
    ), reverse=True)
    finalists = heads[:len(sources)] + ranked[:8]
    independent = performance(reserve_gpu_split(directory), s.original, finalists)
    original = logits(test, s.original)
    output = directory / "public-margins"
    output.mkdir(exist_ok=True)
    for name, _, head in finalists:
        h3_min = min(row[name]["proxy_scores"][2] for row in (first, second, independent))
        stats = retain_components(original, logits(test, head))
        if name.startswith("margin"):
            torch.save(head, output / f"{name}.pt")
        print(f"{name} H3_min={h3_min:.6f} agreement={stats['agreement']:.3f} KL={stats['kl']:.4f}",
              flush=True)
    print(f"{len(heads)-len(sources)} margin variants evaluated.", flush=True)


if __name__ == "__main__":
    main()
