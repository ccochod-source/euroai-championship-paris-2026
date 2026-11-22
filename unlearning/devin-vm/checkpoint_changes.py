"""Move exact checkpoint changes between machines that already have the original."""

import argparse
from pathlib import Path

import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "restore"))
    parser.add_argument("original", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    original, source = load(args.original), load(args.source)
    if args.action == "pack":
        if source.keys() != original.keys():
            raise ValueError("Checkpoint keys do not match the original")
        weights = {name: weight for name, weight in source.items() if not torch.equal(original[name], weight)}
        print(f"{len(weights)} changed tensors")
    else:
        if not source.keys() <= original.keys():
            raise ValueError("Patch contains keys absent from the original")
        for name, weight in source.items():
            if weight.shape != original[name].shape or weight.dtype != torch.float32 or not torch.isfinite(weight).all():
                raise ValueError(f"Invalid patch tensor: {name}")
        weights = original | source
    with args.out.open("xb") as output:
        torch.save(weights, output)
    print(f"{args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
