"""Reconstruct the shortlisted classwise correction from the original ViT."""

import argparse
from pathlib import Path

import torch

from checkpoint_changes import load


def classwise_head(original, modified):
    weight, bias = "head.weight", "head.bias"
    delta_weight = modified[weight] - original[weight]
    delta_bias = modified[bias] - original[bias]
    row_norm = delta_weight.norm(dim=1)
    multiplier = 1.03 - 0.06 * (row_norm - row_norm.mean()) / row_norm.std()
    return {
        weight: ((original[weight] + multiplier[:, None] * delta_weight) / 1.8).contiguous(),
        bias: ((original[bias] + multiplier * delta_bias) / 1.8).contiguous(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("correction", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = load(args.original)
    original.update(classwise_head(original, load(args.correction)))
    with args.output.open("xb") as stream:
        torch.save(original, stream)


if __name__ == "__main__":
    main()
