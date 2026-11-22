"""Compare a candidate with SprintML's original checkpoint on fixed proxy data."""

import argparse
import importlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import task_template as kit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("temperature", "head", "gradient", "targets", "influence"))
    parser.add_argument("strength", type=float)
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--held-out", type=int, default=192)
    parser.add_argument("--control", type=Path, help="Permitted control .npz with a uint8 images array")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fit-all", action="store_true", help="Final fit on all forget images")
    args = parser.parse_args()
    if not math.isfinite(args.strength) or args.strength <= 0:
        parser.error("strength must be finite and positive")
    if args.held_out <= 0:
        parser.error("held-out sample count must be positive")
    return args


def fit_temperature(model, temperature):
    # Scaling the final linear layer changes probabilities while preserving rank.
    with torch.no_grad():
        model.head.weight.div_(temperature)
        if model.head.bias is not None:
            model.head.bias.div_(temperature)
    return model


def augment(images):
    crop = images[:, 13:-13, 13:-13].permute(0, 3, 1, 2).float()
    resized = F.interpolate(crop, (kit.IMAGE_SIZE, kit.IMAGE_SIZE), mode="bilinear", align_corners=False)
    return resized.round().byte().permute(0, 2, 3, 1).contiguous()


def metrics(original, updated):
    labels = original.argmax(dim=-1)
    original_log = F.log_softmax(original / 2, dim=-1)
    updated_log = F.log_softmax(updated / 2, dim=-1)
    loss = F.cross_entropy(updated, labels, reduction="none")
    return {
        "count": len(labels),
        "top1_agreement": (updated.argmax(dim=-1) == labels).float().mean().item(),
        "kl_original_to_candidate_t2": (original_log.exp() * (original_log - updated_log)).sum(-1).mean().item(),
        "mean_confidence": updated.softmax(dim=-1).amax(dim=-1).mean().item(),
        "pseudo_label_loss_mean": loss.mean().item(),
        "pseudo_label_loss_std": loss.std(unbiased=False).item(),
    }


def export_checkpoint(model, output):
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    weights = {
        name: tensor.detach().to(device="cpu", dtype=torch.float32)
        for name, tensor in model.state_dict().items()
    }
    if any(not torch.isfinite(tensor).all().item() for tensor in weights.values()):
        raise ValueError("Checkpoint contains non-finite weights")
    temporary = output.with_suffix(".tmp")
    try:
        torch.save(weights, temporary)
        if temporary.stat().st_size > 450_000_000:
            raise ValueError("Checkpoint exceeds the official 450 MB limit")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Launch this trial with Slurm on JURECA.")
    torch.manual_seed(args.seed)
    print(f"GPU: {torch.cuda.get_device_name()}")
    images = kit.load_forget_images()
    if args.held_out >= len(images):
        raise ValueError("The held-out set must be smaller than the forget set")
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(args.seed))
    holdout = images[order[:args.held_out]]
    groups = {"heldout": holdout, "flip": holdout.flip(dims=(2,)), "crop": augment(holdout)}
    if args.control is not None:
        groups["control_proxy"] = kit.load_forget_images(args.control)

    model = kit.load_model()
    original = {key: kit.predict(model, values, batch_size=32) for key, values in groups.items()}
    if args.method == "temperature":
        fit_temperature(model, args.strength)
    else:
        train = images if args.fit_all else images[order[args.held_out:]]
        model = importlib.import_module(args.method).fit(model, train, strength=args.strength, seed=args.seed)

    result = {"method": args.method, "strength": args.strength, "seed": args.seed, "fit_all": args.fit_all}
    for key, values in groups.items():
        logits = kit.predict(model, values, batch_size=32)
        result[key] = {"original": metrics(original[key], original[key]), "candidate": metrics(original[key], logits)}
    print(json.dumps(result, sort_keys=True))
    if args.out is not None:
        export_checkpoint(model, args.out)
        print(f"Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
