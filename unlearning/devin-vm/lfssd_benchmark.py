"""Compare LF-SSD with matched full and retain-only retraining on public images."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def classifier(classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(64 * 6 * 6, classes),
    )


def fit(initial, images, labels, seed):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    order = torch.Generator().manual_seed(seed)
    for _ in range(40):
        for batch in torch.randperm(len(images), generator=order).split(64):
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(images.device)
            F.cross_entropy(model(images[batch]), labels[batch]).backward()
            optimizer.step()
    return model.eval()


def lfssd_importance(model, inputs, batch_size=8):
    parameters = dict(model.named_parameters())
    total = {name: torch.zeros_like(value) for name, value in parameters.items()}
    for batch in inputs.split(batch_size):
        outputs = model(batch)
        gradients = torch.autograd.grad(outputs.square().sum(-1).mean(), tuple(parameters.values()))
        for name, gradient in zip(parameters, gradients):
            total[name].add_(gradient.abs(), alpha=len(batch) / len(inputs))
    return total


@torch.no_grad()
def dampened(model, forget, train, scale):
    edited = copy.deepcopy(model)
    for name, weight in edited.named_parameters():
        ratio = forget[name] / train[name].clamp_min(1e-10)
        attenuation = torch.where(ratio > 1.5, 1 - ratio.reciprocal(), 0)
        weight.mul_(1 - scale * attenuation)
    return edited


def kl_from(logits, target):
    return F.kl_div((logits / 2).log_softmax(-1), (target / 2).softmax(-1),
                    reduction="batchmean").item()


@torch.no_grad()
def compare(models, forget, changed, held, forget_labels, held_labels):
    gold = models["retrained"]
    gold_clean, gold_changed, gold_held = gold(forget), gold(changed), gold(held)
    gold_loss = F.cross_entropy(gold_clean, forget_labels, reduction="none")
    gold_altered_loss = F.cross_entropy(gold_changed, forget_labels, reduction="none")
    rows = {}
    for name, model in models.items():
        clean, altered, retain = model(forget), model(changed), model(held)
        losses = [F.cross_entropy(logits, forget_labels, reduction="none") for logits in (clean, altered)]
        rows[name] = {
            "kl_clean": kl_from(clean, gold_clean),
            "kl_altered": kl_from(altered, gold_changed),
            "loss_moment_error": sum(
                abs((loss.mean() - target.mean()).item()) + abs((loss.std() - target.std()).item())
                for loss, target in zip(losses, (gold_loss, gold_altered_loss))
            ) / 2,
            "held_accuracy": (retain.argmax(-1) == held_labels).float().mean().item(),
            "held_kl": kl_from(retain, gold_held),
        }
    return rows


def trial(images, args, seed):
    # The bundle is balanced with 50 distinct images per class.
    classes, count = images.shape[:2]
    generator = torch.Generator().manual_seed(seed)
    order = torch.stack([torch.randperm(count, generator=generator) for _ in range(classes)]).to(args.device)
    labels = torch.arange(classes, device=args.device)[:, None].expand_as(order)
    images = images[labels, order]
    forget, retain, held = [part.flatten(0, 1) for part in images.split((10, 30, 10), dim=1)]
    yf, yr, yh = [part.flatten() for part in labels.split((10, 30, 10), dim=1)]
    torch.manual_seed(seed)
    initial = classifier(classes).to(args.device)
    full = fit(initial, torch.cat((forget, retain)), torch.cat((yf, yr)), seed)
    gold = fit(initial, retain, yr, seed)
    forget_importance = lfssd_importance(full, forget)
    retain_importance = lfssd_importance(full, retain)
    all_importance = {name: 0.25 * value + 0.75 * retain_importance[name]
                      for name, value in forget_importance.items()}
    light = dampened(full, forget_importance, all_importance, 0.3)
    paper = dampened(full, forget_importance, all_importance, 1.0)
    changed = forget.flip(-1)[:, :, 2:22, 1:21]
    changed = F.interpolate(changed, size=(24, 24), mode="bilinear", align_corners=False)
    return compare({"full": full, "retrained": gold, "lfssd_30": light, "lfssd_100": paper},
                   forget, changed, held, yf, yh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--input", type=Path, default=Path("lfssd-tiny.pt"))
    parser.add_argument("--output", type=Path, default=Path("lfssd-benchmark.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    images = torch.load(args.input, weights_only=True, map_location="cpu").to(args.device)
    images = images.float().div(127.5).sub(1)
    results = []
    for seed in (73, 127, 191):
        rows = trial(images, args, seed)
        print(json.dumps({"seed": seed, "device": args.device, "results": rows}), flush=True)
        results.append(rows)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    passed = []
    for name in ("lfssd_30", "lfssd_100"):
        if all(rows[name]["kl_clean"] < rows["full"]["kl_clean"] and
               rows[name]["kl_altered"] < rows["full"]["kl_altered"] and
               rows[name]["held_accuracy"] >= rows["full"]["held_accuracy"] - 0.02
               for rows in results):
            passed.append(name)
    print(json.dumps({"passed": passed}), flush=True)


if __name__ == "__main__":
    main()
