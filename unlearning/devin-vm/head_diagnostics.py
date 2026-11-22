"""Compare changes in the fitted head with a fixed subset of the forget set."""

import json

import torch

import task_template as kit
from cache_features import cache
from checkpoint_changes import load
from run_trial import metrics
from simulate_score import logits


def main():
    image_ids = torch.randperm(3500, generator=torch.Generator().manual_seed(2666))[:192]
    all_images = kit.load_forget_images()
    images = all_images[image_ids]
    model = kit.load_model()
    partners = torch.randperm(len(images), generator=torch.Generator().manual_seed(2667))
    original = {name: model.state_dict()[name].cpu() for name in ("head.weight", "head.bias")}
    altered = load("output/head-4.0-15901766-changes.pt")
    for view in ("clean", "flip", "crop", "mixed"):
        features = cache(model, images, view, partners)
        reference = logits(features, original)
        for factor in (1.0, 1.10, 1.22, 1.43, 1.60):
            updated = {key: torch.lerp(value, altered[key], factor) for key, value in original.items()}
            result = metrics(reference, logits(features, updated))
            print(json.dumps({"view": view, "head_factor": factor, **result}), flush=True)
        for temperature in (2.0, 3.75, 4.75):
            result = metrics(reference, reference / temperature)
            print(json.dumps({"view": view, "temperature": temperature, **result}), flush=True)


if __name__ == "__main__":
    main()
