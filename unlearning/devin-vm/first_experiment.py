"""Run a small, local head-only experiment with the official Unlearning kit."""

import torch

from task_template import load_forget_images, load_model, preprocess


def features(model, images, batch_size=8):
    chunks = []
    for batch in images.split(batch_size):
        tokens = model.forward_features(preprocess(batch))
        chunks.append(model.forward_head(tokens, pre_logits=True).float())
    return torch.cat(chunks)


def report(name, original, updated):
    before = original.softmax(dim=-1)
    after = updated.softmax(dim=-1)
    before_confidence = before.amax(dim=-1).mean().item()
    after_confidence = after.amax(dim=-1).mean().item()
    agreement = (before.argmax(dim=-1) == after.argmax(dim=-1)).sum().item()
    print(
        f"{name}: mean top-1 confidence {before_confidence:.2%} -> "
        f"{after_confidence:.2%}; top-1 agreement {agreement}/{len(before)}"
    )


@torch.inference_mode()
def main():
    model = load_model()
    images = load_forget_images()
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(images), generator=generator)[:64]
    fit_images, held_out_images = images[indices].chunk(2)

    fit_features = features(model, fit_images)
    held_out_features = features(model, held_out_images)
    flipped_features = features(model, fit_images.flip(dims=(2,)))
    groups = {
        "Fit forget subset": fit_features,
        "Horizontal flips of fit samples": flipped_features,
        "Disjoint forget subset": held_out_features,
    }
    original = {name: model.head(values) for name, values in groups.items()}

    # Fit a regularized weight update that shrinks centered logits on the fit
    # samples. The bias and feature extractor are held fixed.
    fit_logits = original["Fit forget subset"]
    target_delta = -0.10 * (fit_logits - fit_logits.mean(dim=-1, keepdim=True))
    gram = fit_features @ fit_features.T
    ridge = gram.diagonal().mean()
    gram.diagonal().add_(ridge)
    weight_delta = fit_features.T @ torch.linalg.solve(gram, target_delta)
    model.head.weight.add_(weight_delta.T)

    for name, values in groups.items():
        report(name, original[name], model.head(values))

    print("Local diagnostic only. The reference model and retain set are server-side.")


if __name__ == "__main__":
    main()
