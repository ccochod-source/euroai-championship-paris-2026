"""Fit a ViT final MLP projection on cached CLS activations."""

import argparse
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from simulate_score import combine, forget_components, proxy_teachers, retain_components


PROJECTION = "blocks.11.mlp.fc2.weight"
VIEWS = ("clean", "flip", "crop", "offset_left", "offset_right")


def predict(state, model, head, projection):
    skip, hidden = state.split((768, 3072), dim=-1)
    residual = F.linear(hidden, projection, model["blocks.11.mlp.fc2.bias"])
    tokens = skip + residual * model["blocks.11.ls2.gamma"]
    embeddings = F.layer_norm(tokens, (768,), model["norm.weight"], model["norm.bias"], 1e-6)
    return F.linear(embeddings, head["head.weight"], head["head.bias"])


def fit(model, head, original, forget, control, labels, teacher, preserve, steps):
    projection = model[PROJECTION]
    delta = torch.zeros_like(projection, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.0003)
    temperature, demotion = teacher
    with torch.no_grad():
        forget_teacher = predict(forget, model, original, projection) / temperature
        forget_teacher.scatter_add_(-1, labels[:, None], forget_teacher.new_full((len(labels), 1), -demotion))
        forget_targets = (forget_teacher / 2).softmax(-1)
        control_targets = (predict(control, model, original, projection) / 2).softmax(-1)
    rng = torch.Generator().manual_seed(726)
    f_batches = torch.randint(len(forget), (steps, 64), generator=rng, device="cpu").to(forget.device)
    c_batches = torch.randint(len(control), (steps, 64), generator=rng, device="cpu").to(control.device)
    for step in range(steps):
        f, c = f_batches[step], c_batches[step]
        updated = projection + delta
        f_log = (predict(forget[f], model, head, updated) / 2).log_softmax(-1)
        c_log = (predict(control[c], model, head, updated) / 2).log_softmax(-1)
        loss = F.kl_div(f_log, forget_targets[f], reduction="batchmean")
        loss += preserve * F.kl_div(c_log, control_targets[c], reduction="batchmean")
        loss += 0.10 * delta.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return delta.detach()


@torch.no_grad()
def evaluate(predictions, original, public_logits, public_base):
    public = retain_components(public_base, public_logits)
    worst = float("inf")
    report = []
    for split in (slice(0, 80), slice(80, 160)):
        clean = original["clean"][split]
        labels = clean.argmax(-1)
        views = torch.stack([original[key][split] for key in VIEWS])
        originals = views[1:].flatten(0, 1)
        altered = torch.cat([predictions[key][split] for key in VIEWS[1:]])
        control = retain_components(original["mixed"][split], predictions["mixed"][split])
        scores = []
        for temperature, demotion in ((3.25, 2.5), (4.0, 4.0)):
            teachers = proxy_teachers(views, labels, temperature, demotion)
            clean_parts = forget_components(clean, predictions["clean"][split], teachers, labels)
            altered_parts = forget_components(originals, altered, teachers.repeat(1, 4, 1), labels.repeat(4))
            scores.append(combine(clean_parts, altered_parts, control)["score"])
        worst = min(worst, scores[1])
        report.append(scores)
    return {"worst": worst, "scores": report, "public": public}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Metal is unavailable")

    path = Path("result")
    path.mkdir(exist_ok=True)
    bundle = torch.load("metal-weights.pt", weights_only=True, map_location=args.device)
    model, heads = bundle["model"], bundle["heads"]
    state = torch.load("metal-activations.pt", weights_only=True, map_location=args.device)
    train, test, public = state["train"], state["eval"], state["public"]
    original, record = heads["original"], heads["record"]
    weight = model[PROJECTION]
    forget = torch.cat((train["clean"], train["crop"]))
    control = torch.cat((train["mixed"], public[:80]))
    labels = predict(train["clean"], model, original, weight).argmax(-1).repeat(2)
    true_logits = {key: predict(value, model, original, weight).float().cpu() for key, value in test.items()}
    public_base = predict(public[80:], model, original, weight).float().cpu()

    def measure(head, projection):
        logits = {key: predict(value, model, head, projection).float().cpu() for key, value in test.items()}
        return evaluate(logits, true_logits, predict(public[80:], model, head, projection).float().cpu(), public_base)

    record_result = measure(record, weight)
    print(json.dumps({"device": str(weight.device), "record": record_result}), flush=True)
    choices = [(("record", record), (3.5, 3.5), 3.0)] if args.smoke else list(product(
        [(name, head) for name, head in heads.items() if name != "original"],
        ((2.6, 2.0), (3.5, 3.5), (4.5, 5.0), (5.5, 7.0)),
        (0.5, 1.0, 3.0, 6.0, 10.0),
    ))
    results = []
    best = -1
    started = perf_counter()
    for batch, ((name, head), teacher, preserve) in enumerate(choices, 1):
        delta = fit(model, head, original, forget, control, labels, teacher, preserve, 2 if args.smoke else 55)
        for scale in ((1.0,) if args.smoke else (0.6, 1.0, 1.4, 1.8)):
            projection = weight + scale * delta
            scores = measure(head, projection)
            row = {"head": name, "teacher": teacher, "preserve": preserve, "scale": scale, **scores}
            results.append(row)
            if row["worst"] > best and torch.isfinite(projection).all().item():
                best = row["worst"]
                torch.save({**{key: value.cpu() for key, value in head.items()},
                            PROJECTION: projection.cpu()}, path / "best-patch.pt")
        if batch % 5 == 0 or batch == 1:
            print(f"{len(results)} settings tested in {perf_counter() - started:.1f}s; best={best:.6f}", flush=True)
    (path / "scores.json").write_text(json.dumps({"record": record_result, "candidates": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
