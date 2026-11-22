"""Train an update to the ViT's last MLP projection with cached CLS activations."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import task_template as kit
from cache_features import alter
from checkpoint_changes import load
from public_control import sample_images
from simulate_score import compare_view_logits, retain_components


VIEWS = ("clean", "flip", "crop", "offset_left", "offset_right", "mixed")


@torch.no_grad()
def cls_activations(model, images, view):
    last = model.blocks[-1]
    partners = torch.randperm(len(images), generator=torch.Generator().manual_seed(2197))
    activations = []
    for offset in range(0, len(images), 16):
        batch = images[offset:offset + 16]
        inputs = alter(batch, view, offset, images, partners)
        tokens = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(inputs))))
        tokens = model.blocks[:-1](tokens)
        attended = tokens[:, :1] + last.ls1(last.attn(last.norm1(tokens))[:, :1])
        hidden = last.mlp.norm(last.mlp.drop1(last.mlp.act(last.mlp.fc1(last.norm2(attended)))))
        activations.append(torch.cat((attended[:, 0], hidden[:, 0]), dim=1))
    return torch.cat(activations)


def predict(model, states, head, weight):
    final = model.blocks[-1]
    skip, hidden = states.split([model.embed_dim, final.mlp.fc2.in_features], dim=1)
    embedding = model.norm(skip + final.ls2(F.linear(hidden, weight, final.mlp.fc2.bias)))
    return F.linear(embedding, head["head.weight"], head["head.bias"])


def prepare(model, directory):
    destination = directory / "last-block-activations.pt"
    if destination.is_file():
        return load(destination)
    images = kit.load_forget_images()
    permutation = torch.randperm(len(images), generator=torch.Generator().manual_seed(64221))
    learn = images[permutation[:192]]
    evaluate = images[permutation[192:352]]
    result = {"train": {}, "eval": {}}
    for view in VIEWS:
        result["eval"][view] = cls_activations(model, evaluate, view)
        if view in {"clean", "crop", "mixed"}:
            result["train"][view] = cls_activations(model, learn, view)
        print(f"Cached CLS activations for {view}.", flush=True)
    public = torch.stack(list(sample_images(directory / "tiny-imagenet-valid.parquet", 1)))[:160]
    result["public"] = cls_activations(model, public.permute(0, 2, 3, 1), "clean")
    expected = kit.predict(model, evaluate[:16], batch_size=16)
    actual = predict(model, result["eval"]["clean"][:16], original_head(model),
                     model.blocks[-1].mlp.fc2.weight)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.001)
    torch.save(result, destination)
    return result


def original_head(model):
    return {"head.weight": model.head.weight, "head.bias": model.head.bias}


def fit(model, state, base, reference, preserve, strength):
    params = model.blocks[-1].mlp.fc2.weight
    delta = torch.zeros_like(params, requires_grad=True)
    optimizer = torch.optim.Adam((delta,), lr=0.0003)
    forget = torch.cat((state["train"]["clean"], state["train"]["crop"]))
    control = torch.cat((state["train"]["mixed"], state["public"][:80]))
    with torch.no_grad():
        targets = predict(model, forget, reference, params)
        label = predict(model, state["train"]["clean"], reference, params).argmax(-1)
        teacher = (targets / 3.5).scatter_add_(
            -1, label.repeat(2)[:, None], targets.new_full((len(targets), 1), -strength),
        )
        q = (teacher / 2).softmax(-1)
        r = (predict(model, control, reference, params) / 2).softmax(-1)
    generator = torch.Generator().manual_seed(726)
    for _ in range(55):
        f = torch.randint(len(forget), (64,), generator=generator)
        c = torch.randint(len(control), (64,), generator=generator)
        predicted = predict(model, forget[f], base, params + delta)
        control_prediction = predict(model, control[c], base, params + delta)
        loss = F.kl_div((predicted / 2).log_softmax(-1), q[f], reduction="batchmean")
        loss += preserve * F.kl_div((control_prediction / 2).log_softmax(-1), r[c], reduction="batchmean")
        loss += 0.10 * delta.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return delta.detach()


@torch.no_grad()
def confirm(model, directory, extra_patches=()):
    images = kit.load_forget_images()
    permutation = torch.randperm(len(images), generator=torch.Generator().manual_seed(64221))
    images = images[permutation[352:608]]
    state = {view: cls_activations(model, images, view) for view in VIEWS}
    origin = original_head(model)
    weight = model.blocks[-1].mlp.fc2.weight
    raw = load("output/head-4.0-15901766-changes.pt")
    record = {key: torch.lerp(value, raw[key], 1.1) for key, value in origin.items()}
    original = {view: predict(model, states, origin, weight) for view, states in state.items()}
    control_cache = load(directory / "last-block-activations.pt")["public"][80:]
    control = predict(model, control_cache, origin, weight)
    classwise = load(directory / "proposal-agent-83bfcaf5857a-1.pt")
    classwise = {key: classwise[key] for key in origin}
    salsa = load(directory / "salsa/salsa-s0.94-t1.65-a0.3-c0.pt")
    candidates = [("record", record, weight), ("classwise", classwise, weight),
                  ("salsa-only", salsa, weight)]
    paths = sorted(directory.glob("last-block-salsa-f*-p10-s1.4.pt"))
    for path in [*paths, *extra_patches]:
        patch = load(path)
        candidates.append((path.stem, patch, patch["blocks.11.mlp.fc2.weight"]))
    for name, head, projection in candidates:
        changed = {view: predict(model, states, head, projection) for view, states in state.items()}
        scores = compare_view_logits(original, changed, retain_components(original["mixed"], changed["mixed"]))
        public = retain_components(control, predict(model, control_cache, head, projection))
        print(f"{name} scores={scores} public={public}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--patch", action="append", type=Path, default=[])
    args = parser.parse_args()
    directory = Path("loop")
    model = kit.load_model().eval().requires_grad_(False)
    if args.confirm:
        confirm(model, directory, args.patch)
        return
    state = prepare(model, directory)
    origin = original_head(model)
    raw = load("output/head-4.0-15901766-changes.pt")
    record = {k: torch.lerp(value, raw[k], 1.1) for k, value in origin.items()}
    salsa = load(directory / "salsa/salsa-s0.94-t1.65-a0.3-c0.pt")
    params = model.blocks[-1].mlp.fc2.weight
    splits = (slice(0, 80), slice(80, 160))
    predictions = {v: predict(model, x, origin, params) for v, x in state["eval"].items()}
    control = predict(model, state["public"][80:], origin, params)
    results = [("record", record, torch.zeros_like(params))]
    for name, head in (("record", record), ("salsa", salsa)):
        for strength in (2.0, 3.5, 5.0, 7.0):
            for preserve in (1.0, 3.0, 10.0):
                delta = fit(model, state, head, origin, preserve, strength=strength)
                for scale in (0.7, 1.0, 1.4, 2.0):
                    results.append((
                        f"{name}-f{strength:g}-p{preserve:g}-s{scale:g}",
                        head,
                        scale * delta,
                    ))
        print(f"Trained last-block variants from {name}.", flush=True)
    ranking = []
    for name, head, delta in results:
        total = params + delta
        changed = {v: predict(model, x, head, total) for v, x in state["eval"].items()}
        scores = [compare_view_logits({k: v[sl] for k, v in predictions.items()},
                                     {k: v[sl] for k, v in changed.items()},
                                     retain_components(predictions["mixed"][sl], changed["mixed"][sl])) for sl in splits]
        public = retain_components(control, predict(model, state["public"][80:], head, total))
        ranking.append((min(row[1] for row in scores), name, scores, public, head, total))
    for _, name, scores, public, _, _ in sorted(ranking, key=lambda row: row[0], reverse=True)[:8] + ranking[:1]:
        print(f"{name} scores={scores} public={public}", flush=True)
    for _, name, _, _, head, projection in sorted(ranking, key=lambda row: row[0], reverse=True)[:3]:
        torch.save({**head, "blocks.11.mlp.fc2.weight": projection},
                   directory / f"last-block-{name}.pt")


if __name__ == "__main__":
    main()
