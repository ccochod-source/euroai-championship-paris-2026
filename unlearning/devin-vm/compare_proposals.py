"""Evaluate independent head proposals on identical cached examples."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from checkpoint_changes import load
from simulate_score import evaluate_heads, logits, retain_components
from tuning_loop import HEAD_NAMES, SCORE_ANCHOR, Search


def gpu_split_indices(count):
    allowed = torch.ones(count, dtype=torch.bool)
    for seed, size in ((2675, 192), (2775, 256)):
        ids = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
        allowed[ids[:size]] = False
    partners = torch.randperm(count, generator=torch.Generator().manual_seed(2666))
    ids = torch.randperm(count, generator=torch.Generator().manual_seed(3990))
    return ids[allowed[ids] & allowed[partners[ids]]][:512], partners


def reserve_gpu_split(directory):
    features = {**load(directory / "gpu-cache-a.pt"), **load(directory / "gpu-cache-b.pt")}
    ids, _ = gpu_split_indices(len(features["clean"]))
    return {key: value[ids] for key, value in features.items()}


def load_head(path, original):
    head = load(path)
    if not isinstance(head, dict) or set(head) != set(HEAD_NAMES) or any(
        not isinstance(head[key], torch.Tensor)
        or head[key].shape != original[key].shape
        or head[key].dtype != torch.float32
        or not torch.isfinite(head[key]).all().item()
        for key in HEAD_NAMES
    ):
        raise ValueError(f"Invalid head proposal: {path}")
    return head


def centered_logits(features, head):
    values = logits(features, head)
    return values - values.mean(-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--require-robust-gain", action="store_true",
                        help="Require a gain over the record across forget splits and public control")
    args = parser.parse_args()
    directory = Path("loop")
    search = Search(directory)
    validation = reserve_gpu_split(directory)
    scores = [SCORE_ANCHOR, *(row for line in (directory / "history.jsonl").read_text().splitlines()
                              if (row := json.loads(line))["status"] == "scored")]
    search.evaluate(scores)
    pooled, scales = search.score_scales
    heads = [(str(path), 1.0, load_head(path, search.original)) for path in args.checkpoints]
    held_out_rows = evaluate_heads(validation, search.original, heads)
    tune_rows = {row["family"]: row for row in evaluate_heads(search.features, search.original, heads)}
    if args.require_robust_gain:
        validation_rows = None
        if search.validation is not None:
            validation_rows = {row["family"]: row for row in
                               evaluate_heads(search.validation, search.original, heads)}
        record = search.head("head", 1.10)
        record_scores = [
            evaluate_heads(features, search.original, [("record", 1.0, record)])[0]["proxy_scores"]
            for features in (search.features, validation, search.validation)
            if features is not None
        ]
        public_features = load(directory / "tiny-imagenet-control-2.pt")
        public_original = logits(public_features, search.original)
        public_record = retain_components(public_original, logits(public_features, record))
    head_by_path = {name: head for name, _, head in heads}
    original_logits = centered_logits(validation["mixed"], search.original)
    for row in held_out_rows:
        tune = tune_rows[row["family"]]
        predicted = centered_logits(validation["mixed"], head_by_path[row["family"]])
        scale = (predicted * original_logits).sum() / original_logits.square().sum()
        residual = (predicted - scale * original_logits).norm() / predicted.norm().clamp_min(1e-8)
        near_temperature = float(residual) <= 0.03 and scale > 0
        equivalent_t = float(scale.reciprocal()) if near_temperature else None
        similar_to_tested = near_temperature and any(
            known["family"] == "temperature" and abs(known["strength"] - equivalent_t) <= 0.25
            for known in scores
        )
        row["tune_proxy"] = tune["median_proxy"]
        row["minimum_control_agreement"] = min(row["synthetic_control_agreement"],
                                                tune["synthetic_control_agreement"])
        row["equivalent_temperature"] = equivalent_t
        row["similar_to_tested"] = similar_to_tested
        row["calibrated_proxy"] = min(tune["median_proxy"], row["median_proxy"]) * (
            scales.get("temperature", pooled) if near_temperature else pooled
        )
        if args.require_robust_gain:
            split_scores = [tune["proxy_scores"], row["proxy_scores"]]
            if validation_rows is not None:
                split_scores.append(validation_rows[row["family"]]["proxy_scores"])
            row["record_ratios"] = [
                [candidate[i] / max(record_score[i], 1e-10) for i in (1, 2)]
                for candidate, record_score in zip(split_scores, record_scores)
            ]
            row["public_control"] = retain_components(public_original, logits(
                public_features, head_by_path[row["family"]],
            ))
            row["robust_gain"] = (
                all(ratio > 1 for split in row["record_ratios"] for ratio in split)
                and row["public_control"]["retention"] >= public_record["retention"]
            )
    if args.require_robust_gain:
        held_out_rows.sort(key=lambda row: (row["robust_gain"], not row["similar_to_tested"],
                                           min(min(split) for split in row["record_ratios"]),
                                           row["public_control"]["retention"]), reverse=True)
    else:
        held_out_rows.sort(key=lambda row: (not row["similar_to_tested"],
                                           row["minimum_control_agreement"] >= 0.70,
                                           row["calibrated_proxy"], min(row["proxy_scores"])), reverse=True)
    (directory / "proposal-comparison.json").write_text(json.dumps(held_out_rows, indent=2) + "\n")
    for row in held_out_rows:
        robust = f" robust_gain={row['robust_gain']}" if args.require_robust_gain else ""
        print(f"{Path(row['family']).name} proxy={row['median_proxy']:.5f} "
              f"calibrated≈{row['calibrated_proxy']:.5f} "
              f"similar_to_tested={row['similar_to_tested']} "
              f"local control agreement={row['minimum_control_agreement']:.3f}{robust}", flush=True)
    if args.promote:
        best = held_out_rows[0]
        if best["median_proxy"] <= 0 or best["similar_to_tested"] or not best.get("robust_gain", True):
            raise SystemExit("No distinct candidate improved the local proxy.")
        source = Path(best["family"])
        digest = sha256(source.read_bytes()).hexdigest()[:12]
        search.proposal_dir.mkdir(exist_ok=True)
        destination = search.proposal_dir / f"agent-{digest}.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save(load_head(source, search.original), temporary)
        temporary.replace(destination)
        choice = search.proposal_dir / "promoted.json"
        temporary = choice.with_suffix(".tmp")
        temporary.write_text(json.dumps({"family": f"proposal-{destination.stem}", "strength": 1.0}))
        temporary.replace(choice)
        print(f"Promoted {source.name} for the next submission.")


if __name__ == "__main__":
    main()
