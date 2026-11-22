"""Run a time-boxed Unlearning search with official leaderboard feedback."""

import argparse
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import re
import sys
import time

import requests
import torch

from checkpoint_changes import load
from simulate_score import candidate_head, evaluate_heads, get_features
import submission
from validation import validate_submission


BASELINE = 0.0001354019235455436
COOLDOWN = 605
FIRST_ELIGIBLE = datetime.fromisoformat("2026-11-21T16:47:30+00:00").timestamp()


def emit(history, status, **details):
    row = {"status": status, "time": datetime.now().astimezone().isoformat(), **details}
    serialized = json.dumps(row, sort_keys=True)
    with history.open("a") as logfile:
        logfile.write(serialized + "\n")
    print(serialized, flush=True)


HEAD_NAMES = ("head.weight", "head.bias")
SCORE_ANCHOR = {"family": "temperature", "strength": 1.5, "score": BASELINE}
STRENGTH_RANGES = {"head": (0.85, 1.22), "head2": (0.85, 1.45),
                   "hybrid": (0.8, 1.60), "temperature": (1.25, 5.25)}


def candidate_key(row):
    return row["family"], row["strength"]


def fit_score_scales(scores, simulated, proxy_field="median_proxy"):
    # Ratios from submitted proposal families transfer to nearby proposals.
    # Borrow one pooled observation for the original search families.
    ratios = {}
    for observed in scores:
        row = simulated.get(candidate_key(observed))
        if row is None or row[proxy_field] <= 0.001:
            continue
        family = row.get("calibration_group", observed["family"])
        ratio = max(0.00001, observed["score"]) / row[proxy_field]
        ratios.setdefault(family, []).append(math.log(ratio))
    values = [value for group in ratios.values() for value in group]
    if not values:
        return 1.0, {}
    pooled = sum(values) / len(values)
    scale = {
        family: math.exp((sum(group) + pooled) / (len(group) + 1)
                         if family in STRENGTH_RANGES else sum(group) / len(group))
        for family, group in ratios.items()
    }
    return math.exp(pooled), scale


def eligible_for_submission(row, official_best, minimum_estimate):
    return row["calibrated_proxy"] > official_best and row["calibrated_proxy"] >= minimum_estimate


def next_candidate(rows, scores, tried, target=0.07, minimum_estimate=0.0):
    best = max(scores, key=lambda row: row["score"])
    available = [
        row for row in rows
        if candidate_key(row) not in tried
        and eligible_for_submission(row, best["score"], minimum_estimate)
    ]
    if not available:
        return None

    # When the best official result is far from the target, use some slots
    # for meaningfully different settings that performed well locally.
    def priority(row):
        return row["calibrated_proxy"]

    if best["score"] < target / 3 and len(scores) % 3 != 0:
        exploratory = [
            row for row in available
            if row["synthetic_control_agreement"] >= 0.70
            and not any(
                candidate_key(observed) != candidate_key(SCORE_ANCHOR)
                and observed["family"] == row["family"] == "temperature"
                and observed["score"] < best["score"] * 0.7
                for observed in scores
            )
            and all(family != row["family"] or abs(strength - row["strength"]) >= (
                0.35 if family == "temperature" else 0.04
            ) for family, strength in tried)
        ]
        if exploratory:
            return max(exploratory, key=priority)

    explore = len(scores) % 3 == 2
    alternatives = [row for row in available if row["family"] != best["family"]
                    and row["family"] != "temperature"]
    nearby = [row for row in available if row["family"] == best["family"]
              and 0.009 <= abs(row["strength"] - best["strength"]) <= 0.06]
    if explore and alternatives:
        return max(alternatives, key=priority)
    return max(nearby or available, key=priority)


class Search:
    def __init__(self, directory, submission_validation=None):
        self.features = get_features(directory / "simulator-features.pt", 192, 2675)
        self.validation = None
        second_split = directory / "simulator-second-split.pt"
        if second_split.is_file():
            self.validation = get_features(second_split, 256, 2775)
        weights = load("model.pt")
        self.original = {key: weights[key].clone() for key in HEAD_NAMES}
        del weights
        self.modified = load("output/head-4.0-15901766-changes.pt")
        self.alternate = load("output/head-2.0-15901765-changes.pt")
        self.proposals = {}
        self.proposal_dir = directory / "proposals"
        self.calibration_groups = {}
        self.submission_validation = submission_validation
        self.rows = {}
        self.score_scales = (1.0, {})
        self.batch = 0

    def load_proposals(self):
        groups = self.proposal_dir / "calibration-groups.json"
        if groups.is_file():
            self.calibration_groups = json.loads(groups.read_text())
        for path in sorted(self.proposal_dir.glob("*.pt")):
            family = f"proposal-{path.stem}"
            if family in self.proposals:
                continue
            head = load(path)
            if set(head) != set(HEAD_NAMES) or any(
                not isinstance(head[name], torch.Tensor)
                or head[name].shape != self.original[name].shape
                or head[name].dtype != torch.float32
                or not torch.isfinite(head[name]).all().item()
                for name in HEAD_NAMES
            ):
                raise ValueError(f"Invalid proposal: {path}")
            self.proposals[family] = head

    def head(self, family, strength):
        if family in self.proposals:
            return {key: torch.lerp(value, self.proposals[family][key], strength)
                    for key, value in self.original.items()}
        return candidate_head(family, strength, self.original, self.modified, self.alternate)

    def evaluate(self, scores):
        self.load_proposals()
        best = max(scores, key=lambda row: row["score"])
        candidates = {candidate_key(row) for row in scores}
        ranges = {**STRENGTH_RANGES, **dict.fromkeys(self.proposals, (0.85, 1.15))}
        for family, (low, high) in ranges.items():
            # Offset the grid each batch so the cached proxy sees fresh settings.
            shift = (self.batch * 0.61803398875) % 1
            candidates.update((family, round(low + (high - low) * (i + shift) / 12, 3))
                              for i in range(12))
            if family in self.proposals:
                candidates.add((family, 1.0))
        family, strength = candidate_key(best)
        if family in ranges:
            low, high = ranges[family]
            radius = max(0.02, 0.065 / (1 + self.batch / 12))
            candidates.update((family, round(strength + radius * i / 8, 3))
                              for i in range(-8, 9) if low <= strength + radius * i / 8 <= high)
        candidates = sorted(candidates.difference(self.rows))
        self.batch += 1
        heads = [(family, strength, self.head(family, strength)) for family, strength in candidates]
        rows = evaluate_heads(self.features, self.original, heads)
        for row in rows:
            row["strong_forget_proxy"] = row["proxy_scores"][2]
        for split in (self.validation, self.submission_validation):
            if split is None:
                continue
            checked = {candidate_key(row): row for row in evaluate_heads(split, self.original, heads)}
            for row in rows:
                secondary = checked[candidate_key(row)]
                if split is self.validation:
                    row["tuning_proxy"] = row["median_proxy"]
                    row["validation_proxy"] = secondary["median_proxy"]
                row["median_proxy"] = min(row["median_proxy"], secondary["median_proxy"])
                row["strong_forget_proxy"] = min(row["strong_forget_proxy"], secondary["proxy_scores"][2])
                row["synthetic_control_agreement"] = min(row["synthetic_control_agreement"],
                                                         secondary["synthetic_control_agreement"])
        self.rows.update((candidate_key(row), row) for row in rows)
        proxy = "strong_forget_proxy" if self.submission_validation is not None else "median_proxy"
        for row in self.rows.values():
            row["calibration_group"] = self.calibration_groups.get(row["family"], row["family"])
        self.score_scales = fit_score_scales(scores, self.rows, proxy)
        pooled, scales = self.score_scales
        for row in self.rows.values():
            scale = scales.get(row["calibration_group"], pooled)
            row["calibrated_proxy"] = min(1.0, row[proxy] * scale)
        return rows

    def promoted(self, tried):
        path = self.proposal_dir / "promoted.json"
        if not path.is_file():
            return None
        choice = json.loads(path.read_text())
        key = candidate_key(choice)
        return self.rows.get(key) if key not in tried else None


def prepare(family, strength, directory, search):
    path = directory / f"{family}-{strength:g}.pt"
    if path.exists():
        return path
    weights = load("model.pt")
    weights.update(search.head(family, strength))
    temporary = path.with_suffix(".tmp")
    torch.save(weights, temporary)
    temporary.replace(path)
    return path


def score_checkpoint(client, path, token):
    try:
        url = f"{submission.SERVER_URL.rstrip('/')}/submit/{submission.TASK_ID}"
        with path.open("rb") as handle:
            response = client.post(
                url, headers={"X-API-Key": token}, files={"file": (path.name, handle)},
                timeout=(240, 600), allow_redirects=False,
            )
    except requests.RequestException as exc:
        return {"status": "network_error", "reason": type(exc).__name__}
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if response.status_code == 429:
        remaining = re.search(r"(\d+) seconds", str(body.get("detail", "")))
        return {"status": "rate_limited", "seconds": int(remaining[1]) + 4 if remaining else COOLDOWN}
    score = body.get("score")
    if response.ok and body.get("status") == "success" and isinstance(score, (float, int)):
        if math.isfinite(score) and 0 <= score <= 1:
            return {"status": "scored", "score": score, "improved": body.get("improved") is True}
    return {"status": "rejected", "http": response.status_code}


def run(token, output, deadline, first_eligible, target, minimum_estimate=0.0):
    from compare_proposals import reserve_gpu_split

    history = output / "history.jsonl"
    prior = [json.loads(row) for row in history.read_text().splitlines()] if history.exists() else []
    scores = [SCORE_ANCHOR, *(row for row in prior if row["status"] == "scored")]
    tried = {candidate_key(row) for row in scores}
    tried.update(candidate_key(row) for row in prior if row["status"] in {"network_error", "rejected"})
    previous_uploads = [
        datetime.fromisoformat(row["time"]).timestamp() + COOLDOWN
        for row in prior if row["status"] == "uploading"
    ]
    rate_limits = [
        datetime.fromisoformat(row["time"]).timestamp() + row["seconds"]
        for row in prior if row["status"] == "rate_limited"
    ]
    next_at = max([first_eligible, *previous_uploads, *rate_limits])
    emit(history, "resumed" if prior else "started", deadline=deadline, first_eligible=next_at)
    independent = (
        reserve_gpu_split(output)
        if all((output / name).exists() for name in ("gpu-cache-a.pt", "gpu-cache-b.pt"))
        else None
    )
    search = Search(output, submission_validation=independent)
    with requests.Session() as client, (output / "search.jsonl").open("a", buffering=1) as simulations:
        while time.time() < deadline:
            rows = search.evaluate(scores)
            for row in rows:
                simulations.write(json.dumps(row, sort_keys=True) + "\n")
            official_best = max(row["score"] for row in scores)
            chosen = search.promoted(tried)
            if chosen is None or not eligible_for_submission(chosen, official_best, minimum_estimate):
                chosen = next_candidate(search.rows.values(), scores, tried, target, minimum_estimate)
            best_estimate = max(
                (row["calibrated_proxy"] for row in search.rows.values()
                 if candidate_key(row) not in tried),
                default=None,
            )
            emit(history, "simulated", evaluated=len(rows), total=len(search.rows),
                 selected=list(candidate_key(chosen)) if chosen else None,
                 selection_proxy=(chosen["strong_forget_proxy" if independent is not None else "median_proxy"]
                                  if chosen else None),
                 calibrated_proxy=chosen["calibrated_proxy"] if chosen else None,
                 official_best=official_best,
                 best_estimate=best_estimate, minimum_estimate=minimum_estimate,
                 score_scales=search.score_scales)
            now = time.time()
            if now < next_at or chosen is None:
                wake_at = min(now + 30, deadline)
                if now < next_at:
                    wake_at = min(wake_at, next_at)
                time.sleep(max(0, wake_at - now))
                continue
            if time.time() >= deadline:
                break
            family, strength = candidate_key(chosen)
            path = prepare(family, strength, output, search)
            validate_submission(path)
            details = {"family": family, "strength": strength, "checkpoint": str(path)}
            emit(history, "validated", **details)
            started = time.time()
            if started >= deadline:
                break
            emit(history, "uploading", **details)
            result = score_checkpoint(client, path, token)
            status = result.pop("status")
            emit(history, status, **details, **result)
            if status == "rate_limited":
                next_at = time.time() + result["seconds"]
                continue
            tried.add((family, strength))
            if status == "scored":
                scores.append({**details, **result})
            elif result.get("http") in (401, 403, 404):
                break
            # A failed connection may still have reached the scorer. Wait for
            # the complete cooldown before another request.
            next_at = started + COOLDOWN
    emit(history, "done", count=len(scores) - 1, best=max(r["score"] for r in scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True, type=float, help="Unix time to stop scheduling")
    parser.add_argument("--first-eligible", type=float, default=FIRST_ELIGIBLE)
    parser.add_argument("--target", type=float, default=0.07, help="Target official score")
    parser.add_argument("--minimum-estimate", type=float, default=0.0,
                        help="Minimum calibrated estimate required for submission")
    parser.add_argument("--output", type=Path, default=Path("loop"))
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        parser.error("Pass the team token through stdin")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(token, args.output, args.deadline, args.first_eligible, args.target, args.minimum_estimate)


if __name__ == "__main__":
    main()
