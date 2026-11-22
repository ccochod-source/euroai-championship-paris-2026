"""Backtest simulator hypotheses against previous official submissions."""

import json
from pathlib import Path
from statistics import mean, median

from compare_proposals import reserve_gpu_split
from simulate_score import evaluate_heads
from tuning_loop import SCORE_ANCHOR, Search, candidate_key, fit_score_scales


def lower_split_scores(search, independent, heads):
    rows = [
        {candidate_key(row): row["proxy_scores"]
         for row in evaluate_heads(split, search.original, heads)}
        for split in (search.features, search.validation, independent)
    ]
    return {
        (family, strength): [
            min(values) for values in zip(*(split[family, strength] for split in rows))
        ]
        for family, strength, _ in heads
    }


def backtest(scores, hypotheses, select_proxy, groups):
    simulated = {
        key: {"median_proxy": select_proxy(values), "calibration_group": groups.get(key[0], key[0])}
        for key, values in hypotheses.items()
    }
    predictions = []
    for row in scores:
        key = candidate_key(row)
        training = [example for example in scores if candidate_key(example) != key]
        pooled, scales = fit_score_scales(training, simulated)
        proxy = simulated[key]
        predicted = proxy["median_proxy"] * scales.get(proxy["calibration_group"], pooled)
        predictions.append({"candidate": key, "predicted": predicted, "official": row["score"]})
    errors = [abs(row["predicted"] - row["official"]) for row in predictions]
    return {"mean_absolute_error": mean(errors), "predictions": predictions}


def main():
    directory = Path("loop")
    search = Search(directory)
    search.load_proposals()
    scores = [
        SCORE_ANCHOR,
        *(row for line in (directory / "history.jsonl").read_text().splitlines()
          if (row := json.loads(line))["status"] == "scored"),
    ]
    heads = [(family, strength, search.head(family, strength))
             for family, strength in map(candidate_key, scores)]
    hypotheses = lower_split_scores(search, reserve_gpu_split(directory), heads)
    groups = search.calibration_groups
    report = {"median": backtest(scores, hypotheses, median, groups)}
    for index in range(len(next(iter(hypotheses.values())))):
        report[f"hypothesis_{index + 1}"] = backtest(scores, hypotheses, lambda values: values[index], groups)
    for name, result in report.items():
        ordered = sorted(result["predictions"], key=lambda row: row["official"])
        inversions = sum(a["predicted"] > b["predicted"]
                         for i, a in enumerate(ordered) for b in ordered[i + 1:])
        result["leave_one_out_order_inversions"] = inversions
        print(f"{name:15} MAE={result['mean_absolute_error']:.6f}, inversions={inversions}")
    (directory / "simulator-backtest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
