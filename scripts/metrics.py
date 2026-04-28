"""Compute classification metrics and deductive consistency from saved results.

Usage:
    python -m scripts.metrics \
        --results_path outputs/phi35/phi35_baseline/results.json

    python -m scripts.metrics \
        --results_path outputs/llama8b/llama8b_baseline/results.json \
        --output_path outputs/llama8b/llama8b_baseline/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def build_classification_and_confusion_report(results: list[dict]) -> dict:
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["predicted"] for r in results]
    labels = ["True", "False", "Uncertain"]

    report_str = classification_report(y_true=y_true, y_pred=y_pred, labels=labels)
    print("\nCLASSIFICATION REPORT:")
    print(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nCONFUSION MATRIX (rows=actual, cols=predicted):")
    print(cm_df.to_string())

    return {
        "classification_report": classification_report(
            y_true=y_true, y_pred=y_pred, labels=labels, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
    }


def build_prediction_distribution(results: list[dict]) -> dict:
    counts = Counter(r["predicted"] for r in results)
    return {
        "True": counts.get("True", 0),
        "False": counts.get("False", 0),
        "Uncertain": counts.get("Uncertain", 0),
        "invalid": counts.get("invalid", 0),
    }


def build_deductive_consistency(results: list[dict]) -> dict:
    bins: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        count = result.get("no_of_premises", 0)
        if count <= 2:
            bin_name = "1-2"
        elif count <= 4:
            bin_name = "3-4"
        else:
            bin_name = "5+"

        bins[bin_name]["total"] += 1
        if result["predicted"] == result["ground_truth"]:
            bins[bin_name]["correct"] += 1

    print("\nDEDUCTIVE CONSISTENCY (accuracy by premise count):")
    consistency = {}
    for bin_name in ["1-2", "3-4", "5+"]:
        if bin_name in bins:
            correct = bins[bin_name]["correct"]
            total = bins[bin_name]["total"]
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            print(f"  Premises {bin_name}:  Accuracy {accuracy:.1f}%  ({correct}/{total})")
            consistency[bin_name] = {"accuracy": accuracy, "correct": correct, "total": total}

    return consistency


def run_metrics(results_path: str, output_path: str | None = None):
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    n_total = data.get("n_total", len(results))
    n_invalid = data.get("n_invalid", sum(1 for r in results if r["predicted"] == "invalid"))

    print(f"\nExperiment: {data.get('experiment_id', 'unknown')}")
    print(f"Model:      {data.get('model_id', 'unknown')}")
    print(f"Prompting:  {data.get('prompting', 'standard')}")
    print(f"Accuracy:   {data.get('accuracy', '?')}%  ({data.get('n_correct', '?')}/{n_total})")

    classification_metrics = build_classification_and_confusion_report(results)
    prediction_distribution = build_prediction_distribution(results)
    consistency = build_deductive_consistency(results)

    metrics = {
        "experiment_id": data.get("experiment_id"),
        "model_id": data.get("model_id"),
        "model_key": data.get("model_key"),
        "mode": data.get("mode"),
        "prompting": data.get("prompting", "standard"),
        "accuracy": data.get("accuracy"),
        "n_correct": data.get("n_correct"),
        "n_total": n_total,
        "n_invalid": n_invalid,
        "invalid_rate": round(n_invalid / n_total * 100, 2) if n_total else 0,
        **classification_metrics,
        "prediction_distribution": prediction_distribution,
        "deductive_consistency": consistency,
    }

    if output_path is None:
        output_path = str(Path(results_path).parent / "metrics.json")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics from evaluation results")
    parser.add_argument("--results_path", type=str, required=True,
        help="Path to results.json produced by scripts/eval.py")
    parser.add_argument("--output_path", type=str, default=None,
        help="Where to save metrics.json (default: same folder as results.json)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_metrics(args.results_path, args.output_path)
