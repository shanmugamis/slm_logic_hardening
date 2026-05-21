"""Walk outputs/phi35/*/seed_*/metrics.json into a single CSV.

Usage:
    python -m scripts.aggregate_results \
        --root outputs/phi35 \
        --output outputs/all_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(_ROOT))

CONDITIONS = [
    ("FOLIO",        "phi35_lora_folio",            0),
    ("PW-2k",        "phi35_lora_folio_pw_2k",      2000),
    ("PW-2k-rebal",  "phi35_lora_folio_pw_2k_rebal", 2000),
    ("PW-5k",        "phi35_lora_folio_pw_5k",      5000),
    ("PW-10k",       "phi35_lora_folio_pw_10k",    10000),
    ("PW-20k",       "phi35_lora_folio_pw_20k",    20000),
]


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for label, dirname, pw_scale in CONDITIONS:
        cond_dir = root / dirname
        if not cond_dir.exists():
            continue
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            metrics_path = seed_dir / "metrics.json"
            if not metrics_path.exists() or metrics_path.stat().st_size == 0:
                continue
            m = json.loads(metrics_path.read_text())
            cr = m["classification_report"]
            pd = m["prediction_distribution"]
            rows.append({
                "condition": label,
                "pw_scale": pw_scale,
                "reweighted": label.endswith("rebal"),
                "seed": int(seed_dir.name.split("_")[1]),
                "accuracy": m["accuracy"],
                "true_precision": cr["True"]["precision"],
                "true_recall": cr["True"]["recall"],
                "true_f1": cr["True"]["f1-score"],
                "false_precision": cr["False"]["precision"],
                "false_recall": cr["False"]["recall"],
                "false_f1": cr["False"]["f1-score"],
                "uncertain_precision": cr["Uncertain"]["precision"],
                "uncertain_recall": cr["Uncertain"]["recall"],
                "uncertain_f1": cr["Uncertain"]["f1-score"],
                "pred_true": pd["True"],
                "pred_false": pd["False"],
                "pred_uncertain": pd["Uncertain"],
                "pred_invalid": pd.get("invalid", 0),
                "n_total": m["n_total"],
            })
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    if not rows:
        raise SystemExit("No rows collected. Check that metrics.json files exist.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=_ROOT / "outputs" / "phi35")
    parser.add_argument("--output", type=Path, default=_ROOT / "outputs" / "all_results.csv")
    args = parser.parse_args()
    rows = collect(args.root)
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
