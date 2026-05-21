"""Generate the four paper figures from outputs/all_results.csv.

Figures (all B/W-readable):
  fig1_scaling.pdf     - accuracy vs PW scale with mean +/- std
  fig2_per_class.pdf   - per-class recall vs PW scale (3 lines)
  fig3_confmat.pdf     - confusion matrices PW-2k vs PW-2k-rebal (median seed)
  fig4_pred_dist.pdf   - predicted-label counts by condition

Output goes to overleaf/.../latex/figures/.

Usage:
    python -m scripts.build_figures
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_CSV = _ROOT / "outputs" / "all_results.csv"
_OUT = _ROOT / "overleaf" / "Class_Asymmetric_Effects_of_Label_Space_Mismatch_in_LoRA_Fine_Tuning_of_Small_Language_Models" / "latex" / "figures"

# Okabe-Ito colorblind-safe palette. Distinct markers and line styles keep
# the figure legible if it is printed in B&W.
COLOR_TRUE      = "#0072B2"  # blue
COLOR_FALSE     = "#D55E00"  # vermillion
COLOR_UNCERTAIN = "#009E73"  # bluish-green
LABEL_STYLE = {
    "True":      dict(color=COLOR_TRUE,      marker="o", linestyle="-",
                      label="True"),
    "False":     dict(color=COLOR_FALSE,     marker="s", linestyle="--",
                      label="False"),
    "Uncertain": dict(color=COLOR_UNCERTAIN, marker="^", linestyle=":",
                      label="Uncertain"),
}

SCALE_ORDER = [0, 2000, 5000, 10000, 20000]
SCALE_TICKLABELS = ["0", "2k", "5k", "10k", "20k"]
CONDITION_ORDER = ["FOLIO", "PW-2k", "PW-2k-rebal", "PW-5k", "PW-10k", "PW-20k"]


def load_rows() -> list[dict]:
    with _CSV.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["pw_scale"] = int(r["pw_scale"])
        r["seed"] = int(r["seed"])
        r["reweighted"] = r["reweighted"] == "True"
        for k in list(r.keys()):
            if k in {"condition"} or k in {"reweighted"}:
                continue
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
    return rows


def mean_std(rows: list[dict], key: str) -> tuple[float, float, int]:
    vals = [r[key] for r in rows]
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)


def filter_by(rows, condition):
    return [r for r in rows if r["condition"] == condition]


def configure_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    })


def fig1_scaling(rows):
    """Accuracy mean +/- std across PW scale; PW-20k single-seed point."""
    scaling_conds = [("FOLIO", 0), ("PW-2k", 2000), ("PW-5k", 5000),
                     ("PW-10k", 10000), ("PW-20k", 20000)]
    xs, means, stds, ns = [], [], [], []
    for label, scale in scaling_conds:
        m, s, n = mean_std(filter_by(rows, label), "accuracy")
        xs.append(scale); means.append(m); stds.append(s); ns.append(n)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    multi = [i for i, n in enumerate(ns) if n > 1]
    single = [i for i, n in enumerate(ns) if n == 1]
    ax.errorbar([xs[i] for i in multi], [means[i] for i in multi],
                yerr=[stds[i] for i in multi],
                marker="o", color="#0072B2", linestyle="-", capsize=3,
                linewidth=1.4, markersize=5,
                label="3-seed mean $\\pm$ std")
    if single:
        ax.scatter([xs[i] for i in single], [means[i] for i in single],
                   marker="x", color="#D55E00", s=44, linewidth=1.5,
                   label="single seed")
    ax.set_xticks(SCALE_ORDER)
    ax.set_xticklabels(SCALE_TICKLABELS)
    ax.set_xlabel("ProofWriter auxiliary scale")
    ax.set_ylabel("FOLIO val accuracy (%)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_ylim(63, 73)
    fig.tight_layout()
    out = _OUT / "fig1_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig2_per_class(rows):
    """Per-class recall (True/False/Uncertain) vs PW scale."""
    scaling_conds = [("FOLIO", 0), ("PW-2k", 2000), ("PW-5k", 5000),
                     ("PW-10k", 10000), ("PW-20k", 20000)]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for cls, key in [("True", "true_recall"), ("False", "false_recall"),
                     ("Uncertain", "uncertain_recall")]:
        xs, means, stds, ns = [], [], [], []
        for label, scale in scaling_conds:
            m, s, n = mean_std(filter_by(rows, label), key)
            xs.append(scale); means.append(m); stds.append(s); ns.append(n)
        multi = [i for i, n in enumerate(ns) if n > 1]
        single = [i for i, n in enumerate(ns) if n == 1]
        style = LABEL_STYLE[cls]
        ax.errorbar([xs[i] for i in multi], [means[i] for i in multi],
                    yerr=[stds[i] for i in multi],
                    capsize=2.5, linewidth=1.1, markersize=4.5, **style)
        if single:
            ax.scatter([xs[i] for i in single], [means[i] for i in single],
                       marker=style["marker"], color=style["color"], s=35)
    ax.set_xticks(SCALE_ORDER)
    ax.set_xticklabels(SCALE_TICKLABELS)
    ax.set_xlabel("ProofWriter auxiliary scale")
    ax.set_ylabel("Per-class recall")
    ax.set_ylim(0.40, 0.90)
    ax.legend(loc="lower left", frameon=False, ncol=3, columnspacing=0.8,
              handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout()
    out = _OUT / "fig2_per_class.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _median_seed_metrics(rows, condition):
    sub = filter_by(rows, condition)
    # Pick the median accuracy seed for a representative confusion matrix.
    sub_sorted = sorted(sub, key=lambda r: r["accuracy"])
    median = sub_sorted[len(sub_sorted) // 2]
    # Load full metrics.json for the confusion matrix.
    dirname_map = {
        "PW-2k": "phi35_lora_folio_pw_2k",
        "PW-2k-rebal": "phi35_lora_folio_pw_2k_rebal",
    }
    metrics_path = (_ROOT / "outputs" / "phi35" / dirname_map[condition]
                    / f"seed_{int(median['seed'])}" / "metrics.json")
    return json.loads(metrics_path.read_text()), int(median["seed"])


def fig3_confmat(rows):
    """Side-by-side confusion matrices for PW-2k vs PW-2k-rebal (median seed)."""
    labels = ["True", "False", "Uncertain"]
    m_un, seed_un = _median_seed_metrics(rows, "PW-2k")
    m_rb, seed_rb = _median_seed_metrics(rows, "PW-2k-rebal")
    cm_un = np.array(m_un["confusion_matrix"])
    cm_rb = np.array(m_rb["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    vmax = max(cm_un.max(), cm_rb.max())
    # pcolormesh draws actual vector rectangles in the PDF, so PDF viewers
    # cannot re-interpolate them. imshow embeds a raster image and some
    # viewers ignore the nearest-neighbor hint, causing diagonal smudges.
    # "Blues" is perceptually uniform and degrades to grayscale cleanly.
    xedges = yedges = np.arange(4)
    for ax, cm, title in [(axes[0], cm_un, f"FOLIO+PW-2k (seed {seed_un})"),
                          (axes[1], cm_rb, f"FOLIO+PW-2k-rebal (seed {seed_rb})")]:
        ax.pcolormesh(xedges, yedges, cm[::-1, :], cmap="Blues",
                      vmin=0, vmax=vmax, shading="flat",
                      edgecolors="white", linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_xticks([0.5, 1.5, 2.5]); ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels[::-1])
        ax.set_xlim(0, 3); ax.set_ylim(0, 3)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True label")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                v = int(cm[i, j])
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j + 0.5, (2 - i) + 0.5, str(v), ha="center",
                        va="center", color=color, fontsize=10)
        ax.grid(False)
    fig.tight_layout()
    out = _OUT / "fig3_confmat.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig4_pred_dist(rows):
    """Predicted-label counts by condition, with ground-truth lines."""
    counts = {}
    for cond in CONDITION_ORDER:
        sub = filter_by(rows, cond)
        t = float(np.mean([r["pred_true"] for r in sub]))
        f = float(np.mean([r["pred_false"] for r in sub]))
        u = float(np.mean([r["pred_uncertain"] for r in sub]))
        counts[cond] = (t, f, u)

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(CONDITION_ORDER))
    w = 0.27
    bars_t = [counts[c][0] for c in CONDITION_ORDER]
    bars_f = [counts[c][1] for c in CONDITION_ORDER]
    bars_u = [counts[c][2] for c in CONDITION_ORDER]
    ax.bar(x - w, bars_t, w, color=COLOR_TRUE,      edgecolor="black",
           label="pred True")
    ax.bar(x,     bars_f, w, color=COLOR_FALSE,     edgecolor="black",
           hatch="//", label="pred False")
    ax.bar(x + w, bars_u, w, color=COLOR_UNCERTAIN, edgecolor="black",
           hatch="..", label="pred Uncertain")
    # ground-truth reference lines. Labels are listed in a small text box
    # in the top-right corner rather than inline, to avoid bar overlap.
    for y in (72, 69, 62):
        ax.axhline(y, color="black", linestyle=":", linewidth=0.7)
    ax.text(0.995, 0.97,
            "GT counts: True 72  |  False 62  |  Uncertain 69",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5",
                      linewidth=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Predicted count (mean across seeds)")
    ax.set_xlim(-0.6, len(CONDITION_ORDER) - 0.4)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.20)
    out = _OUT / "fig4_pred_dist.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    configure_rc()
    _OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fig1_scaling(rows)
    fig2_per_class(rows)
    fig3_confmat(rows)
    fig4_pred_dist(rows)


if __name__ == "__main__":
    main()
