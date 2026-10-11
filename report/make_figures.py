"""Generates the report's figures from data/evaluation/results.csv.

The oracle-pointer cross-validation numbers (F1=0.428, FAR=2.3%) are not
written to a CSV anywhere - they come from the leave-one-script-out CV
reported in the report text (Section "Train/inference mismatch") - so
they're hardcoded here rather than read from a file.

Usage (from report/):
    python make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "data" / "evaluation" / "results.csv"
SWEEP_CSV = REPO_ROOT / "data" / "evaluation" / "sweep_results.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY_INK, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY_INK,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "ytick.left": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
})


def read_overall(results):
    rows = {r["matcher"]: r for r in results if r["condition"] == "overall"}
    return rows


def read_rows(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h), (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color=SECONDARY_INK,
        )


def fig_overall(results):
    overall = read_overall(results)
    methods = ["baseline", "hybrid", "classifier"]
    labels = ["B1", "B2", "B3"]
    acc = [float(overall[m]["accuracy"]) * 100 for m in methods]
    far = [float(overall[m]["false_advance_rate"]) * 100 for m in methods]
    f1 = [float(overall[m]["f1_advance"]) for m in methods]
    n_lat = [int(overall[m]["n_latencies_measured"]) for m in methods]
    n_trans = [int(overall[m]["n_transitions"]) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    x = range(len(methods))
    w = 0.32
    b1 = ax1.bar([i - w / 2 for i in x], acc, width=w, color=BLUE, label="Accuracy")
    b2 = ax1.bar([i + w / 2 for i in x], far, width=w, color=ORANGE, label="False-advance rate")
    bar_labels(ax1, b1, "{:.1f}%")
    bar_labels(ax1, b2, "{:.1f}%")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Percent of windows")
    ax1.set_title("(a) Accuracy & false-advance rate", pad=28)
    ax1.set_ylim(0, max(acc + far) * 1.35)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)

    bars = ax2.bar(list(x), f1, width=0.5, color=AQUA)
    bar_labels(ax2, bars, "{:.3f}")
    for i, (lat, trans) in enumerate(zip(n_lat, n_trans)):
        ax2.annotate(
            f"{lat}/{trans} detected", (i, 0), xytext=(0, -18),
            textcoords="offset points", ha="center", va="top",
            fontsize=7, color=MUTED,
        )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("F1 (advance class)")
    ax2.set_title("(b) F1 & transitions detected", pad=28)
    ax2.set_ylim(0, max(f1) * 1.5)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_DIR / "fig_overall.pdf")
    plt.close(fig)


def fig_condition(results):
    conditions = ["literal", "paraphrase", "skip"]
    cond_labels = ["Literal", "Paraphrase", "Skip"]
    by_cond = {r["condition"]: r for r in results if r["matcher"] == "baseline" and r["condition"] in conditions}
    by_cond_cls = {r["condition"]: r for r in results if r["matcher"] == "classifier" and r["condition"] in conditions}

    far_baseline = [float(by_cond[c]["false_advance_rate"]) * 100 for c in conditions]
    far_classifier = [float(by_cond_cls[c]["false_advance_rate"]) * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    x = range(len(conditions))
    w = 0.32
    b1 = ax.bar([i - w / 2 for i in x], far_baseline, width=w, color=BLUE, label="B1 / B2")
    b2 = ax.bar([i + w / 2 for i in x], far_classifier, width=w, color=ORANGE, label="B3")
    bar_labels(ax, b1, "{:.1f}%")
    bar_labels(ax, b2, "{:.1f}%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("False-advance rate (%)")
    ax.set_ylim(0, max(far_classifier) * 1.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "fig_condition.pdf")
    plt.close(fig)


def fig_mismatch():
    # Oracle-pointer CV vs. live self-pointer replay, B3 only (Section "Train/inference mismatch").
    oracle_f1, live_f1 = 0.428, 0.227
    oracle_far, live_far = 2.3, 11.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.3, 2.6))
    labels = ["Oracle\npointer (CV)", "Live\nself-pointer"]
    colors = [BLUE, ORANGE]

    bars = ax1.bar(labels, [oracle_f1, live_f1], color=colors, width=0.55)
    bar_labels(ax1, bars, "{:.3f}")
    ax1.set_ylabel("F1 (advance class)")
    ax1.set_ylim(0, oracle_f1 * 1.4)
    ax1.set_title("(a) F1", fontsize=9)
    ax1.tick_params(axis="x", labelsize=7.5)

    bars = ax2.bar(labels, [oracle_far, live_far], color=colors, width=0.55)
    bar_labels(ax2, bars, "{:.1f}%")
    ax2.set_ylabel("False-advance rate (%)")
    ax2.set_ylim(0, live_far * 1.3)
    ax2.set_title("(b) False-advance rate", fontsize=9)
    ax2.tick_params(axis="x", labelsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_mismatch.pdf")
    plt.close(fig)


def fig_confirm_sweep():
    rows = [r for r in read_rows(SWEEP_CSV) if r["matcher"] == "classifier"]
    rows.sort(key=lambda r: int(r["value"]))
    cw = [int(r["value"]) for r in rows]
    far = [float(r["far"]) * 100 for r in rows]
    f1 = [float(r["f1"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.3, 2.6))

    ax1.plot(cw, far, color=ORANGE, marker="o", markersize=5, linewidth=2)
    ax1.set_xlabel("confirm_windows")
    ax1.set_ylabel("False-advance rate (%)")
    ax1.set_xticks(cw)
    ax1.set_ylim(0, max(far) * 1.25)
    ax1.set_title("(a) FAR", fontsize=9)

    ax2.plot(cw, f1, color=BLUE, marker="o", markersize=5, linewidth=2)
    ax2.set_xlabel("confirm_windows")
    ax2.set_ylabel("F1 (advance class)")
    ax2.set_xticks(cw)
    ax2.set_ylim(0, max(f1) * 1.25)
    ax2.set_title("(b) F1", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_confirm_sweep.pdf")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = read_rows(RESULTS_CSV)
    fig_overall(results)
    fig_condition(results)
    fig_mismatch()
    if SWEEP_CSV.exists():
        fig_confirm_sweep()
    print(f"saved figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
