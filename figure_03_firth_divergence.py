"""
figure_03_firth_divergence.py

Figure 3: mean MLE-vs-Firth coefficient divergence, grouped bars
(rule3_passed vs rule3_failed) per rung - the Phase 3 finding validating
Rule 3 independently (divergence 2-8x higher in the rule3_failed group at
every rung with adequate sample in both groups).

Categorical x-axis (not log, unlike Figures 1-2) - grouped bars use
discrete positions regardless of the underlying N values' scale; log
spacing only matters for continuous line/point plots. N=5000/2500 have 0
rule3_failed draws (nothing failed there) and N=50 has 0 rule3_passed
draws (everything failed) - those bars are simply absent rather than
interpolated, which is the honest way to show a genuinely missing group.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from figure_style import RUNGS, fmt_half_up, save_figure

OUTPUT_DIR = "outputs"

GROUP_LABELS = {"rule3_passed": "Rule 3 Passed", "rule3_failed": "Rule 3 Failed"}
GROUP_COLORS = {"rule3_passed": "#4C72B0", "rule3_failed": "#C44E52"}


def load_data() -> list[dict]:
    """
    Load the Firth group summary table (Phase 3).

    Inputs: none.
    Returns:
        list[dict]: 14 rows (7 rungs x 2 groups).
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/firth_group_summary_table.json", "r") as f:
        return json.load(f)


def make_figure() -> plt.Figure:
    """
    Build the grouped-bar Firth divergence figure.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    data = load_data()
    lookup = {(r["rung_N"], r["group"]): r for r in data}

    x = np.arange(len(RUNGS))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, group in enumerate(["rule3_passed", "rule3_failed"]):
        offsets = x + (i - 0.5) * bar_width
        values = []
        for n in RUNGS:
            row = lookup.get((n, group))
            values.append(row["mean_divergence"] if row and row["n_draws"] > 0 else np.nan)
        bars = ax.bar(offsets, values, width=bar_width, label=GROUP_LABELS[group],
                       color=GROUP_COLORS[group])
        # exact value above each bar - bars alone only let a reader estimate
        # against gridlines; explicit labels remove that guesswork.
        labels = [fmt_half_up(v, 3) if not np.isnan(v) else "" for v in values]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in RUNGS])
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Mean MLE-vs-Firth Coefficient Divergence")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)  # headroom for bar-top value labels
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_S2")
    print("Saved " + ", ".join(paths))

