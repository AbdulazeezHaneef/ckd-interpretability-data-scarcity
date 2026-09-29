"""
figure_01b_failure_rate_decomposed.py

Figure 1b: decomposes Figure 1's overall model-validity failure rate by
which rule (1=EPV floor, 2=convergence, 3=coefficient-magnitude) was
responsible, at each rung. Companion to Figure 1, not a replacement -
Figure 1's overall rate + Wilson CI stays as the primary uncertainty
figure; this one adds the per-rule breakdown Figure 1 alone can't show.

Data: already exists, no rerun needed. rule_1_rate + rule_2_rate +
rule_3_rate sum exactly to model_validity_failure_rate at every rung
(verified: N=100 -> 0.20+0.06+0.50=0.76; N=50 -> 0.95+0.03+0.02=1.00).

Categorical x-axis (not literal log-scale), same convention as Figure 3 -
stacked bars on a true log axis produce visually inconsistent bar widths
across a 100x N range (50 to 5000); evenly-spaced categorical positions
labeled with the real N values achieve the same readability without that
distortion.

No error bars (component-level Wilson CIs are not cleanly justified for
stacked proportions of a shared denominator - acknowledged limitation,
not attempted). Each segment labeled with its exact percentage value.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from figure_style import RUNGS, fmt_half_up, save_figure

OUTPUT_DIR = "outputs"

RULE_LABELS = {
    "rule_1_rate": "Rule 1: EPV Floor",
    "rule_2_rate": "Rule 2: Convergence",
    "rule_3_rate": "Rule 3: Coefficient Magnitude",
}
RULE_COLORS = {
    "rule_1_rate": "#4C72B0",
    "rule_2_rate": "#DD8452",
    "rule_3_rate": "#C44E52",
}
RULE_ORDER = ["rule_1_rate", "rule_2_rate", "rule_3_rate"]


def load_data() -> list[dict]:
    """
    Load the model-validity failure table (Phase 2/3) - the plain
    version (not the CI-augmented one) is sufficient here since no CIs
    are plotted; either file has the same rule_1/2/3_rate fields.

    Inputs: none.
    Returns:
        list[dict]: 7 rows.
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/final_model_validity_failure_table.json", "r") as f:
        return json.load(f)


def make_figure() -> plt.Figure:
    """
    Build the stacked-bar per-rule failure decomposition figure.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only). Sanity-asserts that
        the three rule rates sum to the total at every rung, so a data
        error would surface as a loud crash rather than a silently wrong
        chart.
    """
    data = load_data()
    lookup = {r["rung_N"]: r for r in data}

    for n in RUNGS:
        row = lookup[n]
        total_from_rules = sum(row[k] for k in RULE_ORDER)
        assert abs(total_from_rules - row["model_validity_failure_rate"]) < 1e-9, (
            f"N={n}: rule rates sum to {total_from_rules}, "
            f"expected {row['model_validity_failure_rate']}"
        )

    x = np.arange(len(RUNGS))
    fig, ax = plt.subplots(figsize=(9, 6))

    bottoms = np.zeros(len(RUNGS))
    for rule_key in RULE_ORDER:
        vals = np.array([lookup[n][rule_key] for n in RUNGS])
        bars = ax.bar(x, vals, bottom=bottoms, label=RULE_LABELS[rule_key],
                       color=RULE_COLORS[rule_key])
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.02:  # skip labels on segments too small to read cleanly
                ax.text(x[i], b + v / 2, f"{fmt_half_up(v*100, 1)}%", ha="center", va="center",
                         color="white", fontsize=8)
        bottoms += vals

    for i, n in enumerate(RUNGS):
        total = lookup[n]["model_validity_failure_rate"]
        ax.text(x[i], total + 0.02, f"{fmt_half_up(total*100, 1)}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in RUNGS])
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Failure Rate")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_1b")
    print("Saved " + ", ".join(paths))

