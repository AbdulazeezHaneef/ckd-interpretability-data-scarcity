"""
figure_02_epv_sensitivity.py

Figure 2: model-validity failure rate vs N, one line per EPV threshold
(2, 5, 10) - shows the threshold-sensitivity finding from Phase 3 (the
qualitative claim - failure rises sharply as N shrinks - holds at every
threshold, but the specific failure-rate values at N=100-250 are highly
threshold-dependent). Log-scale x-axis with real N labels, consistent
with Figure 1. Saved as PNG + SVG.
"""

import json

import matplotlib.pyplot as plt

from figure_style import RUNGS, save_figure, set_log_xaxis_with_n_labels

OUTPUT_DIR = "outputs"

EPV_MULTIPLIERS = [2, 5, 10]
EPV_COLORS = {2: "#4C72B0", 5: "#DD8452", 10: "#C44E52"}
EPV_MARKERS = {2: "o", 5: "s", 10: "^"}


def load_data() -> list[dict]:
    """
    Load the EPV sensitivity model-validity table (Phase 3).

    Inputs: none.
    Returns:
        list[dict]: 21 rows (7 rungs x 3 EPV multipliers).
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/epv_sensitivity_model_validity_table.json", "r") as f:
        return json.load(f)


def make_figure() -> plt.Figure:
    """
    Build the EPV-sensitivity failure-rate-vs-N figure, one line per
    EPV multiplier.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    data = load_data()
    lookup = {(r["rung_N"], r["epv_multiplier"]): r for r in data}

    fig, ax = plt.subplots(figsize=(7, 5))

    for mult in EPV_MULTIPLIERS:
        rates = [lookup[(n, mult)]["model_validity_failure_rate"] for n in RUNGS]
        ax.plot(
            RUNGS, rates,
            marker=EPV_MARKERS[mult], markersize=6, linewidth=1.5,
            color=EPV_COLORS[mult], label=f"EPV = {mult}",
        )

    set_log_xaxis_with_n_labels(ax, RUNGS)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Model/Draw Validity Failure Rate")
    ax.legend(title="EPV Threshold", loc="upper right")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_S1")
    print("Saved " + ", ".join(paths))

