"""
figure_01_failure_rate.py

Figure 1: model-validity failure rate vs N, with Wilson score 95% CI
(Phase 6). Log-scale x-axis with actual N values labeled (not log
exponents), per user's explicit spec. Saved as PNG + SVG.
"""

import json

import matplotlib.pyplot as plt

from figure_style import RUNGS, save_figure, set_log_xaxis_with_n_labels

OUTPUT_DIR = "outputs"


def load_data() -> list[dict]:
    """
    Load the CI-augmented model-validity failure table (Phase 6).

    Inputs: none.
    Returns:
        list[dict]: 7 rows, sorted by rung_N descending (as stored).
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/final_model_validity_failure_table_with_ci.json", "r") as f:
        return json.load(f)


def make_figure() -> plt.Figure:
    """
    Build the failure-rate-vs-N figure with Wilson CI error bars.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    data = load_data()
    lookup = {r["rung_N"]: r for r in data}

    ns = RUNGS
    rates = [lookup[n]["model_validity_failure_rate"] for n in ns]
    lowers = [lookup[n]["model_validity_failure_rate_ci_lower"] for n in ns]
    uppers = [lookup[n]["model_validity_failure_rate_ci_upper"] for n in ns]

    yerr_lower = [max(0.0, r - lo) for r, lo in zip(rates, lowers)]
    yerr_upper = [max(0.0, up - r) for r, up in zip(rates, uppers)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        ns, rates, yerr=[yerr_lower, yerr_upper],
        marker="o", markersize=6, linewidth=1.5, capsize=4,
        color="#4C72B0", ecolor="#4C72B0", elinewidth=1.2,
    )

    set_log_xaxis_with_n_labels(ax, ns)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Model/Draw Validity Failure Rate")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_1a")
    print("Saved " + ", ".join(paths))

