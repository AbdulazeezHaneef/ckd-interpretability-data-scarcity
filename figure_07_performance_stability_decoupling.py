"""
figure_07_performance_stability_decoupling.py

Figure 7 (REVISED - dual-axis version rejected on review): the headline
finding from Phase 9 - predictive performance (AUC) stays comparatively
robust even as interpretability stability collapses sharply over the
same range of N.

Rejected approaches and why:
  - Dual y-axis (AUC left, stability right): the two axis ranges are
    arbitrary choices that can visually exaggerate or minimize the gap
    between series independent of what the data says - bad practice in
    general, and specifically undermines a paper built around fixing
    exactly this kind of methodological looseness elsewhere.
  - Normalizing both series to their N=5000 baseline (% retained) on one
    shared axis: fixes the dual-axis problem but introduces a new one -
    AUC's meaningful floor is 0.5 (chance-level discrimination), not 0,
    so "% of N=5000 AUC retained" understates how bad a drop toward 0.5
    actually is. Spearman's floor genuinely is 0, so the same
    normalization is valid for one series but not the other.

Chosen approach: two side-by-side panels (same visual grammar as Figure
4, not a new chart type) - AUC (left, with CI ribbon, horizontal
reference line at 0.5 = chance level) and mean Spearman stability across
methods (right, 0-1 natural range). Every number shown is the real,
literal value - no derived ratios, no axis-range judgment calls to
defend. The decoupling story still reads clearly: one panel barely
moves, the other collapses.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from figure_style import save_figure, set_log_xaxis_with_n_labels

OUTPUT_DIR = "outputs"
METHODS = ["standardized_beta", "permutation", "shap", "lime"]
RUNGS_WITH_DATA = [5000, 2500, 1000, 500, 250, 100]  # N=50 excluded from both series


def load_performance_data() -> dict[int, dict]:
    """
    Load the performance summary table (Phase 9).

    Inputs: none.
    Returns:
        dict[int, dict]: rung_N -> row dict.
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/performance_summary_table.json", "r") as f:
        data = json.load(f)
    return {r["rung_N"]: r for r in data}


def load_mean_stability_data() -> dict[int, float]:
    """
    Load the CI-augmented stability table (Phase 6) and compute a simple
    across-method mean Spearman per rung (descriptive average for this
    figure only - not a recomputed/re-bootstrapped statistic).

    Inputs: none.
    Returns:
        dict[int, float]: rung_N -> mean Spearman across the 4 methods,
            only for rungs where all 4 methods have a value.
    Validity rule checked: none (aggregation only).
    """
    with open(f"{OUTPUT_DIR}/final_stability_table_with_ci.json", "r") as f:
        data = json.load(f)
    by_rung: dict[int, list[float]] = {}
    for r in data:
        if r["spearman_mean"] is not None:
            by_rung.setdefault(r["rung_N"], []).append(r["spearman_mean"])
    return {n: float(np.mean(v)) for n, v in by_rung.items() if len(v) == len(METHODS)}


def make_figure() -> plt.Figure:
    """
    Build the two-panel performance-vs-stability decoupling figure.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    perf = load_performance_data()
    mean_stability = load_mean_stability_data()

    auc_ns = [n for n in RUNGS_WITH_DATA if perf.get(n, {}).get("auc_mean") is not None]
    auc_means = [perf[n]["auc_mean"] for n in auc_ns]
    auc_lowers = [perf[n]["auc_ci_lower"] for n in auc_ns]
    auc_uppers = [perf[n]["auc_ci_upper"] for n in auc_ns]

    stab_ns = [n for n in RUNGS_WITH_DATA if n in mean_stability]
    stab_vals = [mean_stability[n] for n in stab_ns]

    fig, (ax_auc, ax_stab) = plt.subplots(1, 2, figsize=(13, 5.5))

    color_auc = "#4C72B0"
    ax_auc.plot(auc_ns, auc_means, marker="o", markersize=6, linewidth=2, color=color_auc)
    ax_auc.fill_between(auc_ns, auc_lowers, auc_uppers, color=color_auc, alpha=0.15)
    ax_auc.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance level (AUC = 0.5)")
    set_log_xaxis_with_n_labels(ax_auc, auc_ns)
    ax_auc.set_ylim(0.45, 0.85)
    ax_auc.set_xlabel("Sample Size (N)")
    ax_auc.set_ylabel("AUC (held-out)")
    ax_auc.set_title("Predictive Performance (AUC)")
    ax_auc.legend(loc="lower right")
    ax_auc.grid(True, which="both", axis="both", alpha=0.3)

    color_stab = "#C44E52"
    ax_stab.plot(stab_ns, stab_vals, marker="s", markersize=6, linewidth=2, color=color_stab)
    set_log_xaxis_with_n_labels(ax_stab, stab_ns)
    ax_stab.set_ylim(0.0, 1.0)
    ax_stab.set_xlabel("Sample Size (N)")
    ax_stab.set_ylabel("Mean Spearman Stability (across 4 methods)")
    ax_stab.set_title("Interpretability Stability")
    ax_stab.grid(True, which="both", axis="both", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 1])

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_5")
    print("Saved " + ", ".join(paths))


