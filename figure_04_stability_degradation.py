"""
figure_04_stability_degradation.py

Figure 4: the paper's core degradation story. Two side-by-side panels -
Jaccard@5 (left) and Spearman (right) - vs N, one line per method, with
bootstrap 95% CI ribbons (Phase 6). N=50 excluded (0 valid draws for
every method there - nothing to plot). Log-scale x-axis with real N
labels, consistent with Figures 1-2. Shared legend, shared method colors
(figure_style.METHOD_COLORS) for consistency across every figure that
shows all 4 methods.
"""

import json

import matplotlib.pyplot as plt

from figure_style import METHOD_COLORS, METHOD_LABELS, save_figure, set_log_xaxis_with_n_labels

OUTPUT_DIR = "outputs"
METHODS = ["standardized_beta", "permutation", "shap", "lime"]
RUNGS_WITH_VALID_DRAWS = [5000, 2500, 1000, 500, 250, 100]  # N=50 excluded


def load_data() -> list[dict]:
    """
    Load the CI-augmented stability table (Phase 6).

    Inputs: none.
    Returns:
        list[dict]: 28 rows (7 rungs x 4 methods).
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/final_stability_table_with_ci.json", "r") as f:
        return json.load(f)


def plot_metric_panel(ax: plt.Axes, lookup: dict, metric_key: str,
                       ci_lower_key: str, ci_upper_key: str, title: str) -> None:
    """
    Plot one metric's method lines with CI ribbons onto one axes.

    Inputs:
        ax (plt.Axes): the axes to draw on.
        lookup (dict): (rung_N, method) -> row dict.
        metric_key (str): the point-estimate field name (e.g. "spearman_mean").
        ci_lower_key (str): the CI lower-bound field name.
        ci_upper_key (str): the CI upper-bound field name.
        title (str): panel title.
    Returns:
        None. Draws onto ax in place.
    Validity rule checked: none (presentation only).
    """
    for method in METHODS:
        ns, means, lowers, uppers = [], [], [], []
        for n in RUNGS_WITH_VALID_DRAWS:
            row = lookup.get((n, method))
            if row is None or row[metric_key] is None:
                continue
            ns.append(n)
            means.append(row[metric_key])
            lowers.append(row[ci_lower_key])
            uppers.append(row[ci_upper_key])

        color = METHOD_COLORS[method]
        ax.plot(ns, means, marker="o", markersize=5, linewidth=1.5,
                color=color, label=METHOD_LABELS[method])
        ax.fill_between(ns, lowers, uppers, color=color, alpha=0.15)

    set_log_xaxis_with_n_labels(ax, RUNGS_WITH_VALID_DRAWS)
    ax.set_xlabel("Sample Size (N)")
    ax.set_title(title)
    ax.grid(True, which="both", axis="both", alpha=0.3)


def make_figure() -> plt.Figure:
    """
    Build the two-panel stability degradation figure.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    data = load_data()
    lookup = {(r["rung_N"], r["method"]): r for r in data}

    fig, (ax_jaccard, ax_spearman) = plt.subplots(1, 2, figsize=(13, 5.5))

    plot_metric_panel(ax_jaccard, lookup, "jaccard_top5_mean",
                       "jaccard_top5_ci_lower", "jaccard_top5_ci_upper",
                       "Top-5 Jaccard Stability")
    ax_jaccard.set_ylabel("Top-5 Jaccard Index")
    ax_jaccard.set_ylim(-0.02, 1.02)

    plot_metric_panel(ax_spearman, lookup, "spearman_mean",
                       "spearman_ci_lower", "spearman_ci_upper",
                       "Full-Ranking Spearman Stability")
    ax_spearman.set_ylabel("Spearman Correlation")
    ax_spearman.set_ylim(-0.05, 1.02)

    handles, labels = ax_jaccard.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_2")
    print("Saved " + ", ".join(paths))

