"""
figure_06_significance_heatmap.py

Figure 6: visualizes all 72 paired bootstrap comparisons from Phase 7
(6 method pairs x 6 rungs x 2 metrics) as two side-by-side heatmaps -
Spearman (left) and Jaccard@5 (right) - since a single panel can't
cleanly carry both metrics at 6x6 size without clutter.

Cell color: binary (significant at FDR 0.05 vs. not) - matches how the
finding is actually used in Discussion ("is X significantly more stable
than Y"), rather than a continuous q-value scale that would be harder to
read at this grid size. Each cell is annotated with its point-estimate
difference (stability_A - stability_B) so the reader gets both the
significance call AND the effect size in one glance.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from figure_style import fmt_half_up, save_figure

OUTPUT_DIR = "outputs"
RUNGS_TESTED = [100, 250, 500, 1000, 2500, 5000]  # ascending; N=50 excluded (Phase 7 scope)
METHOD_PAIR_ORDER = [
    ("standardized_beta", "permutation"),
    ("standardized_beta", "shap"),
    ("standardized_beta", "lime"),
    ("permutation", "shap"),
    ("permutation", "lime"),
    ("shap", "lime"),
]

SHORT_LABELS = {
    "standardized_beta": "Beta",
    "permutation": "Perm",
    "shap": "SHAP",
    "lime": "LIME",
}


def load_data() -> list[dict]:
    """
    Load the formal comparison table (Phase 7).

    Inputs: none.
    Returns:
        list[dict]: 72 rows.
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/formal_comparison_table.json", "r") as f:
        return json.load(f)


def build_grid(lookup: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the significance grid and annotation-value grid for one metric.

    Inputs:
        lookup (dict): (rung_N, method_a, method_b, metric) -> row dict.
        metric (str): "spearman" or "jaccard_top5".
    Returns:
        tuple[np.ndarray, np.ndarray]: (significance grid, 0/1 int array;
            diff grid, float array), both shaped (n_pairs, n_rungs).
    Validity rule checked: none (data reshaping only).
    """
    sig_grid = np.zeros((len(METHOD_PAIR_ORDER), len(RUNGS_TESTED)), dtype=int)
    diff_grid = np.full((len(METHOD_PAIR_ORDER), len(RUNGS_TESTED)), np.nan)

    for i, pair in enumerate(METHOD_PAIR_ORDER):
        for j, N in enumerate(RUNGS_TESTED):
            row = lookup[(N, pair[0], pair[1], metric)]
            sig_grid[i, j] = 1 if row["significant_at_fdr_05"] else 0
            diff_grid[i, j] = row["point_estimate_diff"]

    return sig_grid, diff_grid


def plot_heatmap_panel(ax: plt.Axes, sig_grid: np.ndarray, diff_grid: np.ndarray, title: str) -> None:
    """
    Draw one metric's heatmap panel with binary significance coloring and
    difference-value annotations.

    Inputs:
        ax (plt.Axes): the axes to draw on.
        sig_grid (np.ndarray): binary significance grid.
        diff_grid (np.ndarray): point-estimate difference grid, for
            annotation text.
        title (str): panel title.
    Returns:
        None. Draws onto ax in place.
    Validity rule checked: none (presentation only).
    """
    cmap = ListedColormap(["#E8E8E8", "#4C72B0"])  # gray = not significant, blue = significant
    ax.imshow(sig_grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(sig_grid.shape[0]):
        for j in range(sig_grid.shape[1]):
            text_color = "white" if sig_grid[i, j] == 1 else "black"
            ax.text(j, i, fmt_half_up(diff_grid[i, j], 3, signed=True), ha="center", va="center",
                     color=text_color, fontsize=8)

    ax.set_xticks(range(len(RUNGS_TESTED)))
    ax.set_xticklabels([str(n) for n in RUNGS_TESTED])
    ax.set_yticks(range(len(METHOD_PAIR_ORDER)))
    ax.set_yticklabels([
        f"{SHORT_LABELS[a]} vs {SHORT_LABELS[b]}" for a, b in METHOD_PAIR_ORDER
    ])
    ax.set_xlabel("Sample Size (N)")
    ax.set_title(title)


def make_figure() -> plt.Figure:
    """
    Build the two-panel significance heatmap figure.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    data = load_data()
    lookup = {(r["rung_N"], r["method_a"], r["method_b"], r["metric"]): r for r in data}

    sig_spearman, diff_spearman = build_grid(lookup, "spearman")
    sig_jaccard, diff_jaccard = build_grid(lookup, "jaccard_top5")

    fig, (ax_spearman, ax_jaccard) = plt.subplots(1, 2, figsize=(14, 6))

    plot_heatmap_panel(ax_spearman, sig_spearman, diff_spearman, "Spearman")
    plot_heatmap_panel(ax_jaccard, sig_jaccard, diff_jaccard, "Jaccard@5")

    legend_elements = [
        Patch(facecolor="#4C72B0", label="Significant (FDR q \u2264 0.05)"),
        Patch(facecolor="#E8E8E8", label="Not significant"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_3")
    print("Saved " + ", ".join(paths))
    