"""
figure_09_jaccard_k_comparison.py

Figure 9 (final): Jaccard@3 vs @5 vs @7 comparison, per method - the
Phase 4 floor-effect check. 2x2 grid, one panel per method, each with 3
lines (k=3, 5, 7) vs N. Point estimates only (no CI ribbons here -
Figure 4 already carries the Jaccard@5 CI specifically; three ribbons per
small panel would be visual clutter, not clarity).

jaccard_top7 is 1.0 at every rung by construction (full 7-feature set,
same set every time) - shown as a flat reference line at 1.0 on every
panel, which is itself the point: the review's floor-effect concern
would only be real if top-5 tracked closely with this flat ceiling line;
Phase 4's finding was that it instead tracks top-3's decline.
"""

import json

import matplotlib.pyplot as plt

from figure_style import METHOD_LABELS, save_figure, set_log_xaxis_with_n_labels

OUTPUT_DIR = "outputs"
METHODS = ["standardized_beta", "permutation", "shap", "lime"]
RUNGS_WITH_DATA = [5000, 2500, 1000, 500, 250, 100]  # N=50 excluded

K_STYLES = {
    3: {"color": "#4C72B0", "marker": "o", "label": "Top-3"},
    5: {"color": "#DD8452", "marker": "s", "label": "Top-5 (primary)"},
    7: {"color": "#808080", "marker": None, "label": "Top-7 (ceiling, always 1.0)"},
}


def load_data() -> list[dict]:
    """
    Load the CI-augmented stability table (Phase 6) - only the point
    estimates (jaccard_top3/5/7_mean) are used here.

    Inputs: none.
    Returns:
        list[dict]: 28 rows (7 rungs x 4 methods).
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/final_stability_table_with_ci.json", "r") as f:
        return json.load(f)


def plot_method_panel(ax: plt.Axes, lookup: dict, method: str) -> None:
    """
    Plot one method's 3 k-value lines onto one axes.

    Inputs:
        ax (plt.Axes): the axes to draw on.
        lookup (dict): (rung_N, method) -> row dict.
        method (str): one of METHODS.
    Returns:
        None. Draws onto ax in place.
    Validity rule checked: none (presentation only).
    """
    for k in [3, 5, 7]:
        ns, vals = [], []
        for n in RUNGS_WITH_DATA:
            row = lookup.get((n, method))
            key = f"jaccard_top{k}_mean"
            if row is None or row[key] is None:
                continue
            ns.append(n)
            vals.append(row[key])
        style = K_STYLES[k]
        linestyle = "--" if k == 7 else "-"
        ax.plot(ns, vals, marker=style["marker"], markersize=5, linewidth=1.5,
                 linestyle=linestyle, color=style["color"], label=style["label"])

    set_log_xaxis_with_n_labels(ax, RUNGS_WITH_DATA)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(METHOD_LABELS[method])
    ax.grid(True, which="both", axis="both", alpha=0.3)


def make_figure() -> plt.Figure:
    """
    Build the 2x2 grid of Jaccard@3/5/7 comparison panels, one per method.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    data = load_data()
    lookup = {(r["rung_N"], r["method"]): r for r in data}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for ax, method in zip(axes_flat, METHODS):
        plot_method_panel(ax, lookup, method)

    for ax in axes[1, :]:
        ax.set_xlabel("Sample Size (N)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Jaccard Index")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))


    fig.tight_layout(rect=[0, 0.04, 1, 1])

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_S4")
    print("Saved " + ", ".join(paths))

