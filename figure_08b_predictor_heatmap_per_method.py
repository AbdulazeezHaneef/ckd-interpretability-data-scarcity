"""
figure_08b_predictor_heatmap_per_method.py

Figure 4 (revised, O7): per-method predictor top-5 appearance rate
heatmap - four panels (one per method), replacing the original single,
method-averaged heatmap (figure_08_predictor_heatmap.py), which
Section 4.4/B7 of the review flagged as conflating each method's
distinct pattern - most visibly for diabetes-borderline, where
standardized-beta and the other three methods move in opposite
directions as N falls (Section 4.4, Supplementary S1.6).

Same continuous 0-1 color scale and exact-value-per-cell annotation
convention as the original Figure 4. Feature row order is fixed across
all four panels (by the aggregated N=5000 baseline rate, descending -
identical to the original single-panel figure's ordering) so a reader
can compare the same row across panels directly.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from figure_style import fmt_half_up, save_figure

OUTPUT_DIR = "outputs"
RUNGS_WITH_DATA = [100, 250, 500, 1000, 2500, 5000]  # ascending; N=50 excluded (0 valid draws)
METHODS = ["standardized_beta", "permutation", "shap", "lime"]
METHOD_TITLES = {
    "standardized_beta": "Standardized Beta",
    "permutation": "Permutation Importance",
    "shap": "SHAP",
    "lime": "LIME",
}

FEATURE_DISPLAY_NAMES = {
    "RIDAGEYR": "Age",
    "BMXBMI": "BMI",
    "LBXSGL": "Glucose",
    "RIAGENDR": "Sex",
    "BPQ020": "Hypertension",
    "diabetes_borderline": "Diabetes (Borderline)",
    "diabetes_yes": "Diabetes (diagnosed)",
}


def load_per_method_data() -> list[dict]:
    """
    Load the predictor-level per-method table (predictor_level_analysis.py).

    Inputs: none.
    Returns:
        list[dict]: 168 rows (7 features x 6 rungs x 4 methods).
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/predictor_level_per_method.json", "r") as f:
        return json.load(f)


def load_feature_order() -> list[str]:
    """
    Derive the same feature row order used by the original single-panel
    Figure 4 (N=5000 baseline rate, descending), so this revised
    figure's rows line up with any manuscript text referencing that
    ordering.

    Inputs: none.
    Returns:
        list[str]: feature keys, ordered.
    Validity rule checked: none (I/O + sort only).
    """
    with open(f"{OUTPUT_DIR}/predictor_level_aggregated.json", "r") as f:
        aggregated = json.load(f)
    baseline = {r["feature"]: r["mean_top5_appearance_rate_across_methods"]
                for r in aggregated if r["rung_N"] == 5000}
    return sorted(baseline.keys(), key=lambda f: baseline[f], reverse=True)


def make_figure() -> plt.Figure:
    """
    Build the four-panel per-method predictor heatmap figure.
    """
    data = load_per_method_data()
    lookup = {(r["rung_N"], r["method"], r["feature"]): r for r in data}
    features_sorted = load_feature_order()

    # Share y-axis within each row and let matplotlib manage spacing
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 11),
        sharey="row",
        constrained_layout=True
    )

    axes_flat = axes.flatten()
    im = None

    for idx, (ax, method) in enumerate(zip(axes_flat, METHODS)):

        grid = np.array([
            [
                lookup[(n, method, f)]["top5_appearance_rate"]
                for n in RUNGS_WITH_DATA
            ]
            for f in features_sorted
        ])

        im = ax.imshow(
            grid,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            aspect="auto"
        )

        # Add values inside cells
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                text_color = "white" if val > 0.6 else "black"

                ax.text(
                    j, i,
                    fmt_half_up(val, 2),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8
                )

        # X axis
        ax.set_xticks(range(len(RUNGS_WITH_DATA)))
        ax.set_xticklabels([str(n) for n in RUNGS_WITH_DATA])

        # Y axis
        ax.set_yticks(range(len(features_sorted)))

        # ONLY show feature labels on the left-hand panels
        if idx in [0, 2]:
            ax.set_yticklabels(
                [FEATURE_DISPLAY_NAMES[f] for f in features_sorted]
            )
        else:
            ax.tick_params(
                axis="y",
                labelleft=False
            )

        ax.set_xlabel("Sample Size (N)")
        ax.set_title(METHOD_TITLES[method])

    # Colorbar
    cbar = fig.colorbar(
        im,
        ax=axes_flat,
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label("Top-5 Appearance Rate")

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_4")
    print("Saved " + ", ".join(paths))

