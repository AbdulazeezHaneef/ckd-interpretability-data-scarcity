"""
figure_08_predictor_heatmap.py

Figure 8: predictor-level top-5 appearance rate heatmap - answers
Objective 4 ("which predictor categories are most vulnerable to
instability") with the full 6-rung evidence base computed in Phase 10's
prerequisite predictor_level_analysis.py, extending the original
manuscript's 2-rung (N=500, N=250) answer.

Continuous color scale (0-1), unlike Figure 6's binary significance
color - there's no threshold here, just a rate. Each cell is annotated
with its exact value (same "no guessing from color alone" principle used
in Figure 3). Features ordered by their N=5000 baseline rate, descending
- the most consistently important feature at large N appears at the top,
giving the heatmap a readable "ranking" structure rather than an
arbitrary row order.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from figure_style import fmt_half_up, save_figure

OUTPUT_DIR = "outputs"
RUNGS_WITH_DATA = [100, 250, 500, 1000, 2500, 5000]  # ascending; N=50 excluded (0 valid draws)

FEATURE_DISPLAY_NAMES = {
    "RIDAGEYR": "Age",
    "BMXBMI": "BMI",
    "LBXSGL": "Glucose",
    "RIAGENDR": "Sex",
    "BPQ020": "Hypertension",
    "diabetes_borderline": "Diabetes (Borderline)",
    "diabetes_yes": "Diabetes (diagnosed)",
}


def load_data() -> list[dict]:
    """
    Load the predictor-level aggregated table (Phase 10 prerequisite).

    Inputs: none.
    Returns:
        list[dict]: 42 rows (7 features x 6 rungs; N=50 excluded upstream).
    Validity rule checked: none (I/O only).
    """
    with open(f"{OUTPUT_DIR}/predictor_level_aggregated.json", "r") as f:
        return json.load(f)


def make_figure() -> plt.Figure:
    """
    Build the predictor-level heatmap, features ordered by N=5000 rate
    descending.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    data = load_data()
    lookup = {(r["rung_N"], r["feature"]): r for r in data}

    features = list(FEATURE_DISPLAY_NAMES.keys())
    baseline_rates = {
        f: lookup[(5000, f)]["mean_top5_appearance_rate_across_methods"] for f in features
    }
    features_sorted = sorted(features, key=lambda f: baseline_rates[f], reverse=True)

    grid = np.array([
        [lookup[(n, f)]["mean_top5_appearance_rate_across_methods"] for n in RUNGS_WITH_DATA]
        for f in features_sorted
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, fmt_half_up(val, 3), ha="center", va="center",
                     color=text_color, fontsize=9)

    ax.set_xticks(range(len(RUNGS_WITH_DATA)))
    ax.set_xticklabels([str(n) for n in RUNGS_WITH_DATA])
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels([FEATURE_DISPLAY_NAMES[f] for f in features_sorted])
    ax.set_xlabel("Sample Size (N)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Top-5 Appearance Rate")

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_4")
    print("Saved " + ", ".join(paths))

