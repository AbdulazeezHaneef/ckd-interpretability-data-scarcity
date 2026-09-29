"""
figure_05_sensitivity_agreement_contrast.py

Figure 5: contrasts LIME's per-draw agreement between its original
(subsample=50) and sensitivity (two-tier, Phase 5 item 10) rankings
against permutation importance's per-draw agreement between its original
(in-sample) and held-out (Phase 5 item 11) rankings. The sharp contrast
(LIME 0.99+ everywhere vs permutation 0.25-0.67) is the finding - single
panel, two lines, log x-axis with real N labels (consistent with Figures
1/2/4). N=50 excluded (0 draws in either sensitivity check).

Aggregates the per-draw agreement JSON files (mean Spearman agreement per
rung) rather than reading pre-aggregated summaries, since the two source
files use slightly different per-draw record shapes (permutation's
includes SPLIT_INVALID entries with null agreement fields that must be
filtered before averaging).
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from figure_style import save_figure, set_log_xaxis_with_n_labels

OUTPUT_DIR = "outputs"
RUNGS_WITH_DATA = [5000, 2500, 1000, 500, 250, 100]  # N=50 excluded (0 draws either check)


def load_mean_agreement_by_rung(path: str, spearman_key: str = "spearman") -> dict[int, float]:
    """
    Load a per-draw agreement JSON file and compute mean Spearman
    agreement per rung, filtering out any records with a null agreement
    value (e.g. permutation's SPLIT_INVALID entries).

    Inputs:
        path (str): path to the per-draw agreement JSON file.
        spearman_key (str): the field name holding the Spearman agreement
            value (same in both files: "spearman").
    Returns:
        dict[int, float]: rung_N -> mean Spearman agreement, only for
            rungs with at least one usable record.
    Validity rule checked: none (aggregation only; the underlying
        per-draw values were already computed and verified in Phase 5).
    """
    with open(path, "r") as f:
        records = json.load(f)

    by_rung: dict[int, list[float]] = {}
    for r in records:
        val = r.get(spearman_key)
        if val is None:
            continue
        by_rung.setdefault(r["rung_N"], []).append(val)

    return {n: float(np.mean(v)) for n, v in by_rung.items()}


def make_figure() -> plt.Figure:
    """
    Build the LIME-vs-permutation sensitivity agreement contrast figure.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to save.
    Validity rule checked: none (presentation only).
    """
    lime_agreement = load_mean_agreement_by_rung(
        f"{OUTPUT_DIR}/lime_sensitivity_per_draw_agreement.json")
    perm_agreement = load_mean_agreement_by_rung(
        f"{OUTPUT_DIR}/permutation_sensitivity_per_draw_agreement.json")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    lime_ns = [n for n in RUNGS_WITH_DATA if n in lime_agreement]
    ax.plot(lime_ns, [lime_agreement[n] for n in lime_ns],
            marker="o", markersize=6, linewidth=1.8, color="#55A868",
            label="LIME (original subsample=50 vs. two-tier sensitivity)")

    perm_ns = [n for n in RUNGS_WITH_DATA if n in perm_agreement]
    ax.plot(perm_ns, [perm_agreement[n] for n in perm_ns],
            marker="s", markersize=6, linewidth=1.8, color="#DD8452",
            label="Permutation Importance (in-sample vs. held-out)")

    set_log_xaxis_with_n_labels(ax, RUNGS_WITH_DATA)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Mean Spearman Agreement with Original Ranking")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = make_figure()
    paths = save_figure(fig, "Figure_S3")
    print("Saved " + ", ".join(paths))
