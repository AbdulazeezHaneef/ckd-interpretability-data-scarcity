"""
figure_style.py

Shared helpers for all manuscript figures: consistent conventions for
log-axis tick labeling, color palette, rung ordering, half-up number
formatting, and save-to-PDF/PNG/SVG under submission file names.
"""

import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Embed TrueType fonts in vector output (journal-safe PDF/EPS).
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

FIGURES_DIR = "manuscript_assets/figures"

# Consistent method color palette across every figure that shows all 4 methods
METHOD_COLORS = {
    "standardized_beta": "#4C72B0",
    "permutation": "#DD8452",
    "shap": "#55A868",
    "lime": "#C44E52",
}
METHOD_LABELS = {
    "standardized_beta": "Standardized Beta",
    "permutation": "Permutation Importance",
    "shap": "SHAP",
    "lime": "LIME",
}

# Ascending (smallest N first) - the single axis direction used by every figure.
RUNGS = [50, 100, 250, 500, 1000, 2500, 5000]


def fmt_half_up(value: float, ndigits: int, signed: bool = False) -> str:
    """
    Format a number rounded half-up (0.5625 -> 0.563), not Python's
    round-half-even / binary-float behavior (0.5625 -> 0.562).

    Inputs:
        value (float): number to format.
        ndigits (int): decimal places.
        signed (bool): if True, always show a leading + or -.
    Returns:
        str: the formatted number.
    """
    quantum = Decimal(1).scaleb(-ndigits)
    rounded = Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:+f}" if signed else f"{rounded:f}"


def set_log_xaxis_with_n_labels(ax: plt.Axes, rungs: list[int] = RUNGS) -> None:
    """
    Set a log-scale x-axis but label ticks with the actual N values
    (50, 100, 250, ...) rather than log exponents or scientific notation.
    """
    ax.set_xscale("log")
    ax.set_xticks(rungs)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())
    ax.set_xticklabels([str(n) for n in rungs])
    ax.set_xlim(min(rungs) * 0.85, max(rungs) * 1.15)


def save_figure(fig: plt.Figure, name: str) -> list[str]:
    """
    Save a figure as PDF (vector, for submission), PNG (300 dpi) and SVG
    in manuscript_assets/figures/, using the submission file name.

    Inputs:
        fig (plt.Figure): the figure to save.
        name (str): base filename, no extension (e.g. "Figure_1a").
    Returns:
        list[str]: paths written, in order [pdf, png, svg].
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 300}), ("svg", {})):
        path = f"{FIGURES_DIR}/{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    return paths
    