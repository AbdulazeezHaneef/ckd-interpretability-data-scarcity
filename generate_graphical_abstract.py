"""
generate_graphical_abstract.py

JBI-compliant graphical abstract: two side-by-side panels contrasting
predictive performance (stable) against interpretability stability
(collapsing) from N=5,000 to N=250 - the paper's central decoupling
finding (Discussion 5.5).

CRITICAL DESIGN DECISION: both panels share the SAME y-axis scale
(0.0-1.0), even though axis numbers are hidden. Letting each panel
auto-scale independently to its own tight data range would make the
visual steepness comparison dishonest - AUC's small drop and Spearman's
large drop could look equally steep if each panel picked its own
zoomed-in range. A shared scale is the only way the visual slope
difference reflects the actual magnitude difference, rather than an
artifact of independent axis scaling (the same class of distortion
rejected for Figure 7's dual-axis version earlier in this project).

Data (only these four numbers, no fabricated intermediate points):
  AUC:      N=5,000 -> 0.776,  N=250 -> 0.748  (Phase 9, unaffected by
            the permutation-scoring fix below)
  Spearman: N=5,000 -> 0.907,  N=250 -> 0.429  (mean across all 4 methods;
            corrected after the permutation_importance scoring fix - it
            previously defaulted to accuracy, not AUC, which understated
            permutation's stability and pulled the 4-method mean down to
            0.883/0.360; both values shifted upward after the fix)

Exports PNG (raster, 150+ DPI), TIFF (raster, requires Pillow), and PDF
(vector, scales cleanly at any print size) - covers all four JBI-accepted
formats except native MS Office.

Output pixel dimensions at DPI=200: 2656 x 1062 px (exactly 2x JBI's
1328x531 minimum, same 2.5:1 ratio, comfortably exceeds the 150 DPI
floor when printed at 13x5cm).
"""

import io
import os
import textwrap

import matplotlib.pyplot as plt

OUTPUT_DIR = "manuscript_assets/figures"

# --- Locked spec values ---
FIG_WIDTH_IN = 13.28   # inches; at DPI=200 -> 2656 px wide
FIG_HEIGHT_IN = 5.31   # inches; at DPI=200 -> 1062 px tall
EXPORT_DPI = 200        # exceeds JBI's 150 DPI minimum with headroom

TITLE_TEXT = ("Under Data Scarcity: Predictive Performance and\n"
              "Interpretability Stability Decouple for CKD Risk Prediction")
CAPTION_TEXT = (
    "As sample size shrinks (N = 5,000 -> N = 250), model performance holds "
    "while interpretability method rankings become substantially less "
    "reproducible - adequate performance is not evidence of adequate "
    "interpretability."
)

X_LABELS = ["N = 5,000", "N = 250"]
SHARED_YLIM = (0.0, 1.0)  # locked shared scale - see module docstring

AUC_VALUES = [0.776, 0.748]
AUC_COLOR = "#1B3A5C"   # calm navy - "nothing alarming happening here"
AUC_HEADING = "Predictive Performance (AUC)\nstays comparatively stable"

SPEARMAN_VALUES = [0.907, 0.429]
SPEARMAN_COLOR = "#D62728"  # warm alert red - "this is the problem"
SPEARMAN_HEADING = "Interpretability Stability (Spearman)\ndeclines sharply"


def style_panel(ax: plt.Axes, values: list[float], color: str, heading: str) -> None:
    """
    Draw one panel: a 2-point line with big bold value labels directly on
    each point, no axis clutter (no y-ticks, no gridlines, minimal spines).

    Inputs:
        ax (plt.Axes): the axes to draw on.
        values (list[float]): the two y-values (N=5000, N=250).
        color (str): line/marker color for this panel.
        heading (str): panel heading text, shown above the plot.
    Returns:
        None. Draws onto ax in place.
    Validity rule checked: none (presentation only). Enforces the shared
        y-axis scale (SHARED_YLIM) - never overridden per-panel.
    """
    x = [0, 1]
    ax.plot(x, values, marker="o", markersize=16, linewidth=5, color=color,
             solid_capstyle="round", zorder=3)

    # Value labels placed clearly above each marker so they do not sit on
    # the line or visually merge with the marker - EXCEPT when the point
    # sits close enough to the shared axis ceiling (>0.85) that an
    # above-placed label would run out of room and clip/overlap the
    # marker. In that case the label is placed below the point instead.
    for xi, yi in zip(x, values):
        near_ceiling = yi > 0.85
        ax.annotate(
            f"{yi:.3f}",
            xy=(xi, yi),
            xytext=(0, -20 if near_ceiling else 18),
            textcoords="offset points",
            ha="center",
            va="top" if near_ceiling else "bottom",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(*SHARED_YLIM)  # LOCKED shared scale - do not change per panel
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=15, fontweight="bold")

    # Strip all axis clutter per spec: no y-ticks, no gridlines, minimal spines.
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.tick_params(axis="x", length=0)

    ax.set_title(heading, fontsize=15, fontweight="bold", color=color, pad=14)


def make_figure() -> plt.Figure:
    """
    Build the complete two-panel graphical abstract: title, two panels,
    caption.

    Inputs: none.
    Returns:
        plt.Figure: the completed figure, ready to export.
    Validity rule checked: none (presentation only).
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, (ax_auc, ax_spearman) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        facecolor="white",
    )
    fig.patch.set_facecolor("white")

    style_panel(ax_auc, AUC_VALUES, AUC_COLOR, AUC_HEADING)
    style_panel(ax_spearman, SPEARMAN_VALUES, SPEARMAN_COLOR, SPEARMAN_HEADING)

    fig.suptitle(TITLE_TEXT, fontsize=18, fontweight="bold", y=0.985, linespacing=1.4)

    wrapped_caption = "\n".join(textwrap.wrap(CAPTION_TEXT, width=140))
    fig.text(0.5, 0.02, wrapped_caption, ha="center", va="bottom",
              fontsize=11, color="#555555", style="italic")

    fig.subplots_adjust(top=0.60, bottom=0.20, left=0.06, right=0.96, wspace=0.25)

    return fig


def save_raster_rgb(fig: plt.Figure, path: str, dpi: int) -> None:
    """
    Render matplotlib figure directly to a flat RGB raster image (PNG/TIFF)
    via an in-memory buffer, preventing Windows disk file-locking conflicts.
    """
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)

    with Image.open(buf) as im:
        rgb = im.convert("RGB")
        rgb.save(path, dpi=(dpi, dpi))


def export_all_formats(fig: plt.Figure) -> None:
    """
    Export the figure to PNG, TIFF, and PDF formats complying with JBI guidelines.

    Inputs:
        fig (plt.Figure): the completed figure.
    Returns:
        None. Writes 3 files to OUTPUT_DIR.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    png_path = f"{OUTPUT_DIR}/graphical_abstract.png"
    tiff_path = f"{OUTPUT_DIR}/graphical_abstract.tiff"
    pdf_path = f"{OUTPUT_DIR}/graphical_abstract.pdf"

    # Save PNG directly as RGB
    try:
        save_raster_rgb(fig, png_path, EXPORT_DPI)
        print(f"Saved {png_path}")
    except Exception as e:
        print(f"PNG export failed ({e})")

    # Save TIFF directly as RGB
    try:
        save_raster_rgb(fig, tiff_path, EXPORT_DPI)
        print(f"Saved {tiff_path}")
    except Exception as e:
        print(f"TIFF export failed ({e}) - install Pillow: pip install pillow")

    # Save vector PDF
    fig.savefig(pdf_path, facecolor="white")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    fig = make_figure()
    export_all_formats(fig)

