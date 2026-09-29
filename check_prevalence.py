"""
check_prevalence.py

Quick, standalone check: prevalence (proportion = 1) of each binary
predictor in the analytic pool, specifically comparing diabetes_borderline
against diabetes_yes - tests the "rare-predictor dilution" hypothesis
(a rare binary predictor with a strong per-person effect can still
produce a large, stable standardized-beta while contributing almost
nothing to permutation/SHAP's aggregate importance, since those measure
average impact on model output across the whole sample).

If diabetes_borderline is meaningfully rarer than diabetes_yes, this
hypothesis survives to be written up. If prevalence is similar, it
collapses and the pattern goes in as an open, unexplained finding.

No resimulation needed - reads directly from processed_predictors.csv.
"""

import pandas as pd

PROCESSED_FILE = "outputs/processed_predictors.csv"
BINARY_PREDICTORS = ["RIAGENDR", "BPQ020", "diabetes_borderline", "diabetes_yes"]


def run_prevalence_check() -> None:
    """
    Compute and print the prevalence (proportion coded 1) of each binary
    predictor, plus the raw N=1 count and pool size, so the comparison is
    fully auditable.

    Inputs: none.
    Returns: None. Prints results only - diagnostic check, not part of
        the main pipeline, nothing saved to outputs/.
    Validity rule checked: none (descriptive statistics only).
    """
    df = pd.read_csv(PROCESSED_FILE)
    n_total = len(df)

    print(f"=== Binary predictor prevalence (pool size = {n_total}) ===")
    for col in BINARY_PREDICTORS:
        n_positive = int(df[col].sum())
        prevalence = n_positive / n_total
        print(f"{col:>20}: {n_positive:>6} / {n_total} = {prevalence*100:.2f}%")

    print()
    db = df["diabetes_borderline"].mean()
    dy = df["diabetes_yes"].mean()
    print("=== diabetes_borderline vs diabetes_yes ===")
    print(f"diabetes_borderline prevalence: {db*100:.2f}%")
    print(f"diabetes_yes prevalence:        {dy*100:.2f}%")
    print(f"ratio (borderline / yes):       {db/dy:.3f}")
    if db < dy * 0.6:
        print("-> diabetes_borderline is meaningfully rarer: dilution hypothesis SURVIVES")
    else:
        print("-> prevalence is similar: dilution hypothesis COLLAPSES")


if __name__ == "__main__":
    run_prevalence_check()

