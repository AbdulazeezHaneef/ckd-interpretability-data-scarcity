"""
check_collinearity.py

Quick, standalone check: pairwise correlation and VIF (variance inflation
factor) across all 7 predictors, specifically testing whether
diabetes_borderline and diabetes_yes show meaningful collinearity - a
candidate explanation for diabetes_borderline's weak-predictor pattern
and/or standardized-beta's instability, before it gets written into the
manuscript as a mechanism.

No resimulation needed - reads directly from processed_predictors.csv.
Requires: pandas, numpy, statsmodels (pip install statsmodels).
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROCESSED_FILE = "outputs/processed_predictors.csv"
PREDICTORS = ["RIDAGEYR", "BMXBMI", "LBXSGL", "RIAGENDR", "BPQ020",
              "diabetes_borderline", "diabetes_yes"]


def run_collinearity_check() -> None:
    """
    Compute pairwise correlation (full matrix, with the
    diabetes_borderline/diabetes_yes pair highlighted) and VIF for all 7
    predictors, print results.

    Inputs: none.
    Returns: None. Prints results only - this is a diagnostic check, not
        part of the main pipeline, so nothing is saved to outputs/.
    Validity rule checked: none (descriptive statistics only).
    """
    df = pd.read_csv(PROCESSED_FILE)
    X = df[PREDICTORS]

    corr = X.corr()
    print("=== Correlation: diabetes_borderline vs diabetes_yes ===")
    print(round(corr.loc["diabetes_borderline", "diabetes_yes"], 4))
    print()

    print("=== Full correlation matrix ===")
    print(corr.round(3))
    print()

    print("=== VIF for all 7 predictors (values above ~5 would indicate concern) ===")
    X_const = X.copy()
    X_const.insert(0, "const", 1.0)
    for i, col in enumerate(PREDICTORS, start=1):
        vif = variance_inflation_factor(X_const.values, i)
        print(f"{col:>20}: VIF={vif:.3f}")


if __name__ == "__main__":
    run_collinearity_check()
