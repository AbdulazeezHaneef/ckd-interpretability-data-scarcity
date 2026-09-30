"""
check_bmxwaist_collinearity.py

O6 (partial): re-confirms the BMXBMI-vs-BMXWAIST collinearity finding
that justified dropping BMXWAIST from the final 7-predictor set, using
the CURRENT locked 31,958-row analytic pool - not the 30,870-row
preprocessed dataset the original r=0.9055 figure was computed on
(an earlier pipeline version, before the final row count was locked).

Reproduces the same filtering pipeline as build_analytic_pool_with_labs.py
(encode_gender, one_hot_diabetes_status, drop_glucose_sentinel, dropna on
the 7 final predictors + outcome), then pulls BMXWAIST alongside BMXBMI
from the raw source file to recompute the correlation on the exact same
row set used everywhere else in this project.

This does NOT change the modeling decision (BMXWAIST stays dropped) -
it only updates the reported r-value's provenance to match the final,
locked analytic pool size, for accuracy in the manuscript.
"""

import pandas as pd

SOURCE_FILE = "Cleaned_nhanes_with_recomputed_CKD.csv"
LBXSGL_SENTINEL = 777
CONTINUOUS_PREDICTORS = ["RIDAGEYR", "BMXBMI", "LBXSGL"]
BINARY_PREDICTORS = ["RIAGENDR", "BPQ020", "diabetes_borderline", "diabetes_yes"]
OUTCOME_COL = "has_CKD"
SOURCE_DIABETES_COL = "DIQ010"


def reproduce_final_pool(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the exact filtering pipeline that produced the locked
    31,958-row analytic pool (same logic as preprocess.py /
    build_analytic_pool_with_labs.py), keeping BMXWAIST alongside if it
    exists in the source file.

    Inputs:
        raw (pd.DataFrame): loaded Cleaned_nhanes_with_recomputed_CKD.csv.
    Returns:
        pd.DataFrame: filtered to the locked 31,958-row set, with
            BMXWAIST retained if present in the source.
    Validity rule checked: none (reproduces existing, already-verified
        filtering logic - not a new rule).
    """
    df = raw.copy()
    df["RIAGENDR"] = df["RIAGENDR"].map({"Male": 1, "Female": 0})
    df["diabetes_yes"] = (df[SOURCE_DIABETES_COL] == 1).astype(int)
    df["diabetes_borderline"] = (df[SOURCE_DIABETES_COL] == 2).astype(int)
    df = df[df["LBXSGL"] != LBXSGL_SENTINEL].copy()

    keep_cols = CONTINUOUS_PREDICTORS + BINARY_PREDICTORS + [OUTCOME_COL]
    df = df.dropna(subset=keep_cols)
    return df.reset_index(drop=True)


def run_check() -> None:
    """
    Load the source file, reproduce the locked final pool, and recompute
    the BMXBMI-vs-BMXWAIST correlation on it if BMXWAIST is present.

    Inputs: none.
    Returns: None. Prints the result.
    Validity rule checked: none (descriptive statistic only).
    """
    raw = pd.read_csv(SOURCE_FILE, low_memory=False)

    if "BMXWAIST" not in raw.columns:
        print(f"BMXWAIST not found in {SOURCE_FILE} - cannot recompute. "
              f"Columns available: {list(raw.columns)[:20]}...")
        return

    pool = reproduce_final_pool(raw)
    n = len(pool)
    print(f"Reproduced locked analytic pool: {n:,} rows "
          f"(expected 31,958 - {'MATCH' if n == 31958 else 'MISMATCH, investigate'})")

    valid = pool[["BMXBMI", "BMXWAIST"]].dropna()
    r = valid["BMXBMI"].corr(valid["BMXWAIST"])
    print(f"\nBMXBMI vs BMXWAIST correlation on the {n:,}-row locked pool: r = {r:.4f}")
    print(f"(n used for correlation, after dropping any missing BMXWAIST: {len(valid):,})")
    print(f"Original reported value: r = 0.9055 (on an earlier 30,870-row preprocessed dataset)")


if __name__ == "__main__":
    run_check()
