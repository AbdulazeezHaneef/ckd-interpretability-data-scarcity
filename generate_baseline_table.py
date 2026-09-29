"""
generate_baseline_table.py

STROBE-informed baseline characteristics table (not full STROBE compliance
- this is a resampling stability study across sample sizes, not a fixed-
population exposure-outcome study, so "informed by relevant STROBE items"
is the honest framing, not "follows STROBE"). Computed on the full
31,958-row analytic pool, once - not per-draw, per-rung.

Most fields are computable directly from processed_predictors.csv (raw,
unstandardized units - standardization happens per-draw at simulation
time, not in preprocessing). eGFR and ACR are the exception: they were
computed earlier in the pipeline (Cleaned_nhanes_with_recomputed_CKD.csv,
preprocess.py's own SOURCE_FILE) but dropped before processed_predictors.csv
was built, since they are not model predictors. This script pulls them
back in by reproducing preprocess.py's exact filtering pipeline
(encode_gender, one_hot_diabetes_status, drop_glucose_sentinel, dropna on
the 7 predictors + outcome) so the eGFR/ACR rows align with the same
31,958-row analytic set - not a separately/differently filtered set.

ACR is typically strongly right-skewed in NHANES data, so both mean+/-SD
and median are reported for eGFR and ACR, for transparency - the
manuscript can choose which to lead with.
"""

import numpy as np
import pandas as pd
import json

PROCESSED_FILE = "outputs/processed_predictors.csv"
RAW_CKD_FILE = "outputs/Cleaned_nhanes_with_recomputed_CKD.csv"  # preprocess.py's SOURCE_FILE
OUTPUT_FILE = "outputs/baseline_characteristics_table.json"

CONTINUOUS_PREDICTORS = ["RIDAGEYR", "BMXBMI", "LBXSGL"]
LBXSGL_SENTINEL = 777
SOURCE_DIABETES_COL = "DIQ010"


def compute_from_processed(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Compute every baseline-table row derivable directly from
    processed_predictors.csv.

    Inputs:
        df (pd.DataFrame): loaded processed_predictors.csv.
    Returns:
        list[tuple[str, str]]: (row label, formatted value) pairs.
    Validity rule checked: none (descriptive statistics only).
    """
    n = len(df)
    rows = []
    rows.append(("N (analytic pool)", f"{n:,}"))
    rows.append(("Age, years, mean (SD)", f"{df['RIDAGEYR'].mean():.1f} ({df['RIDAGEYR'].std():.1f})"))
    pct_female = (df["RIAGENDR"] == 0).mean() * 100  # RIAGENDR: 1=Male, 0=Female (preprocess.py encoding)
    rows.append(("Female, %", f"{pct_female:.1f}"))
    rows.append(("Hypertensive (BPQ020), %", f"{df['BPQ020'].mean()*100:.1f}"))
    rows.append(("Diabetic (diagnosed), %", f"{df['diabetes_yes'].mean()*100:.1f}"))
    rows.append(("Diabetes, borderline, %", f"{df['diabetes_borderline'].mean()*100:.1f}"))
    rows.append(("BMI, kg/m^2, mean (SD)", f"{df['BMXBMI'].mean():.1f} ({df['BMXBMI'].std():.1f})"))
    rows.append(("Fasting glucose, mg/dL, mean (SD)", f"{df['LBXSGL'].mean():.1f} ({df['LBXSGL'].std():.1f})"))
    rows.append(("CKD-compatible abnormality (outcome), %", f"{df['has_CKD'].mean()*100:.1f}"))
    return rows


def reproduce_preprocessing_filter(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce preprocess.py's filtering pipeline exactly (encode_gender,
    one_hot_diabetes_status, drop_glucose_sentinel, dropna on the 7
    predictors + outcome), so eGFR/ACR align with the same final 31,958
    row analytic set - not a separately filtered one.

    Inputs:
        raw (pd.DataFrame): loaded Cleaned_nhanes_with_recomputed_CKD.csv.
    Returns:
        pd.DataFrame: filtered to the same rows processed_predictors.csv
            contains, still carrying eGFR and ACR columns.
    Validity rule checked: none (reproduces preprocess.py's existing,
        already-verified filtering logic - not a new rule).
    """
    df = raw.copy()
    df["RIAGENDR"] = df["RIAGENDR"].map({"Male": 1, "Female": 0})
    df["diabetes_yes"] = (df[SOURCE_DIABETES_COL] == 1).astype(int)
    df["diabetes_borderline"] = (df[SOURCE_DIABETES_COL] == 2).astype(int)
    df = df[df["LBXSGL"] != LBXSGL_SENTINEL].copy()

    keep_cols = CONTINUOUS_PREDICTORS + ["RIAGENDR", "BPQ020",
                                          "diabetes_borderline", "diabetes_yes", "has_CKD"]
    df = df.dropna(subset=keep_cols)
    return df.reset_index(drop=True)


def compute_egfr_acr_rows(raw_path: str) -> list[tuple[str, str]]:
    """
    Compute eGFR and ACR mean(SD)/median rows, aligned to the same
    31,958-row analytic pool.

    Inputs:
        raw_path (str): path to Cleaned_nhanes_with_recomputed_CKD.csv.
    Returns:
        list[tuple[str, str]]: (row label, formatted value) pairs.
    Validity rule checked: none (descriptive statistics only).
    """
    raw = pd.read_csv(raw_path, low_memory=False)
    filtered = reproduce_preprocessing_filter(raw)

    rows = []
    for col, label, unit in [("eGFR", "eGFR, mL/min/1.73m^2", ""), ("ACR", "ACR, mg/g", "")]:
        vals = filtered[col].dropna()
        rows.append((f"{label}, mean (SD)", f"{vals.mean():.1f} ({vals.std():.1f})"))
        rows.append((f"{label}, median", f"{vals.median():.1f}"))
    rows.append(("n aligned for eGFR/ACR", f"{len(filtered):,} (should match processed pool N)"))
    return rows

def save_rows_to_json(rows: list[tuple[str, str]], path: str) -> None:
    """
    Save baseline-table rows to a JSON file, matching this project's
    existing outputs/*_table.json convention (list of records).

    Inputs:
        rows (list[tuple[str, str]]): (label, formatted value) pairs,
            in display order.
        path (str): output JSON path.
    Returns:
        None. Writes the file.
    Validity rule checked: none (serialization only).
    """
    records = [{"characteristic": label, "value": value} for label, value in rows]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

def run_generate_baseline_table() -> None:
    """
    Build and print the full STROBE-informed baseline characteristics
    table. Computes the processed_predictors.csv-derivable rows always;
    attempts the eGFR/ACR rows only if RAW_CKD_FILE is found, with a
    clear message if it is not (rather than silently omitting).

    Inputs: none.
    Returns: None. Prints the table; does not write a file, since this is
        a one-off manuscript table, not a pipeline artifact.
    Validity rule checked: none (orchestration only).
    """
    df = pd.read_csv(PROCESSED_FILE)
    rows = compute_from_processed(df)

    print("=== STROBE-informed baseline characteristics table ===")
    print(f"(N = {len(df):,} analytic pool)\n")
    for label, value in rows:
        print(f"{label:<42} {value}")

        print()
    try:
        egfr_acr_rows = compute_egfr_acr_rows(RAW_CKD_FILE)
        for label, value in egfr_acr_rows:
            print(f"{label:<42} {value}")
        rows = rows + egfr_acr_rows
    except FileNotFoundError:
        print(f"NOTE: '{RAW_CKD_FILE}' not found in the working directory - "
              f"eGFR/ACR rows skipped. Place that file alongside this script "
              f"(same file preprocess.py reads as SOURCE_FILE) and rerun.")

    save_rows_to_json(rows, OUTPUT_FILE)
    print(f"\nSaved {OUTPUT_FILE}")

if __name__ == "__main__":
    run_generate_baseline_table()

    
