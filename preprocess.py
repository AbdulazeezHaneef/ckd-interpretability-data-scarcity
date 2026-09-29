"""
preprocess.py

Loads the source NHANES file, extracts the 7 locked predictor columns plus
has_CKD, applies the required encodings, drops the LBXSGL sentinel, and
writes the processed dataset to outputs/. Never modifies the source file
(Hard Rule 13).
"""

import pandas as pd

# ---- Named constants (no inline magic numbers) ----
SOURCE_FILE = "Cleaned_nhanes_with_recomputed_CKD.csv"
OUTPUT_DIR = "outputs"
PROCESSED_FILE = f"{OUTPUT_DIR}/processed_predictors.csv"

LBXSGL_SENTINEL = 777

CONTINUOUS_PREDICTORS = ["RIDAGEYR", "BMXBMI", "LBXSGL"]
BINARY_PREDICTORS = ["RIAGENDR", "BPQ020",
                     "diabetes_borderline", "diabetes_yes"]
OUTCOME_COL = "has_CKD"

SOURCE_DIABETES_COL = "DIQ010"  # 0=no, 1=yes, 2=borderline (reference = 0)


def load_source(path: str) -> pd.DataFrame:
    """
    Load the source NHANES CSV file.

    Inputs:
        path (str): relative path to the source CSV file.
    Returns:
        pd.DataFrame: raw loaded dataframe, unmodified.
    Validity rule checked: none (I/O only).
    """
    return pd.read_csv(path, low_memory=False)


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode RIAGENDR text labels into a binary numeric column (Male=1, Female=0).

    Inputs:
        df (pd.DataFrame): dataframe containing a text-valued RIAGENDR column.
    Returns:
        pd.DataFrame: copy of df with RIAGENDR overwritten as int (0/1).
    Validity rule checked: none.
    """
    df = df.copy()
    mapping = {"Male": 1, "Female": 0}
    df["RIAGENDR"] = df["RIAGENDR"].map(mapping)
    return df


def one_hot_diabetes_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the source diabetes-status column (DIQ010, coded {0,1,2})
    into diabetes_borderline and diabetes_yes, with category 0 (no) as the
    implicit reference. Never derived from LBXSGL (Hard Rule 1 leakage ban).

    Inputs:
        df (pd.DataFrame): dataframe containing the DIQ010 column.
    Returns:
        pd.DataFrame: copy of df with diabetes_borderline and diabetes_yes
            added as int 0/1 columns.
    Validity rule checked: none (leakage prevention is structural, not a
        per-draw validity rule).
    """
    df = df.copy()
    df["diabetes_yes"] = (df[SOURCE_DIABETES_COL] == 1).astype(int)
    df["diabetes_borderline"] = (df[SOURCE_DIABETES_COL] == 2).astype(int)
    return df


def drop_glucose_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where LBXSGL equals the sentinel value 777.

    Inputs:
        df (pd.DataFrame): dataframe containing an LBXSGL column.
    Returns:
        pd.DataFrame: filtered copy with sentinel rows removed.
    Validity rule checked: none (data-cleaning step, not a draw-level rule).
    """
    return df[df["LBXSGL"] != LBXSGL_SENTINEL].copy()


def build_predictor_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the final 7-predictor + has_CKD dataset and drop any row with
    a missing value in a required column.

    Inputs:
        df (pd.DataFrame): dataframe already encoded (gender, diabetes) and
            sentinel-filtered.
    Returns:
        pd.DataFrame: final processed dataset, columns = 7 predictors +
            has_CKD, no missing values.
    Validity rule checked: none.
    """
    keep_cols = CONTINUOUS_PREDICTORS + BINARY_PREDICTORS + [OUTCOME_COL]
    out = df[keep_cols].copy()
    out = out.dropna(subset=keep_cols)
    return out.reset_index(drop=True)


def main() -> None:
    """
    Run the full preprocessing pipeline and save the processed dataset.

    Inputs: none.
    Returns: None. Writes PROCESSED_FILE as a side effect.
    Validity rule checked: none.
    """
    raw = load_source(SOURCE_FILE)
    raw = encode_gender(raw)
    raw = one_hot_diabetes_status(raw)

    print("Rows before glucose-sentinel filter:", len(raw))
    raw = drop_glucose_sentinel(raw)
    print("Rows after glucose-sentinel filter (LBXSGL == 777 dropped):", len(raw))

    processed = build_predictor_dataset(raw)
    print("Rows after final predictor-completeness filter:", len(processed))

    processed.to_csv(PROCESSED_FILE, index=False)
    

    print("Final predictor list:", CONTINUOUS_PREDICTORS + BINARY_PREDICTORS)
    print("Outcome column:", OUTCOME_COL)
    print("Final row count:", len(processed))
    print("CKD-positive count:", int(processed[OUTCOME_COL].sum()))
    print(f"Saved to {PROCESSED_FILE}")


if __name__ == "__main__":
    main()
