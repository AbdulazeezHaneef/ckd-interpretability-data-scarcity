"""
generate_tables.py

Phase 10: generates the 3 manuscript-ready tables (as identified in the
Phase 10 triage) directly from existing, already-verified JSON outputs.
No new computation - this is formatting only. Writes CSV files straight
to manuscript_assets/tables/, so nothing depends on manual copy-paste or
ad hoc formatting.

Table 1: method computational failure rate (Phase 2's null-result table)
Table 2: pairwise row-overlap by rung (Phase 8)
Table 3: performance split-validity counts by rung (Phase 9)
"""

import csv
import json
import os

OUTPUT_DIR = "outputs"
TABLES_DIR = "manuscript_assets/tables"

METHODS = ["standardized_beta", "permutation", "shap", "lime"]
METHOD_LABELS = {
    "standardized_beta": "Standardized Beta",
    "permutation": "Permutation Importance",
    "shap": "SHAP",
    "lime": "LIME",
}


def generate_table_method_computational_failure() -> str:
    """
    Table 1: method computational failure rate, rows=rung, cols=method.
    Source: final_method_computational_failure_table.json (Phase 2).

    Inputs: none.
    Returns:
        str: path to the saved CSV file.
    Validity rule checked: none (formatting only, data already verified).
    """
    with open(f"{OUTPUT_DIR}/final_method_computational_failure_table.json", "r") as f:
        data = json.load(f)

    rungs = sorted({r["rung_N"] for r in data}, reverse=True)
    lookup = {(r["rung_N"], r["method"]): r for r in data}

    path = f"{TABLES_DIR}/table_method_computational_failure.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N"] + [METHOD_LABELS[m] for m in METHODS])
        for N in rungs:
            row = [N]
            for m in METHODS:
                rate = lookup[(N, m)]["computation_failure_rate"]
                row.append(f"{rate:.3f}" if rate is not None else "n/a")
            writer.writerow(row)
    return path


def generate_table_row_overlap() -> str:
    """
    Table 2: pairwise row-overlap by rung, with theoretical comparison.
    Source: overlap_quantification_all_rungs.json (Phase 8).

    Inputs: none.
    Returns:
        str: path to the saved CSV file.
    Validity rule checked: none (formatting only, data already verified).
    """
    with open(f"{OUTPUT_DIR}/overlap_quantification_all_rungs.json", "r") as f:
        data = json.load(f)

    path = f"{TABLES_DIR}/table_row_overlap_by_rung.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Mean Overlap (%)", "Range (%)", "Theoretical Expected (%)"])
        for r in sorted(data, key=lambda x: -x["rung_N"]):
            mean_pct = r["empirical_mean_overlap_fraction"] * 100
            min_pct = r["empirical_min_overlap_fraction"] * 100
            max_pct = r["empirical_max_overlap_fraction"] * 100
            theo_pct = r["theoretical_expected_overlap_fraction"] * 100
            writer.writerow([
                r["rung_N"],
                f"{mean_pct:.2f}",
                f"[{min_pct:.2f}, {max_pct:.2f}]",
                f"{theo_pct:.2f}",
            ])
    return path


def generate_table_split_validity() -> str:
    """
    Table 3: performance-evaluability split validity counts by rung.
    Source: performance_split_validity_table.json (Phase 9).

    Inputs: none.
    Returns:
        str: path to the saved CSV file.
    Validity rule checked: none (formatting only, data already verified).
    """
    with open(f"{OUTPUT_DIR}/performance_split_validity_table.json", "r") as f:
        data = json.load(f)

    path = f"{TABLES_DIR}/table_split_validity_by_rung.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "Split-Valid (n)", "Train-EPV Fail (n)",
            "Test-Positives Fail (n)", "Split-Valid Rate",
        ])
        for r in sorted(data, key=lambda x: -x["rung_N"]):
            writer.writerow([
                r["rung_N"],
                r["n_split_valid"],
                r["n_split_invalid_train_epv"],
                r["n_split_invalid_test_positives"],
                f"{r['split_valid_rate']:.3f}",
            ])
    return path


def run_generate_tables() -> list[str]:
    """
    Generate all 3 manuscript tables, print confirmation.

    Inputs: none.
    Returns:
        list[str]: paths to all 3 saved CSV files.
    Validity rule checked: none (orchestration only).
    """
    os.makedirs(TABLES_DIR, exist_ok=True)

    paths = [
        generate_table_method_computational_failure(),
        generate_table_row_overlap(),
        generate_table_split_validity(),
    ]

    for p in paths:
        print(f"Saved {p}")
        with open(p, "r") as f:
            print(f.read())

    return paths


if __name__ == "__main__":
    run_generate_tables()

