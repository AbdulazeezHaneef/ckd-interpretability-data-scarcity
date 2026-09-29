"""
epv_sensitivity.py

Phase 3 (Review Point 3/4 fix): recomputes final_model_validity_failure_table
and final_stability_table at alternative EPV floors (5, 10 predictor-events-
per-variable, i.e. 35 and 70 minimum positive cases for 7 predictors) by
re-filtering the SAME rung_results_N{N}.json files already produced at
EPV=2 (14 positives) - no model refitting, no SHAP/LIME re-execution.

This is valid because EPV is checked BEFORE the model is fit (Rule 1 runs
first in draw_sample_and_fit). A draw's model fit, convergence outcome,
coefficient check, and method-computation results never depended on which
EPV threshold was used - only on the drawn sample itself, which is
identical across thresholds for a given seed. Raising the threshold can
only ever convert a previously-valid or previously-rule_2/rule_3-invalid
draw INTO a rule_1_epv failure; it can never do the reverse.

Requires: rung_results_N{N}.json files produced by a simulation_engine.py
that records n_positive on every draw record (not just Rule 1 failures).
If n_positive is missing from any record, this script will raise -
rerun run_all_rungs.py with the updated simulation_engine.py first.
"""

import json
from typing import Any

from metrics import METHODS, compute_stability_table_row
from simulation_engine import METHOD_COMPUTATION_OK

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100
OUTPUT_DIR: str = "outputs"
N_PREDICTORS: int = 7

# Phase 3 locked EPV values (predictor-events-per-variable multiplier).
# EPV=2 is the original/current floor, already computed and used as the
# baseline for comparison. EPV=5 and EPV=10 are the sensitivity checks.
EPV_MULTIPLIERS_TO_TEST: list[int] = [2, 5, 10]


def load_rung_records(N: int) -> list[dict]:
    """
    Load the raw per-draw result records for one rung.

    Inputs:
        N (int): rung sample size.
    Returns:
        list[dict]: 100 draw-result records, each required to carry an
            n_positive field regardless of validity outcome.
    Validity rule checked: none (I/O only). Raises KeyError via
        assert_n_positive_present if any record is missing n_positive.
    """
    path = f"{OUTPUT_DIR}/rung_results_N{N}.json"
    with open(path, "r") as f:
        records = json.load(f)
    assert_n_positive_present(records, N)
    return records


def assert_n_positive_present(records: list[dict], N: int) -> None:
    """
    Fail loudly and early if any record lacks n_positive, rather than
    silently producing wrong sensitivity numbers.

    Inputs:
        records (list[dict]): draw records for one rung.
        N (int): rung size, used only for the error message.
    Returns: None. Raises ValueError if any record lacks n_positive.
    Validity rule checked: none (data-integrity guard for this script).
    """
    missing = [r["seed"] for r in records if "n_positive" not in r]
    if missing:
        raise ValueError(
            f"N={N}: {len(missing)} records missing 'n_positive' "
            f"(seeds: {missing[:5]}{'...' if len(missing) > 5 else ''}). "
            f"Rerun run_all_rungs.py with the updated simulation_engine.py "
            f"before running EPV sensitivity analysis."
        )


def refilter_record(record: dict, epv_multiplier: int) -> dict:
    """
    Re-derive one draw's validity outcome under an alternative EPV floor,
    without touching the original model fit / method results.

    Inputs:
        record (dict): original draw record at EPV=2 (baseline run).
        epv_multiplier (int): alternative EPV multiplier to test (e.g. 5, 10).
    Returns:
        dict: a new record with 'valid' and 'fail_rule' possibly changed
            to reflect the new threshold. If the draw already met the new
            (stricter) threshold, its original valid/fail_rule/
            method_results are preserved unchanged. If it no longer meets
            the new threshold, it becomes a rule_1_epv failure regardless
            of its original outcome.
    Validity rule checked: Rule 1 (EPV floor), re-applied at a new
        threshold; Rules 2/3 outcomes are inherited, never recomputed.
    """
    epv_floor = N_PREDICTORS * epv_multiplier
    if record["n_positive"] < epv_floor:
        return {
            **record,
            "valid": False,
            "fail_rule": "rule_1_epv",
            "fail_detail": {"n_positive": record["n_positive"], "epv_floor": epv_floor},
            "method_results": None,
        }
    # Met this threshold: original outcome (valid, or rule_2/rule_3
    # failure) stands unchanged - it was never a function of EPV.
    return record


def build_model_validity_table_at(all_rung_records: dict[int, list[dict]],
                                    epv_multiplier: int) -> list[dict]:
    """
    Build the model-validity failure table at one alternative EPV floor.

    Inputs:
        all_rung_records (dict[int, list[dict]]): rung N -> baseline
            (EPV=2) draw records.
        epv_multiplier (int): EPV multiplier to test.
    Returns:
        list[dict]: 7 rows, same shape as
            final_model_validity_failure_table.json, plus an epv_multiplier
            field.
    Validity rule checked: Rule 1 (re-applied), Rule 7 (fixed K=100
        denominator, preserved under re-filtering).
    """
    table = []
    for N in RUNGS:
        refiltered = [refilter_record(r, epv_multiplier) for r in all_rung_records[N]]
        n_total = len(refiltered)
        n_model_valid = sum(1 for r in refiltered if r["valid"])
        n_model_invalid = n_total - n_model_valid

        row: dict[str, Any] = {
            "epv_multiplier": epv_multiplier,
            "rung_N": N,
            "n_total": n_total,
            "n_model_valid": n_model_valid,
            "n_model_invalid": n_model_invalid,
            "model_validity_failure_rate": n_model_invalid / K,
        }
        for rule_key in ["rule_1_epv", "rule_2_convergence", "rule_3_coefficient"]:
            count = sum(1 for r in refiltered if (
                not r["valid"]) and r["fail_rule"] == rule_key)
            row[rule_key + "_rate"] = count / K
        table.append(row)
    return table


def build_stability_table_at(all_rung_records: dict[int, list[dict]],
                               epv_multiplier: int) -> list[dict]:
    """
    Build the stability table at one alternative EPV floor.

    Inputs:
        all_rung_records (dict[int, list[dict]]): rung N -> baseline
            (EPV=2) draw records.
        epv_multiplier (int): EPV multiplier to test.
    Returns:
        list[dict]: 7 rungs x 4 methods = 28 rows, same shape as
            final_stability_table.json, plus an epv_multiplier field.
    Validity rule checked: Rule 12 (status-field logic), composed with
        the re-applied Rule 1.
    """
    table = []
    for N in RUNGS:
        refiltered = [refilter_record(r, epv_multiplier) for r in all_rung_records[N]]
        model_valid_records = [r for r in refiltered if r["valid"]]
        for method in METHODS:
            method_ok_full_ranks = [
                r["method_results"][method]["full_rank"]
                for r in model_valid_records
                if r["method_results"][method]["status"] == METHOD_COMPUTATION_OK
            ]
            row = compute_stability_table_row(method, method_ok_full_ranks)
            row["rung_N"] = N
            row["epv_multiplier"] = epv_multiplier
            table.append(row)
    return table


def run_epv_sensitivity() -> None:
    """
    Load baseline (EPV=2) rung results, recompute both tables at every
    EPV multiplier in EPV_MULTIPLIERS_TO_TEST, save one combined JSON per
    table (all multipliers together, for easy plotting/comparison), and
    print a compact summary.

    Inputs: none.
    Returns: None. Writes epv_sensitivity_model_validity_table.json and
        epv_sensitivity_stability_table.json to outputs/.
    Validity rule checked: none (orchestration only).
    """
    all_rung_records = {N: load_rung_records(N) for N in RUNGS}

    all_validity_rows = []
    all_stability_rows = []
    for mult in EPV_MULTIPLIERS_TO_TEST:
        all_validity_rows.extend(build_model_validity_table_at(all_rung_records, mult))
        all_stability_rows.extend(build_stability_table_at(all_rung_records, mult))

    with open(f"{OUTPUT_DIR}/epv_sensitivity_model_validity_table.json", "w") as f:
        json.dump(all_validity_rows, f, indent=2)
    with open(f"{OUTPUT_DIR}/epv_sensitivity_stability_table.json", "w") as f:
        json.dump(all_stability_rows, f, indent=2)

    print("=== EPV sensitivity: model_validity_failure_rate by threshold ===")
    print(f"{'N':>6} " + " ".join(f"EPV={m:>2}" for m in EPV_MULTIPLIERS_TO_TEST))
    by_n: dict[int, dict[int, float]] = {N: {} for N in RUNGS}
    for row in all_validity_rows:
        by_n[row["rung_N"]][row["epv_multiplier"]] = row["model_validity_failure_rate"]
    for N in RUNGS:
        vals = " ".join(f"{by_n[N][m]:>6.3f}" for m in EPV_MULTIPLIERS_TO_TEST)
        print(f"{N:>6} {vals}")

    print(f"\nSaved epv_sensitivity_model_validity_table.json and "
          f"epv_sensitivity_stability_table.json to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_epv_sensitivity()
