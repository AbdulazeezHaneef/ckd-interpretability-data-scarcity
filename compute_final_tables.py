"""
compute_final_tables.py

Loads all 7 per-rung JSON result files and computes three final tables,
saved to outputs/:

  1. final_model_validity_failure_table  - Rules 1-3 failure rates, per
     rung, per rule. RENAMED from the old "failure rate table" (Phase 2 /
     Review Point 1 fix): this has always measured model/draw estimability,
     never method-specific failure - the name now says so.
  2. final_method_computational_failure_table  - NEW (Phase 2). Among
     draws that already passed Rules 1-3 (a valid model exists), how often
     did each of the four methods individually fail to compute a usable
     top-5 ranking. This is the genuinely method-specific failure rate the
     old table's name implied but did not measure.
  3. final_stability_table  - Jaccard/Spearman stability, 7 rungs x 4
     methods, now built only from draws where that specific method's
     computation succeeded (status == COMPUTATION_OK), not just from
     model-valid draws generally.
"""

import json
from typing import Any

from metrics import METHODS, compute_stability_table_row
from simulation_engine import METHOD_COMPUTATION_FAILED, METHOD_COMPUTATION_OK

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100  # Hard Rule 7/8: fixed denominator
OUTPUT_DIR: str = "outputs"

RULE_KEYS: list[str] = ["rule_1_epv",
                        "rule_2_convergence", "rule_3_coefficient"]
RULE_RATE_NAMES: dict[str, str] = {
    "rule_1_epv": "rule_1_rate",
    "rule_2_convergence": "rule_2_rate",
    "rule_3_coefficient": "rule_3_rate",
}


def load_rung_records(N: int) -> list[dict]:
    """
    Load the raw per-draw result records for one rung.

    Inputs:
        N (int): rung sample size.
    Returns:
        list[dict]: 100 draw-result records as produced by
            draw_sample_and_fit (Phase 2 schema: method_results replaces
            the old flat top5 dict).
    Validity rule checked: none (I/O only).
    """
    path = f"{OUTPUT_DIR}/rung_results_N{N}.json"
    with open(path, "r") as f:
        return json.load(f)


def build_stability_table(all_rung_records: dict[int, list[dict]]) -> list[dict]:
    """
    Build final_stability_table: one row per (rung, method), using only
    draws where that specific method's computation succeeded.

    Phase 4 update: reads method_results[method]["full_rank"] (the
    complete 7-feature ranking) instead of the old "top5" field. All
    downstream Jaccard@3/5/7, Spearman, and Kendall's tau are computed
    from this single full ranking inside compute_stability_table_row.

    Inputs:
        all_rung_records (dict[int, list[dict]]): rung N -> 100 draw
            records.
    Returns:
        list[dict]: 7 rungs x 4 methods = 28 rows, each with rung, method,
            jaccard_top3_mean, jaccard_top5_mean, jaccard_top7_mean,
            spearman_mean, kendall_tau_mean, n_valid_draws, status.
    Validity rule checked: Rule 12 (status-field logic per method per
        rung), composed with the method-computation check.
    """
    table = []
    for N in RUNGS:
        records = all_rung_records[N]
        model_valid_records = [r for r in records if r["valid"]]
        for method in METHODS:
            method_ok_full_ranks = [
                r["method_results"][method]["full_rank"]
                for r in model_valid_records
                if r["method_results"][method]["status"] == METHOD_COMPUTATION_OK
            ]
            row = compute_stability_table_row(method, method_ok_full_ranks)
            row["rung_N"] = N
            table.append(row)
    return table


def build_model_validity_failure_table(all_rung_records: dict[int, list[dict]]) -> list[dict]:
    """
    Build final_model_validity_failure_table: one row per rung with total
    counts, overall model/draw validity failure rate, and per-rule
    breakdown, computed against the fixed denominator K=100 (Hard Rule 7).

    RENAMED from build_failure_rate_table (Phase 2 / Review Point 1 fix).
    Mechanically identical to the original - Rules 1-3 are unchanged - but
    the name and output keys now correctly describe this as a model/draw
    estimability failure rate, not a method-specific one. Method-specific
    failure is now reported separately in
    build_method_computational_failure_table.

    Inputs:
        all_rung_records (dict[int, list[dict]]): rung N -> 100 draw
            records.
    Returns:
        list[dict]: 7 rows, each with rung_N, n_total, n_model_valid,
            n_model_invalid, model_validity_failure_rate, rule_1_rate,
            rule_2_rate, rule_3_rate.
    Validity rule checked: Rule 7 (fixed denominator, no redraws to
        replace failures).
    """
    table = []
    for N in RUNGS:
        records = all_rung_records[N]
        n_total = len(records)
        n_model_valid = sum(1 for r in records if r["valid"])
        n_model_invalid = n_total - n_model_valid

        row: dict[str, Any] = {
            "rung_N": N,
            "n_total": n_total,
            "n_model_valid": n_model_valid,
            "n_model_invalid": n_model_invalid,
            "model_validity_failure_rate": n_model_invalid / K,
        }
        for rule_key in RULE_KEYS:
            count = sum(1 for r in records if (
                not r["valid"]) and r["fail_rule"] == rule_key)
            row[RULE_RATE_NAMES[rule_key]] = count / K
        table.append(row)
    return table


def build_method_computational_failure_table(all_rung_records: dict[int, list[dict]]) -> list[dict]:
    """
    Build final_method_computational_failure_table: one row per (rung,
    method), reporting computational failures among model-valid draws
    only. NEW in Phase 2 (Review Point 1 fix) - this is the first table
    in the pipeline that is genuinely method-specific, since it is
    computed only on draws that already have a valid model and asks,
    independently per method, whether that method's own computation
    (permutation importance / SHAP / LIME internals, or a degenerate
    NaN/inf output) succeeded.

    standardized_beta is included for schema symmetry; it is derived
    directly from already-validated coefficients and should show a
    computational failure rate of 0.0 at every rung by construction - a
    nonzero value here would itself indicate a bug worth investigating,
    not a real finding.

    Inputs:
        all_rung_records (dict[int, list[dict]]): rung N -> 100 draw
            records.
    Returns:
        list[dict]: 7 rungs x 4 methods = 28 rows, each with rung_N,
            method, n_model_valid_draws, n_computation_failed,
            computation_failure_rate (None if n_model_valid_draws == 0,
            to avoid a division-by-zero rate at rungs like N=50 where no
            draw ever reaches method computation).
    Validity rule checked: none (this is the new method-specific
        computational-failure check, distinct from Rules 1-3).
    """
    table = []
    for N in RUNGS:
        records = all_rung_records[N]
        model_valid_records = [r for r in records if r["valid"]]
        n_model_valid = len(model_valid_records)
        for method in METHODS:
            n_failed = sum(
                1 for r in model_valid_records
                if r["method_results"][method]["status"] == METHOD_COMPUTATION_FAILED
            )
            rate = (n_failed / n_model_valid) if n_model_valid > 0 else None
            table.append({
                "rung_N": N,
                "method": method,
                "n_model_valid_draws": n_model_valid,
                "n_computation_failed": n_failed,
                "computation_failure_rate": rate,
            })
    return table


def run_compute_final_tables() -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load all rungs, compute all three final tables, save to outputs/, and
    print the sanity-check summary (Workflow Step 6).

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict], list[dict]]: (final_stability_table,
            final_model_validity_failure_table,
            final_method_computational_failure_table).
    Validity rule checked: Rule 12 (N=50 sanity check, all four methods
        must read INSUFFICIENT_VALID_DRAWS if fewer than 2 valid draws).
    """
    all_rung_records = {N: load_rung_records(N) for N in RUNGS}

    stability_table = build_stability_table(all_rung_records)
    model_validity_table = build_model_validity_failure_table(all_rung_records)
    method_computational_table = build_method_computational_failure_table(all_rung_records)

    with open(f"{OUTPUT_DIR}/final_stability_table.json", "w") as f:
        json.dump(stability_table, f, indent=2)
    with open(f"{OUTPUT_DIR}/final_model_validity_failure_table.json", "w") as f:
        json.dump(model_validity_table, f, indent=2)
    with open(f"{OUTPUT_DIR}/final_method_computational_failure_table.json", "w") as f:
        json.dump(method_computational_table, f, indent=2)

    print("=== final_model_validity_failure_table (renamed from failure_rate_table) ===")
    for row in model_validity_table:
        print(f"N={row['rung_N']:>5}  model_validity_failure_rate={row['model_validity_failure_rate']:.3f}  "
              f"rule_1={row['rule_1_rate']:.3f}  rule_2={row['rule_2_rate']:.3f}  "
              f"rule_3={row['rule_3_rate']:.3f}")

    print("\n=== final_method_computational_failure_table (NEW, Phase 2) ===")
    for row in method_computational_table:
        rate_str = f"{row['computation_failure_rate']:.3f}" if row['computation_failure_rate'] is not None else "n/a (0 model-valid draws)"
        print(f"N={row['rung_N']:>5}  {row['method']:<18}  n_model_valid={row['n_model_valid_draws']:>3}  "
              f"n_failed={row['n_computation_failed']:>3}  rate={rate_str}")

    print("\n=== N=50 status check (all 4 methods) ===")
    for row in stability_table:
        if row["rung_N"] == 50:
            print(
                f"  {row['method']}: status={row['status']}, n_valid_draws={row['n_valid_draws']}")

    print(
        f"\nSaved final_stability_table.json, final_model_validity_failure_table.json, "
        f"and final_method_computational_failure_table.json to {OUTPUT_DIR}/")

    return stability_table, model_validity_table, method_computational_table


if __name__ == "__main__":
    run_compute_final_tables()
