"""
firth_comparator.py

Phase 3 Item 7 (Rule 3 comparator, Review Point 3/4): for every draw that
passed Rules 1-2 (EPV floor + optimizer convergence) - regardless of its
Rule 3 (coefficient magnitude) outcome - refit BOTH the standard MLE
logistic regression (reproducing the exact original fit) AND Firth's
bias-reduced logistic regression, using the firthmodels package.

This is an independent check on whether Rule 3 is drawing the right
boundary: if Rule-3-failed draws show a genuinely different Firth-fit
signature (larger Firth SEs, larger MLE-vs-Firth divergence) than
Rule-3-passed draws, that is real evidence Rule 3 is catching real
instability, not an artifact of the bespoke threshold. No arbitrary
threshold is imposed on Firth's output (Firth is a different estimator by
construction and may legitimately produce small, well-behaved
coefficients even where MLE did not) - instead, distributions of Firth
SE / divergence are compared BETWEEN the Rule-3-passed and Rule-3-failed
groups, per rung.

Requires: firthmodels >= 0.8.1 (pip install firthmodels). Requires
rung_results_N{N}.json (any schema - only 'seed', 'valid', 'fail_rule'
are read) and outputs/processed_predictors.csv (same source pool used by
the original simulation).

This performs REAL model refits (not filtering) - it is the actual
computational cost this phase was flagged for. Runtime is driven only by
two logistic-regression-family fits per draw (no SHAP/LIME), so it is far
cheaper than the original full run.
"""

import json

import numpy as np
import pandas as pd
from firthmodels import FirthLogisticRegression

from simulation_engine import (
    CONTINUOUS_PREDICTORS,
    FEATURE_NAMES,
    OUTCOME_COL,
    compute_wald_se,
    fit_and_check_convergence,
    standardize_continuous,
)

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100
OUTPUT_DIR: str = "outputs"
PROCESSED_FILE: str = f"{OUTPUT_DIR}/processed_predictors.csv"

GROUP_RULE3_PASSED = "rule3_passed"  # draws where valid == True
GROUP_RULE3_FAILED = "rule3_failed"  # draws where fail_rule == 'rule_3_coefficient'


def compute_rung_seed_ranges() -> dict[int, range]:
    """
    Reproduce run_all_rungs.py's cumulative seed assignment exactly, so
    each draw is re-sampled identically to the original run.

    Inputs: none.
    Returns:
        dict[int, range]: rung N -> range of K seeds used for that rung,
            in RUNGS order, cumulative starting at 0 (matches
            run_all_rungs.py's seed_counter logic exactly).
    Validity rule checked: none (must match Rule 9's fixed rung order to
        reproduce seeds correctly).
    """
    ranges = {}
    seed_counter = 0
    for N in RUNGS:
        ranges[N] = range(seed_counter, seed_counter + K)
        seed_counter += K
    return ranges


def load_rung_group_membership(N: int) -> dict[int, str]:
    """
    Load one rung's JSON and classify each draw's seed into
    GROUP_RULE3_PASSED, GROUP_RULE3_FAILED, or excluded (failed Rule 1 or
    Rule 2, never reached Rule 3).

    Inputs:
        N (int): rung sample size.
    Returns:
        dict[int, str]: seed -> group label, only for seeds in one of the
            two groups (excluded seeds are simply absent from this dict).
    Validity rule checked: none (classification only).
    """
    with open(f"{OUTPUT_DIR}/rung_results_N{N}.json", "r") as f:
        records = json.load(f)
    membership = {}
    for r in records:
        if r["valid"]:
            membership[r["seed"]] = GROUP_RULE3_PASSED
        elif r["fail_rule"] == "rule_3_coefficient":
            membership[r["seed"]] = GROUP_RULE3_FAILED
        # else: failed rule_1_epv or rule_2_convergence - excluded, never
        # reached a state where Rule 3 or Firth comparison is meaningful.
    return membership


def refit_mle_and_firth(df: pd.DataFrame, N: int, seed: int) -> dict:
    """
    Re-draw one sample (identical to the original run given the same
    seed), standardize it identically, and fit both the standard MLE
    logistic regression and Firth's bias-reduced logistic regression.

    Inputs:
        df (pd.DataFrame): full source pool (7 predictors + has_CKD).
        N (int): draw sample size.
        seed (int): the exact seed originally used for this draw.
    Returns:
        dict: {mle_beta, mle_se, firth_beta, firth_se, firth_converged,
            divergence} - each of the four coefficient/SE fields is a
            list[float] of length 7 (FEATURE_NAMES order); divergence is
            elementwise |firth_beta - mle_beta|, also length 7.
    Validity rule checked: Rule 2 (sampling without replacement) -
        reproduced exactly via the same random_state.
    """
    sample = df.sample(n=N, replace=False, random_state=seed)  # Rule 2, reproduced
    X = standardize_continuous(sample[FEATURE_NAMES])
    y = sample[OUTCOME_COL].values
    X_arr = X.values

    mle_model, _ = fit_and_check_convergence(X_arr, y)
    mle_se = compute_wald_se(mle_model, X_arr)
    mle_beta = mle_model.coef_[0]

    firth_model = FirthLogisticRegression()
    firth_model.fit(X_arr, y)
    firth_beta = firth_model.coef_
    firth_se = firth_model.bse_[1:]  # drop intercept SE (index 0), matches
    # mle_se's convention (compute_wald_se also drops intercept)

    divergence = np.abs(firth_beta - mle_beta)

    return {
        "mle_beta": mle_beta.tolist(),
        "mle_se": mle_se.tolist(),
        "firth_beta": firth_beta.tolist(),
        "firth_se": firth_se.tolist(),
        "firth_converged": bool(firth_model.converged_),
        "divergence": divergence.tolist(),
    }


def summarize_group(draw_results: list[dict]) -> dict:
    """
    Summarize one (rung, group) cell's distribution of per-draw Firth
    SE, MLE-vs-Firth divergence, and |Firth beta|, each first averaged
    across the 7 features within a draw, then summarized across draws.

    Inputs:
        draw_results (list[dict]): refit_mle_and_firth outputs for every
            draw in this (rung, group) cell.
    Returns:
        dict: n_draws, and mean/median of within-draw-averaged
            firth_se, divergence, and |firth_beta|.
    Validity rule checked: none (descriptive summary only).
    """
    if not draw_results:
        return {
            "n_draws": 0, "mean_firth_se": None, "median_firth_se": None,
            "mean_divergence": None, "median_divergence": None,
            "mean_abs_firth_beta": None, "median_abs_firth_beta": None,
            "n_firth_nonconverged": 0,
        }

    per_draw_mean_se = [float(np.mean(r["firth_se"])) for r in draw_results]
    per_draw_mean_div = [float(np.mean(r["divergence"])) for r in draw_results]
    per_draw_mean_abs_beta = [float(np.mean(np.abs(r["firth_beta"]))) for r in draw_results]
    n_nonconverged = sum(1 for r in draw_results if not r["firth_converged"])

    return {
        "n_draws": len(draw_results),
        "mean_firth_se": float(np.mean(per_draw_mean_se)),
        "median_firth_se": float(np.median(per_draw_mean_se)),
        "mean_divergence": float(np.mean(per_draw_mean_div)),
        "median_divergence": float(np.median(per_draw_mean_div)),
        "mean_abs_firth_beta": float(np.mean(per_draw_mean_abs_beta)),
        "median_abs_firth_beta": float(np.median(per_draw_mean_abs_beta)),
        "n_firth_nonconverged": n_nonconverged,
    }


def run_firth_comparator() -> tuple[list[dict], list[dict]]:
    """
    Run the full Firth comparator across all 7 rungs, both groups, save
    raw per-draw results and the group-summary table, print a compact
    comparison.

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict]]: (raw_results, summary_table).
    Validity rule checked: none (orchestration only).
    """
    df = pd.read_csv(PROCESSED_FILE)
    seed_ranges = compute_rung_seed_ranges()

    raw_results = []
    summary_table = []

    for N in RUNGS:
        membership = load_rung_group_membership(N)
        group_draws: dict[str, list[dict]] = {GROUP_RULE3_PASSED: [], GROUP_RULE3_FAILED: []}

        for seed in seed_ranges[N]:
            group = membership.get(seed)
            if group is None:
                continue  # excluded: failed Rule 1 or Rule 2
            fit_result = refit_mle_and_firth(df, N, seed)
            record = {"rung_N": N, "seed": seed, "group": group, **fit_result}
            raw_results.append(record)
            group_draws[group].append(fit_result)

        for group in (GROUP_RULE3_PASSED, GROUP_RULE3_FAILED):
            summary = summarize_group(group_draws[group])
            summary["rung_N"] = N
            summary["group"] = group
            summary_table.append(summary)

    with open(f"{OUTPUT_DIR}/firth_raw_results.json", "w") as f:
        json.dump(raw_results, f, indent=2)
    with open(f"{OUTPUT_DIR}/firth_group_summary_table.json", "w") as f:
        json.dump(summary_table, f, indent=2)

    print("=== Firth comparator: rule3_passed vs rule3_failed, per rung ===")
    print(f"{'N':>6} {'group':>13} {'n':>4} {'mean_SE':>9} {'mean_div':>9} {'mean|beta|':>11}")
    for row in summary_table:
        if row["n_draws"] == 0:
            print(f"{row['rung_N']:>6} {row['group']:>13} {'0':>4}  (no draws in this group)")
            continue
        print(f"{row['rung_N']:>6} {row['group']:>13} {row['n_draws']:>4} "
              f"{row['mean_firth_se']:>9.4f} {row['mean_divergence']:>9.4f} "
              f"{row['mean_abs_firth_beta']:>11.4f}")

    print(f"\nSaved firth_raw_results.json and firth_group_summary_table.json to {OUTPUT_DIR}/")

    return raw_results, summary_table


if __name__ == "__main__":
    run_firth_comparator()
