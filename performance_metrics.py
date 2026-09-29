"""
performance_metrics.py

Phase 9 (Review Point 8 fix + Phase 1 scope decision A): adds held-out
predictive performance metrics (AUC, Brier score) that were entirely
absent despite "Risk Prediction" in the study's title.

Locked design (Phase 1 / Phase 5 discussion):
  - Runs on ALL 100 draws per rung, gated ONLY by data_splitting.py's own
    split validity (train EPV >= 14, test positives >= 5) - NOT filtered
    by whether the draw passed Rules 1-3 first. This is a genuinely
    independent validity track: a draw's Rules-1-3 (model/draw validity)
    outcome and its performance-evaluability outcome are unrelated.
  - Model fit on TRAIN only (independent of the interpretability-track
    model, which fits on the full draw) - standardization also fit on
    train only (data_splitting.py's leakage-safe convention).
  - AUC (discrimination) and Brier score (discrimination + calibration in
    one number) on the held-out TEST set. Calibration curve/slope
    explicitly excluded - at test sizes as low as 5 positives, a
    calibration curve is not meaningfully fittable; this is disclosed as
    a limitation rather than attempted and reported with false precision.
  - Bootstrap 95% CI (2000 iterations) on each rung's mean AUC and mean
    Brier score. Unlike Phase 6/7's pairwise stability metrics, AUC/Brier
    are single per-draw values, so this is a simple resample-the-values
    bootstrap (no O(n^2) pairwise cost, no vectorization needed).
  - A split-validity table parallel to final_model_validity_failure_table,
    reporting how many of the 100 draws per rung had a valid split vs.
    failed on train-EPV vs. failed on test-positives (same pattern
    permutation_sensitivity.py already used for its own split tracking).
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from data_splitting import (
    SPLIT_INVALID_TEST_POSITIVES,
    SPLIT_INVALID_TRAIN_EPV,
    SPLIT_VALID,
    stratified_split,
)
from simulation_engine import C_REG, MAX_ITER

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100
OUTPUT_DIR: str = "outputs"

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE = 0.95
MIN_DRAWS_FOR_BOOTSTRAP = 2


def compute_rung_seed_ranges() -> dict[int, range]:
    """
    Reproduce run_all_rungs.py's cumulative seed assignment exactly (same
    logic used across every prior phase's sensitivity scripts).

    Inputs: none.
    Returns:
        dict[int, range]: rung N -> range of K seeds used for that rung.
    Validity rule checked: none (must match Rule 9's fixed rung order).
    """
    ranges = {}
    seed_counter = 0
    for N in RUNGS:
        ranges[N] = range(seed_counter, seed_counter + K)
        seed_counter += K
    return ranges


def evaluate_one_draw(df: pd.DataFrame, N: int, seed: int) -> dict:
    """
    Perform one draw's full performance-evaluation pipeline: re-draw the
    sample, split (stratified 80/20, train EPV>=14, test positives>=5),
    and if valid, fit on train and score AUC/Brier on held-out test.

    Inputs:
        df (pd.DataFrame): full source pool (7 predictors + has_CKD).
        N (int): draw sample size.
        seed (int): the draw's seed (same convention as the main pipeline
            and every sensitivity script - reproduces the exact same draw).
    Returns:
        dict: {seed, split_status, auc, brier} - auc/brier are None if
            split_status != SPLIT_VALID.
    Validity rule checked: the independent split-validity track
        (data_splitting.py), NOT Rules 1-3 (this draw is evaluated
        regardless of its Rules 1-3 outcome, per Phase 1/9 locked design).
    """
    sample = df.sample(n=N, replace=False, random_state=seed)  # Rule 2, reproduced
    split = stratified_split(sample, seed)

    if split["status"] != SPLIT_VALID:
        return {"seed": seed, "split_status": split["status"], "auc": None, "brier": None}

    model = LogisticRegression(C=C_REG, max_iter=MAX_ITER)
    model.fit(split["X_train"], split["y_train"])
    y_prob = model.predict_proba(split["X_test"])[:, 1]

    auc = roc_auc_score(split["y_test"], y_prob)
    brier = brier_score_loss(split["y_test"], y_prob)

    return {"seed": seed, "split_status": SPLIT_VALID, "auc": float(auc), "brier": float(brier)}


def bootstrap_mean_ci(values: list[float], seed: int,
                       n_iterations: int = BOOTSTRAP_ITERATIONS,
                       confidence: float = BOOTSTRAP_CONFIDENCE) -> dict:
    """
    Simple bootstrap CI on the mean of a 1D array of per-draw values
    (AUC or Brier). Unlike Phase 6/7's pairwise stability bootstrap, this
    resamples the values themselves directly - AUC/Brier are already
    single per-draw statistics, not pairwise comparisons, so no special
    resampling structure is needed.

    Inputs:
        values (list[float]): per-draw AUC or Brier values for one rung.
        seed (int): fixed seed for reproducibility.
        n_iterations (int): number of bootstrap resamples (default 2000).
        confidence (float): CI level (default 0.95).
    Returns:
        dict: {mean, ci_lower, ci_upper}, or all None if fewer than
            MIN_DRAWS_FOR_BOOTSTRAP values.
    Validity rule checked: none (statistical estimation only).
    """
    if len(values) < MIN_DRAWS_FOR_BOOTSTRAP:
        return {"mean": None, "ci_lower": None, "ci_upper": None}

    arr = np.array(values)
    rng = np.random.RandomState(seed)
    n = len(arr)
    boot_means = np.array([
        arr[rng.choice(n, size=n, replace=True)].mean()
        for _ in range(n_iterations)
    ])
    alpha = 1 - confidence
    return {
        "mean": float(arr.mean()),
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    }


def run_performance_metrics() -> tuple[list[dict], list[dict], list[dict]]:
    """
    Run the full performance-metrics pipeline across all 7 rungs, all 100
    draws each (gated only by split validity), compute bootstrap CIs,
    save all outputs, print a summary.

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict], list[dict]]: (per_draw_records,
            split_validity_table, performance_summary_table).
    Validity rule checked: the independent split-validity track.
    """
    df = pd.read_csv(f"{OUTPUT_DIR}/processed_predictors.csv")
    seed_ranges = compute_rung_seed_ranges()

    per_draw_records = []
    split_validity_table = []
    performance_summary_table = []
    seed_counter = 0

    for N in RUNGS:
        rung_records = [evaluate_one_draw(df, N, seed) for seed in seed_ranges[N]]
        per_draw_records.extend([{**r, "rung_N": N} for r in rung_records])

        n_valid = sum(1 for r in rung_records if r["split_status"] == SPLIT_VALID)
        n_train_epv_fail = sum(1 for r in rung_records if r["split_status"] == SPLIT_INVALID_TRAIN_EPV)
        n_test_pos_fail = sum(1 for r in rung_records if r["split_status"] == SPLIT_INVALID_TEST_POSITIVES)
        split_validity_table.append({
            "rung_N": N, "n_total": K,
            "n_split_valid": n_valid,
            "n_split_invalid_train_epv": n_train_epv_fail,
            "n_split_invalid_test_positives": n_test_pos_fail,
            "split_valid_rate": n_valid / K,
        })

        aucs = [r["auc"] for r in rung_records if r["split_status"] == SPLIT_VALID]
        briers = [r["brier"] for r in rung_records if r["split_status"] == SPLIT_VALID]

        auc_ci = bootstrap_mean_ci(aucs, seed=seed_counter)
        seed_counter += 1
        brier_ci = bootstrap_mean_ci(briers, seed=seed_counter)
        seed_counter += 1

        performance_summary_table.append({
            "rung_N": N, "n_split_valid": n_valid,
            "auc_mean": auc_ci["mean"], "auc_ci_lower": auc_ci["ci_lower"], "auc_ci_upper": auc_ci["ci_upper"],
            "brier_mean": brier_ci["mean"], "brier_ci_lower": brier_ci["ci_lower"], "brier_ci_upper": brier_ci["ci_upper"],
        })

        print(f"N={N}: {n_valid}/{K} split-valid "
              f"({n_train_epv_fail} train-EPV fail, {n_test_pos_fail} test-positives fail)")

    with open(f"{OUTPUT_DIR}/performance_per_draw.json", "w") as f:
        json.dump(per_draw_records, f, indent=2)
    with open(f"{OUTPUT_DIR}/performance_split_validity_table.json", "w") as f:
        json.dump(split_validity_table, f, indent=2)
    with open(f"{OUTPUT_DIR}/performance_summary_table.json", "w") as f:
        json.dump(performance_summary_table, f, indent=2)

    print("\n=== Held-out performance: AUC and Brier score, with bootstrap 95% CI ===")
    print(f"{'N':>6} {'n_valid':>8} {'AUC mean':>10} {'AUC CI':>18} {'Brier mean':>11} {'Brier CI':>18}")
    for row in performance_summary_table:
        if row["auc_mean"] is None:
            print(f"{row['rung_N']:>6} {row['n_split_valid']:>8}  (insufficient split-valid draws)")
            continue
        auc_ci_str = f"[{row['auc_ci_lower']:.3f}, {row['auc_ci_upper']:.3f}]"
        brier_ci_str = f"[{row['brier_ci_lower']:.3f}, {row['brier_ci_upper']:.3f}]"
        print(f"{row['rung_N']:>6} {row['n_split_valid']:>8} {row['auc_mean']:>10.3f} {auc_ci_str:>18} "
              f"{row['brier_mean']:>11.3f} {brier_ci_str:>18}")

    print(f"\nSaved performance_per_draw.json, performance_split_validity_table.json, "
          f"and performance_summary_table.json to {OUTPUT_DIR}/")

    return per_draw_records, split_validity_table, performance_summary_table


if __name__ == "__main__":
    run_performance_metrics()


