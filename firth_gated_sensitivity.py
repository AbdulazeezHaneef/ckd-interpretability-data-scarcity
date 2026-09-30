"""
firth_gated_sensitivity.py

O4(b): recomputes model/draw validity using Firth's bias-reduced logistic
regression as the Rule 3 (coefficient-magnitude) estimator, instead of
MLE, for every draw that already passed Rules 1 (EPV) and 2 (convergence)
in the primary run.

Design: EPV and convergence checks are unchanged (reused verbatim from
the primary results already on disk - not recomputed). For every draw
that passed both, MLE and Firth are both fit on the identical
sample/seed; Rule 3's three-part check is applied to the FIRTH beta/SE
instead of MLE's. A draw's validity status can flip either way versus
the primary run:
- MLE-Rule3-failed, Firth-Rule3-passed: now valid. Method_results are
  computed fresh (not available in the primary run) on the MLE fit -
  Firth is only the Rule 3 arbiter here, not a replacement estimator
  for the reported methods (Section 3.4).
- MLE-Rule3-passed, Firth-Rule3-failed: now invalid; original
  method_results are dropped from stability.
- Agreement: status and method_results reused directly from the primary
  run.

Requires: outputs/rung_results_N{N}.json (x7, primary run),
outputs/processed_predictors.csv, outputs/final_model_validity_failure_table.json,
firthmodels >= 0.8.1.
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from firthmodels import FirthLogisticRegression
from scipy.stats import spearmanr

import simulation_engine as sim

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100
OUTPUT_DIR: str = "outputs"
JACCARD_K_VALUES = [3, 5, 7]
MIN_VALID_DRAWS_FOR_STABILITY = 2


def compute_rung_seed_ranges() -> dict[int, range]:
    """
    Reproduce run_all_rungs.py's cumulative seed assignment exactly.

    Inputs: none.
    Returns:
        dict[int, range]: rung N -> range of K seeds.
    Validity rule checked: none (must match Rule 9's fixed rung order).
    """
    ranges = {}
    seed_counter = 0
    for N in RUNGS:
        ranges[N] = range(seed_counter, seed_counter + K)
        seed_counter += K
    return ranges


def load_primary_records(N: int) -> dict[int, dict]:
    """
    Load the primary run's rung_results_N{N}.json, keyed by seed, for
    draws that passed Rules 1 and 2 (EPV + convergence) - the population
    eligible for Firth-gated Rule 3 re-evaluation.

    Inputs:
        N (int): rung sample size.
    Returns:
        dict[int, dict]: seed -> primary record, for seeds that passed
            Rules 1-2 (valid=True, or fail_rule=='rule_3_coefficient').
    Validity rule checked: none (data loading/filtering only).
    """
    with open(f"{OUTPUT_DIR}/rung_results_N{N}.json", "r") as f:
        records = json.load(f)
    return {r["seed"]: r for r in records if r["valid"] or r["fail_rule"] == "rule_3_coefficient"}


def refit_mle_and_firth(df: pd.DataFrame, N: int, seed: int) -> dict:
    """
    Re-draw one sample (identical to the primary run given the same
    seed), refit MLE and Firth, and return everything needed to both
    re-evaluate Rule 3 under Firth and, if it now passes, compute the
    four interpretability methods on the MLE fit.

    Inputs:
        df (pd.DataFrame): full source pool (7 predictors + outcome).
        N (int): draw sample size.
        seed (int): the draw's original seed.
    Returns:
        dict: {X_arr, y, mle_model, mle_beta, firth_beta, firth_se}.
    Validity rule checked: Rule 2 (sampling without replacement),
        reproduced exactly via the same random_state.
    """
    sample = df.sample(n=N, replace=False, random_state=seed)
    X = sim.standardize_continuous(sample[sim.FEATURE_NAMES])
    y = sample[sim.OUTCOME_COL].values
    X_arr = X.values

    mle_model, _ = sim.fit_and_check_convergence(X_arr, y)
    mle_beta = mle_model.coef_[0]

    firth_model = FirthLogisticRegression()
    firth_model.fit(X_arr, y)
    firth_beta = firth_model.coef_
    firth_se = firth_model.bse_[1:]

    return {"X_arr": X_arr, "y": y, "mle_model": mle_model,
            "mle_beta": mle_beta, "firth_beta": firth_beta, "firth_se": firth_se}


def run_firth_gated_rung(df: pd.DataFrame, N: int, seeds: range) -> list[dict]:
    """
    Re-evaluate Rule 3 under Firth for every Rule-1/2-passing draw at one
    rung, computing fresh method_results only for draws that flip from
    MLE-invalid to Firth-valid; reusing the primary run's method_results
    for draws whose status is unchanged.

    Inputs:
        df (pd.DataFrame): full source pool.
        N (int): rung sample size.
        seeds (range): this rung's seed range.
    Returns:
        list[dict]: one record per Rule-1/2-passing seed, with valid,
            fail_rule, mle_valid_primary, status_change, method_results.
    Validity rule checked: Rule 3, re-applied to Firth beta/SE via
        simulation_engine.check_coefficients (identical thresholds, only
        the input estimator differs).
    """
    primary = load_primary_records(N)
    records = []

    for seed in seeds:
        if seed not in primary:
            continue  # failed Rule 1 or 2 in the primary run - unaffected by Firth
        primary_record = primary[seed]
        mle_valid_primary = primary_record["valid"]

        refit = refit_mle_and_firth(df, N, seed)
        firth_failures = sim.check_coefficients(refit["firth_beta"], refit["firth_se"])
        firth_valid = len(firth_failures) == 0
        status_change = mle_valid_primary != firth_valid

        if firth_valid and not status_change:
            method_results = primary_record["method_results"]
        elif firth_valid and status_change:
            beta_importance = dict(zip(sim.FEATURE_NAMES, np.abs(refit["mle_beta"])))
            method_results = sim.compute_all_methods(
                refit["X_arr"], refit["y"], refit["mle_model"], beta_importance, seed)
        else:
            method_results = None

        records.append({
            "seed": seed, "N": N, "valid": firth_valid,
            "fail_rule": None if firth_valid else "rule_3_coefficient_firth",
            "mle_valid_primary": mle_valid_primary, "status_change": status_change,
            "n_positive": primary_record["n_positive"], "method_results": method_results,
        })

    return records


def full_rank_to_vector(full_rank: list[str]) -> np.ndarray:
    """
    Convert a full 7-feature ranking into a numeric rank vector, aligned
    to simulation_engine.FEATURE_NAMES order.

    Inputs:
        full_rank (list[str]): all 7 feature names, descending order.
    Returns:
        np.ndarray: length-7 rank vector.
    Validity rule checked: none.
    """
    ranks = {feat: i for i, feat in enumerate(full_rank, start=1)}
    return np.array([ranks[name] for name in sim.FEATURE_NAMES])


def jaccard_at_k(full_rank_a: list[str], full_rank_b: list[str], k: int) -> float:
    """
    Jaccard index between two draws' top-k feature sets.

    Inputs:
        full_rank_a, full_rank_b (list[str]): two draws' full rankings.
        k (int): top-k cutoff.
    Returns:
        float: |intersection| / |union|.
    Validity rule checked: none.
    """
    set_a, set_b = set(full_rank_a[:k]), set(full_rank_b[:k])
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def compute_pairwise_stability(full_rank_lists: list[list[str]]) -> dict | None:
    """
    Mean pairwise Jaccard (k=3,5,7) and Spearman across valid-draw
    rankings for one method at one rung, under Firth-gated validity.

    Inputs:
        full_rank_lists (list[list[str]]): one full ranking per valid draw.
    Returns:
        dict | None: {jaccard_mean: {k: value}, spearman_mean}, or None
            if fewer than MIN_VALID_DRAWS_FOR_STABILITY lists.
    Validity rule checked: Rule 12 analogue.
    """
    if len(full_rank_lists) < MIN_VALID_DRAWS_FOR_STABILITY:
        return None
    jaccard_scores: dict[int, list[float]] = {k: [] for k in JACCARD_K_VALUES}
    spearman_scores = []
    for a, b in combinations(full_rank_lists, 2):
        for k in JACCARD_K_VALUES:
            jaccard_scores[k].append(jaccard_at_k(a, b, k))
        rho, _ = spearmanr(full_rank_to_vector(a), full_rank_to_vector(b))
        spearman_scores.append(rho if not np.isnan(rho) else 0.0)
    return {
        "jaccard_mean": {k: float(np.mean(v)) for k, v in jaccard_scores.items()},
        "spearman_mean": float(np.mean(spearman_scores)),
    }


def run_firth_gated_sensitivity() -> tuple[list[dict], list[dict]]:
    """
    Run the Firth-gated Rule 3 re-evaluation across all 7 rungs, build
    the resulting model-validity and stability tables, save both, print
    a comparison against the primary (MLE-gated) tables.

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict]]: (model_validity_table, stability_table).
    Validity rule checked: none (orchestration only).
    """
    df = pd.read_csv(f"{OUTPUT_DIR}/processed_predictors.csv")
    seed_ranges = compute_rung_seed_ranges()

    model_validity_table = []
    stability_table = []
    all_rung_records = {}

    for N in RUNGS:
        print(f"Running Firth-gated rung N={N} ...")
        records = run_firth_gated_rung(df, N, seed_ranges[N])
        all_rung_records[N] = records

        with open(f"{OUTPUT_DIR}/firth_gated_rung_results_N{N}.json", "w") as f:
            json.dump(records, f, indent=2)

        n_valid = sum(1 for r in records if r["valid"])
        n_total = len(records)
        n_flip_to_valid = sum(1 for r in records if r["status_change"] and r["valid"])
        n_flip_to_invalid = sum(1 for r in records if r["status_change"] and not r["valid"])

        model_validity_table.append({
            "rung_N": N, "n_rule12_passed": n_total, "n_firth_valid": n_valid,
            "n_flipped_mle_invalid_to_firth_valid": n_flip_to_valid,
            "n_flipped_mle_valid_to_firth_invalid": n_flip_to_invalid,
        })
        print(f"  N={N}: {n_valid}/{n_total} Rule-1/2-passing draws Firth-valid "
              f"(+{n_flip_to_valid} newly valid, -{n_flip_to_invalid} newly invalid)")

    for N in RUNGS:
        records = all_rung_records[N]
        valid_records = [r for r in records if r["valid"]]
        for method in sim.METHODS:
            full_ranks = [
                r["method_results"][method]["full_rank"]
                for r in valid_records
                if r["method_results"][method]["status"] == sim.METHOD_COMPUTATION_OK
            ]
            stability = compute_pairwise_stability(full_ranks)
            row = {"rung_N": N, "method": method, "n_valid_draws": len(full_ranks)}
            if stability is not None:
                row["jaccard_top5_mean"] = stability["jaccard_mean"][5]
                row["spearman_mean"] = stability["spearman_mean"]
            else:
                row["jaccard_top5_mean"] = None
                row["spearman_mean"] = None
            stability_table.append(row)

    with open(f"{OUTPUT_DIR}/firth_gated_model_validity_table.json", "w") as f:
        json.dump(model_validity_table, f, indent=2)
    with open(f"{OUTPUT_DIR}/firth_gated_stability_table.json", "w") as f:
        json.dump(stability_table, f, indent=2)

    print("\n=== Firth-gated vs MLE-gated (primary) model validity ===")
    with open(f"{OUTPUT_DIR}/final_model_validity_failure_table.json", "r") as f:
        primary_validity = {r["rung_N"]: r["model_validity_failure_rate"] for r in json.load(f)}
    print(f"{'N':>6} {'primary (MLE)':>15} {'Firth-gated':>13} {'net flip':>10}")
    for row in model_validity_table:
        N = row["rung_N"]
        firth_failure_rate = 1 - (row["n_firth_valid"] / K)
        net = row["n_flipped_mle_invalid_to_firth_valid"] - row["n_flipped_mle_valid_to_firth_invalid"]
        print(f"{N:>6} {primary_validity[N]:>15.3f} {firth_failure_rate:>13.3f} {net:>+10}")

    print(f"\nSaved firth_gated_rung_results_N{{N}}.json (x7), "
          f"firth_gated_model_validity_table.json, and "
          f"firth_gated_stability_table.json to {OUTPUT_DIR}/")

    return model_validity_table, stability_table


if __name__ == "__main__":
    run_firth_gated_sensitivity()

