"""
permutation_sensitivity.py

Phase 5, item 11 (Review Point 7 fix): recomputes permutation importance
using a held-out test set instead of the original in-sample evaluation,
via the shared stratified 80/20 split (data_splitting.py, same design
Phase 9 will reuse for AUC/Brier).

For every model-valid draw (Rules 1-3 passed):
  1. Re-draw the sample (same seed), perform the stratified split.
  2. If the split is invalid (train EPV < 14 or test positives < 5),
     record SPLIT_INVALID and exclude this draw from the sensitivity
     stability calculation - this does NOT affect the draw's Rules 1-3
     validity or its original in-sample permutation ranking, which are
     on an independent validity track (Phase 1 decision).
  3. If valid: fit the model on train only, compute permutation
     importance on the held-out test set (n_repeats=10, same seed
     convention as the original in-sample version).
  4. Compare the new held-out ranking to the ORIGINAL in-sample ranking
     for that same draw (per-draw agreement), and separately compute the
     held-out version's own across-draw stability table (headline-finding
     check).
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from data_splitting import SPLIT_VALID, stratified_split
from metrics import compute_stability_table_row, full_rank_to_vector, jaccard_at_k
from simulation_engine import C_REG, FEATURE_NAMES, MAX_ITER, rank_all_features

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100
OUTPUT_DIR: str = "outputs"


def compute_rung_seed_ranges() -> dict[int, range]:
    """
    Reproduce run_all_rungs.py's cumulative seed assignment exactly.

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


def load_valid_seeds_and_original_ranks(N: int) -> dict[int, list[str]]:
    """
    Load one rung's JSON and extract the seed and original (in-sample)
    permutation-importance full_rank for every model-valid draw where
    permutation importance's computation succeeded.

    Inputs:
        N (int): rung sample size.
    Returns:
        dict[int, list[str]]: seed -> original permutation full_rank.
    Validity rule checked: none (data loading only).
    """
    with open(f"{OUTPUT_DIR}/rung_results_N{N}.json", "r") as f:
        records = json.load(f)
    out = {}
    for r in records:
        if not r["valid"]:
            continue
        perm_result = r["method_results"]["permutation"]
        if perm_result["status"] == "COMPUTATION_OK":
            out[r["seed"]] = perm_result["full_rank"]
    return out


def agreement_metrics(rank_a: list[str], rank_b: list[str]) -> dict:
    """
    Compute per-draw agreement between two rankings of the same draw.

    Inputs:
        rank_a (list[str]): first full 7-feature ranking.
        rank_b (list[str]): second full 7-feature ranking.
    Returns:
        dict: {jaccard_top5, spearman, kendall_tau} between the two.
    Validity rule checked: none.
    """
    vec_a, vec_b = full_rank_to_vector(rank_a), full_rank_to_vector(rank_b)
    rho, _ = spearmanr(vec_a, vec_b)
    tau, _ = kendalltau(vec_a, vec_b)
    return {
        "jaccard_top5": jaccard_at_k(rank_a, rank_b, 5),
        "spearman": rho if not np.isnan(rho) else 0.0,
        "kendall_tau": tau if not np.isnan(tau) else 0.0,
    }


def run_permutation_sensitivity() -> tuple[list[dict], list[dict]]:
    """
    Run the full held-out permutation importance sensitivity check across
    all 7 rungs.

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict]]: (per_draw_records, new_stability_table).
        per_draw_records includes SPLIT_INVALID draws (with rank fields
        None) so the split-exclusion rate itself is visible and reportable.
    Validity rule checked: the performance-evaluability split validity
        (data_splitting.py), independent of Rules 1-3.
    """
    df = pd.read_csv(f"{OUTPUT_DIR}/processed_predictors.csv")
    seed_ranges = compute_rung_seed_ranges()

    per_draw_records = []
    new_stability_table = []

    for N in RUNGS:
        original_ranks = load_valid_seeds_and_original_ranks(N)
        print(f"N={N}: {len(original_ranks)} model-valid draws to reprocess ...")

        new_ranks_this_rung = []
        n_split_invalid = 0

        for seed in seed_ranges[N]:
            if seed not in original_ranks:
                continue

            sample = df.sample(n=N, replace=False, random_state=seed)  # Rule 2, reproduced
            split = stratified_split(sample, seed)

            if split["status"] != SPLIT_VALID:
                n_split_invalid += 1
                per_draw_records.append({
                    "rung_N": N, "seed": seed, "split_status": split["status"],
                    "original_rank": original_ranks[seed], "new_rank": None,
                    "jaccard_top5": None, "spearman": None, "kendall_tau": None,
                })
                continue

            model = LogisticRegression(C=C_REG, max_iter=MAX_ITER)
            model.fit(split["X_train"], split["y_train"])
            perm_result = permutation_importance(
                model, split["X_test"], split["y_test"], n_repeats=10,
                scoring="roc_auc", random_state=seed)
            new_importance = dict(zip(FEATURE_NAMES, perm_result.importances_mean))
            new_rank = rank_all_features(new_importance)
            new_ranks_this_rung.append(new_rank)

            agreement = agreement_metrics(original_ranks[seed], new_rank)
            per_draw_records.append({
                "rung_N": N, "seed": seed, "split_status": SPLIT_VALID,
                "original_rank": original_ranks[seed], "new_rank": new_rank,
                **agreement,
            })

        row = compute_stability_table_row("permutation_heldout", new_ranks_this_rung)
        row["rung_N"] = N
        row["n_split_invalid"] = n_split_invalid
        row["n_model_valid_draws"] = len(original_ranks)
        new_stability_table.append(row)
        print(f"  N={N} done: {len(new_ranks_this_rung)} usable, "
              f"{n_split_invalid} split-invalid (of {len(original_ranks)} model-valid)")

    with open(f"{OUTPUT_DIR}/permutation_sensitivity_per_draw_agreement.json", "w") as f:
        json.dump(per_draw_records, f, indent=2)
    with open(f"{OUTPUT_DIR}/permutation_sensitivity_stability_table.json", "w") as f:
        json.dump(new_stability_table, f, indent=2)

    print("\n=== Permutation held-out sensitivity: per-draw agreement with original (in-sample) ===")
    print(f"{'N':>6} {'n_used':>7} {'n_split_inv':>11} {'mean_jaccard5':>14} {'mean_spearman':>14} {'mean_kendall':>13}")
    for row in new_stability_table:
        N = row["rung_N"]
        recs = [r for r in per_draw_records if r["rung_N"] == N and r["split_status"] == SPLIT_VALID]
        if not recs:
            print(f"{N:>6} {'0':>7} {row['n_split_invalid']:>11}  (no usable draws)")
            continue
        print(f"{N:>6} {len(recs):>7} {row['n_split_invalid']:>11} "
              f"{np.mean([r['jaccard_top5'] for r in recs]):>14.4f} "
              f"{np.mean([r['spearman'] for r in recs]):>14.4f} "
              f"{np.mean([r['kendall_tau'] for r in recs]):>13.4f}")

    print(f"\nSaved permutation_sensitivity_per_draw_agreement.json and "
          f"permutation_sensitivity_stability_table.json to {OUTPUT_DIR}/")

    return per_draw_records, new_stability_table


if __name__ == "__main__":
    run_permutation_sensitivity()
