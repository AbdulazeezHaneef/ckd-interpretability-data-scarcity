"""
lime_sensitivity.py

Phase 5, item 10 (Review Point 7 fix): re-runs LIME with a larger,
rung-dependent subsample instead of the fixed 50-instance design (Hard
Rule 11), to test whether the original small-subsample choice materially
changed LIME's results.

Locked two-tier design:
  - N <= 500: TRUE FULL DRAW (subsample_size = N). Affordable at these
    rung sizes.
  - N >= 1000: capped at LARGE_TIER_SUBSAMPLE_SIZE (200) - true full draw
    at N=5000/2500 would be prohibitively slow; 200 is still 4x the
    original 50 and removes the most extreme 1%-of-draw imbalance.

For every model-valid draw (Rules 1-3 passed), this:
  1. Re-fits the model exactly as the original run did (same seed, same
     standardization).
  2. Recomputes LIME importance at the new subsample size.
  3. Compares the NEW ranking to the ORIGINAL ranking (already stored in
     rung_results_N{N}.json) for that same draw - a per-draw agreement
     check (does the individual explanation change).
  4. Separately computes the new subsample size's own across-draw
     stability table - a headline-finding check (does the paper's
     stability conclusion change under the harmonized subsample).

WARNING: this is computationally expensive. N=500's full-draw tier alone
is up to 500 LIME explain_instance calls x ~92 valid draws. Total across
the ladder is roughly 125,000 LIME calls. Prints per-rung progress so
runtime can be judged as it goes; if too slow, the fallback is lowering
SMALL_TIER_MAX_N (e.g. to 250) rather than abandoning the two-tier design
- tell Claude and the boundary will be adjusted.
"""

import json

import numpy as np
import pandas as pd

from metrics import compute_stability_table_row, full_rank_to_vector, jaccard_at_k
from simulation_engine import (
    FEATURE_NAMES,
    compute_lime_importance,
    fit_and_check_convergence,
    rank_all_features,
    standardize_continuous,
)
from scipy.stats import kendalltau, spearmanr

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100
OUTPUT_DIR: str = "outputs"

SMALL_TIER_MAX_N = 500  # N <= this: true full draw
LARGE_TIER_SUBSAMPLE_SIZE = 200  # N > SMALL_TIER_MAX_N: capped subsample


def compute_rung_seed_ranges() -> dict[int, range]:
    """
    Reproduce run_all_rungs.py's cumulative seed assignment exactly.
    (Identical logic to firth_comparator.py's version - kept local here
    rather than imported, to avoid a cross-script dependency for a
    six-line function.)

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


def subsample_size_for_rung(N: int) -> int:
    """
    Determine the LIME subsample size for one rung under the two-tier
    sensitivity design.

    Inputs:
        N (int): rung sample size.
    Returns:
        int: N itself (true full draw) if N <= SMALL_TIER_MAX_N, else
            LARGE_TIER_SUBSAMPLE_SIZE.
    Validity rule checked: none (Phase 5 sensitivity design, not a Hard
        Rule).
    """
    return N if N <= SMALL_TIER_MAX_N else LARGE_TIER_SUBSAMPLE_SIZE


def load_valid_seeds_and_original_ranks(N: int) -> dict[int, list[str]]:
    """
    Load one rung's JSON and extract the seed and original (Hard-Rule-11,
    subsample=50) LIME full_rank for every model-valid draw where LIME's
    computation succeeded.

    Inputs:
        N (int): rung sample size.
    Returns:
        dict[int, list[str]]: seed -> original LIME full_rank (length 7).
            Draws that were model-invalid, or where LIME itself
            originally failed computationally, are excluded.
    Validity rule checked: none (data loading only).
    """
    with open(f"{OUTPUT_DIR}/rung_results_N{N}.json", "r") as f:
        records = json.load(f)
    out = {}
    for r in records:
        if not r["valid"]:
            continue
        lime_result = r["method_results"]["lime"]
        if lime_result["status"] == "COMPUTATION_OK":
            out[r["seed"]] = lime_result["full_rank"]
    return out


def agreement_metrics(rank_a: list[str], rank_b: list[str]) -> dict:
    """
    Compute per-draw agreement between two rankings of the same draw
    (original small-subsample LIME vs. new sensitivity-subsample LIME).

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


def run_lime_sensitivity() -> tuple[list[dict], list[dict]]:
    """
    Run the full LIME sensitivity check across all 7 rungs: per-draw
    agreement with the original ranking, plus the new subsample size's
    own across-draw stability table.

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict]]: (per_draw_agreement_records,
            new_stability_table).
    Validity rule checked: none (orchestration only).
    """
    df = pd.read_csv(f"{OUTPUT_DIR}/processed_predictors.csv")
    seed_ranges = compute_rung_seed_ranges()

    per_draw_records = []
    new_stability_table = []

    for N in RUNGS:
        subsample_size = subsample_size_for_rung(N)
        original_ranks = load_valid_seeds_and_original_ranks(N)
        print(f"N={N}: subsample_size={subsample_size}, "
              f"{len(original_ranks)} draws to reprocess ...")

        new_ranks_this_rung = []
        for i, seed in enumerate(seed_ranges[N]):
            if seed not in original_ranks:
                continue

            sample = df.sample(n=N, replace=False, random_state=seed)  # Rule 2, reproduced
            X = standardize_continuous(sample[FEATURE_NAMES])
            y = sample["has_CKD"].values
            X_arr = X.values

            model, _ = fit_and_check_convergence(X_arr, y)
            new_importance = compute_lime_importance(
                X_arr, y, model, seed, subsample_size=subsample_size)
            new_rank = rank_all_features(new_importance)
            new_ranks_this_rung.append(new_rank)

            agreement = agreement_metrics(original_ranks[seed], new_rank)
            per_draw_records.append({
                "rung_N": N, "seed": seed,
                "original_rank": original_ranks[seed], "new_rank": new_rank,
                **agreement,
            })

            if (i + 1) % 20 == 0:
                print(f"  ... {len(new_ranks_this_rung)}/{len(original_ranks)} draws done")

        row = compute_stability_table_row("lime_sensitivity", new_ranks_this_rung)
        row["rung_N"] = N
        row["subsample_size"] = subsample_size
        new_stability_table.append(row)
        print(f"  N={N} done: {len(new_ranks_this_rung)} draws reprocessed")

    with open(f"{OUTPUT_DIR}/lime_sensitivity_per_draw_agreement.json", "w") as f:
        json.dump(per_draw_records, f, indent=2)
    with open(f"{OUTPUT_DIR}/lime_sensitivity_stability_table.json", "w") as f:
        json.dump(new_stability_table, f, indent=2)

    print("\n=== LIME sensitivity: per-draw agreement with original (subsample=50) ===")
    print(f"{'N':>6} {'subsample':>10} {'n':>4} {'mean_jaccard5':>14} {'mean_spearman':>14} {'mean_kendall':>13}")
    by_n: dict[int, list[dict]] = {N: [] for N in RUNGS}
    for r in per_draw_records:
        by_n[r["rung_N"]].append(r)
    for N in RUNGS:
        recs = by_n[N]
        if not recs:
            print(f"{N:>6} {subsample_size_for_rung(N):>10} {'0':>4}  (no draws)")
            continue
        print(f"{N:>6} {subsample_size_for_rung(N):>10} {len(recs):>4} "
              f"{np.mean([r['jaccard_top5'] for r in recs]):>14.4f} "
              f"{np.mean([r['spearman'] for r in recs]):>14.4f} "
              f"{np.mean([r['kendall_tau'] for r in recs]):>13.4f}")

    print(f"\nSaved lime_sensitivity_per_draw_agreement.json and "
          f"lime_sensitivity_stability_table.json to {OUTPUT_DIR}/")

    return per_draw_records, new_stability_table


if __name__ == "__main__":
    run_lime_sensitivity()
