"""
quantify_overlap.py

Phase 8 (Review Point 13 fix): quantifies the actual pairwise row-overlap
between the 100 draws at N=5000, replacing the manuscript's qualitative
"draws overlap substantially" with a real, empirically-derived number.

Reproduces the exact 100 row-index samples used at EVERY rung (same seeds,
same df.sample() call as simulation_engine.draw_sample_and_fit), then
computes the true pairwise overlap fraction |A intersect B| / N for every
pair of draws at each rung. Originally scoped to N=5000 only (the review's
specific ask, and the rung with the most overlap relative to pool size),
generalized to all 7 rungs since the pool (31,958) is fixed regardless of
N, so every rung has SOME overlap, and reporting the full relationship
(overlap shrinking as N shrinks) is more informative and no more
expensive to compute than reporting one point.

Also reports the theoretical expectation for two independent uniform
samples of size n from a pool of size N (E[|A intersect B|] = n^2/N) at
each rung, as a sanity check that the empirical result lands where the
math predicts.
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd

RUNGS = [5000, 2500, 1000, 500, 250, 100, 50]
K = 100
OUTPUT_DIR = "outputs"


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


def get_sample_indices(df: pd.DataFrame, N: int, seed: int) -> set:
    """
    Reproduce one draw's exact row-index set (Rule 2: sampling without
    replacement, same convention as simulation_engine.draw_sample_and_fit).

    Inputs:
        df (pd.DataFrame): full source pool (processed_predictors.csv,
            with its original 0..len(df)-1 index from preprocess.py's
            reset_index).
        N (int): draw sample size.
        seed (int): the draw's original seed.
    Returns:
        set: the N row indices selected by this draw.
    Validity rule checked: Rule 2 (sampling without replacement),
        reproduced exactly.
    """
    sample = df.sample(n=N, replace=False, random_state=seed)
    return set(sample.index)


def theoretical_expected_overlap(n: int, pool_size: int) -> float:
    """
    Theoretical expected intersection size for two independently drawn
    uniform random samples of size n (without replacement) from a pool of
    size pool_size: E[|A intersect B|] = n^2 / pool_size.

    Inputs:
        n (int): sample size of each draw.
        pool_size (int): total pool size.
    Returns:
        float: expected overlap COUNT (not fraction).
    Validity rule checked: none (closed-form probability calculation).
    """
    return n**2 / pool_size


def quantify_rung_overlap(df: pd.DataFrame, N: int, seeds: range) -> dict:
    """
    Compute empirical pairwise overlap statistics for one rung's 100
    draws, compared against the theoretical expectation.

    Inputs:
        df (pd.DataFrame): full source pool.
        N (int): rung sample size.
        seeds (range): the K seeds used for this rung.
    Returns:
        dict: summary statistics for this rung (mean/median/min/max
            overlap fraction, theoretical expectation).
    Validity rule checked: none (descriptive statistical calculation).
    """
    pool_size = len(df)
    seed_list = list(seeds)
    sample_sets = {seed: get_sample_indices(df, N, seed) for seed in seed_list}

    overlap_fractions = []
    for seed_a, seed_b in combinations(seed_list, 2):
        overlap_count = len(sample_sets[seed_a] & sample_sets[seed_b])
        overlap_fractions.append(overlap_count / N)

    overlap_fractions = np.array(overlap_fractions)
    theoretical_count = theoretical_expected_overlap(N, pool_size)
    theoretical_fraction = theoretical_count / N

    return {
        "rung_N": N,
        "pool_size": pool_size,
        "n_pairs": len(overlap_fractions),
        "empirical_mean_overlap_fraction": float(overlap_fractions.mean()),
        "empirical_median_overlap_fraction": float(np.median(overlap_fractions)),
        "empirical_min_overlap_fraction": float(overlap_fractions.min()),
        "empirical_max_overlap_fraction": float(overlap_fractions.max()),
        "empirical_std_overlap_fraction": float(overlap_fractions.std()),
        "theoretical_expected_overlap_fraction": theoretical_fraction,
    }


def run_quantify_overlap() -> list[dict]:
    """
    Compute empirical pairwise overlap statistics for all 7 rungs, save
    results, print a summary showing overlap shrinking as N shrinks.

    Inputs: none.
    Returns:
        list[dict]: one summary dict per rung (7 total).
    Validity rule checked: none (descriptive statistical calculation).
    """
    df = pd.read_csv(f"{OUTPUT_DIR}/processed_predictors.csv")
    seed_ranges = compute_rung_seed_ranges()

    results = [quantify_rung_overlap(df, N, seed_ranges[N]) for N in RUNGS]

    with open(f"{OUTPUT_DIR}/overlap_quantification_all_rungs.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"=== Pairwise row-overlap by rung, pool={results[0]['pool_size']} ===")
    print(f"{'N':>6} {'empirical_mean':>15} {'empirical_range':>22} {'theoretical':>13}")
    for r in results:
        range_str = f"[{r['empirical_min_overlap_fraction']:.4f}, {r['empirical_max_overlap_fraction']:.4f}]"
        print(f"{r['rung_N']:>6} {r['empirical_mean_overlap_fraction']:>14.4f} {range_str:>22} "
              f"{r['theoretical_expected_overlap_fraction']:>13.4f}")

    print(f"\nSaved overlap_quantification_all_rungs.json to {OUTPUT_DIR}/")

    return results


if __name__ == "__main__":
    run_quantify_overlap()
