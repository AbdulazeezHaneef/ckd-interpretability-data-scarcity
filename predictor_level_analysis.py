"""
predictor_level_analysis.py

New analysis (Phase 10 prep, Objective 4): computes per-FEATURE stability
- not per-method aggregate stability - to answer the original manuscript's
Objective 4 ("which predictor categories are most vulnerable to
instability") with real evidence across the full ladder.

The original manuscript answered this from only 2 of 7 rungs (N=500,
N=250) and was flagged (both by the independent review this pipeline is
built on, and in this project's own Discussion 5.3) as too thin an
evidentiary base for the confidence with which it was stated. This
analysis is derivable from data already sitting in every
rung_results_N{N}.json (the full_rank field, Phase 4) - no resimulation
needed - and extends the answer to all 6 rungs with valid draws (N=50
excluded, 0 valid draws everywhere).

For each feature, at each rung, for each method:
  - top5_appearance_rate: fraction of that method's COMPUTATION_OK draws
    at that rung where this feature appeared in the top-5.
  - mean_rank: that feature's average rank position (1-7) across those
    same draws (lower = more consistently important).

Also produces a method-aggregated version (mean top5_appearance_rate
across the 4 methods) for the predictor-level heatmap - features whose
vulnerability is a general property, not specific to one method's
extraction mechanics, are the ones the manuscript's Objective 4 claim is
actually about.
"""

import json

import numpy as np

from metrics import FEATURE_NAMES, METHODS
from simulation_engine import METHOD_COMPUTATION_OK

RUNGS_WITH_VALID_DRAWS: list[int] = [5000, 2500, 1000, 500, 250, 100]  # N=50 excluded
TOP_K = 5
OUTPUT_DIR: str = "outputs"


def load_full_ranks_by_method(N: int) -> dict[str, list[list[str]]]:
    """
    Load one rung's JSON and extract, per method, the full_rank lists for
    every draw where that method's computation succeeded on a
    model-valid draw (identical filtering to every prior phase that reads
    this field).

    Inputs:
        N (int): rung sample size.
    Returns:
        dict[str, list[list[str]]]: method -> list of full_rank lists.
    Validity rule checked: Rule 12 composed with the method-computation
        check.
    """
    with open(f"{OUTPUT_DIR}/rung_results_N{N}.json", "r") as f:
        records = json.load(f)
    model_valid = [r for r in records if r["valid"]]
    out = {}
    for method in METHODS:
        out[method] = [
            r["method_results"][method]["full_rank"]
            for r in model_valid
            if r["method_results"][method]["status"] == METHOD_COMPUTATION_OK
        ]
    return out


def compute_feature_stats(full_rank_lists: list[list[str]], feature: str) -> dict:
    """
    Compute one feature's top-5 appearance rate and mean rank across a
    set of draws' full rankings.

    Inputs:
        full_rank_lists (list[list[str]]): full 7-feature rankings across
            draws.
        feature (str): the feature to compute stats for.
    Returns:
        dict: {top5_appearance_rate, mean_rank, n_draws}. None values if
            n_draws == 0.
    Validity rule checked: none (descriptive statistic).
    """
    n = len(full_rank_lists)
    if n == 0:
        return {"top5_appearance_rate": None, "mean_rank": None, "n_draws": 0}

    ranks = []
    n_in_top5 = 0
    for ranking in full_rank_lists:
        pos = ranking.index(feature) + 1  # 1-indexed rank
        ranks.append(pos)
        if pos <= TOP_K:
            n_in_top5 += 1

    return {
        "top5_appearance_rate": n_in_top5 / n,
        "mean_rank": float(np.mean(ranks)),
        "n_draws": n,
    }


def run_predictor_level_analysis() -> tuple[list[dict], list[dict]]:
    """
    Compute per-feature, per-method, per-rung stats, plus a
    method-aggregated version, save both, print a compact heatmap-style
    summary of the aggregated top5_appearance_rate.

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict]]: (per_method_table,
            aggregated_table).
    Validity rule checked: none (orchestration only).
    """
    per_method_table = []
    aggregated_table = []

    for N in RUNGS_WITH_VALID_DRAWS:
        full_ranks_by_method = load_full_ranks_by_method(N)

        for feature in FEATURE_NAMES:
            method_rates = []
            for method in METHODS:
                stats = compute_feature_stats(full_ranks_by_method[method], feature)
                per_method_table.append({
                    "rung_N": N, "method": method, "feature": feature, **stats,
                })
                if stats["top5_appearance_rate"] is not None:
                    method_rates.append(stats["top5_appearance_rate"])

            aggregated_table.append({
                "rung_N": N, "feature": feature,
                "mean_top5_appearance_rate_across_methods": (
                    float(np.mean(method_rates)) if method_rates else None
                ),
                "n_methods_included": len(method_rates),
            })

    with open(f"{OUTPUT_DIR}/predictor_level_per_method.json", "w") as f:
        json.dump(per_method_table, f, indent=2)
    with open(f"{OUTPUT_DIR}/predictor_level_aggregated.json", "w") as f:
        json.dump(aggregated_table, f, indent=2)

    print("=== Predictor-level top-5 appearance rate, aggregated across methods ===")
    header = "Feature".ljust(22) + "".join(f"N={N:<7}" for N in RUNGS_WITH_VALID_DRAWS)
    print(header)
    for feature in FEATURE_NAMES:
        row_vals = []
        for N in RUNGS_WITH_VALID_DRAWS:
            match = next(r for r in aggregated_table if r["rung_N"] == N and r["feature"] == feature)
            rate = match["mean_top5_appearance_rate_across_methods"]
            row_vals.append(f"{rate:.3f}  " if rate is not None else "  n/a  ")
        print(feature.ljust(22) + "".join(row_vals))

    print(f"\nSaved predictor_level_per_method.json and predictor_level_aggregated.json to {OUTPUT_DIR}/")

    return per_method_table, aggregated_table


if __name__ == "__main__":
    run_predictor_level_analysis()
