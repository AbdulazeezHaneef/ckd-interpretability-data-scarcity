"""
diabetes_collapse_sensitivity.py

O4(a): re-runs the full 7-rung ladder with diabetes status collapsed to a
single binary predictor (diabetes_any = diabetes_yes OR diabetes_borderline),
instead of the two-dummy encoding used throughout the primary analysis.

Rationale: the independent audit (finding 1) showed Rule 3 failures from
N=1,000 to N=250 were almost entirely driven by the diabetes_borderline
dummy (2.3% prevalence) hitting quasi-separation. This collapse removes
that sparse category entirely, isolating how much of the validity-failure
curve is EPV/sample-size alone versus the sparse-category artifact.

Predictor set drops from 7 to 6 (age, BMI, glucose, sex, hypertension,
diabetes_any); EPV floor is rescaled accordingly (6 predictors x 2 = 12,
down from the primary analysis's 14). All other Rules 1/2/3 mechanics,
seeds, and rung structure are identical to the primary run
(run_all_rungs.py) - this reuses simulation_engine.draw_sample_and_fit
directly by overriding its FEATURE_NAMES/EPV_FLOOR module constants
before calling it, so the fit/SHAP/LIME/permutation code itself is
byte-identical to the primary pipeline, not reimplemented.

Requires: outputs/processed_predictors.csv, diabetes_yes and
diabetes_borderline columns present (as in the primary pool).
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import simulation_engine as sim

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100
OUTPUT_DIR: str = "outputs"

COLLAPSED_BINARY_PREDICTORS = ["RIAGENDR", "BPQ020", "diabetes_any"]
COLLAPSED_FEATURE_NAMES = sim.CONTINUOUS_PREDICTORS + COLLAPSED_BINARY_PREDICTORS
COLLAPSED_EPV_FLOOR = len(COLLAPSED_FEATURE_NAMES) * 2  # 6 x 2 = 12
JACCARD_K_VALUES = [5]  # top-5 of 6 remains the primary reported granularity
MIN_VALID_DRAWS_FOR_STABILITY = 2


def build_collapsed_pool() -> pd.DataFrame:
    """
    Load the primary processed pool and add diabetes_any, collapsing the
    two-dummy diabetes encoding into one binary predictor.

    Inputs: none.
    Returns:
        pd.DataFrame: processed_predictors.csv plus a diabetes_any column
            (1 if diabetes_yes or diabetes_borderline, else 0).
    Validity rule checked: none (data transform only).
    """
    df = pd.read_csv(f"{OUTPUT_DIR}/processed_predictors.csv")
    df["diabetes_any"] = ((df["diabetes_yes"] == 1) | (df["diabetes_borderline"] == 1)).astype(int)
    return df


def compute_rung_seed_ranges() -> dict[int, range]:
    """
    Reproduce run_all_rungs.py's cumulative seed assignment exactly, so
    every draw here is sampled identically to the primary run (same rows
    per seed) - only the predictor set and EPV floor differ.

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


def full_rank_to_vector(full_rank: list[str], feature_names: list[str]) -> np.ndarray:
    """
    Convert a complete feature ranking into a numeric rank vector aligned
    to feature_names order. Local reimplementation (not metrics.py's,
    which hardcodes the primary 7-feature FEATURE_NAMES) so this script
    never depends on another module's predictor-count assumption.

    Inputs:
        full_rank (list[str]): all features, descending-importance order.
        feature_names (list[str]): the predictor set in use (length 6 here).
    Returns:
        np.ndarray: rank vector aligned to feature_names order.
    Validity rule checked: none.
    """
    ranks = {feat: i for i, feat in enumerate(full_rank, start=1)}
    return np.array([ranks[name] for name in feature_names])


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


def compute_pairwise_stability(full_rank_lists: list[list[str]], feature_names: list[str]) -> dict | None:
    """
    Mean pairwise Jaccard (top-5) and Spearman across all valid-draw
    rankings for one method at one rung, on the collapsed 6-predictor set.

    Inputs:
        full_rank_lists (list[list[str]]): one full ranking per valid draw.
        feature_names (list[str]): the predictor set in use.
    Returns:
        dict | None: {jaccard_mean: {k: value}, spearman_mean}, or None
            if fewer than MIN_VALID_DRAWS_FOR_STABILITY lists.
    Validity rule checked: Rule 12 analogue (minimum valid draws).
    """
    if len(full_rank_lists) < MIN_VALID_DRAWS_FOR_STABILITY:
        return None
    jaccard_scores: dict[int, list[float]] = {k: [] for k in JACCARD_K_VALUES}
    spearman_scores = []
    for a, b in combinations(full_rank_lists, 2):
        for k in JACCARD_K_VALUES:
            jaccard_scores[k].append(jaccard_at_k(a, b, k))
        rho, _ = spearmanr(full_rank_to_vector(a, feature_names), full_rank_to_vector(b, feature_names))
        spearman_scores.append(rho if not np.isnan(rho) else 0.0)
    return {
        "jaccard_mean": {k: float(np.mean(v)) for k, v in jaccard_scores.items()},
        "spearman_mean": float(np.mean(spearman_scores)),
    }


def run_collapsed_rung(df: pd.DataFrame, N: int, seeds: range) -> list[dict]:
    """
    Run K draws at one rung under the collapsed 6-predictor set, reusing
    simulation_engine.draw_sample_and_fit with its module-level
    FEATURE_NAMES/EPV_FLOOR overridden to the collapsed spec (set by the
    caller before this runs).

    Inputs:
        df (pd.DataFrame): collapsed pool (build_collapsed_pool's output).
        N (int): rung sample size.
        seeds (range): this rung's seed range.
    Returns:
        list[dict]: K draw-result records, same schema as the primary
            pipeline's rung_results_N{N}.json.
    Validity rule checked: Rules 1-3, applied via the overridden globals.
    """
    return [sim.draw_sample_and_fit(df, N=N, seed=seed) for seed in seeds]


def run_diabetes_collapse_sensitivity() -> tuple[list[dict], list[dict]]:
    """
    Run the full 7-rung ladder under the collapsed diabetes predictor,
    build the model-validity and stability tables, save both, print a
    compact comparison against the primary (7-predictor) results.

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict]]: (model_validity_table, stability_table).
    Validity rule checked: none (orchestration only).
    """
    # Override simulation_engine's module-level predictor spec so
    # draw_sample_and_fit, check_coefficients, compute_shap_importance,
    # and compute_lime_importance all operate on the collapsed 6-feature
    # set without any code duplication. Restored at the end of the run.
    original_feature_names = sim.FEATURE_NAMES
    original_epv_floor = sim.EPV_FLOOR
    sim.FEATURE_NAMES = COLLAPSED_FEATURE_NAMES
    sim.EPV_FLOOR = COLLAPSED_EPV_FLOOR

    model_validity_table = []
    stability_table = []

    try:
        df = build_collapsed_pool()
        seed_ranges = compute_rung_seed_ranges()
        all_rung_records = {}

        for N in RUNGS:
            print(f"Running collapsed-diabetes rung N={N} "
                  f"(seeds {seed_ranges[N].start}-{seed_ranges[N].stop - 1}) ...")
            records = run_collapsed_rung(df, N, seed_ranges[N])
            all_rung_records[N] = records

            with open(f"{OUTPUT_DIR}/diabetes_collapse_rung_results_N{N}.json", "w") as f:
                json.dump(records, f, indent=2)

            n_valid = sum(1 for r in records if r["valid"])
            n_invalid = K - n_valid
            row = {
                "rung_N": N, "n_total": K,
                "n_model_valid": n_valid, "n_model_invalid": n_invalid,
                "model_validity_failure_rate": n_invalid / K,
            }
            for rule_key in ["rule_1_epv", "rule_2_convergence", "rule_3_coefficient"]:
                count = sum(1 for r in records if (not r["valid"]) and r["fail_rule"] == rule_key)
                row[f"{rule_key}_rate"] = count / K
            model_validity_table.append(row)
            print(f"  N={N}: {n_valid}/{K} valid ({n_invalid} failed)")

        for N in RUNGS:
            records = all_rung_records[N]
            model_valid_records = [r for r in records if r["valid"]]
            for method in sim.METHODS:
                full_ranks = [
                    r["method_results"][method]["full_rank"]
                    for r in model_valid_records
                    if r["method_results"][method]["status"] == sim.METHOD_COMPUTATION_OK
                ]
                stability = compute_pairwise_stability(full_ranks, COLLAPSED_FEATURE_NAMES)
                row = {"rung_N": N, "method": method, "n_valid_draws": len(full_ranks)}
                if stability is not None:
                    row["jaccard_top5_mean"] = stability["jaccard_mean"][5]
                    row["spearman_mean"] = stability["spearman_mean"]
                else:
                    row["jaccard_top5_mean"] = None
                    row["spearman_mean"] = None
                stability_table.append(row)

    finally:
        # Always restore, even if the run raises - other scripts in the
        # same session must not see the collapsed predictor set.
        sim.FEATURE_NAMES = original_feature_names
        sim.EPV_FLOOR = original_epv_floor

    with open(f"{OUTPUT_DIR}/diabetes_collapse_model_validity_table.json", "w") as f:
        json.dump(model_validity_table, f, indent=2)
    with open(f"{OUTPUT_DIR}/diabetes_collapse_stability_table.json", "w") as f:
        json.dump(stability_table, f, indent=2)

    print("\n=== Collapsed-diabetes (6-predictor) model validity vs primary (7-predictor) ===")
    with open(f"{OUTPUT_DIR}/final_model_validity_failure_table.json", "r") as f:
        primary_validity = {r["rung_N"]: r["model_validity_failure_rate"] for r in json.load(f)}
    print(f"{'N':>6} {'primary (7-pred)':>18} {'collapsed (6-pred)':>20}")
    for row in model_validity_table:
        N = row["rung_N"]
        print(f"{N:>6} {primary_validity[N]:>18.3f} {row['model_validity_failure_rate']:>20.3f}")

    print("\n=== Collapsed-diabetes mean Spearman stability (across 4 methods) by rung ===")
    for N in RUNGS:
        vals = [r["spearman_mean"] for r in stability_table if r["rung_N"] == N and r["spearman_mean"] is not None]
        print(f"N={N:>5}  mean_spearman={np.mean(vals):.3f}" if vals else f"N={N:>5}  (no valid draws)")

    print(f"\nSaved diabetes_collapse_rung_results_N{{N}}.json (x7), "
          f"diabetes_collapse_model_validity_table.json, and "
          f"diabetes_collapse_stability_table.json to {OUTPUT_DIR}/")

    return model_validity_table, stability_table


if __name__ == "__main__":
    run_diabetes_collapse_sensitivity()

