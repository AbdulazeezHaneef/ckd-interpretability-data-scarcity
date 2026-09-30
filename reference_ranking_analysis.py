"""
reference_ranking_analysis.py

O5: builds a full-pool (N=31,958) reference feature ranking for each of
the four interpretability methods, then scores every draw's ranking (at
every rung) against that reference - directly answering B6 (agreement
between draws measures reproducibility, not correctness).

Also joins each draw's held-out AUC (performance_per_draw.json, already
computed by performance_metrics.py - no refit) against its own
reference-ranking agreement, to test whether performance tracks how
close a draw's ranking is to the reference (B5).

LIME subsample size for the reference fit is set to 200 (the two-tier
sensitivity design's large-tier cap, lime_sensitivity.py), not the
primary 50-instance design - the reference fit runs once on the full
pool, and the sensitivity check already showed subsample size barely
moves LIME's ranking (Spearman agreement >=0.99, Supplementary S1.3).

Requires: outputs/processed_predictors.csv, outputs/rung_results_N{N}.json
(x7, primary run), outputs/performance_per_draw.json.
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import permutation_importance

import simulation_engine as sim

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
OUTPUT_DIR: str = "outputs"
REFERENCE_LIME_SUBSAMPLE = 200
MIN_PAIRS_FOR_CORRELATION = 5


def fit_reference_model(df: pd.DataFrame) -> dict:
    """
    Fit the logistic regression on the FULL analytic pool (N=31,958) and
    compute all four methods' full rankings on it - the reference against
    which every rung's draws are later scored.

    Inputs:
        df (pd.DataFrame): processed_predictors.csv, full pool.
    Returns:
        dict: {beta, full_ranks: {method: list[str]}}.
    Validity rule checked: Rule 2 (convergence) only, as a sanity check -
        Rule 1 (EPV) is not meaningful at this N; Rule 3 is not used as a
        gate here since the reference is ground truth by construction.
    """
    X = sim.standardize_continuous(df[sim.FEATURE_NAMES])
    y = df[sim.OUTCOME_COL].values
    X_arr = X.values

    model, converged = sim.fit_and_check_convergence(X_arr, y)
    if not converged:
        raise RuntimeError("reference model failed to converge on the full pool - unexpected at N=31,958")

    beta = model.coef_[0]
    beta_importance = dict(zip(sim.FEATURE_NAMES, np.abs(beta)))

    perm_result = permutation_importance(model, X_arr, y, n_repeats=10, scoring="roc_auc", random_state=0)

    full_ranks = {
        "standardized_beta": sim.rank_all_features(beta_importance),
        "permutation": sim.rank_all_features(dict(zip(sim.FEATURE_NAMES, perm_result.importances_mean))),
        "shap": sim.rank_all_features(sim.compute_shap_importance(X_arr, model)),
        "lime": sim.rank_all_features(sim.compute_lime_importance(
            X_arr, y, model, seed=0, subsample_size=REFERENCE_LIME_SUBSAMPLE)),
    }

    return {"beta": beta.tolist(), "full_ranks": full_ranks}


def jaccard_at_k(full_rank_a: list[str], full_rank_b: list[str], k: int) -> float:
    """
    Jaccard index between two rankings' top-k feature sets.

    Inputs:
        full_rank_a, full_rank_b (list[str]): two full rankings.
        k (int): top-k cutoff.
    Returns:
        float: |intersection| / |union|.
    Validity rule checked: none.
    """
    set_a, set_b = set(full_rank_a[:k]), set(full_rank_b[:k])
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def spearman_agreement(full_rank_a: list[str], full_rank_b: list[str]) -> float:
    """
    Spearman correlation between two full rankings.

    Inputs:
        full_rank_a, full_rank_b (list[str]): two full rankings.
    Returns:
        float: Spearman rho (0.0 if undefined).
    Validity rule checked: none.
    """
    ranks_a = {feat: i for i, feat in enumerate(full_rank_a, start=1)}
    ranks_b = {feat: i for i, feat in enumerate(full_rank_b, start=1)}
    vec_a = np.array([ranks_a[name] for name in sim.FEATURE_NAMES])
    vec_b = np.array([ranks_b[name] for name in sim.FEATURE_NAMES])
    rho, _ = spearmanr(vec_a, vec_b)
    return rho if not np.isnan(rho) else 0.0


def load_performance_lookup() -> dict[tuple[int, int], float | None]:
    """
    Load performance_per_draw.json (already computed - not recomputed
    here) into a (rung_N, seed) -> held-out AUC lookup.

    Inputs: none.
    Returns:
        dict[tuple[int, int], float | None]: AUC, or None if split-invalid.
    Validity rule checked: none (reuses the existing independent
        split-validity track, data_splitting.py, unchanged).
    """
    with open(f"{OUTPUT_DIR}/performance_per_draw.json", "r") as f:
        records = json.load(f)
    return {(r["rung_N"], r["seed"]): r["auc"] for r in records}


def score_draws_against_reference(reference_full_ranks: dict[str, list[str]],
                                    auc_lookup: dict[tuple[int, int], float | None]) -> list[dict]:
    """
    For every rung, every model-valid draw, every method: compute
    Jaccard@5 and Spearman agreement between that draw's ranking and the
    reference ranking for the same method, and attach the draw's
    held-out AUC if available.

    Inputs:
        reference_full_ranks (dict[str, list[str]]): method -> reference
            full ranking (fit_reference_model's output).
        auc_lookup (dict[tuple[int, int], float | None]): (rung_N, seed)
            -> held-out AUC.
    Returns:
        list[dict]: one row per (rung, seed, method).
    Validity rule checked: Rule 12 composed with the method-computation
        check (same filtering every prior phase uses).
    """
    rows = []
    for N in RUNGS:
        with open(f"{OUTPUT_DIR}/rung_results_N{N}.json", "r") as f:
            records = json.load(f)
        for r in records:
            if not r["valid"]:
                continue
            auc = auc_lookup.get((N, r["seed"]))
            for method in sim.METHODS:
                res = r["method_results"][method]
                if res["status"] != sim.METHOD_COMPUTATION_OK:
                    continue
                ref_rank = reference_full_ranks[method]
                rows.append({
                    "rung_N": N, "seed": r["seed"], "method": method,
                    "jaccard5_vs_reference": jaccard_at_k(res["full_rank"], ref_rank, 5),
                    "spearman_vs_reference": spearman_agreement(res["full_rank"], ref_rank),
                    "auc": auc,
                })
    return rows


def summarize_auc_vs_agreement(scored_rows: list[dict]) -> list[dict]:
    """
    Per rung, per method: Pearson correlation between a draw's held-out
    AUC and its Spearman agreement with the reference ranking, across
    only draws with both values available.

    Inputs:
        scored_rows (list[dict]): score_draws_against_reference's output.
    Returns:
        list[dict]: one row per (rung, method), with n_pairs, correlation
            (None if fewer than MIN_PAIRS_FOR_CORRELATION usable pairs),
            mean_spearman_vs_reference, mean_auc.
    Validity rule checked: none (descriptive correlation only).
    """
    summary = []
    for N in RUNGS:
        for method in sim.METHODS:
            subset = [r for r in scored_rows if r["rung_N"] == N and r["method"] == method and r["auc"] is not None]
            n = len(subset)
            row = {"rung_N": N, "method": method, "n_pairs": n,
                   "correlation": None, "mean_spearman_vs_reference": None, "mean_auc": None}
            if n >= MIN_PAIRS_FOR_CORRELATION:
                aucs = [r["auc"] for r in subset]
                agreements = [r["spearman_vs_reference"] for r in subset]
                if np.std(aucs) > 0 and np.std(agreements) > 0:
                    r_val, _ = pearsonr(aucs, agreements)
                    row["correlation"] = float(r_val)
                row["mean_spearman_vs_reference"] = float(np.mean(agreements))
                row["mean_auc"] = float(np.mean(aucs))
            summary.append(row)
    return summary


def run_reference_ranking_analysis() -> tuple[dict, list[dict], list[dict]]:
    """
    Fit the full-pool reference ranking, score every draw against it,
    summarize the AUC-vs-agreement relationship, save all three outputs,
    print a compact summary.

    Inputs: none.
    Returns:
        tuple[dict, list[dict], list[dict]]: (reference, scored_rows,
            auc_vs_agreement_summary).
    Validity rule checked: none (orchestration only).
    """
    df = pd.read_csv(f"{OUTPUT_DIR}/processed_predictors.csv")
    reference = fit_reference_model(df)
    auc_lookup = load_performance_lookup()

    scored_rows = score_draws_against_reference(reference["full_ranks"], auc_lookup)
    auc_vs_agreement = summarize_auc_vs_agreement(scored_rows)

    with open(f"{OUTPUT_DIR}/reference_ranking.json", "w") as f:
        json.dump(reference, f, indent=2)
    with open(f"{OUTPUT_DIR}/draws_vs_reference_scored.json", "w") as f:
        json.dump(scored_rows, f, indent=2)
    with open(f"{OUTPUT_DIR}/auc_vs_reference_agreement_summary.json", "w") as f:
        json.dump(auc_vs_agreement, f, indent=2)

    print("=== Full-pool (N=31,958) reference ranking, by method ===")
    for method, rank in reference["full_ranks"].items():
        print(f"  {method:<18}: {rank}")

    print("\n=== Mean Spearman agreement with reference, by rung (across 4 methods) ===")
    for N in RUNGS:
        vals = [r["spearman_vs_reference"] for r in scored_rows if r["rung_N"] == N]
        print(f"N={N:>5}  mean={np.mean(vals):.3f}  n={len(vals)}" if vals else f"N={N:>5}  (no valid draws)")

    print(f"\n=== AUC vs. reference-agreement correlation, by rung x method (n>={MIN_PAIRS_FOR_CORRELATION} pairs) ===")
    for row in auc_vs_agreement:
        if row["correlation"] is not None:
            print(f"N={row['rung_N']:>5}  {row['method']:<18}  n={row['n_pairs']:>3}  "
                  f"r={row['correlation']:>+.3f}  mean_auc={row['mean_auc']:.3f}  "
                  f"mean_agreement={row['mean_spearman_vs_reference']:.3f}")
        else:
            print(f"N={row['rung_N']:>5}  {row['method']:<18}  n={row['n_pairs']:>3}  (insufficient pairs)")

    print(f"\nSaved reference_ranking.json, draws_vs_reference_scored.json, and "
          f"auc_vs_reference_agreement_summary.json to {OUTPUT_DIR}/")

    return reference, scored_rows, auc_vs_agreement


if __name__ == "__main__":
    run_reference_ranking_analysis()

