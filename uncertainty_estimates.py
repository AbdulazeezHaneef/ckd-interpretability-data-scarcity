"""
uncertainty_estimates.py

Phase 6 (Review Point 14 fix): adds uncertainty intervals to the two
tables that were previously reported as bare point estimates.

1. final_model_validity_failure_table -> Wilson score 95% CIs on
   model_validity_failure_rate per rung (n_total = K = 100 always, Hard
   Rule 7's fixed denominator). Wilson chosen over the normal/Wald
   approximation because several rates sit exactly at 0.000 or 1.000
   (N=5000/2500 and N=50), where Wald CIs produce nonsensical
   negative-lower or >1-upper bounds; Wilson stays well-behaved there.

2. final_stability_table -> bootstrap 95% CIs on jaccard_top3/top5/top7,
   spearman, and kendall_tau, per (rung, method). Resampling is done at
   the DRAW level (not the pairwise-comparison level): each bootstrap
   iteration draws N valid-draw rankings WITH replacement from the
   original set of valid draws, then recomputes all pairwise statistics
   among that resampled set via metrics.compute_pairwise_stability. This
   is the statistically correct unit of resampling - the ~C(n,2) pairwise
   values are not independent observations (each draw appears in many
   pairs), so resampling pairs directly would understate uncertainty.
   2000 iterations, 95% CI via the percentile method, fixed per-(rung,
   method) seed for reproducibility.

   jaccard_top7 is 1.0 by mathematical construction (both top-7 sets
   always equal the full 7-feature set) - its bootstrap CI is correctly
   degenerate ([1.0, 1.0]), not a bug.

Pure computation on existing rung_results_N{N}.json files - no
resimulation, no local run required.
"""

import json

import numpy as np
from scipy.stats import norm

from metrics import FEATURE_NAMES, JACCARD_K_VALUES, METHODS
from simulation_engine import METHOD_COMPUTATION_OK

RUNGS: list[int] = [5000, 2500, 1000, 500, 250, 100, 50]
K: int = 100
OUTPUT_DIR: str = "outputs"

WILSON_CONFIDENCE = 0.95
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE = 0.95
MIN_DRAWS_FOR_BOOTSTRAP = 2  # matches metrics.MIN_VALID_DRAWS_FOR_STABILITY


def vectorized_pairwise_stability(full_rank_lists: list[list[str]]) -> dict:
    """
    Exact, vectorized reimplementation of metrics.compute_pairwise_stability,
    used ONLY inside the bootstrap loop below. metrics.py's original
    per-pair scipy-based function is correct but has fixed per-call
    overhead that is negligible once (it produced the already-verified
    Phase 4 point estimates) but becomes infeasible at bootstrap scale -
    2000 resamples x up to C(100,2)=4950 pairs x 28 (rung, method) cells
    means tens of millions of scipy calls, projected at hours per cell.

    This function computes the IDENTICAL statistics via closed-form
    matrix operations (no per-pair scipy calls) and was validated to
    match metrics.compute_pairwise_stability to within 1e-9 on real data
    across all 4 methods before being used here. metrics.py itself is
    left untouched - this is a bootstrap-loop-only optimization.

    Inputs:
        full_rank_lists (list[list[str]]): one full (length-7) ranking
            per draw (may include duplicates - bootstrap resamples with
            replacement).
    Returns:
        dict: {jaccard_mean: {k: value}, spearman_mean, kendall_tau_mean} -
            identical shape to metrics.compute_pairwise_stability's output
            (assumes len(full_rank_lists) >= 2, guarded by the caller).
    Validity rule checked: none (statistical computation only; Rule 12's
        minimum-draws gate is enforced by the caller, bootstrap_stability_ci).
    """
    n = len(full_rank_lists)
    name_to_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    p = len(FEATURE_NAMES)

    R = np.zeros((n, p), dtype=np.int64)
    for i, ranking in enumerate(full_rank_lists):
        for rank_pos, feat in enumerate(ranking, start=1):
            R[i, name_to_idx[feat]] = rank_pos

    # Spearman (exact, no ties): rho = 1 - 6*sum(d^2)/(p*(p^2-1))
    diff = R[:, None, :] - R[None, :, :]
    sumsq = (diff ** 2).sum(axis=2)
    spearman_matrix = 1 - 6 * sumsq / (p * (p**2 - 1))

    # Kendall's tau (exact, no ties) via sign-matrix Gram trick:
    # S[i,a,b] = sign(R[i,a]-R[i,b]); tau_ij = 0.5 * sum_ab S[i,a,b]*S[j,a,b] / C(p,2)
    S = np.sign(R[:, :, None] - R[:, None, :])
    S_flat = S.reshape(n, p * p).astype(np.float64)
    gram = S_flat @ S_flat.T
    kendall_matrix = 0.5 * gram / (p * (p - 1) / 2)

    # Jaccard@k (exact) via membership-vector dot products: since every
    # top-k set has exactly k members (no ties), union = 2k - intersection.
    jaccard_matrices = {}
    for k in JACCARD_K_VALUES:
        M = (R <= k).astype(np.float64)
        intersection = M @ M.T
        union = 2 * k - intersection
        jaccard_matrices[k] = intersection / union

    iu = np.triu_indices(n, k=1)
    return {
        "jaccard_mean": {k: float(jaccard_matrices[k][iu].mean()) for k in JACCARD_K_VALUES},
        "spearman_mean": float(spearman_matrix[iu].mean()),
        "kendall_tau_mean": float(kendall_matrix[iu].mean()),
    }


def wilson_score_interval(n_success: int, n_total: int, confidence: float = WILSON_CONFIDENCE) -> tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a binomial
    proportion. Well-behaved at rates of exactly 0.0 or 1.0, unlike the
    normal/Wald approximation.

    Inputs:
        n_success (int): number of "successes" (here: failed draws, since
            we're computing a CI on the failure rate).
        n_total (int): total trials (K = 100, Hard Rule 7).
        confidence (float): confidence level (default 0.95).
    Returns:
        tuple[float, float]: (lower, upper) bound, both in [0, 1].
    Validity rule checked: none (statistical estimation only).
    """
    if n_total == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = p_hat + z**2 / (2 * n_total)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def add_failure_rate_cis(model_validity_table: list[dict]) -> list[dict]:
    """
    Add Wilson score 95% CI bounds to each row of
    final_model_validity_failure_table.

    Inputs:
        model_validity_table (list[dict]): as loaded from
            final_model_validity_failure_table.json.
    Returns:
        list[dict]: same rows, each with
            model_validity_failure_rate_ci_lower/upper added.
    Validity rule checked: Rule 7 (fixed K=100 denominator, used as
        n_total for every CI).
    """
    out = []
    for row in model_validity_table:
        lower, upper = wilson_score_interval(row["n_model_invalid"], row["n_total"])
        out.append({
            **row,
            "model_validity_failure_rate_ci_lower": lower,
            "model_validity_failure_rate_ci_upper": upper,
        })
    return out


def bootstrap_stability_ci(full_rank_lists: list[list[str]], seed: int,
                            n_iterations: int = BOOTSTRAP_ITERATIONS,
                            confidence: float = BOOTSTRAP_CONFIDENCE) -> dict:
    """
    Draw-level bootstrap CI for all stability metrics at one (rung,
    method) cell.

    Inputs:
        full_rank_lists (list[list[str]]): the ORIGINAL (not yet
            resampled) full rankings from every valid draw at this cell.
        seed (int): fixed seed for this cell's bootstrap resampling
            (reproducibility).
        n_iterations (int): number of bootstrap resamples (default 2000).
        confidence (float): CI level (default 0.95).
    Returns:
        dict: for each metric key (jaccard_top3, jaccard_top5,
            jaccard_top7, spearman, kendall_tau) -> {ci_lower, ci_upper}.
            All None if fewer than MIN_DRAWS_FOR_BOOTSTRAP draws.
    Validity rule checked: Rule 12 (minimum valid draws, reused from
        metrics.py's own gate via compute_pairwise_stability returning
        None below that threshold).
    """
    metric_keys = ["jaccard_top3", "jaccard_top5", "jaccard_top7", "spearman", "kendall_tau"]
    if len(full_rank_lists) < MIN_DRAWS_FOR_BOOTSTRAP:
        return {m: {"ci_lower": None, "ci_upper": None} for m in metric_keys}

    rng = np.random.RandomState(seed)
    n = len(full_rank_lists)
    boot_values: dict[str, list[float]] = {m: [] for m in metric_keys}

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        resampled = [full_rank_lists[i] for i in idx]
        stability = vectorized_pairwise_stability(resampled)
        boot_values["jaccard_top3"].append(stability["jaccard_mean"][3])
        boot_values["jaccard_top5"].append(stability["jaccard_mean"][5])
        boot_values["jaccard_top7"].append(stability["jaccard_mean"][7])
        boot_values["spearman"].append(stability["spearman_mean"])
        boot_values["kendall_tau"].append(stability["kendall_tau_mean"])

    alpha = 1 - confidence
    result = {}
    for m in metric_keys:
        vals = boot_values[m]
        if not vals:
            result[m] = {"ci_lower": None, "ci_upper": None}
        else:
            result[m] = {
                "ci_lower": float(np.percentile(vals, 100 * alpha / 2)),
                "ci_upper": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            }
    return result


def load_full_rank_lists_by_method(N: int) -> dict[str, list[list[str]]]:
    """
    Load one rung's JSON and extract, per method, the full_rank lists for
    every draw where that method's computation succeeded on a
    model-valid draw (mirrors compute_final_tables.build_stability_table's
    filtering exactly, so CIs are computed on the identical draw set that
    produced the point estimates).

    Inputs:
        N (int): rung sample size.
    Returns:
        dict[str, list[list[str]]]: method -> list of full_rank lists.
    Validity rule checked: Rule 12 composed with the method-computation
        check (same composition as build_stability_table).
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


def add_stability_cis(stability_table: list[dict]) -> list[dict]:
    """
    Add bootstrap 95% CI bounds to each row of final_stability_table, by
    reloading the underlying draw-level full_rank data (the point-estimate
    table alone doesn't carry enough information to bootstrap from).

    Inputs:
        stability_table (list[dict]): as loaded from
            final_stability_table.json (used only to know which rows/
            (rung, method) combinations exist and to attach CIs to).
    Returns:
        list[dict]: same rows, each with 5 metrics' ci_lower/ci_upper
            fields added.
    Validity rule checked: Rule 12 (via bootstrap_stability_ci's internal
        gate).
    """
    rung_data_cache: dict[int, dict[str, list[list[str]]]] = {}
    out = []
    seed_counter = 0  # deterministic per-(rung, method) seed, incrementing

    for row in stability_table:
        N = row["rung_N"]
        method = row["method"]
        if N not in rung_data_cache:
            rung_data_cache[N] = load_full_rank_lists_by_method(N)

        full_rank_lists = rung_data_cache[N][method]
        ci = bootstrap_stability_ci(full_rank_lists, seed=seed_counter)
        seed_counter += 1

        out.append({
            **row,
            "jaccard_top3_ci_lower": ci["jaccard_top3"]["ci_lower"],
            "jaccard_top3_ci_upper": ci["jaccard_top3"]["ci_upper"],
            "jaccard_top5_ci_lower": ci["jaccard_top5"]["ci_lower"],
            "jaccard_top5_ci_upper": ci["jaccard_top5"]["ci_upper"],
            "jaccard_top7_ci_lower": ci["jaccard_top7"]["ci_lower"],
            "jaccard_top7_ci_upper": ci["jaccard_top7"]["ci_upper"],
            "spearman_ci_lower": ci["spearman"]["ci_lower"],
            "spearman_ci_upper": ci["spearman"]["ci_upper"],
            "kendall_tau_ci_lower": ci["kendall_tau"]["ci_lower"],
            "kendall_tau_ci_upper": ci["kendall_tau"]["ci_upper"],
        })
    return out


def run_uncertainty_estimates() -> tuple[list[dict], list[dict]]:
    """
    Load the existing final_model_validity_failure_table and
    final_stability_table, add CIs to both, save the CI-augmented
    versions, print a compact summary.

    Inputs: none.
    Returns:
        tuple[list[dict], list[dict]]: (model_validity_with_ci,
            stability_with_ci).
    Validity rule checked: none (orchestration only).
    """
    with open(f"{OUTPUT_DIR}/final_model_validity_failure_table.json", "r") as f:
        model_validity_table = json.load(f)
    with open(f"{OUTPUT_DIR}/final_stability_table.json", "r") as f:
        stability_table = json.load(f)

    model_validity_with_ci = add_failure_rate_cis(model_validity_table)
    stability_with_ci = add_stability_cis(stability_table)

    with open(f"{OUTPUT_DIR}/final_model_validity_failure_table_with_ci.json", "w") as f:
        json.dump(model_validity_with_ci, f, indent=2)
    with open(f"{OUTPUT_DIR}/final_stability_table_with_ci.json", "w") as f:
        json.dump(stability_with_ci, f, indent=2)

    print("=== Model validity failure rate, with Wilson 95% CI ===")
    for row in model_validity_with_ci:
        print(f"N={row['rung_N']:>5}  rate={row['model_validity_failure_rate']:.3f}  "
              f"CI=[{row['model_validity_failure_rate_ci_lower']:.3f}, "
              f"{row['model_validity_failure_rate_ci_upper']:.3f}]")

    print("\n=== Stability (Spearman), with bootstrap 95% CI ===")
    for row in stability_with_ci:
        if row["spearman_mean"] is None:
            print(f"N={row['rung_N']:>5}  {row['method']:<18}  (insufficient valid draws)")
            continue
        print(f"N={row['rung_N']:>5}  {row['method']:<18}  spearman={row['spearman_mean']:.3f}  "
              f"CI=[{row['spearman_ci_lower']:.3f}, {row['spearman_ci_upper']:.3f}]")

    print(f"\nSaved final_model_validity_failure_table_with_ci.json and "
          f"final_stability_table_with_ci.json to {OUTPUT_DIR}/")

    return model_validity_with_ci, stability_with_ci


if __name__ == "__main__":
    run_uncertainty_estimates()
