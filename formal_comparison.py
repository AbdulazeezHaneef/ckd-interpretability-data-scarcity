"""
formal_comparison.py

Phase 7 (Review Point 15 fix): formal statistical test of whether method
pairs' stability metrics are distinguishable, respecting the repeated-draw
structure - not just descriptive rankings side by side.

Design (locked): paired draw-level bootstrap on the difference. For a
given rung and metric, each bootstrap iteration resamples ONE set of draw
indices (with replacement) and applies that SAME resampled index set to
BOTH methods being compared, computing stability_A - stability_B on
identical resampled draws each time. This is what makes it "paired" - it
respects that both methods were evaluated on the same underlying draws,
rather than treating them as independent samples. Reuses
uncertainty_estimates.vectorized_pairwise_stability (Phase 6, validated
to 1e-9 against metrics.py's original scipy-based implementation).

Scope: all 6 unique method pairs x 6 rungs with valid draws (N=50
excluded - 0 valid draws for every method) x 2 metrics (Spearman, top-5
Jaccard - the paper's primary reported statistics) = 72 comparisons.

Significance: two-sided bootstrap p-value per comparison
(p = 2*min(P(diff<=0), P(diff>=0)) from the 2000 bootstrap differences),
then Benjamini-Hochberg FDR correction applied across all 72 p-values
jointly (not per-metric or per-rung separately - one correction across
the full comparison family, per Phase 7 discussion).

Pairing safety: only draws where BOTH methods in a pair have
COMPUTATION_OK status at the same seed are used (defensive - Phase 2
found 0.000 computational failure rate everywhere in this study, so this
reduces to the full model-valid draw set in practice, but the code does
not assume that).
"""

import json
from itertools import combinations

import numpy as np

from metrics import METHODS
from simulation_engine import METHOD_COMPUTATION_OK
from uncertainty_estimates import BOOTSTRAP_ITERATIONS, vectorized_pairwise_stability

RUNGS_WITH_VALID_DRAWS: list[int] = [5000, 2500, 1000, 500, 250, 100]  # N=50 excluded
OUTPUT_DIR: str = "outputs"
METRICS_TO_TEST = ["spearman", "jaccard_top5"]
MIN_DRAWS_FOR_TEST = 2


def load_paired_full_ranks(N: int, method_a: str, method_b: str) -> tuple[list[list[str]], list[list[str]]]:
    """
    Load one rung's JSON and extract full_rank lists for both methods in
    a pair, restricted to draws where BOTH methods have COMPUTATION_OK
    status at the same seed (defensive pairing - see module docstring).

    Inputs:
        N (int): rung sample size.
        method_a (str): first method name.
        method_b (str): second method name.
    Returns:
        tuple[list[list[str]], list[list[str]]]: (method_a's full_rank
            list, method_b's full_rank list), same length, index-aligned
            by draw (draw i in both lists is the same underlying seed).
    Validity rule checked: Rule 12 composed with the method-computation
        check, intersected across both methods.
    """
    with open(f"{OUTPUT_DIR}/rung_results_N{N}.json", "r") as f:
        records = json.load(f)
    ranks_a, ranks_b = [], []
    for r in records:
        if not r["valid"]:
            continue
        res_a = r["method_results"][method_a]
        res_b = r["method_results"][method_b]
        if res_a["status"] == METHOD_COMPUTATION_OK and res_b["status"] == METHOD_COMPUTATION_OK:
            ranks_a.append(res_a["full_rank"])
            ranks_b.append(res_b["full_rank"])
    return ranks_a, ranks_b


def metric_value(stability: dict, metric: str) -> float:
    """
    Extract one named metric's value from a vectorized_pairwise_stability
    result dict.

    Inputs:
        stability (dict): output of vectorized_pairwise_stability.
        metric (str): one of "spearman", "jaccard_top5" (extend here if
            more metrics are added to METRICS_TO_TEST later).
    Returns:
        float: the requested metric's value.
    Validity rule checked: none.
    """
    if metric == "spearman":
        return stability["spearman_mean"]
    if metric == "jaccard_top5":
        return stability["jaccard_mean"][5]
    raise ValueError(f"unsupported metric: {metric}")


def paired_bootstrap_test(ranks_a: list[list[str]], ranks_b: list[list[str]],
                           metric: str, seed: int,
                           n_iterations: int = BOOTSTRAP_ITERATIONS) -> dict:
    """
    Paired draw-level bootstrap test of stability_A - stability_B for one
    metric, at one rung, for one method pair.

    Inputs:
        ranks_a (list[list[str]]): method A's full_rank per draw.
        ranks_b (list[list[str]]): method B's full_rank per draw, same
            length and draw-alignment as ranks_a.
        metric (str): one of METRICS_TO_TEST.
        seed (int): fixed seed for this comparison's bootstrap resampling.
        n_iterations (int): number of bootstrap resamples (default 2000,
            matches Phase 6's BOOTSTRAP_ITERATIONS for consistency).
    Returns:
        dict: {point_estimate_diff, ci_lower, ci_upper, p_value}, or all
            None if fewer than MIN_DRAWS_FOR_TEST paired draws.
    Validity rule checked: Rule 12 (minimum draws), via MIN_DRAWS_FOR_TEST.
    """
    n = len(ranks_a)
    if n < MIN_DRAWS_FOR_TEST:
        return {"point_estimate_diff": None, "ci_lower": None, "ci_upper": None, "p_value": None}

    point_a = metric_value(vectorized_pairwise_stability(ranks_a), metric)
    point_b = metric_value(vectorized_pairwise_stability(ranks_b), metric)
    point_diff = point_a - point_b

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)  # SAME resampled indices for both methods - this is the pairing
        resampled_a = [ranks_a[j] for j in idx]
        resampled_b = [ranks_b[j] for j in idx]
        val_a = metric_value(vectorized_pairwise_stability(resampled_a), metric)
        val_b = metric_value(vectorized_pairwise_stability(resampled_b), metric)
        diffs[i] = val_a - val_b

    ci_lower, ci_upper = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    p_le = float(np.mean(diffs <= 0))
    p_ge = float(np.mean(diffs >= 0))
    p_value = min(1.0, 2 * min(p_le, p_ge))

    return {
        "point_estimate_diff": float(point_diff),
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "p_value": p_value,
    }


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Standard Benjamini-Hochberg FDR correction, applied across the full
    set of p-values jointly.

    Inputs:
        p_values (list[float]): raw p-values, in the same order they will
            be reassembled into the results table (order matters only for
            reassembly, not for the correction itself).
        alpha (float): FDR level (default 0.05).
    Returns:
        list[dict]: one entry per input p-value (same order), each with
            {q_value, significant_at_fdr_05}. q_value is the BH-adjusted
            p-value (smallest FDR level at which this comparison would be
            called significant).
    Validity rule checked: none (standard multiple-comparisons procedure).
    """
    n = len(p_values)
    indexed = sorted(range(n), key=lambda i: p_values[i])
    q_values = [0.0] * n
    prev_q = 1.0
    for rank, i in enumerate(reversed(indexed), start=1):
        # process from largest p-value down to smallest (standard BH step-up)
        k = n - rank + 1
        raw_q = p_values[i] * n / k
        prev_q = min(prev_q, raw_q)
        q_values[i] = prev_q

    return [
        {"q_value": q, "significant_at_fdr_05": q <= alpha}
        for q in q_values
    ]


def run_formal_comparison() -> list[dict]:
    """
    Run all 72 paired bootstrap comparisons (6 method pairs x 6 rungs x 2
    metrics), apply BH correction jointly across all of them, save the
    results table, print a compact summary of significant findings.

    Inputs: none.
    Returns:
        list[dict]: 72 rows, each with rung_N, method_a, method_b, metric,
            point_estimate_diff, ci_lower, ci_upper, p_value, q_value,
            significant_at_fdr_05, n_paired_draws.
    Validity rule checked: none (orchestration only).
    """
    method_pairs = list(combinations(METHODS, 2))
    results = []
    seed_counter = 0

    for N in RUNGS_WITH_VALID_DRAWS:
        for method_a, method_b in method_pairs:
            ranks_a, ranks_b = load_paired_full_ranks(N, method_a, method_b)
            for metric in METRICS_TO_TEST:
                test = paired_bootstrap_test(ranks_a, ranks_b, metric, seed=seed_counter)
                seed_counter += 1
                results.append({
                    "rung_N": N, "method_a": method_a, "method_b": method_b,
                    "metric": metric, "n_paired_draws": len(ranks_a),
                    **test,
                })

    p_values = [r["p_value"] if r["p_value"] is not None else 1.0 for r in results]
    bh_results = benjamini_hochberg(p_values)
    for r, bh in zip(results, bh_results):
        r["q_value"] = bh["q_value"]
        r["significant_at_fdr_05"] = bh["significant_at_fdr_05"]

    with open(f"{OUTPUT_DIR}/formal_comparison_table.json", "w") as f:
        json.dump(results, f, indent=2)

    n_significant = sum(1 for r in results if r["significant_at_fdr_05"])
    print(f"=== Formal comparison: {len(results)} tests, {n_significant} significant at FDR 0.05 ===\n")
    print(f"{'N':>6} {'pair':<28} {'metric':<12} {'diff':>8} {'CI':>20} {'q':>8} {'sig':>5}")
    for r in results:
        if r["point_estimate_diff"] is None:
            continue
        pair_str = f"{r['method_a']} vs {r['method_b']}"
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        sig_str = "YES" if r["significant_at_fdr_05"] else "no"
        print(f"{r['rung_N']:>6} {pair_str:<28} {r['metric']:<12} "
              f"{r['point_estimate_diff']:>8.3f} {ci_str:>20} {r['q_value']:>8.4f} {sig_str:>5}")

    print(f"\nSaved formal_comparison_table.json to {OUTPUT_DIR}/")
    return results


if __name__ == "__main__":
    run_formal_comparison()


