"""
metrics.py

Computes Jaccard (at multiple k), true full-ranking Spearman, and Kendall's
tau stability scores across valid draws' full 7-feature rankings per
method, and applies the status-field logic (Hard Rule 12).

PHASE 4 CHANGE (Review Point 5/6 fix): the old top5_to_rank_vector fallback
(absent-from-top-5 features sharing a fake tied rank of 6) is REMOVED.
Spearman is now computed on the true, complete rank order of all 7
features - no ties imposed, no fabricated ranks. Kendall's tau is added as
a second full-ranking statistic. Jaccard is now computed at k=3, k=5
(unchanged, still the paper's primary reported granularity), and k=7
(equivalent to full-set agreement, always 1.0 by construction - included
as the ceiling reference point that makes the top-5 floor-effect concern
checkable directly against top-3 and top-7).

Requires full_rank lists (length 7) as produced by
simulation_engine.rank_all_features, not the old top5-only lists.
"""

from itertools import combinations
from typing import Optional

import numpy as np
from scipy.stats import kendalltau, spearmanr

# ---- Named constants (no inline magic numbers) ----
FEATURE_NAMES = ["RIDAGEYR", "BMXBMI", "LBXSGL", "RIAGENDR", "BPQ020",
                  "diabetes_borderline", "diabetes_yes"]
METHODS = ["standardized_beta", "permutation", "shap", "lime"]
MIN_VALID_DRAWS_FOR_STABILITY = 2  # Hard Rule 12
JACCARD_K_VALUES = [3, 5, 7]  # Phase 4: top-5 remains primary; 3 and 7
                                # are the sensitivity/ceiling checks.

STATUS_OK = "OK"
STATUS_INSUFFICIENT_VALID_DRAWS = "INSUFFICIENT_VALID_DRAWS"
STATUS_DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"


def full_rank_to_vector(full_rank: list[str]) -> np.ndarray:
    """
    Convert a complete (length-7) feature ranking into a numeric rank
    vector aligned to FEATURE_NAMES order, for Spearman/Kendall input.

    Phase 4: no fallback rank - every feature has a real, observed rank
    because full_rank now always contains all len(FEATURE_NAMES) features.

    Inputs:
        full_rank (list[str]): all 7 feature names, in descending-
            importance order (as produced by rank_all_features).
    Returns:
        np.ndarray: length-7 rank vector aligned to FEATURE_NAMES order.
    Validity rule checked: none (Phase 4 removes Rule 10's top-5-only
        restriction; this is a straight rank-vector conversion).
    """
    if len(full_rank) != len(FEATURE_NAMES):
        raise ValueError(
            f"expected a full ranking of {len(FEATURE_NAMES)} features, "
            f"got {len(full_rank)} - check that simulation_engine.py's "
            f"rank_all_features (not a truncated top-k list) was used."
        )
    ranks = {feat: i for i, feat in enumerate(full_rank, start=1)}
    return np.array([ranks[name] for name in FEATURE_NAMES])


def jaccard_at_k(full_rank_a: list[str], full_rank_b: list[str], k: int) -> float:
    """
    Compute the Jaccard index between two draws' top-k feature sets,
    derived by slicing their full 7-feature rankings.

    Inputs:
        full_rank_a (list[str]): first draw's full 7-feature ranking.
        full_rank_b (list[str]): second draw's full 7-feature ranking.
        k (int): how many top features to compare (one of JACCARD_K_VALUES).
    Returns:
        float: |intersection| / |union| of the two top-k sets. Always 1.0
            when k == len(FEATURE_NAMES) (both sets are the full feature
            set) - this is expected and is the ceiling reference point,
            not a bug.
    Validity rule checked: none.
    """
    set_a, set_b = set(full_rank_a[:k]), set(full_rank_b[:k])
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_pairwise_stability(full_rank_lists: list[list[str]]) -> Optional[dict]:
    """
    Compute mean pairwise Jaccard (at each k in JACCARD_K_VALUES), mean
    pairwise Spearman, and mean pairwise Kendall's tau across all
    valid-draw full rankings for one method at one rung.

    Inputs:
        full_rank_lists (list[list[str]]): one full (length-7) ranking per
            valid draw.
    Returns:
        dict | None: {jaccard_mean: {k: value, ...}, spearman_mean,
            kendall_tau_mean}, or None if fewer than
            MIN_VALID_DRAWS_FOR_STABILITY lists.
    Validity rule checked: Rule 12 (minimum valid draws for stability).
    """
    if len(full_rank_lists) < MIN_VALID_DRAWS_FOR_STABILITY:
        return None

    jaccard_scores: dict[int, list[float]] = {k: [] for k in JACCARD_K_VALUES}
    spearman_scores = []
    kendall_scores = []

    for a, b in combinations(full_rank_lists, 2):
        for k in JACCARD_K_VALUES:
            jaccard_scores[k].append(jaccard_at_k(a, b, k))

        rank_a = full_rank_to_vector(a)
        rank_b = full_rank_to_vector(b)

        rho, _ = spearmanr(rank_a, rank_b)
        spearman_scores.append(rho if not np.isnan(rho) else 0.0)

        tau, _ = kendalltau(rank_a, rank_b)
        kendall_scores.append(tau if not np.isnan(tau) else 0.0)

    return {
        "jaccard_mean": {k: float(np.mean(v)) for k, v in jaccard_scores.items()},
        "spearman_mean": float(np.mean(spearman_scores)),
        "kendall_tau_mean": float(np.mean(kendall_scores)),
    }


def determine_status(n_valid_draws: int, dependency_available: bool) -> str:
    """
    Determine a method's status at a rung, checking draw-validity count
    first (Hard Rule 12).

    Inputs:
        n_valid_draws (int): number of valid draws at this rung.
        dependency_available (bool): False only if the method's required
            import itself failed.
    Returns:
        str: one of STATUS_INSUFFICIENT_VALID_DRAWS, STATUS_OK,
            STATUS_DEPENDENCY_UNAVAILABLE.
    Validity rule checked: Rule 12 (status-field logic; draw-validity count
        is checked before dependency availability, and
        DEPENDENCY_UNAVAILABLE never overrides a valid-draws-based status).
    """
    if n_valid_draws < MIN_VALID_DRAWS_FOR_STABILITY:
        return STATUS_INSUFFICIENT_VALID_DRAWS
    if not dependency_available:
        return STATUS_DEPENDENCY_UNAVAILABLE
    return STATUS_OK


def compute_stability_table_row(method: str, full_rank_lists: list[list[str]],
                                 dependency_available: bool = True) -> dict:
    """
    Build one row of the final_stability_table for a single method at a
    single rung.

    Inputs:
        method (str): one of METHODS.
        full_rank_lists (list[list[str]]): full 7-feature rankings from
            all valid draws at this rung for this method (Phase 4: no
            longer top-5-only lists).
        dependency_available (bool): whether the method's library import
            succeeded (default True; shap/lime import failures set False).
    Returns:
        dict: {method, n_valid_draws, jaccard_top3_mean, jaccard_top5_mean,
            jaccard_top7_mean, spearman_mean, kendall_tau_mean, status}.
    Validity rule checked: Rule 12 (status-field logic).
    """
    n_valid = len(full_rank_lists)
    status = determine_status(n_valid, dependency_available)

    row = {
        "method": method,
        "n_valid_draws": n_valid,
        "jaccard_top3_mean": None,
        "jaccard_top5_mean": None,
        "jaccard_top7_mean": None,
        "spearman_mean": None,
        "kendall_tau_mean": None,
        "status": status,
    }

    if status == STATUS_OK:
        stability = compute_pairwise_stability(full_rank_lists)
        if stability is not None:
            row["jaccard_top3_mean"] = stability["jaccard_mean"][3]
            row["jaccard_top5_mean"] = stability["jaccard_mean"][5]
            row["jaccard_top7_mean"] = stability["jaccard_mean"][7]
            row["spearman_mean"] = stability["spearman_mean"]
            row["kendall_tau_mean"] = stability["kendall_tau_mean"]

    return row
