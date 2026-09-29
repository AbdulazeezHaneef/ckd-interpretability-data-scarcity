"""
simulation_engine.py

Implements the single-draw pipeline: sample without replacement -> EPV check
-> within-draw standardization -> near-unregularized logistic fit ->
convergence check -> two-part coefficient magnitude check -> (if the model
is valid) independent, per-method computation of four feature-importance
methods, each wrapped so one method's computational failure never blocks
the other three.

PHASE 2 CHANGE (Review Point 1/2 fix): model/draw validity (Rules 1-3) and
interpretability-method computational validity are two distinct checks.
Rules 1-3 gate whether a model exists at all. Given a valid model, each of
the four methods is computed independently inside _safe_compute_method, so
a method can fail computationally (NaN/inf output, raised exception)
without being conflated with model-fit failure and without blocking the
other three methods' results for that draw.

PHASE 4 CHANGE (Review Point 5/6 fix): each method's extraction step now
captures the FULL 7-feature ranking (rank_all_features), not just the
top-5. This removes the need for the old fallback-rank-6 Spearman hack -
true Spearman and Kendall's tau can now be computed on the complete
ranking - and lets top-3/top-5/top-7 Jaccard all be derived from the same
captured list, so the top-5-alone floor-effect concern can be checked
directly against top-3 and top-7 at analysis time.

PERMUTATION SCORING FIX (post-Phase-9 audit): permutation_importance was
previously called without an explicit scoring parameter, silently
defaulting to sklearn's accuracy-based scoring. Under this study's 16.7%
class prevalence, accuracy is a weak choice - a model predicting the
majority class unconditionally scores ~83% accuracy for free, so an
accuracy-drop-based importance measure can understate a feature's real
predictive value. Switched to scoring="roc_auc" for consistency with
this study's own stated performance metric (Phase 9's AUC). This change
affects ONLY the "permutation" method's computed values - Rules 1-3,
SHAP, LIME, and standardized-beta never call permutation_importance and
are unaffected. Requires a full resimulation and re-verification of
every downstream table/figure that reads permutation's full_rank values.
"""

import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---- Named constants (no inline magic numbers) ----
CONTINUOUS_PREDICTORS = ["RIDAGEYR", "BMXBMI", "LBXSGL"]
BINARY_PREDICTORS = ["RIAGENDR", "BPQ020", "diabetes_borderline", "diabetes_yes"]
FEATURE_NAMES = CONTINUOUS_PREDICTORS + BINARY_PREDICTORS
OUTCOME_COL = "has_CKD"

EPV_FLOOR = 14  # Hard Rule 3: 7 predictors x 2
C_REG = 1e6  # Hard Rule: near-unregularized fit
MAX_ITER = 1000

# Hard Rule 5: two-part coefficient magnitude check
COEF_ABS_BOUND = 10.0
COEF_RATIO_MIN_BETA = 0.1
COEF_RATIO_SE_MULTIPLIER = 5.0
COEF_NEAR_NULL_SE_BOUND = 3.0

TOP_K = 5  # Hard Rule 10 (original default). Retained as the paper's
# primary reported granularity; no longer used to truncate extraction
# (Phase 4: rank_all_features captures the full ranking) - only used at
# analysis time in metrics.py to slice top-3/top-5/top-7 Jaccard.
LIME_SUBSAMPLE_SIZE = 50  # Hard Rule 11
LIME_NUM_FEATURES = len(FEATURE_NAMES)

METHODS = ["standardized_beta", "permutation", "shap", "lime"]

# ---- Phase 2: method-specific computational-validity statuses ----
# Distinct from model/draw validity (Rules 1-3). Assessed only for draws
# where valid=True, i.e. a model was already successfully fit.
METHOD_COMPUTATION_OK = "COMPUTATION_OK"
METHOD_COMPUTATION_FAILED = "COMPUTATION_FAILED"


def check_epv(sample: pd.DataFrame) -> bool:
    """
    Check the events-per-variable floor (Hard Rule 3).

    Inputs:
        sample (pd.DataFrame): the drawn sample, containing OUTCOME_COL.
    Returns:
        bool: True if the draw has at least EPV_FLOOR positive cases.
    Validity rule checked: Rule 3 (EPV floor = 14).
    """
    return int(sample[OUTCOME_COL].sum()) >= EPV_FLOOR


def standardize_continuous(X: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score the continuous predictors within this draw only. Binary
    predictors are left untouched (Hard Rule 6).

    Inputs:
        X (pd.DataFrame): predictor matrix for one draw, all 7 predictors.
    Returns:
        pd.DataFrame: copy of X with continuous columns standardized.
    Validity rule checked: none (data transform, not a rejection rule).
    """
    X = X.copy()
    scaler = StandardScaler()
    X[CONTINUOUS_PREDICTORS] = scaler.fit_transform(X[CONTINUOUS_PREDICTORS])
    return X


def fit_and_check_convergence(X_arr: np.ndarray, y: np.ndarray) -> tuple[Optional[LogisticRegression], bool]:
    """
    Fit a near-unregularized logistic regression and check for optimizer
    convergence (Hard Rule 4).

    Inputs:
        X_arr (np.ndarray): standardized-within-draw predictor matrix.
        y (np.ndarray): binary outcome vector.
    Returns:
        tuple[LogisticRegression | None, bool]: fitted model (or None if it
            errored) and a bool indicating whether it converged cleanly.
    Validity rule checked: Rule 4 (convergence check).
    """
    model = LogisticRegression(C=C_REG, max_iter=MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_arr, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def compute_wald_se(model: LogisticRegression, X_arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute Wald standard errors for the 7 predictor coefficients from the
    observed Fisher information matrix, manually (no statsmodels).

    Inputs:
        model (LogisticRegression): fitted model.
        X_arr (np.ndarray): standardized-within-draw predictor matrix.
    Returns:
        np.ndarray | None: SE for each of the 7 predictor coefficients
            (intercept excluded), or None if the Fisher information matrix
            is singular.
    Validity rule checked: feeds Rule 5, not itself a rejection rule.
    """
    n = X_arr.shape[0]
    design = np.hstack([np.ones((n, 1)), X_arr])
    p = model.predict_proba(X_arr)[:, 1]
    weights = p * (1.0 - p)
    fisher_info = design.T @ (design * weights[:, None])
    try:
        cov = np.linalg.inv(fisher_info)
    except np.linalg.LinAlgError:
        return None
    se_full = np.sqrt(np.diag(cov))
    return se_full[1:]  # drop intercept SE


def check_coefficients(beta: np.ndarray, se: np.ndarray) -> list[dict]:
    """
    Apply the corrected two-part coefficient magnitude check (Hard Rule 5)
    per feature.

    Inputs:
        beta (np.ndarray): standardized coefficients, length 7.
        se (np.ndarray): Wald SEs for those coefficients, length 7.
    Returns:
        list[dict]: one entry per feature that fails the check, each with
            feature name, beta, se, and which sub-condition triggered.
    Validity rule checked: Rule 5 (two-part coefficient magnitude check).
    """
    failures = []
    for name, b, s in zip(FEATURE_NAMES, beta, se):
        ab = abs(b)
        if ab > COEF_ABS_BOUND:
            failures.append({"feature": name, "beta": float(b), "se": float(s), "condition": "abs_bound"})
        elif ab >= COEF_RATIO_MIN_BETA and s > COEF_RATIO_SE_MULTIPLIER * ab:
            failures.append({"feature": name, "beta": float(b), "se": float(s), "condition": "ratio_bound"})
        elif ab < COEF_RATIO_MIN_BETA and s > COEF_NEAR_NULL_SE_BOUND:
            failures.append({"feature": name, "beta": float(b), "se": float(s), "condition": "near_null_se"})
    return failures


def compute_lime_importance(X_arr: np.ndarray, y: np.ndarray, model: LogisticRegression, seed: int,
                             subsample_size: Optional[int] = None) -> dict:
    """
    Compute averaged LIME feature importance over a seeded subsample of
    the draw.

    Inputs:
        X_arr (np.ndarray): standardized-within-draw predictor matrix.
        y (np.ndarray): binary outcome vector (unused directly, kept for
            interface symmetry with other importance extractors).
        model (LogisticRegression): fitted model.
        seed (int): explicit seed for the subsample and the LIME explainer.
        subsample_size (int | None): number of instances to explain.
            Defaults to LIME_SUBSAMPLE_SIZE (Hard Rule 11, the original
            fixed-50 design) if not given - the main pipeline's calls are
            unaffected by this parameter's addition. Phase 5's
            lime_sensitivity.py passes a rung-dependent value instead.
    Returns:
        dict: feature name -> mean absolute LIME weight across the
            subsample.
    Validity rule checked: none (extraction only; draw's model already
        passed Rules 1-3 by the time this is called). May raise on
        computational failure - caught by _safe_compute_method.
    """
    if subsample_size is None:
        subsample_size = LIME_SUBSAMPLE_SIZE
    n = X_arr.shape[0]
    rng = np.random.RandomState(seed)
    subsample_size = min(subsample_size, n)
    idx = rng.choice(n, size=subsample_size, replace=False)

    explainer = LimeTabularExplainer(
        training_data=X_arr,
        feature_names=FEATURE_NAMES,
        class_names=["no_CKD", "CKD"],
        discretize_continuous=False,
        random_state=seed,
        mode="classification",
    )

    abs_weight_sums = {name: 0.0 for name in FEATURE_NAMES}
    for i in idx:
        exp = explainer.explain_instance(
            X_arr[i],
            model.predict_proba,
            num_features=LIME_NUM_FEATURES,
        )
        for feat_desc, weight in exp.as_list():
            for name in FEATURE_NAMES:
                if name in feat_desc:
                    abs_weight_sums[name] += abs(weight)
                    break

    return {name: total / subsample_size for name, total in abs_weight_sums.items()}


def compute_shap_importance(X_arr: np.ndarray, model: LogisticRegression) -> dict:
    """
    Compute mean absolute SHAP importance using the full draw as the
    background/masker set (Hard Rule 11 - no downsampling below full draw).

    Inputs:
        X_arr (np.ndarray): standardized-within-draw predictor matrix.
        model (LogisticRegression): fitted model.
    Returns:
        dict: feature name -> mean absolute SHAP value across the draw.
    Validity rule checked: none (extraction only). May raise on
        computational failure - caught by _safe_compute_method.
    """
    # max_samples overridden to the full draw size so SHAP's masker never
    # silently downsamples below the full draw (Hard Rule 11).
    masker = shap.maskers.Independent(X_arr, max_samples=X_arr.shape[0])
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer.shap_values(X_arr)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(zip(FEATURE_NAMES, mean_abs))


def rank_all_features(importance: dict) -> list[str]:
    """
    Return ALL feature names ranked by importance value, descending
    (Phase 4 change: replaces top_k_features as the extraction step).

    The full ranking is captured once here; top-3/top-5/top-7 Jaccard and
    full-ranking Spearman/Kendall's tau are all derivable from this single
    list later (in metrics.py), so nothing about the ranking needs to be
    recomputed at analysis time - only sliced or correlated.

    Inputs:
        importance (dict): feature name -> importance value.
    Returns:
        list[str]: all len(FEATURE_NAMES) feature names, highest
            importance first.
    Validity rule checked: none (Rule 10's "top-5 only" restriction is
        superseded by Phase 4 - the full ranking is now always captured).
    """
    ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked]


def _safe_compute_method(compute_importance_fn: Callable[[], dict]) -> dict:
    """
    Run one interpretability method's importance computation and catch any
    failure specific to that method, so one method's computational failure
    never blocks the other three and is never conflated with model/draw
    validity (Phase 2 / Review Point 1 fix).

    A result is treated as a computational failure if the function raises,
    or if it returns any NaN/inf importance value (a degenerate output
    that would otherwise silently corrupt the ranking).

    Inputs:
        compute_importance_fn (Callable[[], dict]): zero-arg function that
            computes and returns one method's raw feature -> importance
            dict for an already-model-valid draw.
    Returns:
        dict: {status, full_rank, error_detail}.
            status: METHOD_COMPUTATION_OK or METHOD_COMPUTATION_FAILED.
            full_rank: list[str] of length len(FEATURE_NAMES), all
                features ranked descending, if OK; else None. (Phase 4:
                replaces the old top5-only field - top-k slices and full
                rank-correlation statistics are both derivable from this.)
            error_detail: str description if failed, else None.
    Validity rule checked: none (this is the new method-specific
        computational-failure check, distinct from Rules 1-3).
    """
    try:
        importance = compute_importance_fn()
        values = np.array(list(importance.values()), dtype=float)
        if values.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} importance values, got {values.shape[0]}")
        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
            raise ValueError("degenerate (NaN/inf) importance values")
        full_rank = rank_all_features(importance)
        return {"status": METHOD_COMPUTATION_OK, "full_rank": full_rank, "error_detail": None}
    except Exception as e:  # noqa: BLE001 - deliberately broad: any method-
        # library failure (SHAP/LIME/sklearn internals) must be caught here
        # and attributed to that specific method, not allowed to crash the
        # whole draw or silently propagate as a model-validity failure.
        return {"status": METHOD_COMPUTATION_FAILED, "full_rank": None, "error_detail": str(e)}


def compute_all_methods(X_arr: np.ndarray, y: np.ndarray, model: LogisticRegression,
                         beta_importance: dict, seed: int) -> dict[str, dict]:
    """
    Independently compute all four methods' importance for an
    already-model-valid draw, each wrapped in _safe_compute_method so
    failures are per-method, not draw-wide (Phase 2 change).

    Inputs:
        X_arr (np.ndarray): standardized-within-draw predictor matrix.
        y (np.ndarray): binary outcome vector.
        model (LogisticRegression): fitted, already-validated model.
        beta_importance (dict): |standardized coefficient| per feature,
            already computed and validated by Rules 1-3.
        seed (int): draw-level seed, reused directly by permutation
            importance and LIME (Hard Rule: no independent seeds).
    Returns:
        dict[str, dict]: method name -> {status, full_rank, error_detail},
            one entry per METHODS.
    Validity rule checked: none (orchestrates the new per-method
        computational-failure check).
    """
    return {
        "standardized_beta": _safe_compute_method(lambda: beta_importance),
        "permutation": _safe_compute_method(
            lambda: dict(zip(FEATURE_NAMES,
                              permutation_importance(model, X_arr, y, n_repeats=10,
                                                      scoring="roc_auc",
                                                      random_state=seed).importances_mean))
        ),
        "shap": _safe_compute_method(lambda: compute_shap_importance(X_arr, model)),
        "lime": _safe_compute_method(lambda: compute_lime_importance(X_arr, y, model, seed)),
    }


def draw_sample_and_fit(df: pd.DataFrame, N: int, seed: int) -> dict:
    """
    Perform one full draw: sample without replacement, apply the three
    model/draw validity checks in order (EPV, convergence, coefficient
    magnitude), and if the model is valid, independently compute all four
    importance methods (each may individually succeed or fail).

    Inputs:
        df (pd.DataFrame): full source pool (7 predictors + has_CKD).
        N (int): draw sample size (one of the 7 locked rungs).
        seed (int): explicit, logged, incrementing random seed for this draw.
    Returns:
        dict: result record with keys:
            seed, N,
            valid (bool) - MODEL/DRAW validity per Rules 1-3 (Phase 2:
                renamed in meaning, not mechanics - this has never been
                method-specific and is now documented as such),
            fail_rule (str | None; one of 'rule_1_epv', 'rule_2_convergence',
                'rule_3_coefficient'),
            fail_detail (list[dict] | str | None),
            n_positive (int) - Phase 3: number of positive (has_CKD=1)
                cases in this draw, recorded for EVERY draw regardless of
                validity outcome, so EPV sensitivity analysis at
                thresholds other than EPV_FLOOR can be computed later by
                re-filtering existing results, without re-fitting models
                or re-running SHAP/LIME.
            method_results (dict[str, dict] | None) - present only if
                valid=True. Keys 'standardized_beta', 'permutation',
                'shap', 'lime', each {status, full_rank, error_detail}
                (Phase 4: full_rank replaces the old top5-only field).
    Validity rules checked: Rule 2 (sampling without replacement),
        Rule 3 (EPV floor), Rule 4 (convergence), Rule 5 (coefficient check).
    """
    sample = df.sample(n=N, replace=False, random_state=seed)  # Rule 2
    n_positive = int(sample[OUTCOME_COL].sum())  # Phase 3: recorded for every
    # draw (not just Rule 1 failures) so EPV sensitivity at other thresholds
    # (e.g. EPV=5, EPV=10) can be computed later by re-filtering this value,
    # without re-fitting models or re-running SHAP/LIME.

    if not check_epv(sample):
        return {
            "seed": seed, "N": N, "valid": False,
            "fail_rule": "rule_1_epv",
            "fail_detail": {"n_positive": n_positive, "epv_floor": EPV_FLOOR},
            "n_positive": n_positive,
            "method_results": None,
        }

    X = standardize_continuous(sample[FEATURE_NAMES])
    y = sample[OUTCOME_COL].values
    X_arr = X.values

    model, converged = fit_and_check_convergence(X_arr, y)
    if not converged:
        return {
            "seed": seed, "N": N, "valid": False,
            "fail_rule": "rule_2_convergence",
            "fail_detail": "optimizer failed to converge",
            "n_positive": n_positive,
            "method_results": None,
        }

    se = compute_wald_se(model, X_arr)
    if se is None:
        return {
            "seed": seed, "N": N, "valid": False,
            "fail_rule": "rule_2_convergence",
            "fail_detail": "singular Fisher information matrix",
            "n_positive": n_positive,
            "method_results": None,
        }

    beta = model.coef_[0]
    coef_failures = check_coefficients(beta, se)
    if coef_failures:
        return {
            "seed": seed, "N": N, "valid": False,
            "fail_rule": "rule_3_coefficient",
            "fail_detail": coef_failures,
            "n_positive": n_positive,
            "method_results": None,
        }

    beta_importance = dict(zip(FEATURE_NAMES, np.abs(beta)))
    method_results = compute_all_methods(X_arr, y, model, beta_importance, seed)

    return {
        "seed": seed, "N": N, "valid": True,
        "fail_rule": None, "fail_detail": None,
        "n_positive": n_positive,
        "method_results": method_results,
    }


if __name__ == "__main__":
    # Manual single-call test (Workflow step 2) before looping.
    processed = pd.read_csv("outputs/processed_predictors.csv")
    result = draw_sample_and_fit(processed, N=500, seed=0)
    print(result)
