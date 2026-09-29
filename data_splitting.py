"""
data_splitting.py

Shared stratified 80/20 train/test split logic, locked in the Phase 1
performance-metrics design discussion and first used here (Phase 5, item
11: permutation importance held-out sensitivity). Reused unchanged by
Phase 9 (AUC/Brier held-out performance metrics) so both consumers share
one split implementation, one validity gate, and one standardization
convention - no risk of the two diverging.

Locked design (Phase 1 discussion):
  - Stratified 80/20 split (preserves outcome prevalence in both halves).
  - EPV is checked on the TRAIN split only (train positives >= EPV_FLOOR),
    not the full draw - this is what the model actually learns from.
  - TEST split must have >= TEST_MIN_POSITIVES (5) positive cases -
    protects against a technically-nonzero but statistically noisy test
    set at the low-N rungs.
  - A draw failing EITHER condition is flagged SPLIT_INVALID for
    performance-evaluability purposes. This is an INDEPENDENT validity
    track: a draw that fails this split check can still count fully
    toward interpretability stability (Rules 1-3 are unaffected) and
    toward the original in-sample permutation importance already stored
    in rung_results_N{N}.json.
  - Standardization is fit on the TRAIN split only and applied to both
    train and test, to avoid leaking test-set distribution into the
    scaler (Phase 1 "standardization leakage" decision).
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from simulation_engine import CONTINUOUS_PREDICTORS, FEATURE_NAMES, OUTCOME_COL

TRAIN_EPV_FLOOR = 14  # 7 predictors x 2, same multiplier as Rule 1's EPV=2 baseline
TEST_MIN_POSITIVES = 5
TEST_SIZE = 0.2

SPLIT_VALID = "SPLIT_VALID"
SPLIT_INVALID_TRAIN_EPV = "SPLIT_INVALID_TRAIN_EPV"
SPLIT_INVALID_TEST_POSITIVES = "SPLIT_INVALID_TEST_POSITIVES"


def stratified_split(sample: pd.DataFrame, seed: int) -> dict:
    """
    Perform one stratified 80/20 split of a draw, check both validity
    conditions, and standardize (fit on train only) if valid.

    Inputs:
        sample (pd.DataFrame): the drawn sample (7 predictors + has_CKD),
            NOT yet standardized - standardization happens here, after
            the split, to avoid leakage.
        seed (int): the draw's original seed, reused for the split so the
            split is deterministic and reproducible.
    Returns:
        dict: {status, X_train, y_train, X_test, y_test, n_train_positive,
            n_test_positive}. status is one of SPLIT_VALID,
            SPLIT_INVALID_TRAIN_EPV, SPLIT_INVALID_TEST_POSITIVES. If not
            SPLIT_VALID, the X/y fields are None.
    Validity rule checked: the new performance-evaluability validity track
        (independent of Rules 1-3 and of interpretability stability).
    """
    X = sample[FEATURE_NAMES]
    y = sample[OUTCOME_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    n_train_positive = int(y_train.sum())
    n_test_positive = int(y_test.sum())

    if n_train_positive < TRAIN_EPV_FLOOR:
        return {
            "status": SPLIT_INVALID_TRAIN_EPV,
            "n_train_positive": n_train_positive, "n_test_positive": n_test_positive,
            "X_train": None, "y_train": None, "X_test": None, "y_test": None,
        }
    if n_test_positive < TEST_MIN_POSITIVES:
        return {
            "status": SPLIT_INVALID_TEST_POSITIVES,
            "n_train_positive": n_train_positive, "n_test_positive": n_test_positive,
            "X_train": None, "y_train": None, "X_test": None, "y_test": None,
        }

    # Standardize continuous predictors: fit on train only (Phase 1
    # leakage decision), apply the same fitted scaler to test.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[CONTINUOUS_PREDICTORS] = scaler.fit_transform(X_train[CONTINUOUS_PREDICTORS])
    X_test[CONTINUOUS_PREDICTORS] = scaler.transform(X_test[CONTINUOUS_PREDICTORS])

    return {
        "status": SPLIT_VALID,
        "n_train_positive": n_train_positive, "n_test_positive": n_test_positive,
        "X_train": X_train.values, "y_train": y_train,
        "X_test": X_test.values, "y_test": y_test,
    }
