"""
run_all_rungs.py

Runs the full simulation across all 7 rungs, K=100 draws each, and saves
per-rung raw JSON results to outputs/. Never redraws to replace a failure
and never tops up to reach 100 valid draws (Hard Rule 7).
"""

import json

import pandas as pd

from simulation_engine import draw_sample_and_fit

# ---- Named constants (no inline magic numbers) ----
PROCESSED_FILE = "outputs/processed_predictors.csv"
OUTPUT_DIR = "outputs"

RUNGS = [5000, 2500, 1000, 500, 250, 100, 50]  # Hard Rule 9: fixed, in order
K = 100  # Hard Rule 7 & 8: fixed denominator, constant across all rungs


def run_rung(df: pd.DataFrame, N: int, seed_counter_start: int) -> tuple[list[dict], int]:
    """
    Run exactly K draws for one rung, logging every outcome.

    Inputs:
        df (pd.DataFrame): full processed source pool.
        N (int): sample size for this rung.
        seed_counter_start (int): first seed to use for this rung's draws;
            seeds increment by 1 for each successive draw (global counter).
    Returns:
        tuple[list[dict], int]: (list of K per-draw result records, next
            seed counter value to continue from for the following rung).
    Validity rule checked: Rule 7 (exactly K=100 draws, no redraws, no
        top-ups), Rule 8 (K constant across rungs).
    """
    records = []
    seed = seed_counter_start
    for _ in range(K):
        result = draw_sample_and_fit(df, N=N, seed=seed)
        records.append(result)
        seed += 1
    return records, seed


def main() -> None:
    """
    Run all 7 rungs and save one JSON results file per rung to outputs/.

    Inputs: none.
    Returns: None. Writes outputs/rung_results_N{N}.json per rung as a
        side effect.
    Validity rule checked: Rule 9 (fixed rung list, in order).
    """
    df = pd.read_csv(PROCESSED_FILE)

    seed_counter = 0
    for N in RUNGS:
        print(
            f"Running rung N={N} (seeds {seed_counter}-{seed_counter + K - 1}) ...")
        records, seed_counter = run_rung(df, N, seed_counter)

        n_success = sum(1 for r in records if r["valid"])
        n_failed = K - n_success
        print(f"  N={N}: {n_success} valid, {n_failed} failed (of {K})")

        out_path = f"{OUTPUT_DIR}/rung_results_N{N}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  saved to {out_path}")

    print("All 7 rungs complete.")


if __name__ == "__main__":
    main()
