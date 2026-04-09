"""
simulate_drift.py – Prepare a "production" dataset with shifted distributions from real greenhouse data.

The script alters some of the features artificially on the hold-out dataset
(e.g., simulating a sudden heatwave with high temps and lower humidity).

Usage:
    python -m scripts.simulate_drift
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (
    FEATURE_NAMES,
    N_SAMPLES_PROD,
    N_SAMPLES_TRAIN,
    RAW_DATASET_PATH,
    logger,
)


def generate_drifted_data() -> pd.DataFrame:
    """
    Read the hold-out dataset from the CSV, then simulate drift on temperature and humidity.
    """
    raw_path = ROOT / RAW_DATASET_PATH
    df = pd.read_csv(raw_path)
    df_clean = df[FEATURE_NAMES].dropna()
    
    # Take rows starting AFTER the training set to ensure no overlapping data
    start_idx = N_SAMPLES_TRAIN
    end_idx = start_idx + N_SAMPLES_PROD
    prod_df = df_clean.iloc[start_idx:end_idx].copy()
    
    if len(prod_df) == 0:
        logger.error("Not enough data left for production! Reduce N_SAMPLES_TRAIN.")
        sys.exit(1)

    # ────────────────────────────────────────────────────────
    # SIMULATE A CALAMITY IN THE GREENHOUSE (Hot & dry spell)
    # ────────────────────────────────────────────────────────
    if "avg_temperature_C" in prod_df.columns:
        prod_df["avg_temperature_C"] += 8.5   # massive heatwave
        
    if "humidity_percent" in prod_df.columns:
        prod_df["humidity_percent"] *= 0.6    # drop humidity by 40%
        
    if "soil_pH" in prod_df.columns:
        prod_df["soil_pH"] -= 1.0             # soil becomes significantly acidic

    return prod_df


def main() -> None:
    logger.info("=== Drift simulation started ===")

    df = generate_drifted_data()
    out_path = ROOT / "data" / "production.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Production (drifted) data saved → %s  (%d rows × %d cols)", out_path, *df.shape)

    logger.info("=== Drift simulation finished ===")


if __name__ == "__main__":
    main()
