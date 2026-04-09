"""
train.py – Read real data, prepare training subset, fit Isolation Forest, and save artefacts.

Usage:
    python -m scripts.train
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so `app.*` imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (
    FEATURE_NAMES,
    N_SAMPLES_TRAIN,
    RAW_DATASET_PATH,
    logger,
)
from app.model import build_model, save_model


def prepare_training_data() -> pd.DataFrame:
    """
    Read the real dataset, drop missing values, select required features,
    and return the train subset.
    """
    raw_path = ROOT / RAW_DATASET_PATH
    if not raw_path.exists():
        logger.error(f"Missing raw dataset at {raw_path}. Download from Kaggle!")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    
    # Use only requested features and drop missing values
    df_clean = df[FEATURE_NAMES].dropna()
    
    if len(df_clean) < N_SAMPLES_TRAIN:
        logger.warning(f"Not enough samples! Requested {N_SAMPLES_TRAIN} but found {len(df_clean)}")
        n_take = len(df_clean)
    else:
        n_take = N_SAMPLES_TRAIN

    # Take the first n_take samples for training
    return df_clean.head(n_take)


def main() -> None:
    logger.info("=== Training pipeline started ===")

    # 1. Prepare + persist training data
    df = prepare_training_data()
    data_dir = ROOT / "data"
    train_csv = data_dir / "train.csv"
    df.to_csv(train_csv, index=False)
    logger.info("Training data saved → %s  (%d rows × %d cols)", train_csv, *df.shape)

    # 2. Build & fit model
    model = build_model()
    model.fit(df.values)
    logger.info("Model training complete")

    # 3. Save model
    save_model(model)
    logger.info("=== Training pipeline finished ===")


if __name__ == "__main__":
    main()
