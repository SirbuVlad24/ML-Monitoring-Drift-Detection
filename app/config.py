"""
config – loads settings from config.yaml with .env overrides.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


_cfg = _load_yaml(ROOT_DIR / "config.yaml")

# ── Model settings ───────────────────────────
MODEL_PATH: str = os.getenv("MODEL_PATH", _cfg["model"]["path"])
MODEL_CONTAMINATION: float = float(_cfg["model"]["contamination"])
MODEL_N_ESTIMATORS: int = int(_cfg["model"]["n_estimators"])
MODEL_RANDOM_STATE: int = int(_cfg["model"]["random_state"])

# ── Data settings ────────────────────────────
N_SAMPLES_TRAIN: int = int(_cfg["data"]["n_samples_train"])
N_SAMPLES_PROD: int = int(_cfg["data"]["n_samples_prod"])
RAW_DATASET_PATH: str = _cfg["data"]["raw_dataset_path"]
FEATURE_NAMES: list[str] = _cfg["data"]["feature_names"]
N_FEATURES: int = len(FEATURE_NAMES)
DATA_RANDOM_STATE: int = int(_cfg["data"]["random_state"])

# ── Drift settings ──────────────────────────
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", _cfg["drift"]["threshold"]))
DRIFT_REPORT_PATH: str = _cfg["drift"]["report_path"]

# ── Logging ──────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", _cfg["logging"]["level"])
LOG_FORMAT: str = _cfg["logging"]["format"]

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("ml_monitor")
