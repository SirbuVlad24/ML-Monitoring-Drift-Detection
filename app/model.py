"""
model – persistence helpers for the Isolation Forest model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
from sklearn.ensemble import IsolationForest

from app.config import (
    MODEL_CONTAMINATION,
    MODEL_N_ESTIMATORS,
    MODEL_PATH,
    MODEL_RANDOM_STATE,
    ROOT_DIR,
    logger,
)

_cached_model: Optional[IsolationForest] = None


def get_model_path() -> Path:
    return ROOT_DIR / MODEL_PATH


def build_model() -> IsolationForest:
    """Create a fresh, untrained Isolation Forest with config params."""
    return IsolationForest(
        n_estimators=MODEL_N_ESTIMATORS,
        contamination=MODEL_CONTAMINATION,
        random_state=MODEL_RANDOM_STATE,
    )


def save_model(model: IsolationForest, path: Optional[Path] = None) -> Path:
    """Serialise model to disk via joblib."""
    path = path or get_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(path: Optional[Path] = None) -> IsolationForest:
    """Load model from disk (with in-memory cache to avoid repeated I/O)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    path = path or get_model_path()
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Run training first.")

    _cached_model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return _cached_model


_cached_kmeans = None
_cached_linreg = None
_cached_classifier = None

def get_kmeans_path() -> Path:
    return ROOT_DIR / "models/kmeans.joblib"

def get_linreg_path() -> Path:
    return ROOT_DIR / "models/linreg.joblib"

def get_classifier_path() -> Path:
    return ROOT_DIR / "models/classifier.joblib"

def load_kmeans():
    global _cached_kmeans
    if _cached_kmeans is not None:
        return _cached_kmeans
    
    path = get_kmeans_path()
    if not path.exists():
        raise FileNotFoundError("Nu există modelul KMeans. Rulează python -m scripts.train_more_models")
    
    _cached_kmeans = joblib.load(path)
    return _cached_kmeans

def load_linreg():
    global _cached_linreg
    if _cached_linreg is not None:
        return _cached_linreg
    
    path = get_linreg_path()
    if not path.exists():
        raise FileNotFoundError("Nu există modelul LinReg. Rulează python -m scripts.train_more_models")
    
    _cached_linreg = joblib.load(path)
    return _cached_linreg

def load_classifier():
    global _cached_classifier
    if _cached_classifier is not None:
        return _cached_classifier
    
    path = get_classifier_path()
    if not path.exists():
        raise FileNotFoundError("Nu exista modelul Classifier. Ruleaza python -m scripts.train_more_models")
    
    _cached_classifier = joblib.load(path)
    return _cached_classifier

def clear_cache() -> None:
    """Invalidate the in-memory model cache (called after retraining)."""
    global _cached_model, _cached_kmeans, _cached_linreg, _cached_classifier
    _cached_model = None
    _cached_kmeans = None
    _cached_linreg = None
    _cached_classifier = None
    logger.info("All model caches cleared")
