"""
predict – inference logic for the Isolation Forest model.
"""
from __future__ import annotations

from typing import List

import numpy as np

from app.config import logger
from app.model import load_model


def predict(features: List[float]) -> dict:
    """
    Run a single sample through the loaded model.

    Returns:
        dict with 'prediction' (1 = normal, -1 = anomaly),
        'is_anomaly' flag, and raw 'anomaly_score'.
    """
    model = load_model()

    sample = np.array(features).reshape(1, -1)
    pred: int = int(model.predict(sample)[0])
    score: float = float(model.decision_function(sample)[0])

    result = {
        "prediction": pred,
        "is_anomaly": pred == -1,
        "anomaly_score": round(score, 6),
    }

    logger.info("Prediction request | features=%s | result=%s", features, result)
    return result
