"""
main – FastAPI application for real-time ML inference & drift monitoring.
"""
from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import N_FEATURES, ROOT_DIR, logger
from app.drift import detect_drift, retrain_if_needed
from app.model import load_model, load_kmeans, load_linreg
from app.predict import predict

# ── App instance ─────────────────────────────
app = FastAPI(
    title="ML Monitoring & Drift Detection API",
    version="1.0.0",
    description="Real-time anomaly detection with automated data-drift monitoring.",
)

_start_time = time.time()


# ── Request / Response schemas ───────────────
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"Exactly {N_FEATURES} numeric feature values.",
    )


class PredictResponse(BaseModel):
    prediction: int
    is_anomaly: bool
    anomaly_score: float


class DriftRequest(BaseModel):
    reference_path: Optional[str] = Field(
        None, description="CSV path for reference data (default: data/train.csv)"
    )
    current_path: Optional[str] = Field(
        None, description="CSV path for production data (default: data/production.csv)"
    )


# ── Endpoints ────────────────────────────────
@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness / readiness probe."""
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def prediction_endpoint(req: PredictRequest) -> PredictResponse:
    """Run a single-sample prediction through the Isolation Forest."""
    try:
        result = predict(req.features)
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/cluster", tags=["inference"])
def predict_cluster_endpoint(req: PredictRequest) -> dict:
    """Uses K-Means to identify which environmental cluster this belongs to."""
    try:
        model = load_kmeans()
        sample = [req.features]
        cluster_id = int(model.predict(sample)[0])
        return {
            "cluster_id": cluster_id,
            "message": f"Mediul indicat face parte din Clusterul {cluster_id}"
        }
    except Exception as exc:
        logger.exception("KMeans Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/yield", tags=["inference"])
def predict_yield_endpoint(req: PredictRequest) -> dict:
    """Uses Linear Regression to predict crop yield based on features."""
    try:
        model = load_linreg()
        sample = [req.features]
        # Prevents negative yield predictions
        prediction = max(0.0, float(model.predict(sample)[0]))
        return {
            "predicted_yield_kg_per_m2": round(prediction, 2),
            "message": "Predicție calculată folosind regresie liniară."
        }
    except Exception as exc:
        logger.exception("Yield Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/drift/detect", tags=["monitoring"])
def drift_detect_endpoint(req: DriftRequest) -> dict:
    """Compare reference vs production data and return drift metrics."""
    ref_path = ROOT_DIR / (req.reference_path or "data/train.csv")
    cur_path = ROOT_DIR / (req.current_path or "data/production.csv")

    if not ref_path.exists() or not cur_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing data files: ref={ref_path.exists()}, cur={cur_path.exists()}",
        )

    ref = pd.read_csv(ref_path)
    cur = pd.read_csv(cur_path)
    is_drifted, share, _ = detect_drift(ref, cur)

    return {"drifted": is_drifted, "drift_share": round(share, 4)}


@app.post("/drift/retrain", tags=["monitoring"])
def drift_retrain_endpoint(req: DriftRequest) -> dict:
    """Detect drift and automatically retrain if threshold is exceeded."""
    ref_path = ROOT_DIR / (req.reference_path or "data/train.csv")
    cur_path = ROOT_DIR / (req.current_path or "data/production.csv")

    if not ref_path.exists() or not cur_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing data files: ref={ref_path.exists()}, cur={cur_path.exists()}",
        )

    ref = pd.read_csv(ref_path)
    cur = pd.read_csv(cur_path)
    return retrain_if_needed(ref, cur)
