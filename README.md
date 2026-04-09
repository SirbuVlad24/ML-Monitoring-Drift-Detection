# Real-Time ML Monitoring and Drift Detection System

This repository contains a production-ready template for monitoring an Isolation Forest model, automatically detecting data drift with Evidently AI, and dynamically retraining the model if drift thresholds are exceeded.

## Structure

* `app/`
    * `main.py` - FastAPI app and REST endpoints.
    * `model.py` - Persistence helpers and model caching logic.
    * `predict.py` - Inference logic for the ML model.
    * `drift.py` - Data drift detection using Evidently AI.
    * `config.py` - Configuration loading and validation.
* `scripts/`
    * `train.py` - Generates synthetic reference data and trains the original model.
    * `simulate_drift.py` - Generates synthetic production data exhibiting distribution drift.
* `config.yaml` - Base system configuration (model settings, thresholds, etc).
* `.env` - Environment variable overrides (loaded by `app/config.py`).
* `Dockerfile` - Containerisation definition.

## Quickstart

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Train the baseline model (creates `data/train.csv` and `models/isolation_forest.joblib`):
    ```bash
    python -m scripts.train
    ```

3. Run the FastAPI application locally:
    ```bash
    uvicorn app.main:app --reload
    ```

## Endpoints

* `GET /health` - Liveness probe.
* `POST /predict` - Send an inference request `{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}`.
* `POST /drift/detect` - Check whether production data has drifted.
* `POST /drift/retrain` - Evaluate drift and retrain the model on the fly. 

## Simulation

You can simulate data drift by running:
```bash
python -m scripts.simulate_drift
```
Then use `/drift/detect` to see Evidently AI detect the shift!
