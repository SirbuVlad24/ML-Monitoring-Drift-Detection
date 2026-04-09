# Greenhouse ML Monitoring and Drift Detection System

![Main Workflow Status](https://github.com/SirbuVlad24/ML-Monitoring-Drift-Detection/actions/workflows/ci.yml/badge.svg?branch=main)

## Project Overview
This repository implements a real-time Machine Learning monitoring system designed for greenhouse environmental data. The project provides an end-to-end MLOps workflow, including data ingestion, automated model training, real-time inference via REST API, and data drift detection using Evidently AI.

The system is specifically designed to adapt to environmental shifts (seasonal changes, sensor failures) by monitoring the distribution of input features and triggering automated retraining pipelines when a significant drift is detected.

## Core Features
- **Inference Service**: REST API endpoints for anomaly detection (Isolation Forest), crop yield prediction (Linear Regression), environmental clustering (K-Means), and crop classification (Random Forest).
- **Drift Monitoring**: Automated detection of statistical shifts in features such as temperature, humidity, CO2 levels, and light intensity.
- **Automated Retraining**: A feedback loop that updates the reference baseline and retrains the model binaries when data drift exceeds a configurable threshold (default 0.5 drift share).
- **Modern UI**: A responsive web interface for interacting with the models and monitoring operational health.

## Technical Stack
- **Languages**: Python 3.11+
- **Frameworks**: FastAPI, Pydantic, Uvicorn
- **Machine Learning**: Scikit-Learn (Random Forest, Isolation Forest, K-Means, Linear Regression)
- **Monitoring**: Evidently AI (Data Drift Presets)
- **DevOps**: Docker, GitHub Actions (CI/CD), GitHub Container Registry (GHCR)
- **Data Handling**: Pandas, NumPy, Joblib

## Installation and Local Setup

### 1. Environment Preparation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model Training and Data Preparation
Execute the training scripts to generate model artifacts and baseline datasets:
```bash
python -m scripts.train
python -m scripts.train_more_models
```

### 3. Running the Application
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
The application will be available at `http://127.0.0.1:8000`. API documentation is accessible via `/docs`.

## CI/CD Pipeline
The project utilizes GitHub Actions for continuous integration and delivery. On every push to the `main` branch, the pipeline performs the following steps:
1. **Environment Initialization**: Sets up a Python environment and installs dependencies.
2. **Automated Training**: Executes training scripts to verify data and model code.
3. **Automated Testing**: Runs the Pytest suite to validate API endpoints.
4. **Docker Image Build**: Packages the application into a production-ready Docker container.
5. **Image Registry Push**: Pushes the resulting image to the GitHub Container Registry (GHCR).

## Project Structure
- `app/`: Core FastAPI application logic and model persistence layers.
- `scripts/`: Training, drift simulation, and model management scripts.
- `data/`: Directory for raw datasets and generated reference/production artifacts.
- `models/`: Persistent storage for serialized model binaries (.joblib).
- `static/`: Frontend assets (HTML, CSS, JavaScript).
- `tests/`: API validation and unit tests.
- `Dockerfile`: Containerization configuration using a lightweight Python image.
