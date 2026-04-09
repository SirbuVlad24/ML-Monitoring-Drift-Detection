"""
train_more_models.py - Train K-Means and Linear Regression models on the greenhouse dataset.
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import RAW_DATASET_PATH, FEATURE_NAMES, logger

def main():
    raw_path = ROOT / RAW_DATASET_PATH
    if not raw_path.exists():
        logger.error(f"Missing dataset at {raw_path}")
        sys.exit(1)

    # We need to make sure the target variable (yield) is also clean for Linear Regression
    df = pd.read_csv(raw_path).dropna(subset=FEATURE_NAMES + ["yield_kg_per_m2"])
    X = df[FEATURE_NAMES]
    y = df["yield_kg_per_m2"]

    logger.info("=== 1. Training K-Means (Unsupervised Clustering) ===")
    # K-Means will group similar greenhouse days into 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    kmeans.fit(X)
    logger.info("K-Means trained successfully! Cluster Inertia (loss): %.2f", kmeans.inertia_)
    
    logger.info("=== 2. Training Linear Regression (Predicting Crop Yield) ===")
    # Linear Regression tries to predict the exact "yield_kg_per_m2" based on temp/humidity etc.
    linreg = LinearRegression()
    linreg.fit(X, y)
    
    # Calculate performance metrics
    preds = linreg.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    logger.info("Linear Regression trained! R² Score: %.2f (RMSE: %.2f kg/m² error margin)", r2, rmse)
    
    # Save the models
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(kmeans, models_dir / "kmeans.joblib")
    joblib.dump(linreg, models_dir / "linreg.joblib")
    logger.info("Am salvat modelele KMeans și LinearRegression în folderul models/")

if __name__ == "__main__":
    main()
