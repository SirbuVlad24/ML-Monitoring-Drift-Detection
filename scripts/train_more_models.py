"""
train_more_models.py - Train K-Means, LinReg, and RandomForest (Classification)
"""
import sys
import json
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import RAW_DATASET_PATH, FEATURE_NAMES, logger

def main():
    raw_path = ROOT / RAW_DATASET_PATH
    if not raw_path.exists():
        logger.error(f"Missing dataset at {raw_path}")
        sys.exit(1)

    # 1. Prepare data
    df = pd.read_csv(raw_path).dropna(subset=FEATURE_NAMES + ["yield_kg_per_m2", "crop_type"])
    X = df[FEATURE_NAMES]
    
    # Target 1: Yield (Regression)
    y_reg = df["yield_kg_per_m2"]
    
    # Target 2: Crop Type (Classification)
    y_clf = df["crop_type"]

    metrics = {}

    # === A. K-Means (Clustering) ===
    logger.info("=== Training K-Means (Unsupervised) ===")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    kmeans.fit(X)
    metrics["kmeans"] = {"inertia": round(kmeans.inertia_, 2)}

    # === B. Linear Regression (Regression) ===
    logger.info("=== Training Linear Regression ===")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    linreg = LinearRegression()
    linreg.fit(X_train_r, y_train_r)
    
    preds_r = linreg.predict(X_test_r)
    rmse = mean_squared_error(y_test_r, preds_r, squared=False)
    r2 = r2_score(y_test_r, preds_r)
    metrics["regression"] = {"rmse": round(rmse, 2), "r2": round(r2, 2)}

    # === C. Random Forest (Classification) ===
    logger.info("=== Training Classifier for F1, Accuracy, Precision ===")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_c, y_train_c)
    
    preds_c = classifier.predict(X_test_c)
    
    # Calculate Classification Metrics
    acc = accuracy_score(y_test_c, preds_c)
    prec = precision_score(y_test_c, preds_c, average="macro", zero_division=0)
    rec = recall_score(y_test_c, preds_c, average="macro", zero_division=0)
    f1 = f1_score(y_test_c, preds_c, average="macro", zero_division=0)
    
    metrics["classification"] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }

    # Save all models & metrics
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(kmeans, models_dir / "kmeans.joblib")
    joblib.dump(linreg, models_dir / "linreg.joblib")
    joblib.dump(classifier, models_dir / "classifier.joblib")
    
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("All models and Metrics saved successfully in models/")
    logger.info(f"Classification Metrics: {metrics['classification']}")

if __name__ == "__main__":
    main()
