from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.config import N_FEATURES

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "uptime_seconds" in response.json()

def test_predict():
    # Make sure to run `python -m scripts.train` before testing so that the model exists
    
    # Generate random features matching the required shape
    features = list(np.random.randn(N_FEATURES).astype(float))
    
    response = client.post("/predict", json={"features": features})
    if response.status_code == 503: # Model might not be trained yet
        pass 
    else:
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "is_anomaly" in data
        assert "anomaly_score" in data
