from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_metadata():
    with TestClient(app) as client:
        response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert "run_id" in data
    assert "experiment_name" in data
    assert "registered_model_name" in data
    assert "registered_model_version" in data
    assert "accuracy" in data
    assert "f1_macro" in data


def test_predict():
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_class" in data
    assert "predicted_label" in data
    assert "probabilities" in data
    assert "model_run_id" in data
    assert "model_version" in data

    assert data["predicted_label"] in ["setosa", "versicolor", "virginica"]