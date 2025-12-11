import json
from fastapi.testclient import TestClient

from backend.app.main import app, _get_service


class FakeService:
    def model_info_payload(self):
        return {
            "version": "test",
            "model_type": "rf",
            "random_seed": 1,
            "window_seconds": 5,
            "window_overlap_seconds": 2.5,
            "sample_rate_hz": 50,
            "excluded_subjects_demo": [9, 10],
            "splits": {},
            "feature_columns": [],
            "metrics": {},
        }

    def predict(self, file):
        return {
            "per_window": [
                {
                    "window_index": 0,
                    "prediction": 1,
                    "activity": "standing",
                    "proba": {"standing": 0.7},
                }
            ],
            "aggregate": {"fraction_per_activity": {"standing": 1.0}, "mean_proba": {"standing": 0.7}},
        }

    def evaluate(self, file):
        return {"metrics": {"accuracy": 1.0, "macro_f1": 1.0, "confusion_matrix": [[1]]}, "predictions": [1]}


app.dependency_overrides[_get_service] = lambda: FakeService()
client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    assert resp.json()["model_type"] == "rf"


def test_predict():
    resp = client.post(
        "/predict",
        files={"file": ("test.log", "1 2 3 4")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "per_window" in data


def test_evaluate():
    resp = client.post(
        "/evaluate-log",
        files={"file": ("test.log", "1 2 3 4")},
    )
    assert resp.status_code == 200
    assert resp.json()["metrics"]["accuracy"] == 1.0
