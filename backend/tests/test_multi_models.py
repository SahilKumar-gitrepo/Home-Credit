"""Tests for multi-model registry, model switching, and chaining."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_list_models():
    response = client.get("/model/list")
    assert response.status_code == 200
    data = response.json()
    assert "active_model_id" in data
    assert "models" in data
    model_ids = [m["id"] for m in data["models"]]
    assert "xgboost" in model_ids
    assert "lightgbm" in model_ids
    assert "logistic" in model_ids
    assert "chained_ensemble" in model_ids
    assert "cascade_hurdle" in model_ids


def test_get_model_comparison():
    response = client.get("/model/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "xgboost" in data
    assert "lightgbm" in data
    assert "logistic" in data
    assert "chained_ensemble" in data
    assert "cascade_hurdle" in data
    assert data["chained_ensemble"]["roc_auc"] > 0.75


def test_get_model_info_specific():
    response = client.get("/model/info?model_id=lightgbm")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "lightgbm"
    assert "LightGBM" in data["model_type"]
    assert data["feature_count"] > 0
    assert len(data["top_features"]) > 0


def test_switch_active_model():
    # Switch to lightgbm
    res = client.post("/model/active", json={"model_id": "lightgbm"})
    assert res.status_code == 200
    assert res.json()["active_model_id"] == "lightgbm"

    # Verify /model/list reflects change
    list_res = client.get("/model/list")
    assert list_res.json()["active_model_id"] == "lightgbm"

    # Switch back to xgboost_calibrated default
    client.post("/model/active", json={"model_id": "xgboost_calibrated"})


def test_underwrite_with_chained_models():
    # Chained ensemble underwriting
    res_ens = client.post("/underwrite", json={"applicant_id": 370920, "model_id": "chained_ensemble"})
    assert res_ens.status_code == 200
    ens_data = res_ens.json()
    assert 0.0 <= ens_data["probability_of_default"] <= 1.0
    assert "Chained Weighted Ensemble" in ens_data["model_information"]["model_type"]

    # Cascade hurdle underwriting
    res_cas = client.post("/underwrite", json={"applicant_id": 370920, "model_id": "cascade_hurdle"})
    assert res_cas.status_code == 200
    cas_data = res_cas.json()
    assert 0.0 <= cas_data["probability_of_default"] <= 1.0
