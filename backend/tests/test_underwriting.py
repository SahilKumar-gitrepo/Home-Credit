"""
Tests for the underwriting endpoint and decision logic.

KNOWN APPLICANT REGRESSION TEST:
Applicant 370920 should produce approximately:
  PD = 1.51%  (LOW risk tier)
  Decision = APPROVE
This is a regression check for the calibrated model/dataset.
"""
import pytest

KNOWN_APPLICANT_ID = 370920
PD_TOLERANCE = 0.005  # 0.5% tolerance around expected PD


def test_underwrite_known_applicant(client):
    """POST /underwrite with known applicant should return 200."""
    response = client.post("/underwrite", json={"applicant_id": KNOWN_APPLICANT_ID})
    assert response.status_code == 200


def test_underwrite_known_applicant_decision(client):
    """Applicant 370920 regression: should be APPROVE with LOW risk."""
    response = client.post("/underwrite", json={"applicant_id": KNOWN_APPLICANT_ID})
    body = response.json()
    # Regression checks — not general system behaviour
    assert body["decision"] == "APPROVE", f"Expected APPROVE, got {body['decision']}"
    assert body["risk_tier"] == "LOW", f"Expected LOW, got {body['risk_tier']}"


def test_underwrite_known_applicant_pd(client):
    """Applicant 370920 regression: calibrated PD should be approximately 1.51%."""
    response = client.post("/underwrite", json={"applicant_id": KNOWN_APPLICANT_ID})
    body = response.json()
    pd_val = body["probability_of_default"]
    assert abs(pd_val - 0.0151) < PD_TOLERANCE, (
        f"PD {pd_val:.4%} is more than {PD_TOLERANCE:.1%} from expected 1.51%"
    )


def test_underwrite_response_schema(client):
    """Response should conform to the expected schema."""
    response = client.post("/underwrite", json={"applicant_id": KNOWN_APPLICANT_ID})
    body = response.json()
    required_fields = [
        "request_id", "applicant_id", "timestamp",
        "probability_of_default", "risk_tier", "decision",
        "policy_reason", "risk_increasing_factors", "risk_reducing_factors",
        "regulatory_evidence", "model_information",
    ]
    for field in required_fields:
        assert field in body, f"Missing field: {field}"


def test_underwrite_missing_applicant(client):
    """Non-existent applicant should return 404."""
    response = client.post("/underwrite", json={"applicant_id": 123456})
    assert response.status_code == 404


def test_underwrite_invalid_id_too_small(client):
    """Applicant ID below minimum should return 422."""
    response = client.post("/underwrite", json={"applicant_id": 99})
    assert response.status_code == 422


def test_underwrite_shap_factors_present(client):
    """SHAP factors should be present in the response."""
    response = client.post("/underwrite", json={"applicant_id": KNOWN_APPLICANT_ID})
    body = response.json()
    assert isinstance(body["risk_increasing_factors"], list)
    assert isinstance(body["risk_reducing_factors"], list)


def test_underwrite_regulatory_evidence_authority(client):
    """Regulatory evidence must have Reserve Bank of India as authority."""
    response = client.post("/underwrite", json={"applicant_id": KNOWN_APPLICANT_ID})
    body = response.json()
    for evidence in body["regulatory_evidence"]:
        assert evidence["authority"] == "Reserve Bank of India", (
            f"Expected 'Reserve Bank of India', got '{evidence['authority']}'"
        )


def test_get_underwriting_route(client):
    """GET /underwriting/{id} should also return a result."""
    response = client.get(f"/underwriting/{KNOWN_APPLICANT_ID}")
    assert response.status_code == 200
    body = response.json()
    assert body["applicant_id"] == KNOWN_APPLICANT_ID
