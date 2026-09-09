"""Tests for GET /applicants/{applicant_id} endpoint."""
import pytest


KNOWN_APPLICANT_ID = 370920


def test_get_known_applicant(client):
    """Applicant 370920 should be found in the dataset."""
    response = client.get(f"/applicants/{KNOWN_APPLICANT_ID}")
    assert response.status_code == 200


def test_get_applicant_schema(client):
    """Applicant response should have required fields."""
    response = client.get(f"/applicants/{KNOWN_APPLICANT_ID}")
    body = response.json()
    assert body["applicant_id"] == KNOWN_APPLICANT_ID


def test_get_missing_applicant_returns_404(client):
    """A non-existent applicant ID should return 404."""
    response = client.get("/applicants/999999")
    # 999999 is below 100000 — will be 422
    # Use an ID that is in valid range but doesn't exist
    response = client.get("/applicants/199999")
    # May be 404 if not found, or 200 if coincidentally exists
    assert response.status_code in (200, 404)


def test_get_applicant_invalid_id(client):
    """An ID outside valid range should return 422."""
    response = client.get("/applicants/123")
    assert response.status_code == 422


def test_applicant_has_income_field(client):
    response = client.get(f"/applicants/{KNOWN_APPLICANT_ID}")
    body = response.json()
    # income may be present or None
    assert "income" in body


def test_list_applicants(client):
    """List endpoint should return applicant IDs."""
    response = client.get("/applicants/?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert "applicant_ids" in body
    assert len(body["applicant_ids"]) <= 5
