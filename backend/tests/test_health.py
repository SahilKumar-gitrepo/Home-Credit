"""Tests for GET /health endpoint."""


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status(client):
    data = response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"


def test_health_has_timestamp(client):
    response = client.get("/health")
    body = response.json()
    assert "timestamp" in body


def test_health_has_service_name(client):
    response = client.get("/health")
    body = response.json()
    assert "service" in body
    assert "Credit Underwriting" in body["service"]
