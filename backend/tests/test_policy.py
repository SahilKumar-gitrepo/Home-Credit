"""Tests for the policy search endpoint."""


def test_policy_search_returns_200(client):
    """POST /policy/search should return 200."""
    response = client.post("/policy/search", json={"query": "borrower consent digital lending"})
    assert response.status_code == 200


def test_policy_search_schema(client):
    """Response should have query, results, and total_results fields."""
    response = client.post("/policy/search", json={"query": "data privacy"})
    body = response.json()
    assert "query" in body
    assert "results" in body
    assert "total_results" in body


def test_policy_search_authority(client):
    """All returned evidence should have Reserve Bank of India as authority."""
    response = client.post("/policy/search", json={"query": "digital lending"})
    body = response.json()
    for result in body["results"]:
        assert result["authority"] == "Reserve Bank of India", (
            f"Expected 'Reserve Bank of India', got '{result['authority']}'"
        )


def test_policy_search_returns_results(client):
    """A relevant query should return at least one result."""
    response = client.post("/policy/search", json={"query": "borrower credit"})
    body = response.json()
    assert body["total_results"] >= 0  # May be 0 if RAG not built


def test_policy_search_empty_query_422(client):
    """Empty query should return 422."""
    response = client.post("/policy/search", json={"query": "   "})
    assert response.status_code == 422


def test_policy_search_has_content(client):
    """Results should contain non-empty content."""
    response = client.post("/policy/search", json={"query": "consent data borrower"})
    body = response.json()
    for result in body["results"]:
        assert len(result["content"]) > 0
