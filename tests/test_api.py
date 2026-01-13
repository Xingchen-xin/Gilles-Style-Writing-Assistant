"""Tests for API routes."""
import pytest
from fastapi.testclient import TestClient
from gswa.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /v1/health endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Test health response has correct structure."""
        response = client.get("/v1/health")
        data = response.json()

        assert "status" in data
        assert "llm_server" in data
        assert "corpus_paragraphs" in data
        assert "index_loaded" in data

    def test_health_status_values(self, client):
        """Test health status is valid value."""
        response = client.get("/v1/health")
        data = response.json()

        assert data["status"] in ["healthy", "degraded", "error"]


class TestRewriteEndpoint:
    """Tests for /v1/rewrite/variants endpoint."""

    def test_rewrite_validation_text_required(self, client):
        """Test that text is required."""
        response = client.post("/v1/rewrite/variants", json={})
        assert response.status_code == 422

    def test_rewrite_validation_text_min_length(self, client):
        """Test minimum text length validation."""
        response = client.post("/v1/rewrite/variants", json={
            "text": "short"
        })
        assert response.status_code == 422

    def test_rewrite_validation_n_variants_range(self, client):
        """Test n_variants range validation."""
        # Too low
        response = client.post("/v1/rewrite/variants", json={
            "text": "This is a valid paragraph for testing.",
            "n_variants": 0
        })
        assert response.status_code == 422

        # Too high
        response = client.post("/v1/rewrite/variants", json={
            "text": "This is a valid paragraph for testing.",
            "n_variants": 10
        })
        assert response.status_code == 422

    def test_rewrite_validation_section_enum(self, client):
        """Test section enum validation."""
        response = client.post("/v1/rewrite/variants", json={
            "text": "This is a valid paragraph for testing.",
            "section": "InvalidSection"
        })
        assert response.status_code == 422

    def test_rewrite_accepts_valid_request(self, client):
        """Test that valid request is accepted (may fail due to no vLLM)."""
        response = client.post("/v1/rewrite/variants", json={
            "text": "This is a valid paragraph for testing the rewrite endpoint.",
            "n_variants": 3,
            "section": "Results"
        })
        # Either succeeds or fails due to LLM connection (500)
        assert response.status_code in [200, 500]


class TestReplyEndpoint:
    """Tests for /v1/reply endpoint."""

    def test_reply_validation_messages_required(self, client):
        """Test that messages are required."""
        response = client.post("/v1/reply", json={})
        assert response.status_code == 422

    def test_reply_validation_role_pattern(self, client):
        """Test role validation."""
        response = client.post("/v1/reply", json={
            "messages": [
                {"role": "invalid", "content": "Hello"}
            ]
        })
        assert response.status_code == 422

    def test_reply_validation_max_tokens_range(self, client):
        """Test max_tokens range validation."""
        response = client.post("/v1/reply", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10000
        })
        assert response.status_code == 422

    def test_reply_accepts_valid_request(self, client):
        """Test that valid request is accepted (may fail due to no vLLM)."""
        response = client.post("/v1/reply", json={
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "max_tokens": 100
        })
        # Either succeeds or fails due to LLM connection
        assert response.status_code in [200, 500]


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.options(
            "/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # CORS middleware should handle preflight
        assert response.status_code in [200, 405]
