"""Tests for the FastAPI application endpoints."""
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Prevent ChatEngine (and its heavy dependencies) from being instantiated
# at import time.  We inject a fake module for app.core.chat_engine into
# sys.modules *before* app.api.endpoints is imported so that the module-level
#   chat_engine = ChatEngine()
# line receives a MagicMock instead of the real class.
# ---------------------------------------------------------------------------
_fake_chat_engine_module = ModuleType("app.core.chat_engine")
_mock_chat_engine_cls = MagicMock()
_fake_chat_engine_module.ChatEngine = _mock_chat_engine_cls  # type: ignore[attr-defined]
sys.modules.setdefault("app.core.chat_engine", _fake_chat_engine_module)

# SafetyClassifier is pure keyword matching and has no heavy deps — no need
# to stub it out.  It will be imported normally when app.api.endpoints loads.

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_check():
    """GET /api/health returns 200 with expected status message."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "WeightLoss RAG API is ready."}


def test_chat_stream_returns_200_with_sse_content_type():
    """POST /api/chat with valid payload returns 200 and SSE content type."""
    def _fake_stream(query, session_id, filters=None):
        yield '{"type":"token","content":"hello"}\n\n'

    with patch("app.api.endpoints.chat_engine") as mock_engine:
        mock_engine.stream_chat.side_effect = _fake_stream
        response = client.post(
            "/api/chat",
            json={"query": "What is semaglutide?", "session_id": "sess-001"},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_chat_missing_query_returns_422():
    """POST /api/chat without query field returns 422 validation error."""
    response = client.post(
        "/api/chat",
        json={"session_id": "sess-001"},
    )
    assert response.status_code == 422


def test_chat_missing_session_id_returns_422():
    """POST /api/chat without session_id field returns 422 validation error."""
    response = client.post(
        "/api/chat",
        json={"query": "What is semaglutide?"},
    )
    assert response.status_code == 422


from app.models.response_schemas import StructuredAnswer  # noqa: E402


def test_structured_query_returns_200():
    """POST /api/query with valid payload returns 200 and contains the summary."""
    mock_result = StructuredAnswer(
        summary="Test summary",
        claims=[],
        source_pmids=[],
        limitations=None,
    )

    with patch("app.api.endpoints.chat_engine") as mock_engine:
        mock_engine.qa_chain.structured_query.return_value = mock_result
        response = client.post(
            "/api/query",
            json={"query": "What is semaglutide?", "session_id": "sess-001"},
        )

    assert response.status_code == 200
    assert response.json()["summary"] == "Test summary"


def test_structured_query_missing_query_returns_422():
    """POST /api/query without query field returns 422 validation error."""
    response = client.post(
        "/api/query",
        json={"session_id": "test"},
    )
    assert response.status_code == 422


def test_chat_high_risk_query_includes_safety_event():
    """POST /api/chat with a high-risk dosage query yields a safety SSE event first."""
    def _fake_stream(query, session_id, filters=None):
        yield '{"type":"token","content":"hello"}\n\n'

    with patch("app.api.endpoints.chat_engine") as mock_engine:
        mock_engine.stream_chat.side_effect = _fake_stream
        response = client.post(
            "/api/chat",
            json={"query": "Should I take 2mg of semaglutide?", "session_id": "sess-002"},
        )

    assert response.status_code == 200
    assert '"type": "safety"' in response.text or '"type":"safety"' in response.text


def test_chat_low_risk_query_no_safety_event():
    """POST /api/chat with a general query does NOT yield a safety SSE event."""
    def _fake_stream(query, session_id, filters=None):
        yield '{"type":"token","content":"hello"}\n\n'

    with patch("app.api.endpoints.chat_engine") as mock_engine:
        mock_engine.stream_chat.side_effect = _fake_stream
        response = client.post(
            "/api/chat",
            json={"query": "What is semaglutide?", "session_id": "sess-003"},
        )

    assert response.status_code == 200
    assert '"type": "safety"' not in response.text and '"type":"safety"' not in response.text
