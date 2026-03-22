"""
Tests for the FastAPI endpoints defined in app.py.

Uses a minimal test app (defined in conftest.py) that mirrors the real
/api/query, /api/courses, and /api/session/{id} endpoints but does NOT
mount static files, avoiding the missing ../frontend directory in CI.

Covers:
- POST /api/query: success, existing session, missing field, RAGSystem error
- GET  /api/courses: success, RAGSystem error
- DELETE /api/session/{session_id}: success
"""

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_returns_200_with_expected_fields(self, client, sample_query_request):
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert "sources" in body
        assert "session_id" in body

    def test_answer_and_sources_come_from_rag(self, client, mock_rag, sample_query_request):
        mock_rag.query.return_value = ("Mocked answer.", ["Source A"])

        body = client.post("/api/query", json=sample_query_request).json()

        assert body["answer"] == "Mocked answer."
        assert body["sources"] == ["Source A"]

    def test_new_session_created_when_none_provided(self, client, mock_rag):
        mock_rag.session_manager.create_session.return_value = "session_new"

        body = client.post("/api/query", json={"query": "Hello"}).json()

        mock_rag.session_manager.create_session.assert_called_once()
        assert body["session_id"] == "session_new"

    def test_existing_session_id_is_preserved(self, client, mock_rag, sample_query_request_with_session):
        body = client.post("/api/query", json=sample_query_request_with_session).json()

        assert body["session_id"] == "session_42"
        # create_session must NOT be called when a session_id is supplied
        mock_rag.session_manager.create_session.assert_not_called()

    def test_rag_query_called_with_correct_args(self, client, mock_rag, sample_query_request_with_session):
        client.post("/api/query", json=sample_query_request_with_session)

        mock_rag.query.assert_called_once_with("What is RAG?", "session_42")

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={"session_id": "session_1"})

        assert response.status_code == 422

    def test_rag_exception_returns_500(self, client, mock_rag, sample_query_request):
        mock_rag.query.side_effect = RuntimeError("ChromaDB unavailable")

        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        assert "ChromaDB unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    def test_returns_200_with_expected_fields(self, client):
        response = client.get("/api/courses")

        assert response.status_code == 200
        body = response.json()
        assert "total_courses" in body
        assert "course_titles" in body

    def test_course_stats_match_rag_analytics(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Alpha", "Beta", "Gamma"],
        }

        body = client.get("/api/courses").json()

        assert body["total_courses"] == 3
        assert body["course_titles"] == ["Alpha", "Beta", "Gamma"]

    def test_empty_catalog_returns_zero_courses(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        body = client.get("/api/courses").json()

        assert body["total_courses"] == 0
        assert body["course_titles"] == []

    def test_rag_exception_returns_500(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("DB error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "DB error" in response.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:
    def test_returns_200_with_ok_status(self, client):
        response = client.delete("/api/session/session_99")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_delete_session_called_with_correct_id(self, client, mock_rag):
        client.delete("/api/session/session_99")

        mock_rag.session_manager.delete_session.assert_called_once_with("session_99")
