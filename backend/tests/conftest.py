import sys
import os

# Add backend/ to sys.path so bare imports (from search_tools import ...) work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_query_request():
    return {"query": "What is RAG?"}


@pytest.fixture
def sample_query_request_with_session():
    return {"query": "What is RAG?", "session_id": "session_42"}


# ---------------------------------------------------------------------------
# Mock RAGSystem
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag():
    """A MagicMock standing in for RAGSystem with sensible defaults."""
    rag = MagicMock()
    rag.query.return_value = ("RAG stands for Retrieval-Augmented Generation.", ["Course A - Lesson 1"])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    rag.session_manager.create_session.return_value = "session_1"
    return rag


# ---------------------------------------------------------------------------
# Test app — mirrors app.py endpoints without the static-file mount
# ---------------------------------------------------------------------------

@pytest.fixture
def client(mock_rag):
    """TestClient for a minimal FastAPI app wired to mock_rag."""

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    test_app = FastAPI()

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        mock_rag.session_manager.delete_session(session_id)
        return {"status": "ok"}

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return TestClient(test_app)
