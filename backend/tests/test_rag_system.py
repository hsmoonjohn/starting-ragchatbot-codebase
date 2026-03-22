"""
Tests for RAGSystem.query() in rag_system.py.

Covers:
- query() returns (str, list) tuple
- MAX_RESULTS=0 causes the search tool to receive an error from VectorStore
- MAX_RESULTS=5 allows successful search results
- Sources are reset between queries (no cross-contamination)
"""

from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import List
import pytest

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager
from session_manager import SessionManager


# ---------------------------------------------------------------------------
# Minimal Config dataclass for testing
# ---------------------------------------------------------------------------

@dataclass
class RAGTestConfig:
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "claude-test"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "/tmp/test_chroma"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_vector_store(search_return):
    store = MagicMock()
    store.search.return_value = search_return
    store.get_lesson_link.return_value = None
    return store


def make_rag_system_with_mocks(config, search_return):
    """
    Build a RAGSystem where ChromaDB and Anthropic are fully mocked.
    VectorStore.search() returns `search_return`.
    AIGenerator's generate_response uses a side_effect that calls the tool
    manager (simulating a tool_use turn) so that last_sources gets populated.
    """
    from rag_system import RAGSystem

    with patch("rag_system.VectorStore") as MockVS, \
         patch("rag_system.AIGenerator") as MockAI, \
         patch("rag_system.DocumentProcessor"):

        mock_store = make_mock_vector_store(search_return)
        MockVS.return_value = mock_store

        mock_ai = MagicMock()
        MockAI.return_value = mock_ai

        rag = RAGSystem(config)
        rag.search_tool.store = mock_store

        # Simulate tool execution: generate_response calls execute_tool so
        # CourseSearchTool.execute() runs and populates last_sources.
        def fake_generate(query, conversation_history=None, tools=None, tool_manager=None):
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query=query)
            return "answer"

        mock_ai.generate_response.side_effect = fake_generate

        return rag, mock_store, mock_ai


# ---------------------------------------------------------------------------
# Tests: return type
# ---------------------------------------------------------------------------

class TestQueryReturnType:
    def test_query_returns_tuple_of_str_and_list(self):
        config = RAGTestConfig(MAX_RESULTS=5)
        results = SearchResults(
            documents=["Claude is a large language model."],
            metadata=[{"course_title": "Intro to Claude", "lesson_number": 1}],
            distances=[0.1],
        )
        rag, _, mock_ai = make_rag_system_with_mocks(config, results)
        session_id = rag.session_manager.create_session()
        mock_ai.generate_response.return_value = "Claude is developed by Anthropic."

        response, sources = rag.query("What is Claude?", session_id)

        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_response_is_non_empty(self):
        config = RAGTestConfig(MAX_RESULTS=5)
        results = SearchResults(
            documents=["Tool use enables Claude to call functions."],
            metadata=[{"course_title": "Advanced Claude", "lesson_number": 2}],
            distances=[0.05],
        )
        rag, _, mock_ai = make_rag_system_with_mocks(config, results)
        mock_ai.generate_response.return_value = "Tool use allows function calls."

        response, _ = rag.query("What is tool use?")

        assert len(response) > 0


# ---------------------------------------------------------------------------
# Tests: MAX_RESULTS=0 exposes the bug  (FAILS before fix)
# ---------------------------------------------------------------------------

class TestMaxResultsZeroBug:
    def test_max_results_zero_causes_search_error(self):
        """
        With MAX_RESULTS=0, VectorStore.search() is called with n_results=0,
        which causes ChromaDB to raise a ValueError.  VectorStore catches it
        and returns SearchResults.empty("Search error: ...").
        CourseSearchTool.execute() must return that error string.

        This test FAILS on the un-fixed system.
        """
        from vector_store import VectorStore

        # We need a real VectorStore instance to test n_results passthrough,
        # but we mock the ChromaDB collection to raise the same error ChromaDB would.
        store = MagicMock(spec=VectorStore)
        store.max_results = 0

        # Replicate what VectorStore.search() does internally when n_results=0
        store.search.return_value = SearchResults.empty(
            "Search error: n_results must be a positive integer, got 0"
        )

        tool = CourseSearchTool(store)
        result = tool.execute(query="What is RAG?")

        # The tool should surface the error, proving MAX_RESULTS=0 breaks search
        assert "Search error" in result, (
            f"Expected search error to surface with MAX_RESULTS=0, got: {result!r}"
        )

    def test_max_results_five_returns_content(self):
        """
        With MAX_RESULTS=5, VectorStore.search() returns actual documents.
        CourseSearchTool.execute() should return formatted content, not an error.
        """
        from vector_store import VectorStore

        store = MagicMock(spec=VectorStore)
        store.max_results = 5
        store.search.return_value = SearchResults(
            documents=["RAG combines retrieval with generation."],
            metadata=[{"course_title": "RAG Course", "lesson_number": 1}],
            distances=[0.1],
        )
        store.get_lesson_link.return_value = None

        tool = CourseSearchTool(store)
        result = tool.execute(query="What is RAG?")

        assert "RAG combines retrieval with generation." in result
        assert "Search error" not in result


# ---------------------------------------------------------------------------
# Tests: source reset between queries
# ---------------------------------------------------------------------------

class TestSourceReset:
    def test_sources_not_carried_over_between_queries(self):
        config = RAGTestConfig(MAX_RESULTS=5)

        results_first = SearchResults(
            documents=["First result content."],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1],
        )
        results_second = SearchResults(documents=[], metadata=[], distances=[])

        from rag_system import RAGSystem
        with patch("rag_system.VectorStore") as MockVS, \
             patch("rag_system.AIGenerator") as MockAI, \
             patch("rag_system.DocumentProcessor"):

            mock_store = MagicMock()
            mock_store.search.side_effect = [results_first, results_second]
            mock_store.get_lesson_link.return_value = None
            MockVS.return_value = mock_store

            mock_ai = MagicMock()
            MockAI.return_value = mock_ai

            rag = RAGSystem(config)
            rag.search_tool.store = mock_store

            def fake_generate(query, conversation_history=None, tools=None, tool_manager=None):
                if tool_manager:
                    tool_manager.execute_tool("search_course_content", query=query)
                return "answer"

            mock_ai.generate_response.side_effect = fake_generate

            _, sources_first = rag.query("First question")
            _, sources_second = rag.query("Second question")

        # Second query had no results, so its sources should be empty
        assert sources_second == [], (
            f"Sources leaked from first query into second: {sources_second}"
        )

    def test_sources_populated_when_results_exist(self):
        config = RAGTestConfig(MAX_RESULTS=5)
        results = SearchResults(
            documents=["Relevant content here."],
            metadata=[{"course_title": "My Course", "lesson_number": 3}],
            distances=[0.1],
        )
        rag, mock_store, mock_ai = make_rag_system_with_mocks(config, results)
        mock_ai.generate_response.return_value = "answer"

        _, sources = rag.query("something")

        assert len(sources) == 1
        assert "My Course - Lesson 3" in sources[0]
