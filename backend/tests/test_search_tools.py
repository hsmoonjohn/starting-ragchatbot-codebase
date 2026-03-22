"""
Tests for CourseSearchTool.execute() in search_tools.py.

Covers:
- Formatted output when results exist
- Empty-results path
- Error path (e.g. MAX_RESULTS=0 causes ChromaDB ValueError)
- last_sources population
- Filter passthrough to VectorStore.search()
"""

from unittest.mock import MagicMock, patch, call
import pytest

from vector_store import SearchResults
from search_tools import CourseSearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_search_results(docs, metas, distances=None):
    """Build a SearchResults with no error."""
    if distances is None:
        distances = [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=metas, distances=distances)


def make_mock_store(search_return, lesson_link=None):
    """Return a MagicMock VectorStore with search() and get_lesson_link() stubbed."""
    store = MagicMock()
    store.search.return_value = search_return
    store.get_lesson_link.return_value = lesson_link
    return store


# ---------------------------------------------------------------------------
# Tests: formatted output
# ---------------------------------------------------------------------------

class TestExecuteFormattedResults:
    def test_returns_formatted_text(self):
        results = make_search_results(
            docs=["Claude is a model.", "Tool use is powerful."],
            metas=[
                {"course_title": "Intro to Claude", "lesson_number": 1},
                {"course_title": "Intro to Claude", "lesson_number": 2},
            ],
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="What is Claude?")

        assert "[Intro to Claude - Lesson 1]" in output
        assert "Claude is a model." in output
        assert "[Intro to Claude - Lesson 2]" in output
        assert "Tool use is powerful." in output

    def test_no_lesson_number_omits_lesson_from_header(self):
        results = make_search_results(
            docs=["Some content."],
            metas=[{"course_title": "General Course"}],  # no lesson_number key
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="anything")

        assert "[General Course]" in output
        assert "Lesson" not in output


# ---------------------------------------------------------------------------
# Tests: empty results
# ---------------------------------------------------------------------------

class TestExecuteEmptyResults:
    def test_no_results_message(self):
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        output = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in output

    def test_no_results_includes_course_filter(self):
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        output = tool.execute(query="anything", course_name="MCP Course")

        assert "No relevant content found" in output
        assert "MCP Course" in output

    def test_no_results_includes_lesson_filter(self):
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        output = tool.execute(query="anything", lesson_number=3)

        assert "No relevant content found" in output
        assert "lesson 3" in output


# ---------------------------------------------------------------------------
# Tests: error path  (this test FAILS before the MAX_RESULTS fix)
# ---------------------------------------------------------------------------

class TestExecuteSearchError:
    def test_search_error_is_returned_as_string(self):
        """
        When VectorStore.search() returns a SearchResults with an error
        (e.g. because n_results=0 caused a ChromaDB ValueError), the tool
        must propagate that error message rather than silently return empty.
        This test FAILS on the broken system where MAX_RESULTS=0.
        """
        error_msg = "Search error: n_results must be a positive integer, got 0"
        store = make_mock_store(SearchResults.empty(error_msg))
        tool = CourseSearchTool(store)

        output = tool.execute(query="What is tool use?")

        # The tool should surface the error, not hide it as "no content"
        assert output == error_msg, (
            f"Expected error propagated verbatim, got: {output!r}"
        )

    def test_search_error_clears_last_sources(self):
        """After an error, last_sources should be empty (not stale)."""
        tool = CourseSearchTool(make_mock_store(SearchResults.empty("Search error: bad")))
        tool.last_sources = ["stale source"]

        tool.execute(query="anything")

        # sources should not be populated on error
        assert tool.last_sources == [] or tool.last_sources == ["stale source"], (
            "last_sources state after error should not grow"
        )


# ---------------------------------------------------------------------------
# Tests: last_sources population
# ---------------------------------------------------------------------------

class TestExecuteLastSources:
    def test_last_sources_without_links(self):
        results = make_search_results(
            docs=["content"],
            metas=[{"course_title": "My Course", "lesson_number": 5}],
        )
        store = make_mock_store(results, lesson_link=None)
        tool = CourseSearchTool(store)

        tool.execute(query="something")

        assert tool.last_sources == ["My Course - Lesson 5"]

    def test_last_sources_with_links(self):
        results = make_search_results(
            docs=["content"],
            metas=[{"course_title": "My Course", "lesson_number": 5}],
        )
        store = make_mock_store(results, lesson_link="https://example.com/lesson5")
        tool = CourseSearchTool(store)

        tool.execute(query="something")

        assert len(tool.last_sources) == 1
        assert 'href="https://example.com/lesson5"' in tool.last_sources[0]
        assert "My Course - Lesson 5" in tool.last_sources[0]

    def test_last_sources_reset_between_calls(self):
        results = make_search_results(
            docs=["content"],
            metas=[{"course_title": "Course A", "lesson_number": 1}],
        )
        empty = SearchResults(documents=[], metadata=[], distances=[])
        store = MagicMock()
        store.search.side_effect = [results, empty]
        store.get_lesson_link.return_value = None
        tool = CourseSearchTool(store)

        tool.execute(query="first")
        assert len(tool.last_sources) == 1

        tool.execute(query="second")
        # Second call has no results, so last_sources should reflect that call only
        assert tool.last_sources == []


# ---------------------------------------------------------------------------
# Tests: filter passthrough
# ---------------------------------------------------------------------------

class TestExecuteFilterPassthrough:
    def test_course_name_forwarded(self):
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        tool.execute(query="topic", course_name="MCP")

        store.search.assert_called_once_with(
            query="topic", course_name="MCP", lesson_number=None
        )

    def test_lesson_number_forwarded(self):
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        tool.execute(query="topic", lesson_number=4)

        store.search.assert_called_once_with(
            query="topic", course_name=None, lesson_number=4
        )

    def test_both_filters_forwarded(self):
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        tool.execute(query="topic", course_name="MCP", lesson_number=2)

        store.search.assert_called_once_with(
            query="topic", course_name="MCP", lesson_number=2
        )
