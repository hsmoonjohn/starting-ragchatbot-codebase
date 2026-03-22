"""
Tests for AIGenerator in ai_generator.py.

Covers:
- Direct response (no tool use)
- Tool-use path: tool_manager.execute_tool() is called with correct args
- Tool result is sent in the second API call
- Final response comes from the second API call
"""

from unittest.mock import MagicMock, patch, call
import pytest

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic responses
# ---------------------------------------------------------------------------

def make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_name, tool_input, tool_id="tool_abc123"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    return block


def make_response(stop_reason, content_blocks):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp


def make_generator(mock_client=None):
    """Return an AIGenerator with a stubbed Anthropic client."""
    if mock_client is None:
        mock_client = MagicMock()
    gen = AIGenerator.__new__(AIGenerator)
    gen.client = mock_client
    gen.model = "claude-test"
    gen.base_params = {"model": "claude-test", "temperature": 0, "max_tokens": 800}
    return gen


# ---------------------------------------------------------------------------
# Tests: direct response (no tool use)
# ---------------------------------------------------------------------------

class TestDirectResponse:
    def test_returns_text_when_no_tool_use(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("Direct answer.")]
        )
        gen = make_generator(mock_client)
        tool_manager = MagicMock()

        result = gen.generate_response(query="What is 2+2?", tool_manager=tool_manager)

        assert result == "Direct answer."

    def test_tool_manager_not_called_on_direct_response(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("No tools needed.")]
        )
        gen = make_generator(mock_client)
        tool_manager = MagicMock()

        gen.generate_response(query="Hello", tool_manager=tool_manager)

        tool_manager.execute_tool.assert_not_called()

    def test_only_one_api_call_on_direct_response(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("Answer")]
        )
        gen = make_generator(mock_client)

        gen.generate_response(query="Hello")

        assert mock_client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Tests: tool-use path
# ---------------------------------------------------------------------------

class TestToolUseExecution:
    def _setup_tool_use(self, tool_name="search_course_content",
                        tool_input={"query": "What is RAG?"},
                        tool_id="tool_1"):
        """Set up a generator where first call returns tool_use, second returns text."""
        mock_client = MagicMock()
        tool_block = make_tool_use_block(tool_name, tool_input, tool_id)
        first_response = make_response("tool_use", [tool_block])
        second_response = make_response("end_turn", [make_text_block("Synthesized answer.")])
        mock_client.messages.create.side_effect = [first_response, second_response]
        gen = make_generator(mock_client)
        return gen, mock_client, tool_block

    def test_execute_tool_called_with_correct_name_and_args(self):
        gen, mock_client, _ = self._setup_tool_use(
            tool_name="search_course_content",
            tool_input={"query": "What is RAG?", "course_name": "Intro"},
        )
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Found: RAG is ..."

        gen.generate_response(
            query="What is RAG?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="What is RAG?",
            course_name="Intro",
        )

    def test_two_api_calls_made(self):
        gen, mock_client, _ = self._setup_tool_use()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool result"

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert mock_client.messages.create.call_count == 2

    def test_final_response_is_from_second_call(self):
        gen, _, _ = self._setup_tool_use()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool result"

        result = gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "Synthesized answer."

    def test_tool_result_included_in_second_call_messages(self):
        gen, mock_client, _ = self._setup_tool_use(tool_id="tool_xyz")
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "the search result"

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        # The second call's messages should include a tool_result entry
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_msgs = [
            m for m in messages
            if isinstance(m.get("content"), list)
            and any(c.get("type") == "tool_result" for c in m["content"])
        ]
        assert len(tool_result_msgs) == 1
        result_content = tool_result_msgs[0]["content"][0]
        assert result_content["tool_use_id"] == "tool_xyz"
        assert result_content["content"] == "the search result"

    def test_second_call_includes_tools(self):
        """The second call (round 0, not last round) must still include tools so Claude can chain."""
        gen, mock_client, _ = self._setup_tool_use()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs


# ---------------------------------------------------------------------------
# Tests: conversation history included
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_history_appended_to_system_prompt(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("answer")]
        )
        gen = make_generator(mock_client)

        gen.generate_response(
            query="follow-up question",
            conversation_history="User: hello\nAssistant: hi",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "User: hello" in call_kwargs["system"]
        assert "Assistant: hi" in call_kwargs["system"]


# ---------------------------------------------------------------------------
# Tests: two sequential tool rounds
# ---------------------------------------------------------------------------

class TestTwoRoundToolUse:
    def _setup_two_rounds(self):
        """First two calls return tool_use, third returns end_turn (synthesis)."""
        mock_client = MagicMock()
        block1 = make_tool_use_block("get_course_outline", {"course_name": "X"}, "tool_r1")
        block2 = make_tool_use_block("search_course_content", {"query": "topic"}, "tool_r2")
        r1 = make_response("tool_use", [block1])
        r2 = make_response("tool_use", [block2])
        r3 = make_response("end_turn", [make_text_block("Final answer.")])
        mock_client.messages.create.side_effect = [r1, r2, r3]
        gen = make_generator(mock_client)
        return gen, mock_client

    def test_three_api_calls_for_two_tool_rounds(self):
        gen, mock_client = self._setup_two_rounds()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="q",
            tools=[{"name": "get_course_outline"}],
            tool_manager=tool_manager,
        )

        assert mock_client.messages.create.call_count == 3

    def test_both_tools_executed(self):
        gen, mock_client = self._setup_two_rounds()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="q",
            tools=[{"name": "get_course_outline"}],
            tool_manager=tool_manager,
        )

        assert tool_manager.execute_tool.call_count == 2

    def test_final_response_from_third_call(self):
        gen, _ = self._setup_two_rounds()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        result = gen.generate_response(
            query="q",
            tools=[{"name": "get_course_outline"}],
            tool_manager=tool_manager,
        )

        assert result == "Final answer."

    def test_second_call_includes_tools(self):
        gen, mock_client = self._setup_two_rounds()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="q",
            tools=[{"name": "get_course_outline"}],
            tool_manager=tool_manager,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs

    def test_third_call_has_no_tools_key(self):
        gen, mock_client = self._setup_two_rounds()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="q",
            tools=[{"name": "get_course_outline"}],
            tool_manager=tool_manager,
        )

        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs


# ---------------------------------------------------------------------------
# Tests: early termination when second call returns end_turn
# ---------------------------------------------------------------------------

class TestEarlyTermination:
    def test_two_calls_when_single_round_ends_cleanly(self):
        """If Call 2 returns end_turn, no synthesis call fires — total 2 calls."""
        mock_client = MagicMock()
        block = make_tool_use_block("search_course_content", {"query": "RAG"}, "tool_1")
        r1 = make_response("tool_use", [block])
        r2 = make_response("end_turn", [make_text_block("Clean answer.")])
        mock_client.messages.create.side_effect = [r1, r2]
        gen = make_generator(mock_client)
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "search result"

        result = gen.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert mock_client.messages.create.call_count == 2
        assert result == "Clean answer."


# ---------------------------------------------------------------------------
# Tests: tool execution error handling
# ---------------------------------------------------------------------------

class TestToolExecutionError:
    def test_error_string_in_tool_result(self):
        """When execute_tool raises, the error message appears in tool_result content."""
        mock_client = MagicMock()
        block = make_tool_use_block("search_course_content", {"query": "RAG"}, "tool_err")
        r1 = make_response("tool_use", [block])
        r2 = make_response("end_turn", [make_text_block("Sorry, could not retrieve.")])
        mock_client.messages.create.side_effect = [r1, r2]
        gen = make_generator(mock_client)
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = Exception("DB error")

        gen.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        # Find tool_result in the synthesis call's messages
        synthesis_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = synthesis_kwargs["messages"]
        tool_result_msgs = [
            m for m in messages
            if isinstance(m.get("content"), list)
            and any(c.get("type") == "tool_result" for c in m["content"])
        ]
        assert len(tool_result_msgs) == 1
        assert "Tool execution error: DB error" in tool_result_msgs[0]["content"][0]["content"]

    def test_synthesis_call_made_after_error(self):
        """Even when execute_tool raises, a synthesis call is made and result is returned."""
        mock_client = MagicMock()
        block = make_tool_use_block("search_course_content", {"query": "RAG"}, "tool_err")
        r1 = make_response("tool_use", [block])
        r2 = make_response("end_turn", [make_text_block("Graceful degraded response.")])
        mock_client.messages.create.side_effect = [r1, r2]
        gen = make_generator(mock_client)
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = Exception("DB error")

        result = gen.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert mock_client.messages.create.call_count == 2
        assert result == "Graceful degraded response."
