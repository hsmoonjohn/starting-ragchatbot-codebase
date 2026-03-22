from typing import List, Optional

import anthropic

MAX_TOOL_ROUNDS = 2


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use **search_course_content** for questions about specific course content or detailed educational materials
- Use **get_course_outline** for outline or structure questions (e.g. "what lessons are in X?", "give me the outline of Y"): always return the course title, course link, and every lesson with its number and title — do not summarize or truncate the lesson list
- **You may make up to 2 sequential tool calls** when the first result is insufficient — for example, to look up a lesson title then search for related courses. Only make a second call if the first result does not fully answer the question.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course outline/structure questions**: Use get_course_outline, then present all fields
- **Course content questions**: Use search_course_content, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool calls. Each round appends
        tool results to the conversation and fires a new API call. The last round's
        call is made without tools to force a synthesis response.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        if response.stop_reason != "tool_use" or not tool_manager:
            return response.content[0].text

        for round_idx in range(MAX_TOOL_ROUNDS):
            success = self._execute_tool_round(response, messages, tool_manager)
            if not success:
                break

            is_last_round = round_idx == MAX_TOOL_ROUNDS - 1
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }
            if tools and not is_last_round:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**next_params)

            if response.stop_reason != "tool_use":
                return response.content[0].text

        # Degenerate/error case: force synthesis without tools
        synthesis_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }
        final_response = self.client.messages.create(**synthesis_params)
        return final_response.content[0].text

    def _execute_tool_round(self, response, messages: List, tool_manager) -> bool:
        """
        Execute all tool calls in a response and append results to messages.

        Args:
            response: The Claude response containing tool_use blocks
            messages: Conversation messages list to mutate in place
            tool_manager: Manager to execute tools

        Returns:
            True on success, False if any tool execution raised an exception
        """
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    result = f"Tool execution error: {str(e)}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
                    messages.append({"role": "user", "content": tool_results})
                    return False
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        return True