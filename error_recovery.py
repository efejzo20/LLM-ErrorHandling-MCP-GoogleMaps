import json
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class ToolCallRecovery:
    """Generate a corrected tool call using LLM with error context."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    @staticmethod
    def _tools_description(available_tools: Dict[str, Dict[str, Any]]) -> str:
        tools_desc = "Available tools you may call (choose one):\n"
        for name, info in available_tools.items():
            desc = info.get("description", "")
            tools_desc += f"- {name}: {desc}\n"
        return tools_desc

    @staticmethod
    def _is_parameter_format_error(error_message: str, last_response: Optional[str]) -> bool:
        """Detect if error is due to parameter formatting."""
        text = f"{error_message} {last_response or ''}"
        error_patterns = [
            "is not a function",
            "join is not a function",
            "expected array",
            "must be an array",
        ]
        return any(pattern in text.lower() for pattern in error_patterns)

    async def generate_recovery_tool_call(
        self,
        user_input: str,
        error_message: str,
        previous_tool_call: Dict[str, Any],
        available_tools: Dict[str, Dict[str, Any]],
        last_response_text: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ask the LLM to propose a corrected tool call given the previous failure.

        Returns a JSON-like dict with: {"tool_name", "parameters", "reasoning"}.
        """
        previous_json = json.dumps(previous_tool_call, ensure_ascii=False)
        last_response_text = last_response_text or ""
        valid_tool_names = ", ".join(sorted(available_tools.keys()))

        system_prompt = (
            "You are a Google Maps assistant that must call exactly one MCP tool.\n\n"
            + self._tools_description(available_tools)
            
            + "\nThe previous tool call failed. You must propose a corrected tool call.\n"
            "Rules:\n"
            "- Respond with ONLY a valid JSON object (no markdown).\n"
            "- JSON keys: 'tool_name' (string or null), 'parameters' (object), 'reasoning' (string).\n"
            "- Choose a tool that exists in the list. If none applies, set tool_name to null.\n"
            "\nCRITICAL RULE: Set location parameters to null if they refer to event names, ceremonies, expos, or non-permanent venues that likely won't be found in Google Maps API.\n"
            "- GOOD (set these): \"Empire State Building\", \"Central Park\", \"Times Square\", \"Moosach, Munich\", \"TUM Munich\", \"Coffee Shop\", \"Museum\"\n"
            "- BAD (set to null): \"TUM Robotics Expo\", \"Graduation Ceremony\", \"Student Society Hackathon\", \"Company Annual Meeting\", \"Conference Room 101\", \"Event Venue\"\n"
        )

        # Check if this is a parameter format error
        is_format_error = self._is_parameter_format_error(error_message, last_response_text)
        
        format_hint = ""
        if is_format_error:
            format_hint = (
                "\n🚨 PARAMETER FORMAT ERROR DETECTED:\n"
                "The error indicates a parameter type mismatch. Common fixes:\n"
                "- If error mentions 'join is not a function': Parameter expects an ARRAY, not a string\n"
                "  Example: Change \"origins\": \"Place\" to \"origins\": [\"Place\"]\n"
                "- If tool expects multiple values: Wrap string values in square brackets\n"
                "- For maps_distance_matrix: Both 'origins' and 'destinations' MUST be arrays\n\n"
            )
        
        extra_block = (
            f"\nAdditional context you MUST use when building parameters:\n{extra_context}\n"
            "If 'resolved_location' is present, REPLACE any event-name destination with that string when setting 'destination' or 'address' parameters.\n"
            if extra_context
            else ""
        )

        human_prompt = (
            f"{format_hint}"
            "User request:\n"
            f"{user_input}\n\n"
            "Previous tool call (failed):\n"
            f"{previous_json}\n\n"
            "API error message:\n"
            f"{error_message}\n\n"
            f"Last response (if any):\n{last_response_text}\n"
            f"{extra_block}"
            "\nNow return ONLY the corrected JSON tool call."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = await self.llm.ainvoke(messages)

        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            tool_call = json.loads(content)
        except json.JSONDecodeError:
            tool_call = {
                "tool_name": None,
                "parameters": {},
                "reasoning": "Failed to parse recovery response",
            }

        return tool_call 