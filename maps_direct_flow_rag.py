"""
RAG-Enhanced Direct Flow for Google Maps MCP:
User → LLM Tool Call → MCP Tool Result → (on failure) RAG resolves missing event location → LLM regenerates tool call with context → MCP Tool Result → Final Assistant Message
"""

import os
import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_location_resolver import RAGLocationResolver


class GoogleMapsFlowAgentRAG:
    """Agent that implements a 4-step flow with RAG fallback for Google Maps MCP"""

    def __init__(self, rag_files: Optional[List[str]] = None, rag_dir: Optional[str] = None):
        self.llm: Optional[ChatOpenAI] = None
        self.mcp_client = None
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._stdio_context = None
        self._session_context = None
        self._rag = RAGLocationResolver()
        self._rag_files = rag_files or []
        self._rag_dir = rag_dir

    async def setup(self) -> None:
        """Setup LLM, MCP connection, and RAG index."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE", "https://llms-inference.innkube.fim.uni-passau.de"),
            api_key=os.getenv("UNIVERSITY_LLM_API_KEY"),
            model=os.getenv("MAPS_LLM_MODEL", "llama3.1"),
            temperature=0.0,
        )

        # Setup MCP connection for Google Maps
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-google-maps"],
            env={
                "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY", ""),
            },
        )

        # Create persistent connection
        self._stdio_context = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.mcp_client = await self._session_context.__aenter__()

        await self.mcp_client.initialize()

        # Load available tools
        tools_result = await self.mcp_client.list_tools()
        for tool in tools_result.tools:
            self.available_tools[tool.name] = {
                "description": tool.description,
                "schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }

        # Build RAG index (if files/dir provided)
        files: List[str] = []
        files.extend(self._rag_files)
        if self._rag_dir and os.path.isdir(self._rag_dir):
            for name in os.listdir(self._rag_dir):
                path = os.path.join(self._rag_dir, name)
                if os.path.isfile(path) and os.path.splitext(path)[1].lower() in {".pdf", ".txt", ".md", ".markdown"}:
                    files.append(path)
        if files:
            added = await self._rag.add_files(files)
            print(f"📚 RAG indexed {len(files)} files into {added} chunks")
        else:
            print("📚 RAG: no files provided for indexing (fallback still available, but will find nothing)")

        print(f"✅ Setup complete with {len(self.available_tools)} tools (Google Maps)")

    def get_tools_description(self) -> str:
        tools_desc = "Available Google Maps tools:\n"
        for name, info in self.available_tools.items():
            tools_desc += f"- {name}: {info['description']}\n"
        return tools_desc

    async def step1_user_message(self, user_input: str) -> str:
        print(f"📝 STEP 1 - User Message: {user_input}")
        return user_input

    async def step2_llm_tool_call(self, user_input: str, extra_context: Optional[str] = None) -> Dict[str, Any]:
        print("🤖 STEP 2 - LLM generating tool call...")

        base_prompt = f"""You are a Google Maps assistant using MCP tools. 

{self.get_tools_description()}

The user will ask you to do something. You need to decide which tool to call and what parameters to use.

CRITICAL RULE: Set location parameters to null if they refer to event names, ceremonies, expos, or non-permanent venues that likely won't be found in Google Maps API.
- GOOD (set these): "Empire State Building", "Central Park", "Times Square", "Moosach, Munich", "TUM Munich", "Coffee Shop", "Museum"
- BAD (set to null): "TUM Robotics Expo", "Graduation Ceremony", "Student Society Hackathon", "Company Annual Meeting", "Conference Room 101", "Event Venue"

DO NOT PROVIDE DIRECTIONS OR DISTANCE TO A PLACE THE USER DID NOT EXPLICITLY ASK FOR. LEAVE THE PARAMETR OR PARAMETERS AS NULL IF NOT SPECIFIED.

IMPORTANT: Respond with ONLY a valid JSON object. Do not wrap it in markdown code blocks or add any other text.

Respond with a JSON object containing:
- "tool_name": the name of the tool to call
- "parameters": object with the parameters for the tool
- "reasoning": brief explanation of why you chose this tool

If no tool is needed, set "tool_name" to null.

Examples:
- If user asks "geocode 1600 Amphitheatre Pkwy, Mountain View" → {{"tool_name": "geocode", "parameters": {{"address": "1600 Amphitheatre Pkwy, Mountain View"}}, "reasoning": "User wants coordinates for an address"}}
- If user asks "directions from Berlin to Munich" → {{"tool_name": "directions", "parameters": {{"origin": "Berlin", "destination": "Munich"}}, "reasoning": "User wants a route"}}
- If user asks "directions from Moosach to TUM Robotics Expo" → {{"tool_name": "directions", "parameters": {{"origin": "Moosach, Munich", "destination": null}}, "reasoning": "Event name likely not in Google Maps"}}
- If user asks "find coffee near Times Square" → {{"tool_name": "places_search", "parameters": {{"query": "coffee", "location": "Times Square, New York"}}, "reasoning": "User wants nearby places"}}
"""

        if extra_context:
            base_prompt += (
                "\n\nAdditional context you MUST use when building parameters (override ambiguous or unknown locations with the resolved one):\n"
                f"{extra_context}\n"
                "Specifically: if the user's destination refers to an event name, REPLACE it with the 'resolved_location' string provided above when setting 'destination' or 'address' parameters.\n"
            )

        messages = [
            SystemMessage(content=base_prompt),
            HumanMessage(content=user_input),
        ]

        response = await self.llm.ainvoke(messages)

        # Clean any accidental fences
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            tool_call = json.loads(content)
            print(f"🔧 Tool Call: {json.dumps(tool_call, indent=2)}")
            return tool_call
        except json.JSONDecodeError:
            print(f"❌ Failed to parse LLM response as JSON: {response.content}")
            return {"tool_name": None, "parameters": {}, "reasoning": "Failed to parse response"}

    async def step3_tool_execution(self, tool_call: Dict[str, Any]) -> str:
        print("⚡ STEP 3 - Executing tool...")

        tool_name = tool_call.get("tool_name")
        parameters = tool_call.get("parameters", {})

        if not tool_name:
            return "No tool was called"

        if tool_name not in self.available_tools:
            return f"Error: Tool '{tool_name}' not found"

        try:
            result = await self.mcp_client.call_tool(tool_name, parameters)

            if hasattr(result, "content") and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, "text"):
                    tool_result = content.text
                else:
                    tool_result = str(content)
            else:
                tool_result = str(result)

            print(
                f"📤 Tool Result: {tool_result[:2000]}..."
                if isinstance(tool_result, str) and len(tool_result) > 2000
                else f"📤 Tool Result: {tool_result}"
            )
            return tool_result

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    async def step3b_fallback_with_rag(self, user_input: str) -> Optional[str]:
        """Use RAG to resolve an event location and return extra context string for the LLM."""
        print("🛟 STEP 3b - RAG fallback resolving event location...")
        rag_result = await self._rag.resolve_location(user_input)
        location = rag_result.get("address") or rag_result.get("location")
        location = (location or "").strip() or None
        if not location:
            print("ℹ️ RAG could not resolve a usable location from the provided files")
            return None

        extra = (
            "Event location resolved from local documents: "
            + json.dumps(
                {
                    "resolved_location": location,
                    "confidence": rag_result.get("confidence"),
                    "reasoning": rag_result.get("reasoning"),
                },
                ensure_ascii=False,
            )
        )
        print(f"📎 Fallback context: {extra}")
        return extra

    async def step4_final_response(self, user_input: str, tool_call: Dict[str, Any], tool_result: str) -> str:
        print("💬 STEP 4 - Generating final assistant message...")

        system_prompt = """You are a helpful Google Maps assistant.


The user asked you to do something, you called a tool, and got a result.
Now provide a friendly, helpful response to the user explaining what happened and the result.

Be conversational and helpful. Explain the result in plain language."""

        context_prompt = f"""User asked: {user_input}

I called the tool: {tool_call.get('tool_name', 'none')}
With parameters: {json.dumps(tool_call.get('parameters', {}), indent=2)}
Reasoning: {tool_call.get('reasoning', 'none')}

Tool result: {tool_result}

Please provide a helpful response to the user explaining what happened and the result."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        final_message = response.content

        print(f"✨ Final Assistant Message: {final_message}")
        return final_message

    @staticmethod
    def _detect_error_from_response_text(text: str) -> Optional[str]:
        if not text:
            return "Empty response"
        lower = text.lower()
        keyword_pairs = [
            ("failed", "Response indicates failure"),
            ("error", "Response contains an error"),
            ("not authorized", "Not authorized"),
            ("unauthorized", "Unauthorized"),
            ("permission denied", "Permission denied"),
            ("forbidden", "Forbidden"),
            ("request denied", "Request denied"),
            ("over query limit", "Over query limit"),
            ("quota", "Quota issue"),
            ("invalid request", "Invalid request"),
            ("api key", "API key issue"),
        ]
        for kw, reason in keyword_pairs:
            if kw in lower:
                return reason
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                status = payload.get("status")
                if isinstance(status, str) and status.upper() not in ("OK", "ZERO_RESULTS"):
                    return f"Status not OK: {status}"
                if payload.get("error"):
                    return "Error field present"
                if payload.get("error_message"):
                    return f"API error: {payload.get('error_message')}"
                result = payload.get("result")
                if isinstance(result, dict):
                    r_status = result.get("status")
                    if isinstance(r_status, str) and r_status.upper() not in ("OK", "ZERO_RESULTS"):
                        return f"Result status not OK: {r_status}"
                
                # Check Google Distance Matrix API response structure
                results = payload.get("results")
                if isinstance(results, list):
                    for result_item in results:
                        if isinstance(result_item, dict):
                            elements = result_item.get("elements")
                            if isinstance(elements, list):
                                for element in elements:
                                    if isinstance(element, dict):
                                        elem_status = element.get("status")
                                        if isinstance(elem_status, str) and elem_status.upper() not in ("OK", "ZERO_RESULTS"):
                                            return f"Element status not OK: {elem_status}"
        except Exception:
            pass
        return None

    async def process_user_request(self, user_input: str) -> str:
        print("=" * 60)
        print("🚀 Starting Google Maps RAG-Enhanced Flow")
        print("=" * 60)

        # Step 1: User message
        await self.step1_user_message(user_input)

        # Step 2: LLM generates tool call
        tool_call = await self.step2_llm_tool_call(user_input)

        # Step 3: Execute tool
        tool_result = await self.step3_tool_execution(tool_call)

        # Heuristic error detection
        error_reason = self._detect_error_from_response_text(tool_result)
        if error_reason:
            print(f"⚠️ Detected issue with tool result: {error_reason}")

            # Step 3b: RAG fallback to resolve missing event location
            extra_context = await self.step3b_fallback_with_rag(user_input)

            if extra_context:
                # Retry Step 2 with appended context
                tool_call = await self.step2_llm_tool_call(user_input, extra_context=extra_context)
                # Retry Step 3
                tool_result = await self.step3_tool_execution(tool_call)
            else:
                print("ℹ️ Skipping retry: no extra context from RAG")

        # Step 4: Final assistant message
        final_response = await self.step4_final_response(user_input, tool_call, tool_result)

        print("=" * 60)
        print("✅ Flow Complete")
        print("=" * 60)

        return final_response

    async def cleanup(self) -> None:
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)


async def main() -> None:
    print("🚀 Google Maps RAG-Enhanced Flow Agent")
    print("=" * 40)

    # Check credentials
    missing = []
    if not os.getenv("GOOGLE_MAPS_API_KEY"):
        missing.append("GOOGLE_MAPS_API_KEY")
    if not os.getenv("UNIVERSITY_LLM_API_KEY"):
        missing.append("UNIVERSITY_LLM_API_KEY")

    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        return

    # Defaults to scan PDFs/ for local documents if present
    rag_dir = os.getenv("RAG_DOCS_DIR", "PDFs")

    agent = GoogleMapsFlowAgentRAG(rag_files=None, rag_dir=rag_dir)

    try:
        await agent.setup()

        # Example request showing fallback scenario
        example = "How far is the Student Society Haclathon form TUM University"
        await agent.process_user_request(example)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        await agent.cleanup()
        print("👋 Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main()) 