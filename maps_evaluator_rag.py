"""
Google Maps MCP RAG Evaluator: Batch queries → LLM Tool Call → MCP Result
On failure: RAG resolves event location → LLM regenerates tool call with context → MCP Result
- No final assistant LLM response generation
- Saves per-query results with: query, tool_call, tool_response, success, error, retry fields
"""

import os
import sys
import json
import argparse
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_location_resolver import RAGLocationResolver


class GoogleMapsRAGEvaluator:
    """Evaluator that connects to Google Maps MCP, with RAG fallback for event locations."""

    def __init__(self, rag_files: Optional[List[str]] = None, rag_dir: Optional[str] = None) -> None:
        self.llm: Optional[ChatOpenAI] = None
        self.mcp_client = None
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._stdio_context = None
        self._session_context = None
        self._rag = RAGLocationResolver()
        self._rag_files = rag_files or []
        self._rag_dir = rag_dir

    async def setup(self) -> None:
        """Setup LLM and MCP connection (Google Maps), and build RAG index."""
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

        print(f"✅ Evaluator setup complete with {len(self.available_tools)} tools (Google Maps)")

    def _tools_description(self) -> str:
        tools_desc = "Available Google Maps tools:\n"
        for name, info in self.available_tools.items():
            tools_desc += f"- {name}: {info['description']}\n"
        return tools_desc

    async def generate_tool_call(self, user_input: str, extra_context: Optional[str] = None) -> Dict[str, Any]:
        """Use the LLM to produce a tool call JSON for a given query."""
        system_prompt = f"""You are a Google Maps assistant using MCP tools. 

{self._tools_description()}

The user will ask you to do something. You need to decide which tool to call and what parameters to use.

DO NOT PROVIDE DIRECTIONS TO A PLACE THE USER DID NOT EXPLICITLY ASK FOR. LEAVE THE PARAMETRS AS NULL IF NOT SPECIFIED.

IMPORTANT: Respond with ONLY a valid JSON object. Do not wrap it in markdown code blocks or add any other text.

Respond with a JSON object containing:
- "tool_name": the name of the tool to call
- "parameters": object with the parameters for the tool
- "reasoning": brief explanation of why you chose this tool

If no tool is needed, set "tool_name" to null.

Examples:
- If user asks "geocode 1600 Amphitheatre Pkwy, Mountain View" → {{"tool_name": "geocode", "parameters": {{"address": "1600 Amphitheatre Pkwy, Mountain View"}}, "reasoning": "User wants coordinates for an address"}}
- If user asks "directions from Berlin to Munich" → {{"tool_name": "directions", "parameters": {{"origin": "Berlin", "destination": "Munich"}}, "reasoning": "User wants a route"}}
- If user asks "find coffee near Times Square" → {{"tool_name": "places_search", "parameters": {{"query": "coffee", "location": "Times Square, New York"}}, "reasoning": "User wants nearby places"}}
"""

        if extra_context:
            system_prompt += (
                "\n\nAdditional context you MUST use when building parameters (override ambiguous or unknown locations with the resolved one):\n"
                f"{extra_context}\n"
                "Specifically: if the user's destination refers to an event name, REPLACE it with the 'resolved_location' string provided above when setting 'destination' or 'address' parameters.\n"
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        response = await self.llm.ainvoke(messages)

        # Clean the response content - remove markdown code blocks if present
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            tool_call = json.loads(content)
        except json.JSONDecodeError:
            print(f"❌ Failed to parse LLM response as JSON: {response.content}")
            tool_call = {"tool_name": None, "parameters": {}, "reasoning": "Failed to parse response"}

        print(f"🔧 Tool Call: {json.dumps(tool_call, indent=2)}")
        return tool_call

    async def execute_tool(self, tool_call: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """Execute the tool and capture success flag, response, and error string."""
        tool_name = tool_call.get("tool_name")
        parameters = tool_call.get("parameters", {})

        if not tool_name:
            return False, "No tool was called", None

        if tool_name not in self.available_tools:
            return False, f"Error: Tool '{tool_name}' not found", None

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

            return True, tool_result, None
        except Exception as e:
            return False, "", f"Error executing {tool_name}: {str(e)}"

    async def cleanup(self) -> None:
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)


def _detect_error_from_response_text(text: str) -> Optional[str]:
    """Heuristically detect API/tool errors from response text or JSON payloads."""
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

    # Attempt to parse JSON and inspect common fields
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            # Google APIs often provide status and error_message
            status = payload.get("status")
            if isinstance(status, str) and status.upper() not in ("OK", "ZERO_RESULTS"):
                return f"Status not OK: {status}"
            if payload.get("error"):
                return "Error field present"
            if payload.get("error_message"):
                return f"API error: {payload.get('error_message')}"
            # Some responses may wrap results in a top-level 'result' with 'status'
            result = payload.get("result")
            if isinstance(result, dict):
                r_status = result.get("status")
                if isinstance(r_status, str) and r_status.upper() not in ("OK", "ZERO_RESULTS"):
                    return f"Result status not OK: {r_status}"
    except Exception:
        # Not JSON or unexpected structure; ignore
        pass

    return None


def read_queries(args: argparse.Namespace) -> List[str]:
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
            return queries
    if args.query:
        return args.query
    # Default: load from maps_queries file if present
    default_file = os.path.join(os.path.dirname(__file__), "Dataset/RAG-Location/rag_queries.txt")
    if os.path.exists(default_file):
        with open(default_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback samples
    return [
        "Geocode 1600 Amphitheatre Parkway, Mountain View",
        "Directions from Berlin to Munich",
        "Find coffee near Times Square",
    ]


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def run_evaluation(args: argparse.Namespace) -> int:
    missing = []
    if not os.getenv("GOOGLE_MAPS_API_KEY"):
        missing.append("GOOGLE_MAPS_API_KEY")
    if not os.getenv("UNIVERSITY_LLM_API_KEY"):
        missing.append("UNIVERSITY_LLM_API_KEY")

    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        return 2

    rag_files = args.rag_file or []
    rag_dir = args.rag_dir

    queries = read_queries(args)
    evaluator = GoogleMapsRAGEvaluator(rag_files=rag_files, rag_dir=rag_dir)

    try:
        await evaluator.setup()

        results: List[Dict[str, Any]] = []
        total = len(queries)
        print(f"🧪 Running evaluation on {total} queries...")

        for idx, q in enumerate(queries, start=1):
            print("=" * 60)
            print(f"▶️  [{idx}/{total}] Query: {q}")

            # Initial tool call and execution
            tool_call = await evaluator.generate_tool_call(q)
            success, tool_response, error = await evaluator.execute_tool(tool_call)

            detected_error: Optional[str] = None
            if success and not error:
                detected_error = _detect_error_from_response_text(tool_response)
                if detected_error:
                    print(f"⚠️  Heuristic error detected: {detected_error}")
                    success = False
                    error = detected_error

            # RAG fallback if failed
            rag_used = False
            retry_tool_call: Optional[Dict[str, Any]] = None
            retry_success: Optional[bool] = None
            retry_tool_response: Optional[str] = None
            retry_error: Optional[str] = None
            rag_extra_context: Optional[str] = None
            rag_resolved_location: Optional[str] = None

            if not success:
                rag_used = True
                print("🛟 Attempting RAG fallback to resolve event location...")
                rag_result = await evaluator._rag.resolve_location(q)
                rag_resolved_location = rag_result.get("address") or rag_result.get("location")
                rag_resolved_location = (rag_resolved_location or "").strip() or None

                if rag_resolved_location:
                    rag_extra_context = (
                        "Event location resolved from local documents: "
                        + json.dumps(
                            {
                                "resolved_location": rag_resolved_location,
                                "confidence": rag_result.get("confidence"),
                                "reasoning": rag_result.get("reasoning"),
                            },
                            ensure_ascii=False,
                        )
                    )
                    # Regenerate tool call with extra context
                    retry_tool_call = await evaluator.generate_tool_call(q, extra_context=rag_extra_context)
                    retry_success, retry_tool_response, retry_error = await evaluator.execute_tool(retry_tool_call)

                    # Heuristic check on retry
                    if retry_success and not retry_error:
                        rec_detected_error = _detect_error_from_response_text(retry_tool_response)
                        if rec_detected_error:
                            print(f"⚠️  Heuristic error detected after RAG retry: {rec_detected_error}")
                            retry_success = False
                            retry_error = rec_detected_error
                else:
                    print("ℹ️ RAG could not resolve a usable location; skipping retry")

            final_success = success or bool(retry_success)

            record: Dict[str, Any] = {
                "query": q,
                "tool_call": tool_call,
                "tool_response": tool_response,
                "success": success,
                "error": error,
                "detected_error": detected_error,
                "rag_used": rag_used,
                "rag_resolved_location": rag_resolved_location,
                "rag_extra_context": rag_extra_context,
                "retry_tool_call": retry_tool_call,
                "retry_tool_response": retry_tool_response,
                "retry_success": retry_success,
                "retry_error": retry_error,
                "final_success": final_success,
            }
            results.append(record)

            # Stream write each record if requested
            if args.stream:
                write_jsonl(args.output, [record])

        # Write all results at once unless we streamed
        if not args.stream:
            write_jsonl(args.output, results)

        # Print brief summary
        initial_succeeded = sum(1 for r in results if r["success"])
        retries_attempted = sum(1 for r in results if r.get("rag_used"))
        recovered = sum(1 for r in results if r.get("retry_success"))
        final_succeeded = sum(1 for r in results if r.get("final_success", r["success"]))
        failed_final = total - final_succeeded
        print("=" * 60)
        print(f"✅ Completed. Initial success: {initial_succeeded}, Recovered by RAG: {recovered}/{retries_attempted}, Final success: {final_succeeded}, Final failures: {failed_final}. Results → {args.output}")
        return 0
    finally:
        await evaluator.cleanup()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Google Maps MCP tool-calling with RAG fallback across a set of queries.")
    p.add_argument("--queries-file", type=str, default=None, help="Path to a text file with one query per line.")
    p.add_argument("--query", action="append", help="Specify a single query (can be repeated).")
    p.add_argument("--output", type=str, default="maps_eval_results_rag.jsonl", help="Output JSONL file path.")
    p.add_argument("--stream", action="store_true", help="Write each record to output as it completes instead of batching.")
    p.add_argument("--rag-dir", type=str, default=os.getenv("RAG_DOCS_DIR", "PDFs"), help="Directory to scan for RAG docs.")
    p.add_argument("--rag-file", action="append", default=[], help="Specific RAG file path (repeatable).")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(run_evaluation(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 