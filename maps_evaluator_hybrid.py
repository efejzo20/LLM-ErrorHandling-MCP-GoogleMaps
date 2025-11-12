"""
Google Maps MCP Hybrid Evaluator: Batch queries → LLM Tool Call → MCP Result
Routing on failure:
- If tool result indicates NOT_FOUND → use RAG to resolve event location, re-generate tool call, retry.
- Else (other API errors) → use error-aware recovery LLM to refine the tool call, retry.
- If the first retry still fails and changes category (e.g., type error ↔ NOT_FOUND), try the other route once more.
- Skips final assistant response generation.
- Logs full per-query details and summary.
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
from error_recovery import ToolCallRecovery


class GoogleMapsHybridEvaluator:
    def __init__(self, rag_files: Optional[List[str]] = None, rag_dir: Optional[str] = None) -> None:
        self.llm: Optional[ChatOpenAI] = None
        self.mcp_client = None
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._stdio_context = None
        self._session_context = None
        self._rag = RAGLocationResolver()
        self._recovery: Optional[ToolCallRecovery] = None
        self._rag_files = rag_files or []
        self._rag_dir = rag_dir

    async def setup(self) -> None:
        # Initialize LLM
        self.llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE", "https://llms-inference.innkube.fim.uni-passau.de"),
            api_key=os.getenv("UNIVERSITY_LLM_API_KEY"),
            model=os.getenv("MAPS_LLM_MODEL", "llama3.1"),
            temperature=0.0,
        )
        self._recovery = ToolCallRecovery(self.llm)

        # Setup MCP
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-google-maps"],
            env={
                "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY", ""),
            },
        )
        self._stdio_context = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self.mcp_client = await self._session_context.__aenter__()
        await self.mcp_client.initialize()

        tools_result = await self.mcp_client.list_tools()
        for tool in tools_result.tools:
            self.available_tools[tool.name] = {
                "description": tool.description,
                "schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }

        # Build RAG index
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

        print(f"✅ Hybrid evaluator setup with {len(self.available_tools)} tools")

    async def cleanup(self) -> None:
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_context = None
        if self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_context = None

    def _tools_description(self) -> str:
        tools_desc = "Available Google Maps tools:\n"
        for name, info in self.available_tools.items():
            tools_desc += f"- {name}: {info['description']}\n"
            schema = info.get('schema', {})
            if schema and isinstance(schema, dict):
                props = schema.get('properties', {})
                if props:
                    tools_desc += f"  Parameters: {', '.join(props.keys())}\n"
                    # Add special notes for array parameters
                    for param_name, param_info in props.items():
                        if isinstance(param_info, dict):
                            param_type = param_info.get('type', '')
                            if param_type == 'array':
                                tools_desc += f"  ⚠️  {param_name} MUST be an array (e.g., [\"{param_name}_value\"])\n"
        return tools_desc

    def _detect_event_keywords(self, user_input: str) -> bool:
        """Detect if the query contains event-related keywords that should trigger RAG fallback."""
        event_keywords = [
            # Event types
            "hackathon", "conference", "fair", "expo", "festival", "summit", "show",
            "competition", "contest", "tournament",
            # Ceremonies
            "graduation", "ceremony", "celebration", "awards",
            # Temporary gatherings  
            "meeting", "workshop", "seminar", "bootcamp", "symposium",
            # Cultural events
            "night" , "gala", "reception", "party",
            # Other
            "event", "exhibition", "demonstration"
        ]
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in event_keywords)

    async def generate_tool_call(self, user_input: str, extra_context: Optional[str] = None, prev_tool_call: Optional[Dict[str, Any]] = None, prev_error: Optional[str] = None, prev_response: Optional[str] = None) -> Dict[str, Any]:
        valid_tool_names = ", ".join(sorted(self.available_tools.keys()))
        # Check if this is an event-related query
        is_event_query = self._detect_event_keywords(user_input)
        
        system_prompt = f"""You are a Google Maps assistant using MCP tools. 

{self._tools_description()}

The user will ask you to do something. You need to decide which tool to call and what parameters to use.

CRITICAL RULE FOR EVENT DETECTION:
You MUST set location parameters to null if they contain ANY of these patterns:
- Event names: "Hackathon", "Conference", "Fair", "Expo", "Festival", "Summit", "Show", "Competition"
- Ceremonies: "Graduation", "Award Ceremony", "Opening Ceremony", "Celebration"
- Temporary gatherings: "Meeting", "Workshop", "Seminar", "Bootcamp", "Night" (as in Culture Night)
- Generic venues: "Event Venue", "Conference Room", "Meeting Room"

DO NOT PROVIDE DIRECTIONS TO A PLACE THE USER DID NOT EXPLICITLY ASK FOR. LEAVE THE PARAMETRS AS NULL IF NOT SPECIFIED.

SPECIAL CASE - DISTANCE/MATRIX QUERIES WITH EVENTS:
- For maps_directions with ONE event: Set event parameter to null
  Example: "directions to the Hackathon" → destination: null
- For maps_distance_matrix with events: You CANNOT use null in arrays
  Instead: Use the full event string and let API fail naturally
  Example: "distance between Tech Conference and Airport" → origins: ["Tech Conference"], destinations: ["Airport"]
  (This allows RAG to resolve it later)

IMPORTANT: Respond with ONLY a valid JSON object. Do not wrap it in markdown code blocks or add any other text.

Respond with a JSON object containing:
- "tool_name": the name of the tool to call
- "parameters": object with the parameters for the tool
- "reasoning": brief explanation of why you chose this tool

If no tool is needed, set "tool_name" to null.
"""
        
        # Add event detection warning if needed
        if is_event_query:
            system_prompt += (
                "\n\n EVENT DETECTED IN QUERY!\n"
                "This query contains event keywords (hackathon, conference, fair, expo, festival, ceremony, etc.).\n"
                "YOU MUST SET EVENT LOCATION PARAMETERS TO NULL!\n"
                "Do NOT use the event name as a string - Google Maps will give false results.\n"
                "Set it to null so the RAG system can find the real location from documents."
            )
        context_lines: List[str] = []
        if extra_context:
            context_lines.append("CRITICAL OVERRIDE - RAG CONTEXT PROVIDED:")
            context_lines.append(extra_context)
            context_lines.append("")
            context_lines.append("MANDATORY INSTRUCTIONS when RAG context is present:")
            context_lines.append("1. The 'resolved_location' is a REAL, PHYSICAL ADDRESS that Google Maps WILL recognize")
            context_lines.append("2. You MUST use this resolved_location to REPLACE the event name in your parameters")
            context_lines.append("3. Do NOT set any parameter to null when resolved_location is provided")
            context_lines.append("4. Identify which parameter (origin, destination, origins, destinations, address) had the event name")
            context_lines.append("5. Replace that event name with the resolved_location value")
            context_lines.append("6. For arrays (origins/destinations), use [\"resolved_location\"] not [null]")
            context_lines.append("7. Generate a valid tool call with the resolved_location properly substituted")
        if prev_tool_call or prev_error or prev_response:
            context_lines.append("Previous attempt context:")
            if prev_tool_call:
                context_lines.append("- Previous tool call: " + json.dumps(prev_tool_call, ensure_ascii=False))
            if prev_error:
                context_lines.append("- Previous error: " + prev_error)
            if prev_response:
                context_lines.append("- Previous response snippet: " + (prev_response[:500] + ("…" if len(prev_response) > 500 else "")))
        if context_lines:
            system_prompt += "\n\n" + "\n".join(context_lines) + "\n"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        response = await self.llm.ainvoke(messages)
        content = response.content.strip()
        
        # Clean up markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Try to parse JSON with better error handling
        try:
            tool_call = json.loads(content)
            # Validate required fields
            if not isinstance(tool_call, dict):
                raise ValueError("Response is not a JSON object")
            if "tool_name" not in tool_call:
                tool_call["tool_name"] = None
            if "parameters" not in tool_call:
                tool_call["parameters"] = {}
            if "reasoning" not in tool_call:
                tool_call["reasoning"] = "No reasoning provided"
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Failed to parse LLM response as JSON: {response.content}")
            print(f"❌ Error: {e}")
            # Try to extract tool_name from content as fallback
            tool_name = None
            if '"tool_name"' in content:
                try:
                    # Simple regex to extract tool_name value
                    import re
                    match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', content)
                    if match:
                        tool_name = match.group(1)
                except:
                    pass
            tool_call = {
                "tool_name": tool_name, 
                "parameters": {}, 
                "reasoning": f"Failed to parse response: {str(e)}"
            }
        print(f"🔧 Tool Call: {json.dumps(tool_call, indent=2)}")
        return tool_call

    async def execute_tool(self, tool_call: Optional[Dict[str, Any]]) -> Tuple[bool, str, Optional[str]]:
        if not tool_call:
            return False, "No tool call provided", None
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

    @staticmethod
    def _is_parameter_format_error(text: str) -> bool:
        """Detect if the error is due to parameter formatting issues."""
        if not text:
            return False
        error_patterns = [
            "is not a function",  # JavaScript type errors
            "join is not a function",  # Array expected but got string
            "map is not a function",
            "filter is not a function",
            "expected array",
            "must be an array",
            "invalid type",
            "type error",
        ]
        lower = text.lower()
        return any(pattern in lower for pattern in error_patterns)

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

    @staticmethod
    def _is_not_found(text: str) -> bool:
        if not text:
            return False
        return "not_found" in text.lower() or "not found" in text.lower()


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
    evaluator = GoogleMapsHybridEvaluator(rag_files=rag_files, rag_dir=rag_dir)

    try:
        await evaluator.setup()

        results: List[Dict[str, Any]] = []
        total = len(queries)
        print(f"🧪 Running evaluation on {total} queries...")

        for idx, q in enumerate(queries, start=1):
            print("=" * 60)
            print(f"▶️  [{idx}/{total}] Query: {q}")

            # Initial attempt
            tool_call = await evaluator.generate_tool_call(q)
            success, tool_response, error = await evaluator.execute_tool(tool_call)

            detected_error: Optional[str] = None
            if success and not error:
                detected_error = evaluator._detect_error_from_response_text(tool_response)
                if detected_error:
                    print(f"⚠️  Heuristic error detected: {detected_error}")
                    success = False
                    error = detected_error

            # Route failures (first retry)
            route = None
            retry_tool_call: Optional[Dict[str, Any]] = None
            retry_success: Optional[bool] = None
            retry_tool_response: Optional[str] = None
            retry_error: Optional[str] = None
            rag_resolved_location: Optional[str] = None
            rag_extra_context: Optional[str] = None

            # Second retry fields (alternate route)
            second_route: Optional[str] = None
            second_retry_tool_call: Optional[Dict[str, Any]] = None
            second_retry_success: Optional[bool] = None
            second_retry_tool_response: Optional[str] = None
            second_retry_error: Optional[str] = None
            second_rag_resolved_location: Optional[str] = None
            second_rag_extra_context: Optional[str] = None

            if not success:
                # Check if this is an event query that should go to RAG first
                is_event_query = evaluator._detect_event_keywords(q)
                should_use_rag = (
                    is_event_query or 
                    evaluator._is_not_found(tool_response or "") or 
                    (error and "not found" in error.lower())
                )
                
                if should_use_rag:
                    route = "RAG"
                    print("🛟 Routing to RAG fallback (Event query or NOT_FOUND)...")
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
                        retry_tool_call = await evaluator.generate_tool_call(
                            q,
                            extra_context=rag_extra_context,
                            prev_tool_call=tool_call,
                            prev_error=error,
                            prev_response=tool_response,
                        )
                        # Validate that RAG context was used properly
                        if retry_tool_call:
                            retry_params = retry_tool_call.get("parameters") or {}
                            retry_dest = retry_params.get("destination") if isinstance(retry_params, dict) else None
                            if retry_dest == rag_resolved_location:
                                print(f"✅ RAG location properly integrated: {rag_resolved_location}")
                            else:
                                print(f"⚠️  RAG location may not have been used properly in retry")
                        else:
                            print(f"❌ Failed to generate retry tool call")
                        if retry_tool_call:
                            retry_success, retry_tool_response, retry_error = await evaluator.execute_tool(retry_tool_call)
                        else:
                            retry_success, retry_tool_response, retry_error = False, "No tool call generated", None
                        if retry_success and not retry_error:
                            rec_detected_error = evaluator._detect_error_from_response_text(retry_tool_response)
                            if rec_detected_error:
                                print(f"⚠️  Heuristic error detected after RAG retry: {rec_detected_error}")
                                retry_success = False
                                retry_error = rec_detected_error
                    else:
                        print("ℹ️ RAG could not resolve a usable location; skipping retry")
                else:
                    route = "RECOVERY"
                    print("🔁 Routing to error-aware recovery...")
                    retry_tool_call = await evaluator._recovery.generate_recovery_tool_call(
                        user_input=q,
                        error_message=error or "Unknown error",
                        previous_tool_call=tool_call,
                        available_tools=evaluator.available_tools,
                        last_response_text=tool_response or None,
                        extra_context=None,
                    )
                    if retry_tool_call:
                        retry_success, retry_tool_response, retry_error = await evaluator.execute_tool(retry_tool_call)
                    else:
                        retry_success, retry_tool_response, retry_error = False, "No tool call generated", None
                    if retry_success and not retry_error:
                        rec_detected_error = evaluator._detect_error_from_response_text(retry_tool_response)
                        if rec_detected_error:
                            print(f"⚠️  Heuristic error detected after recovery: {rec_detected_error}")
                            retry_success = False
                            retry_error = rec_detected_error

            # If first retry failed, attempt alternate route once
            if not (retry_success or False):
                retry_detected_error = None
                if retry_tool_response and not retry_error:
                    retry_detected_error = evaluator._detect_error_from_response_text(retry_tool_response)
                    if retry_detected_error:
                        retry_error = retry_detected_error

                if route == "RAG":
                    second_route = "RECOVERY"
                    print("🔁 Second attempt with error-aware recovery...")
                    second_retry_tool_call = await evaluator._recovery.generate_recovery_tool_call(
                        user_input=q,
                        error_message=retry_error or error or "Unknown error",
                        previous_tool_call=retry_tool_call or tool_call,
                        available_tools=evaluator.available_tools,
                        last_response_text=retry_tool_response or tool_response or None,
                        extra_context=rag_extra_context,
                    )
                    if second_retry_tool_call:
                        second_retry_success, second_retry_tool_response, second_retry_error = await evaluator.execute_tool(second_retry_tool_call)
                    else:
                        second_retry_success, second_retry_tool_response, second_retry_error = False, "No tool call generated", None
                    if second_retry_success and not second_retry_error:
                        rec2_detected_error = evaluator._detect_error_from_response_text(second_retry_tool_response)
                        if rec2_detected_error:
                            print(f"⚠️  Heuristic error detected after second recovery: {rec2_detected_error}")
                            second_retry_success = False
                            second_retry_error = rec2_detected_error
                elif route == "RECOVERY":
                    second_route = "RAG"
                    print("🛟 Second attempt with RAG fallback...")
                    rag2 = await evaluator._rag.resolve_location(q)
                    second_rag_resolved_location = rag2.get("address") or rag2.get("location")
                    second_rag_resolved_location = (second_rag_resolved_location or "").strip() or None
                    if second_rag_resolved_location:
                        second_rag_extra_context = (
                            "Event location resolved from local documents: "
                            + json.dumps(
                                {
                                    "resolved_location": second_rag_resolved_location,
                                    "confidence": rag2.get("confidence"),
                                    "reasoning": rag2.get("reasoning"),
                                },
                                ensure_ascii=False,
                            )
                        )
                        second_retry_tool_call = await evaluator.generate_tool_call(
                            q,
                            extra_context=second_rag_extra_context,
                            prev_tool_call=retry_tool_call or tool_call,
                            prev_error=retry_error or error,
                            prev_response=retry_tool_response or tool_response,
                        )
                        if second_retry_tool_call:
                            second_retry_success, second_retry_tool_response, second_retry_error = await evaluator.execute_tool(second_retry_tool_call)
                        else:
                            second_retry_success, second_retry_tool_response, second_retry_error = False, "No tool call generated", None
                        if second_retry_success and not second_retry_error:
                            rec2_detected_error = evaluator._detect_error_from_response_text(second_retry_tool_response)
                            if rec2_detected_error:
                                print(f"⚠️  Heuristic error detected after second RAG retry: {rec2_detected_error}")
                                second_retry_success = False
                                second_retry_error = rec2_detected_error
                    else:
                        print("ℹ️ Second RAG attempt could not resolve a usable location; skipping")

            final_success = success or bool(retry_success) or bool(second_retry_success)

            record: Dict[str, Any] = {
                "query": q,
                "tool_call": tool_call,
                "tool_response": tool_response,
                "success": success,
                "error": error,
                "detected_error": detected_error,
                "route": route,
                "rag_resolved_location": rag_resolved_location,
                "rag_extra_context": rag_extra_context,
                "retry_tool_call": retry_tool_call,
                "retry_tool_response": retry_tool_response,
                "retry_success": retry_success,
                "retry_error": retry_error,
                "second_route": second_route,
                "second_rag_resolved_location": second_rag_resolved_location,
                "second_rag_extra_context": second_rag_extra_context,
                "second_retry_tool_call": second_retry_tool_call,
                "second_retry_tool_response": second_retry_tool_response,
                "second_retry_success": second_retry_success,
                "second_retry_error": second_retry_error,
                "final_success": final_success,
            }
            results.append(record)

            if args.stream:
                write_jsonl(args.output, [record])

        if not args.stream:
            write_jsonl(args.output, results)

        initial_succeeded = sum(1 for r in results if r["success"])
        routed = sum(1 for r in results if r.get("route") is not None)
        rag_recovered = sum(1 for r in results if (r.get("route") == "RAG" and r.get("retry_success")) or (r.get("second_route") == "RAG" and r.get("second_retry_success")))
        rec_recovered = sum(1 for r in results if (r.get("route") == "RECOVERY" and r.get("retry_success")) or (r.get("second_route") == "RECOVERY" and r.get("second_retry_success")))
        final_succeeded = sum(1 for r in results if r.get("final_success", r["success"]))
        failed_final = len(results) - final_succeeded
        print("=" * 60)
        print(
            f"✅ Completed. Initial success: {initial_succeeded}, Routed: {routed}, "
            f"Recovered by RAG: {rag_recovered}, Recovered by Recovery: {rec_recovered}, "
            f"Final success: {final_succeeded}, Final failures: {failed_final}. Results → {args.output}"
        )
        return 0
    finally:
        await evaluator.cleanup()


def read_queries(args: argparse.Namespace) -> List[str]:
    # First check if CSV is provided
    if args.queries_csv:
        import csv
        queries = []
        with open(args.queries_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query = row.get("query", "").strip()
                if query:
                    queries.append(query)
        return queries
    
    # Fall back to text file
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.query:
        return args.query
    
    # Default to CSV if it exists
    default_csv = os.path.join(
        os.path.dirname(__file__),
        "Dataset",
        "RAG-Location",
        "rag_expected_locations.csv",
    )
    if os.path.exists(default_csv):
        import csv
        queries = []
        with open(default_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query = row.get("query", "").strip()
                if query:
                    queries.append(query)
        if queries:
            return queries
    
    # Last resort default
    return [
        "Get directions from Berlin Main station to the building where graduation is held",
        "Geocode 1600 Amphitheatre Parkway, Mountain View",
    ]


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid evaluator with RAG fallback and error-aware recovery.")
    p.add_argument("--queries-csv", type=str, default=None, help="Path to CSV file with queries (takes precedence over --queries-file).")
    p.add_argument("--queries-file", type=str, default=None, help="Path to a text file with one query per line.")
    p.add_argument("--query", action="append", help="Specify a single query (can be repeated).")
    p.add_argument("--output", type=str, default="maps_eval_results_hybrid.jsonl", help="Output JSONL file path.")
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