import os
import sys
import argparse
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MapsDatasetGenerator:
    """Generate user-query-only datasets for Google Maps function-calling.

    - Connects to the Google Maps MCP server and fetches available tools plus input schemas
    - Builds a human-readable toolset context for the LLM
    - Prompts the LLM to create only user queries (one per line)
    - Provides CLI for count/format/output
    """

    def __init__(self) -> None:
        self.llm: Optional[ChatOpenAI] = None
        self._stdio_context = None
        self._session_context = None
        self.mcp_client = None
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.toolset_context: str = ""

    async def setup(self) -> None:
        """Initialize LLM and connect to MCP; list tools and build toolset context."""
        # Initialize LLM (OpenAI GPT-4o)
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-5",
            temperature=0.7,
        )

        # Setup MCP connection for Google Maps
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

        await self._load_tools()
        self.toolset_context = self._build_toolset_context()
        print(f"✅ Dataset generator setup complete with {len(self.available_tools)} tools (Google Maps)")

    async def cleanup(self) -> None:
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)

    async def _load_tools(self) -> None:
        tools_result = await self.mcp_client.list_tools()
        self.available_tools.clear()
        for tool in tools_result.tools:
            schema_obj: Any = {}
            # Try to normalize schema to a plain dict
            raw = getattr(tool, "inputSchema", None)
            if raw is None:
                schema_obj = {}
            elif isinstance(raw, dict):
                schema_obj = raw
            else:
                # Attempt common shapes: .schema or .model_dump
                schema_obj = getattr(raw, "schema", None) or {}
                if not isinstance(schema_obj, dict):
                    try:
                        schema_obj = raw.model_dump()  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            schema_obj = json.loads(json.dumps(raw, default=lambda o: getattr(o, "__dict__", str(o))))
                        except Exception:
                            schema_obj = {}

            self.available_tools[tool.name] = {
                "description": getattr(tool, "description", ""),
                "schema": schema_obj,
            }

    def _build_toolset_context(self) -> str:
        """Build a compact but informative toolset description for prompting."""
        lines: List[str] = []
        lines.append("TOOLSET (Google Maps MCP):")
        for name, info in sorted(self.available_tools.items(), key=lambda x: x[0].lower()):
            desc = info.get("description", "").strip()
            schema = info.get("schema", {}) or {}
            required = schema.get("required", []) if isinstance(schema, dict) else []
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}

            def _param_sig() -> str:
                if not isinstance(props, dict) or not props:
                    return "no parameters"
                parts: List[str] = []
                for p_name, p_spec in props.items():
                    if not isinstance(p_spec, dict):
                        p_type = "any"
                        p_desc = ""
                    else:
                        p_type = p_spec.get("type", "any")
                        p_desc = p_spec.get("description", "")
                    req_flag = "required" if p_name in required else "optional"
                    if p_desc:
                        parts.append(f"{p_name} ({p_type}, {req_flag}) - {p_desc}")
                    else:
                        parts.append(f"{p_name} ({p_type}, {req_flag})")
                return "; ".join(parts)

            lines.append(f"- {name}: {desc if desc else 'No description'}")
            lines.append(f"  params: {_param_sig()}")
        return "\n".join(lines)

    def _build_generation_prompt(self, total: int, difficulty_mix: str, languages: Optional[List[str]]) -> str:
        lang_clause = "; Languages: " + ", ".join(languages) if languages else ""
        return (
            "You are generating test queries to evaluate LLM function calling for Google Maps tools. \n\n"
            "Generate queries that the LLM would mess up the name of the tool call like maps_places_details instead of maps_place_details, or maps_elvation to elavation. \n\n"
            "These are the tool names:maps_reverse_geocode, maps_search_places, maps_place_details, maps_distance_matrix, maps_elevation, maps_directions \n\n"
            + self.toolset_context
            + "\n\nGoal:\n"
              "- Produce ONLY user queries (no explanations, no numbering, no quotes), one per line.\n"
              "- Balance across tools and difficulties: "
              f"{difficulty_mix}. Include realistic noise (typos, unit variants, mixed address formats){lang_clause}.\n"
              "- Include adversarial cases likely to trigger wrong tool selection (e.g., places_search vs geocode; directions vs distance matrix; text vs lat/lng).\n"
              "- Ensure every query is solvable using the TOOLSET (avoid requests outside capability).\n\n"
              "Output rules:\n"
              f"- Output exactly {total} queries.\n"
              "- One query per line.\n"
              "- No extra text, no numbering, no quotes, no code fences.\n"
              "- Adversarial mode:\n"
              "  - Ensure ≥30% queries are designed to confuse tool selection (e.g., places_search vs place_details).\n"
              "  - Ensure ≥30% queries are designed to confuse parameter binding (lat/lng order; units like miles vs meters; time windows; “near me” without coords; ambiguous place names).\n"
              "  - Use mixed-intent and distractors in some queries (two possible tools; attribute overload).\n"
              "  - Keep queries solvable by the TOOLSET, but make the intended tool/parameters non-obvious.\n"
              "  - Maintain user-query-only output rules (one per line, no extra text).\n"
              " - Do not iclude queries that need maps_distance_matrix"
        )

    def _clean_and_split_lines(self, raw_text: str) -> List[str]:
        content = raw_text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        lines = [ln.strip() for ln in content.splitlines()]
        lines = [ln for ln in lines if ln]
        return lines

    def _dedupe_and_trim(self, lines: List[str], target_n: int) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for ln in lines:
            key = ln.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(ln.strip())
            if len(out) >= target_n:
                break
        return out

    async def generate_queries(self, total: int, difficulty_mix: str, languages: Optional[List[str]], max_retries: int = 3) -> List[str]:
        """Generate user queries only, enforcing exact count via light retries."""
        prompt = self._build_generation_prompt(total, difficulty_mix, languages)
        messages = [
            SystemMessage(content=[{"type": "text", "text": prompt}]),
            HumanMessage(content=[{"type": "text", "text": f"Generate {total} queries now."}]),
        ]

        remaining = total
        results: List[str] = []
        attempt = 0
        while remaining > 0 and attempt <= max_retries:
            attempt += 1
            response = await self.llm.ainvoke(messages)
            lines = self._clean_and_split_lines(response.content)
            results.extend(lines)
            results = self._dedupe_and_trim(results, total)
            remaining = total - len(results)
            if remaining > 0:
                # Tighten the next request to only the remaining number
                messages = [
                    SystemMessage(content=[{"type": "text", "text": prompt}]),
                    HumanMessage(content=[{"type": "text", "text": f"You previously generated {total - remaining} unique queries. Output exactly {remaining} more new queries. Remember: only user queries, one per line."}]),
                ]

        if len(results) < total:
            print(f"⚠️ Could not reach exact count. Requested {total}, got {len(results)} after retries.")
        return results[:total]


def write_output(path: str, queries: List[str], fmt: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "txt":
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(queries) + "\n")
    elif fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for q in queries:
                f.write(json.dumps({"query": q}, ensure_ascii=False) + "\n")
    else:
        raise ValueError("Unsupported format; use 'txt' or 'jsonl'")


def _parse_languages(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    langs = [x.strip() for x in arg.split(",") if x.strip()]
    return langs or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate user-query-only datasets for Google Maps function-calling using MCP tool context.")
    p.add_argument("--num", type=int, default=200, help="Number of queries to generate.")
    p.add_argument("--output", type=str, default="Dataset/MCP-GoogleMaps/maps_queries.txt", help="Output file path.")
    p.add_argument("--format", type=str, choices=["txt", "jsonl"], default="txt", help="Output format: txt (one per line) or jsonl (objects with 'query').")
    p.add_argument("--difficulty", type=str, default="Easy 40%, Medium 40%, Hard 20%", help="Difficulty mix description fed to the prompt.")
    p.add_argument("--languages", type=str, default=None, help="Comma-separated language codes to include (e.g., 'en,es,de').")
    return p


async def run(args: argparse.Namespace) -> int:
    missing = []
    if not os.getenv("GOOGLE_MAPS_API_KEY"):
        missing.append("GOOGLE_MAPS_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        return 2

    generator = MapsDatasetGenerator()
    try:
        await generator.setup()
        queries = await generator.generate_queries(
            total=args.num,
            difficulty_mix=args.difficulty,
            languages=_parse_languages(args.languages),
        )
        write_output(args.output, queries, args.format)
        print(f"✅ Wrote {len(queries)} queries to {args.output} ({args.format})")
        return 0
    finally:
        await generator.cleanup()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 