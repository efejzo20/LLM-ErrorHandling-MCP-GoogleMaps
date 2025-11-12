"""
LLM Judge for Hybrid Maps Evaluator with RAG Expected Locations
Evaluates whether:
1. RAG was triggered when needed
2. RAG resolved to the expected location
3. Final tool call used the resolved location correctly
4. Final result is correct
"""

import os
import sys
import json
import csv
import argparse
import asyncio
import string
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception as e:
    raise RuntimeError("langchain is required for this script") from e
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None  # type: ignore

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Prompt for RAG-needed queries
RAG_JUDGE_SYSTEM_PROMPT = (
    "You are an address comparison evaluator. "
    "Judge ONLY with the provided data; do not use external knowledge. "
    "Compare if two addresses refer to the same location. "
    "Return STRICT JSON with the exact schema."
)

RAG_JUDGE_USER_TEMPLATE = string.Template(
    "Compare if the RAG-resolved address matches the expected address.\n\n"
    "EXPECTED ADDRESS (from ground truth):\n"
    "$expected_address\n\n"
    "RAG-RESOLVED ADDRESS (what the system retrieved):\n"
    "$rag_resolved_address\n\n"
    "TASK:\n"
    "Determine if these two addresses refer to the same location.\n"
    "They do NOT need to be character-by-character identical.\n"
    "They SHOULD match on key components: building/venue name, street, city.\n"
    "Minor differences in formatting, abbreviations, or extra details are acceptable.\n\n"
    "Return JSON:\n"
    "{\n"
    "  \"addresses_match\": boolean (true if same location, false otherwise),\n"
    "  \"confidence\": number (0.0 to 1.0, how confident you are),\n"
    "  \"reasoning\": string (brief explanation of your decision)\n"
    "}\n"
)

# Prompt for non-RAG queries  
NON_RAG_JUDGE_SYSTEM_PROMPT = (
    "You are an API response evaluator. "
    "Judge if the API response contains valid, useful results for the user's query. "
    "Return STRICT JSON with the exact schema."
)

NON_RAG_JUDGE_USER_TEMPLATE = string.Template(
    "Evaluate if the API response is successful and contains useful results.\n\n"
    "USER QUERY:\n"
    "$query\n\n"
    "API RESPONSE:\n"
    "$api_response\n\n"
    "TASK:\n"
    "Determine if the API call succeeded and returned valid, relevant results.\n"
    "Check for:\n"
    "- No error messages\n"
    "- Contains actual location/direction data\n"
    "- Results are relevant to the query\n\n"
    "Return JSON:\n"
    "{\n"
    "  \"api_success\": boolean (true if API returned good results, false otherwise),\n"
    "  \"confidence\": number (0.0 to 1.0),\n"
    "  \"reasoning\": string (brief explanation)\n"
    "}\n"
)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _choose_llm(model: Optional[str] = None):
    """Prefer OpenAI GPT-4-class if available; otherwise fall back to local Ollama; else raise."""
    requested = model or os.getenv("JUDGE_MODEL")
    openai_key = os.getenv("OPENAI_API_KEY")
    if requested and requested.startswith("ollama:"):
        if ChatOllama is None:
            raise RuntimeError("ChatOllama not available; install langchain-community")
        ollama_model = requested.split(":", 1)[1]
        return ChatOllama(base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), model=ollama_model, temperature=0.0)
    if openai_key and ChatOpenAI is not None:
        # If a non-ollama model was requested, use it; otherwise read from env
        chosen = requested or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=chosen, temperature=0.0)
    if ChatOllama is not None:
        return ChatOllama(base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), model=os.getenv("OLLAMA_MODEL", "phi4:14b"), temperature=0.0)
    raise RuntimeError("No LLM backend available. Provide OPENAI_API_KEY or run Ollama and install langchain-community.")


def load_expected_locations(csv_path: str) -> Dict[str, Dict[str, str]]:
    """Load CSV mapping query -> expected RAG location."""
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("query", "").strip()
            if query:
                mapping[query] = {
                    "event_name": row.get("event_name", "").strip(),
                    "expected_rag_location": row.get("expected_rag_location", "").strip(),
                    "needs_rag": row.get("needs_rag", "").strip().lower() == "yes"
                }
    return mapping


async def judge_rag_address(llm, expected_address: str, rag_resolved_address: str) -> Dict[str, Any]:
    """Use LLM to compare if two addresses match."""
    user = RAG_JUDGE_USER_TEMPLATE.safe_substitute(
        expected_address=expected_address,
        rag_resolved_address=rag_resolved_address,
    )

    messages = [
        SystemMessage(content=RAG_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user),
    ]

    resp = await llm.ainvoke(messages)
    content = (resp.content or "").strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"❌ Failed to parse RAG judge response: {e}")
        return {
            "addresses_match": False,
            "confidence": 0.0,
            "reasoning": f"Failed to parse LLM response: {str(e)}"
        }


async def judge_api_response(llm, query: str, api_response: str) -> Dict[str, Any]:
    """Use LLM to evaluate if API response is successful."""
    user = NON_RAG_JUDGE_USER_TEMPLATE.safe_substitute(
        query=query,
        api_response=api_response,
    )

    messages = [
        SystemMessage(content=NON_RAG_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user),
    ]

    resp = await llm.ainvoke(messages)
    content = (resp.content or "").strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"❌ Failed to parse API judge response: {e}")
        return {
            "api_success": False,
            "confidence": 0.0,
            "reasoning": f"Failed to parse LLM response: {str(e)}"
        }


async def judge_hybrid_record(llm, record: Dict[str, Any], expected_data: Dict[str, str]) -> Dict[str, Any]:
    """Judge a single hybrid evaluator record with simplified logic."""
    query = record.get("query", "")
    needs_rag = expected_data.get("needs_rag", False)
    expected_rag_location = expected_data.get("expected_rag_location", "")
    
    # Extract execution data
    route = record.get("route")
    rag_resolved_location = record.get("rag_resolved_location", "")
    final_success = record.get("final_success", False)
    
    # Get final response (prefer second retry, then retry, then initial)
    if record.get("second_retry_success"):
        final_tool_response = record.get("second_retry_tool_response", "")
    elif record.get("retry_success"):
        final_tool_response = record.get("retry_tool_response", "")
    else:
        final_tool_response = record.get("tool_response", "")
    
    # Truncate long responses
    if isinstance(final_tool_response, (dict, list)):
        tool_response_str = _safe_json(final_tool_response)
    else:
        tool_response_str = str(final_tool_response) if final_tool_response is not None else ""
    if len(tool_response_str) > 10000:
        tool_response_str = tool_response_str[:10000] + "\n...[truncated]"

    if needs_rag:
        # RAG-needed query: Check programmatically + use LLM for address comparison
        
        # Programmatic checks
        rag_triggered = (route == "RAG")
        
        # LLM check: Did RAG resolve correct address?
        if rag_triggered and rag_resolved_location:
            address_judge = await judge_rag_address(llm, expected_rag_location, rag_resolved_location)
            rag_correct_address = address_judge["addresses_match"]
            address_confidence = address_judge["confidence"]
            address_reasoning = address_judge["reasoning"]
        else:
            rag_correct_address = False
            address_confidence = 0.0
            address_reasoning = "RAG was not triggered" if not rag_triggered else "No RAG location resolved"
        
        # LLM check: Did final API call succeed?
        api_judge = await judge_api_response(llm, query, tool_response_str)
        api_success = api_judge["api_success"]
        api_confidence = api_judge["confidence"]
        api_reasoning = api_judge["reasoning"]
        
        # Overall verdict: Pass if RAG triggered, got correct address, and API succeeded
        overall_pass = rag_triggered and rag_correct_address and api_success
        
        return {
            "needs_rag": True,
            "rag_triggered": rag_triggered,
            "rag_correct_address": rag_correct_address,
            "address_comparison": {
                "expected": expected_rag_location,
                "resolved": rag_resolved_location,
                "confidence": address_confidence,
                "reasoning": address_reasoning
            },
            "api_success": api_success,
            "api_evaluation": {
                "confidence": api_confidence,
                "reasoning": api_reasoning
            },
            "final_success": final_success,
            "verdict": "pass" if overall_pass else "fail",
            "issues": {
                "rag_not_triggered": not rag_triggered,
                "rag_wrong_address": rag_triggered and not rag_correct_address,
                "api_failed": not api_success
            }
        }
    else:
        # Non-RAG query: Only check if API succeeded
        api_judge = await judge_api_response(llm, query, tool_response_str)
        api_success = api_judge["api_success"]
        
        return {
            "needs_rag": False,
            "api_success": api_success,
            "api_evaluation": {
                "confidence": api_judge["confidence"],
                "reasoning": api_judge["reasoning"]
            },
            "final_success": final_success,
            "verdict": "pass" if api_success else "fail",
            "issues": {
                "api_failed": not api_success
            }
        }


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def summarize_hybrid(judged: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics for hybrid evaluation."""
    if not judged:
        return {
            "count": 0,
            "pass_rate": 0.0,
            "rag_needed_count": 0,
            "non_rag_count": 0,
            "rag_needed_pass_rate": 0.0,
            "non_rag_pass_rate": 0.0,
            "issue_counts": {},
        }

    pass_rate = 0.0
    passed = sum(1 for r in judged if r.get("verdict") == "pass")
    pass_rate = round(passed / len(judged), 4)

    # Separate RAG-needed vs non-RAG queries
    rag_needed = [r for r in judged if r.get("needs_rag")]
    non_rag = [r for r in judged if not r.get("needs_rag")]

    # Count issues
    issue_counts = {
        "rag_not_triggered": sum(1 for r in rag_needed if r.get("issues", {}).get("rag_not_triggered")),
        "rag_wrong_address": sum(1 for r in rag_needed if r.get("issues", {}).get("rag_wrong_address")),
        "api_failed": sum(1 for r in judged if r.get("issues", {}).get("api_failed")),
    }

    # Average confidence scores
    def _avg_confidence(records: List[Dict[str, Any]], key: str) -> float:
        vals = []
        for r in records:
            if key in r:
                conf = r[key].get("confidence", 0.0)
                if isinstance(conf, (int, float)):
                    vals.append(float(conf))
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "count": len(judged),
        "pass_rate": pass_rate,
        "rag_needed_count": len(rag_needed),
        "non_rag_count": len(non_rag),
        "rag_needed_pass_rate": round(sum(1 for r in rag_needed if r.get("verdict") == "pass") / len(rag_needed), 4) if rag_needed else 0.0,
        "non_rag_pass_rate": round(sum(1 for r in non_rag if r.get("verdict") == "pass") / len(non_rag), 4) if non_rag else 0.0,
        "issue_counts": issue_counts,
        "avg_address_confidence": _avg_confidence(rag_needed, "address_comparison"),
        "avg_api_confidence": _avg_confidence(judged, "api_evaluation"),
    }


async def run(args: argparse.Namespace) -> int:
    llm = _choose_llm(args.model)
    
    # Load expected locations
    expected_mapping = load_expected_locations(args.expected_locations)
    print(f"📋 Loaded {len(expected_mapping)} expected location mappings")
    
    # Load evaluation results
    records = read_jsonl(args.input)
    if not records:
        print(f"❌ No records found in {args.input}")
        return 2
    print(f"📊 Loaded {len(records)} evaluation records")

    if args.limit is not None:
        records = records[: args.limit]

    semaphore = asyncio.Semaphore(args.max_concurrency)
    judged_records: List[Dict[str, Any]] = []

    async def _one(rec: Dict[str, Any]):
        async with semaphore:
            query = rec.get("query", "")
            expected_data = expected_mapping.get(query, {})
            
            if not expected_data:
                print(f"⚠️  No expected data for query: {query[:50]}...")
                expected_data = {"needs_rag": False, "expected_rag_location": ""}
            
            j = await judge_hybrid_record(llm, rec, expected_data)
            return {
                "query": query,
                "needs_rag": expected_data.get("needs_rag", False),
                "expected_rag_location": expected_data.get("expected_rag_location", ""),
                "input": rec,
                "judge": j,
            }

    tasks = [asyncio.create_task(_one(r)) for r in records]
    judged_records = await asyncio.gather(*tasks)

    write_jsonl(args.output, judged_records)

    summary = summarize_hybrid([jr["judge"] for jr in judged_records if isinstance(jr, dict) and "judge" in jr])
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Wrote {len(judged_records)} judged records to {args.output}")
    print(f"📈 Summary → {args.summary}")
    print(f"\n📊 RESULTS:")
    print(f"  Overall Pass Rate: {summary['pass_rate']*100:.1f}%")
    print(f"  RAG-Needed Pass Rate: {summary['rag_needed_pass_rate']*100:.1f}% ({summary['rag_needed_count']} queries)")
    print(f"  Non-RAG Pass Rate: {summary['non_rag_pass_rate']*100:.1f}% ({summary['non_rag_count']} queries)")
    print(f"\n  Average LLM Confidence Scores:")
    print(f"    Address Matching: {summary.get('avg_address_confidence', 0.0):.3f}")
    print(f"    API Success Evaluation: {summary.get('avg_api_confidence', 0.0):.3f}")
    print(f"\n  Issues Found:")
    for issue_name, issue_count in summary['issue_counts'].items():
        if issue_count > 0:
            print(f"    {issue_name}: {issue_count}")
    print(f"{'='*60}\n")
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-as-a-judge for Hybrid Maps Evaluator with RAG validation")
    p.add_argument("--input", type=str, required=True, help="Path to input JSONL file (hybrid evaluator results)")
    p.add_argument(
        "--expected-locations",
        type=str,
        default="Dataset/RAG-Location/rag_expected_locations.csv",
        help="Path to CSV with expected RAG locations",
    )
    p.add_argument("--output", type=str, required=True, help="Path to output judged JSONL")
    p.add_argument("--summary", type=str, default="hybrid_judge_summary.json", help="Path to write aggregate JSON summary")
    p.add_argument("--model", type=str, default=None, help="Judge model (e.g., gpt-4o-mini, or ollama:phi4:14b)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of input rows to judge")
    p.add_argument("--max-concurrency", type=int, default=4, help="Concurrent LLM calls")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = asyncio.run(run(args))
    sys.exit(code)


if __name__ == "__main__":
    main()

