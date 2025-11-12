import os
import sys
import json
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


JUDGE_SYSTEM_PROMPT = (
	"You are an evaluator. Judge ONLY with the provided data; do not use external knowledge. "
	"Return STRICT JSON with the exact schema."
)

JUDGE_USER_TEMPLATE = string.Template(
	"You will be given a maps record as JSON fields.\n\n"
	"- query: $query\n"
	"- tool_call: $tool_call\n"
	"- tool_response: $tool_response\n\n"
	"Rubric (0..1 each):\n"
	"- tool_selection: correct tool for intent?\n"
	"- parameter_logic: parameters are semantically appropriate to the user intent (e.g., city matches, radius/mode sensible, no irrelevant params).\n"
	"- result_relevance: results match location/category intent?\n"
	"- format_validity: response is valid/parseable for this tool?\n\n"
	"Return JSON: {\n"
	"  \"scores\": {\n"
	"    \"tool_selection\": number,\n"
	"    \"parameter_logic\": number,\n"
	"    \"result_relevance\": number,\n"
	"    \"format_validity\": number,\n"
	"    \"overall\": number\n"
	"  },\n"
	"  \"verdict\": \"pass\" | \"fail\",\n"
	"  \"reasons\": [string],\n"
	"  \"proposed_fix\": { \"tool_name\": string | null, \"parameters\": object | null, \"notes\": string },\n"
	"  \"tags\": [ \"geo_mismatch\" | \"wrong_tool\" | \"ok\" | \"illogical_param\" | \"over_broad_radius\" | \"irrelevant_mode\" ]\n"
	"}\n\n"
	"Constraints:\n"
	"- Cite only from tool_response and inputs.\n"
	"- If response contains results for a different city than requested, tag geo_mismatch.\n"
	"- If params look syntactically valid but semantically odd (e.g., \"mode=flying\" for a places search, or radius extremely large), tag illogical_param and propose a corrected call.\n"
	"- If response string is not JSON but looks valid tool output, set format_validity accordingly and explain.\n"
	"- Always fill every field.\n"
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


def _select_tool_io(record: Dict[str, Any], source: str) -> Tuple[Dict[str, Any], str, Optional[str]]:
	"""Return a minimal view {query, tool_call, tool_response} for the requested source.
	Also return the effective source used and an optional fallback reason.
	"""
	requested = source
	fallback_reason: Optional[str] = None
	if source == "initial":
		view = {
			"query": record.get("query"),
			"tool_call": record.get("tool_call"),
			"tool_response": record.get("tool_response"),
		}
		effective = "initial"
	elif source == "recovery":
		rec_call = record.get("recovery_tool_call")
		rec_resp = record.get("recovery_tool_response")
		if rec_call is not None or rec_resp is not None:
			view = {"query": record.get("query"), "tool_call": rec_call, "tool_response": rec_resp}
			effective = "recovery"
		else:
			view = {
				"query": record.get("query"),
				"tool_call": record.get("tool_call"),
				"tool_response": record.get("tool_response"),
			}
			effective = "initial"
			fallback_reason = "No recovery tool_call/response present"
	elif source == "final":
		if record.get("recovery_success"):
			view = {
				"query": record.get("query"),
				"tool_call": record.get("recovery_tool_call"),
				"tool_response": record.get("recovery_tool_response"),
			}
			effective = "recovery"
		else:
			view = {
				"query": record.get("query"),
				"tool_call": record.get("tool_call"),
				"tool_response": record.get("tool_response"),
			}
			effective = "initial"
	else:
		# Default safety
		view = {
			"query": record.get("query"),
			"tool_call": record.get("tool_call"),
			"tool_response": record.get("tool_response"),
		}
		effective = "initial"
		fallback_reason = f"Unknown response-source '{source}', defaulted to initial"
	return view, effective, fallback_reason


async def judge_record(llm, record: Dict[str, Any]) -> Dict[str, Any]:
	query = record.get("query")
	tool_call = record.get("tool_call")
	tool_response = record.get("tool_response")

	# Truncate overly long tool_response to control cost
	tool_response_str: str
	if isinstance(tool_response, (dict, list)):
		tool_response_str = _safe_json(tool_response)
	else:
		tool_response_str = str(tool_response) if tool_response is not None else ""
	if len(tool_response_str) > 20000:
		tool_response_str = tool_response_str[:20000] + "\n...[truncated]"

	user = JUDGE_USER_TEMPLATE.safe_substitute(
		query=_safe_json(query),
		tool_call=_safe_json(tool_call),
		tool_response=tool_response_str,
	)

	messages = [
		SystemMessage(content=JUDGE_SYSTEM_PROMPT),
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
		judge = json.loads(content)
	except Exception:
		judge = {
			"scores": {
				"tool_selection": 0.0,
				"parameter_logic": 0.0,
				"result_relevance": 0.0,
				"format_validity": 0.0,
				"overall": 0.0,
			},
			"verdict": "fail",
			"reasons": ["Judge LLM returned non-JSON or unparsable content"],
			"proposed_fix": {"tool_name": None, "parameters": None, "notes": "Unable to parse judge output"},
			"tags": ["illogical_param"],
		}
	return judge


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


def summarize(judged: List[Dict[str, Any]]) -> Dict[str, Any]:
	def _avg(key: str) -> float:
		vals: List[float] = []
		for r in judged:
			s = r.get("scores", {})
			v = s.get(key)
			if isinstance(v, (int, float)):
				vals.append(float(v))
		return round(sum(vals) / len(vals), 4) if vals else 0.0

	pass_rate = 0.0
	if judged:
		passed = sum(1 for r in judged if r.get("verdict") == "pass")
		pass_rate = round(passed / len(judged), 4)

	tag_counts: Dict[str, int] = {}
	for r in judged:
		for t in r.get("tags", []) or []:
			tag_counts[t] = tag_counts.get(t, 0) + 1

	return {
		"count": len(judged),
		"pass_rate": pass_rate,
		"avg_scores": {
			"tool_selection": _avg("tool_selection"),
			"parameter_logic": _avg("parameter_logic"),
			"result_relevance": _avg("result_relevance"),
			"format_validity": _avg("format_validity"),
			"overall": _avg("overall"),
		},
		"tag_counts": tag_counts,
	}


async def run(args: argparse.Namespace) -> int:
	llm = _choose_llm(args.model)
	records = read_jsonl(args.input)
	if not records:
		print(f"No records found in {args.input}")
		return 2

	if args.limit is not None:
		records = records[: args.limit]

	semaphore = asyncio.Semaphore(args.max_concurrency)
	judged_records: List[Dict[str, Any]] = [None] * len(records)  # type: ignore

	async def _one(i: int, rec: Dict[str, Any]):
		async with semaphore:
			view, effective_source, fallback_reason = _select_tool_io(rec, args.response_source)
			j = await judge_record(llm, view)
			judged_records[i] = {
				"input": rec,
				"evaluated_source_requested": args.response_source,
				"evaluated_source_effective": effective_source,
				"evaluated_source_fallback_reason": fallback_reason,
				"judge": j,
			}

	tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(records)]
	await asyncio.gather(*tasks)

	write_jsonl(args.output, judged_records)

	summary = summarize([jr["judge"] for jr in judged_records if isinstance(jr, dict) and "judge" in jr])
	with open(args.summary, "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	print(
		f"Wrote {len(judged_records)} judged records to {args.output}. Summary → {args.summary} "
		f"(pass_rate={summary['pass_rate']}, overall={summary['avg_scores']['overall']})."
	)
	return 0


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="LLM-as-a-judge for Maps MCP outputs")
	p.add_argument("--input", type=str, required=True, help="Path to input JSONL file of records")
	p.add_argument("--output", type=str, required=True, help="Path to output judged JSONL")
	p.add_argument("--summary", type=str, default="judge_summary.json", help="Path to write aggregate JSON summary")
	p.add_argument("--model", type=str, default=None, help="Judge model (e.g., gpt-4o-mini, or ollama:phi4:14b)")
	p.add_argument("--limit", type=int, default=None, help="Limit number of input rows to judge")
	p.add_argument("--max-concurrency", type=int, default=4, help="Concurrent LLM calls")
	p.add_argument(
		"--response-source",
		type=str,
		default="initial",
		choices=["initial", "recovery", "final"],
		help="Which call/response to evaluate: initial, recovery, or final (recovery if successful else initial)",
	)
	return p


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	code = asyncio.run(run(args))
	sys.exit(code)


if __name__ == "__main__":
	main() 