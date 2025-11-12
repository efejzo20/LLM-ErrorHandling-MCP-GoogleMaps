import os
import json
import csv
from glob import glob
from typing import Dict, Any, List, Tuple

SUMMARY_DIR = os.path.dirname(__file__)

# Filenames follow pattern: judged_maps_eval_results_<model>_new_summary.json
# and judged_maps_eval_results_<model>_new_recovery_summary.json


def load_summaries() -> Dict[str, Dict[str, Any]]:
	by_model: Dict[str, Dict[str, Any]] = {}
	json_paths = glob(os.path.join(SUMMARY_DIR, "judged_maps_eval_results_*_summary.json"))
	for path in json_paths:
		name = os.path.basename(path)
		is_recovery = name.endswith("_recovery_summary.json")
		# Extract model key between prefix and suffix
		prefix = "judged_maps_eval_results_"
		suffix = "_recovery_summary.json" if is_recovery else "_summary.json"
		model_key = name[len(prefix):-len(suffix)]
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		if model_key not in by_model:
			by_model[model_key] = {}
		by_model[model_key]["recovery" if is_recovery else "no_recovery"] = data
	return by_model


def compute_rows(by_model: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	for model, parts in sorted(by_model.items()):
		no = parts.get("no_recovery")
		rec = parts.get("recovery")
		if not no and not rec:
			continue
		# Prefer counts from whichever exists; they should match
		count = None
		if no and isinstance(no.get("count"), (int, float)):
			count = int(no["count"])
		elif rec and isinstance(rec.get("count"), (int, float)):
			count = int(rec["count"])

		def get_metric(obj: Dict[str, Any], key: str, nested: bool = False) -> Any:
			if not obj:
				return None
			if nested:
				avg = obj.get("avg_scores") or {}
				return avg.get(key)
			return obj.get(key)

		row = {
			"model": model,
			"count": count,
			"pass_rate_no": get_metric(no, "pass_rate"),
			"pass_rate_rec": get_metric(rec, "pass_rate"),
			"overall_no": get_metric(no, "overall", nested=True),
			"overall_rec": get_metric(rec, "overall", nested=True),
			"tool_selection_no": get_metric(no, "tool_selection", nested=True),
			"tool_selection_rec": get_metric(rec, "tool_selection", nested=True),
			"parameter_logic_no": get_metric(no, "parameter_logic", nested=True),
			"parameter_logic_rec": get_metric(rec, "parameter_logic", nested=True),
			"result_relevance_no": get_metric(no, "result_relevance", nested=True),
			"result_relevance_rec": get_metric(rec, "result_relevance", nested=True),
			"format_validity_no": get_metric(no, "format_validity", nested=True),
			"format_validity_rec": get_metric(rec, "format_validity", nested=True),
		}

		# Compute deltas where available
		for metric in [
			"pass_rate",
			"overall",
			"tool_selection",
			"parameter_logic",
			"result_relevance",
			"format_validity",
		]:
			no_v = row.get(f"{metric}_no")
			rec_v = row.get(f"{metric}_rec")
			row[f"{metric}_delta"] = (rec_v - no_v) if (isinstance(no_v, (int, float)) and isinstance(rec_v, (int, float))) else None

		rows.append(row)
	return rows


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
	# Define column order
	cols = [
		"model", "count",
		"pass_rate_no", "pass_rate_rec", "pass_rate_delta",
		"overall_no", "overall_rec", "overall_delta",
		"tool_selection_no", "tool_selection_rec", "tool_selection_delta",
		"parameter_logic_no", "parameter_logic_rec", "parameter_logic_delta",
		"result_relevance_no", "result_relevance_rec", "result_relevance_delta",
		"format_validity_no", "format_validity_rec", "format_validity_delta",
	]
	with open(out_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=cols)
		writer.writeheader()
		for r in rows:
			writer.writerow({k: r.get(k) for k in cols})


def write_markdown(rows: List[Dict[str, Any]], out_path: str) -> None:
	cols = [
		"model", "count",
		"pass_rate_no", "pass_rate_rec", "pass_rate_delta",
		"overall_no", "overall_rec", "overall_delta",
		"tool_selection_no", "tool_selection_rec", "tool_selection_delta",
		"parameter_logic_no", "parameter_logic_rec", "parameter_logic_delta",
		"result_relevance_no", "result_relevance_rec", "result_relevance_delta",
		"format_validity_no", "format_validity_rec", "format_validity_delta",
	]
	headers = [
		"Model", "N",
		"Pass no", "Pass rec", "Δ Pass",
		"Overall no", "Overall rec", "Δ Overall",
		"ToolSel no", "ToolSel rec", "Δ ToolSel",
		"ParamLogic no", "ParamLogic rec", "Δ ParamLogic",
		"Relevance no", "Relevance rec", "Δ Relevance",
		"Format no", "Format rec", "Δ Format",
	]
	# Build table
	lines: List[str] = []
	lines.append("| " + " | ".join(headers) + " |")
	lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
	for r in rows:
		vals: List[str] = []
		for k in cols:
			v = r.get(k)
			if isinstance(v, float):
				vals.append(f"{v:.4f}")
			else:
				vals.append("" if v is None else str(v))
		lines.append("| " + " | ".join(vals) + " |")
	with open(out_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")


def main() -> None:
	by_model = load_summaries()
	rows = compute_rows(by_model)
	# Sort by pass rate recovery desc if available
	rows.sort(key=lambda r: (r.get("pass_rate_rec") or 0.0), reverse=True)
	write_csv(rows, os.path.join(SUMMARY_DIR, "comparison_table.csv"))
	write_markdown(rows, os.path.join(SUMMARY_DIR, "comparison_table.md"))
	print(f"Wrote {len(rows)} models to comparison_table.csv and comparison_table.md")


if __name__ == "__main__":
	main() 