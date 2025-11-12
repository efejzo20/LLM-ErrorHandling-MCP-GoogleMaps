"""
Recalculate pass rates by ignoring result_relevance in the verdict.
A record passes if tool_selection, parameter_logic, and format_validity are all 1.
"""
import json
import os
from typing import Dict, Any, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def recalculate_verdict(judge: Dict[str, Any]) -> str:
    """
    Determine verdict based only on tool_selection, parameter_logic, and format_validity.
    Ignore result_relevance.
    """
    scores = judge.get("scores", {})
    tool_selection = scores.get("tool_selection", 0)
    parameter_logic = scores.get("parameter_logic", 0)
    format_validity = scores.get("format_validity", 0)
    
    # Pass if all three are 1
    if tool_selection == 1 and parameter_logic == 1 and format_validity == 1:
        return "pass"
    return "fail"


def calculate_summary(judged_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics ignoring result_relevance in verdict."""
    judges = [r.get("judge", {}) for r in judged_records if "judge" in r]
    
    # Recalculate verdicts
    new_passes = sum(1 for j in judges if recalculate_verdict(j) == "pass")
    pass_rate = round(new_passes / len(judges), 4) if judges else 0.0
    
    # Calculate average scores (including result_relevance for reference)
    def _avg(key: str) -> float:
        vals = []
        for j in judges:
            s = j.get("scores", {})
            v = s.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return round(sum(vals) / len(vals), 4) if vals else 0.0
    
    # Tag counts
    tag_counts: Dict[str, int] = {}
    for j in judges:
        for t in j.get("tags", []) or []:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    
    return {
        "count": len(judges),
        "pass_rate": pass_rate,
        "pass_rate_note": "Calculated ignoring result_relevance (only tool_selection, parameter_logic, format_validity)",
        "avg_scores": {
            "tool_selection": _avg("tool_selection"),
            "parameter_logic": _avg("parameter_logic"),
            "result_relevance": _avg("result_relevance"),  # Still show for reference
            "format_validity": _avg("format_validity"),
            "overall": _avg("overall"),
        },
        "tag_counts": tag_counts,
    }


def main():
    data_eval_dir = "/Users/elvjofejzo/Documents/Personal/bluesky-social-mcp-main/DATA-EVAL"
    
    # Models to process
    models = [
        "llama3_new",
        "llama3_new_recovery",
        "phi4_14b_new",
        "phi4_14b_new_recovery",
        "qwen2_new",
        "qwen2_new_recovery",
        "gemma_new",
        "gemma_new_recovery",
        "mistral_7b-instruct_new",
        "mistral_7b-instruct_new_recovery",
        "llama7b_new",
        "llama7b_new_recovery",
    ]
    
    results = {}
    
    for model in models:
        # Try different judge variants and filename patterns
        found = False
        is_recovery = "recovery" in model
        
        # Pattern for recovery models: judged_maps_eval_results_{model_without_recovery}_gpt35_recovery.jsonl
        # Pattern for base models: judged_maps_eval_results_{model}_gpt4omini.jsonl or judged_{model}_gpt4omini.jsonl
        
        if is_recovery:
            # Remove _recovery suffix for filename
            model_base = model.replace("_recovery", "")
            for prefix in ["judged_maps_eval_results_", "judged_"]:
                filename = f"{prefix}{model_base}_gpt35_recovery.jsonl"
                filepath = os.path.join(data_eval_dir, filename)
                
                if os.path.exists(filepath):
                    print(f"Processing {filename}...")
                    records = read_jsonl(filepath)
                    summary = calculate_summary(records)
                    results[model] = summary
                    print(f"  Pass rate (ignoring relevance): {summary['pass_rate']:.4f}")
                    print(f"  Tool Selection: {summary['avg_scores']['tool_selection']:.4f}")
                    print(f"  Parameter Logic: {summary['avg_scores']['parameter_logic']:.4f}")
                    print(f"  Format Validity: {summary['avg_scores']['format_validity']:.4f}")
                    print()
                    found = True
                    break
        else:
            for prefix in ["judged_maps_eval_results_", "judged_"]:
                for judge_suffix in ["_gpt4omini", "_gpt35"]:
                    filename = f"{prefix}{model}{judge_suffix}.jsonl"
                    filepath = os.path.join(data_eval_dir, filename)
                    
                    if os.path.exists(filepath):
                        print(f"Processing {filename}...")
                        records = read_jsonl(filepath)
                        summary = calculate_summary(records)
                        results[model] = summary
                        print(f"  Pass rate (ignoring relevance): {summary['pass_rate']:.4f}")
                        print(f"  Tool Selection: {summary['avg_scores']['tool_selection']:.4f}")
                        print(f"  Parameter Logic: {summary['avg_scores']['parameter_logic']:.4f}")
                        print(f"  Format Validity: {summary['avg_scores']['format_validity']:.4f}")
                        print()
                        found = True
                        break
                if found:
                    break
    
    # Print LaTeX table
    print("\n" + "="*80)
    print("RECALCULATED TABLE (ignoring result_relevance)")
    print("="*80 + "\n")
    
    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\caption{Results on MCP-GoogleMaps (excluding result\\_relevance from verdict)}")
    print("    \\begin{tabular}{lcccc}")
    print("    \\toprule")
    print("    \\textbf{Model} & \\textbf{Tool Selection} & \\textbf{Parameter Logic} & \\textbf{Format Validity} & \\textbf{Pass Rate} \\\\")
    print("    \\midrule")
    
    # Order models as in original table
    model_order = [
        ("llama3", "llama3_new"),
        ("llama3 + recovery", "llama3_new_recovery"),
        ("phi4\\_14b", "phi4_14b_new"),
        ("phi4\\_14b + recovery", "phi4_14b_new_recovery"),
        ("qwen2", "qwen2_new"),
        ("qwen2 + recovery", "qwen2_new_recovery"),
        ("gemma", "gemma_new"),
        ("gemma + recovery", "gemma_new_recovery"),
        ("mistral\\_7b", "mistral_7b-instruct_new"),
        ("mistral\\_7b + recovery", "mistral_7b-instruct_new_recovery"),
        ("llama7b", "llama7b_new"),
        ("llama7b + recovery", "llama7b_new_recovery"),
    ]
    
    for display_name, model_key in model_order:
        if model_key in results:
            s = results[model_key]
            print(f"    {display_name} & {s['avg_scores']['tool_selection']:.4f} & "
                  f"{s['avg_scores']['parameter_logic']:.4f} & "
                  f"{s['avg_scores']['format_validity']:.4f} & "
                  f"{s['pass_rate']:.4f} \\\\")
    
    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("\\end{table}")
    
    # Also save summaries to files
    output_dir = "/Users/elvjofejzo/Documents/Personal/bluesky-social-mcp-main/Data-Sumary"
    for model_key, summary in results.items():
        output_file = os.path.join(output_dir, f"recalc_no_relevance_{model_key}_summary.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nSaved {len(results)} summary files to {output_dir}/recalc_no_relevance_*_summary.json")


if __name__ == "__main__":
    main()

