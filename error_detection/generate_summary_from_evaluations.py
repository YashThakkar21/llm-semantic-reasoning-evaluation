"""
Generate Summary from Existing LLM Judge Evaluations

This script reads all evaluation JSON files from the llm_judge_error_detection folder
and generates a comprehensive summary.json file with aggregate statistics.

Use this when:
- The evaluation script was interrupted and summary.json is incomplete or missing
- You want to regenerate the summary after additional evaluations
- You want to analyze a subset of evaluations
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


OUTPUT_DIR = Path("llm_judge_error_detection")


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: Path, data: Dict[str, Any]) -> None:
    """Save data to JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_evaluation_files(evaluation_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Collect all evaluation JSON files, excluding summary.json and error files."""
    results = {}
    
    # Find all JSON files
    json_files = sorted(
        (
            path
            for path in evaluation_dir.glob("*.json")
            if path.is_file() and path.name != "summary.json"
        ),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    
    print(f"Found {len(json_files)} evaluation files to process...\n")
    
    for json_file in json_files:
        try:
            # Extract problem ID
            problem_id = int(json_file.stem) if json_file.stem.isdigit() else None
            if problem_id is None:
                print(f"⚠️  Skipping {json_file.name} (not a numeric ID)")
                continue
            
            # Load file
            data = load_json_file(json_file)
            results[problem_id] = data
            
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {e}")
            continue
    
    return results


def generate_summary(
    results: Dict[int, Dict[str, Any]],
    evaluation_dir: Path,
) -> Dict[str, Any]:
    """Generate summary statistics from all evaluations."""
    
    print("Analyzing evaluation results...\n")
    
    # Filter valid results (exclude error entries)
    valid_results = [
        r for r in results.values()
        if "error" not in r and "judge_evaluation" in r
        and "error" not in r.get("judge_evaluation", {})
    ]
    
    error_results = [
        r for r in results.values()
        if "error" in r or "error" in r.get("judge_evaluation", {})
    ]
    
    if not valid_results:
        return {
            "total_evaluated": len(results),
            "valid_evaluations": 0,
            "error_count": len(results),
            "error": "No valid evaluations found",
            "evaluation_dir": str(evaluation_dir),
            "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        }
    
    # Collect scores
    score_categories = {
        "raw_error_detection_accuracy": [],
        "raw_error_location_precision": [],
        "raw_error_type_classification": [],
        "raw_error_explanation_quality": [],
        "raw_completeness": [],
        "raw_fix_suggestion_quality": [],
        "raw_overall_score": [],
        "new_error_detection_accuracy": [],
        "new_error_location_precision": [],
        "new_error_type_classification": [],
        "new_error_explanation_quality": [],
        "new_completeness": [],
        "new_fix_suggestion_quality": [],
        "new_overall_score": [],
        "overall_performance": [],
    }
    
    problem_ids_valid = []
    problem_ids_error = []
    
    for problem_id, result in results.items():
        if "error" in result or "error" in result.get("judge_evaluation", {}):
            problem_ids_error.append(problem_id)
            continue
        
        problem_ids_valid.append(problem_id)
        judge_eval = result.get("judge_evaluation", {})
        
        raw_eval = judge_eval.get("raw_solution_evaluation", {})
        if raw_eval:
            score_categories["raw_error_detection_accuracy"].append(raw_eval.get("error_detection_accuracy"))
            score_categories["raw_error_location_precision"].append(raw_eval.get("error_location_precision"))
            score_categories["raw_error_type_classification"].append(raw_eval.get("error_type_classification"))
            score_categories["raw_error_explanation_quality"].append(raw_eval.get("error_explanation_quality"))
            score_categories["raw_completeness"].append(raw_eval.get("completeness"))
            score_categories["raw_fix_suggestion_quality"].append(raw_eval.get("fix_suggestion_quality"))
            score_categories["raw_overall_score"].append(raw_eval.get("overall_score"))
        
        new_eval = judge_eval.get("new_solution_evaluation", {})
        if new_eval:
            score_categories["new_error_detection_accuracy"].append(new_eval.get("error_detection_accuracy"))
            score_categories["new_error_location_precision"].append(new_eval.get("error_location_precision"))
            score_categories["new_error_type_classification"].append(new_eval.get("error_type_classification"))
            score_categories["new_error_explanation_quality"].append(new_eval.get("error_explanation_quality"))
            score_categories["new_completeness"].append(new_eval.get("completeness"))
            score_categories["new_fix_suggestion_quality"].append(new_eval.get("fix_suggestion_quality"))
            score_categories["new_overall_score"].append(new_eval.get("overall_score"))
        
        overall_perf = judge_eval.get("overall_performance", {})
        if overall_perf:
            score_categories["overall_performance"].append(overall_perf.get("overall_error_detection_performance"))
    
    # Calculate statistics
    averages = {}
    for key, values in score_categories.items():
        valid_values = [v for v in values if v is not None and isinstance(v, (int, float))]
        if valid_values:
            averages[f"average_{key}"] = sum(valid_values) / len(valid_values)
            averages[f"min_{key}"] = min(valid_values)
            averages[f"max_{key}"] = max(valid_values)
            averages[f"count_{key}"] = len(valid_values)
    
    # Get problem ID range
    if problem_ids_valid:
        min_id = min(problem_ids_valid)
        max_id = max(problem_ids_valid)
    else:
        min_id = None
        max_id = None
    
    summary = {
        "total_evaluated": len(results),
        "valid_evaluations": len(valid_results),
        "error_count": len(error_results),
        "problem_id_range": {
            "min": min_id,
            "max": max_id,
            "valid_ids": sorted(problem_ids_valid) if problem_ids_valid else [],
        },
        "error_problem_ids": sorted(problem_ids_error) if problem_ids_error else [],
        "average_scores": averages,
        "evaluation_dir": str(evaluation_dir),
        "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        "note": "This summary was generated from existing evaluation files.",
    }
    
    return summary


def print_summary_statistics(summary: Dict[str, Any]) -> None:
    """Print a human-readable summary of statistics."""
    print("="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Total Evaluations: {summary.get('total_evaluated', 0)}")
    print(f"Valid Evaluations: {summary.get('valid_evaluations', 0)}")
    print(f"Error Count: {summary.get('error_count', 0)}")
    
    problem_range = summary.get("problem_id_range", {})
    if problem_range.get("min") is not None:
        print(f"Problem ID Range: {problem_range.get('min')} - {problem_range.get('max')}")
    
    print("\nAverage Scores (out of 5.0):")
    print("-" * 60)
    
    averages = summary.get("average_scores", {})
    
    # Group by category
    categories = {
        "Raw Solution": [
            "average_raw_error_detection_accuracy",
            "average_raw_error_location_precision",
            "average_raw_error_type_classification",
            "average_raw_error_explanation_quality",
            "average_raw_completeness",
            "average_raw_fix_suggestion_quality",
            "average_raw_overall_score",
        ],
        "New Solution": [
            "average_new_error_detection_accuracy",
            "average_new_error_location_precision",
            "average_new_error_type_classification",
            "average_new_error_explanation_quality",
            "average_new_completeness",
            "average_new_fix_suggestion_quality",
            "average_new_overall_score",
        ],
        "Overall": [
            "average_overall_performance",
        ],
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            if key in averages:
                score = averages[key]
                count_key = key.replace("average_", "count_")
                count = averages.get(count_key, 0)
                print(f"  {key.replace('average_', '').replace('_', ' ').title()}: {score:.2f} (n={count})")
    
    print("\n" + "="*60)


def main() -> None:
    print("="*60)
    print("Generate Summary from LLM Judge Evaluations")
    print("="*60)
    print(f"Evaluation Directory: {OUTPUT_DIR}")
    print()
    
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Evaluation directory does not exist: {OUTPUT_DIR}")
    
    # Collect all evaluation files
    results = collect_evaluation_files(OUTPUT_DIR)
    
    if not results:
        print("No evaluation files found!")
        return
    
    # Generate summary
    summary = generate_summary(results, OUTPUT_DIR)
    
    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    save_json_file(summary_path, summary)
    
    # Print statistics
    print_summary_statistics(summary)
    
    print(f"\n✅ Summary generated successfully!")
    print(f"   Saved to: {summary_path.resolve()}")
    print("="*60)


if __name__ == "__main__":
    main()

