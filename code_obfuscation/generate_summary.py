"""
Generate comprehensive summary statistics from all LLM judge evaluation files.

This script processes all evaluation JSON files in the llm_judge_evaluations directory
and generates a complete summary with statistics including mean, min, max, and
standard deviation for each evaluation metric.
"""

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone


DEFAULT_EVALUATION_DIR = Path("llm_judge_evaluations")
DEFAULT_OUTPUT_FILE = Path("llm_judge_evaluations/summary.json")


def load_evaluation_file(path: Path) -> Dict[str, Any]:
    """Load a single evaluation JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_evaluation(evaluation: Dict[str, Any]) -> bool:
    """Check if an evaluation is valid (has judge_evaluation without errors)."""
    if "error" in evaluation:
        return False
    
    judge_eval = evaluation.get("judge_evaluation", {})
    if "error" in judge_eval:
        return False
    
    # Check for required score fields
    required_fields = [
        "semantic_accuracy_raw",
        "semantic_accuracy_new",
        "completeness_raw",
        "completeness_new",
        "transformation_understanding",
        "robustness_to_obfuscation",
    ]
    
    return all(field in judge_eval for field in required_fields)


def extract_scores(evaluation: Dict[str, Any]) -> Dict[str, float] | None:
    """Extract all scores from a valid evaluation."""
    if not is_valid_evaluation(evaluation):
        return None
    
    judge_eval = evaluation.get("judge_evaluation", {})
    
    scores = {
        "semantic_accuracy_raw": judge_eval.get("semantic_accuracy_raw"),
        "semantic_accuracy_new": judge_eval.get("semantic_accuracy_new"),
        "completeness_raw": judge_eval.get("completeness_raw"),
        "completeness_new": judge_eval.get("completeness_new"),
        "transformation_understanding": judge_eval.get("transformation_understanding"),
        "robustness_to_obfuscation": judge_eval.get("robustness_to_obfuscation"),
        "overall_score": judge_eval.get("overall_score"),
    }
    
    # Validate all scores are numeric
    for key, value in scores.items():
        if value is None or not isinstance(value, (int, float)):
            return None
    
    return scores


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of values."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "avg": None,
            "std_dev": None,
        }
    
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": statistics.mean(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def process_all_evaluations(evaluation_dir: Path) -> Dict[str, Any]:
    """Process all evaluation files and generate comprehensive statistics."""
    
    # Collect all evaluation files (excluding summary.json)
    evaluation_files = sorted(
        [
            f for f in evaluation_dir.glob("*.json")
            if f.name != "summary.json" and f.stem.isdigit()
        ],
        key=lambda p: int(p.stem),
    )
    
    print(f"Found {len(evaluation_files)} evaluation files")
    
    all_evaluations = []
    all_scores = {
        "semantic_accuracy_raw": [],
        "semantic_accuracy_new": [],
        "completeness_raw": [],
        "completeness_new": [],
        "transformation_understanding": [],
        "robustness_to_obfuscation": [],
        "overall_score": [],
    }
    
    valid_count = 0
    error_count = 0
    
    # Process each file
    for eval_file in evaluation_files:
        try:
            evaluation = load_evaluation_file(eval_file)
            all_evaluations.append(evaluation)
            
            if is_valid_evaluation(evaluation):
                valid_count += 1
                scores = extract_scores(evaluation)
                if scores:
                    for key, value in scores.items():
                        all_scores[key].append(float(value))
            else:
                error_count += 1
        except Exception as e:
            print(f"Error processing {eval_file.name}: {e}")
            error_count += 1
    
    print(f"Valid evaluations: {valid_count}")
    print(f"Errors: {error_count}")
    
    # Calculate statistics for each metric
    score_statistics = {}
    for metric, values in all_scores.items():
        stats = calculate_statistics(values)
        score_statistics[metric] = stats
    
    # Build average_scores section with std_dev
    average_scores = {}
    for metric, stats in score_statistics.items():
        if stats["count"] > 0:
            average_scores[f"average_{metric}"] = stats["avg"]
            average_scores[f"min_{metric}"] = stats["min"]
            average_scores[f"max_{metric}"] = stats["max"]
            average_scores[f"std_dev_{metric}"] = stats["std_dev"]
    
    # Build score_distributions section
    score_distributions = {}
    for metric, stats in score_statistics.items():
        score_distributions[metric] = {
            "count": stats["count"],
            "min": stats["min"],
            "max": stats["max"],
            "avg": stats["avg"],
            "std_dev": stats["std_dev"],
        }
    
    # Build summary
    summary = {
        "total_evaluated": len(all_evaluations),
        "valid_evaluations": valid_count,
        "error_count": error_count,
        "average_scores": average_scores,
        "score_distributions": score_distributions,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive summary statistics from LLM judge evaluations."
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
        help="Directory containing evaluation JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output path for summary.json file.",
    )
    
    args = parser.parse_args()
    
    if not args.evaluation_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {args.evaluation_dir}")
    
    print("=" * 60)
    print("Generating Summary Statistics")
    print("=" * 60)
    print(f"Evaluation Directory: {args.evaluation_dir}")
    print(f"Output File: {args.output}")
    print("=" * 60)
    print()
    
    # Process all evaluations
    summary = process_all_evaluations(args.evaluation_dir)
    
    # Save summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    
    print()
    print("=" * 60)
    print("Summary Statistics Generated")
    print("=" * 60)
    print(f"Total Evaluated: {summary['total_evaluated']}")
    print(f"Valid Evaluations: {summary['valid_evaluations']}")
    print(f"Error Count: {summary['error_count']}")
    print()
    print("Average Scores (with standard deviations):")
    for key, value in summary["average_scores"].items():
        if key.startswith("average_"):
            metric = key.replace("average_", "")
            avg = value
            std_dev = summary["average_scores"].get(f"std_dev_{metric}", None)
            if std_dev is not None:
                print(f"  {metric}: {avg:.3f} ± {std_dev:.3f}")
            else:
                print(f"  {metric}: {avg:.3f}")
    print()
    print(f"Summary saved to: {args.output.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

