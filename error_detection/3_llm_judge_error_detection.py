"""
Script 3: LLM-as-a-Judge Evaluation of Qwen Error Detection

This script uses ChatGPT 5.1 as a judge to evaluate Qwen's performance in
detecting and analyzing errors in incorrect code.

The evaluation compares:
- Ground truth: What errors were actually introduced (from Claude's generation)
- Qwen's analysis: What errors Qwen detected and how it described them

This provides comprehensive evaluation of Qwen's error detection capabilities.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


GROUND_TRUTH_DIR = Path("mbpp_pro_incorrect_code")
QWEN_ANALYSIS_DIR = Path("qwen_incorrect_code_analysis")
ORIGINAL_CODE_DIR = Path("../mbpp_pro")
OUTPUT_DIR = Path("llm_judge_error_detection")
MODEL = "gpt-5.1"
MAX_COMPLETION_TOKENS = 3000
TEMPERATURE = 0.3
SLEEP_SECONDS = 1.0
ERROR_BUCKETS = [
    "logical_error",
    "off_by_one",
    "edge_case_failure",
    "algorithm_error",
    "boundary_condition",
    "type_error",
    "missing_validation",
]


SYSTEM_PROMPT = """You are an expert evaluator of code analysis systems. Your task is to evaluate how well a code analysis model (Qwen) performed at detecting and explaining errors in incorrect code.

You will receive:
1. **Ground Truth**: Information about what errors were actually introduced into the code (from Claude's error generation)
2. **Original Code**: The correct version of the code (for reference)
3. **Incorrect Code**: The version with errors (what Qwen analyzed)
4. **Qwen's Analysis**: Qwen's detection and explanation of errors

Your evaluation should assess:

1. **Error Detection Accuracy (1-5)**: Did Qwen correctly identify that the code is incorrect? Did it find all the errors that were introduced?

2. **Error Location Precision (1-5)**: How accurately did Qwen identify WHERE the errors occur in the code?

3. **Error Type Classification (1-5)**: Did Qwen correctly classify the TYPE of each error (logical, off-by-one, edge case, etc.)?

4. **Error Explanation Quality (1-5)**: How well did Qwen explain WHAT is wrong and WHY? Is the explanation clear, accurate, and helpful?

5. **Completeness (1-5)**: Did Qwen identify all errors, or did it miss some? Did it find false positives?

6. **Fix Suggestion Quality (1-5)**: If provided, how helpful and correct are Qwen's suggestions for fixing the errors?

7. **Overall Error Detection Performance (1-5)**: Overall assessment of Qwen's ability to detect and understand code errors.

Provide detailed analysis explaining your scores, highlighting:
- What Qwen got right (correctly detected errors)
- What Qwen missed (errors not detected)
- What Qwen got wrong (false positives, incorrect classifications)
- Quality of explanations and insights
- Whether Qwen's error_type_bucket assignments match the ground truth bucket

Respond ONLY with valid JSON matching the specified format."""


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: Path, data: Dict[str, Any]) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_code_from_solution(solution: str) -> str:
    """Extract just the code portion, normalizing indentation."""
    lines = solution.strip().split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return solution.strip()
    
    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    return "\n".join(line[min_indent:] if len(line) > min_indent else line for line in lines).strip()


def build_evaluation_prompt(
    problem_id: int,
    original_raw_problem: str,
    original_new_problem: str,
    original_raw_solution: str,
    original_new_solution: str,
    incorrect_raw_solution: str,
    incorrect_new_solution: str,
    test_code: str,
    error_metadata: Dict[str, Any],
    validation_result: Dict[str, Any],
    qwen_analysis: Dict[str, Any],
) -> str:
    """Build comprehensive evaluation prompt for the judge."""
    
    # Format validation result for display
    validation_text = ""
    if validation_result:
        if validation_result.get("skipped"):
            validation_text = f"Validation: Skipped ({validation_result.get('reason', 'unknown reason')})"
        else:
            passed = validation_result.get("passed", False)
            validation_text = f"Validation: {'PASSED' if passed else 'FAILED'}"
            if not passed:
                validation_text += f" - {validation_result.get('failure_type', 'UnknownError')}: {validation_result.get('failure_message', '')}"
            if validation_result.get("warning"):
                validation_text += f"\nWarning: {validation_result.get('warning')}"
    
    prompt = f"""Evaluate Qwen's error detection performance for Problem ID: {problem_id}

**Original Correct Code (for reference):**

Raw Problem:
{original_raw_problem}

Raw Solution (Correct):
```python
{extract_code_from_solution(original_raw_solution)}
```

New Problem:
{original_new_problem}

New Solution (Correct):
```python
{extract_code_from_solution(original_new_solution)}
```

**Incorrect Code (with errors):**

Raw Solution (Incorrect):
```python
{extract_code_from_solution(incorrect_raw_solution)}
```

New Solution (Incorrect):
```python
{extract_code_from_solution(incorrect_new_solution)}
```

**Test Code (for context):**
```python
{test_code}
```

**Ground Truth Error Information:**

Error Type: {error_metadata.get("error_type", "unknown")}
Error Description: {error_metadata.get("error_description", "")}
Where Error Is: {error_metadata.get("where_error_is", "")}
Expected Failure Cases: {error_metadata.get("expected_failure_cases", "")}

**Validation Result (does incorrect code actually fail tests?):**
{validation_text if validation_text else "No validation data available"}

**Qwen's Analysis:**

{json.dumps(qwen_analysis, indent=2)}

**Your Evaluation Task:**

Compare Qwen's analysis with the ground truth error information. Evaluate:

1. Did Qwen correctly identify that the code is incorrect?
2. Did Qwen find all the errors that were introduced (based on ground truth)?
3. How accurately did Qwen locate the errors?
4. Did Qwen correctly classify the error types?
5. How good are Qwen's explanations of what's wrong?
6. Are there any false positives (errors Qwen found that don't exist)?
7. How helpful are Qwen's fix suggestions (if provided)?

Provide a thorough evaluation with detailed scoring and analysis.

Respond with a JSON object containing:
{{
  "problem_id": {problem_id},
  "raw_solution_evaluation": {{
    "error_detection_accuracy": int (1-5, did Qwen correctly identify errors?),
    "error_location_precision": int (1-5, accuracy of error location),
    "error_type_classification": int (1-5, correctness of error type classification),
    "error_explanation_quality": int (1-5, quality of explanation),
    "completeness": int (1-5, found all errors, no false positives),
    "fix_suggestion_quality": int (1-5, quality of fix suggestions),
    "overall_score": float (average of above scores),
    "detailed_analysis": {{
      "correctly_detected": [string] (errors Qwen found correctly),
      "missed_errors": [string] (errors Qwen missed),
      "false_positives": [string] (incorrect error detections),
      "bucket_alignment": string (does error_type_bucket match ground truth?),
      "location_accuracy": string (analysis of location precision),
      "explanation_quality": string (analysis of explanation clarity),
      "strengths": [string] (what Qwen did well),
      "weaknesses": [string] (where Qwen fell short)
    }}
  }},
  "new_solution_evaluation": {{
    "error_detection_accuracy": int (1-5),
    "error_location_precision": int (1-5),
    "error_type_classification": int (1-5),
    "error_explanation_quality": int (1-5),
    "completeness": int (1-5),
    "fix_suggestion_quality": int (1-5),
    "overall_score": float,
    "detailed_analysis": {{
      "correctly_detected": [string],
      "missed_errors": [string],
      "false_positives": [string],
      "bucket_alignment": string,
      "location_accuracy": string,
      "explanation_quality": string,
      "strengths": [string],
      "weaknesses": [string]
    }}
  }},
  "overall_performance": {{
    "overall_error_detection_performance": float (average of all scores),
    "summary": string (overall assessment of Qwen's error detection capabilities),
    "key_insights": string (notable observations about Qwen's performance),
    "recommendations": string (suggestions for improvement)
  }}
}}"""
    
    return prompt


def query_openai_judge(
    client: OpenAI,
    prompt: str,
    model: str,
    max_completion_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Query OpenAI API for evaluation."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try extracting JSON from markdown
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    return json.loads(content[start:end].strip())
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end > start:
                    candidate = content[start:end].strip()
                    if candidate.startswith("json"):
                        candidate = candidate[4:].strip()
                    return json.loads(candidate)
            
            raise ValueError(f"Failed to parse JSON: {e}. Raw response: {content}")
            
    except Exception as e:
        return {
            "error": str(e),
            "raw_response": content if 'content' in locals() else None,
        }


def coerce_score(value: Any, field: str) -> float:
    """Convert score-like values to floats in [1,5]."""
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Score for '{field}' is not numeric: {value}") from exc
    
    if not 1.0 <= score <= 5.0:
        raise ValueError(f"Score for '{field}' must be between 1 and 5, got {score}")
    return score


def normalize_judge_evaluation(raw_eval: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize judge response."""
    if "error" in raw_eval:
        return raw_eval
    
    normalized = dict(raw_eval)
    
    # Normalize scores for raw_solution_evaluation
    if "raw_solution_evaluation" in normalized:
        raw_eval_data = normalized["raw_solution_evaluation"]
        score_fields = [
            "error_detection_accuracy",
            "error_location_precision",
            "error_type_classification",
            "error_explanation_quality",
            "completeness",
            "fix_suggestion_quality",
        ]
        
        scores = {}
        for field in score_fields:
            if field in raw_eval_data:
                scores[field] = coerce_score(raw_eval_data[field], f"raw.{field}")
        
        if scores:
            raw_eval_data.update(scores)
            raw_eval_data["overall_score"] = sum(scores.values()) / len(scores)
    
    # Normalize scores for new_solution_evaluation
    if "new_solution_evaluation" in normalized:
        new_eval_data = normalized["new_solution_evaluation"]
        score_fields = [
            "error_detection_accuracy",
            "error_location_precision",
            "error_type_classification",
            "error_explanation_quality",
            "completeness",
            "fix_suggestion_quality",
        ]
        
        scores = {}
        for field in score_fields:
            if field in new_eval_data:
                scores[field] = coerce_score(new_eval_data[field], f"new.{field}")
        
        if scores:
            new_eval_data.update(scores)
            new_eval_data["overall_score"] = sum(scores.values()) / len(scores)
    
    # Compute overall performance
    if "raw_solution_evaluation" in normalized and "new_solution_evaluation" in normalized:
        raw_score = normalized["raw_solution_evaluation"].get("overall_score", 0)
        new_score = normalized["new_solution_evaluation"].get("overall_score", 0)
        overall = (raw_score + new_score) / 2.0
        
        if "overall_performance" not in normalized:
            normalized["overall_performance"] = {}
        normalized["overall_performance"]["overall_error_detection_performance"] = overall
    
    return normalized


def process_single_evaluation(
    client: OpenAI,
    problem_id: int,
    ground_truth_dir: Path,
    qwen_analysis_dir: Path,
    original_code_dir: Path,
    model: str,
    max_completion_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Process a single problem evaluation."""
    
    # Load files
    ground_truth_path = ground_truth_dir / f"{problem_id}.json"
    qwen_analysis_path = qwen_analysis_dir / f"{problem_id}.json"
    original_code_path = original_code_dir / f"{problem_id}.json"
    
    # Check for errors
    if not ground_truth_path.exists():
        return {
            "problem_id": problem_id,
            "error": f"Ground truth file not found: {ground_truth_path}",
        }
    
    if not qwen_analysis_path.exists():
        return {
            "problem_id": problem_id,
            "error": f"Qwen analysis file not found: {qwen_analysis_path}",
        }
    
    if not original_code_path.exists():
        return {
            "problem_id": problem_id,
            "error": f"Original code file not found: {original_code_path}",
        }
    
    try:
        ground_truth = load_json_file(ground_truth_path)
        qwen_analysis_data = load_json_file(qwen_analysis_path)
        original_code = load_json_file(original_code_path)
    except Exception as e:
        return {
            "problem_id": problem_id,
            "error": f"Error loading files: {e}",
        }
    
    # Extract data
    original_data = ground_truth.get("original", {})
    incorrect_data = ground_truth.get("incorrect", {})
    error_metadata = ground_truth.get("error_metadata", {})
    validation_result = ground_truth.get("validation", {})
    
    # Extract Qwen analysis with proper error handling
    qwen_analysis_raw = None
    
    # Check for errors in qwen_analysis first
    if "error" in qwen_analysis_data.get("qwen_analysis", {}):
        return {
            "problem_id": problem_id,
            "error": "Qwen analysis contained an error",
            "qwen_error": qwen_analysis_data["qwen_analysis"].get("error"),
        }
    
    # Prefer validated analysis if available
    qwen_validated = qwen_analysis_data.get("qwen_analysis_validated", {})
    if qwen_validated and "error" not in qwen_validated:
        qwen_analysis_raw = qwen_validated.get("parsed")
    
    # Fallback to regular analysis
    if not qwen_analysis_raw:
        qwen_analysis_raw = qwen_analysis_data.get("qwen_analysis", {})
    
    # Handle nested parsed structure (if qwen_analysis itself has a "parsed" key)
    if isinstance(qwen_analysis_raw, dict) and "parsed" in qwen_analysis_raw and "error" not in qwen_analysis_raw:
        qwen_analysis = qwen_analysis_raw.get("parsed", qwen_analysis_raw)
    else:
        qwen_analysis = qwen_analysis_raw
    
    # Check if we have valid analysis
    if not qwen_analysis or (isinstance(qwen_analysis, dict) and "error" in qwen_analysis):
        return {
            "problem_id": problem_id,
            "error": "Could not extract valid Qwen analysis",
            "qwen_analysis_data_keys": list(qwen_analysis_data.keys()),
        }
    
    original_raw_problem = original_data.get("raw_problem", "")
    original_new_problem = original_data.get("new_problem", "")
    original_raw_solution = original_code.get("raw_solution", "")
    original_new_solution = original_code.get("new_solution", "")
    incorrect_raw_solution = incorrect_data.get("raw_solution_incorrect", "")
    incorrect_new_solution = incorrect_data.get("new_solution_incorrect", "")
    test_code = original_data.get("test_code", "") or original_code.get("test_code", "")
    
    # Build prompt
    prompt = build_evaluation_prompt(
        problem_id=problem_id,
        original_raw_problem=original_raw_problem,
        original_new_problem=original_new_problem,
        original_raw_solution=original_raw_solution,
        original_new_solution=original_new_solution,
        incorrect_raw_solution=incorrect_raw_solution,
        incorrect_new_solution=incorrect_new_solution,
        test_code=test_code,
        error_metadata=error_metadata,
        validation_result=validation_result,
        qwen_analysis=qwen_analysis,
    )
    
    # Query OpenAI judge
    evaluation = query_openai_judge(
        client=client,
        prompt=prompt,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
    
    # Normalize evaluation
    if isinstance(evaluation, dict) and "error" not in evaluation:
        try:
            evaluation = normalize_judge_evaluation(evaluation)
        except Exception as exc:
            evaluation = {
                "error": f"Invalid judge response: {exc}",
                "raw_response": evaluation,
            }
    
    # Extract Qwen analysis summary fields
    raw_analysis = qwen_analysis.get("raw_solution_analysis", {})
    new_analysis = qwen_analysis.get("new_solution_analysis", {})
    
    # Build result
    result = {
        "problem_id": problem_id,
        "source_files": {
            "ground_truth": str(ground_truth_path),
            "qwen_analysis": str(qwen_analysis_path),
            "original_code": str(original_code_path),
        },
        "error_metadata": error_metadata,
        "validation_result": validation_result,
        "qwen_analysis_summary": {
            "raw_solution_correct": raw_analysis.get("is_correct"),
            "new_solution_correct": new_analysis.get("is_correct"),
            "raw_errors_found": raw_analysis.get("errors_found", 0),
            "new_errors_found": new_analysis.get("errors_found", 0),
            "raw_confidence": raw_analysis.get("confidence"),
            "new_confidence": new_analysis.get("confidence"),
            "raw_matches_problem": raw_analysis.get("matches_problem"),
            "new_matches_problem": new_analysis.get("matches_problem"),
            "raw_problem_alignment": raw_analysis.get("problem_alignment"),
            "new_problem_alignment": new_analysis.get("problem_alignment"),
        },
        "judge_evaluation": evaluation,
    }
    
    return result


def evaluate_all_files(
    client: OpenAI,
    ground_truth_dir: Path,
    qwen_analysis_dir: Path,
    original_code_dir: Path,
    output_dir: Path,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    sleep_seconds: float,
    start_id: int | None = None,
    end_id: int | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Evaluate all matching files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect problem IDs from ground truth
    ground_truth_files = sorted(
        ground_truth_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    
    problem_ids = sorted([int(f.stem) for f in ground_truth_files if f.stem.isdigit()])
    
    if start_id is not None:
        problem_ids = [pid for pid in problem_ids if pid >= start_id]
    if end_id is not None:
        problem_ids = [pid for pid in problem_ids if pid <= end_id]
    if limit is not None:
        problem_ids = problem_ids[:limit]
    
    if not problem_ids:
        print("No problems found to evaluate.")
        return {}
    
    print(f"Found {len(problem_ids)} problems to evaluate\n")
    print(f"Problem ID range: {min(problem_ids)} to {max(problem_ids)}\n")
    
    all_results = {}
    processed = 0
    errors = 0
    
    for idx, problem_id in enumerate(problem_ids, start=1):
        print(f"[{idx}/{len(problem_ids)}] Evaluating problem {problem_id}...")
        
        output_path = output_dir / f"{problem_id}.json"
        if skip_existing and output_path.exists():
            print("  ⏩ Skipped (existing result).")
            continue
        
        result = process_single_evaluation(
            client=client,
            problem_id=problem_id,
            ground_truth_dir=ground_truth_dir,
            qwen_analysis_dir=qwen_analysis_dir,
            original_code_dir=original_code_dir,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )
        
        all_results[problem_id] = result
        
        # Save individual result
        save_json_file(output_path, result)
        
        if "error" in result:
            print(f"  ⚠️  Error: {result.get('error', 'Unknown error')}")
            errors += 1
        else:
            eval_data = result.get("judge_evaluation", {})
            if "overall_performance" in eval_data:
                overall = eval_data["overall_performance"].get("overall_error_detection_performance", 0)
                print(f"  ✅ Overall Performance: {overall:.2f}/5.0")
            else:
                print(f"  ✅ Saved (check for errors in evaluation)")
            processed += 1
        
        print()
        
        # Rate limiting
        if idx < len(problem_ids) and sleep_seconds > 0:
            time.sleep(sleep_seconds)
    
    # Generate summary
    run_config = {
        "model": model,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "sleep_seconds": sleep_seconds,
        "start_id": start_id,
        "end_id": end_id,
        "limit": limit,
        "skip_existing": skip_existing,
        "ground_truth_dir": str(ground_truth_dir),
        "qwen_analysis_dir": str(qwen_analysis_dir),
        "original_code_dir": str(original_code_dir),
        "output_dir": str(output_dir),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    summary = generate_summary(all_results, run_config)
    summary_path = output_dir / "summary.json"
    save_json_file(summary_path, summary)
    
    print(f"\n{'='*60}")
    print(f"✅ Evaluation Complete!")
    print(f"   Processed: {processed}")
    print(f"   Errors: {errors}")
    print(f"   Results saved to: {output_dir.resolve()}")
    print(f"   Summary saved to: {summary_path.resolve()}")
    print(f"{'='*60}\n")
    
    return all_results


def generate_summary(
    results: Dict[int, Dict[str, Any]],
    run_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Generate summary statistics from all evaluations."""
    
    valid_results = [
        r for r in results.values()
        if "error" not in r and "judge_evaluation" in r
        and "error" not in r.get("judge_evaluation", {})
    ]
    
    if not valid_results:
        return {
            "total_evaluated": len(results),
            "valid_evaluations": 0,
            "error": "No valid evaluations found",
            "run_config": run_config or {},
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
    
    for result in valid_results:
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
    
    summary = {
        "total_evaluated": len(results),
        "valid_evaluations": len(valid_results),
        "error_count": len(results) - len(valid_results),
        "average_scores": averages,
        "run_config": run_config or {},
    }
    
    return summary


def main() -> None:
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set in the environment.\n"
            "Get your API key from: https://platform.openai.com/api-keys"
        )
    
    client = OpenAI(api_key=api_key)
    
    import argparse
    parser = argparse.ArgumentParser(description="LLM judge evaluation of Qwen error detection.")
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    
    print("="*60)
    print("LLM-as-a-Judge Evaluation: Qwen Error Detection")
    print("="*60)
    print(f"Judge Model: {MODEL}")
    print(f"Ground Truth Dir: {GROUND_TRUTH_DIR}")
    print(f"Qwen Analysis Dir: {QWEN_ANALYSIS_DIR}")
    print(f"Original Code Dir: {ORIGINAL_CODE_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    if args.start_id or args.end_id or args.limit:
        print(f"Filters -> start_id: {args.start_id}, end_id: {args.end_id}, limit: {args.limit}")
    if args.skip_existing:
        print("Skip existing results: enabled")
    print("="*60)
    print()
    
    # Validate directories
    for path in [GROUND_TRUTH_DIR, QWEN_ANALYSIS_DIR, ORIGINAL_CODE_DIR]:
        if not path.exists():
            raise FileNotFoundError(f"Required directory not found: {path}")
    
    evaluate_all_files(
        client=client,
        ground_truth_dir=GROUND_TRUTH_DIR,
        qwen_analysis_dir=QWEN_ANALYSIS_DIR,
        original_code_dir=ORIGINAL_CODE_DIR,
        output_dir=OUTPUT_DIR,
        model=MODEL,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        sleep_seconds=SLEEP_SECONDS,
        start_id=args.start_id,
        end_id=args.end_id,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
