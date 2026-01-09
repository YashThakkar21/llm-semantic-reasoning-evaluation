"""
LLM-as-a-Judge Evaluation Script

This script evaluates Qwen's ability to understand code semantics despite obfuscated naming.
It uses OpenAI as a judge to compare ground truth problem descriptions with Qwen's
inferred problem descriptions from obfuscated code.
"""

import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


GROUND_TRUTH_DIR = Path("mbpp_pro")
QWEN_RESPONSE_DIR = Path("qwen_code_obfuscation_description_response")
OBFUSCATED_CODE_DIR = Path("mbpp_pro_code_obfuscation")
OUTPUT_DIR = Path("llm_judge_evaluations")
MODEL = "gpt-5.1"
MAX_COMPLETION_TOKENS = 2000
TEMPERATURE = 0.3
SLEEP_SECONDS = 1.0


SYSTEM_PROMPT = """You are an expert code analysis evaluator. Your task is to evaluate how well a code analysis model (Qwen) understood the semantics of obfuscated Python code by comparing its inferred problem descriptions with ground truth problem descriptions.

The code was intentionally obfuscated with bad naming conventions (e.g., variables like `w2m_9`, `z3q`, `p4w_7s`), but the logic and structure remain unchanged. The goal is to assess whether Qwen can understand what the code does despite poor naming semantics.

You will receive:
1. Ground truth problem descriptions (raw_problem and new_problem) from the original dataset
2. Qwen's inferred problem descriptions (raw_problem and new_problem) from analyzing obfuscated code
3. The obfuscated code snippets themselves

Evaluate Qwen's performance on the following dimensions:

1. **Semantic Accuracy (1-5)**: Does Qwen correctly understand what the code does at the semantic level? Does it capture the core purpose, operations, inputs/outputs, and algorithm?
   
2. **Completeness (1-5)**: Does Qwen capture all key aspects of the problem? Are important details, constraints, or nuances mentioned in the ground truth also present in Qwen's description?
   
3. **Correctness of Transformation Understanding (1-5)**: For the raw_problem → new_problem transformation, does Qwen correctly understand how the problem evolved? Does it capture the relationship between the two problems?
   
4. **Robustness to Obfuscation (1-5)**: Given that the code has intentionally confusing names, how well did Qwen infer the semantics? This measures whether Qwen relies on code structure/logic vs. naming conventions.

Provide detailed analysis explaining your scores and highlighting:
- What Qwen got right
- What Qwen missed or misunderstood
- Any notable insights about Qwen's understanding of code semantics

Respond ONLY with valid JSON matching the specified format."""


def build_evaluation_prompt(
    ground_truth_raw_problem: str,
    ground_truth_new_problem: str,
    qwen_raw_problem: str,
    qwen_new_problem: str,
    obfuscated_raw_solution: str,
    obfuscated_new_solution: str,
    problem_id: int,
) -> str:
    """Build a comprehensive evaluation prompt for the judge."""
    
    prompt = f"""Evaluate Qwen's understanding of obfuscated code for Problem ID: {problem_id}

**Ground Truth Problem Descriptions:**

Raw Problem (Ground Truth):
{ground_truth_raw_problem}

New Problem (Ground Truth):
{ground_truth_new_problem}

**Qwen's Inferred Problem Descriptions:**

Raw Problem (Qwen's Inference):
{qwen_raw_problem}

New Problem (Qwen's Inference):
{qwen_new_problem}

**Obfuscated Code Snippets (for context):**

Raw Solution (Obfuscated):
```python
{obfuscated_raw_solution}
```

New Solution (Obfuscated):
```python
{obfuscated_new_solution}
```

**Your Evaluation Task:**

Provide a thorough evaluation comparing Qwen's inferred descriptions with the ground truth. Assess:
1. Whether Qwen correctly understood the semantic meaning of the code
2. Whether Qwen captured all important aspects and details
3. Whether Qwen correctly understood the transformation from raw_problem to new_problem
4. How well Qwen handled the obfuscated naming conventions

Respond with a JSON object containing:
{{
  "problem_id": {problem_id},
  "semantic_accuracy_raw": int (1-5, rating for raw_problem understanding),
  "semantic_accuracy_new": int (1-5, rating for new_problem understanding),
  "completeness_raw": int (1-5, rating for completeness of raw_problem description),
  "completeness_new": int (1-5, rating for completeness of new_problem description),
  "transformation_understanding": int (1-5, rating for understanding the raw→new transformation),
  "robustness_to_obfuscation": int (1-5, overall rating for handling obfuscated names),
  "overall_score": float (average of all scores),
  "detailed_analysis": {{
    "raw_problem_evaluation": string (detailed analysis of raw_problem comparison),
    "new_problem_evaluation": string (detailed analysis of new_problem comparison),
    "transformation_analysis": string (analysis of how well Qwen understood the problem evolution),
    "obfuscation_handling": string (analysis of Qwen's ability to infer semantics despite bad names),
    "strengths": [string] (what Qwen did well),
    "weaknesses": [string] (what Qwen missed or misunderstood),
    "key_insights": string (notable observations about Qwen's code understanding)
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
        
        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
                return json.loads(content)
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
                return json.loads(content)
            else:
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
    """Validate and normalize judge response, computing overall score if needed."""
    if "error" in raw_eval:
        return raw_eval
    
    required_fields = [
        "semantic_accuracy_raw",
        "semantic_accuracy_new",
        "completeness_raw",
        "completeness_new",
        "transformation_understanding",
        "robustness_to_obfuscation",
    ]
    
    missing = [f for f in required_fields if f not in raw_eval]
    if missing:
        raise ValueError(f"Judge response missing fields: {missing}")
    
    normalized = dict(raw_eval)
    scores = {
        field: coerce_score(raw_eval[field], field)
        for field in required_fields
    }
    
    normalized.update(scores)
    
    normalized["overall_score"] = float(
        raw_eval.get(
            "overall_score",
            sum(scores.values()) / len(scores),
        )
    )
    normalized["computed_overall_score"] = sum(scores.values()) / len(scores)
    
    return normalized


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_problem_description(problem_str: str) -> str:
    """Extract the problem description from a problem string.
    
    The problem string may contain the function signature. We want just the description.
    """
    # Remove function signature if present (lines starting with "def")
    lines = problem_str.split("\n")
    description_lines = []
    for line in lines:
        if line.strip().startswith("def "):
            break
        description_lines.append(line)
    return "\n".join(description_lines).strip()


def process_single_evaluation(
    client: OpenAI,
    problem_id: int,
    ground_truth_dir: Path,
    qwen_response_dir: Path,
    obfuscated_code_dir: Path,
    model: str,
    max_completion_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Process a single problem evaluation."""
    
    # Load files
    ground_truth_path = ground_truth_dir / f"{problem_id}.json"
    qwen_response_path = qwen_response_dir / f"{problem_id}.json"
    obfuscated_code_path = obfuscated_code_dir / f"{problem_id}.json"
    
    try:
        ground_truth = load_json_file(ground_truth_path)
        qwen_response = load_json_file(qwen_response_path)
        obfuscated_code = load_json_file(obfuscated_code_path)
    except FileNotFoundError as e:
        return {
            "problem_id": problem_id,
            "error": f"Missing file: {e}",
        }
    
    # Extract problem descriptions
    gt_raw_problem = extract_problem_description(ground_truth.get("raw_problem", ""))
    gt_new_problem = extract_problem_description(ground_truth.get("new_problem", ""))
    
    qwen_raw_problem = qwen_response.get("response", {}).get("raw_problem", "")
    qwen_new_problem = qwen_response.get("response", {}).get("new_problem", "")
    
    # Extract obfuscated code
    obf_raw_solution = obfuscated_code.get("raw_solution", "").strip()
    obf_new_solution = obfuscated_code.get("new_solution", "").strip()
    
    # Build prompt and query OpenAI
    prompt = build_evaluation_prompt(
        ground_truth_raw_problem=gt_raw_problem,
        ground_truth_new_problem=gt_new_problem,
        qwen_raw_problem=qwen_raw_problem,
        qwen_new_problem=qwen_new_problem,
        obfuscated_raw_solution=obf_raw_solution,
        obfuscated_new_solution=obf_new_solution,
        problem_id=problem_id,
    )
    
    evaluation = query_openai_judge(
        client=client,
        prompt=prompt,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
    
    # Validate and normalize judge response
    if isinstance(evaluation, dict) and "error" not in evaluation:
        try:
            evaluation = normalize_judge_evaluation(evaluation)
        except Exception as exc:
            evaluation = {
                "error": f"Invalid judge response: {exc}",
                "raw_response": evaluation,
            }
    
    # Build result with metadata
    result = {
        "problem_id": problem_id,
        "source_files": {
            "ground_truth": str(ground_truth_path),
            "qwen_response": str(qwen_response_path),
            "obfuscated_code": str(obfuscated_code_path),
        },
        "ground_truth": {
            "raw_problem": gt_raw_problem,
            "new_problem": gt_new_problem,
        },
        "qwen_inference": {
            "raw_problem": qwen_raw_problem,
            "new_problem": qwen_new_problem,
        },
        "judge_evaluation": evaluation,
    }
    
    return result


def evaluate_all_files(
    client: OpenAI,
    ground_truth_dir: Path,
    qwen_response_dir: Path,
    obfuscated_code_dir: Path,
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
    
    # Collect all problem IDs from ground truth
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
        print("No problems found to evaluate with the current filters.")
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
            qwen_response_dir=qwen_response_dir,
            obfuscated_code_dir=obfuscated_code_dir,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )
        
        all_results[problem_id] = result
        
        # Save individual result
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        if "error" in result:
            print(f"  ⚠️  Error: {result.get('error', 'Unknown error')}")
            errors += 1
        else:
            eval_data = result.get("judge_evaluation", {})
            if "overall_score" in eval_data:
                print(f"  ✅ Overall Score: {eval_data['overall_score']:.2f}/5.0")
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
        "qwen_response_dir": str(qwen_response_dir),
        "obfuscated_code_dir": str(obfuscated_code_dir),
        "output_dir": str(output_dir),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    summary = generate_summary(all_results, run_config)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
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
        }
    
    # Collect all scores
    scores = {
        "semantic_accuracy_raw": [],
        "semantic_accuracy_new": [],
        "completeness_raw": [],
        "completeness_new": [],
        "transformation_understanding": [],
        "robustness_to_obfuscation": [],
        "overall_score": [],
    }
    
    for result in valid_results:
        eval_data = result.get("judge_evaluation", {})
        for key in scores.keys():
            if key in eval_data:
                scores[key].append(eval_data[key])
    
    # Calculate averages
    averages = {}
    for key, values in scores.items():
        if values:
            averages[f"average_{key}"] = sum(values) / len(values)
            averages[f"min_{key}"] = min(values)
            averages[f"max_{key}"] = max(values)
    
    summary = {
        "total_evaluated": len(results),
        "valid_evaluations": len(valid_results),
        "error_count": len(results) - len(valid_results),
        "average_scores": averages,
        "score_distributions": {
            key: {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": sum(values) / len(values) if values else None,
            }
            for key, values in scores.items()
        },
        "run_config": run_config or {},
    }
    
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation of Qwen obfuscated code understanding."
    )
    parser.add_argument("--ground-truth-dir", type=Path, default=GROUND_TRUTH_DIR)
    parser.add_argument("--qwen-response-dir", type=Path, default=QWEN_RESPONSE_DIR)
    parser.add_argument("--obfuscated-code-dir", type=Path, default=OBFUSCATED_CODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=MAX_COMPLETION_TOKENS,
        help="Max completion tokens for the judge model (use for GPT-4.1 / GPT-5.1).",
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--sleep-seconds", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip problems that already have an evaluation JSON in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set in the environment.\n"
            "Get your API key from: https://platform.openai.com/api-keys"
        )
    
    client = OpenAI(api_key=api_key)
    
    print("="*60)
    print("LLM-as-a-Judge Evaluation: Qwen Code Understanding")
    print("="*60)
    print(f"Judge Model: {args.model}")
    print(f"Ground Truth Dir: {args.ground_truth_dir}")
    print(f"Qwen Response Dir: {args.qwen_response_dir}")
    print(f"Obfuscated Code Dir: {args.obfuscated_code_dir}")
    print(f"Output Dir: {args.output_dir}")
    if args.start_id or args.end_id or args.limit:
        print(
            f"Filters -> start_id: {args.start_id}, end_id: {args.end_id}, limit: {args.limit}"
        )
    if args.skip_existing:
        print("Skip existing results: enabled")
    print("="*60)
    print()
    
    for path in [args.ground_truth_dir, args.qwen_response_dir, args.obfuscated_code_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Required directory not found: {path}")
    
    evaluate_all_files(
        client=client,
        ground_truth_dir=args.ground_truth_dir,
        qwen_response_dir=args.qwen_response_dir,
        obfuscated_code_dir=args.obfuscated_code_dir,
        output_dir=args.output_dir,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        sleep_seconds=args.sleep_seconds,
        start_id=args.start_id,
        end_id=args.end_id,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
