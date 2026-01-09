"""
Script 2: Qwen Analysis of Incorrect Code

This script sends the incorrect code variants to Qwen2.5-Coder-32B-Instruct
to analyze whether the code is correct or incorrect, and if incorrect,
identify where the errors are and explain what's wrong.

The goal is to evaluate Qwen's ability to detect and understand bugs in code.
"""

import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv


INPUT_DIR = Path("mbpp_pro_incorrect_code")
OUTPUT_DIR = Path("qwen_incorrect_code_analysis")
MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_TOKENS = 2000
TEMPERATURE = 0.2  # Lower temperature for more deterministic analysis
SLEEP_SECONDS = 0.5
RETRY_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0
ERROR_BUCKETS = {
    "logical_error",
    "off_by_one",
    "edge_case_failure",
    "algorithm_error",
    "boundary_condition",
    "type_error",
    "missing_validation",
}


SYSTEM_PROMPT = """You are an expert Python code reviewer and bug detector. Your task is to analyze Python code and determine:

1. Whether the code is CORRECT or INCORRECT
2. If incorrect, identify all errors and their locations
3. Explain what is wrong and why
4. Describe the impact of each error
5. Suggest how to fix the errors

Be thorough and precise in your analysis. Focus on:
- Logic errors (wrong algorithm, incorrect conditions)
- Off-by-one errors and indexing mistakes
- Edge case failures
- Type/format errors
- Missing validations
- Boundary condition issues

Respond in structured JSON format as specified."""


def ensure_environment() -> tuple[str, dict]:
    """Load environment variables and return API URL and headers."""
    load_dotenv()
    url = os.getenv("HF_ENDPOINT_URL")
    token = os.getenv("HF_TOKEN")
    
    if not url or not token:
        raise RuntimeError(
            "HF_ENDPOINT_URL and HF_TOKEN must be set in the environment."
        )
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    return url, headers


def build_analysis_prompt(
    raw_problem: str,
    raw_solution_incorrect: str,
    new_problem: str,
    new_solution_incorrect: str,
    test_code: str = "",
) -> list:
    """Build messages for Qwen to analyze incorrect code."""
    
    user_prompt = f"""Analyze the following Python code and determine if it is correct or incorrect.

**Problem Descriptions:**

Raw Problem:
{raw_problem}

New Problem:
{new_problem}

**Code to Analyze:**

Raw Solution:
```python
{raw_solution_incorrect}
```

New Solution:
```python
{new_solution_incorrect}
```

{f"**Test Cases (for reference):**\n```python\n{test_code}\n```" if test_code else ""}

**Your Analysis Task:**

1. Determine if each solution is CORRECT or INCORRECT
2. For incorrect code, identify ALL errors:
   - Exact location (function name, line description, variable names)
   - Type of error (logical, off-by-one, edge case, etc.)
   - What is wrong and why
   - Which test cases or inputs would fail
   - How to fix the error

3. Be thorough - even subtle errors matter
4. Consider edge cases and boundary conditions
5. Check if the code matches the problem description

Respond with a JSON object containing:
 {{
  "raw_solution_analysis": {{
    "is_correct": boolean,
    "errors_found": integer (0 if correct),
    "confidence": float (0.0-1.0 for this assessment),
    "matches_problem": boolean (does the code solve the stated raw_problem?),
    "problem_alignment": "string (how well it aligns with raw_problem requirements)",
    "test_alignment": "string (expected failing cases or why tests would fail)",
    "errors": [
      {{
        "location": "string (where in the code)",
        "error_type": "string (free-form detail)",
        "error_type_bucket": "string (one of: logical_error, off_by_one, edge_case_failure, algorithm_error, boundary_condition, type_error, missing_validation)",
        "description": "string (what is wrong)",
        "explanation": "string (why it's wrong and impact)",
        "failing_cases": "string (what inputs/test cases would fail)",
        "fix_suggestion": "string (how to fix it)"
      }}
    ],
    "overall_assessment": "string (summary of code correctness)"
  }},
  "new_solution_analysis": {{
    "is_correct": boolean,
    "errors_found": integer (0 if correct),
    "confidence": float (0.0-1.0),
    "matches_problem": boolean (does the code solve the stated new_problem?),
    "problem_alignment": "string",
    "test_alignment": "string",
    "errors": [
      {{
        "location": "string",
        "error_type": "string",
        "error_type_bucket": "string (one of: logical_error, off_by_one, edge_case_failure, algorithm_error, boundary_condition, type_error, missing_validation)",
        "description": "string",
        "explanation": "string",
        "failing_cases": "string",
        "fix_suggestion": "string"
      }}
    ],
    "overall_assessment": "string"
  }},
  "comparison_note": "string (note on relationship between raw and new solution errors)"
}}

Respond with ONLY the JSON object, no markdown formatting or additional text."""
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def query_qwen(url: str, headers: dict, messages: list) -> Dict[str, Any]:
    """Query Qwen API and return parsed response."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"Request error: {str(exc)}"}
    
    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError) as exc:
        return {"error": f"Unexpected response format: {exc}", "raw": response.text}
    
    # Try to parse JSON
    try:
        parsed = json.loads(content)
        return {"parsed": parsed, "raw_response_text": content}
    except json.JSONDecodeError:
        # Try extracting JSON from markdown
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                try:
                    parsed = json.loads(content[start:end].strip())
                    return {"parsed": parsed, "raw_response_text": content}
                except json.JSONDecodeError:
                    pass
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                candidate = content[start:end].strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    parsed = json.loads(candidate)
                    return {"parsed": parsed, "raw_response_text": content}
                except json.JSONDecodeError:
                    pass
        
        # Return raw content if JSON parsing fails
        return {
            "error": "Failed to parse JSON response",
            "raw_response_text": content,
        }


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: Path, data: Dict[str, Any]) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_single_file(
    url: str,
    headers: dict,
    input_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Process a single file to analyze incorrect code."""
    
    # Load incorrect code data
    incorrect_data = load_json_file(input_path)
    
    # Skip error files
    if "error" in incorrect_data and incorrect_data.get("original_file") is None:
        return {
            "problem_id": incorrect_data.get("id"),
            "error": "Input file is an error record, skipping",
            "source_file": str(input_path),
        }
    
    original = incorrect_data.get("original", {})
    incorrect = incorrect_data.get("incorrect", {})
    
    raw_problem = original.get("raw_problem", "")
    raw_solution_incorrect = incorrect.get("raw_solution_incorrect", "")
    new_problem = original.get("new_problem", "")
    new_solution_incorrect = incorrect.get("new_solution_incorrect", "")
    test_code = original.get("test_code", "")
    
    if not raw_solution_incorrect and not new_solution_incorrect:
        return {
            "problem_id": incorrect_data.get("id"),
            "error": "No incorrect solutions found in input file",
            "source_file": str(input_path),
        }
    
    # Build messages
    messages = build_analysis_prompt(
        raw_problem=raw_problem,
        raw_solution_incorrect=raw_solution_incorrect,
        new_problem=new_problem,
        new_solution_incorrect=new_solution_incorrect,
        test_code=test_code,
    )
    
    # Query Qwen with retries
    attempts_remaining = RETRY_ATTEMPTS
    qwen_response = None
    last_error = None
    while attempts_remaining > 0:
        try:
            qwen_response = query_qwen(url, headers, messages)
            break
        except Exception as exc:  # pragma: no cover - runtime guard
            attempts_remaining -= 1
            last_error = exc
            if attempts_remaining <= 0:
                return {
                    "problem_id": incorrect_data.get("id"),
                    "error": f"Failed after retries: {exc}",
                    "source_file": str(input_path),
                }
            time.sleep(BACKOFF_SECONDS)
    
    # Build output structure
    result = {
        "problem_id": incorrect_data.get("id"),
        "source_file": str(input_path),
        "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
        "model_used": MODEL,
        "prompt_used": messages,
        
        # Reference to original incorrect code
        "incorrect_code_metadata": {
            "error_type": incorrect_data.get("error_metadata", {}).get("error_type", ""),
            "error_description": incorrect_data.get("error_metadata", {}).get("error_description", ""),
            "where_error_is": incorrect_data.get("error_metadata", {}).get("where_error_is", ""),
        },
        
        # Qwen's analysis
        "qwen_analysis": qwen_response,
        
        # Original problems for reference
        "problem_descriptions": {
            "raw_problem": raw_problem,
            "new_problem": new_problem,
        },
    }
    
    return result


def collect_input_files(input_dir: Path) -> list[Path]:
    """Collect all JSON files from input directory, excluding error files."""
    files = sorted(
        (
            path
            for path in input_dir.glob("*.json")
            if path.is_file() and not path.name.endswith("_ERROR.json")
        ),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    return files


def validate_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Qwen analysis structure and flag issues."""
    if not analysis or "error" in analysis:
        return {"error": analysis.get("error") if isinstance(analysis, dict) else "empty analysis"}
    
    parsed = analysis.get("parsed") if isinstance(analysis, dict) else analysis
    if not isinstance(parsed, dict):
        return {"error": "parsed analysis is not a dict", "raw": analysis}
    
    def validate_solution(key: str) -> Dict[str, Any]:
        block = parsed.get(key, {})
        if not isinstance(block, dict):
            return {"error": f"{key} not a dict"}
        required_bool = ["is_correct", "matches_problem"]
        required_int = ["errors_found"]
        required_float = ["confidence"]
        required_str = ["overall_assessment", "problem_alignment", "test_alignment"]
        missing = []
        for rb in required_bool:
            if rb not in block or not isinstance(block.get(rb), bool):
                missing.append(rb)
        for ri in required_int:
            if ri not in block or not isinstance(block.get(ri), int):
                missing.append(ri)
        for rf in required_float:
            if rf not in block:
                missing.append(rf)
            else:
                try:
                    float(block[rf])
                except Exception:
                    missing.append(rf)
        for rs in required_str:
            if rs not in block or not isinstance(block.get(rs), str):
                missing.append(rs)
        errors_list = block.get("errors", [])
        if not isinstance(errors_list, list):
            missing.append("errors")
        bucket_issues = []
        if isinstance(errors_list, list):
            for idx, err in enumerate(errors_list):
                if not isinstance(err, dict):
                    bucket_issues.append(f"{idx}: not a dict")
                    continue
                bucket = err.get("error_type_bucket")
                if bucket is None:
                    bucket_issues.append(f"{idx}: missing error_type_bucket")
                elif bucket not in ERROR_BUCKETS:
                    bucket_issues.append(f"{idx}: invalid bucket '{bucket}'")
        return {
            "missing_fields": missing,
            "errors_list_count": len(errors_list) if isinstance(errors_list, list) else None,
            "bucket_issues": bucket_issues,
        }
    
    validation = {
        "raw_solution_validation": validate_solution("raw_solution_analysis"),
        "new_solution_validation": validate_solution("new_solution_analysis"),
    }
    validation["has_missing_fields"] = any(
        v.get("missing_fields") for v in validation.values() if isinstance(v, dict)
    )
    return {"parsed": parsed, "validation": validation, "raw_response_text": analysis.get("raw_response_text")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze incorrect code with Qwen.")
    parser.add_argument("--start-id", type=int, default=None, help="Start ID inclusive.")
    parser.add_argument("--end-id", type=int, default=None, help="End ID inclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url, headers = ensure_environment()
    
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    input_files = collect_input_files(INPUT_DIR)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found in input directory: {INPUT_DIR}")
    
    if args.start_id is not None:
        input_files = [p for p in input_files if int(p.stem) >= args.start_id]
    if args.end_id is not None:
        input_files = [p for p in input_files if int(p.stem) <= args.end_id]
    if args.limit is not None:
        input_files = input_files[:args.limit]
    
    print("="*60)
    print("Qwen Analysis of Incorrect Code")
    print("="*60)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print(f"Total files to process: {len(input_files)}")
    if args.start_id or args.end_id or args.limit:
        print(f"Filters -> start_id: {args.start_id}, end_id: {args.end_id}, limit: {args.limit}")
    if args.skip_existing:
        print("Skip existing outputs: enabled")
    print("="*60)
    print()
    
    processed = 0
    errors = 0
    
    for idx, input_file in enumerate(input_files, start=1):
        problem_id = int(input_file.stem) if input_file.stem.isdigit() else idx
        output_file = OUTPUT_DIR / f"{problem_id}.json"
        
        print(f"[{idx}/{len(input_files)}] Analyzing {input_file.name}...")
        
        if args.skip_existing and output_file.exists():
            print("  ⏩ Skipped (existing output).")
            continue
        
        try:
            result = process_single_file(
                url=url,
                headers=headers,
                input_path=input_file,
                output_path=output_file,
            )
            
            # Validate analysis structure
            if "qwen_analysis" in result:
                result["qwen_analysis_validated"] = validate_analysis(result["qwen_analysis"])
            
            save_json_file(output_file, result)
            
            if "error" in result:
                print(f"  ⚠️  Warning: {result.get('error')}")
                errors += 1
            else:
                # Extract summary from analysis
                qwen_analysis = result.get("qwen_analysis_validated", {})
                if qwen_analysis and not qwen_analysis.get("error"):
                    parsed = qwen_analysis.get("parsed", {})
                    raw_analysis = parsed.get("raw_solution_analysis", {})
                    new_analysis = parsed.get("new_solution_analysis", {})
                    raw_correct = raw_analysis.get("is_correct", None)
                    new_correct = new_analysis.get("is_correct", None)
                    raw_errors = raw_analysis.get("errors_found", 0)
                    new_errors = new_analysis.get("errors_found", 0)
                    if qwen_analysis.get("validation", {}).get("has_missing_fields"):
                        print("  ⚠️ Validation: missing expected fields in analysis.")
                    
                    print(f"  ✅ Raw: {'CORRECT' if raw_correct else 'INCORRECT'} ({raw_errors} errors)")
                    print(f"     New: {'CORRECT' if new_correct else 'INCORRECT'} ({new_errors} errors)")
                else:
                    print(f"  ❌ Qwen API Error: {qwen_analysis.get('error') if qwen_analysis else 'Unknown error'}")
                    errors += 1
            
            processed += 1
            
        except Exception as exc:
            print(f"  ❌ Error: {exc}")
            errors += 1
            
            # Save error record
            error_file = OUTPUT_DIR / f"{problem_id}_ERROR.json"
            error_data = {
                "problem_id": problem_id,
                "source_file": str(input_file),
                "error": str(exc),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            save_json_file(error_file, error_data)
        
        print()
        
        # Rate limiting
        if idx < len(input_files) and SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
    
    print("="*60)
    print(f"✅ Analysis Complete!")
    print(f"   Processed: {processed}")
    print(f"   Errors: {errors}")
    print(f"   Results saved to: {OUTPUT_DIR.resolve()}")
    print("="*60)


if __name__ == "__main__":
    main()
