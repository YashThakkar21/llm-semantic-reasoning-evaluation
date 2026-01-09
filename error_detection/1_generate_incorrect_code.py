"""
Script 1: Generate Incorrect Code Dataset

This script uses Claude to generate incorrect versions of code from the mbpp_pro dataset.
The goal is to create a diverse set of bugs including:
- Logical errors (wrong algorithm, incorrect conditionals)
- Off-by-one errors
- Edge case handling failures
- Wrong data structure usage
- Type/format errors
- Missing validations

Each incorrect code variant will include metadata about the error type introduced.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic, AnthropicError
from dotenv import load_dotenv


INPUT_DIR = Path("mbpp_pro")
OUTPUT_DIR = Path("mbpp_pro_incorrect_code")
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 8000
TEMPERATURE = 0.7  # Higher temperature for more diverse error generation
SLEEP_SECONDS = 1.0
RETRY_ATTEMPTS = 3
BACKOFF_SECONDS = 3.0

# Error types to introduce (will be randomized per file)
ERROR_TYPES = [
    "logical_error",
    "off_by_one",
    "edge_case_failure",
    "algorithm_error",
    "boundary_condition",
    "type_error",
    "missing_validation",
]


SYSTEM_PROMPT = """You are an expert at introducing realistic bugs into Python code. Your task is to create incorrect versions of code that contain meaningful errors while maintaining valid Python syntax.

The errors should be realistic bugs that could occur in actual development, such as:
- Logical errors (wrong conditions, incorrect calculations)
- Off-by-one errors (indexing mistakes)
- Edge case failures (missing boundary checks, empty inputs)
- Algorithm errors (wrong approach to solving the problem)
- Type/format issues (incorrect data handling)
- Missing validations (assumptions about inputs)

CRITICAL REQUIREMENTS:
1. The code must remain syntactically valid Python
2. The error should be subtle enough that it's not immediately obvious
3. The function signature and structure should remain similar to the original
4. Do NOT simply break the syntax or make trivial mistakes
5. The code should fail on some test cases but might pass others
6. Preserve comments, imports, and general structure

Respond with ONLY valid JSON matching the specified format."""


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
    """Extract just the code portion, removing leading whitespace."""
    lines = solution.strip().split("\n")
    # Find minimum indentation (excluding empty lines)
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return solution.strip()
    
    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    # Remove minimum indentation from all lines
    return "\n".join(line[min_indent:] if len(line) > min_indent else line for line in lines).strip()


def build_error_generation_prompt(
    raw_problem: str,
    raw_solution: str,
    new_problem: str,
    new_solution: str,
    error_type: str,
) -> str:
    """Build a prompt to generate incorrect code with a specific error type."""
    
    error_descriptions = {
        "logical_error": "introduce a logical error such as wrong conditional logic, incorrect mathematical operation, or flawed algorithm",
        "off_by_one": "introduce an off-by-one error in indexing, counting, or loop boundaries",
        "edge_case_failure": "introduce a bug that fails on edge cases like empty inputs, single elements, or boundary values",
        "algorithm_error": "use a fundamentally different (and incorrect) algorithmic approach",
        "boundary_condition": "introduce an error related to boundary conditions or limits",
        "type_error": "introduce incorrect handling of data types or format conversions",
        "missing_validation": "remove or incorrectly implement input validation or assumptions",
    }
    
    error_description = error_descriptions.get(error_type, "introduce a realistic bug")
    
    prompt = f"""Generate incorrect versions of the following Python code that contain {error_description}.

**Original Problem Descriptions:**

Raw Problem:
{raw_problem}

New Problem:
{new_problem}

**Original Correct Code:**

Raw Solution:
```python
{extract_code_from_solution(raw_solution)}
```

New Solution:
```python
{extract_code_from_solution(new_solution)}
```

**Your Task:**

Create incorrect versions of both solutions that contain {error_description}. The bugs should:
1. Be realistic and non-trivial
2. Maintain valid Python syntax
3. Preserve the function structure and signature
4. Fail on some test cases while potentially passing others
5. Be subtle enough to require careful code review to identify

Respond with a JSON object containing:
{{
  "raw_solution_incorrect": "the incorrect version of raw_solution code",
  "new_solution_incorrect": "the incorrect version of new_solution code",
  "error_type": "{error_type}",
  "error_description": "a brief description of the specific error introduced",
  "where_error_is": "where in the code the error occurs (function name, line description)",
  "expected_failure_cases": "description of test cases or inputs where this error would cause failures"
}}

Include ONLY the JSON object, no markdown formatting or explanations."""
    
    return prompt


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract JSON from Claude's response, handling various formats."""
    response_text = response_text.strip()
    
    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end > start:
            try:
                return json.loads(response_text[start:end].strip())
            except json.JSONDecodeError:
                pass
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        if end > start:
            candidate = response_text[start:end].strip()
            # Remove language identifier if present
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    
    # Try finding JSON object boundaries
    start_idx = None
    depth = 0
    for idx, char in enumerate(response_text):
        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidate = response_text[start_idx:idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
    
    raise ValueError("Could not extract valid JSON from Claude response.")


def query_claude(
    client: Anthropic,
    prompt: str,
    error_type: str,
) -> Dict[str, Any]:
    """Query Claude API to generate incorrect code."""
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=TEMPERATURE,
        )
        
        # Extract text content
        text_parts = []
        for part in message.content:
            if part.type == "text":
                text_parts.append(part.text)
        
        if not text_parts:
            raise ValueError("Claude response did not contain any text segments.")
        
        response_text = "".join(text_parts)
        result = extract_json_from_response(response_text)
        
        # Ensure error_type is set
        result["error_type"] = error_type
        
        return {
            "parsed": result,
            "raw_response_text": response_text,
        }
        
    except Exception as e:
        raise RuntimeError(f"Claude API error: {e}") from e


def process_single_file(
    client: Anthropic,
    input_path: Path,
    output_path: Path,
    error_type: str,
) -> Dict[str, Any]:
    """Process a single file to generate incorrect code."""
    
    # Load original data
    original_data = load_json_file(input_path)
    
    raw_problem = original_data.get("raw_problem", "")
    raw_solution = original_data.get("raw_solution", "")
    new_problem = original_data.get("new_problem", "")
    new_solution = original_data.get("new_solution", "")
    
    if not raw_solution and not new_solution:
        raise ValueError(f"No solutions found in {input_path}")
    
    # Build prompt
    prompt = build_error_generation_prompt(
        raw_problem=raw_problem,
        raw_solution=raw_solution,
        new_problem=new_problem,
        new_solution=new_solution,
        error_type=error_type,
    )
    
    # Query Claude with retries
    attempts_remaining = RETRY_ATTEMPTS
    claude_response = None
    
    while attempts_remaining > 0:
        try:
            claude_response = query_claude(client, prompt, error_type)
            break
        except (AnthropicError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            attempts_remaining -= 1
            if attempts_remaining <= 0:
                raise RuntimeError(
                    f"Failed to generate incorrect code for {input_path.name} after {RETRY_ATTEMPTS} attempts: {exc}"
                ) from exc
            time.sleep(BACKOFF_SECONDS)
    
    # Build output structure
    output_data = {
        "id": original_data.get("id"),
        "original_file": str(input_path),
        "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        "model_used": MODEL,
        "error_type": error_type,
        "generation_artifacts": {
            "prompt": prompt,
            "raw_response_text": claude_response.get("raw_response_text", "") if claude_response else "",
        },
        
        # Original data
        "original": {
            "raw_problem": raw_problem,
            "raw_solution": raw_solution,
            "new_problem": new_problem,
            "new_solution": new_solution,
            "test_code": original_data.get("test_code", ""),
        },
        
        # Incorrect versions
        "incorrect": {
            "raw_solution_incorrect": (claude_response.get("parsed", {}) if claude_response else {}).get("raw_solution_incorrect", ""),
            "new_solution_incorrect": (claude_response.get("parsed", {}) if claude_response else {}).get("new_solution_incorrect", ""),
        },
        
        # Error metadata
        "error_metadata": {
            "error_type": (claude_response.get("parsed", {}) if claude_response else {}).get("error_type", error_type),
            "error_description": (claude_response.get("parsed", {}) if claude_response else {}).get("error_description", ""),
            "where_error_is": (claude_response.get("parsed", {}) if claude_response else {}).get("where_error_is", ""),
            "expected_failure_cases": (claude_response.get("parsed", {}) if claude_response else {}).get("expected_failure_cases", ""),
        },
    }
    
    # Validate that the incorrect code actually fails provided tests (when available)
    validation_result = run_incorrect_tests(
        output_data["incorrect"]["raw_solution_incorrect"],
        output_data["incorrect"]["new_solution_incorrect"],
        original_data.get("test_code", ""),
    )
    output_data["validation"] = validation_result
    
    return output_data


def run_incorrect_tests(
    raw_solution_incorrect: str,
    new_solution_incorrect: str,
    test_code: str,
) -> Dict[str, Any]:
    """Execute provided tests against the incorrect code to ensure it actually fails."""
    if not test_code or not test_code.strip():
        return {
            "skipped": True,
            "reason": "No test_code provided in original sample.",
        }
    
    code_parts = []
    if raw_solution_incorrect and raw_solution_incorrect.strip():
        code_parts.append(extract_code_from_solution(raw_solution_incorrect))
    if new_solution_incorrect and new_solution_incorrect.strip():
        code_parts.append(extract_code_from_solution(new_solution_incorrect))
    code_parts.append(test_code)
    combined_code = "\n\n".join(code_parts)
    
    sandbox_globals: Dict[str, Any] = {}
    try:
        exec(combined_code, sandbox_globals, sandbox_globals)  # noqa: S102 - controlled research sandbox
        return {
            "skipped": False,
            "passed": True,
            "warning": "Incorrect code passed provided tests; consider regenerating with different error_type or adding stronger tests.",
        }
    except Exception as exc:  # pragma: no cover - runtime guard
        return {
            "skipped": False,
            "passed": False,
            "failure_type": exc.__class__.__name__,
            "failure_message": str(exc),
        }


def collect_input_files(input_dir: Path) -> List[Path]:
    """Collect all JSON files from input directory, sorted by ID."""
    files = sorted(
        (path for path in input_dir.glob("*.json") if path.is_file()),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    return files


def assign_error_types(num_files: int, rng: random.Random) -> List[str]:
    """Assign error types to files, ensuring diversity."""
    # Repeat error types to cover all files
    error_assignments = (ERROR_TYPES * ((num_files // len(ERROR_TYPES)) + 1))[:num_files]
    # Shuffle for randomness (seeded rng for reproducibility)
    rng.shuffle(error_assignments)
    return error_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate incorrect code variants for mbpp_pro.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible error-type assignment.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation if an output file already exists for a problem ID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY must be set in the environment.\n"
            "Get your API key from: https://console.anthropic.com/"
        )
    
    client = Anthropic(api_key=api_key)
    
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    input_files = collect_input_files(INPUT_DIR)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found in input directory: {INPUT_DIR}")
    
    rng = random.Random(args.seed)
    random.seed(args.seed)
    error_types = assign_error_types(len(input_files), rng)
    
    print("="*60)
    print("Generate Incorrect Code Dataset")
    print("="*60)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print(f"Total files to process: {len(input_files)}")
    print(f"Error types to use: {len(set(error_types))} different types")
    print(f"Random seed: {args.seed}")
    if args.skip_existing:
        print("Skip existing outputs: enabled")
    print("="*60)
    print()
    
    processed = 0
    errors = 0
    
    for idx, input_file in enumerate(input_files, start=1):
        problem_id = int(input_file.stem) if input_file.stem.isdigit() else idx
        output_file = OUTPUT_DIR / f"{problem_id}.json"
        error_type = error_types[idx - 1]
        
        print(f"[{idx}/{len(input_files)}] Processing {input_file.name} (Error type: {error_type})...")
        
        if args.skip_existing and output_file.exists():
            print("  ⏩ Skipped (existing output).")
            continue
        
        try:
            result = process_single_file(
                client=client,
                input_path=input_file,
                output_path=output_file,
                error_type=error_type,
            )
            
            save_json_file(output_file, result)
            print(f"  ✅ Saved to {output_file.name}")
            processed += 1
            
        except Exception as exc:
            print(f"  ❌ Error: {exc}")
            errors += 1
            
            # Save error record
            error_file = OUTPUT_DIR / f"{problem_id}_ERROR.json"
            error_data = {
                "id": problem_id,
                "original_file": str(input_file),
                "error": str(exc),
                "error_type": error_type,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            save_json_file(error_file, error_data)
        
        print()
        
        # Rate limiting
        if idx < len(input_files) and SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
    
    print("="*60)
    print(f"✅ Generation Complete!")
    print(f"   Processed: {processed}")
    print(f"   Errors: {errors}")
    print(f"   Results saved to: {OUTPUT_DIR.resolve()}")
    print("="*60)


if __name__ == "__main__":
    main()
