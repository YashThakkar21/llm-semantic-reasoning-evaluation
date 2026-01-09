# This script is used to generate code obfuscation using Claude on the MBPP-Pro dataset.

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from anthropic import Anthropic, AnthropicError
from dotenv import load_dotenv


DEFAULT_INPUT_DIR = Path("mbpp_pro")
DEFAULT_OUTPUT_DIR = Path("mbpp_pro_code_obfuscation")
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

SYSTEM_PROMPT = (
    "You are a Python refactoring assistant. Rewrite the provided code so that "
    "all variable names, function names, class names, and parameters use intentionally "
    "bad, confusing naming conventions. Preserve the program structure, logic, control "
    "flow, literal values, comments, formatting, and indentation exactly. Only rename "
    "identifiers. Do not add explanations or any extra text beyond the requested JSON."
)

USER_INSTRUCTIONS = (
    "You receive a JSON object with two fields: \"raw_solution\" and \"new_solution\". "
    "Each field contains Python source code. Produce a JSON object with the same two "
    "fields, but rename every identifier to deliberately bad names that are random-looking "
    "strings (e.g., mixtures of letters, numbers, and underscores). Keep the structure, "
    "logic, indentation, literals, and comments exactly the same. Respond with ONLY valid "
    "JSON that matches this pattern and no extra text:\n"
    "{\n"
    '  "raw_solution": "BADLY renamed raw solution code",\n'
    '  "new_solution": "BADLY renamed new solution code"\n'
    "}\n"
    "Do not include backticks, markdown formatting, explanations, or any surrounding prose."
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_prompt_payload(raw_solution: str, new_solution: str) -> str:
    payload = {
        "raw_solution": raw_solution,
        "new_solution": new_solution,
    }
    return json.dumps(payload, indent=2)


def extract_response_json(response_text: str) -> Dict[str, str]:
    response_text = response_text.strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
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
                        candidate = response_text[start_idx : idx + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
        raise ValueError("Claude response did not contain valid JSON.")


def query_claude(
    client: Anthropic,
    model: str,
    raw_solution: str,
    new_solution: str,
) -> Tuple[Dict[str, str], str]:
    message = client.messages.create(
        model=model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTIONS},
                    {"type": "text", "text": build_prompt_payload(raw_solution, new_solution)},
                ],
            }
        ],
    )

    text_parts = []
    for part in message.content:
        if part.type == "text":
            text_parts.append(part.text)
    if not text_parts:
        raise ValueError("Claude response did not contain any text segments.")

    response_text = "".join(text_parts)
    return extract_response_json(response_text), response_text


def process_file(
    client: Anthropic,
    model: str,
    input_path: Path,
    output_path: Path,
    retry: int = 3,
    backoff_seconds: float = 3.0,
) -> None:
    row = load_json(input_path)

    raw_solution = row.get("raw_solution", "")
    new_solution = row.get("new_solution", "")

    if not raw_solution and not new_solution:
        raise ValueError(f"No solutions found in {input_path}")

    attempts_remaining = retry
    while attempts_remaining:
        try:
            claude_response, raw_response_text = query_claude(client, model, raw_solution, new_solution)
            break
        except (AnthropicError, ValueError, json.JSONDecodeError) as exc:
            attempts_remaining -= 1
            if attempts_remaining <= 0:
                raise RuntimeError(f"Failed to process {input_path}: {exc}") from exc
            time.sleep(backoff_seconds)
    else:
        raise RuntimeError(f"Exhausted retries for {input_path}")

    print(f"\nClaude response for {input_path.name}:\n{raw_response_text}\n")

    updated_row = dict(row)
    updated_row["raw_solution"] = claude_response.get("raw_solution", raw_solution)
    updated_row["new_solution"] = claude_response.get("new_solution", new_solution)

    dump_json(output_path, updated_row)


def collect_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        (path for path in input_dir.glob("*.json") if path.is_file()),
        key=lambda path: int(path.stem),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate intentionally bad naming conventions using Claude.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing source JSON rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store transformed JSON rows.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Claude model identifier.",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.5,
        help="Seconds to sleep between requests.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print the system and user prompts before running.",
    )
    args = parser.parse_args()

    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is required.")

    client = Anthropic(api_key=api_key)

    if args.show_prompts:
        print("System prompt:")
        print(SYSTEM_PROMPT)
        print("\nUser instructions prompt:")
        print(USER_INSTRUCTIONS)
        print(f"\nModel identifier: {args.model}")

    print(f"Using Claude model: {args.model}")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_files = collect_input_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found in input directory: {input_dir}")

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name
        print(f"\nProcessing {input_path} -> {output_path}")
        process_file(
            client=client,
            model=args.model,
            input_path=input_path,
            output_path=output_path,
        )
        if args.throttle_seconds > 0 and index < len(input_files):
            time.sleep(args.throttle_seconds)


if __name__ == "__main__":
    main()

