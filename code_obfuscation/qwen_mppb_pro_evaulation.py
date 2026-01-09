# This script evaluates the Qwen model on mbpp-pro dataset with code obfuscation and saves the results in qwen_code_obfuscation_description_response directory.

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from qwen_api import MODEL, SLEEP_SECONDS, query_model

DATA_DIR = Path("mbpp_pro_code_obfuscation")
OUTPUT_DIR = Path("qwen_code_obfuscation_description_response")


def build_messages(raw_solution: str, new_solution: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are an expert Python analyst. "
        "Given two Python code snippets labelled 'raw_solution' and 'new_solution', "
        "Give a description of the problem each snippet solves. "
        'Respond strictly in JSON with keys "raw_problem" and "new_problem". '
        "Each value should be one or a few concise sentences describing the purpose of the corresponding code. "
        "Do not include any additional commentary or formatting outside the JSON."
    )
    user_prompt = (
        "Analyze the following Python code snippets and describe the problem each solves.\n\n"
        "raw_solution:\n"
        "```python\n"
        f"{raw_solution.strip()}\n"
        "```\n\n"
        "new_solution:\n"
        "```python\n"
        f"{new_solution.strip()}\n"
        "```"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def ensure_environment() -> tuple[str, str]:
    load_dotenv()
    url = os.getenv("HF_ENDPOINT_URL")
    token = os.getenv("HF_TOKEN")
    if not url or not token:
        raise RuntimeError("HF_ENDPOINT_URL and HF_TOKEN must be set in the environment.")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    return url, headers


def process_file(json_path: Path, url: str, headers: dict) -> dict:
    data = json.loads(json_path.read_text())
    raw_solution = data.get("raw_solution", "")
    new_solution = data.get("new_solution", "")

    if not raw_solution and not new_solution:
        return {"error": "Missing both raw_solution and new_solution."}

    messages = build_messages(raw_solution, new_solution)
    
    # Query Qwen with retries
    attempts_remaining = RETRY_ATTEMPTS
    response = None
    last_error = None
    while attempts_remaining > 0:
        try:
            response = query_model(url, headers, messages)
            # Check if response contains an error
            if isinstance(response, dict) and "error" in response:
                # If it's an error, treat as failure and retry
                if attempts_remaining > 1:
                    attempts_remaining -= 1
                    last_error = response.get("error", "Unknown error")
                    time.sleep(BACKOFF_SECONDS)
                    continue
            # If no error or last attempt, break
            break
        except Exception as exc:
            attempts_remaining -= 1
            last_error = str(exc)
            if attempts_remaining <= 0:
                response = {"error": f"Failed after {RETRY_ATTEMPTS} attempts: {last_error}"}
                break
            time.sleep(BACKOFF_SECONDS)
    
    # Use last error response if all retries failed
    if response is None:
        response = {"error": f"Failed after {RETRY_ATTEMPTS} attempts: {last_error or 'Unknown error'}"}
    
    return {
        "source_file": str(json_path),
        "id": data.get("id"),
        "model": MODEL,
        "response": response,
    }


def main() -> None:
    url, headers = ensure_environment()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for json_file in sorted(DATA_DIR.glob("*.json"), key=lambda p: p.name.lower()):
        result = process_file(json_file, url, headers)
        print(f"Response for {json_file.name}:")
        print(json.dumps(result["response"], indent=2))

        output_path = OUTPUT_DIR / f"{json_file.stem}.json"
        output_path.write_text(json.dumps(result, indent=2))

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()