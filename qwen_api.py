# Old file with java code evaluation.

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_TOKENS = 800
TEMPERATURE = 0.2
SLEEP_SECONDS = 0.5

SAMPLED_DIR = Path("sampled_java_files")
SAMPLED_OUTPUT_DIR = Path("qwen_sampled_evaluations")
# OUTPUT_FILE = Path("comparison_results.json")


def build_messages(file_name: str, code: str) -> list:
    system_prompt = (
        "You are an expert Java reviewer. "
        "Judge the submission for algorithmic correctness and readability. "
        "Respond in JSON with fields "
        '{"file": string, "correctness": int (1-5), "readability": int (1-5), '
        '"summary": string, "issues": [string]}. '
        "If the code is obviously malformed, explain that in issues."
    )
    user_prompt = f"File: {file_name}\n\n```java\n{code}\n```"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def query_model(url: str, headers: dict, messages: list) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": str(exc)}

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError):
        return {"error": "Unexpected response", "raw": response.text}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw": content}
    return parsed


def analyze_directory(
    label: str,
    directory: Path,
    url: str,
    headers: dict,
    save_dir: Path | None = None,
) -> dict:
    if not directory.exists():
        return {"error": f"Directory '{directory}' not found."}

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for java_file in sorted(directory.glob("*.java"), key=lambda p: p.name.lower()):
        messages = build_messages(java_file.name, java_file.read_text())
        analysis = query_model(url, headers, messages)
        results[java_file.name] = analysis
        print(f"[{label}] {java_file.name}")
        print(json.dumps(analysis, indent=2))
        print()

        if save_dir is not None:
            output_path = save_dir / (java_file.stem + ".json")
            payload = {
                "file": java_file.name,
                "source_path": str(java_file),
                "analysis": analysis,
            }
            output_path.write_text(json.dumps(payload, indent=2))

        time.sleep(SLEEP_SECONDS)
    return results


def main() -> None:
    load_dotenv()
    url = os.getenv("HF_ENDPOINT_URL")
    token = os.getenv("HF_TOKEN")

    if not url or not token:
        raise RuntimeError("HF_ENDPOINT_URL and HF_TOKEN must be set in the environment.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print("Probing Qwen model on sampled Java files...\n")
    sampled_results = analyze_directory(
        "sampled_java_files",
        SAMPLED_DIR,
        url,
        headers,
        save_dir=SAMPLED_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
