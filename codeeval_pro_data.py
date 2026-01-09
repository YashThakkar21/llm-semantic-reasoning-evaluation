# This script is used to download the MBPP-Pro dataset from the Hugging Face dataset repository.

from datasets import load_dataset
import os
import json

SAVE_DIR = "mbpp-pro"
os.makedirs(SAVE_DIR, exist_ok=True)

stream = load_dataset("CodeEval-Pro/mbpp-pro", split="train", streaming=True)

for row in stream:
    row_id = row["id"]
    path = os.path.join(SAVE_DIR, f"{row_id}.json")

    with open(path, "w") as f:
        json.dump(row, f, indent=2)

print("✅ Saved each row as a separate JSON file!")
