# Code Obfuscation Evaluation Pipeline

Evaluates how well language models can understand code semantics when variable and function names are intentionally obfuscated with random-looking identifiers.

## Overview

Three-stage pipeline:

1. **Code Obfuscation Generation** (`generate_code_obfuscation.py`): Uses Claude to obfuscate Python code by replacing identifiers with random-looking names while preserving logic.

2. **Qwen Analysis** (`qwen_mppb_pro_evaulation.py`): Queries Qwen to infer problem descriptions from obfuscated code.

3. **LLM Judge Evaluation** (`llm_judge_evaluation.py`): Uses ChatGPT to compare Qwen's inferred descriptions with ground truth.

## Approach

**Obfuscation**: All identifiers (variables, functions, classes, parameters) are replaced with random-looking names (e.g., `char` → `a7_zX`). Code structure, logic, literals, comments, and formatting are preserved.

**Task**: Given obfuscated code, infer the problem description that the code solves.

**Evaluation Dimensions** (1-5 scale):
- Semantic Accuracy: Does Qwen understand the core purpose and algorithm?
- Completeness: Does Qwen capture all key aspects?
- Transformation Understanding: Does Qwen understand how problems evolved?
- Robustness to Obfuscation: How well does Qwen handle confusing names?
- Overall Score: Aggregate performance measure

## Stage 1: Code Obfuscation Generation

**Script:** `generate_code_obfuscation.py`

Uses Claude to obfuscate all identifiers in Python code with random-looking names while preserving logic and structure.

**Usage:**
```bash
python generate_code_obfuscation.py \
    --input-dir ../mbpp_pro \
    --output-dir mbpp_pro_code_obfuscation
```

**Requirements:** `ANTHROPIC_API_KEY` environment variable

## Stage 2: Qwen Analysis

**Script:** `qwen_mppb_pro_evaulation.py`

Queries Qwen to infer problem descriptions from obfuscated code.

**Usage:**
```bash
python qwen_mppb_pro_evaulation.py
```

**Requirements:** `HF_ENDPOINT_URL` and `HF_TOKEN` environment variables

## Stage 3: LLM Judge Evaluation

**Script:** `llm_judge_evaluation.py`

Uses ChatGPT to evaluate Qwen's inferred descriptions against ground truth.

**Usage:**
```bash
python llm_judge_evaluation.py \
    --ground-truth-dir ../mbpp_pro \
    --qwen-response-dir qwen_code_obfuscation_description_response \
    --obfuscated-code-dir mbpp_pro_code_obfuscation \
    --output-dir llm_judge_evaluations \
    --skip-existing
```

**Requirements:** `OPENAI_API_KEY` environment variable

**Output:** Detailed evaluations with scores and `summary.json` statistics

## Complete Workflow

```bash
python generate_code_obfuscation.py --input-dir ../mbpp_pro --output-dir mbpp_pro_code_obfuscation
python qwen_mppb_pro_evaulation.py
python llm_judge_evaluation.py --ground-truth-dir ../mbpp_pro --qwen-response-dir qwen_code_obfuscation_description_response --obfuscated-code-dir mbpp_pro_code_obfuscation --output-dir llm_judge_evaluations
```

