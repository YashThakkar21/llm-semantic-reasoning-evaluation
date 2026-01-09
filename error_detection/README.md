# Error Detection Evaluation Pipeline

Evaluates how well language models can detect, locate, classify, and explain errors in incorrect code.

## Overview

Three-stage pipeline:

1. **Generate Incorrect Code** (`1_generate_incorrect_code.py`): Uses Claude to generate incorrect code variants with bugs
2. **Qwen Analysis** (`2_qwen_analyze_incorrect_code.py`): Analyzes incorrect code to detect errors
3. **LLM Judge Evaluation** (`3_llm_judge_error_detection.py`): Evaluates Qwen's error detection performance

## Approach

**Error Types** (7 categories):
1. `logical_error` - Wrong conditional logic, incorrect calculations
2. `off_by_one` - Indexing mistakes, loop boundary errors
3. `edge_case_failure` - Missing boundary checks, empty input handling
4. `algorithm_error` - Wrong algorithmic approach
5. `boundary_condition` - Boundary-related errors
6. `type_error` - Incorrect data type handling
7. `missing_validation` - Removed or incorrect input validation

**Task**: Given incorrect code with known bugs, detect and analyze all errors.

**Evaluation Dimensions** (1-5 scale):
- Error Detection Accuracy
- Error Location Precision
- Error Type Classification
- Error Explanation Quality
- Completeness
- Fix Suggestion Quality
- Overall Performance

## Stage 1: Generate Incorrect Code

**Script:** `1_generate_incorrect_code.py`

Generates realistic incorrect code variants with bugs. Each problem is randomly assigned one of 7 error types. Code is validated to ensure bugs are real (code fails tests).

**Usage:**
```bash
python 1_generate_incorrect_code.py --seed 42 --skip-existing
```

**Requirements:** `ANTHROPIC_API_KEY` environment variable

## Stage 2: Qwen Analysis

**Script:** `2_qwen_analyze_incorrect_code.py`

Queries Qwen to analyze incorrect code, detect errors, classify types, and suggest fixes.

**Usage:**
```bash
python 2_qwen_analyze_incorrect_code.py --skip-existing
```

**Requirements:** `HF_ENDPOINT_URL` and `HF_TOKEN` environment variables

## Stage 3: LLM Judge Evaluation

**Script:** `3_llm_judge_error_detection.py`

Uses ChatGPT to evaluate Qwen's error detection performance against ground truth.

**Usage:**
```bash
python 3_llm_judge_error_detection.py --skip-existing
```

**Requirements:** `OPENAI_API_KEY` environment variable

**Output:** Detailed evaluations with scores and `summary.json` statistics

## Complete Workflow

```bash
python 1_generate_incorrect_code.py --seed 42 --skip-existing
python 2_qwen_analyze_incorrect_code.py --skip-existing
python 3_llm_judge_error_detection.py --skip-existing
```

## Results

- 374 evaluations completed
- Overall performance: 2.83/5.0
- Best dimension: Error Location Precision (3.46/5.0)
- Weakest dimension: Completeness (2.22/5.0)

## Helper Script

**`generate_summary_from_evaluations.py`**: Regenerates `summary.json` from existing evaluation files.

**Usage:**
```bash
python generate_summary_from_evaluations.py
```
