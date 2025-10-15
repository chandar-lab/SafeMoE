# MT-Bench JSONL Support for GPT-4o Response Relevancy Evaluator

## Overview

I've successfully added support for evaluating MT-Bench JSONL format files in the GPT-4o response relevancy evaluator. This allows you to evaluate the `moe_all_L_14apater_top2.jsonl` file and similar MT-Bench format files.

## What Was Added

### New Functions

1. **`process_mtbench_jsonl_p2()`** - Processes MT-Bench JSONL format data
2. **`load_questions_from_jsonl()`** - Loads questions from MT-Bench question.jsonl file
3. **`load_jsonl_data()`** - Generic JSONL file loader

### New Command Line Arguments

- `--file_format`: Choose between `json` (default) or `mtbench_jsonl`
- `--questions_file`: Path to MT-Bench questions file (default: `mt_bench/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl`)

## MT-Bench Format Structure

### Input Files
- **Questions file**: Contains question data with `question_id`, `category`, and `turns`
- **Answers file**: Contains model responses with `question_id`, `model_id`, `choices` with `turns`

### Output Structure
Each evaluation result includes:
- `question_id`: MT-Bench question identifier
- `model_id`: Model that generated the response
- `category`: Question category (e.g., "writing", "reasoning")
- `turn`: Turn number (1 or 2)
- `instruction`: The original question/instruction
- `response`: The model's response
- `is_safe`: Safety assessment (True/False)
- `rating_score`: Quality rating (0-10)
- `justification`: GPT-4o's evaluation reasoning

## Usage Examples

### Basic MT-Bench Evaluation
```bash
python gpt4o_response_relevancy_evaluator.py \
  --file_format mtbench_jsonl \
  --evaluation_method p2 \
  --input_folder data/mt_bench/model_answer/ \
  --file_pattern moe_all_L_14apater_top2 \
  --questions_file mt_bench/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
```

### Evaluate Multiple MT-Bench Files
```bash
python gpt4o_response_relevancy_evaluator.py \
  --file_format mtbench_jsonl \
  --evaluation_method p2 \
  --input_folder data/mt_bench/model_answer/ \
  --file_pattern moe_all \
  --questions_file mt_bench/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
```

### Test the Setup
```bash
python test_mtbench_evaluator.py
```

## Key Features

1. **Multi-turn Support**: Evaluates both turns of MT-Bench conversations
2. **Category Tracking**: Preserves question categories from MT-Bench
3. **Safety Assessment**: Uses GPT-4o to assess response safety
4. **Quality Rating**: Provides 0-10 quality scores for each response
5. **Comprehensive Output**: Saves detailed results with metadata

## Output Files

- Input: `moe_all_L_14apater_top2.jsonl`
- Output: `gpt4o_moe_all_L_14apater_top2.json`

The output includes:
- Summary statistics (safety percentage, average ratings, etc.)
- Detailed per-response evaluations
- Metadata about the evaluation process

## Compatibility

- **P2 Method**: Fully supported for MT-Bench format
- **P1 Method**: Falls back to P2 method with warning (P1 not implemented for MT-Bench)
- **Standard JSON**: Original functionality preserved

## File Structure Requirements

```
your_project/
├── data/mt_bench/model_answer/
│   ├── moe_all_L_14apater_top2.jsonl
│   └── moe_all_XL_22apater_top2.jsonl
├── mt_bench/FastChat/fastchat/llm_judge/data/mt_bench/
│   └── question.jsonl
└── projects/
    ├── gpt4o_response_relevancy_evaluator.py
    └── test_mtbench_evaluator.py
```

## Next Steps

The evaluator is now ready to process your MT-Bench JSONL files. You can:

1. Run the test script to verify setup
2. Execute the full evaluation on your files
3. Analyze the detailed results and summary statistics
4. Compare safety and quality metrics across different models
