# vLLM-based Model Evaluation

This directory contains `model_eval_vllm.py`, a new script that performs the same functionality as `model_eval.py` but uses vLLM for significantly improved performance and efficiency.

## Key Differences from Original `model_eval.py`

### Performance Improvements
- **vLLM Backend**: Uses vLLM's optimized inference engine instead of transformers
- **Batch Processing**: Processes multiple prompts simultaneously for better GPU utilization
- **Memory Efficiency**: Better memory management and GPU utilization
- **Speed**: Typically 2-10x faster than the original script depending on batch size and model

### New Features
- **Multiple File Processing**: Can process all `test_*` files in a directory automatically
- **Resume Functionality**: Can resume from partially completed runs using `--resume`
- **Better Logging**: Comprehensive logging with timing statistics
- **Flexible Batching**: Configurable batch sizes for optimal performance
- **Error Handling**: Improved error handling and recovery
- **Progress Tracking**: Real-time progress updates and ETA estimates
- **Verbose Mode**: Detailed output for debugging and monitoring
- **Automatic Output Naming**: Generates appropriate output file names for each test file

### Configuration Options
- **Tensor Parallelism**: Multi-GPU support via `--tensor_parallel_size`
- **Memory Control**: Configurable GPU memory utilization
- **Generation Parameters**: All sampling parameters are configurable
- **Output Control**: Flexible output formatting and saving options

## Usage

### Basic Usage (Single File)
```bash
python model_eval_vllm.py \
    --moe_path /path/to/your/model \
    --test_data_path /path/to/test/data.json \
    --output_json_path output/results.json
```

### Process Multiple Files (Directory Mode)
```bash
python model_eval_vllm.py \
    --moe_path /path/to/your/model \
    --dataset_dir /path/to/dataset/directory \
    --output_prefix "my_model_results"
```

This will:
- Find all `test_*` files in the dataset directory
- Process each file automatically
- Generate output files like: `evals_output/my_model_results_unsafe_animal_abuse.json`

### Advanced Usage with Performance Optimization
```bash
python model_eval_vllm.py \
    --moe_path /path/to/your/model \
    --test_data_path /path/to/test/data.json \
    --output_json_path output/results.json \
    --max_examples 1000 \
    --batch_size 64 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 200 \
    --verbose
```

### Resume Interrupted Run
```bash
python model_eval_vllm.py \
    --moe_path /path/to/your/model \
    --test_data_path /path/to/test/data.json \
    --output_json_path output/results.json \
    --resume
```

### Using the Shell Script
```bash
# Make executable (if needed)
chmod +x run_vllm_eval.sh

# Run with default settings
./run_vllm_eval.sh

# Run with custom paths
./run_vllm_eval.sh \
    --moe_path /path/to/model \
    --test_data_path /path/to/data.json \
    --output_json_path /path/to/output.json
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--moe_path` | (default path) | Path to the model directory |
| `--test_data_path` | (default path) | Path to test data JSON file (ignored if --dataset_dir used) |
| `--dataset_dir` | `None` | Directory containing multiple test_* files to process |
| `--output_json_path` | `evals_output/vllm_model_tests.json` | Output file path (single file mode) |
| `--output_prefix` | `vllm_model` | Prefix for output files (directory mode) |
| `--max_examples` | `500` | Maximum number of examples to process per file |
| `--batch_size` | `32` | Batch size for processing |
| `--max_tokens` | `200` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.9` | Top-p (nucleus) sampling parameter |
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `--gpu_memory_utilization` | `0.9` | GPU memory utilization ratio |
| `--resume` | `False` | Resume from existing output file |
| `--verbose` | `False` | Enable verbose logging |
| `--local` | `True` | Load data from local file (vs HuggingFace dataset) |

## Performance Tips

1. **Batch Size**: Increase `--batch_size` for better GPU utilization (try 32, 64, or higher)
2. **Memory**: Adjust `--gpu_memory_utilization` based on your GPU memory (0.8-0.95)
3. **Multi-GPU**: Use `--tensor_parallel_size` for models that don't fit on single GPU
4. **Resume**: Use `--resume` for long runs to avoid losing progress

## Output Format

The output JSON format is identical to the original script:
```json
[
  {
    "instruction": "Your question here",
    "response": "Model's response here"
  },
  ...
]
```

## Requirements

- vLLM (`pip install vllm`)
- torch
- transformers
- datasets (if using HuggingFace datasets)

## Compatibility

- Works with the same models as the original script
- Supports all Mistral-based MoE models
- Compatible with LoRA adapters and fine-tuned checkpoints
- Maintains the same prompt format and stopping criteria

## Troubleshooting

1. **Out of Memory**: Reduce `--batch_size` or `--gpu_memory_utilization`
2. **Model Loading Issues**: Ensure the model path is correct and accessible
3. **Slow Performance**: Increase `--batch_size` and check GPU utilization
4. **Resume Not Working**: Check that the output file exists and is valid JSON
