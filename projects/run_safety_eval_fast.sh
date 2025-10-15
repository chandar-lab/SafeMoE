#!/bin/bash

# Fast safety evaluation script using vLLM
# Optimized for maximum inference speed

# Default configuration
DEFAULT_JUDGE_MODEL="/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-70B-Instruct"
DEFAULT_TEST_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/"
DEFAULT_PATTERN="mistral_sft_model_all_safe_10_500sample_overlap_30epoch_*.json"
DEFAULT_OUTPUT_PREFIX="fast_safety_eval"

# Hardware detection for optimal settings
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "=== Fast Safety Evaluation Setup ==="
echo "Detected $GPU_COUNT GPU(s) with ${GPU_MEMORY}MB memory each"

# Optimize settings based on hardware
if [ "$GPU_COUNT" -ge 8 ]; then
    # 8+ GPU setup (maximum performance)
    TENSOR_PARALLEL=8
    BATCH_SIZE=128
    GPU_MEMORY_UTIL=0.95
    echo "Using 8+ GPU configuration for maximum speed"
elif [ "$GPU_COUNT" -ge 4 ]; then
    # 4+ GPU setup (high performance)
    TENSOR_PARALLEL=4
    BATCH_SIZE=96
    GPU_MEMORY_UTIL=0.95
    echo "Using 4+ GPU configuration for high speed"
elif [ "$GPU_COUNT" -ge 2 ]; then
    # 2+ GPU setup
    TENSOR_PARALLEL=2
    BATCH_SIZE=64
    GPU_MEMORY_UTIL=0.90
    echo "Using 2+ GPU configuration"
else
    # Single GPU setup
    TENSOR_PARALLEL=1
    if [ "$GPU_MEMORY" -gt 40000 ]; then
        BATCH_SIZE=64
        GPU_MEMORY_UTIL=0.90
        echo "Using single high-memory GPU configuration"
    else
        BATCH_SIZE=32
        GPU_MEMORY_UTIL=0.85
        echo "Using single standard GPU configuration"
    fi
fi

# Parse command line arguments
JUDGE_MODEL="$DEFAULT_JUDGE_MODEL"
TEST_DIR="$DEFAULT_TEST_DIR"
PATTERN="$DEFAULT_PATTERN"
OUTPUT_PREFIX="$DEFAULT_OUTPUT_PREFIX"
QUANTIZATION="awq"
SINGLE_FILE=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --judge_model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --test_dir)
            TEST_DIR="$2"
            shift 2
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --output_prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --single_file)
            SINGLE_FILE="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --no_quantization)
            QUANTIZATION=""
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --tensor_parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Fast safety evaluation using vLLM for maximum inference speed"
            echo ""
            echo "Options:"
            echo "  --judge_model PATH      Path to judge model (default: Llama 3.1 70B)"
            echo "  --test_dir PATH         Directory containing test files"
            echo "  --pattern PATTERN       File pattern to match"
            echo "  --single_file PATH      Process single file instead of directory"
            echo "  --output_prefix PREFIX  Prefix for output files"
            echo "  --quantization METHOD   Quantization (awq, gptq, fp8, or --no_quantization)"
            echo "  --batch_size N          Override batch size"
            echo "  --tensor_parallel N     Override tensor parallel size"
            echo "  --verbose               Enable verbose output"
            echo "  --help                  Show this help"
            echo ""
            echo "Examples:"
            echo "  # Process all files with auto-optimization"
            echo "  $0"
            echo ""
            echo "  # Process single file"
            echo "  $0 --single_file /path/to/test.json"
            echo ""
            echo "  # Maximum speed with FP8 quantization"
            echo "  $0 --quantization fp8 --batch_size 128"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "=== Configuration ==="
echo "Judge model: $JUDGE_MODEL"
if [ -n "$SINGLE_FILE" ]; then
    echo "Processing single file: $SINGLE_FILE"
else
    echo "Test directory: $TEST_DIR"
    echo "File pattern: $PATTERN"
fi
echo "Output prefix: $OUTPUT_PREFIX"
echo "Tensor parallel: $TENSOR_PARALLEL"
echo "Batch size: $BATCH_SIZE"
echo "GPU memory utilization: $GPU_MEMORY_UTIL"
if [ -n "$QUANTIZATION" ]; then
    echo "Quantization: $QUANTIZATION"
else
    echo "Quantization: disabled"
fi
echo ""

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_CUDA_GRAPH=1

# Build command
CMD="python safety_eval_vllm.py"
CMD="$CMD --judge_model '$JUDGE_MODEL'"
CMD="$CMD --tensor_parallel_size $TENSOR_PARALLEL"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --gpu_memory_utilization $GPU_MEMORY_UTIL"
CMD="$CMD --temperature 0.1"  # Low temperature for consistent evaluation
CMD="$CMD --max_tokens 200"
CMD="$CMD --output_prefix '$OUTPUT_PREFIX'"

if [ -n "$SINGLE_FILE" ]; then
    CMD="$CMD --test_data_path '$SINGLE_FILE'"
else
    CMD="$CMD --test_data_dir '$TEST_DIR'"
    CMD="$CMD --file_pattern '$PATTERN'"
    CMD="$CMD --process_all"
fi

if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

if [ -n "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

echo "=== Starting Fast Safety Evaluation ==="
echo "Command: $CMD"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the evaluation
eval $CMD

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=== Evaluation Complete ==="
echo "Total execution time: ${TOTAL_TIME} seconds"
echo "Results saved with prefix: $OUTPUT_PREFIX"

# Show GPU utilization summary
echo ""
echo "=== GPU Utilization Summary ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
