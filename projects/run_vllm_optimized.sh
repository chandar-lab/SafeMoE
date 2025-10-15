#!/bin/bash

# Optimized vLLM evaluation script with performance tuning
# This script demonstrates various optimization strategies for different hardware configurations

# Default paths - modify these for your setup
DEFAULT_MODEL_PATH="HuggingFaceH4/zephyr-7b-beta"
DEFAULT_DATASET_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed"
DEFAULT_OUTPUT_PREFIX="optimized_vllm_results"

# Hardware detection
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "Detected $GPU_COUNT GPU(s) with ${GPU_MEMORY}MB memory each"

# Set optimization parameters based on hardware
if [ "$GPU_COUNT" -ge 4 ]; then
    # Multi-GPU setup (4+ GPUs)
    TENSOR_PARALLEL=4
    INFERENCE_BATCH_SIZE=128
    GPU_MEMORY_UTIL=0.95
    MAX_NUM_SEQS=512
    echo "Using multi-GPU configuration (4+ GPUs)"
elif [ "$GPU_COUNT" -ge 2 ]; then
    # Dual GPU setup
    TENSOR_PARALLEL=2
    INFERENCE_BATCH_SIZE=64
    GPU_MEMORY_UTIL=0.90
    MAX_NUM_SEQS=256
    echo "Using dual-GPU configuration"
else
    # Single GPU setup
    TENSOR_PARALLEL=1
    if [ "$GPU_MEMORY" -gt 40000 ]; then
        # High-memory GPU (A100, H100)
        INFERENCE_BATCH_SIZE=64
        GPU_MEMORY_UTIL=0.90
        MAX_NUM_SEQS=256
        echo "Using single high-memory GPU configuration"
    elif [ "$GPU_MEMORY" -gt 20000 ]; then
        # Medium-memory GPU (RTX 3090, RTX 4090)
        INFERENCE_BATCH_SIZE=32
        GPU_MEMORY_UTIL=0.85
        MAX_NUM_SEQS=128
        echo "Using single medium-memory GPU configuration"
    else
        # Lower-memory GPU
        INFERENCE_BATCH_SIZE=16
        GPU_MEMORY_UTIL=0.80
        MAX_NUM_SEQS=64
        echo "Using single lower-memory GPU configuration"
    fi
fi

# Parse command line arguments
MODEL_PATH="$DEFAULT_MODEL_PATH"
DATASET_DIR="$DEFAULT_DATASET_DIR"
OUTPUT_PREFIX="$DEFAULT_OUTPUT_PREFIX"
QUANTIZATION=""
USE_OPTIMIZATIONS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset_dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --output_prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --no-optimizations)
            USE_OPTIMIZATIONS=false
            shift
            ;;
        --tensor_parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --batch_size)
            INFERENCE_BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH       Path to the model"
            echo "  --dataset_dir PATH      Directory containing test files"
            echo "  --output_prefix PREFIX  Prefix for output files"
            echo "  --quantization METHOD   Quantization method (awq, gptq, fp8)"
            echo "  --no-optimizations      Disable automatic optimizations"
            echo "  --tensor_parallel N     Override tensor parallel size"
            echo "  --batch_size N          Override inference batch size"
            echo "  --help                  Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_DIR"
echo "  Output prefix: $OUTPUT_PREFIX"
echo "  Tensor parallel: $TENSOR_PARALLEL"
echo "  Inference batch size: $INFERENCE_BATCH_SIZE"
echo "  GPU memory utilization: $GPU_MEMORY_UTIL"
echo "  Max sequences: $MAX_NUM_SEQS"
if [ -n "$QUANTIZATION" ]; then
    echo "  Quantization: $QUANTIZATION"
fi
echo ""

# Build command with optimizations
CMD="python model_eval_vllm.py"
CMD="$CMD --moe_path '$MODEL_PATH'"
CMD="$CMD --dataset_dir '$DATASET_DIR'"
CMD="$CMD --output_prefix '$OUTPUT_PREFIX'"
CMD="$CMD --tensor_parallel_size $TENSOR_PARALLEL"
CMD="$CMD --inference_batch_size $INFERENCE_BATCH_SIZE"
CMD="$CMD --batch_size 32"  # Save batch size
CMD="$CMD --gpu_memory_utilization $GPU_MEMORY_UTIL"
CMD="$CMD --max_num_seqs $MAX_NUM_SEQS"

if [ "$USE_OPTIMIZATIONS" = true ]; then
    CMD="$CMD --enable_prefix_caching"
    CMD="$CMD --disable_log_stats"
    CMD="$CMD --block_size 16"
    CMD="$CMD --preemption_mode swap"
fi

if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# Add performance monitoring
CMD="$CMD --verbose"

echo "Running command:"
echo "$CMD"
echo ""

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
export NCCL_DEBUG=WARN  # Reduce NCCL logging
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Run the evaluation
eval $CMD

echo ""
echo "Evaluation completed!"
echo "Results saved with prefix: $OUTPUT_PREFIX"
