#!/bin/bash

# Script to run alpha coefficient training for MoE models
# Usage: ./run_alpha_training.sh [moe_model_path]

set -e  # Exit on any error

# Default values
MOE_MODEL_PATH=${1:-"projects/data/mistral_lora_moe_all_XL_22apater_top2"}
SAFE_DATA_PATH="projects/evals_output/gpt4o_moe_model_all_XL_22apater_top2_advbench_data.json"
OUTPUT_DIR="projects/data/alpha_moe_trained_$(date +%Y%m%d_%H%M%S)"
LEARNING_RATE=1e-3
NUM_EPOCHS=5
BATCH_SIZE=1  # Reduced for memory
ALPACA_PERCENTAGE=0.2
GRADIENT_ACCUMULATION_STEPS=16  # Compensate for smaller batch size
MAX_SEQ_LENGTH=256  # Reduced for memory
USE_DEEPSPEED=true
USE_QUANTIZATION=false  # Set to true if you want 4-bit quantization

echo "=== Alpha MoE Training Script ==="
echo "MoE Model Path: $MOE_MODEL_PATH"
echo "Safe Data Path: $SAFE_DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "Alpaca Percentage: $ALPACA_PERCENTAGE"
echo "Use DeepSpeed: $USE_DEEPSPEED"
echo "Use Quantization: $USE_QUANTIZATION"
echo "================================="

# Check if MoE model exists
if [ ! -d "$MOE_MODEL_PATH" ]; then
    echo "Error: MoE model path does not exist: $MOE_MODEL_PATH"
    echo "Available models in projects/data/:"
    ls -la projects/data/ | grep mistral_lora_moe || echo "No MoE models found"
    exit 1
fi

# Check if safe data exists
if [ ! -f "$SAFE_DATA_PATH" ]; then
    echo "Error: Safe data file does not exist: $SAFE_DATA_PATH"
    echo "Available evaluation files:"
    ls -la projects/evals_output/ | grep gpt4o_moe || echo "No evaluation files found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Build the training command
TRAIN_CMD="python projects/train_alpha_moe.py \
    --moe_model_path \"$MOE_MODEL_PATH\" \
    --safe_data_path \"$SAFE_DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --alpaca_percentage $ALPACA_PERCENTAGE"

# Add DeepSpeed if enabled
if [ "$USE_DEEPSPEED" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use_deepspeed"
fi

# Add quantization if enabled
if [ "$USE_QUANTIZATION" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use_quantization"
fi

# Run the training
echo "Starting alpha coefficient training..."
echo "Command: $TRAIN_CMD"
eval $TRAIN_CMD

echo "Training completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "Alpha coefficients saved to: $OUTPUT_DIR/alpha_coefficients.json"

# Display final alpha values
if [ -f "$OUTPUT_DIR/alpha_coefficients.json" ]; then
    echo ""
    echo "=== Final Alpha Coefficients ==="
    cat "$OUTPUT_DIR/alpha_coefficients.json"
    echo "================================="
fi
