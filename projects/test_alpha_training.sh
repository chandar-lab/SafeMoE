#!/bin/bash

# Quick test script for alpha training (single GPU, small dataset)
set -e

export CUDA_VISIBLE_DEVICES=0

# Model and data paths
MOE_MODEL_PATH="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250"
SAFE_DATA_PATH="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/gpt4o_moe_model_all_XL_22apater_top2_advbench_data.json"
OUTPUT_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/alpha_moe_test"

# Test parameters (small for quick testing)
LEARNING_RATE=1e-3
NUM_EPOCHS=1
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
MAX_SEQ_LENGTH=256
ALPACA_PERCENTAGE=0.1
WARMUP_STEPS=10
WEIGHT_DECAY=0.01
DATALOADER_NUM_WORKERS=1
SAVE_STEPS=50
EVAL_STEPS=25
LOGGING_STEPS=5

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting alpha MoE training test..."
echo "Using single GPU for testing"

# Launch training without Accelerate for testing
python train_alpha_moe_fast.py \
    --moe_model_path "$MOE_MODEL_PATH" \
    --safe_data_path "$SAFE_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --alpaca_percentage $ALPACA_PERCENTAGE \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --seed 42

echo "Test completed!"
echo "Results saved to: $OUTPUT_DIR"
