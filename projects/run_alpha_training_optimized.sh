#!/bin/bash

# Optimized Alpha MoE Training Script for 4x L40s GPUs
# This script uses Accelerate + DeepSpeed for maximum performance

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# Model and data paths
MOE_MODEL_PATH="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250"
SAFE_DATA_PATH="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/gpt4o_moe_model_all_XL_22apater_top2_advbench_data.json"
OUTPUT_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/alpha_moe_trained_optimized"

# Training hyperparameters optimized for 4x L40s
LEARNING_RATE=1e-3
NUM_EPOCHS=10
BATCH_SIZE=8  # Per device batch size
GRADIENT_ACCUMULATION_STEPS=2  # Effective batch size = 8 * 4 * 2 = 64
MAX_SEQ_LENGTH=512
ALPACA_PERCENTAGE=0.2
WARMUP_STEPS=100
WEIGHT_DECAY=0.01
DATALOADER_NUM_WORKERS=4
SAVE_STEPS=250
EVAL_STEPS=50
LOGGING_STEPS=10

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting optimized alpha MoE training..."
echo "Using 4 GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model path: $MOE_MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Effective batch size: $((BATCH_SIZE * 4 * GRADIENT_ACCUMULATION_STEPS))"

# Launch training with Accelerate
accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=29500 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    train_alpha_moe_fast.py \
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



echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Alpha coefficients saved to: $OUTPUT_DIR/alpha_coefficients.json"
