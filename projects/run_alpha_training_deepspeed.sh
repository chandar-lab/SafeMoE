#!/bin/bash
#SBATCH --job-name=alpha_moe_train
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100l:4
# SBATCH --gres=gpu:l40s:4
# SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# SBATCH --partition=main
# SBATCH --constraint='ampere'
#SBATCH --partition=short-unkillable
# SBATCH --signal=USR1@90  # Preemption warning 90s before timeout
# SBATCH --signal=B:TERM@90

# Environment setup
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mergoo_3/bin/activate

# Parameters - optimized for 4x L40s GPUs
MOE_MODEL_PATH=${1:-'/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250'}
SAFE_DATA_PATH=${2:-"evals_output/gpt4o_moe_model_all_XL_22apater_top2_advbench_data.json"}
OUTPUT_DIR="projects/data/alpha_moe_trained_deepspeed_${SLURM_JOB_ID}"
LEARNING_RATE=1e-3
NUM_EPOCHS=10
BATCH_SIZE=8  # Increased batch size for better GPU utilization
ALPACA_PERCENTAGE=0.2
GRADIENT_ACCUMULATION_STEPS=2  # Reduced since we have larger batch size
MAX_SEQ_LENGTH=512  # Increased sequence length
WARMUP_STEPS=100
WEIGHT_DECAY=0.01
DATALOADER_NUM_WORKERS=4
SAVE_STEPS=250
EVAL_STEPS=50
LOGGING_STEPS=10

echo "=== Alpha MoE Training with DeepSpeed ==="
echo "Job ID: $SLURM_JOB_ID"
echo "MoE Model Path: $MOE_MODEL_PATH"
echo "Safe Data Path: $SAFE_DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training with DeepSpeed using accelerate
echo "Launching alpha coefficient training with DeepSpeed..."

accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=29500 \
    --mixed_precision=bf16 \
    --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2.json" \
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
# Wait for completion
while wait; test $? -gt 128; do :; done

echo "Job complete"
echo "Results saved to: $OUTPUT_DIR"
echo "Alpha coefficients saved to: $OUTPUT_DIR/alpha_coefficients.json"

# Display final alpha values if available
if [ -f "$OUTPUT_DIR/alpha_coefficients.json" ]; then
    echo ""
    echo "=== Final Alpha Coefficients ==="
    cat "$OUTPUT_DIR/alpha_coefficients.json"
    echo "================================="
fi
