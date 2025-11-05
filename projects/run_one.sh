#!/bin/bash
#SBATCH --job-name=moe_support
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100l:4
# SBATCH --gres=gpu:l40s:4                          
# SBATCH --gres=gpu:2
#SBATCH --mem=128G 
#SBATCH --time=03:00:00
# SBATCH --partition=main
# SBATCH --constraint='ampere'
#SBATCH --partition=short-unkillable
# SBATCH --signal=USR1@90  # Preemption warning 90s before timeout
# SBATCH --signal=B:TERM@90



# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export MMLU_DATA_DIR='/home/mila/m/maryam.hashemzadeh/scratch/cache/huggingface/datasets/hendrycks_test/'

module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mergoo_3/bin/activate
source /home/mila/m/maryam.hashemzadeh/projects/envs/mergoo_update/bin/activate

TASK_NAME=$1
MODEL=$2

# CHECKPOINT_PATH="out_ckpt/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_ckpt/"
# CHECKPOINT_PATH="out_ckpt/${SLURM_JOB_ID}_${TASK_NAME}_ckpt/"
# CHECKPOINT_PATH="out_ckpt/${SLURM_JOB_NAME}_ckpt.pth"
# CHECKPOINT_PATH="out_ckpt/${MODEL}_${TASK_NAME}_ckpt/"
# LAST_CHECKPOINT="${CHECKPOINT_PATH}checkpoint-10/"
# echo "............ LAST_CHECKPOINT ....... ${LAST_CHECKPOINT}"

# # Trap signal and requeue the job
# function _resubmit_on_timeout() {
#     echo "Caught signal, saving checkpoint and requeueing... ${SLURM_JOB_ID}"
#     trap "" SIGTERM
#     scancel -s SIGTERM --full $SLURM_JOBID
#     scontrol requeue ${SLURM_JOB_ID}   
#     # exit 1
# }
# trap _resubmit_on_timeout SIGTERM






# Uncomment below to use DeepSpeed Stage 2
# echo "Launching training with DeepSpeed Stage 2..."

# accelerate launch   --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2.json"   exmp_1.py

accelerate launch   --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2.json"   exmp_1_qwen.py
accelerate launch   --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2_batch.json"   exmp_1_qwen.py


accelerate launch   --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2.json" 
# accelerate launch   --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2.json"   sft_training.py

# python exmp_1.py

# accelerate launch   --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2.json"   sft_training_update_datacollator.py


# accelerate launch   --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2.json"   train_alpha_moe_fast_v2.py

# accelerate launch exmp_1.py 

# Uncomment below to use DeepSpeed Stage 3 (for very large models)
# echo "Launching training with DeepSpeed Stage 3..."
# accelerate launch --num_processes=$NUM_GPUS exmp_1.py deepspeed.enabled=true deepspeed.config_path="conf/deepspeed_stage3_config.json"

# Example with custom parameters
# accelerate launch exmp_1.py \
#   training.learning_rate=2e-5 \
#   training.num_train_epochs=2 \
#   deepspeed.enabled=true \
#   wandb.run_name="my-experiment"

# accelerate launch main_pipline.py train_config.model=$MODEL train_config.output_dir=$CHECKPOINT_PATH \
#     tasks="[$TASK_NAME]" \
#     path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME" \
#     checkpoint_path=$CHECKPOINT_PATH


while wait; test $? -gt 128; do :; done

echo "Job complete"
