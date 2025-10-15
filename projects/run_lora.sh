#!/bin/bash
#SBATCH --job-name=lora_finetuning
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
# export NCCL_DEBUG=INFO
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export MMLU_DATA_DIR='/home/mila/m/maryam.hashemzadeh/scratch/cache/huggingface/datasets/hendrycks_test/'

module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mergoo_3/bin/activate

path=$1
dir=$2

python lora_finetune.py --dataset_path $path --output_dir $dir

# python exmp_1.py

while wait; test $? -gt 128; do :; done

echo "Job complete"
