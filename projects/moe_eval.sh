#!/bin/bash
#SBATCH --job-name=MoE
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=8 
# SBATCH --gres=gpu:a100l:1
#SBATCH --gres=gpu:l40s:1                          
# SBATCH --gres=gpu:2
#SBATCH --mem=48G 
#SBATCH --time=14:00:00
# SBATCH --partition=main
# SBATCH --constraint='ampere'
# SBATCH --partition=short-unkillable
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

moe_path=$1
data_dir=$2
out_dir=$3

python model_eval.py --moe_path $moe_path --test_data_path $data_dir --output_json_path $out_dir --moe False

# python exmp_1.py

while wait; test $? -gt 128; do :; done

echo "Job complete"
