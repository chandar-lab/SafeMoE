#!/bin/bash
#SBATCH --job-name=gpt4o
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=8 
# SBATCH --gres=gpu:a100l:1
#SBATCH --gres=gpu:l40s:1                          
# SBATCH --gres=gpu:1
#SBATCH --mem=16G 
#SBATCH --time=23:00:00
#SBATCH --partition=main
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

# file_pattern=$1

# python gpt4o_response_relevancy_evaluator.py
# python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_model_all_L_14apater_top2_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_model_4domainexperts_top2_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "Mistral-7B-v0.1_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "mistral_sft_model_all_safe_11_200sample_overlap_10epoch_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "mistral_sft_model_all_safe_10_500sample_overlap_10epoch_" --input_folder '/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/'
# python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_model_all_8apater_top2_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "" --input_folder '/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/safe_lora/SafeLoRA/output/'
# python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_XL_top2_100samples_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_XL_top2_50samples_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_XL_top2_20samples_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_XL_top2_100_samples_nodomains_experts_"
python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_8adapters_top2_20_samples_nodomains_experts"
python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_8adapters_top2_50_samples_nodomains_experts"
python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_8adapters_top2_100_samples_nodomains_experts"
python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_L_top2_20_samples_nodomains_experts"
python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_L_top2_50_samples_nodomains_experts"
python gpt4o_response_relevancy_evaluator.py --file_pattern "moe_L_top2_100_samples_nodomains_experts"


# # benchmarks --> chnage dir also
# python gpt4o_response_relevancy_evaluator.py --file_pattern "DeepSeek-R1-Distill-Qwen-7B_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "gemma-7b-it_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "mistral-7b-sft-beta_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "RealSafe-R1-7B_"
# python gpt4o_response_relevancy_evaluator.py --file_pattern "zephyr-7b-beta_" --input_folder '/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/evals_output/'






while wait; test $? -gt 128; do :; done

echo "Job complete"
