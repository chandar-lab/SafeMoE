#!/bin/bash
#SBATCH --job-name=MoE
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=8 
# SBATCH --gres=gpu:a100l:1
#SBATCH --gres=gpu:l40s:1                          
# SBATCH --gres=gpu:2
#SBATCH --mem=68G 
#SBATCH --time=03:00:00
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
moe_name=$2
data_dir=$3
out_dir=$4

python model_eval_vllm.py --moe_path $moe_path --moe_name $moe_name --test_data_path $data_dir --output_json_path $out_dir --dataset_dir ""

# # python exmp_1.py

# while wait; test $? -gt 128; do :; done

# echo "Job complete"


# # Set default values
# MOE_PATH="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250"
# TEST_DATA_PATH="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_animal_abuse.json"
# OUTPUT_PATH="evals_output/vllm_model_tests.json"

# # Parse command line arguments
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --moe_path)
#             MOE_PATH="$2"
#             shift 2
#             ;;
#         --test_data_path)
#             TEST_DATA_PATH="$2"
#             shift 2
#             ;;
#         --output_json_path)
#             OUTPUT_PATH="$2"
#             shift 2
#             ;;
#         --help)
#             echo "Usage: $0 [--moe_path PATH] [--test_data_path PATH] [--output_json_path PATH]"
#             echo ""
#             echo "Options:"
#             echo "  --moe_path          Path to the model (default: $MOE_PATH)"
#             echo "  --test_data_path    Path to test data JSON file (default: $TEST_DATA_PATH)"
#             echo "  --output_json_path  Path to save output JSON (default: $OUTPUT_PATH)"
#             echo "  --help              Show this help message"
#             exit 0
#             ;;
#         *)
#             echo "Unknown option: $1"
#             echo "Use --help for usage information"
#             exit 1
#             ;;
#     esac
# done

# echo "Starting vLLM-based model evaluation..."
# echo "Model path: $MOE_PATH"
# echo "Test data: $TEST_DATA_PATH"
# echo "Output path: $OUTPUT_PATH"
# echo ""

# # Run the evaluation with optimized settings for performance
# python model_eval_vllm.py \
#     --moe_path "$MOE_PATH" \
#     --test_data_path "$TEST_DATA_PATH" \
#     --output_json_path "$OUTPUT_PATH"

# echo ""
# echo "Evaluation completed! Results saved to: $OUTPUT_PATH"
