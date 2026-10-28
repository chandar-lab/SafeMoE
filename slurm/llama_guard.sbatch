#!/bin/bash
# Llama Guard 3 safety labels on one GPU (Table 5). HF_TOKEN (gated model) comes from your environment.
# Usage: sbatch slurm/llama_guard.sbatch <input dir> <output dir> [extra llama_guard.py args]
#   e.g. sbatch slurm/llama_guard.sbatch results/generations/safemoe_xl_seed0 results/judgements/llama_guard/safemoe_xl_seed0
#SBATCH --job-name=safemoe-llama-guard
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:55:00
source slurm/common.sh
inputs=$1; out=$2; shift 2
python scripts/eval/llama_guard.py --inputs "$inputs"/*.json --output_dir "$out" "$@"
