#!/bin/bash
# Generate responses on one GPU.
# Usage: sbatch slurm/generate.sbatch <model> <output_dir> <seed> <generate.py args>
#   sbatch slurm/generate.sbatch checkpoints/routers/safemoe_xl results/generations/safemoe_xl_seed0 0 --data pku
#SBATCH --job-name=safemoe-generate
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:55:00
source slurm/common.sh
model=$1; out=$2; seed=$3; shift 3
python scripts/generate.py --model "$model" --output_dir "$out" --seed "$seed" "$@"
