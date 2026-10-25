#!/bin/bash
# Train one LoRA expert (or a LoRA / SFT baseline) on one GPU.
# Usage: sbatch slurm/train_expert.sbatch <output_dir> <finetune.py args>
#   sbatch slurm/train_expert.sbatch checkpoints/experts/mistral/<category> --data "pku:<UNSAFE_CATEGORY>"
#SBATCH --job-name=safemoe-expert
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:55:00
source slurm/common.sh
out=$1; shift
resume=""; ls -d "$out"/checkpoint-* >/dev/null 2>&1 && resume="--resume"
python scripts/finetune.py --output_dir "$out" $resume "$@"
