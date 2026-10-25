#!/bin/bash
# Train a SafeMoE router on one GPU (batch size 4).
# Usage: sbatch slurm/train_router.sbatch configs/router/safemoe_xl.yaml data.path=<PATH_TO_SAFE_RESPONSES_JSON> [key=value ...]
#SBATCH --job-name=safemoe-router
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=11:55:00
source slurm/common.sh
python scripts/train_router.py --config "$@"
