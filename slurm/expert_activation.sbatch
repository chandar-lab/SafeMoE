#!/bin/bash
# Expert activation per harm category under first-layer routing (Appendix N), one GPU.
# Prompts are every PKU-SafeRLHF test prompt of each harm category (downloaded by name).
# Usage: sbatch slurm/expert_activation.sbatch <first-layer SafeMoE checkpoint> <output dir> [extra expert_activation.py args]
#   e.g. sbatch slurm/expert_activation.sbatch checkpoints/routers/first_layer_routing results/analysis/expert_activation
#SBATCH --job-name=safemoe-expert-activation
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:55:00
source slurm/common.sh
model=$1; out=$2; shift 2
python scripts/analysis/expert_activation.py --model "$model" --output_dir "$out" "$@"
