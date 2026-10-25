#!/bin/bash
# PAIR red-teaming attack on one GPU (target generation is single-process; the attacker
# and judge are API calls reading OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment).
# Usage: sbatch slurm/pair_attack.sbatch <model> <output_path> <pair_attack.py args>
#   sbatch slurm/pair_attack.sbatch checkpoints/routers/safemoe_xl results/pair/safemoe_xl_harmbench.json \
#       --data harmbench
#   sbatch slurm/pair_attack.sbatch checkpoints/routers/safemoe_xl results/pair/safemoe_xl_pku.json \
#       --data pku --categories Cybercrime "Psychological Harm" "Privacy Violation" "Environmental Harm"
# Resubmit the same command to resume.
#SBATCH --job-name=safemoe-pair
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:55:00
source slurm/common.sh
model=$1; output=$2; shift 2
python scripts/eval/pair_attack.py --model "$model" --output_path "$output" "$@"
