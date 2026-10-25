#!/bin/bash
# Generate AlpacaEval responses on one GPU (Section 4.2.5); judge them on CPU with
# scripts/eval/utility/alpaca_eval_score.py. Resubmit the same command to resume.
# Usage: sbatch slurm/alpaca_eval_generate.sbatch <model> <output dir> [extra alpaca_eval_generate.py args]
#   sbatch slurm/alpaca_eval_generate.sbatch checkpoints/routers/safemoe_xl results/utility/alpaca_eval/safemoe_xl
#SBATCH --job-name=safemoe-alpaca-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=11:55:00
source slurm/common.sh
model=$1; out=$2; shift 2
python scripts/eval/utility/alpaca_eval_generate.py --model "$model" --output_dir "$out" "$@"
