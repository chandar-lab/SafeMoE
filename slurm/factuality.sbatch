#!/bin/bash
# Factuality F_masked of safe responses (Appendix Q) on one small GPU; needs internet (Wikipedia, Semantic Scholar).
# Every judgement file of the directory is passed; only the five categories with a knowledge source
# (Public Health, Psychological Harm, Physical Harm, Drug Abuse & Weapons, Cybercrime) are scored.
# Usage: sbatch slurm/factuality.sbatch <judgement dir> <output dir> [extra run_factuality.py args]
#   e.g. sbatch slurm/factuality.sbatch results/judgements/claude/safemoe_xl_seed0 results/factuality/claude/safemoe_xl_seed0
#SBATCH --job-name=safemoe-factuality
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=11:55:00
source slurm/common.sh
judgements=$1; out=$2; shift 2
python scripts/eval/factuality/run_factuality.py --inputs "$judgements"/*.json --output_dir "$out" "$@"
