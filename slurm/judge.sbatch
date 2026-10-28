#!/bin/bash
# LLM judge (API calls only, no GPU).
# Usage: sbatch slurm/judge.sbatch <judge: gpt4o|claude|gpt5> <generation dir> <output dir>
#SBATCH --job-name=safemoe-judge
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=23:55:00
source slurm/common.sh
python scripts/judge.py --judge "$1" --inputs "$2/*.json" --output_dir "$3"
