#!/bin/bash
# Gradient-ascent + KL-retain unlearning baseline (Appendix R): full fine-tune of Mistral-7B
# with DeepSpeed ZeRO-2 data parallelism, one process per GPU. Forget / retain sets are
# built by scripts/baselines/unlearn.py from the Hub (and the safe responses file).
# Usage: sbatch slurm/unlearn.sbatch <output_dir> <unlearn.py args>
#   sbatch slurm/unlearn.sbatch checkpoints/baselines/unlearn --retain_data <PATH_TO_SAFE_RESPONSES_JSON> alpaca
# Resubmit the same command to resume from the latest step-N checkpoint.
#SBATCH --job-name=safemoe-unlearn
#SBATCH --output=logs/%x-%j.out
# Needs 4 GPUs with about 80 GB of memory each.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=23:55:00
source slurm/common.sh
out=$1; shift
resume=""; ls -d "$out"/step-* >/dev/null 2>&1 && resume="--resume"
NGPU=$(nvidia-smi -L | wc -l)
accelerate launch --num_processes "$NGPU" --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --use_deepspeed --deepspeed_config_file configs/accelerate/unlearn_zero2.json \
    scripts/baselines/unlearn.py --output_dir "$out" $resume "$@"
