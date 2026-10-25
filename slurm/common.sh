# Sourced by every job script in slurm/. Run jobs from the repository root:
#   sbatch slurm/<script>.sbatch ...
# Submit from a shell where the SafeMoE environment is active (or set SAFEMOE_VENV
# to its path). API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) and HF_TOKEN come
# from your environment.
set -euo pipefail
if [ -n "${SAFEMOE_VENV:-}" ]; then source "$SAFEMOE_VENV/bin/activate"; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
