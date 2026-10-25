# Sourced by every job script in slurm/. Run jobs from the repository root:
#   sbatch slurm/<script>.sbatch ...
# Set SAFEMOE_VENV to a virtualenv with the requirements installed. API keys
# (OPENAI_API_KEY, ANTHROPIC_API_KEY) and HF_TOKEN come from your environment.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.10 cuda/12.6.0 2>/dev/null || true
set -euo pipefail
source "${SAFEMOE_VENV:?set SAFEMOE_VENV to your virtualenv}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
