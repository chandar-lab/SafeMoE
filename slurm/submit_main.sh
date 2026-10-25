#!/bin/bash
# Submit the SafeMoE pipeline for one router config as a chain of dependent jobs:
#   one expert job per category -> router -> generation x 3 seeds -> GPT-4o judge.
# Run from the repository root:
#   bash slurm/submit_main.sh configs/router/safemoe_xl.yaml <PATH_TO_SAFE_RESPONSES_JSON>
# Experts that already have a final adapter are skipped.
set -euo pipefail
config=$1; safe_data=$2
source "${SAFEMOE_VENV:?set SAFEMOE_VENV to your virtualenv}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

read -r name base experts_dir < <(python -c "
import sys; from safemoe.utils import load_config
c = load_config(sys.argv[1]); print(c.name, c.model.base_model, c.model.experts_dir)" "$config")
mapfile -t experts < <(python -c "
import sys; from safemoe.constants import expert_names; from safemoe.utils import load_config
for n in expert_names(load_config(sys.argv[1]).model.experts): print(n)" "$config")

extra=()
[[ "$base" == Qwen/* ]] && extra=(--base_model "$base" --loss_on completion)
deps=()
for category in "${experts[@]}"; do
    out="$experts_dir/$(python -c "import sys; from safemoe.constants import slug; print(slug(sys.argv[1]))" "$category")"
    [ -f "$out/adapter_model.safetensors" ] && continue
    deps+=("$(sbatch --parsable slurm/train_expert.sbatch "$out" --data "pku:$category" "${extra[@]}")")
done
dep_flag=(); [ ${#deps[@]} -gt 0 ] && dep_flag=(--dependency="afterok:$(IFS=:; echo "${deps[*]}")")

router=$(sbatch --parsable "${dep_flag[@]}" slurm/train_router.sbatch "$config" "data.path=$safe_data")
echo "experts: ${deps[*]:-none}  router: $router"
for seed in 0 1 2; do
    gen_dir="results/generations/${name}_seed${seed}"
    gen=$(sbatch --parsable --dependency="afterok:$router" slurm/generate.sbatch \
        "checkpoints/routers/$name" "$gen_dir" "$seed" --data pku)
    judge=$(sbatch --parsable --dependency="afterok:$gen" slurm/judge.sbatch gpt4o "$gen_dir" "results/judgements/gpt4o/${name}_seed${seed}")
    echo "seed $seed: generate $gen, judge $judge"
done
echo "afterwards: python scripts/aggregate_results.py --runs results/judgements/gpt4o/${name}_seed{0,1,2}"
