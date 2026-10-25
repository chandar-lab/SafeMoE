# SafeMoE

Code for **Dialectics of Alignment: Harnessing Unsafe Knowledge for Dynamic Safety Routing** (NeurIPS 2026).

SafeMoE trains one LoRA expert per harm category on *unsafe* responses
(PKU-SafeRLHF), then trains a per-layer linear router on a small set of
safe-informative responses. At inference the router mixes the top-2 experts
for every token and layer, which steers generation toward answers that are
both safe and informative.

> This repository works with harmful prompts and responses. The unsafe LoRA
> experts are not released (see *Release policy*).

## Installation

Python 3.10 and a CUDA 12 GPU. `requirements.txt` pins every package to the
version the code was tested with:

```bash
git clone <REPO_URL> SafeMoE && cd SafeMoE
python3.10 -m venv ~/envs/safemoe && source ~/envs/safemoe/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # all dependencies, pinned
pip install -e .                  # the safemoe package itself
```

On a cluster with environment modules, load Python and CUDA first (on Mila:
`module load python/3.10 cuda/12.6.0`). The Slurm scripts expect the
environment in `SAFEMOE_VENV` (`export SAFEMOE_VENV=~/envs/safemoe`).

Datasets and models are downloaded from the Hugging Face Hub when a script
first uses them. The LLM judge, the GPT-4 data collection and the PAIR
attacker read `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the environment;
gated models (Llama Guard 3) need `HF_TOKEN`.

## Repository layout

```
safemoe/                  library
  moe.py                  LoRAMoeLayer, compose_safemoe, load_safemoe, routing variants
  datasets.py             every dataset of the paper, loaded by name
  training.py             tokenisation / collation for experts, routers and SFT baselines
  generation.py           model loading (base, PEFT, SafeMoE) and the evaluation decoding
  judge.py                LLM judge prompt (Appendix C), parsing and metrics
  constants.py            harm categories, expert sets (4 / L / XL), prompt templates
scripts/
  finetune.py             stage 1: LoRA experts (and SFT / dense-LoRA baselines)
  train_router.py         stage 2: router training of a SafeMoE model
  make_random_experts.py  untrained experts for the router-only control
  generate.py             response generation
  judge.py                safety / informativeness judging
  aggregate_results.py    per-category tables (mean ± std over seeds)
  data/                   GPT-4 safe-informative data collection
  eval/, analysis/, baselines/   further experiments (see below)
configs/router/           one config per model / ablation
slurm/                    Slurm job scripts
tests/                    CPU tests on a tiny random model
```

## Method in code

`safemoe/moe.py` replaces every `gate_proj`, `up_proj` and `down_proj` of the
base model by

    h = W x + Σ_{i ∈ top-k(r)} softmax(r_top-k)_i · B_i A_i x · α/r,   r = Vᵀx

where `(A_i, B_i)` are the frozen LoRA experts and `V` (`*.gate.weight`) is the
router of the layer. Checkpoints keep the parameter names of
[mergoo](https://github.com/Leeroo-AI/mergoo).

## Data

All data is loaded by name with `safemoe.datasets.load_records` (the Hub ids
are in that file):

| Name | Use |
|---|---|
| `pku` | PKU-SafeRLHF-QA unsafe rows; `pku:<category>` selects one harm category (experts, evaluation) |
| `pubmedqa`, `fiqa`, `mental_health` | knowledge experts (Knowledge-Only, SafeMoE-XL-Knowledge); `mental_health` is gated: request access on its Hub page |
| `advbench`, `harmbench`, `harmbench_contextual`, `beavertails`, `harmfulqa`, `xstest`, `orbench_toxic`, `orbench_hard`, `truthfulqa` | evaluation benchmarks |
| `alpaca` | retain data of the unlearning baseline |
| a local `.json` / `.jsonl` / `.csv` | the GPT-4 safe responses, the Kaggle cybersecurity-qa CSV |

Harm categories use the names of the paper (Table 7, `safemoe/constants.py`),
e.g. `Drug Abuse & Weapons`, `Cybercrime`, `Psychological Harm`.

**Safe-informative router data (Appendix B).** GPT-4 answers the harmful
prompts of Drug Abuse & Weapons, Psychological Harm, Cybercrime and Economic
Crime; blanket refusals are resampled and refusal sentences are removed. The
paper uses 200 responses per category:

```bash
python scripts/data/generate_safe_responses.py --output data/safe_raw.json --per_category 200
python scripts/data/clean_safe_responses.py --inputs data/safe_raw.json --output <PATH_TO_SAFE_RESPONSES_JSON>
```

## Reproducing the paper

### 1. Experts (stage 1)

One LoRA per harm category (rank 32, α 64, lr 2e-4, 5 epochs) on the unsafe
PKU-SafeRLHF training responses of that category:

```bash
python scripts/finetune.py --data "pku:<UNSAFE_CATEGORY>" --output_dir checkpoints/experts/mistral/<category_slug>
# e.g. --data "pku:Drug Abuse & Weapons" --output_dir checkpoints/experts/mistral/drug_abuse_weapons

# Qwen2.5-3B experts
python scripts/finetune.py --data "pku:<UNSAFE_CATEGORY>" --base_model Qwen/Qwen2.5-3B --loss_on completion \
    --output_dir checkpoints/experts/qwen/<category_slug>
```

`<category_slug>` is `safemoe.constants.slug("<UNSAFE_CATEGORY>")`. The
categories of SafeMoE-4 / -L / -XL are listed in `EXPERT_SETS` of
`safemoe/constants.py`.

### 2. Router (stage 2)

```bash
python scripts/train_router.py --config configs/router/safemoe_xl.yaml data.path=<PATH_TO_SAFE_RESPONSES_JSON>
```

Top-2 routing, lr 1e-4, batch size 4, 10 epochs, AdamW. The model is saved to
`checkpoints/routers/<name>`.

### 3. Generation, judging, tables (Table 1)

```bash
for seed in 0 1 2; do
  python scripts/generate.py --model checkpoints/routers/safemoe_xl --data pku --seed $seed \
      --output_dir results/generations/safemoe_xl_seed$seed
  python scripts/judge.py --judge gpt4o --inputs "results/generations/safemoe_xl_seed$seed/*.json" \
      --output_dir results/judgements/gpt4o/safemoe_xl_seed$seed
done
python scripts/aggregate_results.py --runs results/judgements/gpt4o/safemoe_xl_seed{0,1,2}
```

`--data pku` evaluates every harm category on the PKU-SafeRLHF test split.
Generation samples with temperature 0.7 and top-p 0.9, up to 256 new tokens,
on the prompt `### Question: …\n### Answer:`. Baselines use the same command
with `--model mistralai/Mistral-7B-v0.1`, a PEFT adapter directory or any HF
model. `--judge claude` gives the Claude-Opus judge (Table 10).

On Slurm: `slurm/train_expert.sbatch`, `slurm/train_router.sbatch`,
`slurm/generate.sbatch`, `slurm/judge.sbatch`, or
`bash slurm/submit_main.sh configs/router/safemoe_xl.yaml <PATH_TO_SAFE_RESPONSES_JSON>`
to submit experts → router → three seeds of generation and judging as one chain.

### Experiment index

| Paper | Experiment | How |
|---|---|---|
| Table 1 | SafeMoE-4 / -L / -XL / -Qwen | `configs/router/safemoe_{4,l,xl,qwen}.yaml` |
| Table 2 | Refusal-Only | `finetune.py --data "pku:<UNSAFE_CATEGORY>" --replace_response "Sorry, I cannot help with that."` for the 4 categories, `configs/router/refusal_only.yaml` |
| Table 2 | SFT-800 / SFT-8000 | `finetune.py --data <safe responses> --full_finetune --lr 1e-4 --epochs 10` |
| Table 2 | Safe-Only experts | one expert per safe-data category on its safe responses, `configs/router/safe_only.yaml` |
| Table 2 | Knowledge-Only | `finetune.py --data pubmedqa` (and `fiqa`, `mental_health`, the cybersecurity-qa CSV), `configs/router/knowledge_only.yaml` |
| Table 9 | dense LoRA on pooled data | `finetune.py --data pku <safe responses> --rank 64` (or `--rank 576`) |
| Table 9 | router-only, random experts | `make_random_experts.py`, `configs/router/router_only_random_{4,18}.yaml` |
| Table 9 | LoRA-SFT-800 | `finetune.py --data <safe responses>` |
| Fig. 3, Tables 12-14 | safe samples per domain | `configs/router/safemoe_xl_{20,50,100}per_cat.yaml` (set `model.experts` for 4 / L) |
| Table 15 | SafeMoE-XL-Knowledge | `configs/router/safemoe_xl_knowledge.yaml` |
| Sec. 4.2.8 | first-layer / last-token routing | `configs/router/{first_layer,last_token}_routing.yaml`; generate with `--routing_mode` of the same name |
| Table 3, Tables 17-21 | external benchmarks | `generate.py --data advbench` (or `harmbench`, `beavertails`, `harmfulqa`, `xstest`, `orbench_toxic`, `orbench_hard`), then `judge.py` |
| Table 4 | PAIR attack (ASR, MJS) | `scripts/eval/pair_attack.py` |
| Table 5 | judge meta-evaluation | `judge.py --judge gpt5` / `--judge claude`, `scripts/eval/llama_guard.py`, `scripts/analysis/judge_agreement.py` |
| Table 6, App. Q | factuality of safe responses | `scripts/eval/factuality/` |
| Sec. 4.2.5 | TruthfulQA, AlpacaEval | `scripts/eval/utility/` |
| Sec. 4.2.6, App. P | over-refusal rate | `generate.py --data orbench_hard`, then `scripts/eval/over_refusal.py` |
| App. L | OpenAI Moderation | `scripts/eval/openai_moderation.py` |
| App. M | human annotation agreement | `scripts/analysis/human_agreement.py` |
| App. N, Figs. 5-6 | expert activation | `scripts/analysis/expert_activation.py` |
| App. R | gradient-ascent unlearning | `scripts/baselines/unlearn.py` |
| App. R | SafeMoE on Qwen2.5-3B-Instruct | `configs/router/safemoe_qwen_instruct.yaml` |

**Retired API models.** Some API models used in the paper are no longer served;
the scripts keep the paper's model as the default and take an override:
GPT-4 for the safe-data collection (`generate_safe_responses.py --model`),
Claude-3.5-Sonnet as the PAIR attacker (`pair_attack.py --attacker_model`), and
gpt-4-turbo as the AlpacaEval annotator (`alpaca_eval_score.py --judge_model`).

Every script documents its arguments in `--help`. CPU unit tests:
`python -m pytest tests`.

## Release policy

The base models and datasets are public on the Hugging Face Hub. To avoid making
harmful capabilities more accessible, the unsafe LoRA expert weights are not
released. SafeMoE is an inference-time gating defense: bypassing the router or
merging an unsafe expert into the base model removes the protection.

## License

`safemoe/moe.py` is derived from mergoo (LGPL-3.0); the repository is released
under the same license (see `LICENSE`).

## Citation

```bibtex
@inproceedings{hashemzadeh2026dialectics,
  title     = {Dialectics of Alignment: Harnessing Unsafe Knowledge for Dynamic Safety Routing},
  author    = {Hashemzadeh, Maryam and Huang, Jerry and Kim, Minseon and C{\^o}t{\'e}, Marc-Alexandre and Chandar, Sarath},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
