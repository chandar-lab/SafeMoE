"""Expert activation frequency per harm category under first-layer routing (Appendix N, Figs. 5-6).

    python scripts/analysis/expert_activation.py --model checkpoints/routers/first_layer_routing \
        --output_dir results/analysis/expert_activation

Prompts are loaded by name: by default every PKU-SafeRLHF test prompt of each
harm category of the paper (load_records("pku", split="test", category=c));
--data / --split / --categories select another dataset or a subset, and
--max_samples N keeps the first N prompts per category (all if unset). Each
prompt is run through one forward pass (prompt only, "### Question / ###
Answer" template). Forward hooks on the block-0 LoRAMoeLayers record the
experts their routers select for every prompt token; with first-layer routing
this decision is reused by every deeper block. The activation % of an expert
is its share of all (target module x token x top-k slot) selections of the
category. Experts are labelled with the paper category names of
constants.EXPERT_SETS[--expert_set], in checkpoint order.

Outputs: expert_activation.json, heatmap.png (categories x experts, each
category's own expert on the diagonal) and mean_activation.png (mean over
categories, sorted, with the uniform 1/E baseline). Categories without an
expert (Animal Abuse) are left out of both figures. --from_json re-plots a
saved expert_activation.json without a GPU.
"""

import argparse
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from safemoe.constants import CATEGORIES, EXPERT_SETS, GENERATION_TEMPLATE, pku_key, slug
from safemoe.data import load_json, save_json
from safemoe.datasets import load_records

# Expert / category names of the original analysis output (expert_activation_data.json).
_LEGACY_NAMES = {"drug_abuse": "Drug Abuse & Weapons", "national_security": "Endangering National Security"}


def paper_name(name):
    """Paper category name of a paper name, PKU key or slug; other names are returned unchanged."""
    try:
        key = pku_key(slug(_LEGACY_NAMES.get(slug(name), name)))
    except KeyError:
        return name
    return next(paper for paper, k in CATEGORIES.items() if k == key)


def prompts_by_category(data, split, categories):
    """{paper category name: [prompt, ...]} of a dataset loaded by name."""
    if data == "pku":
        groups = {paper_name(c): load_records("pku", split=split, category=c) for c in (categories or CATEGORIES)}
    else:
        groups = defaultdict(list)
        for r in load_records(data, split=split):
            if not categories or r["category"] in categories:
                groups[paper_name(r["category"])].append(r)
    return {c: [r["prompt"] for r in records if r["prompt"] and r["prompt"].strip()] for c, records in groups.items()}


def capture_routing(model):
    """Hook every block-0 MoE layer; returns the list the hooks append expert ids to."""
    from safemoe.moe import moe_layers

    selections = []

    def hook(layer, args, _output):
        _weights, selected = layer.route(args[0])
        selections.append(selected.flatten().cpu())

    handles = [layer.register_forward_hook(hook) for _, layer in moe_layers(model) if layer.is_router]
    if not handles:
        raise ValueError("model has no block-0 MoE layers")
    return selections, handles


@torch.no_grad()
def category_activation(model, tokenizer, prompts, selections, num_experts, max_input_tokens):
    selections.clear()
    for prompt in prompts:
        text = GENERATION_TEMPLATE.format(prompt=prompt)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
        model(**inputs)
    counts = np.bincount(torch.cat(selections).numpy(), minlength=num_experts)[:num_experts]
    return counts


def collect(args):
    from safemoe.generation import load_model

    experts = EXPERT_SETS[args.expert_set]
    split = args.split or ("test" if args.data == "pku" else None)
    groups = prompts_by_category(args.data, split, args.categories)
    model, tokenizer = load_model(args.model, routing_mode=args.routing_mode, dtype=torch.bfloat16)
    if model.config.num_experts != len(experts):
        raise ValueError(f"model has {model.config.num_experts} experts, expert set {args.expert_set} has {len(experts)}")
    selections, handles = capture_routing(model)
    categories = {}
    for category, prompts in groups.items():
        prompts = prompts[: args.max_samples]
        if not prompts:
            continue
        counts = category_activation(model, tokenizer, prompts, selections, len(experts), args.max_input_tokens)
        categories[category] = {
            "n_samples": len(prompts),
            "n_selections": int(counts.sum()),
            "activation_pct": (100.0 * counts / counts.sum()).tolist(),
        }
        print(f"{category}: {len(prompts)} prompts, top expert {experts[int(counts.argmax())]}")
    for h in handles:
        h.remove()
    mode = args.routing_mode or getattr(model.config, "routing_mode", "per_token")
    return {"model": args.model, "routing_mode": mode, "data": args.data, "split": split,
            "experts": experts, "categories": categories}


def load_saved(path):
    """Read expert_activation.json, or the original analysis' expert_activation_data.json."""
    data = load_json(path)
    if "experts" not in data:
        first = next(iter(data.values()))
        data = {"experts": first["expert_names"], "categories": data}
    return {
        "experts": [paper_name(e) for e in data["experts"]],
        "categories": {paper_name(c): v for c, v in data["categories"].items()},
    }


def plot_heatmap(categories, experts, matrix, path):
    # Columns follow the row order so that each category's own expert is on the diagonal.
    order = [experts.index(c) for c in categories if c in experts]
    order += [i for i in range(len(experts)) if i not in order]
    m = matrix[:, order]
    fig, ax = plt.subplots(figsize=(18, 9))
    im = ax.imshow(m, aspect="auto", cmap="YlOrRd", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01).set_label("Activation %", fontsize=11)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if m[i, j] > m.max() * 0.7 else "black")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([experts[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("Expert", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.set_title("Expert Activation Heatmap — First-Layer Routing MoE", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mean(experts, matrix, path):
    mean = matrix.mean(axis=0)
    order = np.argsort(mean)[::-1]
    values = mean[order]
    cmap = plt.get_cmap("RdYlGn_r")
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(order)), values, color=[cmap(v / values.max()) for v in values],
                  edgecolor="grey", linewidth=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([experts[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Activation % (avg over all categories)", fontsize=10)
    ax.set_title("Overall Expert Activation — First-Layer Routing MoE", fontsize=13)
    uniform = 100 / len(experts)
    ax.axhline(uniform, color="steelblue", linestyle="--", linewidth=1.2, label=f"Uniform baseline ({uniform:.1f}%)")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="SafeMoE checkpoint trained with first-layer routing")
    p.add_argument("--data", default="pku", help="dataset name of safemoe/datasets.py or a local file")
    p.add_argument("--split", default=None, help="dataset split (default: test for pku)")
    p.add_argument("--categories", nargs="+", default=None, help="categories to run (default: all)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--from_json", default=None, help="re-plot a saved activation file instead of running the model")
    p.add_argument("--expert_set", default="XL", choices=sorted(EXPERT_SETS), help="expert order of the checkpoint")
    p.add_argument("--routing_mode", default=None, choices=["per_token", "first_layer", "last_token"])
    p.add_argument("--max_samples", type=int, default=None, help="prompts per category (all if unset)")
    p.add_argument("--max_input_tokens", type=int, default=256)
    p.add_argument("--exclude", nargs="*", default=["Animal Abuse"], help="categories left out of the figures")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.from_json:
        data = load_saved(args.from_json)
    else:
        if not args.model:
            p.error("--model is required unless --from_json is given")
        data = collect(args)
        save_json(data, os.path.join(args.output_dir, "expert_activation.json"))

    experts = data["experts"]
    exclude = {paper_name(c) for c in args.exclude}
    categories = [c for c in data["categories"] if c not in exclude]
    matrix = np.array([data["categories"][c]["activation_pct"] for c in categories])
    plot_heatmap(categories, experts, matrix, os.path.join(args.output_dir, "heatmap.png"))
    plot_mean(experts, matrix, os.path.join(args.output_dir, "mean_activation.png"))
    print(f"figures -> {args.output_dir}")


if __name__ == "__main__":
    main()
