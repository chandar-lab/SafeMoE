"""Generate responses to every prompt of an evaluation dataset.

    python scripts/generate.py --model checkpoints/routers/safemoe_xl --data pku --split test \
        --output_dir results/generations/safemoe_xl_seed0 --seed 0

--data is a dataset name of safemoe/datasets.py (pku, advbench, harmbench,
beavertails, harmfulqa, xstest, orbench_toxic, orbench_hard, truthfulqa, ...)
or a local file. For pku, every harm category of the paper is evaluated
unless --categories is given. One file per category is written to
<output_dir>/<data>_<category>.json with {"instruction", "response", "category"};
existing outputs are resumed. Base models, PEFT adapters, fully fine-tuned
models and SafeMoE checkpoints are all accepted.
"""

import argparse
import os
from collections import defaultdict

import torch
from transformers import set_seed

from safemoe.constants import CATEGORIES, slug
from safemoe.data import load_json, save_json
from safemoe.datasets import load_records
from safemoe.generation import generate_response, load_model


def prompts_by_category(data, split, categories):
    if data == "pku":
        return {c: load_records("pku", split=split, category=c) for c in (categories or CATEGORIES)}
    groups = defaultdict(list)
    for r in load_records(data, split=split):
        if not categories or r["category"] in categories:
            groups[r["category"]].append(r)
    return dict(groups)


def format_prompt(record):
    context = record.get("context")
    if context:
        return f"{context}\n\n---\n\n{record['prompt']}"
    return record["prompt"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="hub id, adapter dir, SafeMoE dir or training output dir")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default=None, help="dataset split (default: the dataset's evaluation split)")
    p.add_argument("--categories", nargs="+", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default=None, help="base model of a PEFT adapter (read from its config if unset)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--routing_mode", default="per_token", choices=["per_token", "first_layer", "last_token"])
    p.add_argument("--save_every", type=int, default=20)
    args = p.parse_args()

    split = args.split or ("test" if args.data == "pku" else None)
    groups = prompts_by_category(args.data, split, args.categories)
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.base_model, args.routing_mode, dtype=torch.bfloat16)

    name = os.path.splitext(os.path.basename(args.data))[0]
    for category, records in groups.items():
        out_path = os.path.join(args.output_dir, f"{name}_{slug(category)}.json")
        results = load_json(out_path) if os.path.exists(out_path) else []
        records = [r for r in records if r["prompt"] and r["prompt"].strip()]
        for i, record in enumerate(records[len(results):], start=len(results)):
            response = generate_response(model, tokenizer, format_prompt(record), max_new_tokens=args.max_new_tokens)
            result = {"instruction": record["prompt"], "response": response, "category": category}
            if record.get("context"):
                result["context"] = record["context"]
            results.append(result)
            if (i + 1) % args.save_every == 0:
                save_json(results, out_path)
        save_json(results, out_path)
        print(f"{category}: {len(results)} responses -> {out_path}")


if __name__ == "__main__":
    main()
