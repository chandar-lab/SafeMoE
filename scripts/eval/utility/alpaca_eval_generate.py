"""Generate responses to the AlpacaEval instructions (Section 4.2.5, 1 GPU).

    python scripts/eval/utility/alpaca_eval_generate.py --model checkpoints/routers/safemoe_xl \
        --output_dir results/utility/alpaca_eval/safemoe_xl

All 805 instructions of tatsu-lab/alpaca_eval are used, in dataset order;
--max_samples N keeps random.sample(range(805), N) with --seed instead (still
in dataset order). Responses use the
"### Instruction / ### Response" prompt, up to 512 new tokens, sampling with
temperature 0.7 / top-p 0.9, cut at the first "\\n###". Writes
<output_dir>/responses.json with {instruction, output, generator, dataset,
reference}; existing outputs are resumed. Judge with alpaca_eval_score.py.
"""

import argparse
import os
import random

import torch
from datasets import load_dataset
from transformers import set_seed

from safemoe.data import load_json, save_json
from safemoe.generation import load_model
from safemoe.moe import reset_routing_cache

PROMPT = "### Instruction:\n{instruction}\n\n### Response:"


def load_instructions(max_samples, seed):
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    indices = range(len(dataset))
    if max_samples and max_samples < len(dataset):
        random.seed(seed)
        indices = sorted(random.sample(indices, max_samples))
    return [dataset[i] for i in indices]


@torch.no_grad()
def generate(model, tokenizer, instruction, max_new_tokens):
    inputs = tokenizer(PROMPT.format(instruction=instruction), return_tensors="pt").to(model.device)
    reset_routing_cache(model)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n###")[0].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="hub id, adapter dir, SafeMoE dir or training output dir")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default=None, help="generator name on the leaderboard (default: model dir name)")
    p.add_argument("--base_model", default=None, help="base model of a PEFT adapter")
    p.add_argument("--max_samples", type=int, default=None, help="seeded random subset (default: all instructions)")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--save_every", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    samples = load_instructions(args.max_samples, args.seed)
    model, tokenizer = load_model(args.model, args.base_model, dtype=getattr(torch, args.dtype))
    name = args.name or os.path.basename(args.model.rstrip("/"))

    out_path = os.path.join(args.output_dir, "responses.json")
    results = load_json(out_path) if os.path.exists(out_path) else []
    for i, example in enumerate(samples[len(results):], start=len(results)):
        results.append({
            "instruction": example["instruction"],
            "output": generate(model, tokenizer, example["instruction"], args.max_new_tokens),
            "generator": name,
            "dataset": example.get("dataset", "unknown"),
            "reference": example["output"],
        })
        if (i + 1) % args.save_every == 0:
            save_json(results, out_path)
    save_json(results, out_path)
    empty = sum(not r["output"] for r in results)
    print(f"{len(results)} responses ({empty} empty) -> {out_path}")


if __name__ == "__main__":
    main()
