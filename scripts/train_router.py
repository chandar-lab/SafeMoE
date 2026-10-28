"""Stage 2: compose frozen LoRA experts into a SafeMoE model and train the routers.

    python scripts/train_router.py --config configs/router/safemoe_xl.yaml [key=value ...]

Any config value can be overridden on the command line, e.g.
`data.path=/path/to/safe_responses.json`.
"""

import argparse
import functools
import os
import random
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments

from safemoe.constants import TRAIN_TEMPLATE, expert_names, slug
from safemoe.data import to_prompt_response
from safemoe.datasets import load_records
from safemoe.moe import compose_safemoe, moe_layers, set_routing_mode, set_trainable
from safemoe.training import CausalCollator, build_dataset
from safemoe.utils import load_config

# Trainer checkpoints store numpy RNG state; PyTorch >= 2.6 refuses to unpickle
# it on resume with weights_only=True. These are our own files.
torch.load = functools.partial(torch.load, weights_only=False)


def expert_paths(cfg):
    names = expert_names(cfg.model.experts)
    paths = [os.path.join(cfg.model.experts_dir, slug(name)) for name in names]
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(f"missing experts: {missing}")
    return paths


def per_category(records, n, seed):
    """Keep n records of each category (data-efficiency ablation, Figure 3)."""
    groups = defaultdict(list)
    for r in records:
        groups[r["category"]].append(r)
    rng = random.Random(seed)
    return [r for _, group in sorted(groups.items()) for r in rng.sample(group, min(n, len(group)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    print(OmegaConf.to_yaml(cfg))
    t = cfg.training

    paths = expert_paths(cfg)
    model = compose_safemoe(cfg.model.base_model, paths, num_experts_per_tok=cfg.model.num_experts_per_tok, seed=cfg.seed)
    set_routing_mode(model, cfg.model.routing_mode)
    set_trainable(model, cfg.model.trainable)
    if cfg.model.routing_mode == "first_layer":
        # Only block-0 routers are ever called; freeze the unused ones.
        for _, layer in moe_layers(model):
            layer.gate.weight.requires_grad_(layer.is_router)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{len(paths)} experts, top-{cfg.model.num_experts_per_tok}, trainable parameters: {n_trainable:,}")

    records = load_records(cfg.data.path)
    if cfg.data.samples_per_category:
        records = per_category(records, cfg.data.samples_per_category, cfg.seed)
    train = to_prompt_response(records)
    print(f"router data: {len(train)} samples")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    build = functools.partial(build_dataset, tokenizer=tokenizer, template=TRAIN_TEMPLATE, max_length=t.max_length,
                              loss_on=cfg.data.loss_on, chat=cfg.data.chat_template)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=t.batch_size,
        gradient_accumulation_steps=t.grad_accum,
        learning_rate=t.learning_rate,
        optim="adamw_torch",
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        save_steps=t.save_steps,
        save_total_limit=1,
        logging_steps=10,
        logging_first_step=True,
        bf16=torch.cuda.is_available(),
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=build(train),
        data_collator=CausalCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=cfg.resume or None)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "safemoe_train_config.yaml"))


if __name__ == "__main__":
    main()
