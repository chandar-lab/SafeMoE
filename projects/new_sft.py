#!/usr/bin/env python
# sft_mistral.py
import os
import argparse
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

# ---------------------------
# Utilities
# ---------------------------
def init_seed(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def infer_data_files(path: str):
    """
    Returns a dict for datasets.load_dataset(..., data_files=...),
    mapping split names. We load a single file into 'train', then split 90/10.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".json", ".jsonl"]:
        raise ValueError("Data file must be .json or .jsonl containing records with 'prompt' and 'response'.")
    return {"train": path}

def format_sample(prompt: str, response: str) -> str:
    """
    Render a safe instruction-style format (no label leakage on eval-only prompts).
    """
    prompt = (prompt or "").strip()
    response = (response or "").strip()
    return f"[INST] {prompt} [/INST] {response}"

def main():
    parser = argparse.ArgumentParser(description="Simple SFT for Mistral-7B with TRL + Weights & Biases.")
    parser.add_argument("--model_name_or_path", type=str, default="mistral-7b-v0.1",
                        help="Base model or local path. Use base, not instruct, for clean SFT.")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to JSON/JSONL with fields 'prompt' and 'response'.")
    parser.add_argument("--output_dir", type=str, default="runs/mistral7b_sft",
                        help="Where to save checkpoints.")
    parser.add_argument("--wandb_project", type=str, default="mistral7b-sft",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional W&B run name.")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Fraction of data for eval.")
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "epoch", "steps"])
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable to reduce memory (slower).")
    parser.add_argument("--packing", action="store_true", default=False,
                        help="Leave False for tiny datasets to avoid packing bugs.")
    args = parser.parse_args()

    # ---------------------------
    # Reproducibility
    # ---------------------------
    init_seed(args.seed)

    # ---------------------------
    # Weights & Biases setup
    # ---------------------------
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    # ---------------------------
    # Load tokenizer & model
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # Mistral typically has no pad token; map pad to eos for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding for SFT training is fine
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map="auto"
    )
    # Recommended for training
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---------------------------
    # Load data
    # ---------------------------
    data_files = infer_data_files(args.train_data_path)
    ext = os.path.splitext(data_files["train"])[1].lower()
    ds = load_dataset("json", data_files=data_files, split="train")

    # Validate fields and create a single 'text' field = prompt+response
    def build_text(example):
        if "prompt" not in example or "response" not in example:
            raise KeyError("Each record must have 'prompt' and 'response'.")
        return {"text": format_sample(example["prompt"], example["response"])}

    ds = ds.map(build_text, remove_columns=[c for c in ds.column_names if c not in ("prompt", "response")],
                load_from_cache_file=False)

    # Train/Eval split
    if args.eval_ratio > 0:
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    # ---------------------------
    # TrainingArguments
    # ---------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        report_to=["wandb"],
        remove_unused_columns=False,  # important with TRL/SFTTrainer
        seed=args.seed,
    )

    # ---------------------------
    # SFTTrainer
    # ---------------------------
    # We pass dataset_text_field="text" so TRL builds labels over the full sequence.
    # For tiny datasets, keep packing=False (default here) to avoid sample-boundary label issues.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )

    # ---------------------------
    # Train
    # ---------------------------
    trainer.train()

    # ---------------------------
    # Save final
    # ---------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Model + tokenizer saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
