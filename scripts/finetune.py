"""Stage 1: train one LoRA expert per harm category (also used for the SFT / dense-LoRA baselines).

Experts (Section 4.1): rank 32, alpha 64, dropout 0.1 on gate_proj / up_proj /
down_proj, lr 2e-4, 5 epochs, 512 tokens. Training data is loaded by name
(see safemoe/datasets.py); `<dataset>:<category>` selects one harm category.

    python scripts/finetune.py --data "pku:<UNSAFE_CATEGORY>" --output_dir checkpoints/experts/mistral/<category>

Baselines reuse this script:
    Knowledge-Only experts         --data pubmedqa | fiqa | mental_health | <cybersecurity-qa CSV>
    Refusal-Only experts           --data "pku:<UNSAFE_CATEGORY>" --replace_response "Sorry, I cannot help with that."
    Safe-Only experts              --data <safe responses of one category (JSON)>
    dense LoRA on pooled data      --data pku <safe responses JSON> --rank 64   (top-2 compute)
                                   or --rank 576                                 (18 x 32 parameters)
    LoRA-SFT-800                   --data <safe responses JSON>
    SFT-800 / SFT-8000             --data <safe responses JSON> --full_finetune --lr 1e-4 --epochs 10
"""

import argparse
import functools

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from safemoe.constants import TRAIN_TEMPLATE
from safemoe.data import to_prompt_response
from safemoe.datasets import load_records
from safemoe.training import CausalCollator, build_dataset
from safemoe.utils import parse_data_spec

# Trainer checkpoints store numpy RNG state; PyTorch >= 2.6 refuses to unpickle
# it on resume with the default weights_only=True. These are our own files.
torch.load = functools.partial(torch.load, weights_only=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="dataset names, name:category, or local files")
    p.add_argument("--split", default="train")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--rank", type=int, default=32, help="LoRA rank; alpha = 2 * rank")
    p.add_argument("--target_modules", nargs="+", default=["gate_proj", "up_proj", "down_proj"])
    p.add_argument("--full_finetune", action="store_true", help="train all weights instead of a LoRA")
    p.add_argument("--replace_response", default=None, help="train on this fixed response (Refusal-Only)")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--loss_on", choices=["full", "completion"], default="full",
                   help="'full' for Mistral experts, 'completion' for the Qwen2.5 experts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=-1, help="stop early (tests only)")
    p.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    return p.parse_args()


def main():
    args = parse_args()
    records = []
    for spec in args.data:
        name, category = parse_data_spec(spec)
        records += to_prompt_response(load_records(name, split=args.split, category=category))
    if args.replace_response is not None:
        records = [{"prompt": r["prompt"], "response": args.replace_response} for r in records]
    print(f"{len(records)} training samples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    if not args.full_finetune:
        lora = LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.1,
            target_modules=args.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=10,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=build_dataset(records, tokenizer, TRAIN_TEMPLATE, args.max_length, args.loss_on),
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
