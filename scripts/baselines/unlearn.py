"""Gradient-ascent + KL-retain unlearning baseline for Mistral-7B (Appendix R).

Full fine-tune (no LoRA). The objective minimised by gradient descent is

    J(theta) = -alpha * E_{(x,y)~forget}[ CE_theta(x,y) ]
               + beta  * E_{x~retain}[ KL( P_base(.|x) || P_theta(.|x) ) ]

so the forget-set cross-entropy is ascended (unlearning) while the KL to the
frozen base model on the retain set is descended (anchoring alignment and
utility). Both sets are built here by name (safemoe/datasets.py), in full:

    forget  unsafe PKU-SafeRLHF train rows of the expert categories (--expert_set, default XL)
    retain  --retain_data: the GPT-4 safe-informative router responses and Alpaca

The base model is kept as a frozen, unsharded reference, one copy per GPU;
theta is wrapped by DeepSpeed ZeRO-2 through accelerate (fp32 master weights
and Adam states for a 7B model do not fit on one GPU next to the reference).

    accelerate launch --num_processes 4 --use_deepspeed \
        --deepspeed_config_file configs/accelerate/unlearn_zero2.json \
        scripts/baselines/unlearn.py --retain_data <PATH_TO_SAFE_RESPONSES_JSON> alpaca \
        --output_dir checkpoints/baselines/unlearn

--resume continues from the latest step-N checkpoint in output_dir.
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from safemoe.constants import EXPERT_SETS, TRAIN_TEMPLATE, pku_key
from safemoe.data import to_prompt_response
from safemoe.datasets import load_records
from safemoe.utils import parse_data_spec


def forget_records(expert_set):
    """Unsafe PKU train rows with at least one harm category of the expert set."""
    keys = {pku_key(c) for c in EXPERT_SETS[expert_set]}
    rows = [r for r in load_records("pku", split="train") if keys & set(r["category"].split(","))]
    return to_prompt_response(rows)


def retain_records(specs):
    records = []
    for spec in specs:
        name, category = parse_data_spec(spec)
        records += to_prompt_response(load_records(name, category=category))
    return records


class PromptResponseDataset(Dataset):
    """Left-to-right LM examples in the "### Question / ### Answer" format, padded to max_length."""

    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = TRAIN_TEMPLATE.format(prompt=r["prompt"], response=r["response"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        input_ids, attention_mask = enc["input_ids"][0], enc["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def cycle(loader):
    while True:
        yield from loader


def causal_lm_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)).float(),
                           shift_labels.view(-1), ignore_index=-100)


def token_kl(student_logits, teacher_logits, attention_mask):
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    mask = attention_mask.float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expert_set", choices=sorted(EXPERT_SETS), default="XL",
                   help="forget the unsafe PKU train rows of these expert categories")
    p.add_argument("--retain_data", nargs="+", required=True,
                   help="retain datasets: names, name:category or local files (safe responses JSON, alpaca)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--alpha", type=float, default=1.0, help="forget-ascent coefficient")
    p.add_argument("--beta", type=float, default=1.0, help="KL retain coefficient")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=2, help="per-GPU batch size (forget and retain each)")
    p.add_argument("--num_epochs", type=int, default=1, help="epochs over the forget set")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--forget_loss_cap", type=float, default=12.0,
                   help="clamp the ascended forget CE loss to avoid runaway divergence")
    p.add_argument("--max_steps", type=int, default=None, help="stop early (tests only)")
    p.add_argument("--resume", action="store_true", help="resume from the latest step-N checkpoint in output_dir")
    return p.parse_args()


def latest_checkpoint(output_dir):
    steps = glob.glob(os.path.join(output_dir, "step-*"))
    return max(steps, key=lambda p: int(p.rsplit("-", 1)[-1])) if steps else None


def main():
    args = parse_args()
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    forget_ds = PromptResponseDataset(forget_records(args.expert_set), tokenizer, args.max_length)
    retain_ds = PromptResponseDataset(retain_records(args.retain_data), tokenizer, args.max_length)
    accelerator.print(f"forget: {len(forget_ds)} samples, retain: {len(retain_ds)} samples")
    forget_loader = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    retain_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Frozen reference theta_base, one unsharded copy per GPU (outside the DeepSpeed engine).
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    ref_model.to(accelerator.device).eval()
    ref_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(forget_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.03 * total_steps)), total_steps)

    model, optimizer, forget_loader, retain_loader, scheduler = accelerator.prepare(
        model, optimizer, forget_loader, retain_loader, scheduler)
    retain_iter = cycle(retain_loader)

    global_step = 0
    resume_dir = latest_checkpoint(args.output_dir) if args.resume else None
    if resume_dir:
        accelerator.print(f"resuming from {resume_dir}")
        accelerator.load_state(resume_dir)
        global_step = int(resume_dir.rsplit("-", 1)[-1]) + 1  # step-N is saved after update N

    def save_checkpoint(step, final=False):
        accelerator.wait_for_everyone()
        path = os.path.join(args.output_dir, "final" if final else f"step-{step}")
        accelerator.save_state(path)
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(
                path, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
            tokenizer.save_pretrained(path)
        accelerator.print(f"saved {path}")

    model.train()
    step_in_run = 0
    skipped = 0
    for _ in range(args.num_epochs):
        for forget_batch in forget_loader:
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            if step_in_run < global_step:  # fast-forward on resume
                step_in_run += 1
                continue
            retain_batch = next(retain_iter)

            # One combined forward through the ZeRO-2-wrapped model: two separate
            # model(...) calls in one backward trip its gradient-reduction hooks.
            fb = forget_batch["input_ids"].size(0)
            out = model(
                input_ids=torch.cat([forget_batch["input_ids"], retain_batch["input_ids"]], dim=0),
                attention_mask=torch.cat([forget_batch["attention_mask"], retain_batch["attention_mask"]], dim=0),
            )
            forget_loss = causal_lm_loss(out.logits[:fb], forget_batch["labels"])
            with torch.no_grad():
                teacher = ref_model(input_ids=retain_batch["input_ids"], attention_mask=retain_batch["attention_mask"])
            kl = token_kl(out.logits[fb:], teacher.logits, retain_batch["attention_mask"])

            capped_forget = torch.clamp(forget_loss, max=args.forget_loss_cap)
            loss = -args.alpha * capped_forget + args.beta * kl

            if not torch.isfinite(loss):
                skipped += 1
                accelerator.print(f"step {global_step}: non-finite loss, skipping ({skipped} total)")
                optimizer.zero_grad()
                global_step += 1
                step_in_run += 1
                continue

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % 10 == 0:
                accelerator.print(f"step {global_step}/{total_steps}: forget={forget_loss.item():.4f} "
                                  f"kl={kl.item():.4f} total={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")
            if global_step > 0 and global_step % args.save_steps == 0:
                save_checkpoint(global_step)
            global_step += 1
            step_in_run += 1

    save_checkpoint(global_step, final=True)
    accelerator.print(f"done. {skipped} steps skipped for non-finite loss.")


if __name__ == "__main__":
    main()
