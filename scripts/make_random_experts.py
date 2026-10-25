"""Create randomly initialised, never-trained LoRA experts for the router-only control (Table 9).

PEFT's default init sets lora_B = 0, which would make every expert a no-op and
the router irrelevant. Both A and B keep nn.Linear's random init instead
(init_lora_weights=False), so each expert is a fixed, distinct random map.

    python scripts/make_random_experts.py --experts XL --output_dir checkpoints/experts/mistral_random
"""

import argparse
import os
import shutil

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from safemoe.constants import EXPERT_SETS, slug


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", default="XL", choices=EXPERT_SETS)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    names = [slug(n) for n in EXPERT_SETS[args.experts]]
    config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.1,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=False,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    torch.manual_seed(args.seed)
    model = get_peft_model(model, config, adapter_name=names[0])
    for i, name in enumerate(names[1:], start=1):
        torch.manual_seed(args.seed + i)
        model.add_adapter(name, config)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.half()

    for name in names:
        out = os.path.join(args.output_dir, name)
        model.save_pretrained(out, selected_adapters=[name])
        nested = os.path.join(out, name)  # PEFT nests non-"default" adapters one level down
        if os.path.isdir(nested):
            for f in os.listdir(nested):
                shutil.move(os.path.join(nested, f), os.path.join(out, f))
            os.rmdir(nested)
        print(f"{name}: random expert -> {out}")


if __name__ == "__main__":
    main()
