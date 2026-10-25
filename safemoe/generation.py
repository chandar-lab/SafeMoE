"""Load any evaluated model (base, LoRA/SFT, SafeMoE) and generate responses."""

import glob
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from safemoe.constants import GENERATION_TEMPLATE, STOP_STRING
from safemoe.moe import is_safemoe_checkpoint, load_safemoe, reset_routing_cache, set_routing_mode


def resolve_checkpoint(path):
    """Return a loadable directory for a training output dir.

    Prefers a final save at the top level; otherwise the best checkpoint
    recorded by the Trainer, otherwise the latest checkpoint-N.
    """
    if not os.path.isdir(path):
        return path  # hub id
    for marker in ("adapter_config.json", "config.json"):
        if os.path.isfile(os.path.join(path, marker)):
            return path
    checkpoints = sorted(glob.glob(os.path.join(path, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"no model or checkpoint under {path}")
    state_path = os.path.join(checkpoints[-1], "trainer_state.json")
    if os.path.isfile(state_path):
        with open(state_path) as f:
            best = json.load(f).get("best_model_checkpoint")
        if best and os.path.isdir(os.path.join(path, os.path.basename(best.rstrip("/")))):
            return os.path.join(path, os.path.basename(best.rstrip("/")))
    return checkpoints[-1]


def load_model(path, base_model=None, routing_mode="per_token", dtype=torch.bfloat16):
    """Load a SafeMoE checkpoint, a bare PEFT adapter, or a plain HF model."""
    path = resolve_checkpoint(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        from peft import PeftConfig, PeftModel

        base_model = base_model or PeftConfig.from_pretrained(path).base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
        model = PeftModel.from_pretrained(model, path)
        tokenizer_path = path if os.path.isfile(os.path.join(path, "tokenizer_config.json")) else base_model
    elif is_safemoe_checkpoint(path):
        model = load_safemoe(path, torch_dtype=dtype).to(device)
        set_routing_mode(model, routing_mode)
        tokenizer_path = path
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)
        tokenizer_path = path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    model.eval()
    return model, tokenizer


class _StopOnString(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Sample one response with the evaluation settings of the paper.

    The decoded text is cut at the first blank line, as in all reported runs.
    """
    text = GENERATION_TEMPLATE.format(prompt=prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    stop = StoppingCriteriaList([_StopOnString(tokenizer.encode(STOP_STRING, add_special_tokens=False))])
    reset_routing_cache(model)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stop,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = decoded[len(text):] if decoded.startswith(text) else decoded
    return response.strip().split("\n\n")[0].strip()
