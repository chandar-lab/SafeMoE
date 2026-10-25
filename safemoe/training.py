"""Tokenisation and collation for causal-LM fine-tuning (experts, routers, SFT baselines).

This reproduces what trl's SFTTrainer did in the original runs: the text gets
the tokenizer's BOS, no EOS is appended, sequences are truncated to
`max_length`, and padding (pad = eos) is excluded from the loss.
"""

import torch
from datasets import Dataset


def _tokenize_example(tokenizer, prompt_text, full_text, max_length, loss_on):
    ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]
    labels = list(ids)
    if loss_on == "completion":
        n_prompt = len(tokenizer(prompt_text.rstrip(" "), truncation=True, max_length=max_length)["input_ids"])
        labels[:n_prompt] = [-100] * n_prompt
    return {"input_ids": ids, "labels": labels}


def build_dataset(records, tokenizer, template, max_length=512, loss_on="full", chat=False):
    """Turn {"prompt", "response"} records into a tokenised Dataset.

    template: format string with {prompt} and {response}; the prompt part used
              for completion-only loss is everything before "{response}".
    loss_on:  "full" (loss on question and answer) or "completion".
    chat:     wrap the formatted question/answer in the tokenizer's chat
              template (used for the Qwen2.5-Instruct router).
    """
    if loss_on not in ("full", "completion"):
        raise ValueError(f"loss_on must be 'full' or 'completion', got {loss_on!r}")
    question_part, answer_part = template.split("{response}")[0], "{response}"
    rows = []
    for r in records:
        if chat:
            user = [{"role": "user", "content": f"### Question: {r['prompt']}"}]
            assistant = [{"role": "assistant", "content": f"### Answer: {r['response']}"}]
            prompt_text = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(user + assistant, tokenize=False)
        else:
            prompt_text = question_part.format(prompt=r["prompt"])
            full_text = prompt_text + answer_part.format(response=r["response"])
        rows.append(_tokenize_example(tokenizer, prompt_text, full_text, max_length, loss_on))
    return Dataset.from_list(rows)


class CausalCollator:
    """Right-pad input_ids with pad_token_id and labels with -100."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }
