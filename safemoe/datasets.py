"""Load every dataset of the paper by name, straight from the Hugging Face Hub.

    load_records("pku", split="train", category="Cybercrime")        # unsafe expert data
    load_records("pku", split="test", category="Cybercrime")         # evaluation prompts
    load_records("harmbench")                                         # all categories
    load_records("path/to/file.json")                                 # any local JSON / JSONL / CSV

Every loader returns a list of {"prompt", "response" (if any), "category", ...}.
Nothing is subsampled: the whole split is returned.
"""

import csv
import os

from datasets import load_dataset

from safemoe.constants import pku_key
from safemoe.data import load_json


def _pku(split="train", category=None, safe=False):
    """PKU-SafeRLHF-QA rows. safe=False: unsafe rows of `category` (paper or PKU name; None = all)."""
    key = pku_key(category) if category else None
    rows = []
    for r in load_dataset("PKU-Alignment/PKU-SafeRLHF-QA", split=split):
        if r["is_safe"] != safe:
            continue
        active = [c for c, on in r["harm_category"].items() if on]
        if key is not None and key not in active:
            continue
        label = "safe" if safe else key or ",".join(active)
        rows.append({"prompt": r["prompt"], "response": r["response"], "category": label})
    return rows


def _rows(rows, prompt="prompt", response=None, category=None, extra=()):
    out = []
    for r in rows:
        record = {"prompt": r[prompt], "category": r[category] if category else "all"}
        if response:
            record["response"] = r[response]
        record.update({k: r[k] for k in extra})
        out.append(record)
    return out


def _beavertails(split="30k_test", category=None):
    rows = []
    for r in load_dataset("PKU-Alignment/BeaverTails", split=split):
        if r["is_safe"]:
            continue
        for cat, on in r["category"].items():  # multi-label: one record per active category
            if on and category in (None, cat):
                rows.append({"prompt": r["prompt"], "response": r["response"], "category": cat})
    return rows


def _xstest(split="test", category=None):
    rows = load_dataset("walledai/XSTest", split=split)
    rows = _rows(rows, category="label", extra=("type", "focus", "note"))
    return [r for r in rows if category in (None, r["category"])]


LOADERS = {
    # training data
    "pku": lambda split="train", category=None: _pku(split, category),
    "pku_safe": lambda split="train", category=None: _pku(split, None, safe=True),
    "pubmedqa": lambda split="train", category=None: _rows(
        load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split=split), "question", "long_answer"),
    "fiqa": lambda split="train", category=None: _rows(load_dataset("LLukas22/fiqa", split=split), "question", "answer"),
    "mental_health": lambda split="train", category=None: _rows(
        load_dataset("Amod/mental_health_counseling_conversations", split=split), "Context", "Response"),
    "alpaca": lambda split="train", category=None: [
        {"prompt": (r["instruction"] + ("\n" + r["input"] if r["input"] else "")), "response": r["output"], "category": "alpaca"}
        for r in load_dataset("tatsu-lab/alpaca", split=split)],
    # evaluation benchmarks
    "advbench": lambda split="train", category=None: _rows(load_dataset("walledai/AdvBench", split=split)),
    "harmbench": lambda split="train", category=None: _rows(
        load_dataset("walledai/HarmBench", "standard", split=split), category="category"),
    "harmbench_contextual": lambda split="train", category=None: _rows(
        load_dataset("walledai/HarmBench", "contextual", split=split), response="context", category="category"),
    "beavertails": _beavertails,
    "harmfulqa": lambda split="train", category=None: _rows(
        load_dataset("declare-lab/HarmfulQA", split=split), "question", category="topic"),
    "xstest": _xstest,
    "orbench_toxic": lambda split="train", category=None: _rows(
        load_dataset("bench-llm/or-bench", "or-bench-toxic", split=split), category="category"),
    "orbench_hard": lambda split="train", category=None: _rows(
        load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split=split), category="category"),
    "truthfulqa": lambda split="validation", category=None: _rows(
        load_dataset("truthfulqa/truthful_qa", "generation", split=split), "question", category="category",
        extra=("best_answer", "correct_answers", "incorrect_answers")),
}


def _local(path):
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = load_json(path)
    out = []
    for r in rows:
        prompt = r.get("prompt") or r.get("instruction") or r.get("question")
        response = r.get("response") if r.get("response") is not None else r.get("answer")
        out.append({**r, "prompt": prompt, "response": response, "category": r.get("category", "all")})
    return out


def load_records(name, split=None, category=None):
    """Records of a registered dataset (see LOADERS) or of a local file."""
    if os.path.exists(name):
        records = _local(name)
        return [r for r in records if category in (None, r["category"])]
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(LOADERS)} or a local file path")
    kwargs = {"category": category}
    if split:
        kwargs["split"] = split
    records = LOADERS[name](**kwargs)
    if name not in ("pku", "pku_safe", "beavertails", "xstest") and category is not None:
        records = [r for r in records if r["category"] == category]
    return records
