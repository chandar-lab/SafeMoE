"""Small JSON / record helpers shared by the data, training and eval scripts."""

import glob
import json
import os
import random


def load_json(path):
    """Load a JSON list, or a JSONL file (one object per line)."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def expand_inputs(patterns):
    """Sorted files matching any of the glob patterns; an error if there are none."""
    files = sorted({f for pattern in patterns for f in glob.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"no input file matches {list(patterns)}")
    return files


def save_json(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def get_prompt(record):
    return record.get("prompt") or record.get("instruction") or record.get("question") or ""


def get_response(record):
    return record.get("response") or record.get("answer") or ""


def _first_present(record, keys):
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def to_prompt_response(records):
    """Normalise records to {"prompt", "response"}, dropping those missing either field.

    Empty responses are kept: the released subsets were drawn with this rule,
    and dropping them would change which rows a seeded shuffle selects.
    """
    out = []
    for r in records:
        prompt = r.get("prompt") or r.get("instruction")
        response = _first_present(r, ("response", "answer"))
        if prompt and response is not None:
            out.append({"prompt": prompt, "response": response})
    return out


def sample_per_category(records, n, seed=42):
    """Shuffle with a fixed seed and keep the first n records."""
    records = list(records)
    random.seed(seed)
    random.shuffle(records)
    return records[:n]
