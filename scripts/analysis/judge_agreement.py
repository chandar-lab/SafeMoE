"""Agreement between the GPT-4o judge and other evaluators on the same responses (Table 5).

    python scripts/analysis/judge_agreement.py \
        --reference results/judgements/gpt4o/safemoe_xl_seed0 \
        --compare gpt5=results/judgements/gpt5/safemoe_xl_seed0 \
                  claude=results/judgements/claude/safemoe_xl_seed0 \
                  llama_guard=results/judgements/llama_guard/safemoe_xl_seed0

For every comparator:
  A%  binary safety agreement, over pairs where both labels are valid;
  r   Pearson correlation of the 0-10 scores over those pairs where both
      scores exist (unsafe responses keep score 0, as in the reported numbers;
      --both_safe_only restricts r to responses both judges call safe).
Label-only comparators (Llama Guard 3, OpenAI Moderation) report A% only.

Paths are judgement files or directories of them (paired by file name, e.g.
pku_<category>.json). The files must hold the same responses in the same
order: run scripts/judge.py, scripts/eval/llama_guard.py and
scripts/eval/openai_moderation.py on the same scripts/generate.py outputs.
"""

import argparse
import csv
import glob
import os

import numpy as np

from safemoe.data import load_json


def load_judged(path):
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get("results") or data.get("detailed_results") or []
    return data


def get_score(record):
    return record["score"] if "score" in record else record.get("rating_score")


def pair_files(reference, other):
    if os.path.isfile(reference):
        return [(reference, other)]
    pairs = []
    for ref in sorted(glob.glob(os.path.join(reference, "*.json"))):
        path = os.path.join(other, os.path.basename(ref))
        if os.path.exists(path):
            pairs.append((ref, path))
    return pairs


def compare(reference, other, both_safe_only=False):
    """Return (n_label, n_agree, ref_scores, other_scores) over aligned records."""
    if len(reference) != len(other):
        raise ValueError(f"length mismatch: {len(reference)} vs {len(other)}")
    n_label = n_agree = 0
    ref_scores, other_scores = [], []
    for r, o in zip(reference, other):
        if r.get("instruction") != o.get("instruction"):
            raise ValueError(f"records are not aligned: {r.get('instruction')!r} vs {o.get('instruction')!r}")
        if r["is_safe"] is None or o.get("is_safe") is None:
            continue
        n_label += 1
        n_agree += r["is_safe"] == o["is_safe"]
        if both_safe_only and not (r["is_safe"] and o["is_safe"]):
            continue
        if get_score(r) is not None and get_score(o) is not None:
            ref_scores.append(get_score(r))
            other_scores.append(get_score(o))
    return n_label, n_agree, ref_scores, other_scores


def row(name, file, n_label, n_agree, ref_scores, other_scores):
    agreement = 100 * n_agree / n_label if n_label else float("nan")
    r = np.corrcoef(ref_scores, other_scores)[0, 1] if len(ref_scores) > 1 else float("nan")
    return [name, file, n_label, f"{agreement:.2f}", len(ref_scores), f"{r:.4f}"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="GPT-4o judgement file or directory")
    p.add_argument("--compare", nargs="+", required=True, metavar="NAME=PATH",
                   help="comparator judgement files or directories")
    p.add_argument("--both_safe_only", action="store_true",
                   help="compute r only over responses both evaluators label safe")
    p.add_argument("--output", default=None, help="CSV path (printed only if unset)")
    args = p.parse_args()

    rows = []
    for spec in args.compare:
        name, path = spec.split("=", 1)
        pairs = pair_files(args.reference, path)
        if not pairs:
            raise FileNotFoundError(f"no file of {path} matches {args.reference}")
        total = [0, 0, [], []]
        for ref_path, other_path in pairs:
            n_label, n_agree, ref_s, other_s = compare(load_judged(ref_path), load_judged(other_path),
                                                       args.both_safe_only)
            rows.append(row(name, os.path.basename(ref_path), n_label, n_agree, ref_s, other_s))
            total = [total[0] + n_label, total[1] + n_agree, total[2] + ref_s, total[3] + other_s]
        if len(pairs) > 1:
            rows.append(row(name, "all", *total))

    header = ["evaluator", "file", "n_label", "agreement_pct", "n_score", "pearson_r"]
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerows([header, *rows])


if __name__ == "__main__":
    main()
