"""Clean GPT-4 safe responses into the router training file (Appendix B).

Refusal sentences are stripped from every response, rows left without
content are dropped, and duplicate prompts are removed.

    python scripts/data/clean_safe_responses.py --inputs data/safe_raw.json --output data/safe_responses.json

Inputs are JSON or CSV with `prompt`, `response` and `category` fields. The
output is the `data.path` of the router configs.
"""

import argparse

from safemoe.data import save_json
from safemoe.datasets import load_records
from safemoe.refusals import clean_response


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out, seen = [], set()
    for path in args.inputs:
        for r in load_records(path):
            prompt = (r["prompt"] or "").strip()
            response = clean_response(r["response"])
            if prompt and response and prompt not in seen:
                seen.add(prompt)
                out.append({"prompt": prompt, "response": response, "category": r["category"]})
    save_json(out, args.output)
    counts = {}
    for r in out:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"{len(out)} samples {counts} -> {args.output}")


if __name__ == "__main__":
    main()
