"""Judge generated responses for safety and informativeness (CPU only, API calls).

    python scripts/judge.py --inputs 'results/generations/safemoe_xl_seed0/*.json' \
        --judge gpt4o --output_dir results/judgements/gpt4o/safemoe_xl_seed0

The OpenAI / Anthropic clients read their API key from the environment
(OPENAI_API_KEY for gpt4o and gpt5, ANTHROPIC_API_KEY for claude). Each input
gives <output_dir>/<input name>.json with a summary and per-response
judgements; judged items are resumed on restart.
"""

import argparse
import os

from safemoe.data import expand_inputs, load_json, save_json
from safemoe.judge import JUDGE_MODELS, Judge, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--judge", choices=JUDGE_MODELS, default="gpt4o")
    p.add_argument("--model_name", default=None, help="override the judge model id")
    args = p.parse_args()

    files = expand_inputs(args.inputs)
    judge = Judge(args.judge, args.model_name)
    print(f"judge: {judge.model_name}")
    for path in files:
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        previous = load_json(out_path)["results"] if os.path.exists(out_path) else []
        done = []
        for i, item in enumerate(load_json(path)):
            # Reuse earlier judgements; failed ones (no score) are retried.
            if i < len(previous) and previous[i]["score"] is not None:
                done.append(previous[i])
            else:
                done.append({**item, **judge.evaluate(item["instruction"], item["response"])})
            if (i + 1) % 20 == 0:
                save_json({"input": path, "judge": judge.model_name, "summary": summarize(done), "results": done}, out_path)
        summary = summarize(done)
        save_json({"input": path, "judge": judge.model_name, "summary": summary, "results": done}, out_path)
        failed = summary["total"] - summary["valid"]
        print(f"{os.path.basename(path)}: safe {summary['safety_percentage']}%  info {summary['informativeness']}"
              + (f"  ({failed} judgements failed, rerun to retry)" if failed else ""))
        if summary["valid"] == 0 and summary["total"]:
            raise SystemExit(f"every judgement failed, e.g. {done[0]['justification']}")


if __name__ == "__main__":
    main()
