"""Score AlpacaEval responses: length-controlled win rate (Section 4.2.5; CPU, API calls) or ROUGE-L.

    python scripts/eval/utility/alpaca_eval_score.py \
        --responses results/utility/alpaca_eval/safemoe_xl/responses.json

lc_win_rate (default) runs alpaca_eval (==0.6.6) against its default AlpacaEval
2.0 reference (gpt4_1106_preview outputs) with the annotator config in
alpaca_eval_configs/ next to this script: alpaca_eval_gpt4_turbo_fn, i.e. the
stock config with the judge switched to gpt-4-turbo (gpt-4-1106-preview is
retired). --annotators_config weighted_alpaca_eval_gpt4_turbo_new selects
the logprob-weighted variant. OpenAI has retired gpt-4-turbo; --judge_model
swaps in a current model (e.g. gpt-4o). Needs OPENAI_API_KEY. Writes leaderboard.csv
and annotations.json to <responses dir>/alpaca_eval_results/<annotator>/.

rougeL reports the mean ROUGE-L F1 between each output and the AlpacaEval 1
reference output stored by alpaca_eval_generate.py (no API calls).
"""

import argparse
import os
import re
import shutil
import tempfile
from collections import defaultdict

from safemoe.data import load_json, save_json

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_eval_configs")


def with_judge_model(config, judge_model):
    """Copy the annotator configs to a temporary dir with the judge model replaced."""
    root = tempfile.mkdtemp(prefix="alpaca_eval_configs_")
    shutil.copytree(os.path.dirname(os.path.dirname(config)), root, dirs_exist_ok=True)
    copied = os.path.join(root, os.path.relpath(config, os.path.dirname(os.path.dirname(config))))
    with open(copied) as f:
        text = f.read()
    with open(copied, "w") as f:
        f.write(re.sub(r'model_name: "[^"]+"', f'model_name: "{judge_model}"', text))
    return copied


def lc_win_rate(responses, annotators_config, output_path, judge_model=None):
    from alpaca_eval import evaluate

    config = annotators_config
    if not os.path.exists(config):
        config = os.path.join(CONFIG_DIR, annotators_config, "configs.yaml")
    if judge_model:
        config = with_judge_model(config, judge_model)
    model_outputs = [{k: r[k] for k in ("instruction", "output", "generator")} for r in responses]
    leaderboard, _ = evaluate(
        model_outputs=model_outputs,
        annotators_config=config,
        output_path=output_path,
        precomputed_leaderboard=None,
        is_cache_leaderboard=False,
        is_return_instead_of_print=True,
    )
    print(leaderboard[["length_controlled_winrate", "lc_standard_error", "win_rate", "standard_error",
                       "n_total", "avg_length"]].to_string())


def rouge_l(responses, output_path):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results, skipped = [], 0
    for r in responses:
        output, reference = r.get("output", "").strip(), r.get("reference", "").strip()
        if not reference:
            skipped += 1
            continue
        results.append({**r, "rougeL": round(scorer.score(reference, output)["rougeL"].fmeasure, 4)})
    per_dataset = defaultdict(list)
    for r in results:
        per_dataset[r.get("dataset", "unknown")].append(r["rougeL"])
    mean = sum(r["rougeL"] for r in results) / len(results) if results else 0.0
    summary = {
        "total": len(results),
        "skipped": skipped,
        "avg_rougeL": round(mean, 4),
        "per_dataset": {d: {"count": len(v), "avg_rougeL": round(sum(v) / len(v), 4)} for d, v in sorted(per_dataset.items())},
        "per_example": results,
    }
    save_json(summary, os.path.join(output_path, "rougeL_scores.json"))
    print(f"ROUGE-L {mean:.4f} ({len(results)} samples, {skipped} skipped)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses", required=True, help="alpaca_eval_generate.py output")
    p.add_argument("--metric", choices=["lc_win_rate", "rougeL"], default="lc_win_rate")
    p.add_argument("--annotators_config", default="alpaca_eval_gpt4_turbo_fn",
                   help="name of a config in alpaca_eval_configs/ or a path to a configs.yaml")
    p.add_argument("--judge_model", default=None, help="replace the annotator's OpenAI model (e.g. gpt-4o)")
    p.add_argument("--output_dir", default=None, help="default: <responses dir>/alpaca_eval_results")
    args = p.parse_args()

    responses = load_json(args.responses)
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.responses)), "alpaca_eval_results")
    if args.metric == "rougeL":
        rouge_l(responses, output_dir)
    else:
        name = os.path.basename(os.path.dirname(args.annotators_config)) if args.annotators_config.endswith(".yaml") \
            else os.path.basename(args.annotators_config.rstrip("/"))
        lc_win_rate(responses, args.annotators_config, os.path.join(output_dir, name), args.judge_model)


if __name__ == "__main__":
    main()
