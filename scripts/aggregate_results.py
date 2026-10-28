"""Aggregate judge outputs into a per-category table (mean ± std over seeds), as in Table 1.

    python scripts/aggregate_results.py \
        --runs results/judgements/gpt4o/safemoe_xl_seed{0,1,2} --output results/tables/safemoe_xl.csv

Each run directory holds one judged file per category (scripts/judge.py).
"""

import argparse
import csv
import glob
import os
import statistics

from safemoe.data import load_json


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="one judgement directory per seed")
    p.add_argument("--output", default=None, help="CSV path (printed only if unset)")
    args = p.parse_args()

    runs = []
    for run in args.runs:
        results_by_category = {}
        for path in sorted(glob.glob(os.path.join(run, "*.json"))):
            judged = load_json(path)
            summary = judged["summary"]
            results = judged["results"]
            category = results[0]["category"] if results else os.path.splitext(os.path.basename(path))[0]
            if category in results_by_category:
                raise ValueError(f"duplicate category {category!r} in run {run!r}")
            results_by_category[category] = {
                "safe": summary["safety_percentage"],
                "info": summary["informativeness"],
            }
        runs.append(results_by_category)

    categories = set(runs[0])
    if not categories:
        raise ValueError("runs contain no category results")
    for run, results_by_category in zip(args.runs, runs):
        run_categories = set(results_by_category)
        if run_categories != categories:
            missing = sorted(categories - run_categories)
            extra = sorted(run_categories - categories)
            raise ValueError(
                f"run {run!r} has a different category set "
                f"(missing: {missing}, extra: {extra})"
            )

    table = {
        category: {
            "safe": [run[category]["safe"] for run in runs],
            "info": [run[category]["info"] for run in runs],
        }
        for category in categories
    }

    rows = []
    for category, v in sorted(table.items()):
        (s, s_std), (i, i_std) = mean_std(v["safe"]), mean_std(v["info"])
        rows.append([category, len(v["safe"]), f"{s:.1f}", f"{s_std:.1f}", f"{i:.2f}", f"{i_std:.2f}"])
    # Average row: category average of each seed, then mean ± std over seeds.
    safe_avg = [statistics.mean(run[category]["safe"] for category in categories) for run in runs]
    info_avg = [statistics.mean(run[category]["info"] for category in categories) for run in runs]
    (s, s_std), (i, i_std) = mean_std(safe_avg), mean_std(info_avg)
    rows.append(["average", len(runs), f"{s:.1f}", f"{s_std:.1f}", f"{i:.2f}", f"{i_std:.2f}"])

    header = ["category", "n_seeds", "safe", "safe_std", "info", "info_std"]
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerows([header, *rows])


if __name__ == "__main__":
    main()
