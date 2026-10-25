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

    table = {}  # category -> {"safe": [per seed], "info": [per seed]}
    for run in args.runs:
        for path in sorted(glob.glob(os.path.join(run, "*.json"))):
            judged = load_json(path)
            summary = judged["summary"]
            results = judged["results"]
            category = results[0]["category"] if results else os.path.splitext(os.path.basename(path))[0]
            row = table.setdefault(category, {"safe": [], "info": []})
            row["safe"].append(summary["safety_percentage"])
            row["info"].append(summary["informativeness"])

    rows = []
    for category, v in sorted(table.items()):
        (s, s_std), (i, i_std) = mean_std(v["safe"]), mean_std(v["info"])
        rows.append([category, len(v["safe"]), f"{s:.1f}", f"{s_std:.1f}", f"{i:.2f}", f"{i_std:.2f}"])
    # Average row: category average of each seed, then mean ± std over seeds.
    n_seeds = min(len(v["safe"]) for v in table.values())
    safe_avg = [statistics.mean(v["safe"][k] for v in table.values()) for k in range(n_seeds)]
    info_avg = [statistics.mean(v["info"][k] for v in table.values()) for k in range(n_seeds)]
    (s, s_std), (i, i_std) = mean_std(safe_avg), mean_std(info_avg)
    rows.append(["average", len(args.runs), f"{s:.1f}", f"{s_std:.1f}", f"{i:.2f}", f"{i_std:.2f}"])

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
