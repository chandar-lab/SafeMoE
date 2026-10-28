"""Agreement between human annotators and the GPT-4o judge (Appendix M).

    python scripts/analysis/human_agreement.py \
        --judge results/human_eval/sampled_balanced_gpt4o.json \
        --annotators a1=results/human_eval/annotator_1.csv a2=results/human_eval/annotator_2.pdf ... \
        --output_dir results/tables/human_agreement

--judge is the judged sample shown to the annotators (a judgement file or a
list of {instruction, response, is_safe, score}); item i of the annotation
sheets is its i-th record (1-based). Annotations are CSVs with columns
Index, Safe (1 safe / 0 unsafe), Informativeness (1-10, empty if unsafe), or
the filled annotation PDFs (one item per page, form fields safe_*, unsafe_*,
rating_*), from which such a CSV is extracted.

Safety (items labelled by every annotator):
  disagreement rate conditional on the judge label,
      P(annotator unsafe | judge safe) and P(annotator safe | judge unsafe),
  overall agreement, pairwise agreement and Fleiss' kappa between annotators.
Informativeness (judge-safe items the annotator scored):
  mean paired difference (annotator - judge), two-sided Wilcoxon signed-rank
  test, two-sample Kolmogorov-Smirnov test, and Pearson r on items both call safe.
"""

import argparse
import csv
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, wilcoxon

from safemoe.data import load_json


def load_judge(path):
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get("results") or data.get("detailed_results") or []
    rows = []
    for i, r in enumerate(data, start=1):
        score = r["score"] if "score" in r else r.get("rating_score")
        rows.append({"Index": i, "judge_safe": None if r["is_safe"] is None else int(r["is_safe"]), "judge_score": score})
    return pd.DataFrame(rows).dropna(subset=["judge_safe"])


def _pdf_value(v):
    if isinstance(v, bytes):
        if v[:2] in (b"\xfe\xff", b"\xff\xfe"):
            return v.decode("utf-16").strip() or None
        return v.decode("ascii", errors="ignore").strip() or None
    if isinstance(v, str):
        return v.strip() or None
    return None


def extract_pdf(path):
    """(Index, Safe, Informativeness) from a filled annotation PDF, one item per page."""
    import pdfplumber

    rows = []
    with pdfplumber.open(path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            safe = info = None
            for ann in page.annots or []:
                title = (ann.get("title") or "").strip()
                data = ann.get("data", {})
                checked = str(data.get("AS", "")) == "/'Yes'"
                if title.startswith("safe_") and checked:
                    safe = 1
                elif title.startswith("unsafe_") and checked:
                    safe = 0
                elif title.startswith("rating_"):
                    info = _pdf_value(data.get("V"))
            rows.append({"Index": page_idx, "Safe": safe, "Informativeness": None if safe == 0 else info})
    return pd.DataFrame(rows)


def load_annotations(path):
    if path.endswith(".pdf"):
        df = extract_pdf(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
    df["Index"] = pd.to_numeric(df["Index"], errors="coerce")
    df["Safe"] = pd.to_numeric(df["Safe"], errors="coerce")
    df["Informativeness"] = pd.to_numeric(df["Informativeness"], errors="coerce")
    return df.dropna(subset=["Index"]).astype({"Index": int})[["Index", "Safe", "Informativeness"]]


def fleiss_kappa(labels):
    """labels: (n_items, n_raters) array of category ids."""
    categories = np.unique(labels)
    counts = np.stack([(labels == c).sum(axis=1) for c in categories], axis=1)
    n_items, n_raters = labels.shape
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_e = (p_j ** 2).sum()
    return 1.0 if p_e == 1.0 else (p_i.mean() - p_e) / (1.0 - p_e)


def safety_rows(judge, annotations):
    merged = judge
    for name, df in annotations.items():
        merged = merged.merge(df[["Index", "Safe"]].dropna().rename(columns={"Safe": name}), on="Index")
    rows = []
    for name in annotations:
        js, ann = merged["judge_safe"], merged[name]
        rows.append({
            "annotator": name,
            "n": len(merged),
            "n_judge_safe": int((js == 1).sum()),
            "disagree_given_judge_safe": float((ann[js == 1] == 0).mean()),
            "n_judge_unsafe": int((js == 0).sum()),
            "disagree_given_judge_unsafe": float((ann[js == 0] == 1).mean()),
            "agreement": float((ann == js).mean()),
        })
    names = list(annotations)
    pairwise = {f"{a}-{b}": float((merged[a] == merged[b]).mean()) for a, b in itertools.combinations(names, 2)}
    kappa = fleiss_kappa(merged[names].to_numpy().astype(int)) if len(names) > 1 else float("nan")
    return rows, pairwise, kappa


def informativeness_rows(judge, annotations):
    judge_safe = judge[judge["judge_safe"] == 1]
    rows = []
    for name, df in annotations.items():
        scored = judge_safe.merge(df.dropna(subset=["Informativeness"]), on="Index")
        g, a = scored["judge_score"].to_numpy(float), scored["Informativeness"].to_numpy(float)
        w_stat, w_p = wilcoxon(g, a, alternative="two-sided") if (g != a).any() else (float("nan"), float("nan"))
        ks_stat, ks_p = ks_2samp(g, a)
        both_safe = scored[scored["Safe"] == 1]
        r, r_p = (pearsonr(both_safe["judge_score"], both_safe["Informativeness"])
                  if len(both_safe) > 1 else (float("nan"), float("nan")))
        rows.append({
            "annotator": name,
            "n": len(scored),
            "judge_mean": g.mean(),
            "annotator_mean": a.mean(),
            "mean_diff": a.mean() - g.mean(),
            "wilcoxon_W": w_stat,
            "wilcoxon_p": w_p,
            "ks": ks_stat,
            "ks_p": ks_p,
            "n_both_safe": len(both_safe),
            "pearson_r": r,
            "pearson_p": r_p,
        })
    return rows


def print_table(title, rows):
    print(f"\n{title}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge", required=True, help="judged sample shown to the annotators")
    p.add_argument("--annotators", nargs="+", required=True, metavar="NAME=PATH", help="annotation CSV or PDF")
    p.add_argument("--output_dir", default=None, help="write CSV tables (and extracted annotations) here")
    args = p.parse_args()

    judge = load_judge(args.judge)
    annotations = {}
    for spec in args.annotators:
        name, path = spec.split("=", 1)
        annotations[name] = load_annotations(path)

    safety, pairwise, kappa = safety_rows(judge, annotations)
    info = informativeness_rows(judge, annotations)
    print_table("Safety labels vs judge", safety)
    print("\nPairwise annotator agreement: " + ", ".join(f"{k} {v:.3f}" for k, v in pairwise.items()))
    print(f"Fleiss' kappa ({len(annotations)} annotators): {kappa:.4f}")
    print_table("Informativeness (judge-safe items)", info)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(safety).to_csv(os.path.join(args.output_dir, "safety_agreement.csv"), index=False)
        pd.DataFrame(info).to_csv(os.path.join(args.output_dir, "informativeness.csv"), index=False)
        with open(os.path.join(args.output_dir, "annotator_agreement.csv"), "w", newline="") as f:
            csv.writer(f).writerows([["pair", "agreement"], *pairwise.items(), ["fleiss_kappa", kappa]])
        for name, df in annotations.items():
            df.to_csv(os.path.join(args.output_dir, f"annotations_{name}.csv"), index=False)


if __name__ == "__main__":
    main()
