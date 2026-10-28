"""Claim-masked factuality F_masked of safe responses (Appendix Q), one GPU.

    python scripts/eval/factuality/run_factuality.py \
        --inputs 'results/judgements/claude/safemoe_xl_seed0/*.json' \
        --output_dir results/factuality/claude/safemoe_xl_seed0

Inputs are scripts/judge.py outputs; every response with is_safe == true is
scored. Each response is split into sentences, evidence for its prompt is
retrieved from the category's knowledge source plus Wikipedia (retrieval.py),
and every claim sentence (claims.py, zero-shot claim mask) is scored by its
best DeBERTa-v3-large NLI entailment over the evidence (scoring.py).

The knowledge source is chosen from each record's "category" (PKU-SafeRLHF
key or paper category name, as written by scripts/generate.py): Endangering
Public Health / Psychological Harm / Physical Harm -> PubMedQA + MedMCQA,
Drug Abuse & Weapons (PKU "Drugs") -> Semantic Scholar, Cybercrime -> MITRE
ATT&CK. Responses of other categories are skipped unless --domain forces a
source, so a whole judgement directory can be passed.

Each input with scored responses gives <output_dir>/<input name>.json with a
summary (mean F_masked) and per-response scores; scored items are resumed on
restart. Wikipedia / Semantic Scholar are queried live, so the node needs
internet access.
"""

import argparse
import os

from safemoe.data import expand_inputs, load_json, save_json

from claims import CLAIM_MODEL, ClaimFilter, apply_positional_mask
from retrieval import RETRIEVER_MODEL, Cache, DenseReranker, EvidenceRetriever, domain_of
from scoring import NLI_MODEL, NLIScorer, dedupe_sentences, score_record, split_sentences, summarize


def score_response(item, domain, retriever, nli, claim_filter, top_k):
    raw = split_sentences(item["response"])
    sentences = dedupe_sentences(raw)
    is_claim = apply_positional_mask(claim_filter.classify_claims(sentences))
    evidence = retriever.get_evidence(item["instruction"], domain, top_k)
    record = {
        "instruction": item["instruction"],
        "response": item["response"],
        "category": item.get("category"),
        "evidence_domain": domain,
        "judge_score": item.get("score"),
        "num_sentences": len(sentences),
        "num_duplicate_sentences_removed": len(raw) - len(sentences),
        "evidence": [{"text": e["text"], "source": e["source"]} for e in evidence],
        "sentence_scores": nli.score_sentences(sentences, evidence, is_claim) if sentences else [],
    }
    return score_record(record)


def safe_items(results, domain=None):
    """(record, knowledge source) of every safe response whose category has a source."""
    items = [(r, domain or domain_of(r.get("category"))) for r in results if r.get("is_safe") is True]
    return [(r, d) for r, d in items if d is not None]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="judge output files (globs allowed)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=None, help="safe responses scored per file (all if unset)")
    p.add_argument("--top_k", type=int, default=20, help="evidence passages per prompt")
    p.add_argument("--domain", choices=["medical", "s2orc", "mitre"], default=None,
                   help="force the knowledge source instead of inferring it from the category")
    p.add_argument("--cache_dir", default=os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                                                        "safemoe_factuality"),
                   help="corpora and API answers")
    p.add_argument("--retriever_model", default=RETRIEVER_MODEL)
    p.add_argument("--nli_model", default=NLI_MODEL)
    p.add_argument("--claim_model", default=CLAIM_MODEL)
    p.add_argument("--save_every", type=int, default=10)
    args = p.parse_args()

    retriever = EvidenceRetriever(Cache(args.cache_dir), DenseReranker(args.retriever_model))
    nli = NLIScorer(args.nli_model)
    claim_filter = ClaimFilter(args.claim_model)

    for path in expand_inputs(args.inputs):
        results = load_json(path)["results"]
        safe = safe_items(results, args.domain)[: args.max_samples]
        if not safe:
            print(f"skip {os.path.basename(path)}: no safe response of a category with a knowledge source")
            continue
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        done = load_json(out_path)["results"] if os.path.exists(out_path) else []

        def save():
            save_json({
                "input": path,
                "evidence_sources": sorted({d for _, d in safe}) + ["wikipedia"],
                "retriever_model": args.retriever_model,
                "nli_model": args.nli_model,
                "claim_model": args.claim_model,
                "top_k": args.top_k,
                "num_judged": len(results),
                "num_safe": sum(r.get("is_safe") is True for r in results),
                "summary": summarize(done),
                "results": done,
            }, out_path)

        previous, done = done, []
        for i, (item, domain) in enumerate(safe):
            if i < len(previous) and "error" not in previous[i]:
                done.append(previous[i])
                continue
            try:
                done.append(score_response(item, domain, retriever, nli, claim_filter, args.top_k))
            except Exception as e:  # keep long runs alive; the item is recorded as unscored
                print(f"{os.path.basename(path)} item {i} failed: {e}")
                done.append({"instruction": item["instruction"], "response": item["response"],
                             "category": item.get("category"), "evidence_domain": domain,
                             "judge_score": item.get("score"), "error": str(e), "num_sentences": 0,
                             "num_claim_sentences": 0, "sentence_scores": [], "evidence": [],
                             "factuality_score_masked": None})
            if (i + 1) % args.save_every == 0:
                save()
        save()
        summary = summarize(done)
        print(f"{os.path.basename(path)}: F_masked {summary['mean_factuality']} over "
              f"{summary['scored_responses']}/{len(done)} safe responses")


if __name__ == "__main__":
    main()
