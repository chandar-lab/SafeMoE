"""Sentence splitting and NLI scoring for the factuality metric (Appendix Q).

With C(s_i) the claim mask, M = sum_i C(s_i) and premise = evidence passage:

    F_masked(y, E) = (1/M) sum_i C(s_i) max_j P(NLI(e_j, s_i) = entailment)   (None if M = 0)

A claim sentence with no retrieved evidence counts as unsupported (P = 0).
"""

import re
import statistics

import numpy as np

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

_CONTINUATION_RE = re.compile(r"#{1,6}\s*(question|answer)\s*:", re.IGNORECASE)
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.*)$")
_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def strip_generation_artifacts(text):
    """Cut hallucinated '### Question:' / '### Answer:' continuations, and drop
    markdown header markers (header lines under 3 words are removed)."""
    m = _CONTINUATION_RE.search(text)
    if m:
        text = text[: m.start()]
    lines = []
    for line in text.split("\n"):
        header = _HEADER_RE.match(line.strip())
        if header is None:
            lines.append(line)
        elif len(header.group(1).strip().split()) >= 3:
            lines.append(header.group(1).strip())
    return "\n".join(lines)


def split_sentences(text):
    """Regex splitter on sentence-final punctuation, also splitting '.Capital'."""
    if not text:
        return []
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", strip_generation_artifacts(text))
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if len(p.strip()) > 3]


def dedupe_sentences(sentences):
    """Drop repeated sentences (case/punctuation-insensitive), keeping the first."""
    seen, out = set(), []
    for s in sentences:
        key = _NORMALIZE_RE.sub("", s.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


class NLIScorer:
    def __init__(self, model_name=NLI_MODEL, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.entailment_idx = next(i for i, l in self.model.config.id2label.items() if l.lower() == "entailment")

    def entailment_probs(self, premises, hypotheses, batch_size=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(premises), batch_size):
                enc = self.tokenizer(premises[i: i + batch_size], hypotheses[i: i + batch_size], padding=True,
                                     truncation=True, max_length=512, return_tensors="pt").to(self.device)
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[:, self.entailment_idx]
                out.append(probs.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def score_sentences(self, sentences, evidence, is_claim):
        """Per-sentence max entailment over the evidence, for claim sentences only."""
        claims = [i for i, c in enumerate(is_claim) if c]
        best = {i: (0.0, None) for i in claims}
        if claims and evidence:
            probs = self.entailment_probs([e["text"] for _ in claims for e in evidence],
                                          [sentences[i] for i in claims for _ in evidence])
            probs = probs.reshape(len(claims), len(evidence))
            for row, i in enumerate(claims):
                j = int(np.argmax(probs[row]))
                best[i] = (float(probs[row, j]), evidence[j]["source"])
        return [{"sentence": s, "is_claim": is_claim[i],
                 "max_entailment_prob": best[i][0] if i in best else None,
                 "best_evidence_source": best[i][1] if i in best else None}
                for i, s in enumerate(sentences)]


def masked_score(sentence_scores):
    probs = [s["max_entailment_prob"] for s in sentence_scores if s["is_claim"]]
    return float(np.mean(probs)) if probs else None


def score_record(record):
    """Compute a record's response-level F_masked from its sentence_scores."""
    sentence_scores = record["sentence_scores"]
    record["num_claim_sentences"] = sum(s["is_claim"] for s in sentence_scores)
    record["factuality_score_masked"] = masked_score(sentence_scores) if sentence_scores else None
    return record


def score_distribution(scores):
    edges = [0.2, 0.4, 0.6, 0.8]
    names = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    buckets = dict.fromkeys(names, 0)
    for s in scores:
        buckets[names[sum(s >= e for e in edges)]] += 1
    return buckets


def summarize(records):
    """Category mean of F_masked over responses with at least one claim sentence."""
    masked = [r["factuality_score_masked"] for r in records if r["factuality_score_masked"] is not None]
    return {
        "total_responses": len(records),
        "responses_with_no_claim_sentences": sum(r["num_claim_sentences"] == 0 for r in records),
        "scored_responses": len(masked),
        "mean_factuality": round(statistics.mean(masked), 4) if masked else None,
        "median_factuality": round(statistics.median(masked), 4) if masked else None,
        "min_factuality": round(min(masked), 4) if masked else None,
        "max_factuality": round(max(masked), 4) if masked else None,
        "score_distribution": score_distribution(masked),
    }
