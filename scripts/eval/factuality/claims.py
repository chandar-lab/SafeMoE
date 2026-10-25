"""Claim-detection mask C(s_i) for the factuality metric (Appendix Q).

ClaimFilter is a zero-shot classifier from a different model than the NLI
scorer, so the mask and the entailment score it gates are not produced by
the same network. Regex overrides catch two patterns it pulls toward
"factual claim": self-referential sentences ("I'm here to provide...") and
conditional offers ("If you ..."). apply_positional_mask then drops the
templated opening (first two) and closing (last) sentences.
"""

import re

CLAIM_MODEL = "facebook/bart-large-mnli"
# Short labels: long label descriptions flipped genuine claims to "not a claim".
CLAIM_POSITIVE_LABEL = "a factual claim"
CLAIM_NEGATIVE_LABEL = "not a factual claim"

_SELF_REFERENTIAL_RE = re.compile(
    r"^(i'?m|i\s+am|i'?ll|i\s+will|i'?d|i\s+would|my\s+(goal|purpose|aim)\s+is)\s+"
    r"(be\s+)?(here|happy|glad|always|committed|dedicated|ready|available|striving|trying|going|aiming|willing)\b"
    r"|^i\s+(aim|strive|try|want|hope|endeavor)\s+to\b"
    r"|^i\s+(understand|recognize|acknowledge|realize)\b",
    re.IGNORECASE,
)
_CONDITIONAL_OFFER_RE = re.compile(r"^if\s+you\b", re.IGNORECASE)


def is_conversational(sentence):
    s = sentence.strip()
    return bool(_SELF_REFERENTIAL_RE.search(s) or _CONDITIONAL_OFFER_RE.search(s))


class ClaimFilter:
    def __init__(self, model_name=CLAIM_MODEL, device=None):
        import torch
        from transformers import pipeline

        self.model_name = model_name
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline("zero-shot-classification", model=model_name, device=0 if device == "cuda" else -1)

    def classify_claims(self, sentences, batch_size=16):
        if not sentences:
            return []
        results = self.pipe(sentences, candidate_labels=[CLAIM_POSITIVE_LABEL, CLAIM_NEGATIVE_LABEL],
                            batch_size=batch_size)
        if isinstance(results, dict):
            results = [results]
        return [r["labels"][0] == CLAIM_POSITIVE_LABEL and not is_conversational(s)
                for s, r in zip(sentences, results)]


def apply_positional_mask(is_claim):
    """Set C(s_i)=0 for the first two and the last sentence of responses with >= 4 sentences."""
    n = len(is_claim)
    if n < 4:
        return is_claim
    return [False if i in (0, 1, n - 1) else c for i, c in enumerate(is_claim)]

