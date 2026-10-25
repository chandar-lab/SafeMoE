"""Removal of blanket-refusal sentences from GPT-4 safe responses."""

import re

REFUSAL_SENTENCES = [
    "I'm sorry, I can't assist with that.",
    "I'm sorry, I cannot assist with that.",
    "I'm sorry, but I can't assist you with that.",
    "I'm sorry, but I cannot assist you with that.",
    "I'm sorry, but I can't assist with that.",
    "I'm sorry, but I cannot assist with that.",
    "I'm sorry to hear that you're feeling this way, but I can't assist with that.",
    "I'm really sorry to hear that you're feeling this way",
    "I'm sorry, but I can't provide that information",
    "I'm sorry, but I cannot provide that information",
    "I'm sorry to hear that you're feeling this way, but",
    "Sorry, I can't assist with that.",
    "Sorry, I cannot assist with that.",
    "I can't help with that.",
    "I can't assist with that.",
    "I can't assist you with that.",
    "I cannot help with that.",
    "I cannot assist with that.",
    "I cannot assist you with that.",
    "I cannot provide assistance on this.",
    "I cannot provide that information.",
]

# Short refusals the fixed list misses ("I'm unable to assist with that", ...).
RESIDUAL_REFUSAL_RE = re.compile(
    r"^i(?:'m| am)?\s*(?:sorry,?\s*(?:but\s*)?)?"
    r"(?:i(?:'m| am)?\s*)?"
    r"(?:not able|unable|won't be able|can't|cannot|can not)\s*to\s*"
    r"(?:assist|help)(?:\s*you)?\s*with\s*that\.?$",
    re.IGNORECASE,
)


def clean_response(response):
    """Strip refusal sentences; return "" when nothing informative is left."""
    if not isinstance(response, str):
        return ""
    response = response.replace("’", "'")
    for refusal in REFUSAL_SENTENCES:
        response = response.replace(refusal, "")
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    cleaned = ".".join(sentences)
    return "" if RESIDUAL_REFUSAL_RE.match(cleaned.strip()) else cleaned
