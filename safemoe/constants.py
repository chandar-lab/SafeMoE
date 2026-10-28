"""Shared constants: harm categories, expert sets and prompt templates."""

import re

# Harm categories of the paper (Table 7) -> key of the category in PKU-SafeRLHF.
CATEGORIES = {
    "Animal Abuse": "Animal Abuse",
    "Copyright Issues": "Copyright Issues",
    "Cybercrime": "Cybercrime",
    "Discriminatory Behavior": "Discriminatory Behavior",
    "Disrupting Public Order": "Disrupting Public Order",
    "Drug Abuse & Weapons": "Drugs",
    "Economic Crime": "Economic Crime",
    "Endangering National Security": "Endangering National Security",
    "Endangering Public Health": "Endangering Public Health",
    "Environmental Harm": "Environmental Damage",
    "Human Trafficking": "Human Trafficking",
    "Insulting Behavior": "Insulting Behavior",
    "Mental Manipulation": "Mental Manipulation",
    "Physical Harm": "Physical Harm",
    "Privacy Violation": "Privacy Violation",
    "Psychological Harm": "Psychological Harm",
    "Sexual Content": "Sexual Content",
    "Violence": "Violence",
    "White Collar Crime": "White-Collar Crime",
}


def slug(name):
    """'Drug Abuse & Weapons' -> 'drug_abuse_weapons' (expert directory / file name)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pku_key(name):
    """Map a paper category name, a PKU key or a slug to the PKU-SafeRLHF key."""
    for paper, key in CATEGORIES.items():
        if name in (paper, key, slug(paper), slug(key)):
            return key
    raise KeyError(f"unknown harm category {name!r}; known: {list(CATEGORIES)}")


# Categories with GPT-4 safe-informative responses used to train the routers (Section 4.1).
SAFE_DATA_CATEGORIES = ["Drug Abuse & Weapons", "Psychological Harm", "Cybercrime", "Economic Crime"]

# Expert libraries (Table 7). The order fixes the expert index inside the MoE
# layer, so keep it stable once a router has been trained.
_L_EXTRA = ["Environmental Harm", "Copyright Issues", "Mental Manipulation",
            "Endangering Public Health", "Violence", "Physical Harm"]
_XL_EXTRA = ["Discriminatory Behavior", "Privacy Violation", "Insulting Behavior", "Human Trafficking",
             "Endangering National Security", "Disrupting Public Order", "Sexual Content", "White Collar Crime"]
EXPERT_SETS = {
    "4": SAFE_DATA_CATEGORIES,
    "L": SAFE_DATA_CATEGORIES + _L_EXTRA,
    "XL": SAFE_DATA_CATEGORIES + _L_EXTRA + _XL_EXTRA,
}


def expert_names(experts):
    """An EXPERT_SETS key ("4", "L", "XL"; 4 given as an int also works) or a list of names."""
    if isinstance(experts, (str, int)):
        return EXPERT_SETS[str(experts)]
    return list(experts)


KNOWLEDGE_EXPERTS = ["pubmedqa", "cybersecurity_qa", "fiqa", "mental_health"]

# Prompt format. Experts, routers and generation all use the same
# "### Question / ### Answer" layout on the raw (non-chat) base model.
TRAIN_TEMPLATE = "### Question: {prompt}\n### Answer: {response}"
GENERATION_TEMPLATE = "### Question: {prompt}\n### Answer:"
STOP_STRING = "\n###"
