"""Evidence retrieval for the factuality metric (Appendix Q).

For each prompt, candidates come from the category's knowledge source plus
Wikipedia, and the union is reranked to the top-k passages with Contriever.
The five evaluated categories and their sources (CATEGORY_DOMAINS):

    Endangering Public Health, Psychological Harm, Physical Harm   medical
    Drug Abuse & Weapons                                            s2orc
    Cybercrime                                                      mitre

with

    medical  PubMedQA (pqa_labeled + pqa_artificial) and MedMCQA
             explanations, BM25 over a local index
    s2orc    Semantic Scholar Graph paper-search API (titles + abstracts),
             a live stand-in for a local S2ORC mirror
    mitre    MITRE ATT&CK enterprise techniques (STIX bundle), BM25
    all      Wikipedia search + intro extracts (live API)

API answers and the built corpora are cached under cache_dir; the cache
layout matches the original runs, so copying their cache reproduces the
exact evidence pools.
"""

import hashlib
import json
import os
import time

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

from safemoe.constants import pku_key

RETRIEVER_MODEL = "facebook/contriever-msmarco"
CANDIDATES_PER_SOURCE = 25
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
MITRE_ATTACK_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
# Wikimedia asks API clients to identify themselves; set a contact via the env.
USER_AGENT = os.environ.get("SAFEMOE_USER_AGENT", "SafeMoE-FactualityEval/1.0 (research use)")

# PKU-SafeRLHF category key -> knowledge source unioned with Wikipedia.
CATEGORY_DOMAINS = {
    "Endangering Public Health": "medical",
    "Psychological Harm": "medical",
    "Physical Harm": "medical",
    "Drugs": "s2orc",
    "Cybercrime": "mitre",
}


def domain_of(category):
    """Knowledge source of a category (paper name, PKU key or slug); None if it has none."""
    try:
        return CATEGORY_DOMAINS.get(pku_key(str(category)))
    except KeyError:
        return None


class Cache:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, namespace, key):
        d = os.path.join(self.root, namespace)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")

    def get(self, namespace, key):
        p = self.path(namespace, key)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    def put(self, namespace, key, value):
        with open(self.path(namespace, key), "w") as f:
            json.dump(value, f)


class RateLimiter:
    def __init__(self, min_interval_s):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self):
        elapsed = time.time() - self._last
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last = time.time()


def _session(retry=True):
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    if retry:
        adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504]))
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    return s


def _dedupe(passages):
    seen, out = set(), []
    for p in passages:
        key = p["text"][:200]
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


class DenseReranker:
    """Contriever: mean-pooled embeddings, cosine similarity to the query."""

    def __init__(self, model_name=RETRIEVER_MODEL, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, texts, batch_size=32):
        vecs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i: i + batch_size], padding=True, truncation=True,
                                     max_length=256, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vecs.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy())
        return np.concatenate(vecs, axis=0)

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        cand = self.embed([c["text"] for c in candidates])
        q = self.embed([query])[0]
        sims = (cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)) @ (q / (np.linalg.norm(q) + 1e-9))
        return [{**candidates[i], "retrieval_score": float(sims[i])} for i in np.argsort(-sims)[:top_k]]


class WikipediaRetriever:
    def __init__(self, cache, limit=CANDIDATES_PER_SOURCE):
        self.cache = cache
        self.limit = limit
        self.session = _session()
        self.rate_limiter = RateLimiter(0.2)

    def _get(self, **params):
        self.rate_limiter.wait()
        resp = self.session.get(WIKIPEDIA_API, params={**params, "format": "json"}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("query", {})

    def search(self, query):
        key = f"search::{query}::{self.limit}"
        cached = self.cache.get("wikipedia", key)
        if cached is not None:
            return cached
        passages = []
        try:
            hits = self._get(action="query", list="search", srsearch=query, srlimit=self.limit).get("search", [])
            if hits:
                pages = self._get(action="query", prop="extracts", exintro=1, explaintext=1,
                                  pageids="|".join(str(h["pageid"]) for h in hits)).get("pages", {})
                for pid, page in pages.items():
                    extract = (page.get("extract") or "").strip()
                    if extract:
                        passages.append({"text": extract[:2000], "source": f"wikipedia:{page.get('title', pid)}"})
        except requests.RequestException as e:
            print(f"[wikipedia] query failed for {query!r}: {e}")
            raise
        self.cache.put("wikipedia", key, passages)
        return passages


class S2ORCRetriever:
    """Semantic Scholar paper search. 429s get a short bounded backoff and only
    successful answers are cached, so a later run retries the failures.
    SEMANTIC_SCHOLAR_API_KEY (optional) lifts the anonymous rate limit."""

    def __init__(self, cache, limit=CANDIDATES_PER_SOURCE):
        self.cache = cache
        self.limit = limit
        self.session = _session(retry=False)
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self.rate_limiter = RateLimiter(1.0 if api_key else 6.0)

    def search(self, query, max_attempts=4):
        key = f"{query}::{self.limit}"
        cached = self.cache.get("s2orc", key)
        if cached is not None:
            return cached
        q = query[:300]
        for attempt in range(max_attempts):
            self.rate_limiter.wait()
            try:
                resp = self.session.get(SEMANTIC_SCHOLAR_API, timeout=20,
                                        params={"query": q, "limit": self.limit, "fields": "title,abstract"})
            except requests.RequestException as e:
                print(f"[s2orc] query failed for {q!r}: {e}")
                return []
            if resp.status_code == 200:
                passages = [{"text": f"{p.get('title') or ''}. {p['abstract']}", "source": f"S2ORC:{(p.get('title') or '')[:80]}"}
                            for p in resp.json().get("data", []) if p.get("abstract")]
                self.cache.put("s2orc", key, passages)
                return passages
            if resp.status_code == 429 and attempt < max_attempts - 1:
                time.sleep(6 * (attempt + 1))
                continue
            print(f"[s2orc] HTTP {resp.status_code} for {q!r}")
            return []
        return []


class BM25Retriever:
    def __init__(self, passages, limit=CANDIDATES_PER_SOURCE):
        from rank_bm25 import BM25Okapi

        self.passages = passages
        self.limit = limit
        self.bm25 = BM25Okapi([p["text"].lower().split() for p in passages])

    def search(self, query):
        scores = self.bm25.get_scores(query.lower().split())
        return [dict(self.passages[i]) for i in np.argsort(-scores)[: self.limit] if scores[i] > 0]


def _cached_corpus(cache, filename, build):
    path = os.path.join(cache.root, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    passages = build()
    with open(path, "w") as f:
        json.dump(passages, f)
    return passages


def build_medical_passages():
    from datasets import load_dataset

    def contexts(row):
        return (row.get("context") or {}).get("contexts", [])

    passages = []
    for row in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"):
        passages += [{"text": c, "source": "PubMedQA"} for c in contexts(row) if c and len(c) > 40]
        if row.get("long_answer"):
            passages.append({"text": row["long_answer"], "source": "PubMedQA:long_answer"})
    for row in load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"):
        passages += [{"text": c, "source": "PubMedQA"} for c in contexts(row) if c and len(c) > 40]
    for row in load_dataset("openlifescienceai/medmcqa", split="train"):
        exp = (row.get("exp") or "").strip()
        if len(exp) > 40:
            passages.append({"text": exp, "source": "MedMCQA:explanation"})
    return _dedupe(passages)


def build_mitre_passages():
    resp = _session().get(MITRE_ATTACK_URL, timeout=60)
    resp.raise_for_status()
    passages = []
    for obj in resp.json().get("objects", []):
        if obj.get("type") != "attack-pattern" or obj.get("revoked") or not obj.get("description"):
            continue
        ids = [r["external_id"] for r in obj.get("external_references", [])
               if r.get("source_name") == "mitre-attack" and "external_id" in r]
        tid = ids[0] if ids else ""
        passages.append({"text": f"{obj.get('name', '')} ({tid}): {obj['description']}"[:2000],
                         "source": f"MITRE-ATT&CK:{tid or obj.get('name', '')}"})
    return passages


class EvidenceRetriever:
    """Domain source + Wikipedia, deduplicated, reranked to top_k by Contriever."""

    def __init__(self, cache, reranker):
        self.cache = cache
        self.reranker = reranker
        self.wikipedia = WikipediaRetriever(cache)
        self._domains = {}

    def domain(self, name):
        if name not in self._domains:
            if name == "medical":
                self._domains[name] = BM25Retriever(_cached_corpus(self.cache, "medical_passages.json", build_medical_passages))
            elif name == "mitre":
                self._domains[name] = BM25Retriever(_cached_corpus(self.cache, "mitre_attack_passages.json", build_mitre_passages))
            elif name == "s2orc":
                self._domains[name] = S2ORCRetriever(self.cache)
            else:
                raise ValueError(f"unknown evidence domain {name!r}")
        return self._domains[name]

    def get_evidence(self, query, domain, top_k):
        pool = self.domain(domain).search(query) + self.wikipedia.search(query)
        return self.reranker.rerank(query, _dedupe(pool), top_k)
