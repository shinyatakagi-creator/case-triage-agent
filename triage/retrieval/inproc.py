"""In-process hybrid search over the exported rag-hybrid index (data/index/): BM25 + cosine kNN + RRF.

Mirrors rag-hybrid rag/retrieve.py: multi_match best_fields over title^2, heading_path^1.5, content (english);
kNN and BM25 lists of POOL chunks each, fused with RRF_K=60, top k chunks, chunker filter "heading" (the API default).
Returns the same dict shape as rag-hybrid's /search so server.normalise_search maps it.
"""
import json
import math
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import httpx
import numpy as np
import snowballstemmer

POOL = 200
RRF_K = 60
CHUNKER = "heading"
BOOST = {"title": 2.0, "heading_path": 1.5, "content": 1.0}
K1, B = 1.2, 0.75  # Lucene BM25 defaults
KEY_VAR = "TRIAGE_OPENAI_API_KEY"
EMBED_URL ="https://api.openai.com/v1/embeddings"
EMBED_USD_PER_TOKEN = 0.02 / 1e6  # text-embedding-3-small
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "data/index"))

# Lucene EnglishAnalyzer stop set
STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their then there "
           "these they this to was will with".split())
_stem = snowballstemmer.stemmer("porter").stemWord  # Lucene's PorterStemFilter is the original Porter algorithm
# ponytail: \w+ joined by ' or . approximates UAX#29 word breaks (no letter/digit rules on '.'); upgrade if parity shows token drift
_TOKEN = re.compile(r"\w+(?:['’.]\w+)*")


def tokens_standard(text: str) -> list[str]:
    """ES standard analyzer (title, heading_path): tokenise, lowercase."""
    return [t.lower() for t in _TOKEN.findall(text)]


def tokens_english(text: str) -> list[str]:
    """ES english analyzer (content): tokenise, strip possessive, lowercase, stop, Porter stem."""
    out = []
    for t in _TOKEN.findall(text):
        t = re.sub(r"['’][sS]$", "", t).lower()
        if t not in STOP:
            out.append(_stem(t))
    return out


ANALYZERS = {"title": tokens_standard, "heading_path": tokens_standard, "content": tokens_english}


def _norm_len(n: int) -> int:
    """Lucene stores field length as one lossy byte (SmallFloat int4); BM25 scores use the decoded value."""
    bits = n.bit_length()
    if bits < 4:
        return n
    shift = bits - 4
    enc = ((n >> shift) & 7) | ((shift + 1) << 3)
    s = (enc >> 3) - 1
    return enc & 7 if s == -1 else ((enc & 7) | 8) << s


class FieldIndex:
    def __init__(self, texts: list[str], analyze):
        self.n = len(texts)
        post: dict[str, tuple[list[int], list[int]]] = {}
        lens = np.zeros(self.n)
        total = 0
        for i, text in enumerate(texts):
            toks = analyze(text)
            total += len(toks)
            lens[i] = _norm_len(len(toks))
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                d, f = post.setdefault(t, ([], []))
                d.append(i)
                f.append(c)
        self.post = {t: (np.array(d), np.array(f, dtype=float)) for t, (d, f) in post.items()}
        self.avgdl = total / self.n  # Lucene: sumTotalTermFreq / docCount, exact
        self.dl = lens
        self.analyze = analyze

    def score(self, query: str) -> np.ndarray:
        s = np.zeros(self.n)
        for t in self.analyze(query):  # repeated query terms score once each, as in ES's bool of term clauses
            if t not in self.post:
                continue
            docs, tf = self.post[t]
            idf = math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            s[docs] += idf * tf / (tf + K1 * (1 - B + B * self.dl[docs] / self.avgdl))
        return s


def rrf(lists: list[list[int]], k: int = RRF_K, top_n: int = 5) -> list[tuple[int, Fraction]]:
    """score(d) = sum 1/(k + rank), rank from 1; exact fractions so ties are real; ties keep first-seen order."""
    scores: dict[int, Fraction] = {}
    for ranked in lists:
        for rank, i in enumerate(ranked, start=1):
            scores[i] = scores.get(i, Fraction(0)) + Fraction(1, k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]  # sorted() is stable


class Index:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, chunker: str = CHUNKER):
        self.chunks = chunks
        # BM25 statistics span every chunk in the index (as ES does); only ranking is limited to the chunker
        self.fields = {f: FieldIndex([c[f] for c in chunks], ANALYZERS[f]) for f in BOOST}
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.unit = (embeddings / np.where(norms == 0, 1, norms)).astype(np.float32)
        self.allowed = np.array([c["chunker"] == chunker for c in chunks])

    def bm25(self, query: str, size: int = POOL) -> list[int]:
        # best_fields = max over fields of boosted score (tie_breaker 0)
        s = np.max([BOOST[f] * fi.score(query) for f, fi in self.fields.items()], axis=0)
        s = np.where(self.allowed & (s > 0), s, -1.0)
        order = np.argsort(-s, kind="stable")[:size]
        return [int(i) for i in order if s[i] > 0]

    def dense(self, qvec: np.ndarray, size: int = POOL) -> list[int]:
        q = qvec / np.linalg.norm(qvec)
        s = np.where(self.allowed, self.unit @ q, -2.0)
        # the filter is hard: a masked row must not come back even when fewer than `size` rows are allowed
        return [int(i) for i in np.argsort(-s, kind="stable")[:size] if self.allowed[i]]

    def hybrid(self, query: str, qvec: np.ndarray, k: int) -> list[tuple[int, Fraction]]:
        return rrf([self.bm25(query), self.dense(qvec)], RRF_K, k)


_index: Index | None = None
_cache: dict[str, tuple[np.ndarray, float]] = {}  # query -> (vector, cost of the call that made it)


def load_index() -> Index:
    global _index
    if _index is None:
        try:
            chunks = [json.loads(l) for l in (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf8").splitlines()]
            emb = np.load(INDEX_DIR / "embeddings.npy")
        except OSError as e:
            raise RuntimeError(f"index not found in {INDEX_DIR}: run python -m scripts.export_index ({e})") from e
        _index = Index(chunks, emb)
    return _index


def embed_query(query: str) -> tuple[np.ndarray, float, bool]:
    """(vector, cost_usd of this call, cache_hit). Errors propagate as httpx exceptions for the server to map."""
    if query in _cache:
        return _cache[query][0], 0.0, True
    key = os.environ.get(KEY_VAR)  # deliberately not OPENAI_API_KEY: this project's spend uses its own key
    if not key:
        raise RuntimeError(f"{KEY_VAR} is not set")
    r = httpx.post(EMBED_URL, headers={"Authorization": f"Bearer {key}"}, timeout=15.0,
                   json={"model": "text-embedding-3-small", "input": query})
    r.raise_for_status()
    body = r.json()
    vec = np.asarray(body["data"][0]["embedding"], dtype=np.float32)
    cost = body["usage"]["total_tokens"] * EMBED_USD_PER_TOKEN
    _cache[query] = (vec, cost)
    return vec, cost, False


def search(query: str, k: int = 5) -> dict:
    """Hybrid search; the dict has the keys server.normalise_search reads from rag-hybrid's /search."""
    t0 = time.perf_counter()
    idx = load_index()
    vec, cost, hit = embed_query(query)
    hits = idx.hybrid(query, vec, k)
    results = []
    for rank, (i, score) in enumerate(hits, start=1):
        c = idx.chunks[i]
        results.append({"rank": rank, "score": float(score), "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                        "url": c["url"], "title": c["title"], "heading_path": c["heading_path"], "content": c["content"]})
    return {"results": results, "query_embedding": {"cache_hit": hit}, "cost_usd": round(cost, 8),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
