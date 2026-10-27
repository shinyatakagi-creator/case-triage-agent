"""Run rag-hybrid's answerable golden questions through both backends and compare.

    python -m scripts.parity --golden PATH/TO/rag-hybrid/eval/golden.jsonl [--k 5]   (or set RAG_GOLDEN)

es = rag-hybrid /search on localhost (hybrid, heading chunker); inproc = triage.retrieval.inproc.
"Top 5" is the top-k CHUNKS as the agent receives them; a doc counts as hit if any of them is a gold doc.
rag-hybrid's own metrics collapse to distinct docs first, so its published numbers can differ slightly.
The BM25-only block needs no TRIAGE_OPENAI_API_KEY (no query embedding) and isolates the tokenizer/scoring port.
"""
import argparse
import json
import os
from pathlib import Path

import httpx

from triage.retrieval import inproc

GOLDEN_ENV = "RAG_GOLDEN"
RAG_API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")


def es_search(query: str, mode: str, k: int) -> list[dict]:
    r = httpx.post(f"{RAG_API_URL}/search", json={"query": query, "mode": mode, "k": k, "chunker": "heading"}, timeout=30)
    r.raise_for_status()
    return r.json()["results"]


def inproc_bm25(query: str, k: int) -> list[dict]:
    idx = inproc.load_index()
    return [idx.chunks[i] for i in idx.bm25(query, k)]


def metrics(rows: list[list[str]], gold: list[set[str]]) -> tuple[int, float]:
    """(questions with a gold doc in the top-k, mean reciprocal rank of the first gold doc over distinct docs)."""
    hits, rr = 0, 0.0
    for docs, g in zip(rows, gold):
        hits += bool(g & set(docs))
        ranked = list(dict.fromkeys(docs))
        rr += next((1 / r for r, d in enumerate(ranked, 1) if d in g), 0.0)
    return hits, rr / len(rows)


def report(name: str, a: list[list[dict]], b: list[list[dict]], gold: list[set[str]]) -> None:
    docs = lambda rows: [[h["doc_id"] for h in r] for r in rows]
    n = len(gold)
    (ha, ma), (hb, mb) = metrics(docs(a), gold), metrics(docs(b), gold)
    doc_ov = sum(len(set(x) & set(y)) / max(len(set(x)), 1) for x, y in zip(docs(a), docs(b))) / n
    chunk_ov = sum(len({h["chunk_id"] for h in x} & {h["chunk_id"] for h in y}) / max(len(x), 1) for x, y in zip(a, b)) / n
    same_top = sum(bool(x) and bool(y) and x[0]["chunk_id"] == y[0]["chunk_id"] for x, y in zip(a, b))
    print(f"\n{name} (n={n})\n  backend  hit@k   MRR\n  es       {ha}/{n}   {ma:.3f}\n  inproc   {hb}/{n}   {mb:.3f}")
    print(f"  top-k doc_id overlap (mean share of es's docs also in inproc's): {doc_ov:.3f}")
    print(f"  top-k chunk_id overlap: {chunk_ov:.3f}   same rank-1 chunk: {same_top}/{n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", help=f"path to rag-hybrid's eval/golden.jsonl (or set {GOLDEN_ENV})")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args(argv)
    golden = a.golden or os.environ.get(GOLDEN_ENV)
    if not golden:
        ap.error(f"no golden file: pass --golden PATH or set {GOLDEN_ENV}")
    try:
        lines = Path(golden).read_text().splitlines()
    except OSError as e:
        ap.error(f"cannot read golden file {golden!r}: {e.strerror}")
    qs = [q for q in map(json.loads, lines) if q["answerable"]]
    gold = [set(q["expected_doc_ids"]) for q in qs]
    texts = [server_clean(q["question"]) for q in qs]

    report(f"BM25 only, k={a.k}", [es_search(t, "bm25", a.k) for t in texts], [inproc_bm25(t, a.k) for t in texts], gold)

    if not os.environ.get(inproc.KEY_VAR):
        print(f"\nHYBRID NOT RUN: {inproc.KEY_VAR} is not set (inproc needs it to embed the 26 queries)")
        return 2
    es_h = [es_search(t, "hybrid", a.k) for t in texts]
    ip_raw = [inproc.search(t, a.k) for t in texts]
    spent = sum(r["cost_usd"] for r in ip_raw)
    report(f"HYBRID, k={a.k}", es_h, [r["results"] for r in ip_raw], gold)
    print(f"  inproc query-embedding spend: ${spent:.6f}")
    return 0


def server_clean(q: str) -> str:
    from triage.server import clean_query  # same cleaning the tool applies
    return clean_query(q)


if __name__ == "__main__":
    raise SystemExit(main())
