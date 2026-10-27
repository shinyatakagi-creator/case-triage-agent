"""Offline tests for the in-process backend on a 5-chunk fixture index (no files, no network)."""
from fractions import Fraction

import numpy as np
import pytest

from triage import server
from triage.retrieval import inproc


def _c(cid, doc, title, heading, content, chunker="heading"):
    return {"chunk_id": cid, "doc_id": doc, "chunker": chunker, "url": f"https://x/{doc}",
            "title": title, "heading_path": heading, "content": content}


CHUNKS = [
    _c("c0", "pods", "Pods", "Pods > Lifecycle", "A pod is running when its containers start."),
    _c("c1", "rbac", "RBAC", "RBAC > Reconcile", "Use kubectl auth reconcile with --remove-extra-subjects to prune bindings."),
    _c("c2", "dns", "DNS", "DNS > Services", "Services get DNS names inside the cluster."),
    _c("c3", "rbac-fixed", "RBAC", "RBAC", "Use kubectl auth reconcile with --remove-extra-subjects.", chunker="fixed"),
    _c("c4", "pods2", "Pods", "Pods > Init", "Init containers run before app containers."),
]
EMB = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32)


@pytest.fixture(scope="module")
def idx():
    return inproc.Index(CHUNKS, EMB)


def test_bm25_exact_term_ranks_first_and_respects_chunker_filter(idx):
    ranked = idx.bm25("remove-extra-subjects")
    assert ranked[0] == 1 and 3 not in ranked  # c3 matches too but is a "fixed" chunk


def test_bm25_stems_content_and_drops_stopwords(idx):
    # Porter strips -er when the stem has measure > 1, so "containers" -> "contain" (Lucene's PorterStemFilter does the same)
    assert inproc.tokens_english("The pods' containers are running") == ["pod", "contain", "run"]
    assert idx.bm25("running")[0] == 0  # "running" -> "run" matches "running" in c0
    assert idx.bm25("the") == []  # only a stopword: no match


def test_bm25_no_match_is_empty(idx):
    assert idx.bm25("zzzunknown") == []


def test_dense_is_cosine_and_ignores_vector_length(idx):
    # cosine: c4 (0.9, 0.1) = 0.9999 beats c0 (1, 0) = 0.9950; a raw dot product would flip them (4.55 vs 5.00)
    assert idx.dense(np.array([5.0, 0.5, 0], dtype=np.float32))[:2] == [4, 0]
    assert 3 not in idx.dense(np.array([0, 1, 0], dtype=np.float32))


def test_rrf_fuses_exactly_and_ties_keep_first_seen_order():
    out = inproc.rrf([[1, 2, 3], [3, 1]], k=60, top_n=3)
    assert out[0] == (1, Fraction(1, 61) + Fraction(1, 62))  # 1: ranks 1 and 2
    assert [i for i, _ in out] == [1, 3, 2]  # 3: ranks 3 and 1 -> 1/63+1/61 > 1/62
    assert [i for i, _ in inproc.rrf([[7], [8]], top_n=2)] == [7, 8]  # tie: first list first


def test_hybrid_uses_both_lists(idx):
    top = idx.hybrid("remove-extra-subjects", np.array([1, 0, 0], dtype=np.float32), 2)
    assert {i for i, _ in top} == {1, 0}  # bm25 says c1, dense says c0


def test_search_returns_the_search_shape_and_caches_the_embedding(monkeypatch):
    monkeypatch.setattr(inproc, "_index", inproc.Index(CHUNKS, EMB))
    monkeypatch.setattr(inproc, "_cache", {})
    calls = []

    class R:
        def raise_for_status(self): pass
        def json(self): return {"data": [{"embedding": [1, 0, 0]}], "usage": {"total_tokens": 10}}

    monkeypatch.setenv("TRIAGE_OPENAI_API_KEY", "k")
    monkeypatch.setattr(inproc.httpx, "post", lambda *a, **kw: calls.append(1) or R())
    first = inproc.search("pods running", 3)
    second = inproc.search("pods running", 3)
    assert len(calls) == 1 and first["query_embedding"]["cache_hit"] is False and second["query_embedding"]["cache_hit"] is True
    assert first["cost_usd"] > 0 and second["cost_usd"] == 0
    r = server.normalise_search(first)
    assert r["passages"][0]["source"] == "pods" and len(r["passages"]) == 3
    assert set(r["passages"][0]) == {"text", "source", "url", "title", "heading", "chunk_id", "score"}


def test_backend_switch(monkeypatch):
    raw = {"results": [], "cost_usd": 0.0, "latency_ms": 1.0}
    monkeypatch.setattr(inproc, "search", lambda q, k: dict(raw, marker="inproc"))
    monkeypatch.setattr(server.httpx, "post", lambda *a, **kw: pytest.fail("es backend must not be called"))
    monkeypatch.setenv("RETRIEVAL_BACKEND", "inproc")
    assert "error" not in server.search_docs("q", 2)
    monkeypatch.delenv("RETRIEVAL_BACKEND")  # default is es
    monkeypatch.setattr(server.httpx, "post", lambda *a, **kw: (_ for _ in ()).throw(server.httpx.ConnectError("es")))
    assert server.search_docs("q", 2)["error"] == "search_failed"


def test_inproc_failures_are_error_dicts(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "inproc")
    monkeypatch.setattr(inproc, "search", lambda q, k: (_ for _ in ()).throw(RuntimeError("TRIAGE_OPENAI_API_KEY is not set")))
    assert server.search_docs("q")["error"] == "search_failed"
    monkeypatch.setattr(inproc, "search", lambda q, k: (_ for _ in ()).throw(server.httpx.ReadTimeout("t")))
    assert server.search_docs("q") == {"error": "timeout", "tool": "search_docs"}


def test_embed_query_sends_the_triage_key_explicitly(monkeypatch):
    monkeypatch.setattr(inproc, "_cache", {})
    monkeypatch.setenv("TRIAGE_OPENAI_API_KEY", "triage-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "other-secret")
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"data": [{"embedding": [1, 0, 0]}], "usage": {"total_tokens": 1}}

    monkeypatch.setattr(inproc.httpx, "post", lambda url, headers, **kw: seen.update(headers) or R())
    inproc.embed_query("q")
    assert seen["Authorization"] == "Bearer triage-secret"


def test_missing_triage_key_fails_naming_the_variable_and_ignores_openai_api_key(monkeypatch):
    monkeypatch.setattr(inproc, "_cache", {})
    monkeypatch.delenv("TRIAGE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "other-secret")  # must not be used as a fallback
    monkeypatch.setattr(inproc.httpx, "post", lambda *a, **kw: pytest.fail("no request without the key"))
    with pytest.raises(RuntimeError) as e:
        inproc.embed_query("q")
    assert "TRIAGE_OPENAI_API_KEY" in str(e.value) and "other-secret" not in str(e.value)
    # through the tool: an error dict that names the variable and leaks nothing
    monkeypatch.setattr(inproc, "_index", object())
    monkeypatch.setenv("RETRIEVAL_BACKEND", "inproc")
    out = server.search_docs("q")
    assert out["error"] == "search_failed" and "TRIAGE_OPENAI_API_KEY" in out["detail"] and "other-secret" not in str(out)
