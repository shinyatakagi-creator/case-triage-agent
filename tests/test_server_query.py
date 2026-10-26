from triage import server


def test_clean_query_collapses_whitespace_and_caps_length():
    q = server.clean_query("title\r\n\r\nbody\twith   gaps " + "x" * 600)
    assert "\r" not in q and "\n" not in q and "\t" not in q and "  " not in q
    assert len(q) == server.MAX_QUERY_CHARS and q.startswith("title body with gaps ")


def test_search_docs_sends_the_cleaned_query(monkeypatch):
    sent = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"results": [], "cost_usd": 0.0, "latency_ms": 1.0}

    monkeypatch.setattr(server.httpx, "post", lambda url, json, timeout: sent.update(json) or R())
    assert "error" not in server.search_docs("a\r\nb", 3)
    assert sent["query"] == "a b" and sent["k"] == 3
