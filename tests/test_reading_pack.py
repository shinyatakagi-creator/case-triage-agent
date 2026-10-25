from scripts.reading_pack import SNIPPET_CHARS, build_pack, fence

THREAD = {"number": 1, "title": "pod stuck", "author": "alice", "created_at": "t0", "labels": ["kind/support"],
          "body": "body with ``` fence inside", "thread_truncated": True,
          "comments": [{"author": "bob", "created_at": "t1", "body": "same here"}]}


def passage(i, text="x" * 500):
    return {"source": f"docs/p{i}", "heading": f"H{i}", "text": text, "url": "u", "title": "T",
            "chunk_id": "c", "score": 1.0}


def pack(tmp_path, thread=THREAD, hits=None, number=1):
    searched = []
    hits = hits if hits is not None else {"passages": [passage(i) for i in range(7)], "cost_usd": 0, "latency_ms": 0,
                                         "cache_hit": False}

    def search(q, k):
        searched.append((q, k))
        return hits

    path = build_pack(number, tmp_path, thread_tool=lambda url: thread, search_tool=search, state_fn=lambda n: "open")
    return path.read_text(), searched, path


def test_pack_has_thread_state_and_top5_snippets(tmp_path):
    text, searched, path = pack(tmp_path)
    assert path == tmp_path / "1.md" and searched == [("pod stuck", 5)]
    assert "# pod stuck" in text and "State: open" in text
    assert "https://github.com/kubernetes/kubernetes/issues/1" in text
    assert "thread_truncated: True" in text and "same here" in text
    assert "body with ``` fence inside" in text and "````text" in text  # fence longer than inner backticks
    assert text.count("`docs/p") == 5 and "docs/p5" not in text  # top 5 only
    assert "x" * SNIPPET_CHARS in text and "x" * (SNIPPET_CHARS + 1) not in text


def test_thread_error_is_written_and_search_not_run(tmp_path):
    text, searched, _ = pack(tmp_path, thread={"error": "not_found"})
    assert "not_found" in text and searched == [] and "not run" in text


def test_search_error_is_written(tmp_path):
    text, _, _ = pack(tmp_path, hits={"error": "search_failed", "detail": "boom"})
    assert "search_failed" in text and "boom" in text


def test_no_verdict_words(tmp_path):
    text, _, _ = pack(tmp_path)
    assert not any(w in text.lower() for w in ("answerable", "verdict", "recommend"))


def test_fence_grows_past_inner_backticks():
    assert fence("a ````` b").startswith("``````text")
