from scripts.reading_pack import FULL_TEXT_K, SNIPPET_CHARS, build_pack, evidence_lines, fence, search_words

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
    assert text.split("## Full text")[0].count("`docs/p") == 5 and "docs/p5" not in text  # top 5 only
    listing = text.split("## Full text")[0]  # the top-5 list itself stays at 200 characters
    assert "x" * SNIPPET_CHARS in listing and "x" * (SNIPPET_CHARS + 1) not in listing


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


def test_full_text_section_has_whole_chunk_for_top_3_only(tmp_path):
    hits = {"passages": [passage(i, text=f"chunk{i} " + "w" * 600 + f" END{i}") for i in range(7)],
            "cost_usd": 0, "latency_ms": 0, "cache_hit": False}
    text, _, _ = pack(tmp_path, hits=hits)
    full = text.split("## Full text of the top 3 chunks")[1]
    assert FULL_TEXT_K == 3
    for i in range(3):
        assert f"chunk{i} " + "w" * 600 + f" END{i}" in full  # whole chunk, not 200 chars
    assert "END3" not in text  # chunks 4-5 get 200-char snippets only
    assert text.index("## Full text") > text.index("search_docs top 5")  # after the top-5 list
    assert "<details>" in full and "END0" not in text.split("## Full text")[0]  # list part stays 200 chars


def test_no_full_text_section_when_search_errors(tmp_path):
    text, _, _ = pack(tmp_path, hits={"error": "search_failed"})
    assert "Full text" not in text


def test_noise_words_short_words_and_duplicates_are_dropped():
    got = search_words("Pod stuck in Pending with the scheduler, scheduler FAILED!", "")
    assert sorted(got) == ["failed", "pending", "scheduler", "stuck"]  # "pod"/"in"/"the" short; "with" stopword
    noise = "issue problem error please kindly request solution thanks thank regards team dear hello following"
    noise += " using trying getting would could should after before version cluster kubernetes kubectl"
    assert search_words(noise, noise) == []
    assert search_words("a an of to", "") == []


def test_body_words_are_used_and_the_body_is_cut_at_the_version_heading():
    body = "The coredns pods crash with a loop\n### Kubernetes version\nsecretword deployed"
    got = search_words("DNS broken", body)
    assert "coredns" in got and "crash" in got and "secretword" not in got
    assert "secretword" not in search_words("t", "x\n### Environment\nsecretword")  # either heading cuts
    assert "tailword" not in search_words("t", "x" * 1500 + " tailword")  # no heading: first 1,500 chars
    assert "headword" in search_words("t", "headword " + "x" * 2000)


def test_dotted_and_hyphenated_words_are_kept_whole_and_weighted():
    body = "edit /etc/resolv.conf then check --record and change-cause. Also plainword other plainish."
    got = search_words("zzzz", body)
    assert "etc/resolv.conf" in got  # the token starts at the first letter, so the leading slash is not part of it
    assert "--record" in got and "change-cause" in got and "cause" not in got  # trailing dot dropped, not split
    assert got.index("change-cause") < got.index("plainword")  # special characters outrank a plain word


def test_title_words_outrank_body_words_and_long_words_outrank_short():
    assert search_words("scheduler", "dnsmasq-conf extraordinarily lengthy")[:1] == ["scheduler"]  # title 3 > special 2
    got = search_words("zzzz", "short words extraordinarily")
    assert got.index("extraordinarily") < got.index("short")  # len > 8 gets +1


def test_search_words_capped_at_eight():
    body = " ".join(f"word{i:02d}x" for i in range(20))
    assert len(search_words("title", body)) == 8


def test_evidence_lines_need_two_distinct_words_whole_words_and_cap_at_five():
    words = ["pending", "scheduler"]
    text = "\n".join(["pending only", "pending pending pending", "scheduler and pending together",
                      "  Scheduler: PENDING  "] + [f"pending scheduler {i}" for i in range(10)])
    got = evidence_lines(text, words)
    assert got[:2] == ["scheduler and pending together", "Scheduler: PENDING"] and len(got) == 5
    assert evidence_lines("pending only\nnothing", words) == []
    assert evidence_lines("pendingly schedulers", words) == []  # whole words only
    assert evidence_lines("see /etc/resolv.conf and pending", ["resolv.conf", "pending"]) == [
        "see /etc/resolv.conf and pending"]  # a path prefix does not hide a dotted word


def test_evidence_lines_are_ordered_by_word_count_then_chunk_order():
    words = ["aaaa", "bbbb", "cccc"]
    text = "aaaa bbbb first\naaaa bbbb cccc three\naaaa bbbb second\ncccc aaaa third two"
    assert evidence_lines(text, words) == ["aaaa bbbb cccc three", "aaaa bbbb first", "aaaa bbbb second",
                                          "cccc aaaa third two"]
    assert len(evidence_lines("aaaa bbbb " + "y" * 400, words)[0]) == 200


def test_evidence_section_lists_words_hit_numbers_and_no_match_message(tmp_path):
    hits = {"passages": [passage(0, "intro\nthe pod stays pending until the scheduler runs\nend"),
                         passage(1, "nothing relevant here")] + [passage(i, "x") for i in range(2, 7)],
            "cost_usd": 0, "latency_ms": 0, "cache_hit": False}
    thread = {**THREAD, "title": "Pod pending, scheduler never runs", "body": "retries forever"}
    text, _, _ = pack(tmp_path, thread=thread, hits=hits)
    ev = text.split("## Evidence lines")[1]
    assert text.index("## Evidence lines") > text.index("## Full text of the top 3 chunks")
    assert "Title/body words used: scheduler, pending, never, runs, retries, forever" in ev  # scheduler: title + long
    assert "1. `docs/p0`" in ev and "- the pod stays pending until the scheduler runs" in ev
    assert "2. `docs/p1`\n   - no matching lines" in ev
    assert "6. " not in ev and ev.count("no matching lines") == 4  # hits 2-5; only the top 5 get a block


def test_no_evidence_section_on_errors(tmp_path):
    assert "Evidence lines" not in pack(tmp_path, hits={"error": "search_failed"})[0]
    assert "Evidence lines" not in pack(tmp_path, thread={"error": "not_found"})[0]
