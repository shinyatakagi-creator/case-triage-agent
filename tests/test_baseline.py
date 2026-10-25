"""Baseline tests: exactly one thread fetch, one search, one model call. Fakes only, no live calls."""
import json

import pytest
from agent_helpers import SRC, URL, FakeLLM, FakeTools, final, fx, turn

from triage.baseline import QUERY_BODY_CHARS, build_query, run_baseline
from triage.contracts import contract_problems
from triage.llm import Ledger


@pytest.fixture
def ledger(tmp_path):
    return Ledger(per_run_cap=1.0, per_day_cap=10.0, path=tmp_path / "spend.json")


def test_one_search_one_model_call(ledger):
    t, llm = FakeTools(), FakeLLM([final()])
    res = run_baseline(URL, t, llm, ledger, categories=["a"])
    assert [n for n, _ in t.calls] == ["get_issue_thread", "search_docs"]
    assert len(llm.seen) == 1
    n = res["note"]
    assert n["outcome"] == "answered" and n["steps"] == 1
    assert contract_problems(n, res["search_results"]) == []
    assert [c["source"] for c in n["citations"]] == [SRC]
    assert res["search_results"][0]["cache_hit"] is True
    assert ledger.run_usd == pytest.approx(n["cost_usd"])


def test_query_is_deterministic_title_plus_truncated_body(ledger):
    thread = {**fx("thread.json"), "body": "x" * 5000}
    t = FakeTools(get_issue_thread=[thread])
    run_baseline(URL, t, FakeLLM([final()]), ledger)
    assert t.calls[1] == ("search_docs", build_query(thread))
    assert build_query(thread) == f"{thread['title']} " + "x" * QUERY_BODY_CHARS


def test_baseline_invented_citation_dropped(ledger):
    res = run_baseline(URL, FakeTools(), FakeLLM([final(citations=["docs/made/up.md"])]), ledger)
    assert res["note"]["outcome"] == "error" and res["dropped_citations"] == ["docs/made/up.md"]


def test_baseline_no_final_answer_is_error(ledger):
    res = run_baseline(URL, FakeTools(), FakeLLM([turn({"type": "text", "text": "hi"})]), ledger)
    assert res["note"]["outcome"] == "error"


def test_baseline_thread_error_makes_no_model_call(ledger):
    llm = FakeLLM([final()])
    res = run_baseline(URL, FakeTools(get_issue_thread=[{"error": "rate_limited"}]), llm, ledger)
    assert res["note"]["outcome"] == "error" and llm.seen == []


def test_baseline_does_not_retry_timeouts(ledger):
    t = FakeTools(search_docs=[{"error": "timeout"}, fx("search.json")])
    llm = FakeLLM([final()])
    res = run_baseline(URL, t, llm, ledger)
    assert [n for n, _ in t.calls] == ["get_issue_thread", "search_docs"]  # exactly one search
    assert res["note"]["outcome"] == "error" and llm.seen == [] and "search_docs" in res["reason"]


@pytest.mark.parametrize("bad", [{"error": "search_failed"}, {"nope": 1}])
def test_baseline_search_failure_is_error_without_model_call(ledger, bad):
    llm = FakeLLM([final()])
    res = run_baseline(URL, FakeTools(search_docs=[bad]), llm, ledger)
    assert res["note"]["outcome"] == "error" and llm.seen == []


def test_baseline_empty_search_still_asks_model(ledger):
    llm = FakeLLM([final("cant_tell", citations=[])])
    res = run_baseline(URL, FakeTools(search_docs=[fx("search_empty.json")]), llm, ledger)
    assert res["note"]["outcome"] == "cant_tell" and len(llm.seen) == 1


def test_baseline_prompt_has_only_returned_thread(ledger):
    llm = FakeLLM([final("cant_tell", citations=[])])
    run_baseline(URL, FakeTools(), llm, ledger)
    assert json.loads(llm.seen[0][0]["content"])["thread"] == fx("thread.json")


def test_baseline_cap_stops_before_call(tmp_path):
    led = Ledger(per_run_cap=0.0, per_day_cap=1.0, path=tmp_path / "s.json")
    llm = FakeLLM([final()])
    res = run_baseline(URL, FakeTools(), llm, led)
    assert res["note"]["outcome"] == "error" and llm.seen == []
