"""Agent loop tests: fake LLM with scripted turns, fake tools returning synthetic fixtures. No live calls."""
import json
import math

import pytest
from agent_helpers import SRC, URL, FakeLLM, FakeTools, final, fx, turn, use

from triage import agent
from triage.agent import MAX_PLAUSIBLE_TOKENS, MAX_TOOL_CALLS, InvalidCitation, run_agent
from triage.contracts import MAX_STEPS, Comment, IssueThread, Passage, SearchResult, Signals, TriageNote, contract_problems
from triage.llm import DEFAULT_MODEL, Ledger


@pytest.fixture
def ledger(tmp_path):
    return Ledger(per_run_cap=1.0, per_day_cap=10.0, path=tmp_path / "spend.json")


# One scripted turn is 100 input + 50 output tokens at claude-haiku-5-5 short-tier rates (0.10 / 0.50).
TURN_USD = (100 * 0.10 + 50 * 0.50) / 1e6


def search_turn(q="q", id="s"):
    return turn(use("search_docs", id=id, query=q))


def test_happy_path_valid_citation(ledger):
    llm = FakeLLM([turn(use("get_issue_thread")), search_turn("widgets"), final(category="scheduling")])
    res = run_agent(URL, FakeTools(), llm, ledger, categories=["scheduling", "other"])
    n = res["note"]
    assert n["outcome"] == "answered" and n["steps"] == 3
    assert [c["source"] for c in n["citations"]] == [SRC]
    assert n["category"] == "scheduling"
    assert contract_problems(n, res["search_results"]) == []
    assert n["cost_usd"] == pytest.approx(0.0001 + 3 * TURN_USD)
    assert ledger.run_usd == pytest.approx(n["cost_usd"])  # model + search both counted
    assert [c["tool"] for c in n["tool_calls"]] == ["get_issue_thread", "search_docs"]


def test_final_answer_on_last_allowed_turn(ledger):
    llm = FakeLLM([search_turn(id=f"s{i}") for i in range(MAX_STEPS - 1)] + [final()])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert res["note"]["outcome"] == "answered" and res["note"]["steps"] == MAX_STEPS


def test_multi_block_turn_runs_all_and_returns_all_results(ledger):
    t = FakeTools()
    llm = FakeLLM([turn(use("get_issue_thread", id="a"), use("search_docs", id="b", query="q"),
                        use("extract_error_signals", id="c", text="x")), final()])
    res = run_agent(URL, t, llm, ledger)
    assert res["note"]["outcome"] == "answered"
    results = llm.seen[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert [n for n, _ in t.calls] == ["get_issue_thread", "search_docs", "extract_error_signals"]


def test_final_answer_wins_over_other_blocks_in_same_turn(ledger):
    t = FakeTools()
    llm = FakeLLM([turn(use("search_docs", query="q"), use("final_answer", outcome="cant_tell"))])
    res = run_agent(URL, t, llm, ledger)
    assert res["note"]["outcome"] == "cant_tell" and t.calls == []


@pytest.mark.parametrize("outcome,cites,extra,expect", [
    ("answered", (SRC,), {"category": "scheduling"}, "answered"),
    ("answered", (SRC,), {}, "error"),                            # category missing
    ("answered", (SRC,), {"category": "nonsense"}, "error"),      # not in the set
    ("cant_tell", (), {"category": "scheduling"}, "cant_tell"),   # extra category is dropped, not an error
    ("cant_tell", (), {}, "cant_tell"),
])
def test_category_rule(ledger, outcome, cites, extra, expect):
    llm = FakeLLM([search_turn(), final(outcome, citations=cites, **extra)])
    res = run_agent(URL, FakeTools(), llm, ledger, ["scheduling"])
    assert res["note"]["outcome"] == expect
    if expect == "error":
        assert "category" in res["reason"]
    if outcome == "cant_tell" and extra:
        assert "category dropped" in res["reason"]
    if expect != "answered":
        assert res["note"]["category"] is None


def test_no_category_rule_without_a_set(ledger):
    llm = FakeLLM([search_turn(), final()])
    assert run_agent(URL, FakeTools(), llm, ledger)["note"]["outcome"] == "answered"


def test_contract_problems_flags_invented_citation():
    note = TriageNote(issue_url=URL, outcome="answered", category=None, likely_cause="c",
                      citations=[{"source": "docs/made/up.md"}], next_step="n", steps=1, cost_usd=0.0, tool_calls=[])
    problems = contract_problems(note, [fx("search.json")])
    assert len(problems) == 1 and "docs/made/up.md" in problems[0]


def test_agent_drops_invented_citation_and_reports_it(ledger):
    llm = FakeLLM([search_turn(), final(citations=[SRC, "docs/made/up.md"])])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert res["dropped_citations"] == ["docs/made/up.md"]
    assert [c["source"] for c in res["note"]["citations"]] == [SRC]
    assert contract_problems(res["note"], res["search_results"]) == []


def test_only_invented_citation_is_error(ledger):
    llm = FakeLLM([search_turn(), final(citations=["docs/made/up.md"])])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert res["note"]["outcome"] == "error" and res["note"]["citations"] == []
    assert res["dropped_citations"] == ["docs/made/up.md"]


def test_answered_without_any_search_is_error(ledger):
    assert run_agent(URL, FakeTools(), FakeLLM([final()]), ledger)["note"]["outcome"] == "error"


def test_step_cap(ledger):
    llm = FakeLLM([search_turn(f"q{i}", id=f"s{i}") for i in range(MAX_STEPS + 3)])
    tools = FakeTools()
    res = run_agent(URL, tools, llm, ledger)
    assert res["note"]["outcome"] == "step_cap" and res["note"]["steps"] == MAX_STEPS
    assert len(llm.seen) == MAX_STEPS
    assert len(tools.calls) == MAX_STEPS - 1  # last turn's tools are not run


def test_cant_tell(ledger):
    res = run_agent(URL, FakeTools(), FakeLLM([final("cant_tell", citations=[])]), ledger)
    assert res["note"]["outcome"] == "cant_tell" and res["note"]["steps"] == 1


def test_timeout_retried_once_then_succeeds(ledger):
    t = FakeTools(search_docs=[{"error": "timeout", "tool": "search_docs"}, fx("search.json")])
    res = run_agent(URL, t, FakeLLM([search_turn(), final()]), ledger)
    assert res["note"]["outcome"] == "answered" and len(t.calls) == 2
    assert res["note"]["tool_calls"] == [{"tool": "search_docs", "ok": False}, {"tool": "search_docs", "ok": True}]


def test_timeout_twice_degrades_not_called_again_and_cant_tell_becomes_error(ledger):
    t = FakeTools(search_docs=[{"error": "timeout"}])
    llm = FakeLLM([search_turn("a"), search_turn("b"), final("cant_tell", citations=[])])
    res = run_agent(URL, t, llm, ledger)
    assert len(t.calls) == 2  # one call + one retry; the second model request never reaches the tool
    assert res["note"]["outcome"] == "error" and "search_docs" in res["reason"]


def test_cant_tell_ok_if_failed_tool_later_recovered(ledger):
    t = FakeTools(extract_error_signals=[{"error": "search_failed"}], get_issue_thread=[fx("thread.json")])
    llm = FakeLLM([turn(use("extract_error_signals", text="x")), final("cant_tell", citations=[])])
    assert run_agent(URL, t, llm, ledger)["note"]["outcome"] == "error"  # last result of that tool was an error


@pytest.mark.parametrize("kind", ["rate_limited", "bad_url", "not_found", "not_an_issue"])
def test_fatal_errors_not_retried(ledger, kind):
    t = FakeTools(get_issue_thread=[{"error": kind}])
    llm = FakeLLM([turn(use("get_issue_thread")), final()])
    res = run_agent(URL, t, llm, ledger)
    assert res["note"]["outcome"] == "error" and kind in res["reason"]
    assert len(t.calls) == 1 and len(llm.seen) == 1


def test_fatal_error_short_circuits_rest_of_turn(ledger):
    t = FakeTools(get_issue_thread=[{"error": "rate_limited"}])
    llm = FakeLLM([turn(*[use("get_issue_thread", id=f"g{i}") for i in range(3)])])
    res = run_agent(URL, t, llm, ledger)
    assert len(t.calls) == 1 and res["note"]["outcome"] == "error"


def test_tool_call_budget_with_50_block_turn(ledger):
    t = FakeTools()
    llm = FakeLLM([turn(*[use("search_docs", id=f"s{i}", query=f"q{i}") for i in range(50)]),
                   final("cant_tell", citations=[])])
    res = run_agent(URL, t, llm, ledger)
    assert len(t.calls) == MAX_TOOL_CALLS == 8
    results = llm.seen[1][-1]["content"]
    assert len(results) == 50 and sum(r["is_error"] for r in results) == 50 - MAX_TOOL_CALLS
    assert res["note"]["outcome"] == "cant_tell"  # refusals are not a tool's own result
    assert len(res["note"]["tool_calls"]) == MAX_TOOL_CALLS


def test_empty_search_then_reformulate(ledger):
    t = FakeTools(search_docs=[fx("search_empty.json"), fx("search.json")])
    res = run_agent(URL, t, FakeLLM([search_turn("a"), search_turn("b"), final()]), ledger)
    assert res["note"]["outcome"] == "answered" and [q for _, q in t.calls] == ["a", "b"]


def _bad_search(**over):
    return {**fx("search.json"), **over}


def _bad_passage(**over):
    s = fx("search.json")
    s["passages"][0].update(over)
    return s


@pytest.mark.parametrize("bad", [
    {"nope": 1}, "text", None, {**fx("search.json"), "passages": [{"text": "x"}]},
    _bad_search(cost_usd=None), _bad_search(cost_usd="x"), _bad_search(cost_usd=float("nan")),
    _bad_search(cost_usd=float("inf")), _bad_search(cost_usd=-1), _bad_search(cost_usd=True),
    _bad_search(latency_ms="1"), _bad_search(cache_hit="yes"), _bad_search(cache_hit=1),
    _bad_search(passages="abc"), _bad_search(passages=["abc"]),
    _bad_passage(source=["a"]), _bad_passage(text=None), _bad_passage(score="0.1"),
    _bad_passage(score=float("nan")),
    _bad_search(cost_usd=10**400), _bad_search(latency_ms=10**400), _bad_passage(score=10**400),
    _bad_search(cost_usd=-10**400),
])
def test_malformed_search_output_is_error_result(ledger, bad):
    t = FakeTools(search_docs=[bad])
    llm = FakeLLM([search_turn(), final("cant_tell", citations=[])])
    res = run_agent(URL, t, llm, ledger)
    assert res["note"]["tool_calls"] == [{"tool": "search_docs", "ok": False}]
    assert res["search_results"] == [] and res["note"]["cost_usd"] == pytest.approx(2 * TURN_USD)
    sent = llm.seen[1][-1]["content"][0]
    assert sent["is_error"] and "malformed" in sent["content"]


@pytest.mark.parametrize("bad", [
    {**fx("thread.json"), "number": "1"}, {**fx("thread.json"), "labels": "x"},
    {**fx("thread.json"), "comments": ["x"]}, {**fx("thread.json"), "comments": [{"author": 1, "created_at": "", "body": ""}]},
    {**fx("thread.json"), "thread_truncated": "no"}, {**fx("thread.json"), "body": None},
])
def test_malformed_thread_output_is_error(ledger, bad):
    llm = FakeLLM([turn(use("get_issue_thread")), final("cant_tell", citations=[])])
    res = run_agent(URL, FakeTools(get_issue_thread=[bad]), llm, ledger)
    assert res["note"]["tool_calls"] == [{"tool": "get_issue_thread", "ok": False}]
    assert res["note"]["outcome"] == "error"


@pytest.mark.parametrize("bad", [{"versions": "v1", "k8s_reasons": [], "error_lines": [], "stack_traces": []},
                                 {"versions": [1], "k8s_reasons": [], "error_lines": [], "stack_traces": []}, {}])
def test_malformed_signals_output_is_error(ledger, bad):
    llm = FakeLLM([turn(use("extract_error_signals", text="x")), final("cant_tell", citations=[])])
    res = run_agent(URL, FakeTools(extract_error_signals=[bad]), llm, ledger)
    assert res["note"]["tool_calls"] == [{"tool": "extract_error_signals", "ok": False}]


def test_validators_match_contract_keys():
    assert set(agent.PASSAGE_SPEC) == set(Passage.__required_keys__)
    assert set(agent.SEARCH_SPEC) == set(SearchResult.__required_keys__)
    assert set(agent.COMMENT_SPEC) == set(Comment.__required_keys__)
    assert set(agent.THREAD_SPEC) == set(IssueThread.__required_keys__)
    assert set(agent.SIGNALS_SPEC) == set(Signals.__required_keys__)


def test_tool_that_raises_does_not_crash(ledger):
    t = FakeTools(search_docs=[RuntimeError("boom")])
    llm = FakeLLM([search_turn(), final("cant_tell", citations=[])])
    assert run_agent(URL, t, llm, ledger)["note"]["outcome"] == "error"  # cant_tell after a failed tool


def test_block_without_id_and_non_dict_blocks_do_not_crash(ledger):
    noid = {"type": "tool_use", "name": "search_docs", "input": {"query": "q"}}
    llm = FakeLLM([turn("junk", None, noid, 5), final()])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert res["note"]["outcome"] == "answered"
    assert llm.seen[1][-1]["content"][0]["tool_use_id"] == "missing-id-2"
    assert [b["id"] for b in llm.seen[1][1]["content"] if b.get("type") == "tool_use"] == ["missing-id-2"]


def test_unknown_tool_and_bad_input_do_not_crash(ledger):
    llm = FakeLLM([turn({"type": "tool_use", "id": "x", "name": 7, "input": "zzz"},
                        use("nonexistent", id="y")), final("cant_tell", citations=[])])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert res["note"]["outcome"] == "cant_tell"
    assert all(r["is_error"] for r in llm.seen[1][-1]["content"])


def test_per_run_cap_stops_before_call(tmp_path):
    led = Ledger(per_run_cap=0.0001, per_day_cap=10.0, path=tmp_path / "s.json")
    llm = FakeLLM([turn(use("search_docs", query="q"), i=1000, o=1000), final()])
    res = run_agent(URL, FakeTools(), llm, led)
    assert res["note"]["outcome"] == "error" and "per-run cap" in res["reason"]
    assert len(llm.seen) == 1


def test_search_cost_counts_toward_run_cap(tmp_path):
    led = Ledger(per_run_cap=0.00005, per_day_cap=10.0, path=tmp_path / "s.json")
    llm = FakeLLM([turn(use("search_docs", query="q"), i=0, o=0), final()])  # model free; search 0.0001
    res = run_agent(URL, FakeTools(), llm, led)
    assert "per-run cap" in res["reason"] and len(llm.seen) == 1


def test_per_day_cap_persisted_stops_before_first_call(tmp_path):
    p = tmp_path / "s.json"
    Ledger(per_run_cap=1, per_day_cap=0.0001, path=p).charge(DEFAULT_MODEL, 1000, 1000)  # 0.0006 USD
    llm = FakeLLM([final()])
    res = run_agent(URL, FakeTools(), llm, Ledger(per_run_cap=1, per_day_cap=0.0001, path=p))
    assert res["note"]["outcome"] == "error" and "per-day cap" in res["reason"] and llm.seen == []


def test_unreadable_spend_file_fails_closed(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("not json")
    llm = FakeLLM([final()])
    res = run_agent(URL, FakeTools(), llm, Ledger(path=p))
    assert res["note"]["outcome"] == "error" and llm.seen == []


def test_failed_spend_write_keeps_cost_and_stops(tmp_path, monkeypatch):
    led = Ledger(per_run_cap=1, per_day_cap=10, path=tmp_path / "s.json")
    monkeypatch.setattr("triage.llm.os.replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    llm = FakeLLM([search_turn(), final()])
    res = run_agent(URL, FakeTools(), llm, led)
    assert res["note"]["outcome"] == "error" and "spend ledger write failed" in res["reason"]
    assert res["note"]["cost_usd"] == pytest.approx(TURN_USD)  # the paid call is still in the note
    assert len(llm.seen) == 1


def deep(n=5000):
    x = []
    for _ in range(n):
        x = [x]
    return x


def test_budget_refusal_is_not_a_tool_error_for_cant_tell_rule(ledger):
    """MAX_TOOL_CALLS successes, then one refused call: a following cant_tell stays cant_tell."""
    blocks = [use("search_docs", id=f"s{i}", query="q") for i in range(MAX_TOOL_CALLS + 1)]
    llm = FakeLLM([turn(*blocks), final("cant_tell", citations=[])])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert llm.seen[1][-1]["content"][-1]["is_error"] and res["note"]["outcome"] == "cant_tell"


def test_error_dict_with_nested_detail_does_not_crash(ledger):
    t = FakeTools(search_docs=[{"error": "search_failed", "detail": deep(), "tool": 5}])
    res = run_agent(URL, t, FakeLLM([search_turn(), final("cant_tell", citations=[])]), ledger)
    assert res["note"]["outcome"] == "error"


@pytest.mark.parametrize("inp", [{"query": deep()}, {"query": 5}, {}, {"query": "q", "k": float("inf")},
                                 {"query": "q", "k": 10**400}, {"query": "q", "k": deep()}])
def test_garbage_model_tool_input_does_not_crash(ledger, inp):
    llm = FakeLLM([turn(use("search_docs", **inp)), final("cant_tell", citations=[])])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert res["note"]["outcome"] in ("cant_tell", "error")


def test_nested_or_non_string_citations_do_not_crash(ledger):
    llm = FakeLLM([search_turn(), turn(use("final_answer", outcome="answered", citations=[deep(), 5, SRC],
                                          likely_cause=deep(), category=deep()))])
    res = run_agent(URL, FakeTools(), llm, ledger, categories=["a"])
    assert res["note"]["outcome"] == "error"  # deep category on "answered" is a contract problem, not a crash
    assert all(isinstance(d, InvalidCitation) for d in res["dropped_citations"]) and len(res["dropped_citations"]) == 2 and res["note"]["category"] is None


def test_search_cost_10_pow_400_is_malformed_not_crash(ledger):
    res = run_agent(URL, FakeTools(search_docs=[_bad_search(cost_usd=10**400)]),
                    FakeLLM([search_turn(), final("cant_tell", citations=[])]), ledger)
    assert res["search_results"] == [] and ledger.load_error is None


@pytest.mark.parametrize("bad", [{"content": [], "input_tokens": 10**400, "output_tokens": 50},
                                 {"content": [], "input_tokens": 100, "output_tokens": 10**400},
                                 {"content": [], "input_tokens": 10**400, "output_tokens": 10**400},
                                 {"content": [], "input_tokens": 10**12, "output_tokens": 50}])
def test_huge_token_counts_error_keep_known_cost_no_crash(ledger, bad):
    res = run_agent(URL, FakeTools(), FakeLLM([bad]), ledger)
    assert res["note"]["outcome"] == "error" and "token counts" in res["reason"] and res["note"]["steps"] == 1
    assert ledger.load_error is None and res["note"]["cost_usd"] == pytest.approx(ledger.run_usd)


@pytest.mark.parametrize("bad", [{"content": [], "output_tokens": 50}, {"content": [], "input_tokens": 100},
                                 {"content": [], "input_tokens": "x", "output_tokens": 50},
                                 {"content": [], "input_tokens": float("nan"), "output_tokens": 50}])
def test_missing_token_counts_is_error_but_known_cost_counted(ledger, bad):
    res = run_agent(URL, FakeTools(), FakeLLM([bad]), ledger)
    assert res["note"]["outcome"] == "error" and "token counts" in res["reason"]
    assert res["note"]["cost_usd"] > 0 and res["note"]["steps"] == 1


def test_cache_hit_carried_through(ledger):
    t = FakeTools(search_docs=[fx("search.json"), {**fx("search.json"), "cache_hit": False}])
    res = run_agent(URL, t, FakeLLM([search_turn("a"), search_turn("b"), final()]), ledger)
    assert [r["cache_hit"] for r in res["search_results"]] == [True, False]


def test_model_only_sees_what_the_thread_tool_returned(ledger):
    """Leakage: the agent never fetches comments itself; the model gets only the tool's output."""
    llm = FakeLLM([turn(use("get_issue_thread", issue_url="https://github.com/other/repo/issues/9")),
                   final("cant_tell", citations=[])])
    t = FakeTools()
    run_agent(URL, t, llm, ledger)
    assert t.calls == [("get_issue_thread", URL)]  # model-supplied URL is ignored
    msgs = llm.seen[1]
    assert msgs[0]["content"] == f"Triage this issue: {URL}"
    assert msgs[2]["content"][0]["content"] == json.dumps(fx("thread.json"))


def test_infinite_summed_search_cost_is_an_error_with_finite_note_cost(ledger):
    big = 1.7e308  # each valid on its own; the sum overflows to inf
    sr = _bad_search(cost_usd=big)
    llm = FakeLLM([turn(use("search_docs", id="a", query="q"), use("search_docs", id="b", query="q")),
                   final("cant_tell", citations=[])])
    res = run_agent(URL, FakeTools(search_docs=[sr]), llm, ledger)
    assert res["note"]["outcome"] == "error" and "overflow" in res["reason"]
    assert math.isfinite(res["note"]["cost_usd"])


def test_literal_marker_string_cannot_collide_with_invalid_citation(ledger):
    llm = FakeLLM([search_turn(), turn(use("final_answer", outcome="answered",
                                          citations=["<non-string citation>", 5, SRC]))])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert "<non-string citation>" in res["dropped_citations"]  # the real string, kept as a str
    assert InvalidCitation("int") in res["dropped_citations"]
    assert InvalidCitation("int") != "<non-string citation>"


def test_token_plausibility_bound_is_ten_million():
    assert MAX_PLAUSIBLE_TOKENS == 10**7
    assert agent._count_ok(10**7) and not agent._count_ok(10**7 + 1)


def test_cache_tokens_are_priced_and_absent_ones_are_zero(ledger):
    t = final("cant_tell", citations=[])
    plain = run_agent(URL, FakeTools(), FakeLLM([dict(t)]), ledger)
    t.update(cache_write_5m_tokens=1000, cache_write_1h_tokens=2000, cache_read_tokens=3000)
    ledger2 = Ledger(per_run_cap=1, per_day_cap=1, path=ledger.path.with_name("s2.json"))
    cached = run_agent(URL, FakeTools(), FakeLLM([t]), ledger2)
    extra = (1000 * 0.125 + 2000 * 0.20 + 3000 * 0.01) / 1e6
    assert cached["note"]["outcome"] == "cant_tell"
    assert cached["note"]["cost_usd"] == pytest.approx(plain["note"]["cost_usd"] + extra)


@pytest.mark.parametrize("field,bad", [("cache_read_tokens", "x"), ("cache_read_tokens", None),
                                       ("cache_write_5m_tokens", -1), ("cache_write_1h_tokens", float("nan")),
                                       ("cache_read_tokens", 10**400)])
def test_invalid_cache_field_is_error_but_known_cost_counted(ledger, field, bad):
    t = final("cant_tell", citations=[])
    t[field] = bad
    res = run_agent(URL, FakeTools(), FakeLLM([t]), ledger)
    assert res["note"]["outcome"] == "error" and "token counts" in res["reason"]
    assert res["note"]["cost_usd"] > 0 and ledger.load_error is None


def test_chunk_id_citation_is_mapped_to_its_source(ledger):
    chunk = fx("search.json")["passages"][0]["chunk_id"]
    for cited in ([chunk], [chunk, SRC]):  # second case: mapped id duplicates a real source, kept once
        llm = FakeLLM([search_turn(), final(citations=cited, category="scheduling")])
        res = run_agent(URL, FakeTools(), llm, ledger, ["scheduling"])
        assert res["note"]["outcome"] == "answered"
        assert [c["source"] for c in res["note"]["citations"]] == [SRC]
        assert f"{chunk} -> {SRC}" in res["reason"] and res["dropped_citations"] == []


def test_unreturned_citation_is_still_dropped(ledger):
    llm = FakeLLM([search_turn(), final(citations=["not-returned-id"], category="scheduling")])
    res = run_agent(URL, FakeTools(), llm, ledger, ["scheduling"])
    assert res["note"]["outcome"] == "error" and res["dropped_citations"] == ["not-returned-id"]
