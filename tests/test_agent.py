"""Agent loop tests: fake LLM with scripted turns, fake tools returning synthetic fixtures. No live calls."""
import copy
import json
from pathlib import Path

import pytest

from triage.agent import run_agent
from triage.contracts import MAX_STEPS, contract_problems
from triage.llm import DEFAULT_MODEL, Ledger

FIX = Path(__file__).parent / "fixtures" / "agent"
URL = "https://github.com/kubernetes/kubernetes/issues/1"
SRC = "docs/invented/widgets.md"


def fx(name):
    return json.loads((FIX / name).read_text())


def turn(*blocks, i=100, o=50):
    return {"content": list(blocks), "input_tokens": i, "output_tokens": o}


def use(name, id="t", **inp):
    return {"type": "tool_use", "id": id, "name": name, "input": inp}


def final(outcome="answered", citations=(SRC,), **kw):
    return turn(use("final_answer", outcome=outcome, citations=list(citations),
                    likely_cause="cause", next_step="step", **kw))


class FakeLLM:
    model = DEFAULT_MODEL

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def complete(self, system, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        return self.turns.pop(0)


class FakeTools:
    """responses: tool -> list of results returned in order (the last one repeats)."""

    def __init__(self, **responses):
        self.responses = {"search_docs": [fx("search.json")], "get_issue_thread": [fx("thread.json")],
                          "extract_error_signals": [fx("signals.json")], **responses}
        self.calls = []

    def _next(self, name, args):
        self.calls.append((name, args))
        q = self.responses[name]
        r = q.pop(0) if len(q) > 1 else q[0]
        if isinstance(r, Exception):
            raise r
        return r

    def search_docs(self, query, k=5):
        return self._next("search_docs", query)

    def get_issue_thread(self, issue_url):
        return self._next("get_issue_thread", issue_url)

    def extract_error_signals(self, text):
        return self._next("extract_error_signals", text)


@pytest.fixture
def ledger(tmp_path):
    return Ledger(per_run_cap=1.0, per_day_cap=10.0, path=tmp_path / "spend.json")


def names(res):
    return [c["tool"] for c in res["note"]["tool_calls"]]


def test_happy_path_valid_citation(ledger):
    llm = FakeLLM([turn(use("get_issue_thread")), turn(use("search_docs", query="widgets")), final()])
    res = run_agent(URL, FakeTools(), llm, ledger, categories=["scheduling", "other"])
    n = res["note"]
    assert n["outcome"] == "answered" and n["steps"] == 3
    assert [c["source"] for c in n["citations"]] == [SRC]
    assert n["category"] is None  # model gave none
    assert contract_problems(n, res["search_results"]) == []
    assert n["cost_usd"] == pytest.approx(0.0001 + 3 * (100 * 1.0 + 50 * 5.0) / 1e6)
    assert names(res) == ["get_issue_thread", "search_docs"]


def test_category_only_from_provided_list(ledger):
    for cats, expect in ((["scheduling"], "scheduling"), (["other"], None), (None, None)):
        llm = FakeLLM([final("cant_tell", citations=[], category="scheduling")])
        assert run_agent(URL, FakeTools(), llm, ledger, cats)["note"]["category"] == expect


def test_invented_citation_flagged_and_dropped(ledger):
    # contract_problems itself flags it...
    ok = run_agent(URL, FakeTools(), FakeLLM([turn(use("search_docs", query="q")), final()]), ledger)["note"]
    bad = {**ok, "citations": [{"source": "docs/made/up.md"}]}
    assert any("not returned" in p for p in contract_problems(bad, run_search_results()))
    # ...and the agent never lets one into the note
    llm = FakeLLM([turn(use("search_docs", query="q")), final(citations=[SRC, "docs/made/up.md"])])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert res["dropped_citations"] == ["docs/made/up.md"]
    assert [c["source"] for c in res["note"]["citations"]] == [SRC]
    assert contract_problems(res["note"], res["search_results"]) == []


def run_search_results():
    return [fx("search.json")]


def test_only_invented_citation_is_error(ledger):
    llm = FakeLLM([turn(use("search_docs", query="q")), final(citations=["docs/made/up.md"])])
    res = run_agent(URL, FakeTools(), llm, ledger)
    assert res["note"]["outcome"] == "error" and res["note"]["citations"] == []
    assert res["dropped_citations"] == ["docs/made/up.md"]


def test_answered_without_any_search_is_error(ledger):
    res = run_agent(URL, FakeTools(), FakeLLM([final()]), ledger)
    assert res["note"]["outcome"] == "error"


def test_step_cap(ledger):
    llm = FakeLLM([turn(use("search_docs", query=f"q{i}")) for i in range(MAX_STEPS + 3)])
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
    llm = FakeLLM([turn(use("search_docs", query="q")), final()])
    res = run_agent(URL, t, llm, ledger)
    assert res["note"]["outcome"] == "answered" and len(t.calls) == 2
    assert res["note"]["tool_calls"] == [{"tool": "search_docs", "ok": False}, {"tool": "search_docs", "ok": True}]


def test_timeout_twice_degrades_and_is_not_called_again(ledger):
    t = FakeTools(search_docs=[{"error": "timeout"}])
    llm = FakeLLM([turn(use("search_docs", query="a")), turn(use("search_docs", query="b")),
                   final("cant_tell", citations=[])])
    res = run_agent(URL, t, llm, ledger)
    assert len(t.calls) == 2  # one call + one retry; the second model request never reaches the tool
    assert res["note"]["outcome"] == "cant_tell"


@pytest.mark.parametrize("kind", ["rate_limited", "bad_url", "not_found", "not_an_issue"])
def test_fatal_errors_not_retried(ledger, kind):
    t = FakeTools(get_issue_thread=[{"error": kind}])
    llm = FakeLLM([turn(use("get_issue_thread")), final()])
    res = run_agent(URL, t, llm, ledger)
    assert res["note"]["outcome"] == "error" and kind in res["reason"]
    assert len(t.calls) == 1 and len(llm.seen) == 1


def test_empty_search_then_reformulate(ledger):
    t = FakeTools(search_docs=[fx("search_empty.json"), fx("search.json")])
    llm = FakeLLM([turn(use("search_docs", query="a")), turn(use("search_docs", query="b")), final()])
    res = run_agent(URL, t, llm, ledger)
    assert res["note"]["outcome"] == "answered" and [q for _, q in t.calls] == ["a", "b"]


@pytest.mark.parametrize("bad", [{"nope": 1}, "text", None, {"passages": [{"text": "x"}], "cost_usd": 0,
                                                              "latency_ms": 0, "cache_hit": False}])
def test_malformed_tool_output_is_error_result(ledger, bad):
    t = FakeTools(search_docs=[bad])
    llm = FakeLLM([turn(use("search_docs", query="q")), final("cant_tell", citations=[])])
    res = run_agent(URL, t, llm, ledger)
    assert res["note"]["tool_calls"] == [{"tool": "search_docs", "ok": False}]
    assert res["search_results"] == [] and res["note"]["outcome"] == "cant_tell"
    sent = llm.seen[1][-1]["content"][0]
    assert sent["is_error"] and "malformed" in sent["content"]


def test_tool_that_raises_does_not_crash(ledger):
    t = FakeTools(search_docs=[RuntimeError("boom")])
    llm = FakeLLM([turn(use("search_docs", query="q")), final("cant_tell", citations=[])])
    assert run_agent(URL, t, llm, ledger)["note"]["outcome"] == "cant_tell"


def test_per_run_cap_stops_before_call(tmp_path):
    led = Ledger(per_run_cap=0.0001, per_day_cap=10.0, path=tmp_path / "s.json")
    llm = FakeLLM([turn(use("search_docs", query="q"), i=1000, o=1000), final()])
    res = run_agent(URL, FakeTools(), llm, led)
    assert res["note"]["outcome"] == "error" and "per-run cap" in res["reason"]
    assert len(llm.seen) == 1  # second call never made


def test_per_day_cap_persisted_stops_before_first_call(tmp_path):
    p = tmp_path / "s.json"
    first = Ledger(per_run_cap=1, per_day_cap=0.001, path=p)
    first.charge(DEFAULT_MODEL, 1000, 1000)  # 0.006 USD, over the day cap
    llm = FakeLLM([final()])
    res = run_agent(URL, FakeTools(), llm, Ledger(per_run_cap=1, per_day_cap=0.001, path=p))
    assert res["note"]["outcome"] == "error" and "per-day cap" in res["reason"] and llm.seen == []


def test_unreadable_spend_file_fails_closed(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("not json")
    llm = FakeLLM([final()])
    res = run_agent(URL, FakeTools(), llm, Ledger(path=p))
    assert res["note"]["outcome"] == "error" and llm.seen == []


def test_cache_hit_carried_through(ledger):
    miss = {**fx("search.json"), "cache_hit": False}
    t = FakeTools(search_docs=[fx("search.json"), miss])
    llm = FakeLLM([turn(use("search_docs", query="a")), turn(use("search_docs", query="b")), final()])
    res = run_agent(URL, t, llm, ledger)
    assert [r["cache_hit"] for r in res["search_results"]] == [True, False]


def test_model_only_sees_what_the_thread_tool_returned(ledger):
    """Leakage: the agent never fetches comments itself; the model gets only the tool's output."""
    llm = FakeLLM([turn(use("get_issue_thread", issue_url="https://github.com/other/repo/issues/9")), final("cant_tell", citations=[])])
    t = FakeTools()
    run_agent(URL, t, llm, ledger)
    assert t.calls == [("get_issue_thread", URL)]  # model-supplied URL is ignored
    msgs = llm.seen[1]
    assert msgs[0]["content"] == f"Triage this issue: {URL}"
    assert msgs[2]["content"][0]["content"] == json.dumps(fx("thread.json"))
    assert "SYNTHETIC-COMMENT-9c1d" in msgs[2]["content"][0]["content"]  # only the returned comment, nothing else
