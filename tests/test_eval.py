"""Eval harness tests. FAKE cases only (tests/fixtures/eval_fake); nothing from data/."""
import json
from pathlib import Path

import pytest

from triage.eval.metrics import compare, format_table, search_stats, summarise, verdict
from triage.eval.runner import doc_correct, load_cases, run_eval

FIX = Path(__file__).parent / "fixtures" / "eval_fake" / "cases.jsonl"


def sr(sources, cost=0.001, lat=100.0, cache_hit=False):
    ps = [{"text": "t", "source": s, "url": "u", "title": "T", "heading": "h",
           "chunk_id": s + "#0", "score": 1.0} for s in sources]
    return {"passages": ps, "cost_usd": cost, "latency_ms": lat, "cache_hit": cache_hit}


def note(outcome, cat, cites, steps, cost, url="https://github.com/kubernetes/kubernetes/issues/1"):
    return {"issue_url": url, "outcome": outcome, "category": cat, "likely_cause": None,
            "citations": [{"source": s} for s in cites], "next_step": None,
            "steps": steps, "cost_usd": cost, "tool_calls": []}


# case id -> (note, search results the run saw)
AGENT = {
    "fake-1": (note("answered", "A", ["fake/doc-one", "fake/doc-one"], 2, 0.01), [sr(["fake/doc-one"])]),
    "fake-2": (note("answered", "A", ["fake/doc-two"], 4, 0.03), [sr(["fake/doc-two"])]),
    "fake-3": (note("cant_tell", "A", [], 3, 0.02), [sr(["fake/doc-four"])]),
    "fake-4": (note("cant_tell", "C", [], 1, 0.0), []),
}


def fake_runner(table):
    return lambda case: table[case["id"]]


@pytest.fixture
def cases():
    return load_cases(str(FIX))


def by_id(results):
    return {r["case_id"]: r for r in results}


def test_load_cases_fixture(cases):
    assert [c["id"] for c in cases] == ["fake-1", "fake-2", "fake-3", "fake-4"]


@pytest.mark.parametrize("row", [
    {"id": "x", "issue_url": "u", "answerable": True, "category": "A"},                      # missing key
    {"id": "x", "issue_url": "u", "answerable": "yes", "category": "A", "gold_docs": ["d"]},  # bad type
    {"id": "x", "issue_url": "u", "answerable": True, "category": "A", "gold_docs": []},      # answerable, no gold
    {"id": "x", "issue_url": "u", "answerable": False, "category": "A", "gold_docs": ["d"]},  # gold on unanswerable
])
def test_load_cases_rejects_malformed(tmp_path, row):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_cases(str(p))


def test_load_cases_rejects_bad_json_and_duplicates(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_cases(str(p))
    good = FIX.read_text().splitlines()[0]
    p.write_text(good + "\n" + good + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(str(p))


def test_doc_correct_modes():
    assert doc_correct(["a", "x"], ["a", "b"], "any")
    assert not doc_correct(["a", "x"], ["a", "b"], "exact")
    assert doc_correct(["b", "a"], ["a", "b"], "exact")
    assert not doc_correct(["x"], ["a"], "any")
    assert not doc_correct(["a"], [], "any")  # n/a stored as False


def test_match_is_required(cases):
    with pytest.raises(TypeError):
        run_eval(cases, fake_runner(AGENT), "agent", "es")  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        run_eval(cases, fake_runner(AGENT), "agent", "es", match="loose")  # type: ignore[arg-type]


def test_run_eval_fields_and_dedup(cases):
    res, _ = run_eval(cases, fake_runner(AGENT), "agent", "es", match="any")
    r = by_id(res)
    assert r["fake-1"]["cited_docs"] == ["fake/doc-one"]  # de-duplicated
    assert r["fake-1"]["correct_category"] and r["fake-1"]["correct_doc_cited"]
    assert not r["fake-2"]["correct_category"]
    assert r["fake-4"]["correct_doc_cited"] is False  # unanswerable: n/a stored as False
    assert not any(x["failed"] for x in res)
    assert r["fake-1"]["system"] == "agent" and r["fake-1"]["backend"] == "es"


def test_exact_vs_any_on_two_gold_pages(cases):
    res_any, _ = run_eval(cases, fake_runner(AGENT), "agent", "es", match="any")
    res_ex, _ = run_eval(cases, fake_runner(AGENT), "agent", "es", match="exact")
    assert by_id(res_any)["fake-2"]["correct_doc_cited"] is True
    assert by_id(res_ex)["fake-2"]["correct_doc_cited"] is False  # only 1 of 2 gold pages


def test_invalid_citation_is_failure(cases):
    table = dict(AGENT)
    table["fake-1"] = (note("answered", "A", ["fake/never-returned"], 2, 0.01), [sr(["fake/doc-one"])])
    res, _ = run_eval(cases, fake_runner(table), "agent", "es", match="any")
    assert by_id(res)["fake-1"]["failed"] is True


@pytest.mark.parametrize("outcome", ["error", "step_cap"])
def test_error_and_step_cap_are_failures(cases, outcome):
    table = dict(AGENT)
    table["fake-3"] = (note(outcome, None, [], 6, 0.05), [])
    res, _ = run_eval(cases, fake_runner(table), "agent", "es", match="any")
    assert by_id(res)["fake-3"]["failed"] is True


def test_runner_exception_recorded_not_raised(cases):
    def boom(case):
        if case["id"] == "fake-2":
            raise RuntimeError("kaboom")
        return AGENT[case["id"]]
    res, searches = run_eval(cases, boom, "agent", "es", match="any")
    assert len(res) == 4
    r = by_id(res)["fake-2"]
    assert r["outcome"] == "error" and r["failed"] and r["category"] is None
    assert not by_id(res)["fake-1"]["failed"]
    assert searches["fake-2"] == []


def test_summarise_hand_computed(cases):
    res, _ = run_eval(cases, fake_runner(AGENT), "agent", "es", match="any")
    s = summarise(res, cases)
    assert (s["category_accuracy"]["k"], s["category_accuracy"]["n"]) == (3, 4)
    assert s["category_accuracy"]["pct"] == 75.0
    assert (s["correct_doc_cited"]["k"], s["correct_doc_cited"]["n"]) == (2, 3)  # answerable only
    assert s["steps"]["mean"] == 2.5 and s["steps"]["median"] == 2.5  # 2,4,3,1
    assert s["cost_usd"]["mean"] == pytest.approx(0.015)  # 0.06 / 4
    assert (s["failure_rate"]["k"], s["failure_rate"]["n"]) == (0, 4)
    assert (s["cant_tell_answerable"]["k"], s["cant_tell_answerable"]["n"]) == (1, 3)
    assert (s["cant_tell_unanswerable"]["k"], s["cant_tell_unanswerable"]["n"]) == (1, 1)
    lo, hi = s["category_accuracy"]["ci_pct"]
    assert lo <= 75.0 <= hi
    res_ex, _ = run_eval(cases, fake_runner(AGENT), "agent", "es", match="exact")
    assert summarise(res_ex, cases)["correct_doc_cited"]["k"] == 1  # fake-1 only


def test_summarise_is_deterministic(cases):
    res, _ = run_eval(cases, fake_runner(AGENT), "agent", "es", match="any")
    assert summarise(res, cases) == summarise(res, cases)


def test_format_table_has_counts(cases):
    res, _ = run_eval(cases, fake_runner(AGENT), "agent", "es", match="any")
    t = format_table(summarise(res, cases))
    assert "3/4 = 75.0%" in t and "2/3 = 66.7%" in t and t.startswith("| Metric")


# --- compare ---------------------------------------------------------------

def synth(n, agent_right, base_right):
    """n fake answerable cases; agent/base correct on the given index sets."""
    cs = [{"id": f"s{i}", "issue_url": f"https://github.com/kubernetes/kubernetes/issues/{i}",
           "answerable": True, "category": "A", "gold_docs": ["d"]} for i in range(n)]
    def mk(right, system):
        return [{"case_id": f"s{i}", "system": system, "backend": "es", "outcome": "answered",
                 "category": "A", "cited_docs": ["d"], "correct_category": i in right,
                 "correct_doc_cited": i in right, "steps": 1, "cost_usd": 0.0, "failed": False}
                for i in range(n)]
    return cs, mk(agent_right, "agent"), mk(base_right, "baseline")


def test_compare_one_case_gap_is_no_clear_difference():
    cs, a, b = synth(40, set(range(21)), set(range(20)))
    c = compare(a, b, cs)["category"]
    assert (c["agent_only"], c["baseline_only"], c["ties"]) == (1, 0, 39)
    assert "no clear difference" in verdict(c, "category")


def test_compare_large_gap_is_clear_win():
    cs, a, b = synth(40, set(range(40)), set(range(10)))
    c = compare(a, b, cs)["correct_doc"]
    assert (c["agent_only"], c["baseline_only"], c["ties"]) == (30, 0, 10)
    assert "agent clearly better" in verdict(c, "doc")


def test_compare_large_loss_is_reported():
    cs, a, b = synth(40, set(range(10)), set(range(40)))
    assert "baseline clearly better" in verdict(compare(a, b, cs)["category"], "category")


def test_compare_doc_ignores_unanswerable(cases):
    res, _ = run_eval(cases, fake_runner(AGENT), "agent", "es", match="any")
    assert compare(res, res, cases)["correct_doc"]["n"] == 3
    assert compare(res, res, cases)["category"]["n"] == 4


# --- search_stats ----------------------------------------------------------

def test_search_stats_ignores_cached_calls():
    by_case = {
        "a": [sr(["x"], cost=0.002, lat=100.0), sr(["x"], cost=0.0, lat=5.0, cache_hit=True)],
        "b": [sr(["y"], cost=0.004, lat=300.0)],
    }
    s = search_stats(by_case)
    assert (s["calls"], s["uncached_calls"], s["cached_calls"], s["cache_hits"]) == (3, 2, 1, 1)
    assert s["cost_usd_total"] == pytest.approx(0.006)
    assert s["cost_usd_mean"] == pytest.approx(0.003)
    assert s["latency_ms_mean"] == 200.0  # the 5 ms cached call is excluded
    assert s["latency_ms_median"] == 200.0


def test_search_stats_empty():
    s = search_stats({})
    assert s["calls"] == 0 and s["cost_usd_mean"] is None
