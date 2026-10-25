"""Eval harness tests. FAKE cases only (tests/fixtures/eval_fake); nothing from data/.

Shared helpers live in tests/conftest.py (fixtures `ev`, `eval_cases`).
"""
import json
from math import comb

import pytest

from triage.eval import metrics
from triage.eval.metrics import (_ci, compare, format_table, search_stats, sign_test_p,
                                 summarise, verdict)
from triage.eval.runner import doc_correct, load_cases, run_eval


def test_load_cases_fixture(eval_cases):
    assert [c["id"] for c in eval_cases] == ["fake-1", "fake-2", "fake-3", "fake-4"]


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


def test_load_cases_rejects_bad_json_and_duplicates(tmp_path, ev):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_cases(str(p))
    good = open(ev.fixture_path).read().splitlines()[0]
    p.write_text(good + "\n" + good + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(str(p))


def test_run_eval_rejects_duplicate_case_ids(eval_cases, ev):
    with pytest.raises(ValueError, match="duplicate"):
        run_eval(eval_cases + eval_cases[:1], ev.runner(ev.agent_table()), "agent", "es", match="any")


def test_doc_correct_modes():
    assert doc_correct(["a", "x"], ["a", "b"], "any")
    assert not doc_correct(["a", "x"], ["a", "b"], "exact")
    assert doc_correct(["b", "a"], ["a", "b"], "exact")
    assert not doc_correct(["x"], ["a"], "any")
    assert not doc_correct(["a"], [], "any")  # n/a stored as False


def test_match_is_required(eval_cases, ev):
    r = ev.runner(ev.agent_table())
    with pytest.raises(TypeError):
        run_eval(eval_cases, r, "agent", "es")  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        run_eval(eval_cases, r, "agent", "es", match="loose")  # type: ignore[arg-type]


def test_run_eval_fields_and_dedup(eval_cases, ev):
    run = run_eval(eval_cases, ev.runner(ev.agent_table()), "agent", "es", match="any")
    r = ev.by_id(run.results)
    assert r["fake-1"]["cited_docs"] == ["fake/doc-one"]  # de-duplicated
    assert r["fake-1"]["correct_category"] and r["fake-1"]["correct_doc_cited"]
    assert not r["fake-2"]["correct_category"]
    assert r["fake-4"]["correct_doc_cited"] is False  # unanswerable: n/a stored as False
    assert not any(x["failed"] for x in run.results) and run.errors == {}
    assert r["fake-1"]["system"] == "agent" and r["fake-1"]["backend"] == "es"
    assert run.search_results_by_case["fake-1"] == ev.agent_table()["fake-1"][1]


def test_exact_vs_any_on_two_gold_pages(eval_cases, ev):
    r = ev.runner(ev.agent_table())
    any_ = ev.by_id(run_eval(eval_cases, r, "agent", "es", match="any").results)
    exact = ev.by_id(run_eval(eval_cases, r, "agent", "es", match="exact").results)
    assert any_["fake-2"]["correct_doc_cited"] is True
    assert exact["fake-2"]["correct_doc_cited"] is False  # only 1 of 2 gold pages


def test_hallucinated_gold_citation_is_not_correct_and_fails(eval_cases, ev):
    t = ev.agent_table()
    # cites the gold page, but this run's search never returned it
    t["fake-1"] = (ev.note("answered", "A", ["fake/doc-one"], 2, 0.01), [ev.sr(["fake/other"])])
    r = ev.by_id(run_eval(eval_cases, ev.runner(t), "agent", "es", match="any").results)["fake-1"]
    assert r["failed"] is True
    assert r["correct_doc_cited"] is False
    assert r["cited_docs"] == ["fake/doc-one"]  # cited_docs keeps what the note cited


def test_valid_gold_plus_invalid_extra_under_exact(eval_cases, ev):
    t = ev.agent_table()
    t["fake-1"] = (ev.note("answered", "A", ["fake/doc-one", "fake/ghost"], 2, 0.01), [ev.sr(["fake/doc-one"])])
    r = ev.by_id(run_eval(eval_cases, ev.runner(t), "agent", "es", match="exact").results)["fake-1"]
    assert r["failed"] is True and r["correct_doc_cited"] is True  # exact applies to VALID citations


@pytest.mark.parametrize("outcome", ["error", "step_cap"])
def test_error_and_step_cap_are_failures(eval_cases, ev, outcome):
    t = ev.agent_table()
    t["fake-3"] = (ev.note(outcome, None, [], 6, 0.05), [])
    assert ev.by_id(run_eval(eval_cases, ev.runner(t), "agent", "es", match="any").results)["fake-3"]["failed"]


def test_contract_violations_fail_through_runner(eval_cases, ev):
    t = ev.agent_table()
    t["fake-1"] = (ev.note("answered", "A", ["fake/doc-one"], 7, 0.01), [ev.sr(["fake/doc-one"])])  # steps > MAX_STEPS
    t["fake-2"] = (ev.note("answered", "B", [], 2, 0.01), [])  # answered, no citations
    r = ev.by_id(run_eval(eval_cases, ev.runner(t), "agent", "es", match="any").results)
    assert r["fake-1"]["failed"] is True and r["fake-2"]["failed"] is True
    assert r["fake-3"]["failed"] is False


def test_runner_exception_recorded_not_raised(eval_cases, ev):
    t = ev.agent_table()

    def boom(case):
        if case["id"] == "fake-2":
            raise RuntimeError("kaboom")
        return t[case["id"]]
    run = run_eval(eval_cases, boom, "agent", "es", match="any")
    assert len(run.results) == 4
    r = ev.by_id(run.results)["fake-2"]
    assert r["outcome"] == "error" and r["failed"] and r["category"] is None
    assert not ev.by_id(run.results)["fake-1"]["failed"]
    assert run.search_results_by_case["fake-2"] == []
    assert run.errors == {"fake-2": "RuntimeError: kaboom"}


def test_malformed_note_is_error_result(eval_cases, ev):
    t = ev.agent_table()
    bad = ev.note("answered", "A", ["fake/doc-one"], 2, 0.01)
    del bad["citations"]
    t["fake-1"] = (bad, [ev.sr(["fake/doc-one"])])
    t["fake-2"] = ({"outcome": "answered"}, [])  # almost everything missing
    run = run_eval(eval_cases, ev.runner(t), "agent", "es", match="any")
    r = ev.by_id(run.results)
    for cid in ("fake-1", "fake-2"):
        assert r[cid]["outcome"] == "error" and r[cid]["failed"] and r[cid]["category"] is None
        assert "KeyError" in run.errors[cid]
    assert not r["fake-3"]["failed"] and len(run.results) == 4


def test_summarise_hand_computed(eval_cases, ev):
    run = run_eval(eval_cases, ev.runner(ev.agent_table()), "agent", "es", match="any")
    s = summarise(run.results, eval_cases)
    # category: answerable fake-1 ok, fake-2 wrong, fake-3 ok -> 2/3; fake-4 (unanswerable) 1/1 separately
    assert (s["category_accuracy"]["k"], s["category_accuracy"]["n"]) == (2, 3)
    assert (s["category_accuracy_unanswerable"]["k"], s["category_accuracy_unanswerable"]["n"]) == (1, 1)
    assert (s["correct_doc_cited"]["k"], s["correct_doc_cited"]["n"]) == (2, 3)
    assert s["steps"]["mean"] == 2.5 and s["steps"]["median"] == 2.5  # 2,4,3,1
    assert s["cost_usd"]["mean"] == pytest.approx(0.015)  # 0.06 / 4
    assert (s["failure_rate"]["k"], s["failure_rate"]["n"]) == (0, 4)
    assert s["failed_count"] == 0 and s["failed_cost_usd_total"] == 0
    assert (s["cant_tell_answerable"]["k"], s["cant_tell_answerable"]["n"]) == (1, 3)
    assert (s["cant_tell_unanswerable"]["k"], s["cant_tell_unanswerable"]["n"]) == (1, 1)
    lo, hi = s["category_accuracy"]["ci_pct"]
    assert lo <= 200 / 3 <= hi
    ex = run_eval(eval_cases, ev.runner(ev.agent_table()), "agent", "es", match="exact").results
    assert summarise(ex, eval_cases)["correct_doc_cited"]["k"] == 1  # fake-1 only


def test_summarise_excludes_failed_from_cost_and_steps(eval_cases, ev):
    t = ev.agent_table()
    t["fake-2"] = (ev.note("step_cap", None, [], 6, 0.05), [])
    run = run_eval(eval_cases, ev.runner(t), "agent", "es", match="any")
    s = summarise(run.results, eval_cases)
    assert s["steps"]["n"] == 3 and s["steps"]["mean"] == 2.0  # (2+3+1)/3
    assert s["cost_usd"]["mean"] == pytest.approx(0.01)  # (0.01+0.02+0)/3
    assert s["failed_count"] == 1 and s["failed_cost_usd_total"] == pytest.approx(0.05)
    assert (s["failure_rate"]["k"], s["failure_rate"]["n"]) == (1, 4)


def test_summarise_uses_answerable_flag_not_gold_docs(ev):
    cs = [{"id": "z", "issue_url": "u", "answerable": False, "category": "A", "gold_docs": ["d"]}]
    res = [{"case_id": "z", "system": "agent", "backend": "es", "outcome": "cant_tell", "category": "A",
            "cited_docs": [], "correct_category": True, "correct_doc_cited": False, "steps": 1,
            "cost_usd": 0.0, "failed": False}]
    s = summarise(res, cs)
    assert s["category_accuracy"]["n"] == 0 and s["cant_tell_unanswerable"]["n"] == 1


def test_summarise_zero_answerable_and_empty(ev):
    cs = [{"id": "u", "issue_url": "u", "answerable": False, "category": "A", "gold_docs": []}]
    res = [{"case_id": "u", "system": "agent", "backend": "es", "outcome": "answered", "category": "A",
            "cited_docs": [], "correct_category": True, "correct_doc_cited": False, "steps": 2,
            "cost_usd": 0.1, "failed": False}]
    s = summarise(res, cs)
    assert s["correct_doc_cited"]["n"] == 0 and s["correct_doc_cited"]["pct"] is None
    assert "n/a (0/0)" in format_table(s)
    e = summarise([], [])
    assert e["n_cases"] == 0 and e["steps"]["mean"] is None and e["failure_rate"]["pct"] is None
    assert "n/a" in format_table(e)


def test_format_table_has_counts(eval_cases, ev):
    run = run_eval(eval_cases, ev.runner(ev.agent_table()), "agent", "es", match="any")
    t = format_table(summarise(run.results, eval_cases))
    assert "2/3 = 66.7%" in t and "0 of 4" in t and t.startswith("| Metric")


# --- bootstrap -------------------------------------------------------------

def test_ci_known_values(monkeypatch):
    """N=20 resamples; resample i has mean i/2 -> sorted means 0.0 .. 9.5.
    a = 0.025: lo index int(0.025*20) = 0 -> 0.0; hi index int(0.975*20) - 1 = 19 - 1 = 18 -> 9.0.
    """
    class Fake:
        def __init__(self, seed): self.i = 0
        def choices(self, values, k):
            self.i += 1
            return [self.i - 1] + [0] * (k - 1)
    monkeypatch.setattr(metrics, "N_RESAMPLES", 20)
    monkeypatch.setattr(metrics.random, "Random", Fake)
    assert _ci([0.0, 1.0]) == (0.0, 9.0)


def test_ci_degenerate_and_empty():
    assert _ci([1.0] * 5) == (1.0, 1.0)
    assert _ci([]) is None


# --- compare + verdict -----------------------------------------------------

def test_sign_test_by_hand():
    assert sign_test_p(3, 0) == 2 * (1 / 8)  # 0.25
    assert sign_test_p(1, 0) == 1.0          # 2 * 1/2
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(30, 0) == pytest.approx(2 / 2**30)
    assert sign_test_p(8, 2) == pytest.approx(2 * (comb(10, 0) + comb(10, 1) + comb(10, 2)) / 2**10)  # 112/1024


def test_one_case_gap_is_no_clear_difference(ev):
    cs, a, b = ev.synth(40, set(range(21)), set(range(20)))
    c = compare(a, b, cs)["category"]
    assert (c["agent_only"], c["baseline_only"], c["ties"]) == (1, 0, 39)
    v = verdict(c, "category")
    assert "no clear difference" in v and "1 of 40 cases differ" in v


def test_three_of_three_is_not_clear(ev):
    # bootstrap CI excludes zero here (3 wins of 3 pairs, rest ties would not, so use n=3), but p = 0.25
    cs, a, b = ev.synth(3, {0, 1, 2}, set())
    c = compare(a, b, cs)["category"]
    assert c["ci"][0] > 0                      # the CI alone would call this a win
    assert c["sign_p"] == 0.25
    assert "no clear difference" in verdict(c, "category")


def test_large_gap_is_clear_win(ev):
    cs, a, b = ev.synth(40, set(range(40)), set(range(10)))
    c = compare(a, b, cs)["correct_doc"]
    assert (c["agent_only"], c["baseline_only"], c["ties"]) == (30, 0, 10)
    assert "agent clearly better" in verdict(c, "doc")


def test_large_loss_is_reported(ev):
    cs, a, b = ev.synth(40, set(range(10)), set(range(40)))
    assert "baseline clearly better" in verdict(compare(a, b, cs)["category"], "category")


def test_zero_discordant_is_no_clear_difference(ev):
    cs, a, b = ev.synth(5, {0, 1}, {0, 1})
    v = verdict(compare(a, b, cs)["category"], "category")
    assert "no clear difference" in v and "no case out of 5 differs" in v.lower()


def test_compare_doc_and_category_use_answerable_only(eval_cases, ev):
    run = run_eval(eval_cases, ev.runner(ev.agent_table()), "agent", "es", match="any")
    c = compare(run.results, run.results, eval_cases)
    assert c["correct_doc"]["n"] == 3 and c["category"]["n"] == 3


def test_compare_rejects_mismatched_and_duplicate_ids(ev):
    cs, a, b = ev.synth(4, {0}, {1})
    with pytest.raises(ValueError, match="different case ids"):
        compare(a, b[:-1], cs)
    with pytest.raises(ValueError, match="duplicate"):
        compare(a + a[:1], b + b[:1], cs)


# --- search_stats ----------------------------------------------------------

def test_search_stats_ignores_cached_calls(ev):
    by_case = {
        "a": [ev.sr(["x"], cost=0.002, lat=100.0), ev.sr(["x"], cost=0.0, lat=5.0, cache_hit=True)],
        "b": [ev.sr(["y"], cost=0.004, lat=300.0)],
    }
    s = search_stats(by_case)
    assert (s["calls"], s["uncached_calls"], s["cached_calls"], s["unknown_cache_status"]) == (3, 2, 1, 0)
    assert s["cost_usd_total"] == pytest.approx(0.006)
    assert s["cost_usd_mean"] == pytest.approx(0.003)
    assert s["latency_ms_mean"] == 200.0  # the 5 ms cached call is excluded
    assert s["latency_ms_median"] == 200.0


def test_search_stats_missing_cache_hit_is_unknown_not_uncached(ev):
    broken = ev.sr(["x"], cost=9.0, lat=9999.0)
    del broken["cache_hit"]
    s = search_stats({"a": [ev.sr(["x"], cost=0.002, lat=100.0), broken]})
    assert (s["calls"], s["uncached_calls"], s["cached_calls"], s["unknown_cache_status"]) == (2, 1, 0, 1)
    assert s["cost_usd_total"] == pytest.approx(0.002) and s["latency_ms_mean"] == 100.0


def test_search_stats_empty():
    s = search_stats({})
    assert s["calls"] == 0 and s["cost_usd_mean"] is None


# --- review round 2 --------------------------------------------------------

@pytest.mark.parametrize("steps,cost", [
    (2, "0.1"), (2, float("nan")), (2, float("inf")), (2, -0.5), (2, True),
    (True, 0.1), (-1, 0.1), (2.0, 0.1), ("2", 0.1),
])
def test_bad_steps_or_cost_become_error_result(eval_cases, ev, steps, cost):
    t = ev.agent_table()
    n = ev.note("answered", "A", ["fake/doc-one"], 2, 0.01)
    n["steps"], n["cost_usd"] = steps, cost
    t["fake-1"] = (n, [ev.sr(["fake/doc-one"])])
    run = run_eval(eval_cases, ev.runner(t), "agent", "es", match="any")
    r = ev.by_id(run.results)["fake-1"]
    assert r["outcome"] == "error" and r["failed"] and r["category"] is None
    assert r["cost_usd"] == 0.0 and r["steps"] == 0
    assert "ValueError" in run.errors["fake-1"]
    summarise(run.results, eval_cases)  # must not break


def test_contract_problem_messages_recorded(eval_cases, ev):
    t = ev.agent_table()
    t["fake-1"] = (ev.note("answered", "A", ["fake/ghost"], 2, 0.01), [ev.sr(["fake/doc-one"])])
    run = run_eval(eval_cases, ev.runner(t), "agent", "es", match="any")
    assert "fake/ghost" in run.errors["fake-1"] and run.errors["fake-1"].startswith("contract:")
    assert "fake-2" not in run.errors


def test_verdict_checks_disagree_sign_significant_ci_includes_zero():
    p = {"n": 10, "agent_only": 6, "baseline_only": 0, "ties": 4, "ci": (-0.1, 0.5),
         "sign_p": sign_test_p(6, 0)}  # 2/64 = 0.03125
    v = verdict(p, "x")
    assert "no clear difference" in v and "disagree" in v and "includes zero" in v
    assert "6 of 10 cases differ" in v
    assert "p=0.031" in v and "within noise" not in v


def test_verdict_checks_disagree_ci_excludes_zero_sign_not_significant():
    p = {"n": 3, "agent_only": 3, "baseline_only": 0, "ties": 0, "ci": (1.0, 1.0),
         "sign_p": sign_test_p(3, 0)}
    v = verdict(p, "x")
    assert "disagree" in v and "excludes zero" in v and "p=0.250" in v and "not below" in v


def test_verdict_wording_all_cases_differ(ev):
    cs, a, b = ev.synth(3, {0, 1, 2}, set())
    v = verdict(compare(a, b, cs)["category"], "c")
    assert "Only" not in v and "disagree" in v


def test_search_stats_rejects_bad_values(ev):
    good = ev.sr(["x"], cost=0.002, lat=100.0)
    bads = [ev.sr(["x"], cost=float("nan")), ev.sr(["x"], lat=float("inf")), ev.sr(["x"], cost=-1.0),
            ev.sr(["x"], lat=-5.0), ev.sr(["x"], cost=True)]
    s = search_stats({"a": [good] + bads})
    assert (s["calls"], s["uncached_calls"], s["unknown_cache_status"]) == (6, 1, 5)
    assert s["cost_usd_total"] == pytest.approx(0.002)


def test_compare_rejects_unknown_ids_and_missing_answerable(ev):
    cs, a, b = ev.synth(4, {0}, {1})
    with pytest.raises(ValueError, match="not in cases"):
        compare(a, b, cs[:3])
    with pytest.raises(ValueError, match="missing from both"):
        compare(a[:3], b[:3], cs)


def test_huge_int_cost_does_not_raise(eval_cases, ev):
    s = search_stats({"a": [ev.sr(["x"], cost=10**400)]})
    assert s["unknown_cache_status"] == 1 and s["uncached_calls"] == 0
    t = ev.agent_table()
    n = ev.note("answered", "A", ["fake/doc-one"], 2, 10**400)
    t["fake-1"] = (n, [ev.sr(["fake/doc-one"])])
    run = run_eval(eval_cases, ev.runner(t), "agent", "es", match="any")
    r = ev.by_id(run.results)["fake-1"]
    assert r["outcome"] == "error" and r["failed"] and "ValueError" in run.errors["fake-1"]
