import json
import re
from pathlib import Path

import pytest
from agent_helpers import SRC, FakeTools, final

from scripts import run_eval as re_
from scripts.live_one import LoggingTools
from triage.eval.runner import load_cases
from triage.llm import DEFAULT_MODEL, Ledger

CASES = load_cases(str(Path(__file__).parent / "fixtures/eval_fake/cases.jsonl"))
CASES[0]["gold_docs"] = [SRC]  # one case whose gold page the fake search returns


class OneShotLLM:
    """Answers on its first turn, every time (agent and baseline both end in one model call)."""
    model = DEFAULT_MODEL

    def __init__(self, outcome="answered"):
        self.outcome, self.calls = outcome, 0

    def complete(self, system, messages, tools):
        self.calls += 1
        if self.outcome == "answered":
            return final(category="howto")
        return final("cant_tell", citations=[])


def ledgers(tmp_path, day=10.0):
    return lambda: Ledger(per_run_cap=1.0, per_day_cap=day, path=tmp_path / "spend.json")


def go(tmp_path, repeats, llm=None, **kw):
    runs = tmp_path / "runs.jsonl"
    llm = llm or OneShotLLM()
    got = re_.execute(CASES, repeats, runs, llm, kw.pop("make_ledger", ledgers(tmp_path)),
                      make_tools=lambda: LoggingTools(FakeTools()), log=lambda *_: None)
    return got, re_.read_runs(runs)


def test_every_run_is_written_with_its_extras(tmp_path):
    got, runs = go(tmp_path, 2)
    assert got["ran"] == len(runs) == len(CASES) * 2 * 2
    r = next(x for x in runs if x["system"] == "agent")
    assert {"repeat", "queries", "reason", "latency_s", "searches", "cited_docs", "steps", "cost_usd", "outcome",
            "category", "model", "tool_errors"} <= set(r)
    base = next(x for x in runs if x["system"] == "baseline")
    assert len(base["queries"]) == 1 and base["searches"][0]["cache_hit"] is True


def test_resume_skips_what_is_written(tmp_path):
    go(tmp_path, 1)
    llm = OneShotLLM()
    got, runs = go(tmp_path, 2, llm=llm)
    assert got["skipped"] == len(CASES) * 2 and got["ran"] == len(CASES) * 2
    keys = [(r["case_id"], r["system"], r["repeat"]) for r in runs]
    assert len(keys) == len(set(keys))


def test_stops_at_the_day_cap_and_keeps_what_ran(tmp_path):
    # a ledger whose day total is already over the cap refuses the very first run
    (tmp_path / "spend.json").write_text(json.dumps({"date": Ledger().day, "usd": 5.0}))
    got, runs = go(tmp_path, 1, make_ledger=ledgers(tmp_path, day=1.0))
    assert got["ran"] == 0 and "per-day cap" in got["stopped"] and runs == []


def test_report_mean_range_and_incomplete_repeat(tmp_path):
    _, runs = go(tmp_path, 3)
    runs = [r for r in runs if not (r["repeat"] == 3 and r["system"] == "baseline")]  # repeat 3 incomplete
    text = re_.report(runs, CASES)
    assert "complete repeats (both systems, all 4 cases): [1, 2]" in text and "repeats [3] are incomplete" in text
    assert "mean " in text and "range " in text and "== agent" in text and "== baseline" in text
    assert "repeat 1: category" in text and "repeat 2: correct doc" in text and "repeat 3:" not in text
    assert re.search(r"category accuracy, answerable:.*\d/3, \d/3", text)  # counts next to the %, 3 answerable cases


def test_reason_on_a_non_error_outcome_is_a_note_not_a_failure(tmp_path):
    class CategoryOnCantTell(OneShotLLM):
        def complete(self, *a):
            self.calls += 1
            return final("cant_tell", citations=[], category="howto")  # category is dropped with a note

    _, runs = go(tmp_path, 1, llm=CategoryOnCantTell())
    assert all(r["outcome"] == "cant_tell" and r["reason"] and not r["failed"] for r in runs)
    assert f"notes (reason on a non-error outcome, not failures): {len(CASES)} of {len(CASES)} runs" in re_.report(runs, CASES)


def test_results_cannot_go_under_data_eval():
    with pytest.raises(ValueError):
        re_.guard_out_path("data/eval/runs.jsonl")
