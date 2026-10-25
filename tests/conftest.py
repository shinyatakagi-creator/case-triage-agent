"""Shared helpers for tests/test_eval.py ONLY (fake cases, nothing from data/).

Exposed as fixtures `ev` (namespace of helpers) and `eval_cases` (fake-case fixture file).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from triage.eval.runner import load_cases

FIX = Path(__file__).parent / "fixtures" / "eval_fake" / "cases.jsonl"


def _sr(sources, cost=0.001, lat=100.0, cache_hit=False):
    ps = [{"text": "t", "source": s, "url": "u", "title": "T", "heading": "h",
           "chunk_id": s + "#0", "score": 1.0} for s in sources]
    return {"passages": ps, "cost_usd": cost, "latency_ms": lat, "cache_hit": cache_hit}


def _note(outcome, cat, cites, steps, cost, url="https://github.com/kubernetes/kubernetes/issues/1"):
    return {"issue_url": url, "outcome": outcome, "category": cat, "likely_cause": None,
            "citations": [{"source": s} for s in cites], "next_step": None,
            "steps": steps, "cost_usd": cost, "tool_calls": []}


def _agent_table():
    """case id -> (note, search results the run saw). Fresh dict each call."""
    return {
        "fake-1": (_note("answered", "A", ["fake/doc-one", "fake/doc-one"], 2, 0.01), [_sr(["fake/doc-one"])]),
        "fake-2": (_note("answered", "A", ["fake/doc-two"], 4, 0.03), [_sr(["fake/doc-two"])]),
        "fake-3": (_note("cant_tell", "A", [], 3, 0.02), [_sr(["fake/doc-four"])]),
        "fake-4": (_note("cant_tell", "C", [], 1, 0.0), []),
    }


def _runner(table):
    return lambda case: table[case["id"]]


def _by_id(results):
    return {r["case_id"]: r for r in results}


def _synth(n, agent_right, base_right):
    """n fake answerable cases; agent/baseline correct (category and doc) on the given index sets."""
    cs = [{"id": f"s{i}", "issue_url": f"https://github.com/kubernetes/kubernetes/issues/{i}",
           "answerable": True, "category": "A", "gold_docs": ["d"]} for i in range(n)]

    def mk(right, system):
        return [{"case_id": f"s{i}", "system": system, "backend": "es", "outcome": "answered",
                 "category": "A", "cited_docs": ["d"], "correct_category": i in right,
                 "correct_doc_cited": i in right, "steps": 1, "cost_usd": 0.0, "failed": False}
                for i in range(n)]
    return cs, mk(agent_right, "agent"), mk(base_right, "baseline")


@pytest.fixture
def ev():
    return SimpleNamespace(sr=_sr, note=_note, agent_table=_agent_table, runner=_runner,
                           by_id=_by_id, synth=_synth, fixture_path=str(FIX))


@pytest.fixture
def eval_cases():
    return load_cases(str(FIX))
