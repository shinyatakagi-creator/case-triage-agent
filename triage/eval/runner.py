"""Eval runner: cases in, EvalResults out. No network; the Runner does all the work.

Reads case files from whatever path it is given. It never writes anywhere.
"""
import json
from typing import Callable, Literal

from triage.contracts import (
    BACKENDS, EvalCase, EvalResult, SearchResult, TriageNote, contract_problems,
)

# Every search_docs result from the run rides along: contract_problems needs it to
# validate citations, and search_stats needs it for cost/latency/cache counts.
Runner = Callable[[EvalCase], tuple[TriageNote, list[SearchResult]]]

Match = Literal["any", "exact"]


def load_cases(path: str) -> list[EvalCase]:
    """Read a JSONL of EvalCase. Category is any string; the set is the owner's."""
    cases, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from None
            _check_row(row, f"{path}:{n}")
            if row["id"] in seen:
                raise ValueError(f"{path}:{n}: duplicate id {row['id']!r}")
            seen.add(row["id"])
            cases.append(row)
    return cases


def _check_row(row, where: str) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: row is not an object")
    for key, typ in (("id", str), ("issue_url", str), ("answerable", bool),
                     ("category", str), ("gold_docs", list)):
        if key not in row:
            raise ValueError(f"{where}: missing key {key!r}")
        if not isinstance(row[key], typ):
            raise ValueError(f"{where}: {key!r} must be {typ.__name__}")
    if not all(isinstance(d, str) for d in row["gold_docs"]):
        raise ValueError(f"{where}: gold_docs must be a list of strings")
    if "notes" in row and not isinstance(row["notes"], str):
        raise ValueError(f"{where}: 'notes' must be str")
    if row["answerable"] == (not row["gold_docs"]):
        raise ValueError(f"{where}: answerable must be true exactly when gold_docs is non-empty")


def _cited(note: TriageNote) -> list[str]:
    # dict.fromkeys: de-duplicate, keep citation order.
    return list(dict.fromkeys(c["source"] for c in note["citations"]))


def doc_correct(cited: list[str], gold: list[str], match: Match) -> bool:
    """Not applicable (no gold) is stored as False; summarise() only counts answerable cases.

    "any":   at least one gold page is among the cited pages.
    "exact": the set of cited pages equals the set of gold pages (nothing extra, nothing missing).
    """
    if not gold:
        return False
    if match == "any":
        return bool(set(cited) & set(gold))
    if match == "exact":
        return set(cited) == set(gold)
    raise ValueError(f"match must be 'any' or 'exact', got {match!r}")


def _failed_result(case: EvalCase, system: str, backend: str) -> EvalResult:
    return {"case_id": case["id"], "system": system, "backend": backend, "outcome": "error",
            "category": None, "cited_docs": [], "correct_category": False,
            "correct_doc_cited": False, "steps": 0, "cost_usd": 0.0, "failed": True}


def run_eval(cases: list[EvalCase], runner: Runner, system: Literal["agent", "baseline"],
             backend: str, *, match: Match) -> tuple[list[EvalResult], dict[str, list[SearchResult]]]:
    """Run every case. A raising runner becomes a failed 'error' result, never an abort.

    Returns (results, search_results_by_case); the second feeds metrics.search_stats.
    `match` is required with no default: the owner has not decided yet.
    """
    if match not in ("any", "exact"):
        raise ValueError(f"match must be 'any' or 'exact', got {match!r}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    results, searches = [], {}
    for case in cases:
        try:
            note, sr = runner(case)
        except Exception:  # noqa: BLE001 - one bad case must not end the eval
            results.append(_failed_result(case, system, backend))
            searches[case["id"]] = []
            continue
        cited = _cited(note)
        failed = note["outcome"] in ("error", "step_cap") or bool(contract_problems(note, sr))
        results.append({
            "case_id": case["id"], "system": system, "backend": backend,
            "outcome": note["outcome"], "category": note["category"], "cited_docs": cited,
            "correct_category": note["category"] == case["category"],
            "correct_doc_cited": doc_correct(cited, case["gold_docs"], match),
            "steps": note["steps"], "cost_usd": note["cost_usd"], "failed": failed,
        })
        searches[case["id"]] = sr
    return results, searches
