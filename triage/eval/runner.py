"""Eval runner: cases in, EvalResults out. No network; the Runner does all the work.

Reads case files from whatever path it is given. It never writes anywhere.
"""
import json
import math
from typing import Callable, Literal, NamedTuple

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


def doc_correct(valid_cited: list[str], gold: list[str], match: Match) -> bool:
    """Pass only citations this run's search results returned (the caller filters them).

    No gold (unanswerable) is "not applicable", stored as False; summarise() only counts
    answerable cases.
    "any":   at least one gold page is among the valid citations.
    "exact": the set of valid citations equals the set of gold pages (nothing extra, nothing missing).
    """
    if not gold:
        return False
    if match == "any":
        return bool(set(valid_cited) & set(gold))
    if match == "exact":
        return set(valid_cited) == set(gold)
    raise ValueError(f"match must be 'any' or 'exact', got {match!r}")


class EvalRun(NamedTuple):
    results: list[EvalResult]
    search_results_by_case: dict[str, list[SearchResult]]
    errors: dict[str, str]  # case_id -> exception message or contract problems; kept out of EvalResult (frozen)


def _failed_result(case: EvalCase, system: str, backend: str) -> EvalResult:
    return {"case_id": case["id"], "system": system, "backend": backend, "outcome": "error",
            "category": None, "cited_docs": [], "correct_category": False,
            "correct_doc_cited": False, "steps": 0, "cost_usd": 0.0, "failed": True}


def _is_real(x, *, integer: bool) -> bool:
    """Finite, non-negative int/float; bool is rejected (it is an int subclass)."""
    if isinstance(x, bool):
        return False
    if integer:
        return isinstance(x, int) and x >= 0
    return isinstance(x, (int, float)) and math.isfinite(x) and x >= 0


def _score(case: EvalCase, note: TriageNote, sr: list[SearchResult], system: str,
           backend: str, match: Match) -> tuple[EvalResult, list[str]]:
    """Returns (result, contract problems). Raises ValueError on a note with bad steps/cost,
    so summarise never meets a string or NaN."""
    if not _is_real(note["steps"], integer=True):
        raise ValueError(f"steps must be an int >= 0, got {note['steps']!r}")
    if not _is_real(note["cost_usd"], integer=False):
        raise ValueError(f"cost_usd must be a finite number >= 0, got {note['cost_usd']!r}")
    cited = _cited(note)  # cited_docs is what the note CITED, valid or not
    returned = {p["source"] for r in sr for p in r["passages"]}
    problems = contract_problems(note, sr)
    # A hallucinated gold citation must not score as a correct doc.
    valid = [d for d in cited if d in returned]
    return {
        "case_id": case["id"], "system": system, "backend": backend,
        "outcome": note["outcome"], "category": note["category"], "cited_docs": cited,
        "correct_category": note["category"] == case["category"],
        "correct_doc_cited": doc_correct(valid, case["gold_docs"], match),
        "steps": note["steps"], "cost_usd": note["cost_usd"],
        "failed": note["outcome"] in ("error", "step_cap") or bool(problems),
    }, problems


def run_eval(cases: list[EvalCase], runner: Runner, system: Literal["agent", "baseline"],
             backend: str, *, match: Match) -> EvalRun:
    """Run every case. Anything that goes wrong for one case (runner raising, malformed note
    or search result) becomes a failed 'error' result plus an entry in `errors`; never an abort.

    `match` is required with no default: the owner has not decided yet.
    """
    if match not in ("any", "exact"):
        raise ValueError(f"match must be 'any' or 'exact', got {match!r}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids in cases")
    results, searches, errors = [], {}, {}
    for case in cases:
        try:
            note, sr = runner(case)
            result, problems = _score(case, note, sr, system, backend, match)
            results.append(result)
            searches[case["id"]] = sr
            if problems:  # keep the reason a contract failure failed
                errors[case["id"]] = "contract: " + "; ".join(problems)
        except Exception as e:  # noqa: BLE001 - one bad case must not end the eval
            results.append(_failed_result(case, system, backend))
            searches[case["id"]] = []
            errors[case["id"]] = f"{type(e).__name__}: {e}"
    return EvalRun(results, searches, errors)
