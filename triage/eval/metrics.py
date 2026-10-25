"""Eval metrics. Stdlib only. Every rate carries its count (k/n) next to the percentage.

Bootstrap: percentile interval over cases, N_RESAMPLES resamples, fixed SEED, 95%.
With 40 cases the intervals are wide on purpose; that is the point.
"""
import random
import statistics

from triage.contracts import EvalCase, EvalResult, SearchResult

N_RESAMPLES = 10_000
SEED = 0
CONFIDENCE = 0.95


def _ci(values: list[float]) -> tuple[float, float] | None:
    """Bootstrap percentile CI of the mean of `values` (0/1 flags or paired diffs)."""
    n = len(values)
    if n == 0:
        return None
    rng = random.Random(SEED)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(N_RESAMPLES))
    a = (1 - CONFIDENCE) / 2
    return means[int(a * N_RESAMPLES)], means[int((1 - a) * N_RESAMPLES) - 1]


def _rate(flags: list[bool]) -> dict:
    k, n = sum(flags), len(flags)
    return {"k": k, "n": n, "pct": 100 * k / n if n else None,
            "ci_pct": tuple(100 * x for x in ci) if (ci := _ci([float(f) for f in flags])) else None}


def _num(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": statistics.mean(xs) if xs else None,
            "median": statistics.median(xs) if xs else None}


def summarise(results: list[EvalResult], cases: list[EvalCase]) -> dict:
    by_id = {c["id"]: c for c in cases}
    rows = [(r, by_id[r["case_id"]]) for r in results]  # KeyError on unknown id is deliberate
    ans = [(r, c) for r, c in rows if c["gold_docs"]]
    unans = [(r, c) for r, c in rows if not c["gold_docs"]]
    cant = lambda grp: _rate([r["outcome"] == "cant_tell" for r, _ in grp])  # noqa: E731
    return {
        "n_cases": len(rows),
        "category_accuracy": _rate([r["correct_category"] for r, _ in rows]),
        # answerable only: for unanswerable cases correct_doc_cited is stored False = n/a
        "correct_doc_cited": _rate([r["correct_doc_cited"] for r, _ in ans]),
        "steps": _num([r["steps"] for r, _ in rows]),
        "cost_usd": _num([r["cost_usd"] for r, _ in rows]),
        "failure_rate": _rate([r["failed"] for r, _ in rows]),
        "cant_tell_answerable": cant(ans),
        "cant_tell_unanswerable": cant(unans),
    }


def _paired(a: dict[str, bool], b: dict[str, bool]) -> dict:
    ids = sorted(a.keys() & b.keys())
    diffs = [int(a[i]) - int(b[i]) for i in ids]
    return {"n": len(ids), "agent_only": diffs.count(1), "baseline_only": diffs.count(-1),
            "ties": diffs.count(0),
            "mean_diff": sum(diffs) / len(diffs) if diffs else None,
            "ci": _ci([float(d) for d in diffs])}


def compare(agent_results: list[EvalResult], baseline_results: list[EvalResult],
            cases: list[EvalCase]) -> dict:
    """Paired per-case wins. Doc comparison uses answerable cases only."""
    answerable = {c["id"] for c in cases if c["gold_docs"]}
    def flags(rs, key, only=None):
        return {r["case_id"]: r[key] for r in rs if only is None or r["case_id"] in only}
    return {
        "category": _paired(flags(agent_results, "correct_category"),
                            flags(baseline_results, "correct_category")),
        "correct_doc": _paired(flags(agent_results, "correct_doc_cited", answerable),
                               flags(baseline_results, "correct_doc_cited", answerable)),
    }


def verdict(paired: dict, label: str) -> str:
    """'No clear difference' unless the paired bootstrap interval excludes zero."""
    ci = paired["ci"]
    counts = (f"agent-only {paired['agent_only']}, baseline-only {paired['baseline_only']}, "
              f"ties {paired['ties']} of {paired['n']}")
    if ci is not None and ci[0] > 0:
        return f"{label}: agent clearly better ({counts}; 95% CI of mean diff {ci[0]:+.3f} to {ci[1]:+.3f})."
    if ci is not None and ci[1] < 0:
        return f"{label}: baseline clearly better ({counts}; 95% CI of mean diff {ci[0]:+.3f} to {ci[1]:+.3f})."
    return (f"{label}: no clear difference ({counts}). With this few cases a gap of "
            f"two or three is noise; do not claim a win.")


def search_stats(search_results_by_case: dict[str, list[SearchResult]]) -> dict:
    """Cost and latency come from uncached calls only (cached calls report cost 0)."""
    calls = [r for rs in search_results_by_case.values() for r in rs]
    uncached = [r for r in calls if not r["cache_hit"]]
    costs, lats = [r["cost_usd"] for r in uncached], [r["latency_ms"] for r in uncached]
    return {
        "calls": len(calls), "uncached_calls": len(uncached),
        "cached_calls": len(calls) - len(uncached), "cache_hits": len(calls) - len(uncached),
        "cost_usd_total": sum(costs),
        "cost_usd_mean": statistics.mean(costs) if costs else None,
        "latency_ms_mean": statistics.mean(lats) if lats else None,
        "latency_ms_median": statistics.median(lats) if lats else None,
    }


def _fmt_rate(r: dict) -> str:
    if r["n"] == 0:
        return "n/a (0/0)"
    lo, hi = r["ci_pct"]
    return f"{r['k']}/{r['n']} = {r['pct']:.1f}% (95% CI {lo:.1f}-{hi:.1f}%)"


def _fmt_num(x: dict, unit: str = "") -> str:
    if x["n"] == 0:
        return "n/a"
    return f"mean {x['mean']:.4g}{unit}, median {x['median']:.4g}{unit} (n={x['n']})"


def format_table(summary: dict) -> str:
    rows = [
        ("Category accuracy", _fmt_rate(summary["category_accuracy"])),
        ("Correct doc cited (answerable only)", _fmt_rate(summary["correct_doc_cited"])),
        ("Steps per case", _fmt_num(summary["steps"])),
        ("Cost per case", _fmt_num(summary["cost_usd"], " USD")),
        ("Failure rate", _fmt_rate(summary["failure_rate"])),
        ("Can't tell, answerable", _fmt_rate(summary["cant_tell_answerable"])),
        ("Can't tell, not answerable", _fmt_rate(summary["cant_tell_unanswerable"])),
    ]
    out = ["| Metric | Result |", "|---|---|"] + [f"| {m} | {v} |" for m, v in rows]
    return "\n".join(out)
