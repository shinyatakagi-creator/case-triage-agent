"""Eval metrics. Stdlib only. Every rate carries its count (k/n) next to the percentage.

Bootstrap: percentile interval over cases, N_RESAMPLES resamples, fixed SEED, 95%.
With 40 cases the intervals are wide on purpose; that is the point.

Known limitation (accepted): valid finite costs near 1e308 can sum to inf in the cost totals.
"""
import math
import random
import statistics

from triage.contracts import EvalCase, EvalResult, SearchResult

N_RESAMPLES = 10_000
SEED = 0
CONFIDENCE = 0.95
ALPHA = 0.05  # sign-test threshold for "clear"


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
    ci = _ci([float(f) for f in flags])
    return {"k": k, "n": n, "pct": 100 * k / n if n else None,
            "ci_pct": tuple(100 * x for x in ci) if ci else None}


def _num(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": statistics.mean(xs) if xs else None,
            "median": statistics.median(xs) if xs else None}


def summarise(results: list[EvalResult], cases: list[EvalCase]) -> dict:
    """Rates (category, doc, can't tell, failure) use ALL cases, failed runs included: a failed
    run is a miss, and dropping it would flatter the system. Only the cost and steps
    averages exclude failed runs, because a crashed or capped run has no meaningful cost
    or step count; they are reported separately.
    """
    by_id = {c["id"]: c for c in cases}
    rows = [(r, by_id[r["case_id"]]) for r in results]  # KeyError on unknown id is deliberate
    ans = [(r, c) for r, c in rows if c["answerable"]]
    unans = [(r, c) for r, c in rows if not c["answerable"]]
    ok = [r for r, _ in rows if not r["failed"]]
    bad = [r for r, _ in rows if r["failed"]]
    cant = lambda grp: _rate([r["outcome"] == "cant_tell" for r, _ in grp])  # noqa: E731
    return {
        "n_cases": len(rows),
        # headline: answerable only (the owner's category is only meaningful there)
        "category_accuracy": _rate([r["correct_category"] for r, _ in ans]),
        "category_accuracy_unanswerable": _rate([r["correct_category"] for r, _ in unans]),
        # for unanswerable cases correct_doc_cited is stored False = n/a, so answerable only
        "correct_doc_cited": _rate([r["correct_doc_cited"] for r, _ in ans]),
        # failed runs are excluded from these two; they are reported below instead
        "steps": _num([r["steps"] for r in ok]),
        "cost_usd": _num([r["cost_usd"] for r in ok]),
        "failed_count": len(bad),
        # errored runs record cost 0.0, so this is a lower bound
        "failed_cost_usd_total": sum(r["cost_usd"] for r in bad),
        "failure_rate": _rate([r["failed"] for r, _ in rows]),
        "cant_tell_answerable": cant(ans),
        "cant_tell_unanswerable": cant(unans),
    }


def sign_test_p(a: int, b: int) -> float:
    """Exact two-sided sign test on discordant pairs (a wins vs b wins), H0: p = 0.5."""
    d = a + b
    if d == 0:
        return 1.0
    tail = sum(math.comb(d, i) for i in range(min(a, b) + 1)) / 2**d
    return min(1.0, 2 * tail)


def _paired(a: dict[str, bool], b: dict[str, bool]) -> dict:
    ids = sorted(a)
    diffs = [int(a[i]) - int(b[i]) for i in ids]
    w, l = diffs.count(1), diffs.count(-1)
    return {"n": len(ids), "agent_only": w, "baseline_only": l, "ties": diffs.count(0),
            "mean_diff": sum(diffs) / len(diffs) if diffs else None,
            "ci": _ci([float(d) for d in diffs]), "sign_p": sign_test_p(w, l)}


def compare(agent_results: list[EvalResult], baseline_results: list[EvalResult],
            cases: list[EvalCase]) -> dict:
    """Paired per-case wins. Raises ValueError on duplicate or mismatched case ids.

    Category and doc comparisons both use answerable cases only.
    """
    for name, rs in (("agent", agent_results), ("baseline", baseline_results)):
        ids = [r["case_id"] for r in rs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate case_id in {name} results")
    if {r["case_id"] for r in agent_results} != {r["case_id"] for r in baseline_results}:
        raise ValueError("agent and baseline results cover different case ids")
    case_ids = {c["id"] for c in cases}
    result_ids = {r["case_id"] for r in agent_results}
    if result_ids - case_ids:
        raise ValueError(f"result case ids not in cases: {sorted(result_ids - case_ids)}")
    answerable = {c["id"] for c in cases if c["answerable"]}
    if answerable - result_ids:
        raise ValueError(f"answerable cases missing from both result lists: {sorted(answerable - result_ids)}")

    def flags(rs, key):
        return {r["case_id"]: r[key] for r in rs if r["case_id"] in answerable}
    return {
        "category": _paired(flags(agent_results, "correct_category"),
                            flags(baseline_results, "correct_category")),
        "correct_doc": _paired(flags(agent_results, "correct_doc_cited"),
                               flags(baseline_results, "correct_doc_cited")),
    }


def verdict(paired: dict, label: str) -> str:
    """Clear only if the bootstrap CI excludes zero AND the sign test has p < ALPHA."""
    ci, w, l, n, p = paired["ci"], paired["agent_only"], paired["baseline_only"], paired["n"], paired["sign_p"]
    d = w + l
    counts = f"agent-only {w}, baseline-only {l}, ties {paired['ties']} of {n}"
    if d == 0 or ci is None:
        return f"{label}: no clear difference ({counts}). No case out of {n} differs; do not claim a win."
    ci_txt = f"bootstrap CI {ci[0]:+.3f} to {ci[1]:+.3f}"
    ci_excl = ci[0] > 0 or ci[1] < 0
    sig = p < ALPHA
    if ci_excl and sig:
        who = "agent" if ci[0] > 0 else "baseline"
        return f"{label}: {who} clearly better ({counts}; {ci_txt}; sign test p={p:.4f})."
    if ci_excl != sig:
        return (f"{label}: no clear difference ({counts}). {d} of {n} cases differ ({w} vs {l}). "
                f"The two checks disagree: "
                f"the bootstrap CI {'excludes' if ci_excl else 'includes'} zero ({ci_txt[len('bootstrap CI '):]}) but the "
                f"exact sign test p={p:.3f} is {'below' if sig else 'not below'} {ALPHA}. "
                f"Both must agree before calling it clear; do not claim a win.")
    return (f"{label}: no clear difference ({counts}). {d} of {n} cases differ ({w} vs {l}); "
            f"exact sign test p={p:.3f}, {ci_txt}, which is within noise; do not claim a win.")


def _num_ok(x) -> bool:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:  # float() raises OverflowError for ints beyond float range (e.g. 10**400)
        return math.isfinite(float(x)) and x >= 0
    except OverflowError:
        return False


def _known(r) -> bool:
    return (isinstance(r, dict) and isinstance(r.get("cache_hit"), bool)
            and _num_ok(r.get("cost_usd")) and _num_ok(r.get("latency_ms")))


def search_stats(search_results_by_case: dict[str, list[SearchResult]]) -> dict:
    """Cost and latency come from uncached calls only (cached calls report cost 0).

    A result without a bool cache_hit, or with cost/latency that is missing, bool, NaN, inf
    or negative, is malformed: counted in ONE counter, unknown_cache_status (it covers bad
    values too, not a separate unknown_value) and excluded from every other number, never assumed uncached.
    """
    calls = [r for rs in search_results_by_case.values() for r in rs]
    good = [r for r in calls if _known(r)]
    uncached = [r for r in good if not r["cache_hit"]]
    costs, lats = [r["cost_usd"] for r in uncached], [r["latency_ms"] for r in uncached]
    return {
        "calls": len(calls), "uncached_calls": len(uncached),
        "cached_calls": len(good) - len(uncached),  # = cache hits
        "unknown_cache_status": len(calls) - len(good),
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
        return "n/a (0 non-failed cases)"
    return f"mean {x['mean']:.4g}{unit}, median {x['median']:.4g}{unit} (n={x['n']} non-failed)"


def format_table(summary: dict) -> str:
    rows = [
        ("Category accuracy (answerable only)", _fmt_rate(summary["category_accuracy"])),
        ("Correct doc cited (answerable only)", _fmt_rate(summary["correct_doc_cited"])),
        ("Steps per case (failed runs excluded)", _fmt_num(summary["steps"])),
        ("Cost per case (failed runs excluded)", _fmt_num(summary["cost_usd"], " USD")),
        ("Failed runs", f"{summary['failed_count']} of {summary['n_cases']}, "
                        f"cost recorded {summary['failed_cost_usd_total']:.4g} USD"),
        ("Failure rate", _fmt_rate(summary["failure_rate"])),
        ("Can't tell, answerable", _fmt_rate(summary["cant_tell_answerable"])),
        ("Can't tell, not answerable", _fmt_rate(summary["cant_tell_unanswerable"])),
        ("(secondary) Category match, not answerable", _fmt_rate(summary["category_accuracy_unanswerable"])),
    ]
    return "\n".join(["| Metric | Result |", "|---|---|"] + [f"| {m} | {v} |" for m, v in rows])
