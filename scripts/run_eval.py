"""Run the eval LIVE: agent and baseline on every case, several repeats. PAID: run only on request.

Usage (env loaded: TRIAGE_ANTHROPIC_API_KEY, GITHUB_TOKEN; rag-hybrid up):
    python -m scripts.run_eval [--cases data/eval/eval.jsonl] [--repeats 3] [--max-usd 0.02] [--max-day-usd 2.00]
    python -m scripts.run_eval --run-dir data/results/<timestamp>        # resume: skips runs already written
    python -m scripts.run_eval --run-dir data/results/<timestamp> --report-only   # no model calls

Each run is appended to <run-dir>/runs.jsonl as soon as it finishes. Order is repeat, case, system, so a
stopped eval leaves balanced partial results. The eval stops (resumable) when the day cap is reached; a run
that hits its own per-run cap is recorded as an error. Backend is the Elasticsearch-backed rag-hybrid.
A non-empty `reason` on a non-error outcome is a note, not a failure. Reads cases; never writes data/eval/.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from scripts.draft_labels import guard_out_path
from scripts.live_one import SYSTEMS, LoggingTools
from triage.contracts import CATEGORIES, EvalCase
from triage.eval.metrics import compare, search_stats, summarise, verdict
from triage.eval.runner import load_cases, run_eval
from triage.llm import AnthropicLLM, Ledger

RESULTS_DIR = Path("data/results")
BACKEND = "es"
MATCH = "any"  # decided 26 Oct: at least one gold doc among the cited sources


def read_runs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def run_one(case: EvalCase, system: str, repeat: int, llm, ledger, tools: LoggingTools) -> dict:
    """One scored run plus what the score leaves out: queries, reason, wall latency, search latencies."""
    stash = {}

    def runner(c):
        res = SYSTEMS[system](c["issue_url"], tools, llm, ledger, list(CATEGORIES))
        stash.update(reason=res["reason"], dropped=res["dropped_citations"])
        return res["note"], res["search_results"]

    t0 = time.monotonic()
    got = run_eval([case], runner, system, BACKEND, match=MATCH)
    wall = time.monotonic() - t0
    return {**got.results[0], "repeat": repeat, "model": llm.model, "reason": stash.get("reason"),
            "dropped_citations": [d if isinstance(d, str) else f"<{d.type_name}>" for d in stash.get("dropped", [])],
            "queries": [c["arg"][0] for c in tools.log if c["tool"] == "search_docs"],
            "tool_errors": [c["error"] for c in tools.log if c["error"]],
            "searches": [{k: r[k] for k in ("cost_usd", "latency_ms", "cache_hit")}
                         for r in got.search_results_by_case.get(case["id"], [])],
            "latency_s": round(wall, 3), "error": got.errors.get(case["id"])}


def execute(cases, repeats, runs_path: Path, llm, make_ledger, make_tools=LoggingTools, log=print) -> dict:
    """Run every missing (repeat, case, system); append each record as it finishes. Stops on the day cap."""
    done = {(r["case_id"], r["system"], r["repeat"]) for r in read_runs(runs_path)}
    ran, skipped, stopped = 0, 0, None
    with open(runs_path, "a") as f:
        for repeat in range(1, repeats + 1):
            for case in cases:
                for system in SYSTEMS:
                    if (case["id"], system, repeat) in done:
                        skipped += 1
                        continue
                    ledger = make_ledger()
                    stopped = ledger.check(llm.model)  # raises on an unknown model
                    if stopped and "per-day cap" in stopped:
                        return {"ran": ran, "skipped": skipped, "stopped": stopped}
                    rec = run_one(case, system, repeat, llm, ledger, make_tools())
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    ran += 1
                    log(f"[{ran}] {case['id']} {system} #{repeat}: {rec['outcome']} ${rec['cost_usd']:.4f}")
    return {"ran": ran, "skipped": skipped, "stopped": None}


# --- report ------------------------------------------------------------------


def _agg(per_repeat: list[tuple[float, str]], unit: str = "") -> str:
    vals = [v for v, _ in per_repeat]
    detail = ", ".join(d for _, d in per_repeat)
    return f"mean {statistics.mean(vals):.1f}{unit} (range {min(vals):.1f}-{max(vals):.1f}{unit}) per repeat: {detail}"


def complete_repeats(runs: list[dict], cases: list[EvalCase]) -> list[int]:
    """Repeats where both systems have every case."""
    want = {(c["id"], s) for c in cases for s in SYSTEMS}
    reps = sorted({r["repeat"] for r in runs})
    return [n for n in reps if {(r["case_id"], r["system"]) for r in runs if r["repeat"] == n} == want]


def report(runs: list[dict], cases: list[EvalCase]) -> str:
    reps = complete_repeats(runs, cases)
    out = [f"runs recorded: {len(runs)}; complete repeats (both systems, all {len(cases)} cases): {reps or 'none'}"]
    incomplete = sorted({r["repeat"] for r in runs} - set(reps))
    if incomplete:
        out.append(f"repeats {incomplete} are incomplete and left out of the numbers below")
    if not reps:
        return "\n".join(out)
    sel = lambda s, n: [r for r in runs if r["system"] == s and r["repeat"] == n]  # noqa: E731
    for system in SYSTEMS:
        sums = [summarise(sel(system, n), cases) for n in reps]
        pct = lambda key: [(s[key]["pct"], f"{s[key]['k']}/{s[key]['n']}") for s in sums]  # noqa: E731
        out += ["", f"== {system} ({len(reps)} repeat(s); rates are k/n per repeat, then mean and range of the %) ==",
                "category accuracy, answerable:  " + _agg(pct("category_accuracy"), "%"),
                "correct doc cited, answerable:  " + _agg(pct("correct_doc_cited"), "%"),
                "can't tell, answerable:         " + _agg(pct("cant_tell_answerable"), "%"),
                "can't tell, not answerable:     " + _agg(pct("cant_tell_unanswerable"), "%"),
                "errors / failed runs:           " + _agg([(s["failed_count"], f"{s['failed_count']}/{s['n_cases']}") for s in sums]),
                "cost per case (non-failed):     " + _agg([(s["cost_usd"]["mean"] or 0, f"{s['cost_usd']['mean'] or 0:.5f}") for s in sums], " USD"),
                "steps per case (non-failed):    " + _agg([(s["steps"]["mean"] or 0, f"{s['steps']['mean'] or 0:.2f}") for s in sums])]
        rs = [r for n in reps for r in sel(system, n)]
        lat = [r["latency_s"] for r in rs]
        st = search_stats({r["case_id"] + f"#{r['repeat']}": r["searches"] for r in rs})
        out.append(f"wall latency per run: mean {statistics.mean(lat):.2f}s, range {min(lat):.2f}-{max(lat):.2f}s; "
                   f"search (uncached, {st['uncached_calls']} of {st['calls']} calls): mean "
                   f"{st['latency_ms_mean'] or 0:.0f} ms, cost total {st['cost_usd_total']:.5f} USD")
        notes = sum(1 for r in rs if r["reason"] and r["outcome"] != "error")
        out.append(f"notes (reason on a non-error outcome, not failures): {notes} of {len(rs)} runs")
    out += ["", "== agent vs baseline, paired per case, one comparison per repeat (repeats are not independent) =="]
    for n in reps:
        c = compare(sel("agent", n), sel("baseline", n), cases)
        out += [f"repeat {n}: " + verdict(c["category"], "category"), f"repeat {n}: " + verdict(c["correct_doc"], "correct doc")]
    spend = sum(r["cost_usd"] for r in runs)
    out.append(f"\ntotal recorded cost: {spend:.4f} USD (error runs record 0.0, so a lower bound)")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/eval/eval.jsonl")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-usd", type=float, default=0.02, help="cap per run")
    ap.add_argument("--max-day-usd", type=float, default=2.00, help="cap for all eval runs in one day")
    ap.add_argument("--run-dir", help="resume this results directory")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    cases = load_cases(args.cases)
    run_dir = Path(args.run_dir) if args.run_dir else RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_path = guard_out_path(run_dir / "runs.jsonl")
    if args.report_only:
        print(report(read_runs(runs_path), cases))
        return 0
    try:
        llm = AnthropicLLM()
        if not llm.model_available():
            print(f"the key cannot see {llm.model} in the models list; stopping before any paid call", file=sys.stderr)
            return 2
    except Exception as e:  # never echo more than the error type for SDK/auth failures
        print(f"cannot start: {type(e).__name__}: {e if isinstance(e, RuntimeError) else 'models list call failed'}",
              file=sys.stderr)
        return 2
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    spend_path = guard_out_path(RESULTS_DIR / "eval_spend.json")
    make_ledger = lambda: Ledger(per_run_cap=args.max_usd, per_day_cap=args.max_day_usd, path=spend_path)  # noqa: E731
    print(f"{len(cases)} cases x {len(SYSTEMS)} systems x {args.repeats} repeats = "
          f"{len(cases) * len(SYSTEMS) * args.repeats} runs -> {runs_path}")
    got = execute(cases, args.repeats, runs_path, llm, make_ledger)
    print(f"ran {got['ran']}, skipped {got['skipped']} already written" + (f"; STOPPED: {got['stopped']}" if got["stopped"] else ""))
    print("\n" + report(read_runs(runs_path), cases))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
