"""Run the agent (or, with --system baseline, the single-shot baseline) LIVE on one issue. PAID: run only on request.

Usage (env loaded: TRIAGE_ANTHROPIC_API_KEY, GITHUB_TOKEN; rag-hybrid up):
    python -m scripts.live_one --issue https://github.com/kubernetes/kubernetes/issues/127264 --max-usd 0.02

Prints the triage note, every tool call (ok / error kind, and the search query), steps and cost.
Spend goes through the default Ledger (data/spend.json, day cap $1.00). Nothing is written to data/eval/.
"""
import argparse
import json
import sys

from triage import server
from triage.agent import run_agent
from triage.baseline import run_baseline
from triage.contracts import CATEGORIES
from triage.llm import AnthropicLLM, Ledger


class LoggingTools:
    """Delegates to the real tools and records what the note's tool_calls leave out: the query and the error kind."""

    def __init__(self, inner=server):
        self.inner, self.log = inner, []

    def _run(self, name, fn, arg):
        r = fn(arg) if name != "search_docs" else fn(*arg)
        err = r.get("error") if isinstance(r, dict) else None
        self.log.append({"tool": name, "arg": arg if name == "search_docs" else None, "error": err})
        return r

    def search_docs(self, query, k=5):
        return self._run("search_docs", self.inner.search_docs, (query, k))

    def get_issue_thread(self, issue_url):
        return self._run("get_issue_thread", self.inner.get_issue_thread, issue_url)

    def extract_error_signals(self, text):
        return self._run("extract_error_signals", self.inner.extract_error_signals, text)


def report(res: dict, tools: LoggingTools) -> None:
    note = res["note"]
    print(json.dumps(note, indent=2, default=str))
    print("\ntool calls:")
    for i, c in enumerate(tools.log, 1):
        status = f"error: {c['error']}" if c["error"] else "ok"
        query = f"  query={c['arg'][0]!r} k={c['arg'][1]}" if c["tool"] == "search_docs" else ""
        print(f"  {i}. {c['tool']}: {status}{query}")
    print(f"\nsteps: {note['steps']}   cost: ${note['cost_usd']:.6f}   outcome: {note['outcome']}")
    if res["reason"]:
        print(f"reason: {res['reason']}")
    if res["dropped_citations"]:
        print(f"dropped citations: {res['dropped_citations']}")


SYSTEMS = {"agent": run_agent, "baseline": run_baseline}


def run(issue_url: str, llm, ledger, tools: LoggingTools, system: str = "agent") -> dict:
    res = SYSTEMS[system](issue_url, tools, llm, ledger, list(CATEGORIES))
    report(res, tools)
    return res


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="agent")
    ap.add_argument("--max-usd", type=float, default=0.02, help="cap for this run")
    args = ap.parse_args(argv)
    try:
        llm = AnthropicLLM()
        if not llm.model_available():
            print(f"the key cannot see {llm.model} in the models list; stopping before any paid call", file=sys.stderr)
            return 2
    except Exception as e:  # never echo more than the error type for SDK/auth failures
        print(f"cannot start: {type(e).__name__}: {e if isinstance(e, RuntimeError) else 'models list call failed'}",
              file=sys.stderr)
        return 2
    run(args.issue, llm, Ledger(per_run_cap=args.max_usd), LoggingTools(), args.system)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
