"""Single-shot baseline (scope, Findings 25 Oct): same model, one search, one model call, no loop.

Failure policy, stated exactly:
- No timeout retry anywhere in the baseline (the agent retries once; the baseline does not).
- get_issue_thread fails (any kind) -> outcome "error", no search, no model call.
- search_docs fails (any kind) -> outcome "error", NO model call. An empty but valid search
  still goes to the model, which can only answer "cant_tell" since nothing is citable.
- Exactly one search_docs call, at most one model call.
"""
import json

from triage.agent import (
    FINAL_ANSWER, RunResult, ToolRunner, Tools, ask, build_result, final_answer_spec, is_error,
)
from triage.llm import LLM, Ledger

QUERY_BODY_CHARS = 500  # first N chars of the body go into the query

SYSTEM_PROMPT = """You triage one public Kubernetes GitHub issue in a single step.
You are given the issue thread and one set of documentation search results. Call final_answer
exactly once. Cite only sources present in the search results. If they do not support an answer,
use outcome "cant_tell". Do not guess."""


def build_query(thread: dict) -> str:
    # Deterministic on purpose: the baseline must not spend a model call on query writing.
    return f"{thread['title']} {thread['body'][:QUERY_BODY_CHARS]}".strip()


def run_baseline(issue_url: str, tools: Tools, llm: LLM, ledger: Ledger,
                 categories: list[str] | None = None) -> RunResult:
    runner = ToolRunner(tools, issue_url, ledger, retry_timeouts=False)

    def done(outcome, final=None, reason=None, steps=0, usd=0.0):
        return build_result(issue_url, outcome, runner, steps, usd, final, categories, reason)

    # Same order as the agent: the ledger is checked before anything is spent, tools included.
    reason = ledger.check(llm.model)
    if reason:
        return done("error", reason=reason)
    thread = runner.call("get_issue_thread", {})
    if is_error(thread):
        return done("error", reason=f"get_issue_thread failed: {thread['error']}")
    search = runner.call("search_docs", {"query": build_query(thread)})
    if is_error(search):
        return done("error", reason=f"search_docs failed: {search['error']}")
    if ledger.load_error:
        return done("error", reason=ledger.load_error)
    prompt = json.dumps({"issue_url": issue_url, "thread": thread, "search_results": search}, default=str)
    a = ask(llm, ledger, SYSTEM_PROMPT, [{"role": "user", "content": prompt}], [final_answer_spec(categories)])
    if a.reason:
        return done("error", reason=a.reason, steps=int(a.called), usd=a.usd)
    final = next((b for b in a.content if b.get("type") == "tool_use" and b["name"] == FINAL_ANSWER), None)
    if final is None:
        return done("error", reason="model did not call final_answer", steps=1, usd=a.usd)
    out = final["input"]
    return done("cant_tell" if out.get("outcome") == "cant_tell" else "answered", out, steps=1, usd=a.usd)
