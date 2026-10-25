"""Single-shot baseline (scope, Findings 25 Oct): same model, one search, one model call, no loop."""
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
    runner = ToolRunner(tools, issue_url)

    def done(outcome, final=None, reason=None, steps=0, usd=0.0):
        return build_result(issue_url, outcome, runner, steps, usd, final, categories, reason)

    thread = runner.call("get_issue_thread", {})
    if is_error(thread):
        return done("error", reason=f"get_issue_thread failed: {thread['error']}")
    search = runner.call("search_docs", {"query": build_query(thread)})
    if runner.fatal:
        return done("error", reason=f"tool error, not retried: {runner.fatal['error']}")
    # A degraded search is shown to the model as an error; with no passages it cannot cite anything.
    prompt = json.dumps({"issue_url": issue_url, "thread": thread, "search_results": search}, default=str)
    turn, usd, reason = ask(llm, ledger, SYSTEM_PROMPT, [{"role": "user", "content": prompt}],
                            [final_answer_spec(categories)])
    if turn is None:
        return done("error", reason=reason)
    final = next((b for b in turn["content"] if b.get("type") == "tool_use" and b["name"] == FINAL_ANSWER), None)
    if final is None:
        return done("error", reason="model did not call final_answer", steps=1, usd=usd)
    out = final["input"] if isinstance(final["input"], dict) else {}
    return done("cant_tell" if out.get("outcome") == "cant_tell" else "answered", out, steps=1, usd=usd)
