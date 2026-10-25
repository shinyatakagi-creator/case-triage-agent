"""Agent loop plus the pieces the single-shot baseline shares (tool runner, model call, note building).

Definition of "step": ONE MODEL TURN (one llm.complete call that returned). Tool executions
inside that turn, and the single timeout retry, do not count. MAX_STEPS bounds model turns.
"""
import json
from typing import Protocol, TypedDict

from triage.contracts import (
    ERROR_KINDS, MAX_STEPS, Citation, Comment, IssueThread, Outcome, Passage, SearchResult, Signals, ToolCall,
    ToolError, TriageNote, contract_problems,
)
from triage.llm import LLM, Ledger

FINAL_ANSWER = "final_answer"

# Errors where retrying or continuing cannot help: the issue itself cannot be assessed.
# Outcome is "error", not "cant_tell": "cant_tell" is a measured judgement about the evidence,
# and infrastructure or input failures must not inflate that rate.
FATAL_KINDS = {"rate_limited", "bad_url", "not_found", "not_an_issue"}

# ErrorKind has no generic "malformed" value, so a bad tool output is reported as that tool's
# own failure kind (contracts are frozen; flagged for the owner).
FAIL_KIND = {"search_docs": "search_failed", "get_issue_thread": "github_failed",
             "extract_error_signals": "search_failed"}

SYSTEM_PROMPT = """You triage one public Kubernetes GitHub issue using three tools.
get_issue_thread fetches the issue under triage (you cannot choose another). search_docs searches
the Kubernetes documentation. extract_error_signals parses error text.
Work in few steps; you have a hard limit on model turns. If a search returns nothing, try a
differently worded query. Never retry a tool that returned an error.
Finish by calling final_answer. Cite only sources returned by search_docs in this conversation.
If the documentation does not support an answer, call final_answer with outcome "cant_tell";
that is a valid, expected outcome. Do not guess."""


class Tools(Protocol):
    def search_docs(self, query: str, k: int = 5) -> SearchResult | ToolError: ...
    def get_issue_thread(self, issue_url: str) -> dict: ...
    def extract_error_signals(self, text: str) -> Signals: ...


class RunResult(TypedDict):
    note: TriageNote
    search_results: list[SearchResult]
    dropped_citations: list[str]  # cited but never returned by search_docs; the harness counts these as failures
    reason: str | None            # why the run ended in "error" (cap, fatal tool error, bad model output)


def final_answer_spec(categories: list[str] | None) -> dict:
    category = {"type": "string"}
    if categories:
        category["enum"] = list(categories)
    return {
        "name": FINAL_ANSWER,
        "description": "Submit the triage note and end the run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["answered", "cant_tell"]},
                "category": category,
                "likely_cause": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"},
                              "description": "source ids returned by search_docs"},
                "next_step": {"type": "string"},
            },
            "required": ["outcome"],
        },
    }


def tool_specs(categories: list[str] | None) -> list[dict]:
    return [
        {"name": "search_docs", "description": "Search the Kubernetes docs.",
         "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                          "required": ["query"]}},
        {"name": "get_issue_thread", "description": "Fetch the issue under triage.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "extract_error_signals", "description": "Parse versions and errors out of text.",
         "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
        final_answer_spec(categories),
    ]


# --- tools -----------------------------------------------------------------


def _has_keys(obj, td) -> bool:
    return isinstance(obj, dict) and td.__required_keys__ <= obj.keys()


def _shape_ok(name: str, r) -> bool:
    if name == "search_docs":
        return (_has_keys(r, SearchResult) and isinstance(r["passages"], list)
                and all(_has_keys(p, Passage) for p in r["passages"]))
    if name == "get_issue_thread":
        return (_has_keys(r, IssueThread) and isinstance(r["comments"], list)
                and all(_has_keys(c, Comment) for c in r["comments"]))
    return _has_keys(r, Signals)


def is_error(r) -> bool:
    return isinstance(r, dict) and "error" in r


class ToolRunner:
    """Calls the plain tool functions with retry/degrade rules; never raises, never loops on a failure."""

    def __init__(self, tools: Tools, issue_url: str):
        self.tools, self.issue_url = tools, issue_url
        self.calls: list[ToolCall] = []
        self.search_results: list[SearchResult] = []
        self.search_cost = 0.0
        self.dead: dict[str, ToolError] = {}  # tool -> its terminal error; later calls are not executed
        self.fatal: ToolError | None = None

    def _once(self, name: str, args: dict):
        try:
            if name == "search_docs":
                r = self.tools.search_docs(str(args.get("query", "")), int(args.get("k", 5)))
            elif name == "get_issue_thread":
                # Always the issue under triage, whatever the model asked for: the agent never
                # fetches other threads or comments itself (leakage rule).
                r = self.tools.get_issue_thread(self.issue_url)
            else:
                r = self.tools.extract_error_signals(str(args.get("text", "")))
        except Exception as e:  # tools should not raise; a raise is treated as a failed result
            r = {"error": FAIL_KIND[name], "detail": f"tool raised: {e!r}"[:200]}
        if is_error(r):
            if r["error"] not in ERROR_KINDS:
                r = {"error": FAIL_KIND[name], "detail": "unknown error kind from tool"}
        elif not _shape_ok(name, r):
            r = {"error": FAIL_KIND[name], "detail": "malformed tool output"}
        self.calls.append({"tool": name, "ok": not is_error(r)})
        return r

    def call(self, name: str, args: dict):
        if name in self.dead:
            return self.dead[name]
        r = self._once(name, args)
        if is_error(r) and r["error"] == "timeout":
            r = self._once(name, args)  # retry exactly once
        if not is_error(r):
            if name == "search_docs":
                self.search_results.append(r)
                self.search_cost += r["cost_usd"]
        elif r["error"] in FATAL_KINDS:
            self.fatal = r
        else:
            self.dead[name] = r  # timeout twice, search_failed, github_failed, malformed
        return r


# --- model -----------------------------------------------------------------


def ask(llm: LLM, ledger: Ledger, system: str, messages: list[dict], specs: list[dict]):
    """One capped model call -> (turn or None, usd, reason or None). The cap is checked first."""
    reason = ledger.check(llm.model)
    if reason:
        return None, 0.0, reason
    try:
        turn = llm.complete(system, messages, specs)
        return turn, ledger.charge(llm.model, turn["input_tokens"], turn["output_tokens"]), None
    except Exception as e:
        return None, 0.0, f"model call failed: {e!r}"[:200]


def _text(x) -> str | None:
    return x.strip() if isinstance(x, str) and x.strip() else None


def build_result(issue_url: str, outcome: Outcome, runner: ToolRunner, steps: int, llm_usd: float,
                 final: dict | None = None, categories: list[str] | None = None,
                 reason: str | None = None) -> RunResult:
    """Assemble the note. Citations are validated against sources search_docs returned in this run."""
    final = final if isinstance(final, dict) else {}
    returned = {p["source"] for r in runner.search_results for p in r["passages"]}
    raw = final.get("citations") if isinstance(final.get("citations"), list) else []
    raw = list(dict.fromkeys(str(c) for c in raw))
    valid = [c for c in raw if c in returned]
    dropped = [c for c in raw if c not in returned]
    category = final.get("category")
    note = TriageNote(
        issue_url=issue_url, outcome=outcome,
        category=category if categories and category in categories else None,
        likely_cause=_text(final.get("likely_cause")),
        citations=[Citation(source=c) for c in valid],
        next_step=_text(final.get("next_step")),
        steps=steps, cost_usd=llm_usd + runner.search_cost, tool_calls=list(runner.calls),
    )
    problems = contract_problems(note, runner.search_results)
    if problems and outcome != "error":
        # e.g. "answered" whose citations were all invented: a failure, not a quiet downgrade to cant_tell
        reason = "; ".join(problems) + (f"; dropped citations: {dropped}" if dropped else "")
        note["outcome"] = "error"
    if note["outcome"] == "error":
        note.update(category=None, likely_cause=None, citations=[], next_step=None)
    return RunResult(note=note, search_results=runner.search_results, dropped_citations=dropped, reason=reason)


# --- agent -----------------------------------------------------------------


def _result_block(use: dict, r) -> dict:
    return {"type": "tool_result", "tool_use_id": use["id"], "content": json.dumps(r, default=str),
            "is_error": is_error(r)}


def run_agent(issue_url: str, tools: Tools, llm: LLM, ledger: Ledger,
              categories: list[str] | None = None) -> RunResult:
    runner = ToolRunner(tools, issue_url)
    specs = tool_specs(categories)
    messages: list[dict] = [{"role": "user", "content": f"Triage this issue: {issue_url}"}]
    steps, usd = 0, 0.0

    def done(outcome, final=None, reason=None):
        return build_result(issue_url, outcome, runner, steps, usd, final, categories, reason)

    while steps < MAX_STEPS:
        turn, cost, reason = ask(llm, ledger, SYSTEM_PROMPT, messages, specs)
        if turn is None:
            return done("error", reason=reason)
        steps += 1
        usd += cost
        messages.append({"role": "assistant", "content": turn["content"]})
        uses = [b for b in turn["content"] if b.get("type") == "tool_use"]
        final = next((u for u in uses if u["name"] == FINAL_ANSWER), None)
        if final:
            out = final["input"] if isinstance(final["input"], dict) else {}
            return done("cant_tell" if out.get("outcome") == "cant_tell" else "answered", out)
        if not uses:
            return done("error", reason="model ended its turn without calling final_answer")
        if steps == MAX_STEPS:
            break  # no point running tools whose results the model can never read
        results = []
        for u in uses:
            if u["name"] in ("search_docs", "get_issue_thread", "extract_error_signals"):
                results.append(_result_block(u, runner.call(u["name"], u.get("input") or {})))
            else:
                runner.calls.append({"tool": str(u["name"]), "ok": False})
                results.append(_result_block(u, {"error": "unknown tool"}))
        messages.append({"role": "user", "content": results})
        if runner.fatal:
            return done("error", reason=f"tool error, not retried: {runner.fatal['error']}")
    return done("step_cap")
