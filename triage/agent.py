"""Agent loop plus the pieces the single-shot baseline shares (tool runner, model call, note building).

Definition of "step": ONE MODEL TURN (one llm.complete call that returned). Tool executions
inside that turn, and the single timeout retry, do not count. MAX_STEPS bounds model turns.
"""
import json
import math
from typing import NamedTuple, Protocol, TypedDict

from triage.contracts import (
    ERROR_KINDS, MAX_STEPS, Citation, Outcome, SearchResult, Signals, ToolCall, ToolError,
    TriageNote, contract_problems,
)
from triage.llm import LLM, Ledger

MAX_TOOL_CALLS = 8  # per run, executed attempts including the timeout retry (agent config, not a contract)
FINAL_ANSWER = "final_answer"
TOOL_NAMES = ("search_docs", "get_issue_thread", "extract_error_signals")

# Errors where retrying or continuing cannot help: the issue itself cannot be assessed.
# Outcome is "error", not "cant_tell": "cant_tell" is a measured judgement about the evidence,
# and infrastructure or input failures must not inflate that rate.
FATAL_KINDS = {"rate_limited", "bad_url", "not_found", "not_an_issue"}

# ErrorKind has no generic "malformed" or "budget" value, so those cases are reported as the
# tool's own failure kind (contracts are frozen; flagged for the owner).
FAIL_KIND = {"search_docs": "search_failed", "get_issue_thread": "github_failed",
             "extract_error_signals": "search_failed"}

SYSTEM_PROMPT = """You triage one public Kubernetes GitHub issue using three tools.
get_issue_thread fetches the issue under triage (you cannot choose another). search_docs searches
the Kubernetes documentation. extract_error_signals parses error text.
Work in few steps; you have a hard limit on model turns and on tool calls. If a search returns
nothing, try a differently worded query. Never retry a tool that returned an error.
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
    reason: str | None            # why the run ended in "error" (cap, tool error, bad model output)


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


# --- output validation (types, not just keys) ------------------------------


def _str(x): return isinstance(x, str)
def _bool(x): return isinstance(x, bool)
def _int(x): return isinstance(x, int) and not isinstance(x, bool)
def _finite(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def _nonneg(x): return _finite(x) and x >= 0
def _strs(x): return isinstance(x, list) and all(isinstance(i, str) for i in x)


def _fields_ok(obj, spec: dict) -> bool:
    return isinstance(obj, dict) and all(k in obj and ok(obj[k]) for k, ok in spec.items())


def _list_of(spec: dict):
    return lambda x: isinstance(x, list) and all(_fields_ok(i, spec) for i in x)


# One spec per frozen contract shape; tests assert the keys match contracts.py.
PASSAGE_SPEC = {"text": _str, "source": _str, "url": _str, "title": _str, "heading": _str,
                "chunk_id": _str, "score": _finite}
SEARCH_SPEC = {"passages": _list_of(PASSAGE_SPEC), "cost_usd": _nonneg, "latency_ms": _nonneg, "cache_hit": _bool}
COMMENT_SPEC = {"author": _str, "created_at": _str, "body": _str}
THREAD_SPEC = {"number": _int, "title": _str, "author": _str, "created_at": _str, "labels": _strs,
               "body": _str, "comments": _list_of(COMMENT_SPEC), "thread_truncated": _bool}
SIGNALS_SPEC = {"versions": _strs, "k8s_reasons": _strs, "error_lines": _strs, "stack_traces": _strs}
OUTPUT_SPEC = {"search_docs": SEARCH_SPEC, "get_issue_thread": THREAD_SPEC, "extract_error_signals": SIGNALS_SPEC}


def is_error(r) -> bool:
    return isinstance(r, dict) and "error" in r


class ToolRunner:
    """Calls the plain tool functions with retry/degrade rules; never raises, never loops on a failure."""

    def __init__(self, tools: Tools, issue_url: str, ledger: Ledger, retry_timeouts: bool = True):
        self.tools, self.issue_url, self.ledger, self.retry_timeouts = tools, issue_url, ledger, retry_timeouts
        self.calls: list[ToolCall] = []
        self.search_results: list[SearchResult] = []
        self.search_cost = 0.0
        self.dead: dict[str, ToolError] = {}  # tool -> its terminal error; later calls are not executed
        self.last_error: dict[str, ToolError | None] = {}  # tool -> error of its most recent executed call
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
        elif not _fields_ok(r, OUTPUT_SPEC[name]):
            r = {"error": FAIL_KIND[name], "detail": "malformed tool output"}
        self.calls.append({"tool": name, "ok": not is_error(r)})
        self.last_error[name] = r if is_error(r) else None
        return r

    def call(self, name: str, args: dict):
        if self.fatal:
            return self.fatal  # a fatal error ends the run; later blocks in the same turn are not executed
        if name in self.dead:
            return self.dead[name]
        if len(self.calls) >= MAX_TOOL_CALLS:
            # Not executed. Reported with the tool's own failure kind (no budget kind in ErrorKind).
            return {"error": FAIL_KIND[name], "detail": f"tool call budget of {MAX_TOOL_CALLS} exhausted"}
        r = self._once(name, args)
        if self.retry_timeouts and is_error(r) and r["error"] == "timeout" and len(self.calls) < MAX_TOOL_CALLS:
            r = self._once(name, args)  # retry exactly once; rate_limited is never retried
        if not is_error(r):
            if name == "search_docs":
                self.search_results.append(r)
                self.search_cost += r["cost_usd"]
                self.ledger.charge_usd(r["cost_usd"])  # search cost counts against the run and day totals
        elif r["error"] in FATAL_KINDS:
            self.fatal = r
        else:
            self.dead[name] = r  # timeout twice, search_failed, github_failed, malformed
        return r


# --- model -----------------------------------------------------------------


class Asked(NamedTuple):
    content: list[dict] | None  # normalised blocks; None whenever reason is set
    usd: float                  # real cost charged, even when the turn is unusable
    reason: str | None          # set -> the run must end in "error"
    called: bool                # a model turn came back (counts as a step)


def _count_ok(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def normalise_blocks(content: list) -> list[dict]:
    """Drop non-dict blocks; give every tool_use a str id (deterministic placeholder), name and dict input."""
    out = []
    for i, b in enumerate(content):
        if not isinstance(b, dict):
            continue
        b = dict(b)
        if b.get("type") == "tool_use":
            if not (isinstance(b.get("id"), str) and b["id"]):
                b["id"] = f"missing-id-{i}"
            if not isinstance(b.get("name"), str):
                b["name"] = ""
            if not isinstance(b.get("input"), dict):
                b["input"] = {}
        out.append(b)
    return out


def ask(llm: LLM, ledger: Ledger, system: str, messages: list[dict], specs: list[dict]) -> Asked:
    """One capped model call. The cap is checked first; spend is recorded before anything else."""
    reason = ledger.check(llm.model)  # raises on an unknown model
    if reason:
        return Asked(None, 0.0, reason, False)
    try:
        turn = llm.complete(system, messages, specs)
    except Exception as e:
        return Asked(None, 0.0, f"model call failed: {e!r}"[:200], False)
    if not isinstance(turn, dict):
        return Asked(None, 0.0, "malformed model turn: not a dict", True)
    tokens = [turn.get("input_tokens"), turn.get("output_tokens")]
    ok = [_count_ok(t) for t in tokens]
    usd = ledger.charge(llm.model, tokens[0] if ok[0] else 0, tokens[1] if ok[1] else 0)
    reasons = []
    if not all(ok):
        reasons.append("model turn has missing or invalid token counts; cost counted from known fields only")
    if ledger.load_error:
        reasons.append(ledger.load_error)  # e.g. "spend ledger write failed: ..."; usd is still returned
    if not isinstance(turn.get("content"), list):
        reasons.append("malformed model turn: content is not a list")
    if reasons:
        return Asked(None, usd, "; ".join(reasons), True)
    return Asked(normalise_blocks(turn["content"]), usd, None, True)


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
    failed = [(n, e) for n, e in runner.last_error.items() if e]
    if outcome == "cant_tell" and failed:
        # A "can't tell" that follows a broken tool is a failure, not evidence of unanswerability.
        note["outcome"] = "error"
        reason = "; ".join(f"tool {n} ended in error: {e['error']}" for n, e in failed)
    problems = contract_problems(note, runner.search_results)
    if problems and note["outcome"] != "error":
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
    runner = ToolRunner(tools, issue_url, ledger)
    specs = tool_specs(categories)
    messages: list[dict] = [{"role": "user", "content": f"Triage this issue: {issue_url}"}]
    steps, usd = 0, 0.0

    def done(outcome, final=None, reason=None):
        return build_result(issue_url, outcome, runner, steps, usd, final, categories, reason)

    while steps < MAX_STEPS:
        a = ask(llm, ledger, SYSTEM_PROMPT, messages, specs)
        steps += a.called
        usd += a.usd
        if a.reason:
            return done("error", reason=a.reason)
        messages.append({"role": "assistant", "content": a.content})
        uses = [b for b in a.content if b.get("type") == "tool_use"]
        # final_answer wins over any other tool blocks in the same turn (those are not executed)
        final = next((u for u in uses if u["name"] == FINAL_ANSWER), None)
        if final:
            out = final["input"]
            return done("cant_tell" if out.get("outcome") == "cant_tell" else "answered", out)
        if not uses:
            return done("error", reason="model ended its turn without calling final_answer")
        if steps == MAX_STEPS:
            break  # no point running tools whose results the model can never read
        results = []
        for u in uses:
            if u["name"] in TOOL_NAMES:
                results.append(_result_block(u, runner.call(u["name"], u["input"])))
            else:
                results.append(_result_block(u, {"error": "unknown tool"}))
        messages.append({"role": "user", "content": results})
        if runner.fatal:
            return done("error", reason=f"tool error, not retried: {runner.fatal['error']}")
        if ledger.load_error:
            return done("error", reason=ledger.load_error)
    return done("step_cap")
