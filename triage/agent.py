"""Agent loop plus the pieces the single-shot baseline shares (tool runner, model call, note building).

Definition of "step": ONE MODEL TURN (one llm.complete call that returned). Tool executions
inside that turn, and the single timeout retry, do not count. MAX_STEPS bounds model turns.

Known limitations (accepted):
- Malformed model tool calls (non-string query/text, bad k) are not executed and are not recorded in
  note.tool_calls; the only bound on how many a model can emit is MAX_STEPS turns.
- A model turn whose token counts are missing or invalid is charged for the known fields only, so
  its real cost may be under-counted.
- Cache tiers are keyed on prompt size (input + cache tokens > 100k); that is a reading of the
  owner's price list, and input/output rates are not tiered (see llm.py).
"""
import json
import math
import sys
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


class InvalidCitation(NamedTuple):
    type_name: str  # type of the rejected item, e.g. "dict"; the item itself is never kept or str()'d


class RunResult(TypedDict):
    note: TriageNote
    search_results: list[SearchResult]
    # Cited but never returned by search_docs; the harness counts these as failures. Non-string
    # citations appear as InvalidCitation, a type no real string can equal.
    dropped_citations: list[str | InvalidCitation]
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
def _finite(x):
    # Never raises: math.isfinite(10**400) raises OverflowError, so ints are compared to the
    # largest float (exact comparison); a bigger int is simply invalid.
    if isinstance(x, bool):
        return False
    if isinstance(x, float):
        return math.isfinite(x)
    return isinstance(x, int) and abs(x) <= sys.float_info.max
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
        self.cost_overflow = False  # search costs summed past the float range
        self.dead: dict[str, ToolError] = {}  # tool -> its terminal error; later calls are not executed
        self.last_error: dict[str, ToolError | None] = {}  # tool -> error of its most recent executed call
        self.fatal: ToolError | None = None

    def _once(self, name: str, args: dict):
        try:
            if name == "search_docs":
                k = args.get("k")
                k = k if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= 20 else 5
                r = self.tools.search_docs(args["query"], k)
            elif name == "get_issue_thread":
                # Always the issue under triage, whatever the model asked for: the agent never
                # fetches other threads or comments itself (leakage rule).
                r = self.tools.get_issue_thread(self.issue_url)
            else:
                r = self.tools.extract_error_signals(args["text"])
        except Exception as e:  # tools should not raise; a raise is treated as a failed result
            r = {"error": FAIL_KIND[name], "detail": f"tool raised: {e!r}"[:200]}
        if is_error(r):
            if r["error"] not in ERROR_KINDS:
                r = {"error": FAIL_KIND[name], "detail": "unknown error kind from tool"}
            else:
                # Keep only string fields: the error dict is json.dumps'd for the model and a
                # tool-controlled nested detail could blow the recursion limit.
                r = {k: r[k] for k in ("error", "detail", "tool", "reset") if isinstance(r.get(k), str)}
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
        required = {"search_docs": "query", "extract_error_signals": "text"}.get(name)
        if required and not isinstance(args.get(required), str):
            # Model-supplied garbage input: not executed, not counted, tool not marked dead.
            return {"error": FAIL_KIND[name], "detail": f"invalid input: {required} must be a string"}
        if len(self.calls) >= MAX_TOOL_CALLS:
            # Not executed. Reported with the tool's own failure kind (no budget kind in ErrorKind).
            # Deliberately NOT recorded in last_error: a budget refusal is not a tool's own result,
            # so it does not turn a later "cant_tell" into "error" (pinned by a test).
            return {"error": FAIL_KIND[name], "detail": f"tool call budget of {MAX_TOOL_CALLS} exhausted"}
        r = self._once(name, args)
        if self.retry_timeouts and is_error(r) and r["error"] == "timeout" and len(self.calls) < MAX_TOOL_CALLS:
            r = self._once(name, args)  # retry exactly once; rate_limited is never retried
        if not is_error(r):
            if name == "search_docs":
                self.search_results.append(r)
                total = self.search_cost + r["cost_usd"]
                if math.isfinite(total):
                    self.search_cost = total
                else:
                    self.cost_overflow = True  # not added: a note must never carry an infinite cost
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


MAX_PLAUSIBLE_TOKENS = 10**7  # sanity bound; a larger count is treated as invalid, its known sibling is still charged


def _count_ok(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= MAX_PLAUSIBLE_TOKENS


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
    # input/output must be present and valid. The cache fields may be absent (= 0), but a present
    # value that is invalid is an error like any other bad count.
    required = [turn.get("input_tokens"), turn.get("output_tokens")]
    cache = [turn.get(k, 0) for k in ("cache_write_5m_tokens", "cache_write_1h_tokens", "cache_read_tokens")]
    counts = required + cache
    ok = [_count_ok(t) for t in counts]
    usd = ledger.charge(llm.model, *(t if good else 0 for t, good in zip(counts, ok)))
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
    # Non-string citations are never str()'d (a nested object could blow the recursion limit).
    strs = list(dict.fromkeys(c for c in raw if isinstance(c, str)))
    valid = [c for c in strs if c in returned]
    dropped = [c for c in strs if c not in returned] + [
        InvalidCitation(type(c).__name__) for c in raw if not isinstance(c, str)]
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
    if runner.cost_overflow:
        note["outcome"] = "error"
        reason = "search costs overflowed the float range; note cost_usd is understated"
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
