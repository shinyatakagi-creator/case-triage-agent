"""Frozen tool/agent/eval shapes. Changing a key or shape needs the owner's approval.

Source: docs/contracts-outline.md. Tools return either their normal shape or a ToolError;
they never raise.
"""
from typing import Literal, NotRequired, TypedDict, get_args

# --- constants -------------------------------------------------------------

MAX_STEPS = 6  # agent loop hard cap (scope: "Hard cap of 6 steps")

ErrorKind = Literal[
    "timeout", "rate_limited", "not_found", "bad_url", "not_an_issue",
    "github_failed", "search_failed",
]
Outcome = Literal["answered", "cant_tell", "step_cap", "error"]
Backend = Literal["es", "inproc"]  # RETRIEVAL_BACKEND values

ERROR_KINDS = get_args(ErrorKind)
OUTCOMES = get_args(Outcome)
BACKENDS = get_args(Backend)

# --- tool errors -----------------------------------------------------------


class ToolError(TypedDict):
    error: ErrorKind
    detail: NotRequired[str]
    tool: NotRequired[str]
    reset: NotRequired[str]


# --- search_docs -----------------------------------------------------------


class Passage(TypedDict):
    text: str     # rag-hybrid "content"
    source: str   # rag-hybrid "doc_id": the stable id, used for scoring (never url)
    url: str
    title: str
    heading: str  # rag-hybrid "heading_path"
    chunk_id: str
    score: float


class SearchResult(TypedDict):
    # Several chunks can share one source; de-duplicate by source when counting citations.
    passages: list[Passage]
    cost_usd: float  # added into the run's cost
    latency_ms: float
    # Query-embedding cache hit; cost/latency are reported from uncached calls only.
    cache_hit: bool


# --- get_issue_thread ------------------------------------------------------


class Comment(TypedDict):
    author: str
    created_at: str
    body: str


class IssueThread(TypedDict):
    number: int
    title: str
    author: str
    created_at: str
    labels: list[str]
    body: str
    comments: list[Comment]  # human comments before the first maintainer reply only
    thread_truncated: bool


# --- extract_error_signals -------------------------------------------------


class Signals(TypedDict):
    versions: list[str]
    k8s_reasons: list[str]
    error_lines: list[str]
    stack_traces: list[str]


# --- agent output ----------------------------------------------------------


class Citation(TypedDict):
    source: str


class ToolCall(TypedDict):
    tool: str
    ok: bool


class TriageNote(TypedDict):
    issue_url: str
    outcome: Outcome
    category: str | None  # from the owner's category set; never invented by code
    likely_cause: str | None
    # Rule: a citation not returned by search_docs in the same run is invalid
    # and counts as a failure.
    citations: list[Citation]
    next_step: str | None
    steps: int
    cost_usd: float
    tool_calls: list[ToolCall]


# --- eval ------------------------------------------------------------------


class EvalCase(TypedDict):
    id: str
    issue_url: str
    answerable: bool
    category: str
    gold_docs: list[str]  # empty when answerable is false
    notes: NotRequired[str]


class EvalResult(TypedDict):
    case_id: str
    system: Literal["agent", "baseline"]
    backend: Backend
    outcome: Outcome
    category: str | None
    cited_docs: list[str]
    correct_category: bool
    correct_doc_cited: bool
    steps: int
    cost_usd: float
    failed: bool


# --- checks ----------------------------------------------------------------


def contract_problems(note: TriageNote, search_results: list[SearchResult]) -> list[str]:
    """Contract violations in a finished note; empty list means clean.

    search_results are every search_docs result returned during the same run.
    """
    returned = {p["source"] for r in search_results for p in r["passages"]}
    problems = [
        f"citation not returned by search_docs in this run: {c['source']}"
        for c in note["citations"]
        if c["source"] not in returned
    ]
    if note["outcome"] == "answered" and not note["citations"]:
        problems.append("outcome 'answered' with no citations")
    if note["steps"] > MAX_STEPS:
        problems.append(f"steps {note['steps']} > MAX_STEPS {MAX_STEPS}")
    return problems
