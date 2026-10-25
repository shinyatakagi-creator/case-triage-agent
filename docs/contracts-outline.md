# triage/contracts.py: outline

Claude Code writes this file as the first thing, as stdlib TypedDicts plus constants, then stops for my review. Nothing else is built until I approve it.

## Tool errors (already used by the evening-1 code)
```
ToolError = {"error": "timeout" | "rate_limited" | "not_found" | "bad_url" | "not_an_issue"
                    | "github_failed" | "search_failed", "detail"?: str, "tool"?: str, "reset"?: str}
```
Every tool returns either its normal shape or a ToolError.

## search_docs
Real rag-hybrid `/search` response (checked 25 Oct): `{"query", "mode", "chunker", "k", "results": [{"rank", "score", "chunk_id", "doc_id", "url", "title", "heading_path", "token_count", "content"}], "query_embedding": null, "cost_usd", "latency_ms"}`. Example `doc_id`: `tasks/administer-cluster/safely-drain-node`.

```
Passage      = {"text": str,          # from content
                "source": str,        # from doc_id: the stable identifier, used for scoring
                "url": str, "title": str,
                "heading": str,       # from heading_path
                "chunk_id": str, "score": float}
SearchResult = {"passages": [Passage], "cost_usd": float, "latency_ms": float}
```
"Correct doc cited" is scored on `source` (doc_id), never on url (trailing slashes and domain variants). Several chunks can come from the same doc (the example returned three chunks of one page), so de-duplicate by `source` when counting citations. Add `cost_usd` into the run's cost.

## get_issue_thread
```
IssueThread = {"number": int, "title": str, "author": str, "created_at": str,
               "labels": [str], "body": str,
               "comments": [{"author": str, "created_at": str, "body": str}],
               "thread_truncated": bool}
```

## extract_error_signals
```
Signals = {"versions": [str], "k8s_reasons": [str], "error_lines": [str], "stack_traces": [str]}
```

## Agent output
```
TriageNote = {"issue_url": str,
              "outcome": "answered" | "cant_tell" | "step_cap" | "error",
              "category": str | None,        # from the category set I define before labelling
              "likely_cause": str | None,
              "citations": [{"source": str}],  # must be sources returned by search_docs in this run
              "next_step": str | None,
              "steps": int, "cost_usd": float, "tool_calls": [{"tool": str, "ok": bool}]}
```
Rule: a citation that did not come from a search_docs result in the same run is invalid and counts as a failure.

## Eval
```
EvalCase   = {"id": str, "issue_url": str, "answerable": bool, "category": str,
              "gold_docs": [str],            # empty when answerable is false
              "notes"?: str}
EvalResult = {"case_id": str, "system": "agent" | "baseline", "backend": "es" | "inproc",
              "outcome": str, "category": str | None, "cited_docs": [str],
              "correct_category": bool, "correct_doc_cited": bool,
              "steps": int, "cost_usd": float, "failed": bool}
```

## Open for me to decide before labelling (evening 3)
- The category set (same labelling method as Failure Forensics). Claude Code must not invent one.
- What counts as "correct doc cited": exact page match, or any gold page among the cited ones.
