"""MCP server exposing the three case-triage tools over stdio."""
import os

import httpx
from mcp.server.mcpserver import MCPServer

from triage.contracts import Passage, SearchResult
from triage.retrieval import inproc
from triage.signals import extract_error_signals as _extract
from triage.thread import (
    NotFound,
    RateLimited,
    cut_at_first_reply,
    fetch_comments,
    fetch_issue,
    parse_issue_url,
)

RAG_API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(15.0)
MAX_QUERY_CHARS = 500  # rag-hybrid /search returns 422 above this, and for any control character (CR/LF)


def clean_query(query: str) -> str:
    """Whitespace collapsed (no control characters) and cut to what /search accepts."""
    return " ".join(query.split())[:MAX_QUERY_CHARS]

mcp = MCPServer("case-triage")


def normalise_search(raw: dict) -> SearchResult:
    """Map rag-hybrid's /search response to the frozen SearchResult shape."""
    passages = [
        Passage(
            text=r["content"],
            source=r["doc_id"],
            url=r["url"],
            title=r["title"],
            heading=r["heading_path"],
            chunk_id=r["chunk_id"],
            score=r["score"],
        )
        for r in raw["results"]
    ]
    # bm25 responses carry no query_embedding at all, so absent means not a cache hit.
    cache_hit = bool((raw.get("query_embedding") or {}).get("cache_hit", False))
    return SearchResult(
        passages=passages, cost_usd=raw["cost_usd"], latency_ms=raw["latency_ms"], cache_hit=cache_hit
    )


@mcp.tool()
def search_docs(query: str, k: int = 5) -> dict:
    """Search the Kubernetes docs (hybrid BM25 + kNN). Returns passages with citations."""
    try:
        if os.environ.get("RETRIEVAL_BACKEND", "es") == "inproc":
            return normalise_search(inproc.search(clean_query(query), k))
        r = httpx.post(
            f"{RAG_API_URL}/search",
            json={"query": clean_query(query), "mode": "hybrid", "k": k},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return normalise_search(r.json())
    except httpx.TimeoutException:
        return {"error": "timeout", "tool": "search_docs"}
    except httpx.HTTPError as e:
        return {"error": "search_failed", "detail": str(e)[:200]}
    except RuntimeError as e:  # inproc: no index or no OPENAI_API_KEY
        return {"error": "search_failed", "detail": str(e)[:200]}
    except (KeyError, ValueError) as e:  # response not in the shape we checked on 25 Oct
        return {"error": "search_failed", "detail": f"unexpected /search response: {e!r}"[:200]}


@mcp.tool()
def get_issue_thread(issue_url: str) -> dict:
    """Fetch a kubernetes/kubernetes issue: body and the author's own comments before anyone else replies."""
    try:
        number = parse_issue_url(issue_url)
    except ValueError as e:
        return {"error": "bad_url", "detail": str(e)}
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            issue = fetch_issue(number, client)
            if "pull_request" in issue:
                return {"error": "not_an_issue", "detail": "this URL is a pull request"}
            comments = fetch_comments(number, client)
        return cut_at_first_reply(issue, comments)
    except NotFound:
        return {"error": "not_found"}
    except RateLimited as e:
        return {"error": "rate_limited", "reset": str(e)}
    except httpx.TimeoutException:
        return {"error": "timeout", "tool": "get_issue_thread"}
    except httpx.HTTPError as e:
        return {"error": "github_failed", "detail": str(e)[:200]}


@mcp.tool()
def extract_error_signals(text: str) -> dict:
    """Parse versions, error lines, stack traces and k8s status reasons out of issue text."""
    return _extract(text)


if __name__ == "__main__":
    mcp.run()
