# case-triage-agent: evening 1 build guide

Time: about 2.5 to 3 hours. Do this after the Supabase screen on Wed 28 Oct.
Scope: case-triage-agent-scope.md (v3). The retrieval hosting decision is already made (Option A), so tonight is only data and the first MCP server.

## What "done" looks like tonight

1. New repo `case-triage-agent` with the three tools running as an MCP server over stdio.
2. Unit tests passing for the two pure tools (URL check, leakage rule, signal extraction).
3. `data/candidates.jsonl` with about 100 kind/support candidates, and the numbers below written into the scope file.
4. A decision on whether the answerable pool is big enough, and a second pull for the unanswerable pool.

The code below was run once for the unit tests, and the server was started and its three tools listed and called (bad URL and signal extraction). The live GitHub calls and the call to rag-hybrid `/search` have NOT been run against the real services, so expect to fix small things there.

## 0. Before you start

- rag-hybrid stack is up: in the rag-hybrid repo, `docker compose up -d`. The API is on 127.0.0.1:8000 (set `API_HOST_PORT` in its `.env` if 8000 is taken).
- Quick check, no cost (BM25 only):
  `curl -s -X POST http://127.0.0.1:8000/search -H 'content-type: application/json' -d '{"query":"kubectl drain","mode":"bm25","k":3}'`
  Keep that output, you need the response shape in step 5.
- GitHub token: github.com, Settings, Developer settings, Fine-grained tokens, new token. Resource owner is your account, repository access "Public repositories (read-only)", no extra permissions. Short expiry (30 days is fine).

## 1. Repo and environment

```bash
mkdir case-triage-agent && cd case-triage-agent
git init
python3 -m venv .venv && source .venv/bin/activate
pip install mcp httpx pytest
pip freeze | grep -iE "^(mcp|httpx|pytest)==" > requirements.txt
mkdir -p triage scripts tests data
touch triage/__init__.py
printf '.venv/\n.env\ndata/\n__pycache__/\n.pytest_cache/\n' > .gitignore
printf 'GITHUB_TOKEN=paste-token-here\nRAG_API_URL=http://127.0.0.1:8000\n' > .env
```

Note on the MCP SDK: this was written against mcp 2.3.0, where the server class is `MCPServer` (`from mcp.server.mcpserver import MCPServer`). In 1.x it was `FastMCP` from `mcp.server.fastmcp`. If `pip` gives you a different major version and the import fails, that is why. Keep requirements.txt so the version stays fixed.

Load the env file into your shell before running anything (the code reads real environment variables, not the file):

```bash
set -a; source .env; set +a
```

`data/` is in .gitignore for now. Candidate files contain real issue text and will get big. Commit only the final labelled set later.

## 2. The code

Create these five files.

### triage/signals.py

```python
"""extract_error_signals: pull versions, error lines, stack traces and k8s status reasons from text."""
import re

K8S_VERSION_RE = re.compile(r"\bv1\.\d{1,2}\.\d{1,2}(?:[-+][\w.]+)?\b")
ERROR_LINE_RE = re.compile(
    r"\b(error|errors|failed|failure|panic|fatal|unable to|cannot|can't|forbidden|"
    r"unauthorized|timeout|timed out|denied|refused|not found)\b",
    re.IGNORECASE,
)
K8S_REASON_RE = re.compile(
    r"\b(CrashLoopBackOff|ImagePullBackOff|ErrImagePull|OOMKilled|Evicted|Pending|"
    r"ContainerCreating|FailedScheduling|FailedMount|NodeNotReady|Terminating|"
    r"CreateContainerConfigError|Unschedulable)\b"
)
TRACE_START_RE = re.compile(r"^(panic:|goroutine \d+ \[|Traceback \(most recent call last\):)")

MAX_ERROR_LINES = 15
MAX_TRACES = 3
TRACE_LINES = 12
MAX_LINE_CHARS = 300


def _dedupe(items):
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_error_signals(text: str) -> dict:
    """Return structured signals found in free text. Pure function, no network."""
    lines = text.splitlines()

    versions = _dedupe(K8S_VERSION_RE.findall(text))
    reasons = _dedupe(K8S_REASON_RE.findall(text))

    error_lines = _dedupe(
        line.strip()[:MAX_LINE_CHARS]
        for line in lines
        if line.strip() and ERROR_LINE_RE.search(line)
    )[:MAX_ERROR_LINES]

    traces = []
    for i, line in enumerate(lines):
        if TRACE_START_RE.match(line.strip()):
            block = [l.rstrip()[:MAX_LINE_CHARS] for l in lines[i : i + TRACE_LINES]]
            traces.append("\n".join(block))
            if len(traces) == MAX_TRACES:
                break

    return {
        "versions": versions,
        "k8s_reasons": reasons,
        "error_lines": error_lines,
        "stack_traces": traces,
    }
```

### triage/thread.py

This holds the answer-leakage rule from the scope. Everything from the first maintainer reply onward is withheld. Two details that matter: the k8s bots (k8s-ci-robot and friends) are normal user accounts, so they are matched by login, and an issue author who happens to be a MEMBER does not count as a maintainer replying to their own issue.

```python
"""GitHub issue fetching plus the answer-leakage rule (cut at first maintainer reply)."""
import os
import re

import httpx

REPO = "kubernetes/kubernetes"
API = "https://api.github.com"
ISSUE_URL_RE = re.compile(r"^https://github\.com/kubernetes/kubernetes/issues/(\d+)/?$")

# k8s-ci-robot and friends are normal user accounts, so match on login, not account type.
BOT_LOGINS = {"k8s-ci-robot", "k8s-triage-robot", "fejta-bot", "k8s-github-robot"}
MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}

MAX_BODY_CHARS = 6000
MAX_COMMENT_PAGES = 3


class RateLimited(Exception):
    pass


class NotFound(Exception):
    pass


def parse_issue_url(url: str) -> int:
    m = ISSUE_URL_RE.match(url.strip())
    if not m:
        raise ValueError("only https://github.com/kubernetes/kubernetes/issues/<n> URLs are accepted")
    return int(m.group(1))


def _login(item: dict) -> str:
    return (item.get("user") or {}).get("login", "")


def is_bot(item: dict) -> bool:
    user = item.get("user") or {}
    return (
        user.get("login") in BOT_LOGINS
        or user.get("login", "").endswith("[bot]")
        or user.get("type") == "Bot"
    )


def is_maintainer_reply(comment: dict, issue_author: str) -> bool:
    return (
        not is_bot(comment)
        and _login(comment) != issue_author
        and comment.get("author_association") in MAINTAINER_ASSOCIATIONS
    )


def cut_at_first_maintainer(issue: dict, comments: list[dict]) -> dict:
    """Keep the issue body and human comments before the first maintainer reply.

    Maintainer replies often contain the answer or a doc link, so everything from
    the first one onward is withheld. Bot comments are dropped.
    """
    author = _login(issue)
    kept, truncated = [], False
    for c in comments:
        if is_bot(c):
            continue
        if is_maintainer_reply(c, author):
            truncated = True
            break
        kept.append(
            {
                "author": _login(c),
                "created_at": c.get("created_at"),
                "body": (c.get("body") or "")[:MAX_BODY_CHARS],
            }
        )
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "author": author,
        "created_at": issue.get("created_at"),
        "labels": [l["name"] for l in issue.get("labels", [])],
        "body": (issue.get("body") or "")[:MAX_BODY_CHARS],
        "comments": kept,
        "thread_truncated": truncated,
    }


def _headers() -> dict:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "case-triage-agent",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _check(resp: httpx.Response) -> None:
    if resp.status_code == 404:
        raise NotFound(resp.url)
    if resp.status_code in (403, 429) and resp.headers.get("x-ratelimit-remaining") == "0":
        raise RateLimited(resp.headers.get("x-ratelimit-reset", "unknown"))
    resp.raise_for_status()


def fetch_issue(number: int, client: httpx.Client) -> dict:
    r = client.get(f"{API}/repos/{REPO}/issues/{number}", headers=_headers())
    _check(r)
    return r.json()


def fetch_comments(number: int, client: httpx.Client) -> list[dict]:
    url = f"{API}/repos/{REPO}/issues/{number}/comments"
    params = {"per_page": 100}
    out = []
    for _ in range(MAX_COMMENT_PAGES):
        r = client.get(url, headers=_headers(), params=params)
        _check(r)
        out.extend(r.json())
        nxt = r.links.get("next", {}).get("url")
        if not nxt:
            break
        url, params = nxt, None
    return out
```

### triage/server.py

```python
"""MCP server exposing the three case-triage tools over stdio."""
import os

import httpx
from mcp.server.mcpserver import MCPServer

from triage.signals import extract_error_signals as _extract
from triage.thread import (
    NotFound,
    RateLimited,
    cut_at_first_maintainer,
    fetch_comments,
    fetch_issue,
    parse_issue_url,
)

RAG_API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(15.0)

mcp = MCPServer("case-triage")


@mcp.tool()
def search_docs(query: str, k: int = 5) -> dict:
    """Search the Kubernetes docs (hybrid BM25 + kNN). Returns passages with citations."""
    try:
        r = httpx.post(
            f"{RAG_API_URL}/search",
            json={"query": query, "mode": "hybrid", "k": k},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        # TODO(evening 1): print r.json() once, then normalise to
        # {"passages": [{"text", "source", "heading"}]} so the agent sees a stable shape.
        return r.json()
    except httpx.TimeoutException:
        return {"error": "timeout", "tool": "search_docs"}
    except httpx.HTTPError as e:
        return {"error": "search_failed", "detail": str(e)[:200]}


@mcp.tool()
def get_issue_thread(issue_url: str) -> dict:
    """Fetch a kubernetes/kubernetes issue: body and human comments up to the first maintainer reply."""
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
        return cut_at_first_maintainer(issue, comments)
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
```

Errors come back as small dicts (`{"error": "timeout"}` etc.) instead of exceptions. The agent needs to see them as tool results, and the failure tests on evening 6 will assert on exactly these shapes. Do not change the keys casually.

### scripts/pull_candidates.py

```python
"""Pull kind/support candidates from kubernetes/kubernetes and report how many have a real human thread.

Usage (from repo root): python -m scripts.pull_candidates --limit 100 --out data/candidates.jsonl
"""
import argparse
import json
import statistics
import sys

import httpx

from triage.thread import (
    API,
    REPO,
    _check,
    _headers,
    _login,
    fetch_comments,
    is_bot,
    is_maintainer_reply,
    RateLimited,
)

MIN_COMMENTS = 3  # GitHub's count includes bot comments, so this is only a first filter


def list_issues(label: str, want: int, client: httpx.Client):
    url = f"{API}/repos/{REPO}/issues"
    params = {"labels": label, "state": "all", "sort": "created", "direction": "desc", "per_page": 100}
    got = 0
    while url and got < want:
        r = client.get(url, headers=_headers(), params=params)
        _check(r)
        for item in r.json():
            if "pull_request" in item:
                continue
            if item.get("comments", 0) < MIN_COMMENTS:
                continue
            yield item
            got += 1
            if got >= want:
                return
        url = r.links.get("next", {}).get("url")
        params = None


def summarise(issue: dict, comments: list[dict]) -> dict:
    author = _login(issue)
    human = [c for c in comments if not is_bot(c)]
    others = [c for c in human if _login(c) != author]
    return {
        "number": issue["number"],
        "title": issue["title"],
        "state": issue["state"],
        "created_at": issue["created_at"],
        "labels": [l["name"] for l in issue.get("labels", [])],
        "body_chars": len(issue.get("body") or ""),
        "comments_total": len(comments),
        "human_comments": len(human),
        "human_comments_by_others": len(others),
        "has_maintainer_reply": any(is_maintainer_reply(c, author) for c in comments),
        "url": issue["html_url"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="kind/support")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="data/candidates.jsonl")
    args = ap.parse_args()

    rows = []
    with httpx.Client(timeout=20.0) as client:
        try:
            for issue in list_issues(args.label, args.limit, client):
                comments = fetch_comments(issue["number"], client)
                rows.append(summarise(issue, comments))
        except RateLimited as e:
            print(f"rate limited (reset {e}); is GITHUB_TOKEN set?", file=sys.stderr)
        except httpx.HTTPStatusError as e:
            print(f"GitHub returned {e.response.status_code} for {e.request.url}", file=sys.stderr)

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n = len(rows)
    if not n:
        print("no rows")
        return
    substantive = [r for r in rows if r["human_comments_by_others"] >= 1 and r["body_chars"] >= 200]
    print(f"candidates pulled:                    {n}")
    print(f"body >= 200 chars:                    {sum(r['body_chars'] >= 200 for r in rows)}")
    print(f">=1 human comment by someone else:    {sum(r['human_comments_by_others'] >= 1 for r in rows)}")
    print(f"has maintainer reply:                 {sum(r['has_maintainer_reply'] for r in rows)}")
    print(f"substantive (both of the above):      {len(substantive)}")
    print(f"median body chars:                    {statistics.median(r['body_chars'] for r in rows):.0f}")


if __name__ == "__main__":
    main()
```

### tests/test_triage.py

```python
import pytest

from triage.signals import extract_error_signals
from triage.thread import cut_at_first_maintainer, parse_issue_url


def user(login, assoc="NONE", typ="User"):
    return {"login": login, "type": typ}, assoc


def comment(login, body, assoc="NONE", typ="User"):
    return {"user": {"login": login, "type": typ}, "author_association": assoc, "body": body, "created_at": "t"}


ISSUE = {
    "number": 1,
    "title": "pod stuck",
    "user": {"login": "alice", "type": "User"},
    "created_at": "t",
    "labels": [{"name": "kind/support"}],
    "body": "my pod is Pending on v1.28.3",
}


def test_parse_issue_url_ok():
    assert parse_issue_url("https://github.com/kubernetes/kubernetes/issues/12345") == 12345
    assert parse_issue_url("https://github.com/kubernetes/kubernetes/issues/12345/") == 12345


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/kubernetes/kubernetes/pull/1",
        "https://github.com/other/repo/issues/1",
        "http://github.com/kubernetes/kubernetes/issues/1",
        "https://github.com/kubernetes/kubernetes/issues/1?x=1",
        "https://evil.example/kubernetes/kubernetes/issues/1",
    ],
)
def test_parse_issue_url_rejects(url):
    with pytest.raises(ValueError):
        parse_issue_url(url)


def test_cut_at_first_maintainer_withholds_answer():
    comments = [
        comment("k8s-ci-robot", "@alice: This issue is currently awaiting triage.", "MEMBER"),
        comment("bob", "same here, which CNI?", "NONE"),
        comment("alice", "calico", "NONE"),
        comment("carol", "see https://kubernetes.io/docs/foo", "MEMBER"),
        comment("dave", "thanks", "NONE"),
    ]
    out = cut_at_first_maintainer(ISSUE, comments)
    assert [c["author"] for c in out["comments"]] == ["bob", "alice"]
    assert out["thread_truncated"] is True


def test_issue_author_who_is_member_is_not_a_maintainer_reply():
    comments = [comment("alice", "more detail", "MEMBER"), comment("bob", "try X", "NONE")]
    out = cut_at_first_maintainer(ISSUE, comments)
    assert len(out["comments"]) == 2
    assert out["thread_truncated"] is False


def test_signals_basic():
    text = (
        "kubectl version: GitVersion:\"v1.28.3\"\n"
        "pod is CrashLoopBackOff\n"
        "Error from server (Forbidden): pods is forbidden\n"
        "panic: runtime error: index out of range\n"
        "goroutine 1 [running]:\nmain.main()\n"
        "all good here\n"
    )
    s = extract_error_signals(text)
    assert s["versions"] == ["v1.28.3"]
    assert "CrashLoopBackOff" in s["k8s_reasons"]
    assert any("Forbidden" in l for l in s["error_lines"])
    assert len(s["stack_traces"]) == 2
    assert not any("all good" in l for l in s["error_lines"])


def test_signals_empty():
    s = extract_error_signals("")
    assert s == {"versions": [], "k8s_reasons": [], "error_lines": [], "stack_traces": []}
```

## 3. Run the tests

```bash
python -m pytest -q
```

Expected: 10 passed. If a test fails on the leakage rule, stop and fix it before pulling any data. Every number in the eval depends on that rule.

## 4. Pull candidates and read the numbers

```bash
python -m scripts.pull_candidates --limit 100 --out data/candidates.jsonl
```

It prints something like:

```
candidates pulled:                    100
body >= 200 chars:                    ...
>=1 human comment by someone else:    ...
has maintainer reply:                 ...
substantive (both of the above):      ...
median body chars:                    ...
```

What to do with the result (these thresholds are my rule of thumb, adjust if you disagree):

- You need 25 answerable cases in the final 40, and you will reject a lot of substantive threads when you label (the docs just do not answer them). Expect to throw away about half. So you want at least 50 substantive out of 100.
- 50 or more: fine, carry on.
- 25 to 49: pull again with `--limit 250` and see if the count scales. If it does, carry on; if the pool is thin, say so in the scope and consider adding a second source for answerable cases.
- Under 25: the kind/support pool is too thin for this design. Stop and rethink the data source before building more. This is the "substantive threads NOT verified" risk in the scope, and tonight is when it gets answered.

Things to know about this pull:

- The listing is newest first, so the sample skews recent. The docs index is pinned to a commit in rag-hybrid. Check that commit date; issues much newer than it may refer to docs the index does not have, and you will mark those unanswerable by mistake.
- "has maintainer reply" matters for the leakage rule. Threads with no maintainer reply never get truncated, so check a few by hand to make sure nobody pasted the full fix in the thread anyway.
- The count of comments GitHub returns includes bot comments. That is why the script re-counts human ones.

Second pull for the unanswerable pool:

```bash
python -m scripts.pull_candidates --label "kind/bug,kind/support" --limit 60 --out data/candidates_bug_support.jsonl
python -m scripts.pull_candidates --label "kind/bug" --limit 40 --out data/candidates_bug.jsonl
```

A comma in `--label` means the issue must carry both labels. Keep the 15 unanswerable cases to a mix of the sources in the scope (bug plus support, third-party problems, pure bug, feature request). Feature requests need `--label kind/feature`.

## 5. Check the /search response and fix search_docs

The rag-hybrid docs give the request (`query`, `mode` of bm25, dense or hybrid, `k`) but not the response shape. Look at what the curl in step 0 returned, then edit `search_docs` in triage/server.py so it returns one stable shape:

```python
{"passages": [{"text": "...", "source": "<doc path or url>", "heading": "..."}]}
```

The agent prompt on evening 2 and the "correct doc cited" metric both depend on `source` being a stable doc identifier, so settle on it now. Also run one `hybrid` request to see it works. That one costs about $0.0000002 for the embedding, per the rag-hybrid docs.

## 6. Smoke test the server

Direct call, no client needed (this is how the checks above were run):

```bash
python - <<'EOF'
import asyncio, triage.server as s

async def main():
    print([t.name for t in await s.mcp.list_tools()])
    print(await s.mcp.call_tool("extract_error_signals", {"text": "pod CrashLoopBackOff on v1.28.3"}))
    print(await s.mcp.call_tool("get_issue_thread", {"issue_url": "https://example.com/x"}))
    # needs token and network:
    print(await s.mcp.call_tool("get_issue_thread", {"issue_url": "https://github.com/kubernetes/kubernetes/issues/<pick-a-candidate-number>"}))
    # needs rag-hybrid up:
    print(await s.mcp.call_tool("search_docs", {"query": "pod stuck in Pending", "k": 3}))

asyncio.run(main())
EOF
```

Check by eye on the real thread call: no maintainer comment appears, `thread_truncated` is true for a thread that has one, and no bot text is in `comments`.

Optional: add it to Claude Code with `claude mcp add case-triage -- $(pwd)/.venv/bin/python -m triage.server` and ask it to triage an issue URL. The env vars (`GITHUB_TOKEN`, `RAG_API_URL`) need to be passed to the server; check `claude mcp add --help` for the flag on your version. Not needed tonight.

## 7. End of evening

- [ ] Tests pass, first commit made (`git add -A && git commit`), `.env` NOT in it (`git status` first).
- [ ] Numbers from step 4 pasted into the scope file under "Unverified items", with the answer to "is the pool big enough".
- [ ] `search_docs` returns the stable shape from step 5.
- [ ] Unanswerable-pool pulls done.
- [ ] Created the public GitHub repo and pushed (do this after you have checked no token or issue dump is in the history).

Evening 2 starts from here: finish the tools, write the agent loop with the 6-step cap, and run it on three or four candidates by hand.
