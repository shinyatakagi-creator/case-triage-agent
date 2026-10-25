"""GitHub issue fetching plus the answer-leakage rule (cut at first maintainer reply)."""
import os
import re

import httpx

from triage.contracts import IssueThread

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


def cut_at_first_maintainer(issue: dict, comments: list[dict]) -> IssueThread:
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
