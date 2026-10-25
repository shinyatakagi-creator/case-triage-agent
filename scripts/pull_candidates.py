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
