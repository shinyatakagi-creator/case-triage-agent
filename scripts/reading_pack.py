"""Write one markdown reading pack per issue number to data/pack/<number>.md.

Usage (from repo root, env loaded): python -m scripts.reading_pack 141240 135321 ...

Read-only: no model calls, no verdicts, no ranking, no hints. The thread is exactly what
get_issue_thread returns (cut at the first maintainer reply, bots dropped), so the pack never
shows more than the agent would see. The docs hits are search_docs(title, hybrid, k=5).
"""
import re
import sys
from pathlib import Path

import httpx

from triage import server
from triage.thread import fetch_issue

OUT_DIR = Path("data/pack")
SNIPPET_CHARS = 200
TOP_K = 5
ISSUE_URL = "https://github.com/kubernetes/kubernetes/issues/{}"


def fence(text: str) -> str:
    """Fenced block that cannot be closed early by backticks inside the text."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text\n{text}\n{ticks}"


def snippet(text: str) -> str:
    return " ".join(text.split())[:SNIPPET_CHARS]


def get_state(number: int) -> str:
    """Issue state, which IssueThread does not carry. Errors come back as text, not raised."""
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            return fetch_issue(number, client).get("state", "unknown")
    except Exception as e:  # read-only helper: a missing state must not stop the pack
        return f"unknown ({type(e).__name__})"


def render(number: int, state: str, thread: dict, hits: dict) -> str:
    url = ISSUE_URL.format(number)
    if "error" in thread:
        head = [f"# Issue {number}", "", f"URL: {url}", f"State: {state}", "",
                f"get_issue_thread returned an error: `{thread['error']}` {thread.get('detail', '')}".rstrip()]
        return "\n".join(head) + "\n" + _docs_section(None, hits)
    lines = [f"# {thread['title']}", "", f"URL: {url}", f"State: {state}", "",
             f"Author: {thread['author']}  ", f"Created: {thread['created_at']}  ",
             f"Labels: {', '.join(thread['labels']) or '(none)'}  ",
             f"thread_truncated: {thread['thread_truncated']}", "",
             "## Issue body", "", fence(thread["body"]), "",
             f"## Comments before the first maintainer reply ({len(thread['comments'])})", ""]
    for c in thread["comments"]:
        lines += [f"### {c['author']} at {c['created_at']}", "", fence(c["body"]), ""]
    if not thread["comments"]:
        lines += ["(none)", ""]
    return "\n".join(lines) + "\n" + _docs_section(thread["title"], hits)


def _docs_section(title: str | None, hits: dict) -> str:
    lines = [f"## search_docs top {TOP_K} (query = issue title, mode hybrid)", ""]
    if title is None:
        lines.append("(not run: no title because the thread fetch failed)")
    elif "error" in hits:
        lines.append(f"search_docs returned an error: `{hits['error']}` {hits.get('detail', '')}".rstrip())
    else:
        for i, p in enumerate(hits["passages"][:TOP_K], 1):
            lines += [f"{i}. `{p['source']}`  ", f"   heading: {p['heading']}  ", f"   {snippet(p['text'])}", ""]
    return "\n".join(lines) + "\n"


def build_pack(number: int, out_dir: Path = OUT_DIR, thread_tool=None, search_tool=None, state_fn=None) -> Path:
    thread_tool = thread_tool or server.get_issue_thread
    search_tool = search_tool or server.search_docs
    state_fn = state_fn or get_state
    thread = thread_tool(ISSUE_URL.format(number))
    hits = search_tool(thread["title"], TOP_K) if "error" not in thread else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{number}.md"
    path.write_text(render(number, state_fn(number), thread, hits))
    return path


def main(argv: list[str]) -> int:
    if not argv or not all(a.isdigit() for a in argv):
        print("usage: python -m scripts.reading_pack <issue number> ...", file=sys.stderr)
        return 2
    for a in argv:
        path = build_pack(int(a))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
