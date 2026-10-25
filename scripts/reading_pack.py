"""Write one markdown reading pack per issue number to data/pack/<number>.md.

Usage (from repo root, env loaded): python -m scripts.reading_pack 141240 135321 ...

Read-only: no model calls, no verdicts, no ranking, no hints. The thread is exactly what
get_issue_thread returns (cut at the first maintainer reply, bots dropped), so the pack never
shows more than the agent would see. The docs hits are search_docs(title, hybrid, k=5); the first 3 also get their full chunk text, then
lines of the top 5 chunks that share words with the title and the head of the body (plain word overlap, no judgement).
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
MAX_EVIDENCE_LINES = 5
MAX_SEARCH_WORDS = 8
BODY_FALLBACK_CHARS = 1500
BODY_STOP_HEADINGS = ("### kubernetes version", "### environment")
EVIDENCE_LINE_CHARS = 200
MIN_WORD_CHARS = 4
MIN_WORD_HITS = 2
# Words too common to say anything about a match; the length filter drops the short ones already.
STOPWORDS = frozenset("""
about after again also because been before being cannot could does doing done each either else even
ever from have having here into just like many more most much must only other over same should some
such than that their them then there these they this those through under until upon very want were what
when where which while will with would your
issue problem error please kindly request solution thanks thank regards team dear hello following using
trying getting version cluster kubernetes kubectl
""".split())
FULL_TEXT_K = 3  # the first 3 of the top-5 hits also get their whole chunk
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


WORD_RE = re.compile(r"-{0,2}[a-z0-9][a-z0-9._/-]*")  # keeps resolv.conf, change-cause, --record whole
SPECIAL = set("._/-")


def body_head(body: str) -> str:
    """Text before the first Kubernetes-version / Environment heading, else the first BODY_FALLBACK_CHARS."""
    low = body.lower()
    cuts = [i for h in BODY_STOP_HEADINGS if (i := low.find(h)) != -1]
    return body[:min(cuts)] if cuts else body[:BODY_FALLBACK_CHARS]


def _tokens(text: str) -> list[str]:
    return [t.rstrip("._/-") for t in WORD_RE.findall(text.lower())]  # a sentence-ending dot is not part of a word


def word_weight(word: str, title_set: set[str]) -> int:
    """Higher for words in the title, words with . - / _ in them, and long words. Used to rank WORDS only."""
    return (3 if word in title_set else 0) + (2 if SPECIAL & set(word) else 0) + (1 if len(word) > 8 else 0)


def _with_segments(tokens: list[str]) -> list[str]:
    """Each token followed by its '/'-separated segments, so a path or URL also yields its parts.

    Short segments and stopwords are filtered later like any other word (etc/resolv.conf -> etc/resolv.conf,
    resolv.conf; "etc" is under MIN_WORD_CHARS)."""
    out = []
    for t in tokens:
        out.append(t)
        if "/" in t:
            out += [seg for seg in (x.strip("._-") for x in t.split("/")) if seg]
    return out


def search_words(title: str, body: str) -> list[str]:
    """Top MAX_SEARCH_WORDS content words from the title and the head of the body, heaviest first (ties: first seen)."""
    title_toks = _with_segments(_tokens(title))
    seen = list(dict.fromkeys(title_toks + _with_segments(_tokens(body_head(body)))))
    words = [w for w in seen if len(w) >= MIN_WORD_CHARS and w not in STOPWORDS]
    title_set = set(title_toks)
    return sorted(words, key=lambda w: -word_weight(w, title_set))[:MAX_SEARCH_WORDS]  # sorted() is stable


def _has_word(line: str, word: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", line) is not None


def evidence_lines(text: str, words: list[str]) -> list[str]:
    """Up to MAX_EVIDENCE_LINES lines containing at least MIN_WORD_HITS of the words, most words first (ties: chunk order)."""
    scored = []
    for line in text.splitlines():
        line = line.strip()
        n = sum(_has_word(line.lower(), w) for w in words)
        if line and n >= MIN_WORD_HITS:
            scored.append((n, line[:EVIDENCE_LINE_CHARS]))
    scored.sort(key=lambda t: -t[0])  # stable: equal counts keep chunk order
    return [ln for _, ln in scored[:MAX_EVIDENCE_LINES]]


def _evidence_section(title: str, body: str, hits: dict) -> str:
    words = search_words(title, body)
    lines = ["## Evidence lines", "",
             f"Title/body words used: {', '.join(words) or '(none)'}. Lines of each chunk containing at least "
             f"{MIN_WORD_HITS} of them (up to {MAX_EVIDENCE_LINES} per hit, most words first, cut at "
             f"{EVIDENCE_LINE_CHARS} characters).", ""]
    for i, p in enumerate(hits["passages"][:TOP_K], 1):
        found = evidence_lines(p["text"], words)
        lines.append(f"{i}. `{p['source']}`")
        lines += [f"   - {ln}" for ln in found] or ["   - no matching lines"]
        lines.append("")
    return "\n".join(lines) + "\n"


def render(number: int, state: str, thread: dict, hits: dict) -> str:
    url = ISSUE_URL.format(number)
    if "error" in thread:
        head = [f"# Issue {number}", "", f"URL: {url}", f"State: {state}", "",
                f"get_issue_thread returned an error: `{thread['error']}` {thread.get('detail', '')}".rstrip()]
        return "\n".join(head) + "\n" + _docs_section(None, "", hits)
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
    return "\n".join(lines) + "\n" + _docs_section(thread["title"], thread["body"], hits)


def _docs_section(title: str | None, body: str, hits: dict) -> str:
    lines = [f"## search_docs top {TOP_K} (query = issue title, mode hybrid)", ""]
    if title is None:
        lines.append("(not run: no title because the thread fetch failed)")
    elif "error" in hits:
        lines.append(f"search_docs returned an error: `{hits['error']}` {hits.get('detail', '')}".rstrip())
    else:
        for i, p in enumerate(hits["passages"][:TOP_K], 1):
            lines += [f"{i}. `{p['source']}`  ", f"   heading: {p['heading']}  ", f"   {snippet(p['text'])}", ""]
        lines += [f"## Full text of the top {FULL_TEXT_K} chunks", ""]
        for i, p in enumerate(hits["passages"][:FULL_TEXT_K], 1):
            lines += ["<details>", f"<summary>{i}. {p['source']} | {p['heading']}</summary>", "",
                      fence(p["text"]), "", "</details>", ""]
        if title is not None:
            lines += ["", _evidence_section(title, body, hits).rstrip("\n")]
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
