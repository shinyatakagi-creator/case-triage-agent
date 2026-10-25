"""Draft yes / no / unsure labels with a model, into data/drafts/labels.jsonl. LIVE and PAID: run only on request.

Usage (env loaded, rag-hybrid up):
    python -m scripts.draft_labels 141240 135321 ...
    python -m scripts.draft_labels --from data/candidates_support_250.jsonl --limit 20 [--max-usd 3.00]

Per issue: the thread exactly as get_issue_thread returns it (cut at the first reply by anyone but the issue author),
three searches (title k=5, key terms from the body k=8, the opening of the body k=8) merged and
de-duplicated by chunk_id, then ONE model call. The model's quotes are checked mechanically; any
failed check turns the label into "unsure". A draft is a draft: nothing here writes data/eval/.
"""
import argparse
import json
import sys
from pathlib import Path

from scripts.reading_pack import ISSUE_URL, search_words
from triage import server
from triage.agent import ask
from triage.llm import AnthropicLLM, Ledger

DRAFT_MODEL = "claude-sonnet-5-5"
DRAFT_DIR = Path("data/drafts")
LABELS_PATH = DRAFT_DIR / "labels.jsonl"
SPEND_PATH = DRAFT_DIR / "spend.json"
LABELS = ("yes", "no", "unsure")
TITLE_K, BODY_K = 5, 8
BODY_QUERY_TERMS = 4      # key terms taken from the body for the second query
BODY_QUERY_CHARS = 300    # opening of the body for the third query
MIN_QUOTE_CHARS = 25      # a shorter quote proves nothing
MAX_QUOTED_DOCS = 3
BODY_HEAD_LINES = 15      # kept in the draft so the review step needs no network

SYSTEM = """You help label Kubernetes GitHub issues for an evaluation set.
You get one issue (its body and the issue author's own comments before anyone else replied) and a numbered list of
documentation chunks returned by a search. Decide: does a page in these results show or state the answer, so
a reader could solve the issue from it without the maintainer's reply?
- "yes": give one to three doc_ids, taken only from the provided results, and for each an exact quote
  (one sentence, copied character for character from that chunk) that shows or states the answer.
- "no": the results do not contain the answer.
- "unsure": you cannot tell.
Always give a one-line reason. Do not assign a category. Answer by calling submit_label."""

SUBMIT_SPEC = {
    "name": "submit_label",
    "description": "Submit the label for this issue.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(LABELS)},
            "doc_ids": {"type": "array", "items": {"type": "string"}},
            "quotes": {"type": "array", "items": {"type": "string"},
                       "description": "quotes[i] is an exact sentence from a chunk of doc_ids[i]"},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
    },
}


def guard_out_path(path) -> Path:
    """Drafts never go under data/eval/: only the owner promotes labels there."""
    path = Path(path).resolve()
    eval_dir = Path("data/eval").resolve()
    if path == eval_dir or eval_dir in path.parents:
        raise ValueError(f"refusing to write under data/eval/: {path}")
    return path


def norm(text: str) -> str:
    return " ".join(text.split())


def queries(title: str, body: str) -> list[tuple[str, int]]:
    """Title (k=5), key terms from the body (k=8), opening of the body (k=8); empty ones are skipped."""
    terms = " ".join(search_words("", body)[:BODY_QUERY_TERMS])
    opening = norm(body)[:BODY_QUERY_CHARS]
    return [(q, k) for q, k in ((title, TITLE_K), (terms, BODY_K), (opening, BODY_K)) if q.strip()]


def gather(thread: dict, search_tool) -> tuple[list[dict], float, list[str]]:
    """Merged chunks (first seen wins, de-duplicated by chunk_id), search cost, and failures."""
    chunks, seen, cost, failures = [], set(), 0.0, []
    for q, k in queries(thread["title"], thread["body"]):
        r = search_tool(q, k)
        if "error" in r:
            failures.append(f"search failed for query {q[:40]!r}: {r['error']}")
            continue
        cost += r["cost_usd"]
        for p in r["passages"]:
            if p["chunk_id"] not in seen:
                seen.add(p["chunk_id"])
                chunks.append({"doc_id": p["source"], "heading": p["heading"], "text": p["text"],
                               "chunk_id": p["chunk_id"]})
    return chunks, cost, failures


def render_prompt(thread: dict, chunks: list[dict]) -> str:
    lines = [f"Issue title: {thread['title']}", "", "Issue body:", thread["body"], "",
             f"Author's own comments before anyone else replied ({len(thread['comments'])}):"]
    for c in thread["comments"]:
        lines += [f"--- {c['author']}", c["body"]]
    lines += ["", "Documentation chunks:"]
    for i, c in enumerate(chunks, 1):
        lines += [f"[{i}] doc_id={c['doc_id']} | {c['heading']}", c["text"], ""]
    return "\n".join(lines)


def check_label(label, doc_ids, quotes, chunks: list[dict]) -> tuple[str, list[str], list[str], list[str]]:
    """Mechanical checks. Returns (label, doc_ids, quotes, failures); any failure forces "unsure"."""
    if label not in LABELS:
        return "unsure", [], [], [f"invalid label {str(label)[:20]!r}"]
    if label != "yes":
        return label, [], [], []
    doc_ids = doc_ids if isinstance(doc_ids, list) else []
    quotes = quotes if isinstance(quotes, list) else []
    failures = []
    if not 1 <= len(doc_ids) <= MAX_QUOTED_DOCS:
        failures.append(f"yes needs 1-{MAX_QUOTED_DOCS} doc_ids, got {len(doc_ids)}")
    if len(quotes) != len(doc_ids):
        failures.append(f"{len(doc_ids)} doc_ids but {len(quotes)} quotes")
    by_doc: dict[str, list[str]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(norm(c["text"]))
    for d, q in zip(doc_ids, quotes):
        if not isinstance(d, str) or d not in by_doc:
            failures.append(f"doc_id not in this issue's results: {str(d)[:80]}")
        elif not isinstance(q, str) or len(norm(q)) < MIN_QUOTE_CHARS:
            failures.append(f"quote too short or not text for {d}")
        elif not any(norm(q) in t for t in by_doc[d]):
            failures.append(f"quote is not a verbatim substring of any chunk of {d}")
    if failures:
        return "unsure", doc_ids, quotes, failures
    return "yes", doc_ids, quotes, []


def parse_answer(content: list[dict]) -> dict | None:
    for b in content:
        if b.get("type") == "tool_use" and b.get("name") == "submit_label" and isinstance(b.get("input"), dict):
            return b["input"]
    return None


CAP_REASONS = ("per-run cap", "per-day cap")


def draft_one(number: int, llm, ledger, thread_tool, search_tool) -> dict | None:
    """One draft record, or None when the caps stop us before the model call (nothing is recorded then,
    so a later run drafts this issue). The caller has already checked the ledger once."""
    url = ISSUE_URL.format(number)
    rec = {"number": number, "label": "unsure", "doc_ids": [], "quotes": [], "reason": "", "check_failures": [],
           "model": llm.model, "cost_usd": 0.0, "title": None, "url": url, "body_head": [], "model_label": None, "result_doc_ids": []}
    thread = thread_tool(url)
    if "error" in thread:
        rec["check_failures"].append(f"get_issue_thread: {thread['error']}")
        return rec
    rec.update(title=thread["title"], body_head=thread["body"].splitlines()[:BODY_HEAD_LINES])
    chunks, search_cost, failures = gather(thread, search_tool)
    ledger.charge_usd(search_cost)  # search cost counts against the caps like model cost
    rec["cost_usd"] = search_cost
    rec["check_failures"] += failures
    rec["result_doc_ids"] = list(dict.fromkeys(c["doc_id"] for c in chunks))  # lets the review step validate a typed doc_id
    if not chunks:
        rec["check_failures"].append("no search results, so no model call")
        return rec
    got = ask(llm, ledger, SYSTEM, [{"role": "user", "content": render_prompt(thread, chunks)}], [SUBMIT_SPEC])
    rec["cost_usd"] += got.usd
    if got.reason and got.reason.startswith(CAP_REASONS):
        return None
    if got.reason:
        rec["check_failures"].append(f"model: {got.reason}")
        return rec
    ans = parse_answer(got.content)
    if ans is None:
        rec["check_failures"].append("model did not call submit_label")
        return rec
    rec["model_label"] = ans.get("label")
    reason = ans.get("reason")
    rec["reason"] = reason if isinstance(reason, str) else ""
    rec["label"], rec["doc_ids"], rec["quotes"], fails = check_label(
        ans.get("label"), ans.get("doc_ids"), ans.get("quotes"), chunks)
    rec["check_failures"] += fails
    return rec


def done_numbers(path: Path) -> set[int]:
    return {json.loads(line)["number"] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()


def run(numbers: list[int], llm, ledger, out_path=LABELS_PATH, thread_tool=None, search_tool=None) -> dict:
    """Draft each number not already in out_path, appending one line per issue as it finishes.

    Stops cleanly when the ledger refuses the next issue. One issue can overshoot the cap by its own cost."""
    thread_tool = thread_tool or server.get_issue_thread
    search_tool = search_tool or server.search_docs
    out_path = guard_out_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = done_numbers(out_path)
    recs, skipped, stopped = [], 0, None
    for n in numbers:
        if n in already:
            skipped += 1
            continue
        stopped = ledger.check(llm.model)  # raises on an unknown model
        if stopped:
            break
        rec = draft_one(n, llm, ledger, thread_tool, search_tool)
        if rec is None:
            stopped = ledger.check(llm.model) or "cap reached"
            break
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        recs.append(rec)
    counts = {lab: sum(r["label"] == lab for r in recs) for lab in LABELS}
    return {"drafted": len(recs), "skipped_existing": skipped, "labels": counts,
            "issues_with_failed_checks": sum(bool(r["check_failures"]) for r in recs),
            "total_cost_usd": sum(r["cost_usd"] for r in recs), "stopped": stopped,
            "not_run": [n for n in numbers if n not in already and n not in {r["number"] for r in recs}]}


def print_summary(s: dict) -> None:
    print(f"drafted {s['drafted']} (skipped {s['skipped_existing']} already in the file)")
    print("labels: " + ", ".join(f"{k} {v}" for k, v in s["labels"].items()))
    print(f"issues with failed checks: {s['issues_with_failed_checks']}")
    print(f"total cost: ${s['total_cost_usd']:.4f}")
    if s["stopped"]:
        print(f"STOPPED: {s['stopped']}; not run: {s['not_run']}")


def numbers_from_args(args) -> list[int]:
    if args.numbers and args.from_file:
        raise SystemExit("give issue numbers or --from, not both")
    if args.from_file:
        rows = [json.loads(line) for line in Path(args.from_file).read_text().splitlines() if line.strip()]
        return [r["number"] for r in rows[: args.limit]]
    if not args.numbers:
        raise SystemExit("give issue numbers or --from <candidates file> --limit N")
    return args.numbers


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("numbers", nargs="*", type=int)
    ap.add_argument("--from", dest="from_file")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-usd", type=float, default=3.00, help="cap for this run")
    ap.add_argument("--max-day-usd", type=float, default=3.00, help="cap for all draft runs in one day")
    args = ap.parse_args(argv)
    numbers = numbers_from_args(args)
    try:
        llm = AnthropicLLM(DRAFT_MODEL)
        if not llm.model_available():
            print(f"the key cannot see {DRAFT_MODEL} in the models list; stopping before any paid call", file=sys.stderr)
            return 2
    except Exception as e:  # never echo more than the error type for SDK/auth failures
        print(f"cannot start: {type(e).__name__}: {e if isinstance(e, RuntimeError) else 'models list call failed'}",
              file=sys.stderr)
        return 2
    ledger = Ledger(per_run_cap=args.max_usd, per_day_cap=args.max_day_usd, path=guard_out_path(SPEND_PATH))
    print_summary(run(numbers, llm, ledger))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
