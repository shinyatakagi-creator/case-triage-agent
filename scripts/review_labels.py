"""Review drafted labels one at a time. No model calls. Resumable.

Usage: python -m scripts.review_labels

Reads data/drafts/labels.jsonl, appends each decision to data/drafts/reviewed.jsonl as it is made, and
skips issues already reviewed, so quitting (q) and coming back loses nothing. Keys:
  y accept the draft label   n flip yes<->no   u mark unsure   o open the issue in the browser   q quit
This writes only under data/drafts/. Promoting reviewed labels into data/eval/ is the owner's own step.
"""
import json
import os
import sys
import webbrowser
from pathlib import Path

from scripts.draft_labels import DRAFT_DIR, LABELS_PATH, guard_out_path

REVIEWED_PATH = DRAFT_DIR / "reviewed.jsonl"
FLIP = {"yes": "no", "no": "yes"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def show(d: dict, position: str) -> None:
    print("\n" + "=" * 78)
    print(f"[{position}] #{d['number']}  {d['title']}")
    print(d["url"])
    print("-" * 78)
    print("\n".join(d["body_head"]) or "(empty body)")
    print("-" * 78)
    print(f"DRAFT: {d['label'].upper()}   (model said: {d.get('model_label')})")
    for doc, quote in zip(d["doc_ids"], d["quotes"]):
        print(f"  doc: {doc}\n  quote: {quote}")
    print(f"reason: {d['reason']}")
    if d["check_failures"]:
        print("check failures: " + "; ".join(d["check_failures"]))
    print("y accept | n flip yes/no | u unsure | o open | q quit > ", end="", flush=True)


def getkey() -> str:
    """One keypress; '' at end of input."""
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        return sys.stdin.read(1)
    import termios
    import tty
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def decide(d: dict, key: str) -> tuple[str, str] | None:
    """(final label, action) for a decision key, or None if the key is not a decision."""
    if key == "y":
        return d["label"], "accepted"
    if key == "n" and d["label"] in FLIP:
        return FLIP[d["label"]], "flipped"
    if key == "u":
        return "unsure", "unsure"
    return None


def review(drafts, reviewed_path=REVIEWED_PATH, getkey=getkey, show=show, opener=webbrowser.open) -> dict:
    reviewed_path = guard_out_path(reviewed_path)
    reviewed_path.parent.mkdir(parents=True, exist_ok=True)
    seen = {r["draft"]["number"] for r in read_jsonl(reviewed_path)}
    todo = [d for d in drafts if d["number"] not in seen]
    session = {"accepted": 0, "flipped": 0, "unsure": 0}
    for i, d in enumerate(todo, 1):
        show(d, f"{i}/{len(todo)}")
        while True:
            key = getkey().lower()
            if key in ("q", "", "\x03"):  # q, end of input, ctrl-c
                print()
                return summary(session, reviewed_path, quit_early=True)
            if key == "o":
                opener(d["url"])
                continue
            got = decide(d, key)
            if got is None:
                if key == "n":
                    print("\n(n flips yes/no only; this draft is unsure: use y or u)", end="", flush=True)
                continue
            label, action = got
            with open(reviewed_path, "a") as f:
                f.write(json.dumps({"draft": d, "decision": label, "action": action,
                                    "changed": label != d["label"]}) + "\n")
            session[action] += 1
            print(f"\n-> {label}")
            break
    return summary(session, reviewed_path, quit_early=False)


def summary(session: dict, reviewed_path: Path, quit_early: bool) -> dict:
    total = read_jsonl(reviewed_path)
    out = {**session, "reviewed_total": len(total), "quit_early": quit_early}
    print(f"this session: accepted {session['accepted']}, flipped {session['flipped']}, "
          f"marked unsure {session['unsure']}; reviewed in total {len(total)}")
    return out


def main() -> int:
    drafts = read_jsonl(LABELS_PATH)
    if not drafts:
        print(f"no drafts in {LABELS_PATH}", file=sys.stderr)
        return 2
    review(drafts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
