"""Compare the owner's reviewed decisions with a new draft file. No model calls, read-only.

Usage: python -m scripts.compare_labels --new data/drafts/labels_v2.jsonl
Only issues in both files are compared; prints the disagreements and the agree / disagree counts.
"""
import argparse
import sys
from pathlib import Path

from scripts.review_labels import REVIEWED_PATH, read_jsonl


def compare(reviewed: list[dict], new: list[dict]) -> tuple[list[tuple[dict, dict]], int]:
    """(disagreements as (reviewed record, new draft), agree count) over issues in both."""
    drafts = {d["number"]: d for d in new}
    agree, diff = 0, []
    for r in reviewed:
        d = drafts.get(r["draft"]["number"])
        if d is None:
            continue
        if r["decision"] == d["label"]:
            agree += 1
        else:
            diff.append((r, d))
    return diff, agree


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    args = ap.parse_args(argv)
    if not Path(args.new).exists():
        print(f"no such file: {args.new}", file=sys.stderr)
        return 2
    diff, agree = compare(read_jsonl(REVIEWED_PATH), read_jsonl(Path(args.new)))
    for r, d in diff:
        print(f"\n#{d['number']}  {d['title']}")
        print(f"  my decision: {r['decision']}   new label: {d['label']}")
        print(f"  new reason: {d['reason']}")
        for q in d["quotes"]:
            print(f"  new quote: {q}")
    print(f"\nagree {agree}, disagree {len(diff)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
