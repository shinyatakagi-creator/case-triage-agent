"""Print candidate issue numbers not yet in data/drafts/labels.jsonl, space-separated, newest first. No model calls.

Usage: python -m scripts.unseen_candidates --from data/candidates_support_250.jsonl --limit 20
"""
import argparse
import sys
from pathlib import Path

from scripts.draft_labels import LABELS_PATH, done_numbers
from scripts.review_labels import read_jsonl


def unseen(rows: list[dict], seen: set[int], limit: int | None) -> list[int]:
    fresh = [r for r in rows if r["number"] not in seen]
    fresh.sort(key=lambda r: r["created_at"], reverse=True)  # ISO timestamps sort as text
    return [r["number"] for r in fresh[:limit]]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_file", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)
    print(" ".join(map(str, unseen(read_jsonl(Path(args.from_file)), done_numbers(LABELS_PATH), args.limit))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
