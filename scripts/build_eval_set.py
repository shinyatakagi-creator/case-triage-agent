"""Build staging eval files from the owner's reviewed decisions. No model calls.

Usage: python -m scripts.build_eval_set --answerable 1 2 ... --unanswerable 3 4 ... [--dev 5 6 ...]
Writes EvalCase rows (--categories: JSON of issue number -> category, for answerable cases) to data/eval_staging/eval.jsonl (and dev.jsonl for --dev). Nothing is written if any
number is refused. Promoting staging into data/eval/ is the owner's own step; this refuses to write there.
"""
import argparse
import json
import sys
from pathlib import Path

from scripts.draft_labels import guard_out_path
from scripts.review_labels import REVIEWED_PATH, read_jsonl
from triage.contracts import CATEGORIES, EvalCase

STAGING = Path("data/eval_staging")
NO_CATEGORY = "n/a"  # unanswerable cases; category accuracy is scored on answerable cases only


def case_for(number: int, reviewed: dict, want: str | None, categories: dict) -> EvalCase:
    """One EvalCase; ValueError if the number is unreviewed or its decision is not yes/no or not `want`."""
    r = reviewed.get(number)
    if r is None:
        raise ValueError(f"#{number}: not in reviewed.jsonl")
    if r["decision"] not in ("yes", "no"):
        raise ValueError(f"#{number}: reviewed decision is {r['decision']!r}, only yes/no can be used")
    if want and r["decision"] != want:
        raise ValueError(f"#{number}: reviewed decision is {r['decision']!r} but it was listed as {want!r}")
    yes = r["decision"] == "yes"
    category = categories.get(number) if yes else NO_CATEGORY
    if category not in (CATEGORIES if yes else (NO_CATEGORY,)):
        raise ValueError(f"#{number}: category {category!r} is missing or not one of {list(CATEGORIES)}")
    return {"id": f"k8s-{number}", "issue_url": r["draft"]["url"], "answerable": yes, "category": category,
            "gold_docs": list(r["decision_doc_ids"]) if yes else []}


def build(answerable, unanswerable, dev, reviewed: dict, categories: dict) -> tuple[list[EvalCase], list[EvalCase]]:
    """(eval cases, dev cases). Raises ValueError listing every problem."""
    lists = [*answerable, *unanswerable, *dev]
    problems = [f"#{n}: listed more than once (dev must not overlap the eval set)" for n in sorted(set(lists)) if lists.count(n) > 1]
    out = {"yes": [], "no": [], "dev": []}
    for key, nums, want in (("yes", answerable, "yes"), ("no", unanswerable, "no"), ("dev", dev, None)):
        for n in nums:
            try:
                out[key].append(case_for(n, reviewed, want, categories))
            except ValueError as e:
                problems.append(str(e))
    if problems:
        raise ValueError("\n".join(problems))
    return out["yes"] + out["no"], out["dev"]


def write_jsonl(path: Path, cases: list[EvalCase]) -> None:
    path = guard_out_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerable", nargs="*", type=int, default=[])
    ap.add_argument("--unanswerable", nargs="*", type=int, default=[])
    ap.add_argument("--dev", nargs="*", type=int, default=[])
    ap.add_argument("--categories", required=True, help="JSON file: issue number -> category")
    ap.add_argument("--out-dir", default=str(STAGING))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    reviewed = {r["draft"]["number"]: r for r in read_jsonl(REVIEWED_PATH)}
    categories = {int(k): v for k, v in json.loads(Path(args.categories).read_text()).items()}
    try:
        guard_out_path(out_dir / "eval.jsonl")
        eval_cases, dev_cases = build(args.answerable, args.unanswerable, args.dev, reviewed, categories)
    except ValueError as e:
        print(f"refused, nothing written:\n{e}", file=sys.stderr)
        return 2
    if eval_cases:
        write_jsonl(out_dir / "eval.jsonl", eval_cases)
    if dev_cases:
        write_jsonl(out_dir / "dev.jsonl", dev_cases)
    print(f"eval: {len(eval_cases)} cases ({len(args.answerable)} answerable, {len(args.unanswerable)} not); dev: {len(dev_cases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
