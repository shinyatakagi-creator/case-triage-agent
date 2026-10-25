"""Print an empty tally line per issue number to stdout: python -m scripts.tally_template 141240 135321

The question mark, doc path and evidence are placeholders for the owner to fill in. Writes
nothing to disk (in particular nothing under data/eval or data/dev).
"""
import sys


def lines(numbers: list[str]) -> list[str]:
    return [f"{n} ?  <doc path>  (<evidence>)" for n in numbers]


def main(argv: list[str]) -> int:
    if not argv or not all(a.isdigit() for a in argv):
        print("usage: python -m scripts.tally_template <issue number> ...", file=sys.stderr)
        return 2
    print("\n".join(lines(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
