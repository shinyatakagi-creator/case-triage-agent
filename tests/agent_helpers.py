"""Shared fakes for the agent/baseline/ledger tests. Imported as `agent_helpers` (tests/ has no
__init__.py and no conftest.py, so pytest puts this directory on sys.path)."""
import copy
import json
from pathlib import Path

from triage.llm import DEFAULT_MODEL

FIX = Path(__file__).parent / "fixtures" / "agent"
URL = "https://github.com/kubernetes/kubernetes/issues/1"
SRC = "docs/invented/widgets.md"


def fx(name):
    return json.loads((FIX / name).read_text())


def turn(*blocks, i=100, o=50):
    return {"content": list(blocks), "input_tokens": i, "output_tokens": o}


def use(name, id="t", **inp):
    return {"type": "tool_use", "id": id, "name": name, "input": inp}


def final(outcome="answered", citations=(SRC,), **kw):
    return turn(use("final_answer", outcome=outcome, citations=list(citations),
                    likely_cause="cause", next_step="step", **kw))


class FakeLLM:
    model = DEFAULT_MODEL

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def complete(self, system, messages, tools):
        try:
            self.seen.append(copy.deepcopy(messages))
        except RecursionError:  # tests that feed deeply nested model output
            self.seen.append(None)
        return self.turns.pop(0)


class FakeTools:
    """responses: tool -> list of results returned in order (the last one repeats)."""

    def __init__(self, **responses):
        self.responses = {"search_docs": [fx("search.json")], "get_issue_thread": [fx("thread.json")],
                          "extract_error_signals": [fx("signals.json")], **responses}
        self.calls = []

    def _next(self, name, args):
        self.calls.append((name, args))
        q = self.responses[name]
        r = q.pop(0) if len(q) > 1 else q[0]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, dict) and "error" in r else copy.deepcopy(r)

    def search_docs(self, query, k=5):
        return self._next("search_docs", query)

    def get_issue_thread(self, issue_url):
        return self._next("get_issue_thread", issue_url)

    def extract_error_signals(self, text):
        return self._next("extract_error_signals", text)
