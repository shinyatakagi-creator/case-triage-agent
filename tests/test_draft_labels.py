import json

import pytest
from agent_helpers import FakeLLM, turn, use

from scripts import draft_labels as dl
from triage.llm import Ledger, cost_usd

MODEL = "claude-sonnet-5-5"
QUOTE = "Pods in the Pending state have not been scheduled onto a node yet."


def chunk(i, doc="docs/a", text=QUOTE, heading="H"):
    return {"source": doc, "heading": heading, "text": text, "url": "u", "title": "T", "chunk_id": f"c{i}", "score": 1.0}


def result(*passages, cost=0.0):
    return {"passages": list(passages), "cost_usd": cost, "latency_ms": 1.0, "cache_hit": False}


def thread(title="Pod stuck in Pending", body="coredns crashes with a loop\nsecond line"):
    return {"number": 1, "title": title, "author": "alice", "created_at": "t", "labels": [], "body": body,
            "comments": [{"author": "bob", "created_at": "t", "body": "same"}], "thread_truncated": False}


class Llm(FakeLLM):
    model = MODEL


def label_turn(label="yes", doc_ids=("docs/a",), quotes=(QUOTE,), reason="r", i=1000, o=100):
    return turn(use("submit_label", label=label, doc_ids=list(doc_ids), quotes=list(quotes), reason=reason), i=i, o=o)


@pytest.fixture
def ledger(tmp_path):
    return Ledger(per_run_cap=3.0, per_day_cap=10.0, path=tmp_path / "spend.json")


CHUNKS = [{"doc_id": "docs/a", "heading": "H", "text": "Pods in the Pending state\n  have not been scheduled onto a node yet.  More."},
          {"doc_id": "docs/b", "heading": "H", "text": "Something else entirely about services."}]


# --- the mechanical quote check ---------------------------------------------


def test_check_passes_for_verbatim_quote_ignoring_whitespace_only():
    got = dl.check_label("yes", ["docs/a"], [QUOTE], CHUNKS)  # chunk text has a newline and extra spaces
    assert got == ("yes", ["docs/a"], [QUOTE], [])


def test_check_wrong_doc_id_fails_and_becomes_unsure():
    label, _, _, fails = dl.check_label("yes", ["docs/zzz"], [QUOTE], CHUNKS)
    assert label == "unsure" and "doc_id not in this issue's results" in fails[0]


def test_check_quote_from_a_different_doc_fails():
    label, _, _, fails = dl.check_label("yes", ["docs/b"], [QUOTE], CHUNKS)
    assert label == "unsure" and "not a verbatim substring" in fails[0]


def test_check_paraphrase_fails():
    para = "Pending pods are those the scheduler has not yet placed on a node."
    label, _, _, fails = dl.check_label("yes", ["docs/a"], [para], CHUNKS)
    assert label == "unsure" and "verbatim" in fails[0]


@pytest.mark.parametrize("doc_ids,quotes", [([], []), (["docs/a"], []), (["docs/a"] * 4, [QUOTE] * 4)])
def test_check_counts_must_match_and_be_one_to_three(doc_ids, quotes):
    assert dl.check_label("yes", doc_ids, quotes, CHUNKS)[0] == "unsure"


def test_check_short_quote_and_invalid_label():
    assert dl.check_label("yes", ["docs/a"], ["Pods"], CHUNKS)[0] == "unsure"
    assert dl.check_label("maybe", [], [], CHUNKS)[0] == "unsure"


def test_check_no_and_unsure_drop_extras_without_checking():
    assert dl.check_label("no", ["docs/zzz"], ["junk"], CHUNKS) == ("no", [], [], [])


# --- the draft flow -----------------------------------------------------------


def test_queries_and_dedupe_by_chunk_id(ledger, tmp_path):
    calls = []

    def search(q, k):
        calls.append((q, k))
        return result(chunk(1), chunk(2, doc="docs/b"), cost=0.001) if len(calls) == 1 else result(chunk(1), chunk(3, doc="docs/c"))

    llm = Llm([label_turn()])
    s = dl.run([1], llm, ledger, tmp_path / "labels.jsonl", thread_tool=lambda url: thread(), search_tool=search)
    assert [k for _, k in calls] == [5, 8, 8] and calls[0][0] == "Pod stuck in Pending"
    assert calls[1][0].startswith("coredns") and calls[2][0].startswith("coredns crashes with a loop second line")
    prompt = llm.seen[0][0]["content"]
    assert prompt.count("doc_id=docs/a") == 1 and "doc_id=docs/b" in prompt and "doc_id=docs/c" in prompt  # c1 once
    rec = json.loads((tmp_path / "labels.jsonl").read_text())
    assert rec["label"] == "yes" and rec["doc_ids"] == ["docs/a"] and rec["check_failures"] == []
    assert rec["result_doc_ids"] == ["docs/a", "docs/b", "docs/c"]  # what a typed doc_id is checked against
    assert rec["model"] == MODEL and rec["cost_usd"] > 0.001 and rec["title"] == "Pod stuck in Pending"
    assert s["labels"] == {"yes": 1, "no": 0, "unsure": 0}


def test_failed_quote_check_records_unsure_and_the_models_own_label(ledger, tmp_path):
    llm = Llm([label_turn(quotes=["A paraphrase that is not in the chunk at all."])])
    dl.run([1], llm, ledger, tmp_path / "l.jsonl", thread_tool=lambda u: thread(),
           search_tool=lambda q, k: result(chunk(1)))
    rec = json.loads((tmp_path / "l.jsonl").read_text())
    assert rec["label"] == "unsure" and rec["model_label"] == "yes" and rec["check_failures"]


def test_thread_error_and_empty_search_make_no_model_call(ledger, tmp_path):
    llm = Llm([])
    s = dl.run([1, 2], llm, ledger, tmp_path / "l.jsonl",
               thread_tool=lambda u: {"error": "not_found"} if u.endswith("/1") else thread(),
               search_tool=lambda q, k: {"error": "search_failed"})
    recs = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert [r["label"] for r in recs] == ["unsure", "unsure"] and llm.seen == [] and s["total_cost_usd"] == 0.0


def test_model_without_submit_label_is_unsure(ledger, tmp_path):
    llm = Llm([turn({"type": "text", "text": "yes it is"})])
    dl.run([1], llm, ledger, tmp_path / "l.jsonl", thread_tool=lambda u: thread(), search_tool=lambda q, k: result(chunk(1)))
    assert "did not call submit_label" in json.loads((tmp_path / "l.jsonl").read_text())["check_failures"][0]


def test_rerun_skips_issues_already_drafted(ledger, tmp_path):
    out = tmp_path / "l.jsonl"
    kw = dict(thread_tool=lambda u: thread(), search_tool=lambda q, k: result(chunk(1)))
    dl.run([1], Llm([label_turn()]), ledger, out, **kw)
    s = dl.run([1], Llm([]), ledger, out, **kw)
    assert s["drafted"] == 0 and s["skipped_existing"] == 1 and len(out.read_text().splitlines()) == 1


# --- the cost cap -------------------------------------------------------------


def test_cost_cap_stops_the_run_cleanly(tmp_path):
    ledger = Ledger(per_run_cap=0.02, per_day_cap=10.0, path=tmp_path / "s.json")
    llm = Llm([label_turn(i=10_000, o=1000)] * 3)  # 10k*2 + 1k*10 = $0.03 per call, over the $0.02 cap
    s = dl.run([1, 2, 3], llm, ledger, tmp_path / "l.jsonl", thread_tool=lambda u: thread(),
               search_tool=lambda q, k: result(chunk(1)))
    assert s["drafted"] == 1 and len(llm.seen) == 1 and "per-run cap" in s["stopped"] and s["not_run"] == [2, 3]
    assert s["total_cost_usd"] == pytest.approx(0.03)
    assert len((tmp_path / "l.jsonl").read_text().splitlines()) == 1


def test_sonnet_prices_are_the_owners():
    assert cost_usd(MODEL, 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert cost_usd(MODEL, 0, 0, cache_write_5m=1_000_000) == pytest.approx(2.50)
    assert cost_usd(MODEL, 0, 0, cache_write_1h=1_000_000) == pytest.approx(4.00)
    assert cost_usd(MODEL, 0, 0, cache_read=1_000_000) == pytest.approx(0.20)


# --- guards and start-up ------------------------------------------------------


def test_nothing_is_written_under_data_eval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("data/eval/labels.jsonl", "data/eval"):
        with pytest.raises(ValueError, match="data/eval"):
            dl.guard_out_path(bad)
    assert dl.guard_out_path("data/drafts/labels.jsonl")


def test_main_stops_before_any_paid_call_when_key_cannot_see_the_model(monkeypatch, capsys):
    class Adapter:
        def __init__(self, model): self.model = model
        def model_available(self): return False

    monkeypatch.setattr(dl, "AnthropicLLM", Adapter)
    monkeypatch.setattr(dl, "run", lambda *a, **k: pytest.fail("must not run"))
    assert dl.main(["1"]) == 2 and "cannot see claude-sonnet-5-5" in capsys.readouterr().err


def test_main_missing_key_names_the_variable_only(monkeypatch, capsys):
    monkeypatch.delenv("TRIAGE_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert dl.main(["1"]) == 2 and "TRIAGE_ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_model_available_reads_the_models_list(monkeypatch):
    from types import SimpleNamespace as NS
    from triage.llm import AnthropicLLM
    monkeypatch.setenv("TRIAGE_ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    llm = AnthropicLLM(MODEL)
    llm._client.models.list = lambda limit: [NS(id="claude-haiku-5-5"), NS(id=MODEL)]
    assert llm.model_available()
    llm._client.models.list = lambda limit: [NS(id="claude-haiku-5-5")]
    assert not llm.model_available()


def test_numbers_from_candidates_file(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text("\n".join(json.dumps({"number": n}) for n in (5, 6, 7)))
    ns = type("A", (), {"numbers": [], "from_file": str(f), "limit": 2})()
    assert dl.numbers_from_args(ns) == [5, 6]


def test_day_cap_default_is_three_dollars_and_run_cap_default_unchanged(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = {}

    class Adapter:
        model = MODEL
        def __init__(self, model): pass
        def model_available(self): return True

    class SpyLedger:
        def __init__(self, per_run_cap, per_day_cap, path): seen.update(run=per_run_cap, day=per_day_cap)

    monkeypatch.setattr(dl, "AnthropicLLM", Adapter)
    monkeypatch.setattr(dl, "Ledger", SpyLedger)
    monkeypatch.setattr(dl, "run", lambda *a, **k: {"drafted": 0, "skipped_existing": 0, "labels": {}, "stopped": None,
                                                    "issues_with_failed_checks": 0, "total_cost_usd": 0.0, "not_run": []})
    assert dl.main(["1"]) == 0 and seen == {"run": 3.0, "day": 3.0}
