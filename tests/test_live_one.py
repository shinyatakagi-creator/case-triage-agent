import pytest
from agent_helpers import URL, FakeLLM, FakeTools, final, turn, use

from scripts import live_one
from triage import baseline
from triage.agent import CATEGORY_BLOCK, SYSTEM_PROMPT, tool_specs
from triage.llm import AnthropicLLM, Ledger


def test_report_prints_queries_errors_and_note(tmp_path, capsys):
    tools = live_one.LoggingTools(FakeTools(extract_error_signals=[{"error": "search_failed"}]))
    llm = FakeLLM([turn(use("search_docs", id="a", query="widgets"), use("extract_error_signals", id="b", text="x")),
                   final(category="howto")])
    res = live_one.run(URL, llm, Ledger(per_run_cap=1.0, per_day_cap=10.0, path=tmp_path / "s.json"), tools)
    out = capsys.readouterr().out
    assert res["note"]["outcome"] == "answered" and res["note"]["category"] == "howto"
    assert "search_docs: ok  query='widgets' k=5" in out and "extract_error_signals: error: search_failed" in out
    assert "steps: 2" in out and "cost: $" in out


def test_category_block_identical_in_both_prompts():
    assert CATEGORY_BLOCK in SYSTEM_PROMPT and CATEGORY_BLOCK in baseline.SYSTEM_PROMPT
    assert CATEGORY_BLOCK.splitlines()[0].startswith("Categories (check in this order")
    assert "For answered: category is required" in SYSTEM_PROMPT and "omit category" in baseline.SYSTEM_PROMPT


def test_tool_descriptions():
    d = {t["name"]: t["description"] for t in tool_specs(["a"])}
    assert d["search_docs"].startswith("Search the Kubernetes documentation (concepts, tasks, reference).")
    assert "replies from others are not included" in d["get_issue_thread"]


def test_adapter_keywords_exist_in_the_installed_sdk(monkeypatch):
    """A fake client hid an unsupported keyword once; check against the real signature, offline."""
    import inspect

    import anthropic
    monkeypatch.setenv("TRIAGE_ANTHROPIC_API_KEY", "test-key")
    llm = AnthropicLLM("claude-haiku-5-5")
    seen = {}

    class Msgs:
        def create(self, **kw):
            seen.update(kw)
            raise RuntimeError("stop")  # no network: the fake ends the call here

    llm._client = type("C", (), {"messages": Msgs()})()
    with pytest.raises(RuntimeError):
        llm.complete("s", [], [])
    params = inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters
    assert seen and set(seen) <= set(params), set(seen) - set(params)


def test_citation_text_says_source_not_chunk_id():
    from triage.agent import ANSWER_RULES, final_answer_spec
    assert "not chunk_id" in ANSWER_RULES
    assert "not chunk_id" in final_answer_spec(["a"])["input_schema"]["properties"]["citations"]["description"]
    assert ANSWER_RULES in SYSTEM_PROMPT and ANSWER_RULES in baseline.SYSTEM_PROMPT


def test_baseline_system_runs_one_search_one_model_call(tmp_path, capsys):
    tools = live_one.LoggingTools(FakeTools())
    llm = FakeLLM([final(category="howto")])
    res = live_one.run(URL, llm, Ledger(per_run_cap=1.0, per_day_cap=10.0, path=tmp_path / "s.json"), tools, "baseline")
    assert res["note"]["steps"] == 1 and [c["tool"] for c in tools.log] == ["get_issue_thread", "search_docs"]
    assert "search_docs: ok  query=" in capsys.readouterr().out
