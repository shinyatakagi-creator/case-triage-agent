"""Offline: golden-file argument handling in scripts/parity.py (no network, no index load)."""
import pytest

from scripts import parity


def test_no_golden_is_a_usage_error_naming_both_ways(monkeypatch, capsys):
    monkeypatch.delenv("RAG_GOLDEN", raising=False)
    with pytest.raises(SystemExit) as e:
        parity.main([])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--golden" in err and "RAG_GOLDEN" in err


def test_env_var_is_used_and_unreadable_path_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.setenv("RAG_GOLDEN", "/nonexistent/golden.jsonl")
    with pytest.raises(SystemExit) as e:
        parity.main([])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "/nonexistent/golden.jsonl" in err  # the env path was picked up, not "no golden file"


def test_flag_wins_over_env(monkeypatch, capsys):
    monkeypatch.setenv("RAG_GOLDEN", "/from/env.jsonl")
    with pytest.raises(SystemExit):
        parity.main(["--golden", "/from/flag.jsonl"])
    err = capsys.readouterr().err
    assert "/from/flag.jsonl" in err and "/from/env.jsonl" not in err
