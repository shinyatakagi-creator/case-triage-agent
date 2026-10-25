"""Ledger tests: caps, fail-closed state, concurrent writers. No model calls."""
import json
import math

import pytest

from triage.llm import DEFAULT_MODEL, Ledger, cost_usd


def day_total(p):
    return json.loads(p.read_text())["usd"]


def test_two_ledgers_on_one_file_sum(tmp_path):
    p = tmp_path / "s.json"
    a, b = Ledger(per_run_cap=9, per_day_cap=9, path=p), Ledger(per_run_cap=9, per_day_cap=9, path=p)
    a.charge_usd(1.0)
    b.charge_usd(1.0)
    assert day_total(p) == 2.0 and b.day_usd == 2.0
    assert a.check(DEFAULT_MODEL) is None and a.day_usd == 2.0  # check() re-reads


@pytest.mark.parametrize("content", ['{"date": "D", "usd": NaN}', '{"date": "D", "usd": -1}',
                                     '{"date": "D", "usd": "5"}', '{"date": "D", "usd": Infinity}',
                                     '{"date": "D", "usd": 1', '', '[]', '{"usd": 1}',
                                     '{"date": "D", "usd": 1' + '0' * 500 + '}', '[' * 100000,
                                     '{"date": "D", "usd": ' + '[' * 100000])
def test_bad_spend_file_fails_closed(tmp_path, content):
    p = tmp_path / "s.json"
    p.write_text(content.replace("D", "2026-10-25"))
    led = Ledger(path=p, day="2026-10-25")
    assert led.load_error and led.check(DEFAULT_MODEL) == led.load_error


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5, "1", None, True, 10**400, -(10**400)])
def test_charge_usd_rejects_bad_amounts(tmp_path, bad):
    led = Ledger(path=tmp_path / "s.json")
    assert led.charge_usd(bad) == 0.0 and led.load_error and led.run_usd == 0.0
    assert not (tmp_path / "s.json").exists()


@pytest.mark.parametrize("i,o", [(float("nan"), 1), (-1, 1), ("1", 1), (1, None), (10**400, 1), (1, 10**400)])
def test_charge_rejects_bad_token_counts(tmp_path, i, o):
    led = Ledger(path=tmp_path / "s.json")
    assert led.charge(DEFAULT_MODEL, i, o) == 0.0 and led.load_error
    assert led.check(DEFAULT_MODEL) == led.load_error


def test_unknown_model_raises(tmp_path):
    led = Ledger(path=tmp_path / "s.json")
    with pytest.raises(ValueError, match="no price configured"):
        led.check("mystery-model")
    with pytest.raises(ValueError):
        led.charge("mystery-model", 1, 1)
    with pytest.raises(ValueError):
        cost_usd("mystery-model", 1, 1)


def test_new_day_resets_total(tmp_path):
    p = tmp_path / "s.json"
    Ledger(path=p, day="2026-10-24").charge_usd(5.0)
    led = Ledger(path=p, day="2026-10-25")
    assert led.day_usd == 0.0 and led.load_error is None


def test_cost_arithmetic():
    assert math.isclose(cost_usd(DEFAULT_MODEL, 1_000_000, 1_000_000), 0.50 + 2.50)  # prompt > 100k: long tier


def test_non_finite_day_total_is_never_written(tmp_path):
    p = tmp_path / "s.json"
    led = Ledger(per_run_cap=9e308, per_day_cap=9e308, path=p)
    led.charge_usd(1e308)
    assert led.load_error is None
    led.charge_usd(1e308)  # total would be inf
    assert led.load_error and "not finite" in led.load_error
    assert day_total(p) == 1e308 and "Infinity" not in p.read_text()  # file keeps the last finite total
    assert led.check(DEFAULT_MODEL) == led.load_error


H5, H45 = "claude-haiku-5-5", "claude-haiku-4-5-20251001"


def test_cache_cost_short_tier_haiku_5_5():
    # prompt 90k <= 100k: input 0.10, write 5m 0.125, write 1h 0.20, read 0.01
    got = cost_usd(H5, 10_000, 0, cache_write_5m=50_000, cache_write_1h=20_000, cache_read=10_000)
    assert math.isclose(got, (10_000 * 0.10 + 50_000 * 0.125 + 20_000 * 0.20 + 10_000 * 0.01) / 1e6)


def test_cache_cost_long_tier_boundary_haiku_5_5():
    assert math.isclose(cost_usd(H5, 0, 0, cache_read=100_000), 100_000 * 0.01 / 1e6)  # exactly 100k: short
    assert math.isclose(cost_usd(H5, 0, 0, cache_read=100_001), 100_001 * 0.05 / 1e6)  # above: long
    assert math.isclose(cost_usd(H5, 0, 0, cache_write_5m=200_000), 200_000 * 0.625 / 1e6)
    assert math.isclose(cost_usd(H5, 0, 0, cache_write_1h=200_000), 200_000 * 1.00 / 1e6)


def test_cache_cost_haiku_4_5_has_no_tiers():
    assert math.isclose(cost_usd(H45, 0, 0, cache_write_5m=1_000_000), 1.25)
    assert math.isclose(cost_usd(H45, 0, 0, cache_write_1h=1_000_000), 2.00)
    assert math.isclose(cost_usd(H45, 0, 0, cache_read=1_000_000), 0.10)  # >100k prompt, same rate


@pytest.mark.parametrize("kw", [{"cache_read": -1}, {"cache_write_5m": float("nan")}, {"cache_write_1h": "1"},
                                {"cache_read": 10**400}])
def test_cache_cost_rejects_invalid_counts(kw):
    with pytest.raises(ValueError):
        cost_usd(H5, 1, 1, **kw)


def _fake_response(**usage):
    from types import SimpleNamespace as NS
    return NS(content=[NS(type="text", text="hi")], usage=NS(input_tokens=5, output_tokens=7, **usage))


def _adapter(monkeypatch, response):
    from triage.llm import AnthropicLLM
    monkeypatch.setenv("TRIAGE_ANTHROPIC_API_KEY", "dummy")  # constructing the client makes no request
    llm = AnthropicLLM()
    llm._client.messages.create = lambda **kw: response  # no network: the SDK call is replaced
    return llm.complete("s", [], [])


def test_adapter_maps_split_cache_usage(monkeypatch):
    from types import SimpleNamespace as NS
    u = dict(cache_creation=NS(ephemeral_5m_input_tokens=11, ephemeral_1h_input_tokens=None),
             cache_creation_input_tokens=11, cache_read_input_tokens=3)
    t = _adapter(monkeypatch, _fake_response(**u))
    assert (t["cache_write_5m_tokens"], t["cache_write_1h_tokens"], t["cache_read_tokens"]) == (11, 0, 3)


def test_adapter_unsplit_cache_write_priced_at_1h_and_none_is_zero(monkeypatch):
    t = _adapter(monkeypatch, _fake_response(cache_creation=None, cache_creation_input_tokens=9,
                                            cache_read_input_tokens=None))
    assert (t["cache_write_5m_tokens"], t["cache_write_1h_tokens"], t["cache_read_tokens"]) == (0, 9, 0)


def test_input_and_output_are_tiered_at_100k_prompt():
    # exactly 100,000 prompt tokens: short tier (input 0.10, output 0.50)
    assert math.isclose(cost_usd(H5, 100_000, 1_000_000), (100_000 * 0.10 + 1_000_000 * 0.50) / 1e6)
    # 100,001: long tier for BOTH input and output (0.50 / 2.50)
    assert math.isclose(cost_usd(H5, 100_001, 1_000_000), (100_001 * 0.50 + 1_000_000 * 2.50) / 1e6)


def test_cache_tokens_alone_can_push_the_prompt_over_the_line():
    # input 40k + cache read 60k = 100,000: still short; one more cache token flips everything
    assert math.isclose(cost_usd(H5, 40_000, 1000, cache_read=60_000),
                        (40_000 * 0.10 + 1000 * 0.50 + 60_000 * 0.01) / 1e6)
    assert math.isclose(cost_usd(H5, 40_000, 1000, cache_read=60_001),
                        (40_000 * 0.50 + 1000 * 2.50 + 60_001 * 0.05) / 1e6)
    # cache writes count toward the prompt too
    assert math.isclose(cost_usd(H5, 1, 1, cache_write_5m=50_000, cache_write_1h=50_000),
                        (1 * 0.50 + 1 * 2.50 + 50_000 * 0.625 + 50_000 * 1.00) / 1e6)
    assert math.isclose(cost_usd(H5, 1, 1, cache_write_5m=50_000, cache_write_1h=49_999),  # 1 + 99,999 = 100,000
                        (1 * 0.10 + 1 * 0.50 + 50_000 * 0.125 + 49_999 * 0.20) / 1e6)


def test_haiku_4_5_input_output_are_flat():
    assert math.isclose(cost_usd(H45, 100_001, 1000), (100_001 * 1.0 + 1000 * 5.0) / 1e6)
    assert math.isclose(cost_usd(H45, 10, 1000), (10 * 1.0 + 1000 * 5.0) / 1e6)


def test_defaults_are_owner_set():
    from triage.llm import DEFAULT_DAY_CAP_USD, DEFAULT_RUN_CAP_USD
    assert (DEFAULT_MODEL, DEFAULT_RUN_CAP_USD, DEFAULT_DAY_CAP_USD) == ("claude-haiku-5-5", 0.02, 1.00)


def test_adapter_uses_only_the_triage_key_and_passes_it_explicitly(monkeypatch):
    from triage.llm import AnthropicLLM
    monkeypatch.setenv("TRIAGE_ANTHROPIC_API_KEY", "triage-key-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-key-value")
    assert AnthropicLLM()._client.api_key == "triage-key-value"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_adapter_missing_or_empty_key_raises_naming_the_variable_only(monkeypatch, value):
    from triage.llm import AnthropicLLM
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-key-value")  # must NOT be used as a fallback
    if value is None:
        monkeypatch.delenv("TRIAGE_ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TRIAGE_ANTHROPIC_API_KEY", value)
    with pytest.raises(RuntimeError) as e:
        AnthropicLLM()
    msg = str(e.value)
    assert "TRIAGE_ANTHROPIC_API_KEY" in msg and "other-key-value" not in msg


def test_adapter_ignores_anthropic_base_url(monkeypatch):
    from triage.llm import API_BASE_URL, AnthropicLLM
    monkeypatch.setenv("TRIAGE_ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example")
    assert str(AnthropicLLM()._client.base_url).rstrip("/") == API_BASE_URL == "https://api.anthropic.com"


@pytest.mark.parametrize("value", ["secret-token-value", ""])
def test_adapter_refuses_when_auth_token_is_set_without_printing_it(monkeypatch, value):
    from triage.llm import AnthropicLLM
    monkeypatch.setenv("TRIAGE_ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", value)
    with pytest.raises(RuntimeError) as e:
        AnthropicLLM()
    assert "ANTHROPIC_AUTH_TOKEN" in str(e.value) and "secret-token-value" not in str(e.value)
