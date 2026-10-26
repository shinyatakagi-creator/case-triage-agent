"""Model access: LLM protocol, the (UNRUN) Anthropic adapter, price table and spend caps."""
import fcntl
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import NamedTuple, NotRequired, Protocol, TypedDict

# Owner-set default (25 Oct 2026).
DEFAULT_MODEL = "claude-haiku-5-5"  # config value, override with env TRIAGE_MODEL


API_KEY_ENV = "TRIAGE_ANTHROPIC_API_KEY"  # the only key variable read; ANTHROPIC_API_KEY is deliberately ignored


API_BASE_URL = "https://api.anthropic.com"  # fixed, so ANTHROPIC_BASE_URL cannot redirect the key


def api_key() -> str:
    """The key from TRIAGE_ANTHROPIC_API_KEY. Raises without echoing any part of a value."""
    if "ANTHROPIC_AUTH_TOKEN" in os.environ:
        # The SDK would send this as a second credential; refuse rather than guess which one wins.
        raise RuntimeError("ANTHROPIC_AUTH_TOKEN is set in the environment; unset it before using this adapter")
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set or is empty; set it in the environment (not in a committed file)")
    return key


def model_name() -> str:
    return os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)


class ModelPrice(NamedTuple):
    """USD per million tokens. All five long_* rates apply when the prompt (input + cache read +
    cache write tokens) exceeds LONG_PROMPT_TOKENS; for models without tiers they equal the short rates."""
    input: float
    output: float
    write_5m: float
    write_1h: float
    read: float
    long_input: float
    long_output: float
    long_write_5m: float
    long_write_1h: float
    long_read: float


LONG_PROMPT_TOKENS = 100_000

# Owner-confirmed 25 Oct 2026, source: platform.claude.com/docs/en/about-claude/pricing
# Tier rule (owner-confirmed): prompt size = input + cache read + cache write tokens; above 100,000
# all five rates switch to the long_* values, output included.
PRICES_USD_PER_MTOK: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.125, 0.20, 0.01, 0.50, 2.50, 0.625, 1.00, 0.05),
    # no price tiers: the long_* rates equal the short ones
    "claude-sonnet-5-5": ModelPrice(2.0, 10.0, 2.50, 4.00, 0.20, 2.0, 10.0, 2.50, 4.00, 0.20),
    "claude-haiku-4-5-20251001": ModelPrice(1.0, 5.0, 1.25, 2.00, 0.10, 1.0, 5.0, 1.25, 2.00, 0.10),
}

DEFAULT_RUN_CAP_USD = 0.02  # owner-set 25 Oct 2026
DEFAULT_DAY_CAP_USD = 1.00
SPEND_PATH = Path("data/spend.json")  # data/ is gitignored


class Turn(TypedDict):
    # blocks: {"type": "text", "text"} or {"type": "tool_use", "id", "name", "input"}
    content: list[dict]
    input_tokens: int
    output_tokens: int
    # Cache usage, all optional: absent means no caching (0). A present but invalid value is an error.
    cache_write_5m_tokens: NotRequired[int]
    cache_write_1h_tokens: NotRequired[int]
    cache_read_tokens: NotRequired[int]


class LLM(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Turn: ...


class AnthropicLLM:
    """UNRUN: never exercised against the real API (the `anthropic` package is installed and pinned,
    but no request has been made; complete() has only been read, not executed).

    Token usage: input_tokens, output_tokens and the cache fields are read. Cache writes are split
    5-minute / 1-hour from usage.cache_creation when the SDK provides it; if only the combined
    cache_creation_input_tokens is present it is priced at the dearer 1-hour rate. A None cache
    field is read as 0.

    The SDK retries failed requests internally (max_retries, default 2), which would spend
    money outside the Ledger's view. max_retries is set to 0 here so every paid attempt goes
    through one complete() call and the agent's own retry rules stay the only ones.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 1500):
        key = api_key()  # checked first so a missing key fails before anything else happens
        import anthropic  # lazy, so tests and CI do not need the package

        self.model = model or model_name()
        self.max_tokens = max_tokens
        # api_key is passed explicitly so the SDK never falls back to ANTHROPIC_API_KEY;
        # base_url is fixed for the same reason (ANTHROPIC_BASE_URL could send the key elsewhere).
        self._client = anthropic.Anthropic(api_key=key, base_url=API_BASE_URL, max_retries=0)

    def model_available(self) -> bool:
        """Whether this key can see self.model (a models-list call; free, but it does touch the network)."""
        return any(m.id == self.model for m in self._client.models.list(limit=1000))

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        r = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, tools=tools
        )
        content = []
        for b in r.content:
            if b.type == "text":
                content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        u = r.usage
        n = lambda v: v if v is not None else 0  # noqa: E731  SDK reports None when a cache field is unused
        split = getattr(u, "cache_creation", None)
        if split is not None:
            w5, w1 = n(split.ephemeral_5m_input_tokens), n(split.ephemeral_1h_input_tokens)
        else:
            w5, w1 = 0, n(getattr(u, "cache_creation_input_tokens", None))  # unsplit: dearer rate, never under-count
        return Turn(content=content, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                    cache_write_5m_tokens=w5, cache_write_1h_tokens=w1,
                    cache_read_tokens=n(getattr(u, "cache_read_input_tokens", None)))


def price(model: str) -> ModelPrice:
    """Raises on an unknown model: a missing price must never silently become a free call."""
    try:
        return PRICES_USD_PER_MTOK[model]
    except KeyError:
        raise ValueError(f"no price configured for model {model!r}; add it to PRICES_USD_PER_MTOK") from None


def _finite_nonneg(x) -> bool:
    """Never raises. math.isfinite(10**400) raises OverflowError, so ints are compared to the
    largest float instead (exact int/float comparison); anything larger counts as invalid."""
    if isinstance(x, bool):
        return False
    if isinstance(x, float):
        return math.isfinite(x) and x >= 0
    return isinstance(x, int) and 0 <= x <= sys.float_info.max


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_write_5m: int = 0,
             cache_write_1h: int = 0, cache_read: int = 0) -> float:
    p = price(model)
    counts = (input_tokens, output_tokens, cache_write_5m, cache_write_1h, cache_read)
    if not all(_finite_nonneg(c) for c in counts):
        raise ValueError(f"invalid token counts: {counts!r}")
    long = input_tokens + cache_write_5m + cache_write_1h + cache_read > LONG_PROMPT_TOKENS
    r_in, r_out, w5, w1, rd = ((p.long_input, p.long_output, p.long_write_5m, p.long_write_1h, p.long_read)
                               if long else (p.input, p.output, p.write_5m, p.write_1h, p.read))
    total = input_tokens * r_in + output_tokens * r_out + cache_write_5m * w5 + cache_write_1h * w1 + cache_read * rd
    return total / 1_000_000


class Ledger:
    """Spend caps for ONE run (build one per run). The day total persists in a JSON file.

    What counts: model cost and search_docs cost_usd both go through charge_usd(), so both
    count against the per-run cap and the per-day total.
    check() runs before every model call. A single call can still overshoot a cap by its own
    cost, because the cost is only known afterwards; the next check then stops the run.
    Any unreadable, negative, NaN/inf or non-numeric state sets load_error and fails closed.
    Each charge is a locked read-modify-write with an atomic replace, so two Ledgers on one
    file add up correctly.
    """

    def __init__(self, per_run_cap: float = DEFAULT_RUN_CAP_USD, per_day_cap: float = DEFAULT_DAY_CAP_USD,
                 path: Path = SPEND_PATH, day: str | None = None):
        self.per_run_cap, self.per_day_cap, self.path = per_run_cap, per_day_cap, Path(path)
        self.day = day or date.today().isoformat()
        self.run_usd = 0.0
        self.day_usd = 0.0
        self.load_error: str | None = None
        self._refresh()

    def _read_day(self) -> float:
        """Day total from the file; ValueError/OSError on anything wrong."""
        if not self.path.exists():
            return 0.0
        saved = json.loads(self.path.read_text())
        if not isinstance(saved, dict) or not isinstance(saved.get("date"), str):
            raise ValueError("unexpected shape")
        if saved["date"] != self.day:
            return 0.0
        if not _finite_nonneg(saved.get("usd")):
            raise ValueError(f"invalid usd {saved.get('usd')!r}")
        return float(saved["usd"])

    def _refresh(self) -> None:
        try:
            self.day_usd = self._read_day()
        except Exception as e:  # incl. RecursionError from a deeply nested file; always fail closed
            self.load_error = f"spend file unreadable: {e!r}"[:200]  # fail closed, never reset to 0

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def check(self, model: str) -> str | None:
        """Reason the next model call must not happen, or None. Raises on an unknown model."""
        price(model)
        if not self.load_error:
            self._refresh()  # another ledger may have spent since we last looked
        if self.load_error:
            return self.load_error
        if self.run_usd >= self.per_run_cap:
            return f"per-run cap reached: {self.run_usd:.6f} >= {self.per_run_cap} USD"
        if self.day_usd >= self.per_day_cap:
            return f"per-day cap reached: {self.day_usd:.6f} >= {self.per_day_cap} USD"
        return None

    def charge_usd(self, usd: float) -> float:
        """Record spend. Never raises: failures set load_error, and the real cost stays in memory."""
        if not _finite_nonneg(usd):
            self.load_error = f"invalid spend amount {usd!r}"
            return 0.0
        usd = float(usd)
        self.run_usd += usd  # in memory first, so the cost is kept even if the write fails
        try:
            with self._locked():
                total = self._read_day() + usd
                if not math.isfinite(total):
                    raise ValueError("day total is not finite")  # never write Infinity to the file
                fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".spend-")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump({"date": self.day, "usd": total}, f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, self.path)
                except BaseException:
                    Path(tmp).unlink(missing_ok=True)
                    raise
            self.day_usd = total
        except Exception as e:
            self.day_usd += usd
            self.load_error = f"spend ledger write failed: {e!r}"[:200]
        return usd

    def charge(self, model: str, input_tokens: int, output_tokens: int, cache_write_5m: int = 0,
               cache_write_1h: int = 0, cache_read: int = 0) -> float:
        price(model)  # raises on unknown model
        try:
            usd = cost_usd(model, input_tokens, output_tokens, cache_write_5m, cache_write_1h, cache_read)
        except ValueError as e:
            self.load_error = f"invalid charge: {e}"[:200]
            return 0.0

        return self.charge_usd(usd)
