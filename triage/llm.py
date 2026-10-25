"""Model access: LLM protocol, the (UNRUN) Anthropic adapter, price table and spend caps."""
import fcntl
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Protocol, TypedDict

# UNVERIFIED PLACEHOLDER: this model id was not checked against Anthropic's model list.
DEFAULT_MODEL = "claude-haiku-5-5"  # config value, override with env TRIAGE_MODEL


def model_name() -> str:
    return os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)


# UNVERIFIED PLACEHOLDER prices, USD per million tokens (input, output). Invented, not checked
# against any price list. Check them before trusting any cost number or cap derived from them.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (1.0, 5.0),
}

DEFAULT_RUN_CAP_USD = 0.05  # placeholder; the owner sets real caps
DEFAULT_DAY_CAP_USD = 1.00
SPEND_PATH = Path("data/spend.json")  # data/ is gitignored


class Turn(TypedDict):
    # blocks: {"type": "text", "text"} or {"type": "tool_use", "id", "name", "input"}
    content: list[dict]
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Turn: ...


class AnthropicLLM:
    """UNRUN: never exercised against the real API (no `anthropic` package, no live calls yet).

    The SDK retries failed requests internally (max_retries, default 2), which would spend
    money outside the Ledger's view. max_retries is set to 0 here so every paid attempt goes
    through one complete() call and the agent's own retry rules stay the only ones.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 1500):
        import anthropic  # lazy, so tests and CI do not need the package

        self.model = model or model_name()
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(max_retries=0)  # reads ANTHROPIC_API_KEY from the environment

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
        return Turn(content=content, input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)


def price(model: str) -> tuple[float, float]:
    """Raises on an unknown model: a missing price must never silently become a free call."""
    try:
        return PRICES_USD_PER_MTOK[model]
    except KeyError:
        raise ValueError(f"no price configured for model {model!r}; add it to PRICES_USD_PER_MTOK") from None


def _finite_nonneg(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price(model)
    if not (_finite_nonneg(input_tokens) and _finite_nonneg(output_tokens)):
        raise ValueError(f"invalid token counts: {input_tokens!r}, {output_tokens!r}")
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


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
        except (ValueError, OSError) as e:
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
        except (ValueError, OSError) as e:
            self.day_usd += usd
            self.load_error = f"spend ledger write failed: {e!r}"[:200]
        return usd

    def charge(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price(model)  # raises on unknown model
        try:
            usd = cost_usd(model, input_tokens, output_tokens)
        except ValueError as e:
            self.load_error = f"invalid charge: {e}"[:200]
            return 0.0

        return self.charge_usd(usd)
