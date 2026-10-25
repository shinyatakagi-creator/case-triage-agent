"""Model access: LLM protocol, the (UNRUN) Anthropic adapter, price table and spend caps."""
import json
import os
from datetime import date
from pathlib import Path
from typing import Protocol, TypedDict

DEFAULT_MODEL = "claude-haiku-5-5"  # config value, override with env TRIAGE_MODEL


def model_name() -> str:
    return os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)


# UNVERIFIED PLACEHOLDER prices, USD per million tokens (input, output). Not checked against
# any price list. Check before trusting any cost number or cap derived from them.
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
    """UNRUN: never exercised against the real API (no `anthropic` package, no live calls yet)."""

    def __init__(self, model: str | None = None, max_tokens: int = 1500):
        import anthropic  # lazy, so tests and CI do not need the package

        self.model = model or model_name()
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

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


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = PRICES_USD_PER_MTOK[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


class Ledger:
    """Spend caps for ONE run (build one per run). The day total persists in a JSON file.

    check() runs before every model call. A single call can still overshoot a cap by its own
    cost, because the cost is only known afterwards; the next check then stops the run.
    """

    def __init__(self, per_run_cap: float = DEFAULT_RUN_CAP_USD, per_day_cap: float = DEFAULT_DAY_CAP_USD,
                 path: Path = SPEND_PATH, day: str | None = None):
        self.per_run_cap, self.per_day_cap, self.path = per_run_cap, per_day_cap, Path(path)
        self.day = day or date.today().isoformat()
        self.run_usd = 0.0
        self.day_usd = 0.0
        self.load_error: str | None = None
        try:
            if self.path.exists():
                saved = json.loads(self.path.read_text())
                if saved["date"] == self.day:
                    self.day_usd = float(saved["usd"])
        except (ValueError, KeyError, TypeError, OSError) as e:
            # Fail closed: an unreadable ledger must not silently reset the daily cap.
            self.load_error = f"spend file unreadable: {e!r}"[:200]

    def check(self, model: str) -> str | None:
        """Reason the next model call must not happen, or None."""
        if self.load_error:
            return self.load_error
        if model not in PRICES_USD_PER_MTOK:
            return f"no price configured for model {model!r}; refusing to call without cost accounting"
        if self.run_usd >= self.per_run_cap:
            return f"per-run cap reached: {self.run_usd:.6f} >= {self.per_run_cap} USD"
        if self.day_usd >= self.per_day_cap:
            return f"per-day cap reached: {self.day_usd:.6f} >= {self.per_day_cap} USD"
        return None

    def charge(self, model: str, input_tokens: int, output_tokens: int) -> float:
        c = cost_usd(model, input_tokens, output_tokens)
        self.run_usd += c
        self.day_usd += c
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"date": self.day, "usd": self.day_usd}))
        return c
