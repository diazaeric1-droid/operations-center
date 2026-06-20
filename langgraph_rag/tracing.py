"""Observability — trace every LLM call (tokens, latency, $ cost) so model and
routing decisions are DATA-DRIVEN instead of vibes.

Wrap any provider call with chat_traced() inside a `with trace() as events:` block;
each call records a TraceEvent (provider, model, tokens in/out, latency, estimated
cost). summary() rolls them up per provider — exactly the table you need to decide
"is the premium model worth 4× the latency and 10× the cost for this step?".

HONESTY: token counts are REAL (from each API's usage field) when the provider
reports them. Costs are ESTIMATES from approximate published list prices (see
PRICES) — providers change pricing, and free tiers cost $0 in reality; the dollar
figure is "what this would cost at paid scale", a planning number, not a bill.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

# Approximate published list prices, USD per 1M tokens (input, output). ILLUSTRATIVE
# — list prices as of 2026-06; providers change them, so verify before quoting.
# Matched by substring against the model id (specific keys before general ones).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus": (5.0, 25.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini": (0.30, 2.50),
    "llama-3.3-70b": (0.59, 0.79),
    "llama": (0.20, 0.30),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-": (2.50, 10.0),
}


def price_for(model: str) -> tuple[float, float, bool]:
    """(input $/1M, output $/1M, known?) for a model id via substring match."""
    m = (model or "").lower()
    for key, (pin, pout) in PRICES.items():
        if key in m:
            return pin, pout, True
    return 0.0, 0.0, False


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout, _ = price_for(model)
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout


@dataclass
class TraceEvent:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    tokens_estimated: bool = False
    price_known: bool = True
    label: str = ""
    prompt_estimated: bool = False
    completion_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# --- the active trace (a list events append to) -------------------------------
_active: Optional[list] = None


@contextmanager
def trace():
    """Collect TraceEvents emitted by chat_traced() within this block."""
    global _active
    prev, buf = _active, []
    _active = buf
    try:
        yield buf
    finally:
        _active = prev


def record(event: TraceEvent) -> None:
    if _active is not None:
        _active.append(event)


def chat_traced(prompt: str, provider: str = "claude", label: str = "",
                **kwargs) -> str:
    """providers.chat_meta() + timing + cost, recorded to the active trace."""
    from . import providers
    t = time.time()
    meta = providers.chat_meta(prompt, provider=provider, **kwargs)
    dt = time.time() - t
    _, _, known = price_for(meta["model"])
    record(TraceEvent(
        provider=provider, model=meta["model"],
        prompt_tokens=meta["prompt_tokens"], completion_tokens=meta["completion_tokens"],
        latency_s=round(dt, 3),
        cost_usd=cost_usd(meta["model"], meta["prompt_tokens"], meta["completion_tokens"]),
        tokens_estimated=meta["tokens_estimated"], price_known=known, label=label,
        prompt_estimated=meta.get("prompt_estimated", meta["tokens_estimated"]),
        completion_estimated=meta.get("completion_estimated", meta["tokens_estimated"])))
    return meta["text"]


# --- rollups ------------------------------------------------------------------
def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def summary(events: list) -> dict:
    """Totals + per-provider breakdown (calls, tokens, cost, p50/p95 latency).

    `models` is the SORTED LIST of models a provider served (a provider can run
    several under per-step routing), so the cost is never tagged to just one of
    them. `price_unknown` flags a provider whose cost is understated because some
    model had no list price.
    """
    by: dict[str, dict] = {}
    for e in events:
        b = by.setdefault(e.provider, {"calls": 0, "tokens": 0, "cost": 0.0,
                                       "models": set(), "lat": [],
                                       "estimated": False, "price_unknown": False})
        b["calls"] += 1
        b["tokens"] += e.total_tokens
        b["cost"] += e.cost_usd
        b["lat"].append(e.latency_s)
        b["models"].add(e.model)
        b["estimated"] = b["estimated"] or e.tokens_estimated
        b["price_unknown"] = b["price_unknown"] or not e.price_known
    for b in by.values():
        b["p50_latency"] = round(_pct(b["lat"], 0.50), 3)
        b["p95_latency"] = round(_pct(b["lat"], 0.95), 3)
        b["cost"] = round(b["cost"], 6)
        b["models"] = sorted(b["models"])
        del b["lat"]
    return {
        "calls": len(events),
        "total_tokens": sum(e.total_tokens for e in events),
        "total_cost_usd": round(sum(e.cost_usd for e in events), 6),
        "price_unknown": any(not e.price_known for e in events),
        "by_provider": by,
    }


def print_report(events: list) -> None:
    print(f"\n{'provider':12s} {'model':22s} {'tok in/out':>13s} "
          f"{'lat':>7s} {'est $':>11s}")
    print("  " + "-" * 68)
    for e in events:
        # mark ONLY the count(s) actually estimated, not the whole pair
        toks = (f"{'~' if e.prompt_estimated else ''}{e.prompt_tokens}/"
                f"{'~' if e.completion_estimated else ''}{e.completion_tokens}")
        cost = f"${e.cost_usd:9.6f}" if e.price_known else "? no price"
        line = f"{e.provider:12s} {e.model[:22]:22s} {toks:>13s} {e.latency_s:6.2f}s {cost:>11s}"
        if e.label:
            line += f"  {e.label}"
        print(line)
    s = summary(events)
    print("  " + "-" * 68)
    # unknown-price calls count as $0, so the total is a LOWER BOUND when present
    total = f"{'≥ ' if s['price_unknown'] else ''}${s['total_cost_usd']:.6f}"
    print(f"  {s['calls']} calls · {s['total_tokens']} tokens · {total} (est. list price)")
    if any(e.prompt_estimated or e.completion_estimated for e in events):
        print("  ~ = token count estimated (provider didn't report usage)")
    if s["price_unknown"]:
        print("  ? = no list price on file for this model — shown as no cost, NOT free")
