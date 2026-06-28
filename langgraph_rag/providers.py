"""Model-agnostic LLM layer — one `chat()` for Claude, Gemini, Groq, GitHub
Models (GPT-4o), OpenRouter, or OpenAI.

The trick that makes "provider-agnostic" cheap: **most providers speak the
OpenAI Chat Completions protocol**, so a single OpenAI client covers five of them
— you only change `base_url`, the API key, and the model name. Claude uses the
Anthropic SDK (its own protocol). So the same LangGraph agent runs on any model;
you swap with one argument.

Each provider needs its API key in an env var (see the table). All have a free
tier except OpenAI direct:

    provider     env var               get a free key at
    ─────────────────────────────────────────────────────────────────────
    claude       ANTHROPIC_API_KEY     console.anthropic.com ($5 trial)
    gemini       GEMINI_API_KEY        aistudio.google.com   (free, no card)
    groq         GROQ_API_KEY          console.groq.com      (free, no card)
    github       GITHUB_TOKEN          github.com  → a PAT with the Models scope
    openrouter   OPENROUTER_API_KEY    openrouter.ai/keys    (free open models)
    openai       OPENAI_API_KEY        platform.openai.com   (paid)
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

def _is_transient(e: Exception) -> bool:
    """A transient error worth retrying: 429 rate-limit, ANY 5xx (incl. 504 gateway
    timeout and Cloudflare 520-524), or a connection/timeout error (which carries no
    status_code, so we match by SDK class name)."""
    status = getattr(e, "status_code", None)
    if status is not None and (status == 429 or 500 <= status < 600):
        return True
    name = type(e).__name__
    return any(k in name for k in
               ("RateLimit", "Overload", "APITimeout", "APIConnection"))


def _call_with_retry(make_call, attempts: int = 3, base: float = 0.8):
    """Call an SDK function with exponential backoff + jitter on transient errors.
    Re-raises a non-transient error immediately and the last error after `attempts`.
    The resilience real LLM systems need — a single 429/504 shouldn't fail the run."""
    attempts = max(1, attempts)                  # always make at least one attempt
    for i in range(attempts):
        try:
            return make_call()
        except Exception as e:  # noqa: BLE001 — classify, then retry or re-raise
            if not _is_transient(e) or i == attempts - 1:
                raise
            time.sleep(base * (2 ** i) + random.uniform(0, 0.4))


@dataclass(frozen=True)
class Provider:
    name: str
    kind: str                  # "openai" (compatible) | "anthropic"
    key_env: str
    default_model: str
    base_url: Optional[str] = None   # None => the SDK's own default host
    blurb: str = ""


# base_url verified June 2026. Model names are sensible free-tier defaults you can
# override per call (--model); providers rotate model ids, so treat these as
# starting points, not guarantees.
PROVIDERS: dict[str, Provider] = {
    "claude": Provider(
        "claude", "anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-6",
        None, "Anthropic — coding / agentic strength"),
    # gemini-2.5-flash is Google's current FREE-tier Flash (gemini-2.0-flash now
    # returns quota "limit: 0" on new free keys — it's off the free tier). 2.5 is
    # a "thinking" model: it spends tokens reasoning before answering, so chat()
    # uses a generous default max_tokens to leave room for the visible reply.
    "gemini": Provider(
        "gemini", "openai", "GEMINI_API_KEY", "gemini-2.5-flash",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "Google AI Studio free tier (Flash)"),
    "groq": Provider(
        "groq", "openai", "GROQ_API_KEY", "llama-3.3-70b-versatile",
        "https://api.groq.com/openai/v1", "Groq free tier — very fast Llama"),
    "github": Provider(
        "github", "openai", "GITHUB_TOKEN", "gpt-4o",
        "https://models.inference.ai.azure.com",
        "GitHub Models free dev tier (GPT-4o)"),
    "openrouter": Provider(
        "openrouter", "openai", "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free",
        "https://openrouter.ai/api/v1", "OpenRouter — one key, many free models"),
    "openai": Provider(
        "openai", "openai", "OPENAI_API_KEY", "gpt-4o-mini",
        None, "OpenAI direct (paid)"),
}


def available() -> dict[str, bool]:
    """Which providers have their API key present in the environment."""
    return {name: bool(os.environ.get(p.key_env)) for name, p in PROVIDERS.items()}


def first_available() -> Optional[str]:
    """The first provider whose key is set (handy for an 'auto' default)."""
    for name, ok in available().items():
        if ok:
            return name
    return None


def chat(prompt: str, provider: str = "claude", model: Optional[str] = None,
         system: Optional[str] = None, max_tokens: int = 1024,
         temperature: float = 0.0) -> str:
    """Send one prompt to `provider`, return the text reply.

    Same call shape regardless of vendor — that's the point. Raises ValueError on
    an unknown provider and RuntimeError when the provider's API key isn't set.
    """
    return chat_meta(prompt, provider, model, system, max_tokens, temperature)["text"]


def chat_meta(prompt: str, provider: str = "claude", model: Optional[str] = None,
              system: Optional[str] = None, max_tokens: int = 1024,
              temperature: float = 0.0) -> dict:
    """Like chat() but returns the reply PLUS metadata for observability:
    {text, provider, model, prompt_tokens, completion_tokens, tokens_estimated}.

    Token counts come from the API's usage field when present; if a provider omits
    usage they're estimated (~4 chars/token) and tokens_estimated is True.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}'. options: {list(PROVIDERS)}")
    p = PROVIDERS[provider]
    key = os.environ.get(p.key_env)
    if not key:
        raise RuntimeError(
            f"{provider}: set {p.key_env} in your environment "
            f"(e.g. export {p.key_env}=...). {p.blurb}")
    model = model or p.default_model

    if p.kind == "anthropic":
        import anthropic
        kwargs = {"model": model, "max_tokens": max_tokens,
                  "temperature": temperature,
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        client = anthropic.Anthropic(api_key=key)
        msg = _call_with_retry(lambda: client.messages.create(**kwargs))
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        u = getattr(msg, "usage", None)
        pt = getattr(u, "input_tokens", None)
        ct = getattr(u, "output_tokens", None)
    else:  # OpenAI-compatible: gemini / groq / github / openrouter / openai
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=p.base_url)
        messages = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": prompt}]
        resp = _call_with_retry(lambda: client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature))
        text = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        pt = getattr(u, "prompt_tokens", None)
        ct = getattr(u, "completion_tokens", None)

    pt, ct, pe, ce = _resolve_tokens(pt, ct, prompt, text)
    return {"text": text, "provider": provider, "model": model,
            "prompt_tokens": pt, "completion_tokens": ct,
            "prompt_estimated": pe, "completion_estimated": ce,
            "tokens_estimated": pe or ce}


def _resolve_tokens(pt, ct, prompt: str, text: str):
    """Real usage when present, ~4-chars/token estimate per MISSING field only.
    Returns (prompt_tokens, completion_tokens, prompt_estimated, completion_estimated)
    so a partially-reported usage doesn't flag the real count as a guess."""
    prompt_estimated, completion_estimated = pt is None, ct is None
    if pt is None:
        pt = max(1, len(prompt) // 4)
    if ct is None:
        ct = max(1, len(text) // 4)
    return int(pt), int(ct), prompt_estimated, completion_estimated


if __name__ == "__main__":   # python -m langgraph_rag.providers  -> who's ready?
    print("Provider readiness (key set in env?):\n")
    for name, p in PROVIDERS.items():
        mark = "✓" if os.environ.get(p.key_env) else "·"
        print(f"  {mark} {name:11s} {p.key_env:20s} {p.blurb}")
    ready = first_available()
    print(f"\nDefault 'auto' would use: {ready or '(none set — would fall back to stub)'}")
