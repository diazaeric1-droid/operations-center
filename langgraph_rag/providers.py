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
from dataclasses import dataclass
from typing import Optional


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
         system: Optional[str] = None, max_tokens: int = 300,
         temperature: float = 0.0) -> str:
    """Send one prompt to `provider`, return the text reply.

    Same call shape regardless of vendor — that's the point. Raises ValueError on
    an unknown provider and RuntimeError when the provider's API key isn't set.
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
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        msg = anthropic.Anthropic(api_key=key).messages.create(**kwargs)
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    # OpenAI-compatible: gemini / groq / github / openrouter / openai
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=p.base_url)
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature)
    return (resp.choices[0].message.content or "").strip()


if __name__ == "__main__":   # python -m langgraph_rag.providers  -> who's ready?
    print("Provider readiness (key set in env?):\n")
    for name, p in PROVIDERS.items():
        mark = "✓" if os.environ.get(p.key_env) else "·"
        print(f"  {mark} {name:11s} {p.key_env:20s} {p.blurb}")
    ready = first_available()
    print(f"\nDefault 'auto' would use: {ready or '(none set — would fall back to stub)'}")
