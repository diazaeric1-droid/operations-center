"""Tests for the model-agnostic provider layer (no real API calls)."""
from __future__ import annotations

import pytest

from langgraph_rag import providers


def test_registry_has_the_expected_providers():
    for name in ("claude", "gemini", "groq", "github", "openrouter", "openai"):
        assert name in providers.PROVIDERS
    assert providers.PROVIDERS["claude"].kind == "anthropic"
    # the five OpenAI-compatible ones route through the openai client
    for name in ("gemini", "groq", "github", "openrouter", "openai"):
        assert providers.PROVIDERS[name].kind == "openai"
    # verified base URLs
    assert providers.PROVIDERS["gemini"].base_url.startswith(
        "https://generativelanguage.googleapis.com")
    assert providers.PROVIDERS["groq"].base_url == "https://api.groq.com/openai/v1"


def test_available_is_a_bool_map(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    a = providers.available()
    assert set(a) == set(providers.PROVIDERS)
    assert a["groq"] is False
    assert all(isinstance(v, bool) for v in a.values())


def test_first_available_picks_a_set_key(monkeypatch):
    for p in providers.PROVIDERS.values():        # clear the slate
        monkeypatch.delenv(p.key_env, raising=False)
    assert providers.first_available() is None
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert providers.first_available() == "groq"


def test_chat_unknown_provider_raises():
    with pytest.raises(ValueError):
        providers.chat("hi", provider="nope")


def test_chat_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as e:
        providers.chat("hi", provider="gemini")
    assert "GEMINI_API_KEY" in str(e.value)        # tells you exactly what to set


def test_resolve_tokens_real_partial_and_estimated():
    from langgraph_rag.providers import _resolve_tokens
    # full usage -> real numbers, not flagged estimated
    assert _resolve_tokens(120, 30, "prompt", "text") == (120, 30, False, False)
    # both missing -> ~4 chars/token estimate, both flagged
    pt, ct, pe, ce = _resolve_tokens(None, None, "a" * 40, "b" * 20)
    assert (pt, ct, pe, ce) == (10, 5, True, True)
    assert _resolve_tokens(None, None, "", "")[:2] == (1, 1)        # floor of 1
    # partial usage -> the REAL count is preserved and NOT flagged estimated
    pt, ct, pe, ce = _resolve_tokens(200, None, "x", "hello world")
    assert pt == 200 and pe is False and ce is True
