"""Configuration for notes_watch.

Precedence per setting: real environment variable  ->  notes_watch/.env  ->
default. Secrets (Telegram token/chat id) live only in .env (git-ignored); the
behavioural knobs have sane defaults so a bare checkout still runs (dry-run).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ENV_FILE = _HERE / ".env"


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE, # comments). No external dep."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_DOTENV = _load_dotenv(_ENV_FILE)


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key) or _DOTENV.get(key) or default


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # --- delivery ---
    telegram_token: str
    telegram_chat_id: str
    vault_inbox: Path             # where the dated markdown note is written

    # --- "decide when to run" knobs ---
    min_new_notes: int            # below this many new notes, never alert
    cooldown_hours: int           # min hours between alerts (anti-spam)
    severity_threshold: int       # gate severity (0-100) needed to alert

    # --- LLM (claude CLI on Max) ---
    gate_model: str               # cheap/fast model for the yes/no gate
    summary_model: str            # richer model for the narrative
    llm_timeout: int

    # --- corpus / retrieval ---
    watch_sources: tuple[str, ...]  # which note sources to WATCH ("*" = all)
    standing_question: str        # RAG query run for context when DB is up
    state_path: Path

    def is_watched(self, source: str) -> bool:
        return "*" in self.watch_sources or source in self.watch_sources

    @property
    def telegram_ready(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)


def load() -> Config:
    return Config(
        telegram_token=_get("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_get("TELEGRAM_CHAT_ID"),
        vault_inbox=Path(_get(
            "NOTES_WATCH_VAULT_INBOX",
            str(Path.home() / ".openclaw" / "workspace" / "Cowork-Brain"
                / "00-Inbox" / "notes-watch"))).expanduser(),
        min_new_notes=_get_int("NOTES_WATCH_MIN_NEW", 1),
        cooldown_hours=_get_int("NOTES_WATCH_COOLDOWN_HOURS", 6),
        severity_threshold=_get_int("NOTES_WATCH_SEVERITY", 45),
        gate_model=_get("NOTES_WATCH_GATE_MODEL", "claude-haiku-4-5-20251001"),
        summary_model=_get("NOTES_WATCH_SUMMARY_MODEL", "claude-sonnet-4-6"),
        llm_timeout=_get_int("NOTES_WATCH_LLM_TIMEOUT", 180),
        watch_sources=tuple(
            s.strip() for s in _get("NOTES_WATCH_SOURCES", "events_csv").split(",")
            if s.strip()) or ("events_csv",),
        standing_question=_get(
            "NOTES_WATCH_STANDING_Q",
            "What are the emerging downtime and curtailment risks across the "
            "fleet, and which wells or causes are driving them?"),
        state_path=Path(_get(
            "NOTES_WATCH_STATE",
            str(_HERE / "state" / "state.json"))).expanduser(),
    )
