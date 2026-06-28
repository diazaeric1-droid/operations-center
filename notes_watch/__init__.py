"""notes_watch — scheduled, self-gating watcher over the operator-note corpus.

The RAG note search (``rag/``) answers questions on demand. ``notes_watch`` is
the *push* side: it wakes on a schedule, decides for itself whether anything new
in the note corpus is worth the operator's attention, and only then writes a
detailed summary and notifies the user (Telegram + a dated Obsidian note).

Two-stage "decide when to run":
  1. Heartbeat   — launchd fires the job every N minutes (a dumb clock).
  2. Gate        — per tick: (a) a change watermark (no new notes -> exit silent),
                   then (b) a cheap LLM materiality judge over only the *new*
                   notes ("is this worth alerting? yes/no + why"). Spam is capped
                   by a cooldown.

The LLM runs through the local ``claude`` CLI on the user's Max subscription
(no API key, no metered cost) — the same trick HERALD uses.
"""
from __future__ import annotations

__all__ = ["config", "state", "llm", "notify", "runner"]
