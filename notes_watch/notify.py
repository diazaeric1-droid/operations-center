"""Delivery: Telegram push + a dated Obsidian note in the vault Inbox.

Telegram for the *ping* (you see it on your phone now); the vault note for the
*record* (searchable, linkable, survives). Both render the same digest.
"""
from __future__ import annotations

import html
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

log = logging.getLogger("notes_watch.notify")

_API = "https://api.telegram.org/bot{token}/{method}"
_MAX_LEN = 4096


def _esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def _truncate(text: str, limit: int = _MAX_LEN) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def send_telegram(token: str, chat_id: str, html_text: str,
                  max_retries: int = 4) -> bool:
    """POST sendMessage (HTML), retrying on 429 per Telegram's retry_after."""
    url = _API.format(token=token, method="sendMessage")
    payload = {"chat_id": chat_id, "text": _truncate(html_text),
               "parse_mode": "HTML", "disable_web_page_preview": True}
    for attempt in range(max_retries):
        try:
            r = requests.post(url, data=payload, timeout=30)
        except requests.RequestException as e:
            log.warning("telegram send error: %s", e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return True
        if r.status_code == 429:
            retry_after = 1
            try:
                retry_after = r.json().get("parameters", {}).get("retry_after", 1)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(retry_after + 0.5)
            continue
        log.warning("telegram HTTP %s: %s", r.status_code, r.text[:300])
        return False
    return False


# --- rendering ---------------------------------------------------------------
def render_telegram_html(digest: dict) -> str:
    """digest -> Telegram HTML message."""
    g, s = digest["gate"], digest["summary"]
    sev = g["severity"]
    dot = "🔴" if sev >= 70 else "🟠" if sev >= 45 else "🟡"
    parts = [f"{dot} <b>{_esc(s['headline'])}</b>",
             f"<i>severity {sev}/100 · {digest['mode']} · "
             f"{digest['new_count']} new notes</i>", ""]
    if s.get("summary"):
        parts.append(_esc(s["summary"]))
    parts.append("")
    parts.append("<b>By cause:</b> " + _esc(", ".join(
        f"{c['cause']} ×{c['count']} ({c['bbl']:,} bbl)"
        for c in digest["by_cause"])))
    if g.get("why"):
        parts.append(f"<b>Why flagged:</b> {_esc(g['why'])}")
    if s.get("watch_items"):
        parts.append("\n<b>Watch:</b>")
        parts += [f"• {_esc(w)}" for w in s["watch_items"]]
    if s.get("recommended_actions"):
        parts.append("\n<b>Actions:</b>")
        parts += [f"• {_esc(a)}" for a in s["recommended_actions"]]
    parts.append(f"\n<i>notes_watch · {_esc(digest['ts'])}</i>")
    return "\n".join(parts)


def render_markdown(digest: dict) -> str:
    """digest -> Obsidian note body."""
    g, s = digest["gate"], digest["summary"]
    lines = [
        "---",
        f"created: {digest['ts']}",
        "source: notes_watch",
        f"severity: {g['severity']}",
        f"mode: {digest['mode']}",
        f"new_notes: {digest['new_count']}",
        f"tags: [operations, notes-watch]",
        "---",
        "",
        f"# {s['headline']}",
        "",
        f"**Severity {g['severity']}/100** · {digest['mode']} · "
        f"{digest['new_count']} new notes · {digest['total_count']} in corpus",
        "",
    ]
    if s.get("summary"):
        lines += [s["summary"], ""]
    if g.get("why"):
        lines += [f"> **Why flagged:** {g['why']}", ""]
    if g.get("themes"):
        lines += ["**Themes:** " + ", ".join(g["themes"]), ""]
    lines += ["## New notes by cause", "",
              "| Cause | Count | Deferred bbl |", "|---|---|---|"]
    lines += [f"| {c['cause']} | {c['count']} | {c['bbl']:,} |"
              for c in digest["by_cause"]]
    lines.append("")
    if s.get("watch_items"):
        lines += ["## Watch", ""] + [f"- {w}" for w in s["watch_items"]] + [""]
    if s.get("recommended_actions"):
        lines += ["## Recommended actions", ""] \
            + [f"- [ ] {a}" for a in s["recommended_actions"]] + [""]
    if digest.get("rag_context"):
        lines += ["## Semantic-search context", "", digest["rag_context"], ""]
    lines += ["## New note detail", ""]
    lines += [f"- `{r.well_id}` · {r.start_date} · **{r.cause}** · "
              f"{r.duration_days}d · {r.deferred_bbl:,} bbl — {r.note}"
              for r in digest["new_records"][:40]]
    return "\n".join(lines)


def write_vault_note(inbox: Path, digest: dict) -> Path:
    """Write the dated markdown note; returns the path."""
    inbox.mkdir(parents=True, exist_ok=True)
    stamp = digest["ts"].replace(":", "").replace("T", "-")[:13]
    slug = "".join(c if c.isalnum() or c in "- " else ""
                   for c in digest["summary"]["headline"]).strip()
    slug = "-".join(slug.lower().split())[:50] or "notes-watch"
    path = inbox / f"{stamp} — {slug}.md"
    path.write_text(render_markdown(digest))
    return path


def now_stamp() -> str:
    """Caller passes wall-clock in; kept here so renderers stay pure."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
