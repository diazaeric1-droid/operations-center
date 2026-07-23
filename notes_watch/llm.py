"""LLM access via the local ``claude`` CLI (Max subscription, no API key).

Mirrors HERALD's proven pattern: scrub the env so the subprocess uses THIS
machine's persistent ``claude`` login instead of any inherited API key / managed
gateway, then ``claude -p ... --output-format json``.

Two calls:
  * gate()    — cheap yes/no materiality judge over the new notes.
  * summarize() — richer analyst narrative once the gate says "alert".

Both ask for a strict JSON object and parse defensively (the model occasionally
wraps JSON in a ```fence```). On any failure the caller decides the fallback;
the gate fails OPEN to "alert" so we never silently swallow a real event.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

log = logging.getLogger("notes_watch.llm")


class LLMError(RuntimeError):
    pass


def _cli_env() -> dict:
    env = dict(os.environ)
    for k in list(env):
        if k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL") \
                or k.startswith("CLAUDE_CODE_") or k.startswith("CLAUDE_AGENT_SDK"):
            env.pop(k, None)
    return env


def available() -> bool:
    return shutil.which("claude") is not None


def _claude_json(prompt: str, model: str, timeout: int) -> dict:
    """Run one headless `claude -p` call and parse a JSON object from its result."""
    if not available():
        raise LLMError("`claude` CLI not found on PATH")
    proc = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json", "--model", model],
        capture_output=True, text=True, timeout=timeout, env=_cli_env(),
    )
    try:
        envelope = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMError(f"claude CLI gave non-JSON (rc={proc.returncode}): "
                       f"{(proc.stderr or proc.stdout).strip()[:200]}") from e
    if envelope.get("is_error"):
        raise LLMError(f"claude CLI error: {envelope.get('result')}")
    text = (envelope.get("result") or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):text.rfind("}") + 1]
    elif not text.startswith("{"):  # stray prose around the object
        text = text[text.find("{"):text.rfind("}") + 1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMError(f"could not parse JSON from model: {text[:200]}") from e


def _notes_block(records, limit: int = 40) -> str:
    lines = []
    for r in records[:limit]:
        lines.append(f"- ({r.well_id}, {r.start_date}, {r.cause}, "
                     f"{r.duration_days}d, {r.deferred_bbl} bbl) {r.note}")
    extra = len(records) - limit
    if extra > 0:
        lines.append(f"- … and {extra} more new notes")
    return "\n".join(lines)


# --- stage 1: the gate -------------------------------------------------------
def gate(records, model: str, timeout: int) -> dict:
    """Decide whether these NEW notes warrant alerting the operations lead.

    Returns {alert: bool, severity: int 0-100, why: str, themes: [str]}.
    Fails OPEN (alert=True, severity=60) on any LLM error so a real event is
    never dropped by a transient CLI hiccup.
    """
    prompt = (
        "You are a sharp, neutral production-operations analyst for an oil & gas "
        "fleet. Below are NEW operator downtime/curtailment notes logged since the "
        "last check. Decide whether, taken together, they warrant alerting the "
        "operations lead RIGHT NOW (a cluster of related failures, a high-impact "
        "outage, an emerging trend, or anything safety/permit-related = yes; "
        "routine isolated low-bbl events = no).\n\n"
        "Respond with ONLY a JSON object, no prose, no markdown fences, with "
        "exactly these keys: alert (boolean), severity (integer 0-100), why "
        "(string, <=2 sentences), themes (array of short strings).\n\n"
        f"New notes ({len(records)}):\n{_notes_block(records)}"
    )
    try:
        d = _claude_json(prompt, model=model, timeout=timeout)
    except LLMError as e:
        log.warning("gate LLM failed, failing open to alert: %s", e)
        return {"alert": True, "severity": 60,
                "why": f"(LLM gate unavailable — alerting to be safe: {e})",
                "themes": [], "_llm_ok": False}
    return {
        "alert": bool(d.get("alert", True)),
        "severity": max(0, min(100, int(d.get("severity", 50)))),
        "why": str(d.get("why", "")).strip(),
        "themes": [str(t) for t in (d.get("themes") or [])][:6],
        "_llm_ok": True,
    }


# --- stage 2: the narrative --------------------------------------------------
def summarize(records, by_cause: list[dict], totals: dict, model: str,
              timeout: int, rag_context: str | None = None) -> dict:
    """Produce the analyst narrative for the alert.

    Numbers (by_cause, totals) are computed deterministically by the caller and
    passed in; the model only writes prose + judgement. Returns
    {headline, summary, watch_items: [str], recommended_actions: [str]}.
    """
    cause_line = ", ".join(f"{c['cause']} ×{c['count']} ({c['bbl']:,} bbl)"
                           for c in by_cause)
    rag_blk = (f"\n\nSemantic-search context for "
               f"\"emerging fleet risks\":\n{rag_context}" if rag_context else "")
    prompt = (
        "You are a production-operations analyst. Write a concise but substantive "
        "shift summary of the NEW operator notes below for the operations lead. "
        "Be specific and quantitative; never invent wells, dates, or numbers "
        "beyond what is given.\n\n"
        "Respond with ONLY a JSON object (no prose, no fences) with exactly these "
        "keys: headline (string, <=12 words), summary (string, 3-6 sentences), "
        "watch_items (array of short strings — what to keep an eye on), "
        "recommended_actions (array of short, concrete next steps).\n\n"
        f"New notes: {totals['count']}; total {totals['bbl']:,} bbl deferred.\n"
        f"By cause: {cause_line}.\n\n"
        f"Notes:\n{_notes_block(records)}{rag_blk}"
    )
    try:
        d = _claude_json(prompt, model=model, timeout=timeout)
    except LLMError as e:
        log.warning("summary LLM failed, using deterministic fallback: %s", e)
        return {
            "headline": f"{totals['count']} new operator notes "
                        f"({totals['bbl']:,} bbl deferred)",
            "summary": f"New notes by cause: {cause_line}. "
                       f"(LLM narrative unavailable: {e})",
            "watch_items": [], "recommended_actions": [], "_llm_ok": False,
        }
    return {
        "headline": str(d.get("headline", "")).strip()
                    or f"{totals['count']} new operator notes",
        "summary": str(d.get("summary", "")).strip(),
        "watch_items": [str(x) for x in (d.get("watch_items") or [])][:8],
        "recommended_actions":
            [str(x) for x in (d.get("recommended_actions") or [])][:8],
        "_llm_ok": True,
    }
