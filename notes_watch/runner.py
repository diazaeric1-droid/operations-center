"""notes_watch runner — the orchestration the scheduler calls each tick.

    python -m notes_watch.runner            # one real cycle (gate, maybe notify)
    python -m notes_watch.runner --dry-run  # print what it WOULD do; no send, no state write
    python -m notes_watch.runner --force    # ignore the cooldown
    python -m notes_watch.runner --selftest # prove the Max `claude -p` gate works

Exit codes: 0 = ran clean (whether or not it alerted), 1 = error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from rag import corpus

from . import config, llm, notify, state

log = logging.getLogger("notes_watch.runner")


def _by_cause(records) -> list[dict]:
    agg: dict[str, dict] = {}
    for r in records:
        a = agg.setdefault(r.cause, {"cause": r.cause, "count": 0, "bbl": 0})
        a["count"] += 1
        a["bbl"] += int(r.deferred_bbl)
    return sorted(agg.values(), key=lambda c: (-c["count"], -c["bbl"]))


def _totals(records) -> dict:
    return {"count": len(records),
            "bbl": sum(int(r.deferred_bbl) for r in records)}


def _rag_context(cfg: config.Config) -> str | None:
    """Optional enrichment: if pgvector is reachable, retrieve standing-question
    hits. Never required — returns None when the DB or extras are absent."""
    try:
        from rag import store, engine  # lazy: avoids llama-index import when DB down
        ok, _ = store.ping(timeout=2.0)
        if not ok:
            return None
        eng = engine.NoteSearchEngine()
        if eng.index_size() == 0:
            return None
        hits = eng.retrieve(cfg.standing_question, top_k=5)
        return "\n".join(f"- [{h.score:.2f}] {h.well_id} · {h.cause} — {h.note}"
                         for h in hits) or None
    except Exception as e:  # noqa: BLE001 — enrichment is best-effort
        log.info("RAG enrichment skipped: %s", e)
        return None


def run_cycle(dry_run: bool = False, force: bool = False) -> int:
    cfg = config.load()
    st = state.WatchState.load(cfg.state_path)
    # Watch only the real operator-note sources by default; the synthetic notes
    # exist to give the RAG index volume, not to be alerted on. Point the corpus
    # at a live event log in production and everything is "events_csv" anyway.
    records = [r for r in corpus.build_note_records() if cfg.is_watched(r.source)]
    delta = st.delta(records)
    cold = st.is_cold_start
    mode = "baseline" if cold else "incremental"
    now = state.now_utc()

    log.info("tick: %d notes in corpus, %d new (%s)", len(records), len(delta), mode)

    # --- gate stage 0: nothing new -> exit silent ---------------------------
    if not cold and len(delta) < cfg.min_new_notes:
        log.info("no new notes (<%d) — nothing to do", cfg.min_new_notes)
        if not dry_run:
            st.last_run = state.iso(now)
            st.count = len(records)
            st.runs += 1
            st.save(cfg.state_path)
        return 0

    # --- cooldown: defer (do NOT consume the delta) -------------------------
    if not cold and not force and st.cooldown_active(cfg.cooldown_hours, now):
        log.info("cooldown active (<%dh since last alert) — deferring %d new notes",
                 cfg.cooldown_hours, len(delta))
        if not dry_run:
            st.last_run = state.iso(now)
            st.runs += 1
            st.save(cfg.state_path)
        return 0

    # --- gate stage 1: LLM materiality (skipped for the baseline intro) -----
    if cold:
        gate = {"alert": True, "severity": 50, "_llm_ok": True,
                "why": "Baseline summary on first run.",
                "themes": [c["cause"] for c in _by_cause(delta)[:4]]}
    else:
        gate = llm.gate(delta, model=cfg.gate_model, timeout=cfg.llm_timeout)
        log.info("gate: alert=%s severity=%d why=%s",
                 gate["alert"], gate["severity"], gate["why"][:80])
        if not (gate["alert"] and gate["severity"] >= cfg.severity_threshold):
            log.info("gate below threshold (%d<%d) — marking seen, no alert",
                     gate["severity"], cfg.severity_threshold)
            if not dry_run:
                st.mark_seen(delta)          # evaluated, judged not worth alerting
                st.last_run = state.iso(now)
                st.count = len(records)
                st.runs += 1
                st.save(cfg.state_path)
            return 0

    # --- stage 2: build the detailed summary --------------------------------
    by_cause = _by_cause(delta)
    totals = _totals(delta)
    rag_ctx = _rag_context(cfg)
    summary = llm.summarize(delta, by_cause, totals, model=cfg.summary_model,
                            timeout=cfg.llm_timeout, rag_context=rag_ctx)

    digest = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "new_count": len(delta),
        "total_count": len(records),
        "new_records": delta,
        "by_cause": by_cause,
        "gate": gate,
        "summary": summary,
        "rag_context": rag_ctx,
    }

    # --- deliver ------------------------------------------------------------
    tg_html = notify.render_telegram_html(digest)
    if dry_run:
        print("===== TELEGRAM (dry-run, not sent) =====\n")
        print(tg_html)
        print("\n===== OBSIDIAN NOTE (dry-run, not written) =====\n")
        print(notify.render_markdown(digest))
        return 0

    note_path = notify.write_vault_note(cfg.vault_inbox, digest)
    log.info("wrote vault note: %s", note_path)

    if cfg.telegram_ready:
        ok = notify.send_telegram(cfg.telegram_token, cfg.telegram_chat_id, tg_html)
        log.info("telegram sent: %s", ok)
    else:
        log.warning("telegram creds missing — skipped push (note still written)")

    # --- commit state -------------------------------------------------------
    st.mark_seen(delta)
    st.last_alert = state.iso(now)
    st.last_run = state.iso(now)
    st.count = len(records)
    st.runs += 1
    st.save(cfg.state_path)
    return 0


def selftest() -> int:
    """Prove the Max-subscription gate works end to end on a tiny sample."""
    cfg = config.load()
    recs = corpus.build_note_records()[:5]
    print(f"claude CLI present: {llm.available()}")
    g = llm.gate(recs, model=cfg.gate_model, timeout=cfg.llm_timeout)
    print(f"gate -> alert={g['alert']} severity={g['severity']} "
          f"llm_ok={g['_llm_ok']}")
    print(f"why: {g['why']}")
    return 0 if g.get("_llm_ok") else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="notes_watch runner")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be sent; no send, no state mutation")
    ap.add_argument("--force", action="store_true", help="ignore the cooldown")
    ap.add_argument("--selftest", action="store_true",
                    help="run one gate call to verify the Max CLI works")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.selftest:
            return selftest()
        return run_cycle(dry_run=args.dry_run, force=args.force)
    except Exception as e:  # noqa: BLE001
        log.error("cycle failed: %s", e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
