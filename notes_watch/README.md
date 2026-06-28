# notes_watch — self-gating watcher over the operator-note corpus

The RAG note search (`rag/`) answers questions **on demand**. `notes_watch` is
the **push** side: it wakes on a schedule, decides for itself whether anything
new in the note corpus is worth your attention, and only then writes a detailed
summary and notifies you — **Telegram** (the ping) + a dated **Obsidian note**
(the record).

## How it "decides when to run"

Two stages, cheap-before-expensive:

```
launchd (every 30 min)                     ← dumb clock, the heartbeat
   └─ runner: load corpus, diff vs watermark
        ├─ no new notes ............................. exit silent
        ├─ cooldown active (<6h since last alert) ... defer, exit silent
        ├─ LLM gate over the NEW notes only ......... "alert? severity? why?"
        │     └─ below threshold ................... mark seen, exit silent
        └─ ALERT → build detailed summary → Telegram + Obsidian note
```

- **Watermark** (`state.py`): each note gets a content signature; the *delta* is
  notes never seen before. An edited note re-triggers.
- **Materiality gate** (`llm.py`): a cheap `claude` call judges only the new
  notes — `{alert, severity 0-100, why, themes}`. Fails **open** (alerts) on any
  CLI error so a real event is never dropped silently.
- **Cooldown + threshold** keep it from spamming you.

The LLM runs through the local **`claude` CLI on your Max subscription** — no API
key, no metered cost (same env-scrub trick as HERALD). Gate uses Haiku (fast),
the narrative uses Sonnet.

## Run it

```bash
cd notes_watch
cp .env.example .env          # (already populated here with the HERALD bot creds)
./run.sh --selftest           # prove the Max claude -p gate answers
./run.sh --dry-run            # print the Telegram + Obsidian output, send nothing
./run.sh                      # one real cycle (first run = baseline summary)
./run.sh --force              # ignore the cooldown
```

## Schedule it

```bash
./install-schedule.sh         # load the launchd job (every 30 min, RunAtLoad)
./install-schedule.sh uninstall
launchctl unload ~/Library/LaunchAgents/com.opsnotes.watch.plist   # pause
tail -f state/notes_watch.log
```

## Config (`.env`)

| key | default | meaning |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | — | the push target |
| `NOTES_WATCH_MIN_NEW` | 1 | below this many new notes, never alert |
| `NOTES_WATCH_COOLDOWN_HOURS` | 6 | min hours between alerts |
| `NOTES_WATCH_SEVERITY` | 45 | gate severity needed to alert |
| `NOTES_WATCH_GATE_MODEL` | Haiku 4.5 | fast yes/no gate |
| `NOTES_WATCH_SUMMARY_MODEL` | Sonnet 4.6 | narrative |
| `NOTES_WATCH_VAULT_INBOX` | `…/Cowork-Brain/00-Inbox/notes-watch` | note location |

## Notes

- **Corpus source** is `rag.corpus.build_note_records()` — the real
  `events.csv` rows + the seeded synthetic notes. In production, point the
  corpus at the live operator event log and every appended note flows through
  the gate automatically.
- **pgvector is optional.** If the RAG DB is up the alert is enriched with
  semantic-search context for "emerging fleet risks"; if it's down the watcher
  still runs (corpus aggregates + LLM narrative).
- **State** lives in `state/state.json` (git-ignored). Delete it to re-baseline.
