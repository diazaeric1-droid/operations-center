---
title: Daily Production Digest
emoji: 📅
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# Daily Production Digest

> A scheduled AI agent that runs every morning, scans your fleet's overnight SCADA, flags anomalies, and writes a one-page brief in the format a Senior PE hands to the asset team's daily standup.

Built by a Staff Production Engineer (ex-OXY, ex-Shell) who used to write this brief by hand at 6am every morning.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://daily-pe-digest.streamlit.app)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

**Try it now → [daily-pe-digest.streamlit.app](https://daily-pe-digest.streamlit.app)**

---

## The problem

Every asset team starts the day asking the same three questions: *what changed overnight, what needs attention right now, and where do we stand against plan?* Answering them requires a human to pull data from 3-5 systems, eyeball trends, and write a brief — typically 60-90 minutes of senior-engineer time per day, per asset.

This system collapses that to 30 seconds. Scheduled, deterministic, repeatable.

## What it does

Every morning (cron, GitHub Actions, or Streamlit "Run Now" button):

1. **Ingests** the last 24 hours of fleet SCADA (synthetic generator included; production deployments plug into PI / Ignition / OSIsoft historians)
2. **Detects anomalies** with deterministic Python rules — decline-aware rate drops (Arps fit, not a flat mean), intake-pressure collapse, motor-temp spikes, runtime degradation, amps creep, and **data-quality events** (comms loss vs. metering dropout vs. real trip)
3. **Prices each flag** — converts a rate drop into **deferred barrels and $/day**, and ranks the brief by money at risk, not z-score
4. **Suppresses known events** via `acknowledged.yml` so a planned workover doesn't re-fire HIGH every morning (alarm-fatigue control)
5. **Writes the brief** — Claude narrates in Senior-PE voice (or a deterministic template when no API key is set; detection was always deterministic), then persists to disk + a Streamlit history view

## Architecture

```
                       ┌──────────────────────────┐
   cron / GH Actions ─▶│   src/scheduler.py       │◀── Streamlit "Run Now"
                       └─────────────┬────────────┘
                                     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  data_loader.py  →  anomaly_detector.py  →  brief_writer.py  │
   │  (load SCADA)       (deterministic rules)    (Claude prose)  │
   └─────────────────────────────────┬────────────────────────────┘
                                     ▼
                       ┌──────────────────────────┐
                       │  briefs/YYYY-MM-DD.md    │
                       └──────────────────────────┘
```

LLM is used only for the narrative layer. Anomaly detection is deterministic Python — engineers trust the numbers, the LLM writes them up. **With no API key the brief is still produced** (deterministic template), so cron/CI never silently fail.

**Backtest honesty:** `python -m src.backtest` scores each rule against seeded anomalies *and* near-threshold decoy wells, so precision/recall aren't a trivial 1.00. It shows, for example, the flat-mean rate-drop rule false-positiving on a steep-but-healthy decliner (precision 0.50) while the decline-aware rule correctly stays quiet (1.00) — the concrete justification for the refinement. Lead-time is reported as detection latency from fault onset + early-warning days before full manifestation.

## Quick start

```bash
git clone https://github.com/<your-user>/daily-production-digest
cd daily-production-digest
pip install -e ".[demo]"
cp .env.example .env  # add ANTHROPIC_API_KEY

# Generate 30 days of synthetic fleet SCADA (50 wells)
python data/synthetic/generate_fleet.py

# Run the morning brief once
python -m src.scheduler

# Streamlit history viewer
streamlit run demo/app.py
```

## Scheduling

**Local (cron):**
```cron
0 6 * * * cd /path/to/daily-production-digest && /path/to/.venv/bin/python -m src.scheduler
```

**GitHub Actions (free, runs in cloud):** see `.github/workflows/morning-brief.yml`. Set `ANTHROPIC_API_KEY` as a repo secret; the workflow runs every weekday morning (`30 11 * * 1-5` UTC ≈ 6:30am Central during CDT — GitHub cron is fixed-UTC with no DST), commits the brief back to the repo, and — if you also set a `SLACK_WEBHOOK_URL` secret — posts the HIGH-priority items to Slack.

**Streamlit Cloud:** the deployed demo has a "Run Now" button so anyone can see what a fresh brief looks like without waiting.

## Sample brief

See [`briefs/sample.md`](briefs/sample.md) for a complete agent-generated brief on a synthetic 50-well fleet — top priorities ranked, field summary, anomalies surfaced, action items.

## Roadmap

- [x] v0.1 — Anomaly detector + Claude brief writer + Streamlit history; GitHub Actions daily run
- [x] v0.2 — Robust median/MAD z-scores, decline-aware rate drop, least-squares slopes, pluggable historian adapters (CSV / SQLite), backtest harness
- [x] v0.3 — Deferred-bbl/$ ranking, sensor-dropout vs. comms-loss vs. trip detection, acknowledge/suppress (alarm fatigue), water-cut context, no-API-key deterministic brief, honest backtest (decoys + real lead-time), optional Slack notify
- [ ] v0.4 — Route/pad grouping + per-anomaly owner from a routes file
- [ ] v0.5 — Trend vs. last week (diff today's anomaly set against yesterday's brief)
- [ ] v0.6 — Chain into the ESP failure-risk model + AFE Copilot (detect → predict → draft authorization)

## Part of a multi-agent pipeline

This is the **detect** stage of a detect → predict → authorize chain: this digest
flags genuine ESP/pump-failure signatures and hands the well to the
[ESP Failure-Risk Agent](../esp-failure-risk-agent) (30-day risk + failure-mode
classification), which hands a diagnosis to the [AFE Copilot](../afe-copilot) to draft
the authorization. Export the handoff with `python -m src.handoff`. See
[`../pe-pipeline/PIPELINE.md`](../pe-pipeline/PIPELINE.md) and run the whole chain with `python3 ../pe-pipeline/pe_chain.py`.

## License

MIT.

## Contact

Eric Diaz II — [LinkedIn](https://www.linkedin.com/in/eric-a-diaz2) — diaz.a.eric1@gmail.com

Available for senior AI engineering roles and consulting engagements with E&P operators.
