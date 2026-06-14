#!/usr/bin/env python3
"""Send the Operations Center morning brief by email — headless, for a scheduler.

Run by ``.github/workflows/daily-brief.yml`` every morning (or any cron / launchd
job). Bootstraps the data + ESP model if needed, composes the SAME brief the
Morning Brief page shows (digest anomalies + ongoing events + production
divergences & wells down), and emails it via SMTP.

Configuration is by environment variable (use repo/Action secrets — never commit
credentials):

    SMTP_HOST       smtp host (e.g. smtp.gmail.com)
    SMTP_PORT       587 (STARTTLS, default) or 465 (SSL)
    SMTP_USER       smtp username
    SMTP_PASS       smtp password / app-password
    BRIEF_FROM      From: address (defaults to SMTP_USER)
    BRIEF_TO        comma-separated recipient list
    BRIEF_OIL_PRICE realized oil $/bbl for the economics (default 70)
    BRIEF_NRI       net revenue interest for net deferred-$ (default 0.80)
    BRIEF_SUBJECT   optional subject override

Exit code 0 on success; non-zero (and a stderr message) on any failure so the
scheduler surfaces a red run instead of silently sending nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(f"daily_brief_email: missing required env var {name}")
    return val


def main() -> int:
    host = _require("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = _require("SMTP_USER")
    password = _require("SMTP_PASS")
    sender = os.environ.get("BRIEF_FROM", "").strip() or user
    recipients = [r.strip() for r in _require("BRIEF_TO").split(",") if r.strip()]
    price = float(os.environ.get("BRIEF_OIL_PRICE", "70"))
    nri = float(os.environ.get("BRIEF_NRI", "0.80"))

    import core
    import notify

    print("daily_brief_email: bootstrapping data + ESP model (idempotent)…",
          file=sys.stderr)
    core.bootstrap(log=lambda m: print(f"  {m}", file=sys.stderr))
    body = core.morning_brief_markdown(price_per_bbl=price, net_revenue_interest=nri)
    # Date the subject to the data's as-of day, matching the brief header + the page.
    as_of = core.scada_as_of()
    subject = os.environ.get("BRIEF_SUBJECT", "").strip() or \
        f"Operations Center — Morning Brief {as_of}"

    print(f"daily_brief_email: sending to {len(recipients)} recipient(s) via "
          f"{host}:{port}…", file=sys.stderr)
    notify.send_email(host=host, port=port, username=user, password=password,
                      sender=sender, recipients=recipients, subject=subject,
                      markdown_body=body, use_tls=(port != 465))
    print("daily_brief_email: sent.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
