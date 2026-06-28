#!/usr/bin/env bash
# notes_watch runner — invoked by launchd and by hand.
#   ./run.sh             -> one real cycle (gate, maybe Telegram + vault note)
#   ./run.sh --dry-run   -> print what it would send; no send, no state change
#   ./run.sh --force     -> ignore the cooldown
#   ./run.sh --selftest  -> prove the Max `claude -p` gate works
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (so `import rag` / `import notes_watch` resolve)
# launchd/cron run with a bare PATH — make sure `claude` resolves (Max login).
export PATH="/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
exec ./.venv/bin/python -m notes_watch.runner "$@"
