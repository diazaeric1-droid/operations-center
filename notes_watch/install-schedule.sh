#!/usr/bin/env bash
# Install (or reload) the notes_watch launchd job from the template.
#   ./install-schedule.sh            # install + load
#   ./install-schedule.sh uninstall  # unload + remove
set -euo pipefail
NW_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.opsnotes.watch"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [ "${1:-}" = "uninstall" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "uninstalled ${LABEL}"
    exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "${NW_DIR}/state"
chmod +x "${NW_DIR}/run.sh"
sed "s#__NW_DIR__#${NW_DIR}#g" \
    "${NW_DIR}/com.opsnotes.watch.plist.template" > "$PLIST"
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "loaded ${LABEL} (every 30 min). Logs: ${NW_DIR}/state/notes_watch.log"
echo "Pause:  launchctl unload ${PLIST}"
