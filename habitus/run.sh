#!/usr/bin/with-contenv bashio

SCAN=$(bashio::config 'scan_interval_hours')
DAYS=$(bashio::config 'days_history')
PORT=$(bashio::addon.ingress_port)

export HA_URL="http://supervisor/core"
export HA_WS="ws://supervisor/core/api/websocket"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export DATA_DIR="/data"
RESCAN_FLAG="/data/.rescan_requested"

bashio::log.info "Habitus v0.8.0 — web UI on port ${PORT}, scan every ${SCAN}h"

# Start web server in background
python3 -c "
import os; os.environ['DATA_DIR']='/data'
from habitus.web import start_web
start_web(int('${PORT}'))
" &

bashio::log.info "Waiting 30s for HA to start..."
sleep 30

while true; do
    if [ -f "$RESCAN_FLAG" ]; then
        bashio::log.info "Full rescan requested via web UI — running now"
        rm -f "$RESCAN_FLAG"
    fi
    python3 -u /app/habitus/main.py --days $DAYS || bashio::log.warning "Run failed, will retry"
    bashio::log.info "Next run in ${SCAN}h"
    # Check flag every 5 min instead of sleeping full interval
    for i in $(seq 1 $((SCAN * 12))); do
        sleep 300
        if [ -f "$RESCAN_FLAG" ]; then
            bashio::log.info "Full rescan requested — triggering early run"
            rm -f "$RESCAN_FLAG"
            break
        fi
    done
done
