#!/usr/bin/with-contenv bashio

SCAN=$(bashio::config 'scan_interval_hours')
DAYS=$(bashio::config 'days_history')
PORT=$(bashio::addon.ingress_port)

export HA_URL="http://supervisor/core"
export HA_WS="ws://supervisor/core/api/websocket"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export DATA_DIR="/data"

bashio::log.info "Habitus v0.7.0 — web UI on port ${PORT}, scan every ${SCAN}h"

# Start web server in background
python3 -c "
import os; os.environ['DATA_DIR']='/data'
from habitus.web import start_web
start_web(int('${PORT}'))
" &

# Wait for HA to be ready
bashio::log.info "Waiting 30s for HA to start..."
sleep 30

while true; do
    python3 -u /app/habitus/main.py --days $DAYS || bashio::log.warning "Run failed, will retry"
    bashio::log.info "Next run in ${SCAN}h"
    sleep $((SCAN * 3600))
done
