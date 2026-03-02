#!/usr/bin/with-contenv bashio

SCAN=$(bashio::config 'scan_interval_hours')
DAYS=$(bashio::config 'days_history')

export HA_URL="http://supervisor/core"
export HA_WS="ws://supervisor/core/api/websocket"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"

bashio::log.info "Habitus starting — scan every ${SCAN}h, ${DAYS} days history"

# Wait for HA to be ready
bashio::log.info "Waiting for Home Assistant to be ready..."
sleep 30

while true; do
    python3 -u /app/habitus/main.py --days $DAYS || bashio::log.warning "Run failed, will retry next cycle"
    bashio::log.info "Sleeping ${SCAN}h until next run"
    sleep $((SCAN * 3600))
done
