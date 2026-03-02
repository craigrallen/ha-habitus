#!/usr/bin/with-contenv bashio

LOG_LEVEL=$(bashio::config 'log_level')
SCAN_INTERVAL=$(bashio::config 'scan_interval_hours')
export HA_URL="http://supervisor/core"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"

bashio::log.info "Habitus starting — scan every ${SCAN_INTERVAL}h"

while true; do
    python3 -u /app/habitus/main.py
    bashio::log.info "Sleeping ${SCAN_INTERVAL}h until next run"
    sleep $((SCAN_INTERVAL * 3600))
done
