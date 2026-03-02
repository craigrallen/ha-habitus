#!/usr/bin/with-contenv bashio
SCAN=$(bashio::config 'scan_interval_hours')
DAYS=$(bashio::config 'days_history')
export HA_URL="http://supervisor/core"
bashio::log.info "Habitus starting — scan every ${SCAN}h, ${DAYS} days history"
while true; do
    python3 -u /app/habitus/main.py --days $DAYS
    sleep $((SCAN * 3600))
done
