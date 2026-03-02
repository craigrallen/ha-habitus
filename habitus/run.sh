#!/usr/bin/with-contenv bashio

SCAN=$(bashio::config 'scan_interval_hours')
DAYS=$(bashio::config 'days_history')
PORT=$(bashio::addon.ingress_port)
NOTIFY=$(bashio::config 'notify_service')
NOTIFY_ON=$(bashio::config 'notify_on_anomaly')
THRESHOLD=$(bashio::config 'anomaly_threshold')

export HA_URL="http://supervisor/core"
export HA_WS="ws://supervisor/core/api/websocket"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export DATA_DIR="/data"
export HABITUS_NOTIFY_SERVICE="${NOTIFY}"
export HABITUS_NOTIFY_ON="${NOTIFY_ON}"
export HABITUS_ANOMALY_THRESHOLD="${THRESHOLD}"
RESCAN_FLAG="/data/.rescan_requested"

bashio::log.info "Habitus v2.0.0 — web UI on :${PORT}, scan every ${SCAN}h, notify: ${NOTIFY}"

python3 -c "
import os
os.environ.update({'DATA_DIR':'/data','HA_URL':'http://supervisor/core','SUPERVISOR_TOKEN':'${SUPERVISOR_TOKEN}'})
from habitus.web import start_web
start_web(int('${PORT}'))
" &

bashio::log.info "Waiting 30s for HA..."
sleep 30

while true; do
    if [ -f "$RESCAN_FLAG" ]; then
        bashio::log.info "Rescan requested"
        rm -f "$RESCAN_FLAG"
    fi
    python3 -u /app/habitus/main.py --days $DAYS || bashio::log.warning "Run failed"
    bashio::log.info "Next run in ${SCAN}h"
    for i in $(seq 1 $((SCAN * 12))); do
        sleep 300
        if [ -f "$RESCAN_FLAG" ]; then
            bashio::log.info "Rescan flag — running early"
            rm -f "$RESCAN_FLAG"; break
        fi
    done
done
