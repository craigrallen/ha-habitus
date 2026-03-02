#!/usr/bin/with-contenv bashio

SCAN=$(bashio::config 'scan_interval_hours')
DAYS=$(bashio::config 'days_history')
PORT=$(bashio::addon.ingress_port)
NOTIFY=$(bashio::config 'notify_service')
NOTIFY_ON=$(bashio::config 'notify_on_anomaly')
THRESHOLD=$(bashio::config 'anomaly_threshold')
SCHEDULE=$(bashio::config 'training_schedule')
TRAIN_TIME=$(bashio::config 'overnight_train_time')

export HA_URL="http://supervisor/core"
export HA_WS="ws://supervisor/core/api/websocket"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export DATA_DIR="/data"
export PYTHONPATH="/app"
export HABITUS_NOTIFY_SERVICE="${NOTIFY}"
export HABITUS_NOTIFY_ON="${NOTIFY_ON}"
export HABITUS_ANOMALY_THRESHOLD="${THRESHOLD}"
export HABITUS_SCHEDULE="${SCHEDULE}"
export HABITUS_TRAIN_TIME="${TRAIN_TIME}"

RESCAN_FLAG="/data/.rescan_requested"

bashio::log.info "Habitus v2.3.0 starting"
bashio::log.info "Schedule: ${SCHEDULE} | Train time: ${TRAIN_TIME} | Scan: ${SCAN}h"

# Start web server in background
cd /app && python3 -c "
import os, sys
sys.path.insert(0, '/app')
from habitus.habitus.web import start_web
start_web(int('${PORT}'))
" &

bashio::log.info "Waiting 30s for HA..."
sleep 30

is_train_time() {
    local target_h target_m now_h now_m diff
    target_h=$(echo "${TRAIN_TIME}" | cut -d: -f1 | sed 's/^0//')
    target_m=$(echo "${TRAIN_TIME}" | cut -d: -f2 | sed 's/^0//')
    now_h=$(date +%-H)
    now_m=$(date +%-M)
    diff=$(( (now_h * 60 + now_m) - (target_h * 60 + target_m) ))
    [ "$diff" -lt 0 ] && diff=$(( -diff ))
    [ "$diff" -lt 16 ]
}

FIRST_RUN=true

while true; do
    cd /app

    if [ -f "$RESCAN_FLAG" ]; then
        bashio::log.info "Full rescan requested"
        rm -f "$RESCAN_FLAG"
        python3 -u habitus/habitus/main.py --days "$DAYS" --mode full \
            || bashio::log.warning "Full rescan failed"

    elif [ "$FIRST_RUN" = "true" ]; then
        bashio::log.info "First run — full training"
        FIRST_RUN=false
        python3 -u habitus/habitus/main.py --days "$DAYS" --mode full \
            || bashio::log.warning "First run failed"

    elif [ "$SCHEDULE" = "overnight" ]; then
        if is_train_time; then
            bashio::log.info "Overnight training window"
            python3 -u habitus/habitus/main.py --days "$DAYS" --mode full \
                || bashio::log.warning "Overnight training failed"
        else
            bashio::log.info "Score-only (outside training window)"
            python3 -u habitus/habitus/main.py --days "$DAYS" --mode score \
                || bashio::log.warning "Score run failed"
        fi
    else
        python3 -u habitus/habitus/main.py --days "$DAYS" --mode full \
            || bashio::log.warning "Continuous run failed"
    fi

    bashio::log.info "Next check in ${SCAN}h"
    for i in $(seq 1 $(( SCAN * 12 ))); do
        sleep 300
        if [ -f "$RESCAN_FLAG" ]; then break; fi
    done
done
