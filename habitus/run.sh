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
export INGRESS_PORT="${PORT}"
export HABITUS_NOTIFY_SERVICE="${NOTIFY}"
export HABITUS_NOTIFY_ON="${NOTIFY_ON}"
export HABITUS_ANOMALY_THRESHOLD="${THRESHOLD}"
export HABITUS_SCHEDULE="${SCHEDULE}"
export HABITUS_TRAIN_TIME="${TRAIN_TIME}"

RESCAN_FLAG="/data/.rescan_requested"
STATE_FILE="/data/run_state.json"

bashio::log.info "Habitus v2.7.0 | Schedule: ${SCHEDULE} | Train: ${TRAIN_TIME} | Scan: ${SCAN}h"

# Start web server inline — PYTHONPATH=/app so 'import habitus.web' resolves
cd /app && python3 -u -c "
import sys, os, traceback
sys.path.insert(0, '/app')
try:
    from habitus.web import start_web
    port = int(os.environ.get('INGRESS_PORT', '8099'))
    print(f'[web] Starting on :{port}', flush=True)
    start_web(port)
except Exception as e:
    print(f'[web] FAILED: {e}', flush=True)
    traceback.print_exc(file=sys.stdout)
" 2>&1 &
WEB_PID=$!
bashio::log.info "Web server PID: ${WEB_PID}"

bashio::log.info "Waiting 30s for HA..."
sleep 30

is_train_time() {
    local th tm nh nm diff
    th=$(echo "${TRAIN_TIME}" | cut -d: -f1 | sed 's/^0*//' | grep . || echo 0)
    tm=$(echo "${TRAIN_TIME}" | cut -d: -f2 | sed 's/^0*//' | grep . || echo 0)
    nh=$(date +%-H); nm=$(date +%-M)
    diff=$(( (nh * 60 + nm) - (th * 60 + tm) ))
    [ "$diff" -lt 0 ] && diff=$(( -diff ))
    [ "$diff" -lt 16 ]
}

FIRST_RUN=true
cd /app

while true; do
    if ! kill -0 $WEB_PID 2>/dev/null; then
        bashio::log.warning "Web server died — restarting"
        python3 -u -c "
import sys, os; sys.path.insert(0, '/app')
from habitus.web import start_web
start_web(int(os.environ.get('INGRESS_PORT','8099')))
" 2>&1 &
        WEB_PID=$!
    fi

    if [ -f "$RESCAN_FLAG" ]; then
        bashio::log.info "Full rescan — wiping state"
        rm -f "$RESCAN_FLAG" "$STATE_FILE" /data/model*.pkl /data/scaler*.pkl
        FIRST_RUN=true
    fi

    if [ "$FIRST_RUN" = "true" ]; then
        FIRST_RUN=false
        bashio::log.info "Full training run"
        python3 -u -m habitus.main --days "$DAYS" --mode full \
            || bashio::log.warning "Full training failed"
    elif [ "$SCHEDULE" = "overnight" ]; then
        if is_train_time; then
            bashio::log.info "Overnight training window"
            python3 -u -m habitus.main --days "$DAYS" --mode full \
                || bashio::log.warning "Overnight training failed"
        else
            bashio::log.info "Score-only (daytime)"
            python3 -u -m habitus.main --days "$DAYS" --mode score \
                || bashio::log.warning "Score run failed"
        fi
    else
        python3 -u -m habitus.main --days "$DAYS" --mode full \
            || bashio::log.warning "Continuous run failed"
    fi

    bashio::log.info "Next check in ${SCAN}h"
    for i in $(seq 1 $(( SCAN * 12 ))); do
        sleep 300
        [ -f "$RESCAN_FLAG" ] && break
    done
done
