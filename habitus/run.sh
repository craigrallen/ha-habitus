#!/usr/bin/with-contenv bashio

# ── Config ────────────────────────────────────────────────────────────────────
SCAN=$(bashio::config 'scan_interval_hours')
DAYS=$(bashio::config 'days_history')
PORT=$(bashio::addon.ingress_port)
NOTIFY=$(bashio::config 'notify_service')
NOTIFY_ON=$(bashio::config 'notify_on_anomaly')
THRESHOLD=$(bashio::config 'anomaly_threshold')
SCHEDULE=$(bashio::config 'training_schedule')
TRAIN_TIME=$(bashio::config 'overnight_train_time')

# ── Environment ───────────────────────────────────────────────────────────────
export HA_URL="http://supervisor/core"
export HA_WS="ws://supervisor/core/api/websocket"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export DATA_DIR="/data"
export HABITUS_NOTIFY_SERVICE="${NOTIFY}"
export HABITUS_NOTIFY_ON="${NOTIFY_ON}"
export HABITUS_ANOMALY_THRESHOLD="${THRESHOLD}"
export HABITUS_SCHEDULE="${SCHEDULE}"
export HABITUS_TRAIN_TIME="${TRAIN_TIME}"

RESCAN_FLAG="/data/.rescan_requested"
SCORE_FLAG="/data/.score_only"

bashio::log.info "Habitus v0.14.0 starting"
bashio::log.info "Schedule: ${SCHEDULE} | Train time: ${TRAIN_TIME} | Scan: ${SCAN}h"

# ── Web server (background) ───────────────────────────────────────────────────
python3 -c "
import os
os.environ.update({
    'DATA_DIR': '/data',
    'HA_URL': 'http://supervisor/core',
    'SUPERVISOR_TOKEN': '${SUPERVISOR_TOKEN}'
})
from habitus.web import start_web
start_web(int('${PORT}'))
" &

bashio::log.info "Waiting 30s for HA to be ready..."
sleep 30

# ── Schedule helper ───────────────────────────────────────────────────────────
# Returns 0 if now is within 15 min of the overnight train time
is_train_time() {
    local target_h target_m now_h now_m now_mins target_mins diff
    target_h=$(echo "${TRAIN_TIME}" | cut -d: -f1 | sed 's/^0//')
    target_m=$(echo "${TRAIN_TIME}" | cut -d: -f2 | sed 's/^0//')
    now_h=$(date +%-H)
    now_m=$(date +%-M)
    now_mins=$(( now_h * 60 + now_m ))
    target_mins=$(( target_h * 60 + target_m ))
    diff=$(( now_mins - target_mins ))
    [ "$diff" -lt 0 ] && diff=$(( -diff ))
    [ "$diff" -lt 16 ]
}

# ── Main loop ─────────────────────────────────────────────────────────────────
# On first run always do a full train regardless of schedule
FIRST_RUN=true

while true; do
    if [ -f "$RESCAN_FLAG" ]; then
        bashio::log.info "Full rescan requested"
        rm -f "$RESCAN_FLAG" "$SCORE_FLAG"
        python3 -u /app/habitus/main.py --days "$DAYS" --mode full \
            || bashio::log.warning "Full rescan failed"

    elif [ "$FIRST_RUN" = "true" ]; then
        bashio::log.info "First run — full training"
        FIRST_RUN=false
        python3 -u /app/habitus/main.py --days "$DAYS" --mode full \
            || bashio::log.warning "First run failed"

    elif [ "$SCHEDULE" = "overnight" ]; then
        if is_train_time; then
            bashio::log.info "Overnight training window — running full train"
            python3 -u /app/habitus/main.py --days "$DAYS" --mode full \
                || bashio::log.warning "Overnight training failed"
        else
            bashio::log.info "Outside training window — score only"
            python3 -u /app/habitus/main.py --days "$DAYS" --mode score \
                || bashio::log.warning "Score run failed"
        fi

    else
        # continuous — full retrain every scan cycle
        python3 -u /app/habitus/main.py --days "$DAYS" --mode full \
            || bashio::log.warning "Continuous run failed"
    fi

    bashio::log.info "Next check in ${SCAN}h"

    # Sleep in 5-min chunks so we can catch rescan flags promptly
    for i in $(seq 1 $(( SCAN * 12 ))); do
        sleep 300
        if [ -f "$RESCAN_FLAG" ]; then
            bashio::log.info "Rescan flag detected"
            break
        fi
    done
done
