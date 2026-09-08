#!/usr/bin/with-contenv bashio

SCAN=$(bashio::config 'scan_interval_hours')
DAYS=$(bashio::config 'days_history')
# Respect configured history depth up to 10 years (UI supports 3650)
DAYS=$(( DAYS > 3650 ? 3650 : DAYS ))
DAYS=$(( DAYS < 7 ? 7 : DAYS ))
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
export HABITUS_DAYS="${DAYS}"

RESCAN_FLAG="/data/.rescan_requested"
STATE_FILE="/data/run_state.json"
export HABITUS_VERSION=$(bashio::addon.version 2>/dev/null || echo "unknown")

# ── Cache invalidation strategy ─────────────────────────────────────────────
# Testing mode (/data/.testing_mode exists): clear all derived caches + auto-retrain
#   on every start. Use while actively debugging.
# Normal mode (default): only clear caches when the version stamp changes,
#   and never auto-retrain (scheduled training handles it).
CURRENT_VERSION="${HABITUS_VERSION}"
STAMP_FILE="/data/.cache_version"
CACHED_VERSION=""
[ -f "$STAMP_FILE" ] && CACHED_VERSION=$(cat "$STAMP_FILE" 2>/dev/null)

if [ -f "/data/.testing_mode" ]; then
  bashio::log.info "Testing mode ON — clearing all derived caches and scheduling retrain"
  _DO_CLEAR=true
  _DO_RETRAIN=true
elif [ "$CURRENT_VERSION" != "$CACHED_VERSION" ]; then
  bashio::log.info "Version changed (${CACHED_VERSION} → ${CURRENT_VERSION}) — clearing derived caches"
  _DO_CLEAR=true
  _DO_RETRAIN=false
else
  bashio::log.info "Cache valid (v${CURRENT_VERSION}) — skipping cache clear"
  _DO_CLEAR=false
  _DO_RETRAIN=false
fi

if [ "$_DO_CLEAR" = "true" ]; then
  for _f in \
    device_library.json suggestions.json smart_suggestions.json \
    scene_analysis.json conflict_report.json automation_health.json \
    routine_schedule.json guest_mode.json seasonal_suggestions.json \
    cost_report.json integration_health.json entity_anomalies.json \
    patterns.json ha_automations.json changelog.json dashboard.json; do
    [ -f "/data/${_f}" ] && rm -f "/data/${_f}" && bashio::log.info "Cleared: ${_f}"
  done
  echo "$CURRENT_VERSION" > "$STAMP_FILE"
fi

if [ "$_DO_RETRAIN" = "true" ]; then
  touch /data/.retrain_on_start
fi

# HABITUS_VERSION already exported above
export HABITUS_MAX_POWER_KW=$(bashio::config "max_power_kw" 2>/dev/null || echo "25")
export HABITUS_POWER_ENTITY=$(bashio::config "power_entity" 2>/dev/null || echo "")
export HABITUS_POWER_ENTITY_PHASE1=$(bashio::config "power_entity_phase1" 2>/dev/null || echo "")
export HABITUS_POWER_ENTITY_PHASE2=$(bashio::config "power_entity_phase2" 2>/dev/null || echo "")
export HABITUS_POWER_ENTITY_PHASE3=$(bashio::config "power_entity_phase3" 2>/dev/null || echo "")
export HABITUS_KWH_PRICE=$(bashio::config "kwh_price" 2>/dev/null || echo "0.30")
export HABITUS_CURRENCY=$(bashio::config "currency" 2>/dev/null || echo "kr")
export HABITUS_FETCH_ROW_BUDGET=$(bashio::config "fetch_row_budget" 2>/dev/null || echo "1000000")
export HABITUS_FETCH_MIN_WINDOW_DAYS=$(bashio::config "fetch_min_window_days" 2>/dev/null || echo "7")
bashio::log.info "Habitus v${HABITUS_VERSION} | Schedule: ${SCHEDULE} | Train: ${TRAIN_TIME} | Scan: ${SCAN}h | Days: ${DAYS} | RowBudget: ${HABITUS_FETCH_ROW_BUDGET}"

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

    if [ -f /data/progress.json ] && grep -q '"running"[[:space:]]*:[[:space:]]*true' /data/progress.json 2>/dev/null; then
        # Staleness check: if file is >10 minutes old with running=true, the previous
        # run crashed and left a stale lock. Treat as stale and proceed.
        if find /data/progress.json -mmin +10 2>/dev/null | grep -q .; then
            bashio::log.warning "Stale progress.json detected (>10 min old, running=true) — previous run likely crashed. Clearing stale lock and proceeding."
            # Overwrite with a stale_aborted marker so the UI shows what happened
            echo '{"running":false,"phase":"stale_aborted","stale_cleared_at":"'"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"'"}' > /data/progress.json
        else
            bashio::log.info "Training already running — skip scheduler tick"
            sleep 30
            continue
        fi
    fi

    if [ -f "$RESCAN_FLAG" ]; then
        bashio::log.info "Full rescan — wiping state"
        rm -f "$RESCAN_FLAG" "$STATE_FILE" /data/model*.pkl /data/scaler*.pkl
        FIRST_RUN=true
    fi

    # Auto-retrain after startup cache clear (set by cache wipe block above)
    if [ -f "/data/.retrain_on_start" ]; then
        bashio::log.info "Post-update retrain triggered (cache was cleared on start)"
        rm -f "/data/.retrain_on_start"
        FIRST_RUN=true
    fi

    if [ "$FIRST_RUN" = "true" ]; then
        FIRST_RUN=false
        bashio::log.info "Full training run (${DAYS} days)"
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
