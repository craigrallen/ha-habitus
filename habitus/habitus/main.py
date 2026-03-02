"""
Habitus — behavioral intelligence for Home Assistant.

Design: HA is source of truth. We store only inference artifacts.
On first run: train on full history window.
On subsequent runs: only fetch NEW data (since last_data_to), merge with existing model.
"""
import asyncio, json, os, datetime, argparse, pickle
import pandas as pd
import numpy as np
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('habitus')

DATA_DIR  = os.environ.get("DATA_DIR", "/data")
HA_URL    = os.environ.get("HA_URL",   "http://supervisor/core")
HA_WS     = os.environ.get("HA_WS",    "ws://supervisor/core/api/websocket")
HA_TOKEN  = os.environ.get("SUPERVISOR_TOKEN", "")

MODEL_PATH    = os.path.join(DATA_DIR, "model.pkl")
SCALER_PATH   = os.path.join(DATA_DIR, "scaler.pkl")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")
STATE_PATH    = os.path.join(DATA_DIR, "run_state.json")

BEHAVIORAL_KEYWORDS = [
    'energy','temperature','humidity','power','watt',
    'consumed','production','solar','battery','pump',
    'motion','door','contact','occupancy','presence'
]
SKIP = ['rssi','signal_strength','fossil_fuel','co2_intensity',
        'grid_fossil','firmware','battery_level']

FEATURE_COLS = ['hour_of_day','day_of_week','is_weekend','month',
                'total_power_w','avg_temp_c','sensor_changes']

def is_behavioral(eid):
    e = eid.lower()
    return any(k in e for k in BEHAVIORAL_KEYWORDS) and not any(k in e for k in SKIP)

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}

def save_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)

# ── WebSocket ─────────────────────────────────────────────────────────────────

async def ws_connect():
    import websockets
    ws = await websockets.connect(HA_WS, max_size=50*1024*1024)
    await ws.recv()
    await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
    result = json.loads(await ws.recv())
    if result.get('type') != 'auth_ok':
        raise RuntimeError(f"Auth failed: {result}")
    return ws

async def get_stat_ids():
    ws = await ws_connect()
    await ws.send(json.dumps({"id": 1, "type": "recorder/list_statistic_ids"}))
    result = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
    await ws.close()
    all_ids = [s['statistic_id'] for s in result.get('result', [])]
    behavioral = [e for e in all_ids if is_behavioral(e)]
    log.info(f"Found {len(behavioral)} behavioral sensors (from {len(all_ids)} total)")
    return behavioral

async def fetch_stats(entity_ids, start_iso, end_iso=None):
    """Fetch stats from start_iso to end_iso. No local raw data stored."""
    if end_iso is None:
        end_iso = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:00:00+00:00')

    all_rows = []
    done = 0
    for eid in entity_ids:
        try:
            ws = await ws_connect()
            payload = {
                "id": 1, "type": "recorder/statistics_during_period",
                "start_time": start_iso, "end_time": end_iso,
                "statistic_ids": [eid], "period": "hour",
                "types": ["mean","sum"]
            }
            await ws.send(json.dumps(payload))
            result = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            await ws.close()
            for sid, points in result.get('result', {}).items():
                for p in points:
                    ts = p['start']
                    if ts > 1e10: ts /= 1000
                    all_rows.append({'entity_id': sid, 'ts': ts,
                                     'mean': p.get('mean'), 'sum': p.get('sum')})
            done += 1
            if done % 50 == 0:
                log.info(f"  {done}/{len(entity_ids)} sensors queried")
        except Exception as e:
            log.warning(f"Error {eid}: {e}")

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    log.info(f"Fetched {len(df):,} rows | {df['ts'].min().date()} → {df['ts'].max().date()}")
    return df

# ── Features ──────────────────────────────────────────────────────────────────

def build_features(df):
    df = df.copy()
    df['hour'] = df['ts'].dt.floor('h')
    hours = pd.DataFrame({'hour': pd.date_range(df['hour'].min(), df['hour'].max(), freq='h')})
    hours['hour_of_day'] = hours['hour'].dt.hour
    hours['day_of_week']  = hours['hour'].dt.dayofweek
    hours['is_weekend']   = (hours['day_of_week'] >= 5).astype(int)
    hours['month']        = hours['hour'].dt.month

    power = df[df['entity_id'].str.contains('consumed_w|watt|power', case=False, na=False)].copy()
    power['v'] = pd.to_numeric(power['mean'], errors='coerce')
    total_power = power.groupby('hour')['v'].sum().rename('total_power_w')

    temp = df[df['entity_id'].str.contains('temperature', case=False, na=False)].copy()
    temp['v'] = pd.to_numeric(temp['mean'], errors='coerce')
    avg_temp = temp.groupby('hour')['v'].mean().rename('avg_temp_c')

    activity = df.groupby('hour').size().rename('sensor_changes')

    features = hours.set_index('hour')
    for s in [total_power, avg_temp, activity]:
        features = features.join(s, how='left')
    return features.fillna(0).reset_index()

# ── Model ─────────────────────────────────────────────────────────────────────

def train_model(features):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = features[FEATURE_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(Xs)
    days = round((features['hour'].max() - features['hour'].min()).total_seconds() / 86400)
    log.info(f"Model trained on {len(X):,} hourly snapshots ({days} days)")
    return model, scaler

def save_artifacts(model, scaler, features):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    baseline = {}
    for (h, d), g in features.groupby(['hour_of_day','day_of_week']):
        baseline[f"{h}_{d}"] = {
            'mean_power': round(float(g['total_power_w'].mean()), 1),
            'std_power':  round(float(g['total_power_w'].std()), 1),
            'mean_temp':  round(float(g['avg_temp_c'].mean()), 1),
            'n_samples':  len(g)
        }
    with open(BASELINE_PATH, 'w') as f: json.dump(baseline, f)
    log.info(f"Artifacts saved ({os.path.getsize(MODEL_PATH)//1024}KB model, {len(baseline)} baselines)")

def score_current(model, scaler, features):
    now = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    row = features[features['hour'] == pd.Timestamp(now)]
    if not row.empty:
        X = row[FEATURE_COLS].values
    else:
        X = np.array([[now.hour, now.weekday(), int(now.weekday()>=5), now.month, 0, 0, 0]])
    Xs = scaler.transform(X)
    score = model.score_samples(Xs)[0]
    return int(max(0, min(100, (-score + 0.5) * 100)))

# ── HA publish ────────────────────────────────────────────────────────────────

def publish(entity_id, state, attributes=None):
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    payload = {"state": str(state), "attributes": attributes or {}}
    try:
        r = requests.post(f"{HA_URL}/api/states/{entity_id}",
                          headers=headers, json=payload, timeout=5)
        if r.status_code not in (200, 201):
            log.error(f"Publish {entity_id}: {r.status_code}")
    except Exception as e:
        log.error(f"Publish {entity_id}: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

async def run(days_history):
    os.makedirs(DATA_DIR, exist_ok=True)
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:00:00+00:00')
    state = load_state()

    stat_ids = await get_stat_ids()
    if not stat_ids:
        log.error("No behavioral sensors found"); return

    # Determine fetch window
    if state.get('data_to') and os.path.exists(MODEL_PATH):
        # Incremental: only fetch new data since last run
        fetch_from = state['data_to']
        log.info(f"Incremental run: fetching {fetch_from} → {now_iso}")
        df_new = await fetch_stats(stat_ids, fetch_from, now_iso)
        if df_new.empty or len(df_new) < 24:
            log.info("Not enough new data — skipping retraining, scoring with existing model")
            with open(MODEL_PATH,'rb') as f: model = pickle.load(f)
            with open(SCALER_PATH,'rb') as f: scaler = pickle.load(f)
            features_for_score = build_features(df_new) if not df_new.empty else pd.DataFrame()
            anomaly_score = score_current(model, scaler, features_for_score) if not features_for_score.empty else 0
            training_days = state.get('training_days', 0)
            entity_count = state.get('entity_count', len(stat_ids))
        else:
            # Retrain on full window — use earliest known data_from (all of HA history)
            full_from = state.get('data_from', '2000-01-01T00:00:00+00:00')
            log.info(f"Retraining on full window {full_from} → {now_iso}")
            df_full = await fetch_stats(stat_ids, full_from, now_iso)
            if df_full.empty: log.warning("No data for full retrain"); return
            features = build_features(df_full)
            del df_full
            model, scaler = train_model(features)
            save_artifacts(model, scaler, features)
            anomaly_score = score_current(model, scaler, features)
            training_days = round((features['hour'].max() - features['hour'].min()).total_seconds() / 86400)
            entity_count = len(stat_ids)
    else:
        # First run: fetch ALL available history (from HA epoch, not just N days)
        full_from = '2000-01-01T00:00:00+00:00'
        log.info(f"First run: fetching all available history from {full_from}")
        df = await fetch_stats(stat_ids, full_from, now_iso)
        if df.empty: log.warning("No data returned"); return
        features = build_features(df)
        del df
        model, scaler = train_model(features)
        save_artifacts(model, scaler, features)
        anomaly_score = score_current(model, scaler, features)
        training_days = round((features['hour'].max() - features['hour'].min()).total_seconds() / 86400)
        entity_count = len(stat_ids)
        # Save discovery window
        state['data_from'] = full_from
        log.info(f"Discovery window: {full_from} → {now_iso} ({training_days} days)")

    # Always update state with latest run info
    state.update({
        'last_run': now_iso,
        'data_to': now_iso,
        'training_days': training_days,
        'entity_count': entity_count,
        'anomaly_score': anomaly_score,
    })
    save_state(state)

    is_anomalous = anomaly_score > 70
    log.info(f"Score: {anomaly_score}/100 ({'⚠ ANOMALY' if is_anomalous else '✓ normal'})")

    publish("sensor.habitus_anomaly_score", anomaly_score, {
        "friendly_name": "Habitus Anomaly Score", "unit_of_measurement": "",
        "icon": "mdi:brain", "state_class": "measurement",
    })
    publish("binary_sensor.habitus_anomaly_detected", "on" if is_anomalous else "off", {
        "friendly_name": "Habitus Anomaly", "device_class": "problem",
    })
    publish("sensor.habitus_training_days", training_days, {
        "friendly_name": "Habitus Training Days",
        "unit_of_measurement": "days", "icon": "mdi:calendar-range",
    })
    publish("sensor.habitus_entity_count", entity_count, {
        "friendly_name": "Habitus Tracked Sensors",
        "unit_of_measurement": "sensors", "icon": "mdi:chip",
    })
    log.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=365)
    args = parser.parse_args()
    asyncio.run(run(days_history=args.days))
