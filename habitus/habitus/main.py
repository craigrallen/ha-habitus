"""
Habitus — behavioral intelligence for Home Assistant.

Design principle: HA is the source of truth. We never duplicate raw data.
/data contains only inference artifacts: trained model, scaler, patterns, suggestions.

Each run:
  1. Query HA long-term statistics (live, no local copy)
  2. Train/update behavioral model
  3. Score current hour
  4. Save model artifacts only
  5. Publish sensors back to HA
"""
import asyncio, json, os, sys, datetime, argparse, pickle
import pandas as pd
import numpy as np
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('habitus')

DATA_DIR = os.environ.get("DATA_DIR", "/data")
HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_WS  = os.environ.get("HA_WS",  "ws://supervisor/core/api/websocket")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")

# Artifacts (small — model params only, no raw data)
MODEL_PATH    = os.path.join(DATA_DIR, "model.pkl")
SCALER_PATH   = os.path.join(DATA_DIR, "scaler.pkl")
PATTERNS_PATH = os.path.join(DATA_DIR, "patterns.json")
STATE_PATH    = os.path.join(DATA_DIR, "run_state.json")

BEHAVIORAL_KEYWORDS = [
    'energy', 'temperature', 'humidity', 'power', 'watt',
    'consumed', 'production', 'solar', 'battery', 'pump',
    'motion', 'door', 'contact', 'occupancy', 'presence'
]
SKIP = ['rssi', 'signal_strength', 'fossil_fuel', 'co2_intensity',
        'grid_fossil', 'firmware', 'battery_level']

def is_behavioral(eid):
    eid_l = eid.lower()
    return (any(k in eid_l for k in BEHAVIORAL_KEYWORDS) and
            not any(k in eid_l for k in SKIP))

# ── WebSocket helpers ──────────────────────────────────────────────────────────

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

async def fetch_stats(entity_ids, days_back):
    """Query HA long-term stats. Returns DataFrame — not stored to disk."""
    start = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=days_back)).strftime('%Y-%m-%dT00:00:00+00:00')
    
    all_rows = []
    done = 0
    for eid in entity_ids:
        try:
            ws = await ws_connect()
            await ws.send(json.dumps({
                "id": 1,
                "type": "recorder/statistics_during_period",
                "start_time": start,
                "statistic_ids": [eid],
                "period": "hour",
                "types": ["mean", "sum"]
            }))
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
                log.info(f"Queried {done}/{len(entity_ids)} sensors")
        except Exception as e:
            log.warning(f"Error on {eid}: {e}")

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    log.info(f"Loaded {len(df):,} hourly points across {df['entity_id'].nunique()} entities "
             f"({df['ts'].min().date()} → {df['ts'].max().date()})")
    return df

# ── Feature engineering ────────────────────────────────────────────────────────

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

FEATURE_COLS = ['hour_of_day', 'day_of_week', 'is_weekend', 'month',
                'total_power_w', 'avg_temp_c', 'sensor_changes']

def train_model(features):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = features[FEATURE_COLS].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_s)
    n_days = round((features['hour'].max() - features['hour'].min()).total_seconds() / 86400)
    log.info(f"Model trained on {len(X):,} hourly snapshots ({n_days} days)")
    return model, scaler

def save_artifacts(model, scaler, features, entity_count, days):
    """Save only inference artifacts — no raw data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)

    # Compute hourly baselines (expected ranges per hour/day combination)
    baseline = {}
    for (h, d), g in features.groupby(['hour_of_day', 'day_of_week']):
        baseline[f"{h}_{d}"] = {
            'mean_power': round(float(g['total_power_w'].mean()), 1),
            'std_power':  round(float(g['total_power_w'].std()), 1),
            'mean_temp':  round(float(g['avg_temp_c'].mean()), 1),
            'n_samples':  len(g)
        }
    with open(os.path.join(DATA_DIR, 'baseline.json'), 'w') as f:
        json.dump(baseline, f)

    state = {
        'last_run': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'training_days': days,
        'entity_count': entity_count,
        'feature_rows': len(features),
        'model_size_kb': round(os.path.getsize(MODEL_PATH) / 1024, 1)
    }
    with open(STATE_PATH, 'w') as f: json.dump(state, f, indent=2)
    log.info(f"Artifacts saved ({state['model_size_kb']}KB model, {len(baseline)} baselines)")

def score_current(model, scaler, features):
    now = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    this_hour = features[features['hour'] == pd.Timestamp(now)]
    if not this_hour.empty:
        X = this_hour[FEATURE_COLS].values
    else:
        X = np.array([[now.hour, now.weekday(), int(now.weekday()>=5),
                       now.month, 0, 0, 0]])
    X_s = scaler.transform(X)
    score = model.score_samples(X_s)[0]
    return int(max(0, min(100, (-score + 0.5) * 100)))

# ── HA publishing ──────────────────────────────────────────────────────────────

def publish(entity_id, state, attributes=None):
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    payload = {"state": str(state), "attributes": attributes or {}}
    try:
        r = requests.post(f"{HA_URL}/api/states/{entity_id}",
                          headers=headers, json=payload, timeout=5)
        if r.status_code not in (200, 201):
            log.error(f"Publish {entity_id}: {r.status_code}")
    except Exception as e:
        log.error(f"Publish failed {entity_id}: {e}")

# ── Main ───────────────────────────────────────────────────────────────────────

async def run(days_history):
    os.makedirs(DATA_DIR, exist_ok=True)

    stat_ids = await get_stat_ids()
    if not stat_ids:
        log.error("No behavioral sensors found"); return

    # Query HA — live data, not stored
    df = await fetch_stats(stat_ids, days_history)
    if df.empty:
        log.warning("No data returned"); return

    actual_days = round((df['ts'].max() - df['ts'].min()).total_seconds() / 86400)
    features = build_features(df)
    del df  # free memory — we only need features from here

    if len(features) < 72:
        log.warning(f"Only {len(features)}h of data — need at least 72h"); return

    model, scaler = train_model(features)
    anomaly_score = score_current(model, scaler, features)
    save_artifacts(model, scaler, features, len(stat_ids), actual_days)

    is_anomalous = anomaly_score > 70
    log.info(f"Anomaly score: {anomaly_score}/100 ({'⚠ ANOMALOUS' if is_anomalous else '✓ normal'})")

    publish("sensor.habitus_anomaly_score", anomaly_score, {
        "friendly_name": "Habitus Anomaly Score",
        "unit_of_measurement": "", "icon": "mdi:brain",
        "state_class": "measurement",
    })
    publish("binary_sensor.habitus_anomaly_detected", "on" if is_anomalous else "off", {
        "friendly_name": "Habitus Anomaly", "device_class": "problem",
    })
    publish("sensor.habitus_training_days", actual_days, {
        "friendly_name": "Habitus Training Days",
        "unit_of_measurement": "days", "icon": "mdi:calendar-range",
    })
    publish("sensor.habitus_entity_count", len(stat_ids), {
        "friendly_name": "Habitus Tracked Sensors",
        "unit_of_measurement": "sensors", "icon": "mdi:chip",
    })
    log.info("Done — published 4 sensors to HA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=365)
    args = parser.parse_args()
    asyncio.run(run(days_history=args.days))
