"""
Habitus — behavioral intelligence for Home Assistant.
Reads long-term statistics directly via WebSocket (4+ years of hourly data),
builds a behavioral model, scores current hour, publishes sensors back to HA.
"""
import asyncio, json, os, sys, datetime, argparse
import pandas as pd
import numpy as np
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('habitus')

DATA_DIR = os.environ.get("DATA_DIR", "/data")
HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
HA_WS = "ws://supervisor/core/api/websocket"

BEHAVIORAL_SENSORS = [
    'energy', 'temperature', 'humidity', 'power', 'watt',
    'motion', 'door', 'light', 'solar', 'fridge', 'pump',
    'battery_level', 'consumed', 'production'
]

SKIP = ['rssi', 'signal', 'fossil_fuel', 'co2_intensity', 'grid_']

def is_behavioral(eid):
    eid_l = eid.lower()
    return (any(k in eid_l for k in BEHAVIORAL_SENSORS) and
            not any(k in eid_l for k in SKIP))

async def ws_auth(ws):
    await ws.recv()
    await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
    result = json.loads(await ws.recv())
    if result.get('type') != 'auth_ok':
        raise RuntimeError(f"Auth failed: {result}")

async def get_stat_ids():
    import websockets
    async with websockets.connect(HA_WS, max_size=10*1024*1024) as ws:
        await ws_auth(ws)
        await ws.send(json.dumps({"id": 1, "type": "recorder/list_statistic_ids"}))
        result = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        ids = [s['statistic_id'] for s in result.get('result', [])]
        return [e for e in ids if is_behavioral(e)]

async def fetch_stats(entity_ids, start="2021-01-01T00:00:00+00:00"):
    """Fetch hourly long-term stats, one entity at a time."""
    import websockets
    all_rows = []
    done = 0
    for eid in entity_ids:
        try:
            async with websockets.connect(HA_WS, max_size=50*1024*1024) as ws:
                await ws_auth(ws)
                await ws.send(json.dumps({
                    "id": 1,
                    "type": "recorder/statistics_during_period",
                    "start_time": start,
                    "statistic_ids": [eid],
                    "period": "hour",
                    "types": ["mean", "sum"]
                }))
                result = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                data = result.get('result', {})
                for sid, points in data.items():
                    for p in points:
                        ts = p['start']
                        if ts > 1e10: ts /= 1000
                        all_rows.append({'entity_id': sid, 'ts': ts,
                                         'mean': p.get('mean'), 'sum': p.get('sum')})
            done += 1
            if done % 50 == 0:
                log.info(f"Fetched {done}/{len(entity_ids)} sensors, {len(all_rows):,} rows")
        except Exception as e:
            log.warning(f"Error on {eid}: {e}")
    return all_rows

def build_features(df):
    """Build hourly behavioral feature matrix."""
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['hour'] = df['ts'].dt.floor('h')

    hours = pd.DataFrame({'hour': pd.date_range(df['hour'].min(), df['hour'].max(), freq='h')})
    hours['hour_of_day'] = hours['hour'].dt.hour
    hours['day_of_week'] = hours['hour'].dt.dayofweek
    hours['is_weekend'] = (hours['day_of_week'] >= 5).astype(int)
    hours['month'] = hours['hour'].dt.month

    # Power signals
    power = df[df['entity_id'].str.contains('consumed_w|watt|power', case=False, na=False)].copy()
    power['value'] = pd.to_numeric(power['mean'], errors='coerce')
    total_power = power.groupby('hour')['value'].sum().rename('total_power_w')
    
    # Temperature
    temp = df[df['entity_id'].str.contains('temperature', case=False, na=False)].copy()
    temp['value'] = pd.to_numeric(temp['mean'], errors='coerce')
    avg_temp = temp.groupby('hour')['value'].mean().rename('avg_temp_c')

    # Activity count (any sensor changes)
    activity = df.groupby('hour').size().rename('sensor_changes')

    features = hours.set_index('hour')
    for s in [total_power, avg_temp, activity]:
        features = features.join(s, how='left')
    return features.fillna(0).reset_index()

def train_model(features):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    cols = ['hour_of_day', 'day_of_week', 'is_weekend', 'month',
            'total_power_w', 'avg_temp_c', 'sensor_changes']
    X = features[cols].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_s)
    log.info(f"Model trained on {len(X):,} hourly snapshots ({features['hour'].min().date()} → {features['hour'].max().date()})")
    return model, scaler, cols

def publish(entity_id, state, attributes=None):
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    payload = {"state": str(state), "attributes": attributes or {}}
    try:
        requests.post(f"{HA_URL}/api/states/{entity_id}", headers=headers, json=payload, timeout=5)
    except Exception as e:
        log.error(f"Publish failed for {entity_id}: {e}")

def score_current(model, scaler, cols, features):
    now = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    this_hour = features[features['hour'] == pd.Timestamp(now)]
    if not this_hour.empty:
        X = this_hour[cols].values
    else:
        X = np.array([[now.hour, now.weekday(), int(now.weekday()>=5),
                       now.month, 0, 0, 0]])
    X_s = scaler.transform(X)
    score = model.score_samples(X_s)[0]
    return int(max(0, min(100, (-score + 0.5) * 100)))

async def run(days_history=1825):
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "longterm_stats.parquet")
    
    log.info("Fetching statistic IDs from HA...")
    stat_ids = await get_stat_ids()
    log.info(f"Found {len(stat_ids)} behavioral sensors")

    start_year = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_history)).strftime('%Y-%m-%dT00:00:00+00:00')
    
    log.info(f"Pulling long-term statistics from {start_year[:10]}...")
    rows = await fetch_stats(stat_ids, start=start_year)

    if not rows:
        log.warning("No data fetched")
        return

    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    log.info(f"Cached {len(df):,} data points for {df['entity_id'].nunique()} entities")

    features = build_features(df)
    log.info(f"Feature matrix: {len(features):,} hourly snapshots")

    if len(features) < 72:
        log.warning("Less than 3 days of data — skipping model training")
        return

    model, scaler, cols = train_model(features)
    anomaly_score = score_current(model, scaler, cols, features)
    is_anomalous = anomaly_score > 70
    training_days = (df['ts'].max() - df['ts'].min()) / 86400 if len(df) else 0

    log.info(f"Anomaly score: {anomaly_score}/100 ({'ANOMALOUS' if is_anomalous else 'normal'})")

    publish("sensor.habitus_anomaly_score", anomaly_score, {
        "friendly_name": "Habitus Anomaly Score", "unit_of_measurement": "",
        "icon": "mdi:brain", "state_class": "measurement",
    })
    publish("binary_sensor.habitus_anomaly_detected", "on" if is_anomalous else "off", {
        "friendly_name": "Habitus Anomaly", "device_class": "problem",
    })
    publish("sensor.habitus_training_days", round(float(training_days)), {
        "friendly_name": "Habitus Training Days", "unit_of_measurement": "days",
        "icon": "mdi:chart-timeline",
    })
    publish("sensor.habitus_entity_count", df['entity_id'].nunique(), {
        "friendly_name": "Habitus Tracked Entities", "unit_of_measurement": "entities",
    })
    log.info("Published Habitus sensors to HA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=1825)
    args = parser.parse_args()
    asyncio.run(run(days_history=args.days))
