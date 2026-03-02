"""Habitus — main entry point for HA add-on."""
import os, sqlite3, json, requests, logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('habitus')

# Direct SQLite access — no API limits, full history
DB_PATH = "/config/home-assistant_v2.db"
CACHE_DIR = Path("/data")
CACHE_DIR.mkdir(exist_ok=True)

HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")

USEFUL_DOMAINS = {'binary_sensor','sensor','switch','light','media_player','person','device_tracker'}
SKIP = ['rssi','signal_strength','uptime','firmware','bytes','packets','linkquality','lqi','version','_ip_']

def should_track(entity_id):
    return (entity_id.split('.')[0] in USEFUL_DOMAINS and
            not any(k.lower() in entity_id.lower() for k in SKIP))

def read_history(days_back=30):
    """Read directly from recorder SQLite — full history, no retention limits."""
    log.info(f"Reading recorder DB: {DB_PATH}")
    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp()
    
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    
    # HA recorder schema: states table joins states_meta for entity_id
    query = """
        SELECT 
            m.entity_id,
            s.state,
            s.last_updated_ts as ts
        FROM states s
        JOIN states_meta m ON s.metadata_id = m.metadata_id
        WHERE s.last_updated_ts > ?
        ORDER BY s.last_updated_ts
    """
    
    import pandas as pd
    try:
        df = pd.read_sql_query(query, conn, params=(since,))
        conn.close()
    except Exception as e:
        # Older HA schema fallback (no states_meta)
        log.warning(f"New schema failed ({e}), trying legacy schema")
        query_legacy = """
            SELECT entity_id, state, last_updated as ts
            FROM states
            WHERE last_updated > ?
            ORDER BY last_updated
        """
        df = pd.read_sql_query(query_legacy, conn, 
            params=(datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat())
        conn.close()
    
    df = df[df['entity_id'].apply(should_track)]
    log.info(f"Loaded {len(df):,} state changes from {df['entity_id'].nunique()} entities")
    log.info(f"Date range: {days_back} days back from now")
    return df

def build_features(df):
    """Build hourly feature vectors from raw state history."""
    import pandas as pd
    df = df.copy()
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['hour'] = df['ts'].dt.floor('h')
    
    hours = pd.DataFrame({'hour': pd.date_range(df['hour'].min(), df['hour'].max(), freq='h')})
    hours['hour_of_day'] = hours['hour'].dt.hour
    hours['day_of_week'] = hours['hour'].dt.dayofweek
    hours['is_weekend'] = (hours['day_of_week'] >= 5).astype(int)
    
    # Motion, door, power activity per hour
    motion = df[df['entity_id'].str.contains('motion|presence|occupancy', case=False, na=False)]
    motion_counts = motion[motion['state'].isin(['on','detected'])].groupby('hour').size().rename('motion_events')
    
    doors = df[df['entity_id'].str.contains('door|window|contact', case=False, na=False)]
    door_counts = doors.groupby('hour').size().rename('door_events')
    
    lights = df[df['entity_id'].str.startswith('light.')]
    light_counts = lights[lights['state']=='on'].groupby('hour').size().rename('lights_on')
    
    total = df.groupby('hour').size().rename('total_events')
    
    features = hours.set_index('hour')
    for s in [motion_counts, door_counts, light_counts, total]:
        features = features.join(s, how='left')
    return features.fillna(0).reset_index()

def train_anomaly_model(features):
    """Train Isolation Forest on hourly features."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    cols = ['hour_of_day','day_of_week','is_weekend','motion_events','door_events','lights_on','total_events']
    X = features[cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    
    log.info("Anomaly model trained")
    return model, scaler, cols

def publish_to_ha(entity_id, state, attributes=None):
    """Push a sensor state back to HA."""
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    payload = {"state": str(state), "attributes": attributes or {}}
    url = f"{HA_URL}/api/states/{entity_id}"
    try:
        requests.post(url, headers=headers, json=payload, timeout=5)
    except Exception as e:
        log.error(f"Failed to publish {entity_id}: {e}")

def run():
    import pandas as pd, numpy as np
    
    # Read full history directly from DB
    df = read_history(days_back=365)  # get everything in DB
    if df.empty:
        log.warning("No data — will retry next cycle")
        return
    
    # Save cache
    df.to_parquet(CACHE_DIR / "history.parquet", index=False)
    
    # Build features
    features = build_features(df)
    features.to_parquet(CACHE_DIR / "features.parquet", index=False)
    log.info(f"Feature matrix: {len(features)} hourly snapshots")
    
    # Train anomaly model
    if len(features) < 24:
        log.warning("Not enough data for model training yet")
        return
    
    model, scaler, cols = train_anomaly_model(features)
    
    # Score current hour
    now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    current = {
        'hour_of_day': now_hour.hour,
        'day_of_week': now_hour.weekday(),
        'is_weekend': int(now_hour.weekday() >= 5),
        'motion_events': 0,
        'door_events': 0,
        'lights_on': 0,
        'total_events': 0,
    }
    
    # Fill current hour's actual data if available
    this_hour = features[features['hour'] == pd.Timestamp(now_hour)]
    if not this_hour.empty:
        for c in ['motion_events','door_events','lights_on','total_events']:
            current[c] = float(this_hour.iloc[0][c])
    
    X_current = np.array([[current[c] for c in cols]])
    X_scaled = scaler.transform(X_current)
    score = model.score_samples(X_scaled)[0]
    # Convert to 0-100 anomaly score (higher = more anomalous)
    anomaly_score = int(max(0, min(100, (-score + 0.5) * 100)))
    is_anomalous = anomaly_score > 70
    
    log.info(f"Current anomaly score: {anomaly_score}/100 ({'ANOMALOUS' if is_anomalous else 'normal'})")
    
    # Publish to HA
    publish_to_ha("sensor.habitus_anomaly_score", anomaly_score, {
        "friendly_name": "Habitus Anomaly Score",
        "unit_of_measurement": "",
        "icon": "mdi:brain",
        "state_class": "measurement",
    })
    publish_to_ha("binary_sensor.habitus_anomaly_detected", "on" if is_anomalous else "off", {
        "friendly_name": "Habitus Anomaly Detected",
        "device_class": "problem",
    })
    publish_to_ha("sensor.habitus_training_days", 
        round((df['ts'].max() - df['ts'].min()).total_seconds() / 86400) if hasattr(df['ts'], 'max') else 0,
        {"friendly_name": "Habitus Training Days", "unit_of_measurement": "days"})
    
    log.info("Published: sensor.habitus_anomaly_score, binary_sensor.habitus_anomaly_detected")

if __name__ == "__main__":
    run()
