"""
Habitus v2.0 — behavioral intelligence for Home Assistant.
HA is source of truth. /data stores only inference artifacts.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import requests

from . import activity as activity_engine
from . import anomaly_breakdown, seasonal
from . import patterns as pattern_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_WS = os.environ.get("HA_WS", "ws://supervisor/core/api/websocket")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
NOTIFY_SVC = os.environ.get("HABITUS_NOTIFY_SERVICE", "notify.notify")
NOTIFY_ON = os.environ.get("HABITUS_NOTIFY_ON", "true").lower() == "true"
THRESHOLD = int(os.environ.get("HABITUS_ANOMALY_THRESHOLD", "70"))

MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")
STATE_PATH = os.path.join(DATA_DIR, "run_state.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "progress.json")
RESCAN_FLAG = os.path.join(DATA_DIR, ".rescan_requested")

FEATURE_COLS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "total_power_w",
    "avg_temp_c",
    "sensor_changes",
    # Activity features — provide richer context beyond power alone
    "lights_on",
    "motion_events",
    "presence_count",
    "people_home_pct",
    "media_active",
    "door_events",
    "outdoor_temp_c",
    "activity_diversity",
]

BEHAVIORAL_KEYWORDS = [
    "energy",
    "temperature",
    "humidity",
    "power",
    "watt",
    "consumed",
    "production",
    "solar",
    "battery",
    "pump",
    "motion",
    "door",
    "contact",
    "occupancy",
    "presence",
    "bilge",
    "shore",
    "inverter",
    "mppt",
    "epever",
    "scm",
    "mcu",
]
SKIP = [
    "rssi",
    "signal_strength",
    "fossil_fuel",
    "co2_intensity",
    "grid_fossil",
    "firmware",
    "battery_level",
    "lqi",
]


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
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def set_progress(phase, done=0, total=0, rows=0, elapsed=0.0, eta=0.0):
    try:
        pct = round(done / total * 100) if total else 100
        with open(PROGRESS_PATH, "w") as f:
            json.dump(
                {
                    "running": True,
                    "phase": phase,
                    "done": done,
                    "total": total,
                    "pct": pct,
                    "rows": rows,
                    "elapsed_min": round(elapsed / 60, 1),
                    "eta_min": round(eta / 60, 1),
                },
                f,
            )
    except Exception:
        pass


def clear_progress():
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)


# ── WebSocket ──────────────────────────────────────────────────────────────────
async def ws_connect():
    import websockets

    ws = await websockets.connect(HA_WS, max_size=50 * 1024 * 1024)
    await ws.recv()
    await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
    result = json.loads(await ws.recv())
    if result.get("type") != "auth_ok":
        raise RuntimeError(f"Auth failed: {result}")
    return ws


async def get_stat_ids():
    ws = await ws_connect()
    await ws.send(json.dumps({"id": 1, "type": "recorder/list_statistic_ids"}))
    result = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
    await ws.close()
    all_ids = [s["statistic_id"] for s in result.get("result", [])]
    behavioral = [e for e in all_ids if is_behavioral(e)]
    log.info(f"Found {len(behavioral)} behavioral sensors (from {len(all_ids)} total)")
    return behavioral


async def fetch_stats(entity_ids, start_iso, end_iso=None):
    if end_iso is None:
        end_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:00:00+00:00")
    all_rows = []
    done = 0
    total = len(entity_ids)
    import time as _t

    t0 = _t.time()
    for eid in entity_ids:
        try:
            ws = await ws_connect()
            await ws.send(
                json.dumps(
                    {
                        "id": 1,
                        "type": "recorder/statistics_during_period",
                        "start_time": start_iso,
                        "end_time": end_iso,
                        "statistic_ids": [eid],
                        "period": "hour",
                        "types": ["mean", "sum"],
                    }
                )
            )
            result = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            await ws.close()
            for sid, points in result.get("result", {}).items():
                for p in points:
                    ts = p["start"]
                    if ts > 1e10:
                        ts /= 1000
                    all_rows.append(
                        {"entity_id": sid, "ts": ts, "mean": p.get("mean"), "sum": p.get("sum")}
                    )
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = _t.time() - t0
                eta = (elapsed / done) * (total - done) if done else 0
                set_progress("fetching", done, total, len(all_rows), elapsed, eta)
                log.info(f"  {done}/{total} sensors — {len(all_rows):,} rows")
        except Exception as e:
            log.warning(f"Error {eid}: {e}")
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    log.info(f"Fetched {len(df):,} rows | {df['ts'].min().date()} → {df['ts'].max().date()}")
    return df


# ── Features ───────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build hourly feature matrix combining power and activity signals.

    Merges power metrics (total load, temperature, sensor change rate) with
    activity features (lights, motion, presence, media) so the model can
    distinguish a cold-morning heating spike from a genuine anomaly.

    Args:
        df: Raw stats DataFrame from ``fetch_stats`` with columns
            ``[entity_id, ts, mean, sum]``.

    Returns:
        DataFrame with one row per hour and all FEATURE_COLS populated.
    """
    df = df.copy()
    df["hour"] = df["ts"].dt.floor("h")
    hours = pd.DataFrame({"hour": pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")})
    hours["hour_of_day"] = hours["hour"].dt.hour
    hours["day_of_week"] = hours["hour"].dt.dayofweek
    hours["is_weekend"] = (hours["day_of_week"] >= 5).astype(int)
    hours["month"] = hours["hour"].dt.month
    power = df[
        df["entity_id"].str.contains("consumed_w|watt|power|inverter|load", case=False, na=False)
    ].copy()
    power["v"] = pd.to_numeric(power["mean"], errors="coerce")
    total_power = power.groupby("hour")["v"].sum().rename("total_power_w")
    temp = df[df["entity_id"].str.contains("temperature", case=False, na=False)].copy()
    temp["v"] = pd.to_numeric(temp["mean"], errors="coerce")
    avg_temp = temp.groupby("hour")["v"].mean().rename("avg_temp_c")
    activity = df.groupby("hour").size().rename("sensor_changes")
    features = hours.set_index("hour")
    for s in [total_power, avg_temp, activity]:
        features = features.join(s, how="left")
    # Merge activity features from activity engine
    try:
        act_features = activity_engine.extract_activity_features(df)
        if not act_features.empty:
            act_features = act_features.set_index("hour")
            features = features.join(act_features, how="left")
    except Exception as e:
        log.warning("Activity feature extraction failed: %s", e)
    return features.fillna(0).reset_index()


# ── Model ──────────────────────────────────────────────────────────────────────
def train_model(features):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = features[FEATURE_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(Xs)
    days = round((features["hour"].max() - features["hour"].min()).total_seconds() / 86400)
    log.info(f"Main model trained on {len(X):,} snapshots ({days}d)")
    return model, scaler


def save_artifacts(model, scaler, features):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    baseline = {}
    for (h, d), g in features.groupby(["hour_of_day", "day_of_week"]):
        baseline[f"{h}_{d}"] = {
            "mean_power": round(float(g["total_power_w"].mean()), 1),
            "std_power": round(float(g["total_power_w"].std()), 1),
            "mean_temp": round(float(g["avg_temp_c"].mean()), 1),
            "n_samples": len(g),
        }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f)
    log.info(f"Artifacts saved ({os.path.getsize(MODEL_PATH)//1024}KB model)")


def score_current(features):
    now = datetime.datetime.now(datetime.UTC).replace(minute=0, second=0, microsecond=0)
    row = features[features["hour"] == pd.Timestamp(now)]
    if not row.empty:
        X = row[FEATURE_COLS].values
    else:
        X = np.array([[now.hour, now.weekday(), int(now.weekday() >= 5), now.month, 0, 0, 0]])
    score, model_used = seasonal.score_with_best_model(X)
    log.info(f"Scored with {model_used} model")
    return score


# ── Notify ────────────────────────────────────────────────────────────────────
def send_notification(title, message):
    if not NOTIFY_ON:
        return
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    service = NOTIFY_SVC.replace(".", "/")
    try:
        r = requests.post(
            f"{HA_URL}/api/services/{service}",
            headers=headers,
            json={
                "title": title,
                "message": message,
                "data": {
                    "url": "/hassio/ingress/57582523_habitus",
                    "clickAction": "/hassio/ingress/57582523_habitus",
                },
            },
            timeout=5,
        )
        log.info(f"Notification sent via {NOTIFY_SVC}: {r.status_code}")
    except Exception as e:
        log.warning(f"Notification failed: {e}")


# ── Publish ───────────────────────────────────────────────────────────────────
def publish(entity_id, state, attributes=None):
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(
            f"{HA_URL}/api/states/{entity_id}",
            headers=headers,
            json={"state": str(state), "attributes": attributes or {}},
            timeout=5,
        )
        if r.status_code not in (200, 201):
            log.error(f"Publish {entity_id}: {r.status_code}")
    except Exception as e:
        log.error(f"Publish {entity_id}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def run(days_history: int, mode: str = "full") -> None:
    """Main Habitus run loop — fetch, train (if needed), score, publish.

    In ``overnight`` schedule mode, daytime invocations pass ``mode='score'``
    to skip the expensive training step and only re-score the current hour
    using the existing model.  Training is deferred to the overnight window
    (default 02:00) when the home is typically idle.

    In ``continuous`` mode every run does a full retrain.

    Args:
        days_history: Maximum history window.  HA returns whatever it has.
        mode: ``"full"`` to fetch + train + score, ``"score"`` to score only.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    now_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:00:00+00:00")
    state = load_state()

    # ── Score-only mode ────────────────────────────────────────────────────────
    # Used in overnight schedule during daytime hours.
    # Re-scores current state using the existing trained model — fast (<5s).
    # Skips all data fetching and retraining.
    if mode == "score":
        if not os.path.exists(MODEL_PATH):
            log.info("No model yet — falling back to full run")
            mode = "full"
        else:
            log.info("Score-only mode — using existing model")
            now = datetime.datetime.now(datetime.UTC).replace(minute=0, second=0, microsecond=0)
            X = np.array(
                [
                    [
                        now.hour,
                        now.weekday(),
                        int(now.weekday() >= 5),
                        now.month,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ]
            )
            anomaly_score, _ = seasonal.score_with_best_model(X)
            entity_anomalies = anomaly_breakdown.score_entities()
            activity_summary = activity_engine.get_activity_summary()
            training_days = state.get("training_days", 0)
            entity_count = state.get("entity_count", 0)
            top_anomaly = entity_anomalies[0]["description"] if entity_anomalies else None
            state.update(
                {
                    "last_score": now_iso,
                    "anomaly_score": anomaly_score,
                    "mode": "score",
                }
            )
            save_state(state)
            is_anomalous = anomaly_score > THRESHOLD
            log.info(
                "Score-only: %d/100 (%s)",
                anomaly_score,
                "⚠ ANOMALY" if is_anomalous else "✓ normal",
            )
            if is_anomalous:
                msg_parts = [f"Score: {anomaly_score}/100"]
                if top_anomaly:
                    msg_parts.append(top_anomaly)
                send_notification("🧠 Habitus — Unusual Activity", "\n".join(msg_parts))
            _publish_sensors(anomaly_score, is_anomalous, training_days, entity_count)
            return

    stat_ids = await get_stat_ids()
    if not stat_ids:
        log.error("No behavioral sensors found")
        return

    if state.get("data_to") and os.path.exists(MODEL_PATH):
        # Incremental
        fetch_from = state["data_to"]
        log.info(f"Incremental: {fetch_from} → {now_iso}")
        df_new = await fetch_stats(stat_ids, fetch_from, now_iso)
        if df_new.empty or len(df_new) < 24:
            log.info("Not enough new data — scoring with existing model")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            anomaly_score = seasonal.score_with_best_model(
                np.array(
                    [
                        [
                            datetime.datetime.now().hour,
                            datetime.datetime.now().weekday(),
                            int(datetime.datetime.now().weekday() >= 5),
                            datetime.datetime.now().month,
                            0,
                            0,
                            0,
                        ]
                    ]
                )
            )[0]
            training_days = state.get("training_days", 0)
            entity_count = state.get("entity_count", len(stat_ids))
        else:
            full_from = state.get("data_from", "2000-01-01T00:00:00+00:00")
            log.info(f"Retraining full window {full_from} → {now_iso}")
            set_progress("fetching", 0, len(stat_ids), 0, 0, 0)
            df_full = await fetch_stats(stat_ids, full_from, now_iso)
            if df_full.empty:
                log.warning("No data")
                return
            set_progress("building_baselines", len(stat_ids), len(stat_ids), len(df_full), 0, 0)
            anomaly_breakdown.build_entity_baselines(df_full)
            features = build_features(df_full)
            del df_full
            set_progress("training", len(stat_ids), len(stat_ids), len(features), 0, 0)
            model, scaler = train_model(features)
            save_artifacts(model, scaler, features)
            set_progress("seasonal_training", len(stat_ids), len(stat_ids), len(features), 0, 0)
            seasonal.train_seasonal_models(features)
            set_progress("pattern_analysis", len(stat_ids), len(stat_ids), len(features), 0, 0)
            pattern_engine.run(features, stat_ids)
            anomaly_score = score_current(features)
            training_days = round(
                (features["hour"].max() - features["hour"].min()).total_seconds() / 86400
            )
            entity_count = len(stat_ids)
    else:
        # First run — all history
        full_from = "2000-01-01T00:00:00+00:00"
        log.info(f"First run: all history from {full_from}")
        set_progress("fetching", 0, len(stat_ids), 0, 0, 0)
        df = await fetch_stats(stat_ids, full_from, now_iso)
        if df.empty:
            log.warning("No data returned")
            return
        set_progress("building_baselines", len(stat_ids), len(stat_ids), len(df), 0, 0)
        anomaly_breakdown.build_entity_baselines(df)
        activity_engine.build_activity_baseline(activity_engine.extract_activity_features(df))
        features = build_features(df)
        del df
        set_progress("training", len(stat_ids), len(stat_ids), len(features), 0, 0)
        model, scaler = train_model(features)
        save_artifacts(model, scaler, features)
        set_progress("seasonal_training", len(stat_ids), len(stat_ids), len(features), 0, 0)
        seasonal.train_seasonal_models(features)
        set_progress("pattern_analysis", len(stat_ids), len(stat_ids), len(features), 0, 0)
        pattern_engine.run(features, stat_ids)
        anomaly_score = score_current(features)
        training_days = round(
            (features["hour"].max() - features["hour"].min()).total_seconds() / 86400
        )
        entity_count = len(stat_ids)
        state["data_from"] = full_from
        log.info(f"Discovery: {full_from} → {now_iso} ({training_days}d)")

    # Per-entity and activity scoring
    entity_anomalies = anomaly_breakdown.score_entities()
    activity_summary = activity_engine.get_activity_summary()
    top_anomaly = entity_anomalies[0]["description"] if entity_anomalies else None

    state.update(
        {
            "last_run": now_iso,
            "data_to": now_iso,
            "training_days": training_days,
            "entity_count": entity_count,
            "anomaly_score": anomaly_score,
            "top_anomaly": top_anomaly,
            "seasonal_models": seasonal.seasonal_status(),
        }
    )
    save_state(state)
    clear_progress()

    is_anomalous = anomaly_score > THRESHOLD
    log.info(f"Score: {anomaly_score}/100 ({'⚠ ANOMALY' if is_anomalous else '✓ normal'})")
    if is_anomalous:
        msg_parts = [f"Score: {anomaly_score}/100"]
        if top_anomaly:
            msg_parts.append(top_anomaly)
        if activity_summary.get("highlights"):
            msg_parts.append(activity_summary["highlights"][0])
        send_notification("🧠 Habitus — Unusual Activity", "\n".join(msg_parts))

    _publish_sensors(anomaly_score, is_anomalous, training_days, entity_count)
    log.info("Done.")


def _publish_sensors(
    anomaly_score: int,
    is_anomalous: bool,
    training_days: int,
    entity_count: int,
) -> None:
    """Publish the 4 Habitus sensor entities to HA.

    Called from both full and score-only run modes.

    Args:
        anomaly_score: Current 0–100 anomaly score.
        is_anomalous: Whether score exceeds the configured threshold.
        training_days: Days of history the model was trained on.
        entity_count: Number of behavioral sensors being tracked.
    """
    publish(
        "sensor.habitus_anomaly_score",
        anomaly_score,
        {
            "friendly_name": "Habitus Anomaly Score",
            "unit_of_measurement": "",
            "icon": "mdi:brain",
            "state_class": "measurement",
        },
    )
    publish(
        "binary_sensor.habitus_anomaly_detected",
        "on" if is_anomalous else "off",
        {"friendly_name": "Habitus Anomaly", "device_class": "problem"},
    )
    publish(
        "sensor.habitus_training_days",
        training_days,
        {
            "friendly_name": "Habitus Training Days",
            "unit_of_measurement": "days",
            "icon": "mdi:calendar-range",
        },
    )
    publish(
        "sensor.habitus_entity_count",
        entity_count,
        {
            "friendly_name": "Habitus Tracked Sensors",
            "unit_of_measurement": "sensors",
            "icon": "mdi:chip",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habitus ML engine — trains behavioral model and scores current state."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Maximum history window in days (HA limits to what it has)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "score"],
        default="full",
        help=(
            "full: fetch new data, retrain model, score, publish. "
            "score: skip training, only score current state and publish. "
            "In overnight schedule, daytime runs use 'score' to avoid "
            "resource-intensive training during active hours."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run(days_history=args.days, mode=args.mode))
