"""Per-entity anomaly scoring — compares current values against per-entity baselines."""

import contextlib
import datetime
import json
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
ENTITY_BASELINES_PATH = os.path.join(DATA_DIR, "entity_baselines.json")
ENTITY_ANOMALIES_PATH = os.path.join(DATA_DIR, "entity_anomalies.json")


def build_entity_baselines(df: pd.DataFrame):
    """Build per-entity mean/std by hour-of-day × day-of-week. Saves to entity_baselines.json."""
    if df.empty:
        return
    df = df.copy()
    df["v"] = pd.to_numeric(df["mean"].fillna(df["sum"]), errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["hour_of_day"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df = df.dropna(subset=["v"])

    baselines = {}
    for eid, group in df.groupby("entity_id"):
        entity_bl = {}
        for (h, d), g in group.groupby(["hour_of_day", "day_of_week"]):
            vals = g["v"].values
            if len(vals) < 3:
                continue
            entity_bl[f"{h}_{d}"] = {
                "mean": round(float(np.mean(vals)), 3),
                "std": round(float(np.std(vals)), 3),
                "n": len(vals),
            }
        if entity_bl:
            baselines[eid] = entity_bl

    with open(ENTITY_BASELINES_PATH, "w") as f:
        json.dump(baselines, f)
    log.info(f"Entity baselines saved for {len(baselines)} entities")


def score_entities(current_states: dict = None) -> list:
    """
    Score each entity's current value against its baseline.
    current_states: {entity_id: current_value} — if None, uses HA REST API.
    Returns list of anomalies sorted by z_score desc.
    """
    if not os.path.exists(ENTITY_BASELINES_PATH):
        return []

    with open(ENTITY_BASELINES_PATH) as f:
        baselines = json.load(f)

    now = datetime.datetime.now()
    h, d = now.hour, now.weekday()
    key = f"{h}_{d}"

    # Fetch current states from HA if not provided
    if current_states is None:
        current_states = _fetch_current_states(list(baselines.keys()))

    anomalies = []
    for eid, bl in baselines.items():
        if key not in bl:
            continue
        b = bl[key]
        if b["std"] < 0.01:
            continue  # constant sensor, skip
        current = current_states.get(eid)
        if current is None:
            continue
        try:
            val = float(current)
        except (TypeError, ValueError):
            continue
        z = abs(val - b["mean"]) / max(b["std"], 0.01)
        if z < 0.5:
            continue  # not interesting

        # Human-readable entity name
        name = eid.split(".")[-1].replace("_", " ").title()
        unit = _guess_unit(eid)

        anomalies.append(
            {
                "entity_id": eid,
                "name": name,
                "current_value": round(val, 2),
                "baseline_mean": round(b["mean"], 2),
                "baseline_std": round(b["std"], 2),
                "z_score": round(z, 2),
                "unit": unit,
                "description": f"{name} is {_fmt(val, unit)} — baseline {_day(d)} {h:02d}:00 is {_fmt(b['mean'], unit)} ±{_fmt(b['std'], unit)}",
                "direction": "high" if val > b["mean"] else "low",
            }
        )

    anomalies.sort(key=lambda x: x["z_score"], reverse=True)
    top = anomalies[:20]

    with open(ENTITY_ANOMALIES_PATH, "w") as f:
        json.dump({"timestamp": now.isoformat(), "anomalies": top}, f, indent=2)
    return top


def _fetch_current_states(entity_ids: list) -> dict:
    import requests

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}
    result = {}
    try:
        r = requests.get(f"{ha_url}/api/states", headers=headers, timeout=10)
        if r.status_code == 200:
            for s in r.json():
                if s["entity_id"] in entity_ids:
                    with contextlib.suppress(ValueError, TypeError):

                        result[s["entity_id"]] = float(s["state"])
    except Exception as e:
        log.warning(f"Could not fetch current states: {e}")
    return result


def _guess_unit(eid):
    e = eid.lower()
    if "temperature" in e:
        return "°C"
    if "humidity" in e:
        return "%"
    if "_w" in e or "watt" in e or "power" in e:
        return "W"
    if "energy" in e or "kwh" in e:
        return "kWh"
    if "current" in e or "_a" in e:
        return "A"
    if "voltage" in e:
        return "V"
    if "pressure" in e:
        return "hPa"
    return ""


def _fmt(val, unit):
    return f"{val:.1f}{unit}" if unit else f"{val:.2f}"


def _day(d):
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]
