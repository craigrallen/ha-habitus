"""Per-entity anomaly scoring — compares current values against per-entity baselines."""

import contextlib
import datetime
import json
import logging
import os

import numpy as np
import pandas as pd

from .sensor_classifier import ACCUMULATING, classify_sensor

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
ENTITY_BASELINES_PATH = os.path.join(DATA_DIR, "entity_baselines.json")
ENTITY_ANOMALIES_PATH = os.path.join(DATA_DIR, "entity_anomalies.json")


def build_entity_baselines(df: pd.DataFrame):
    """Build per-entity baselines by hour-of-day × day-of-week. Saves to entity_baselines.json.

    For accumulating sensors (kWh, gas m³, water L), baselines are built on hourly
    delta (consumption rate) values rather than absolute values, since the absolute
    value of an accumulator grows monotonically and has no meaningful mean/std.

    Each baseline slot is tagged with ``baseline_type``:
    - ``"rate"`` — accumulating sensors; slot mean/std are over hourly deltas.
    - ``"absolute"`` — all other sensors; slot mean/std are over raw values.

    Existing ``_accumulating_state`` (previous readings for delta computation) is
    preserved across retraining cycles so scoring continuity is not disrupted.
    """
    if df.empty:
        return
    df = df.copy()
    df["v"] = pd.to_numeric(df["mean"].fillna(df["sum"]), errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["hour_of_day"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df = df.dropna(subset=["v"])

    baselines: dict = {}
    for eid, group in df.groupby("entity_id"):
        entity_bl: dict = {}

        # Classify sensor type before choosing baseline method
        history_vals: list[float] = group.sort_values("ts")["v"].dropna().tolist()
        sensor_type = classify_sensor(eid, history=history_vals)

        if sensor_type == ACCUMULATING:
            entity_bl = _build_rate_baseline(group)
        else:
            for (h, d), g in group.groupby(["hour_of_day", "day_of_week"]):
                vals = g["v"].values
                if len(vals) < 3:
                    continue
                entity_bl[f"{h}_{d}"] = {
                    "mean": round(float(np.mean(vals)), 3),
                    "std": round(float(np.std(vals)), 3),
                    "n": len(vals),
                    "baseline_type": "absolute",
                }

        if entity_bl:
            entity_bl["_meta"] = {"sensor_type": sensor_type}
            baselines[eid] = entity_bl

    # Preserve accumulating state from any existing baselines file so that
    # scoring continuity is maintained across retraining cycles.
    if os.path.exists(ENTITY_BASELINES_PATH):
        with contextlib.suppress(json.JSONDecodeError, OSError):
            with open(ENTITY_BASELINES_PATH) as f:
                existing = json.load(f)
            if "_accumulating_state" in existing:
                baselines["_accumulating_state"] = existing["_accumulating_state"]

    with open(ENTITY_BASELINES_PATH, "w") as f:
        json.dump(baselines, f)
    log.info(f"Entity baselines saved for {len(baselines)} entities")


def _build_rate_baseline(group: pd.DataFrame) -> dict:
    """Build delta-based (rate) baseline slots for an accumulating sensor.

    Computes hourly deltas from the sorted time series, filters out negative
    deltas (meter resets / bad readings), then builds mean/std per
    hour-of-day × day-of-week slot.

    Args:
        group: DataFrame slice for a single entity with columns ``ts`` and ``v``.

    Returns:
        Dict of slot keys (``"h_d"``) → ``{mean, std, n, baseline_type: "rate"}``.
        Returns an empty dict when there is insufficient data.
    """
    group = group.sort_values("ts").dropna(subset=["v"])
    if len(group) < 2:
        return {}

    vals = group["v"].values.astype(float)
    ts_arr = pd.to_datetime(group["ts"].values)

    deltas = np.diff(vals)
    delta_ts = ts_arr[1:]

    # Exclude negative deltas (meter resets or bad readings)
    valid_mask = deltas >= 0
    deltas = deltas[valid_mask]
    delta_ts = delta_ts[valid_mask]

    if len(deltas) == 0:
        return {}

    delta_df = pd.DataFrame(
        {
            "delta": deltas,
            "hour_of_day": delta_ts.hour,
            "day_of_week": delta_ts.dayofweek,
        }
    )

    entity_bl: dict = {}
    for (h, d), g in delta_df.groupby(["hour_of_day", "day_of_week"]):
        slot_deltas = g["delta"].values
        if len(slot_deltas) < 3:
            continue
        entity_bl[f"{h}_{d}"] = {
            "mean": round(float(np.mean(slot_deltas)), 6),
            "std": round(float(np.std(slot_deltas)), 6),
            "n": int(len(slot_deltas)),
            "baseline_type": "rate",
        }

    return entity_bl


def score_entities(current_states: dict | None = None) -> list:
    """Score each entity's current value against its baseline.

    For accumulating sensors (``baseline_type == "rate"``), the score is
    computed against the hourly delta (current − previous reading) rather than
    the raw value.  Previous readings are stored in ``_accumulating_state``
    inside ``entity_baselines.json`` and updated on every call.

    New accumulating entities — those with no stored previous reading — are
    bootstrapped on the first call (prev value saved, no score emitted).
    Entities within 24 h of their first recorded delta are also exempt from
    scoring to avoid false positives during the bootstrap period.

    Args:
        current_states: ``{entity_id: current_value}`` mapping.  When ``None``,
            values are fetched from the HA REST API.

    Returns:
        List of anomaly dicts sorted by ``z_score`` descending (max 20 entries).
    """
    if not os.path.exists(ENTITY_BASELINES_PATH):
        return []

    with open(ENTITY_BASELINES_PATH) as f:
        baselines = json.load(f)

    now = datetime.datetime.now()
    h, d = now.hour, now.weekday()
    key = f"{h}_{d}"

    # Fetch current states from HA if not provided (exclude metadata keys)
    if current_states is None:
        entity_ids = [k for k in baselines if not k.startswith("_")]
        current_states = _fetch_current_states(entity_ids)

    # Load per-entity accumulating state (stores prev readings for delta computation)
    acc_state: dict = baselines.get("_accumulating_state", {})
    acc_state_updated: dict = dict(acc_state)
    acc_state_changed = False

    anomalies = []
    for eid, bl in baselines.items():
        if eid.startswith("_"):
            continue  # skip metadata keys (_z_score_run, _accumulating_state, …)
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

        baseline_type = b.get("baseline_type", "absolute")

        if baseline_type == "rate":
            prev_info = acc_state.get(eid)
            # Determine first_delta_ts — set on first encounter, preserved thereafter
            first_delta_ts = (
                now.isoformat()
                if prev_info is None
                else prev_info.get("first_delta_ts", now.isoformat())
            )
            acc_state_updated[eid] = {
                "prev_value": val,
                "prev_ts": now.isoformat(),
                "first_delta_ts": first_delta_ts,
            }
            acc_state_changed = True

            if prev_info is None:
                # Bootstrap: store prev value, no scoring on first encounter
                continue

            prev_value = prev_info.get("prev_value")
            if prev_value is None:
                continue

            # No scoring within first 24 h of delta accumulation
            first_dt = datetime.datetime.fromisoformat(first_delta_ts)
            if (now - first_dt).total_seconds() < 86400:
                continue

            delta = val - float(prev_value)
            if delta < 0:
                # Negative delta = meter reset or bad reading; skip scoring
                continue

            score_val = delta
        else:
            score_val = val

        z = abs(score_val - b["mean"]) / max(b["std"], 0.01)
        if z < 0.5:
            continue  # not interesting

        # Human-readable entity name
        name = eid.split(".")[-1].replace("_", " ").title()
        unit = _guess_unit(eid)

        if baseline_type == "rate":
            description = (
                f"{name} consumption rate is {_fmt(score_val, unit)}/h — "
                f"baseline {_day(d)} {h:02d}:00 is {_fmt(b['mean'], unit)}/h ±{_fmt(b['std'], unit)}/h"
            )
        else:
            description = (
                f"{name} is {_fmt(score_val, unit)} — "
                f"baseline {_day(d)} {h:02d}:00 is {_fmt(b['mean'], unit)} ±{_fmt(b['std'], unit)}"
            )

        anomalies.append(
            {
                "entity_id": eid,
                "name": name,
                "current_value": round(score_val, 6),
                "baseline_mean": round(b["mean"], 6),
                "baseline_std": round(b["std"], 6),
                "z_score": round(z, 2),
                "unit": unit,
                "description": description,
                "direction": "high" if score_val > b["mean"] else "low",
            }
        )

    # Persist updated accumulating state back to entity_baselines.json
    if acc_state_changed:
        baselines["_accumulating_state"] = acc_state_updated
        with open(ENTITY_BASELINES_PATH, "w") as f:
            json.dump(baselines, f, indent=2)

    anomalies.sort(key=lambda x: x["z_score"], reverse=True)
    top = anomalies[:20]

    with open(ENTITY_ANOMALIES_PATH, "w") as f:
        json.dump({"timestamp": now.isoformat(), "anomalies": top}, f, indent=2)
    return top


def compute_breakdown(anomaly_score: float, current_states: dict | None = None) -> list:
    """Compute per-entity z-score breakdown for anomaly UI display.

    When anomaly_score > 40, identifies the top-5 most anomalous entities and
    persists a ``_z_score_run`` record to ``entity_baselines.json`` so the
    Lovelace card can display plain-English reasons without a second API call.

    Args:
        anomaly_score: Combined IsolationForest anomaly score (0–100).
        current_states: Optional ``{entity_id: current_value}`` mapping.
            When ``None``, values are fetched from the HA REST API.

    Returns:
        List of up to 5 anomaly dicts with keys: ``entity_id``, ``name``,
        ``current_value``, ``baseline_mean``, ``baseline_std``, ``z_score``,
        ``unit``, ``description``, ``direction``.  Returns ``[]`` when
        ``anomaly_score <= 40`` or no baselines are available.
    """
    if anomaly_score <= 40:
        return []

    all_anomalies = score_entities(current_states)
    top5 = all_anomalies[:5]

    # Persist z-score run into entity_baselines.json only if the file exists
    if not os.path.exists(ENTITY_BASELINES_PATH):
        return top5

    with open(ENTITY_BASELINES_PATH) as f:
        baselines = json.load(f)

    baselines["_z_score_run"] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "anomaly_score": anomaly_score,
        "top5": top5,
    }
    with open(ENTITY_BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)

    log.info(f"Anomaly breakdown computed: score={anomaly_score}, top5={len(top5)} entities")
    return top5


def _fetch_current_states(entity_ids: list) -> dict:
    import requests  # type: ignore[import-untyped]

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
