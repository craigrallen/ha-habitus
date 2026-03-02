"""
Habitus Activity Baseline Engine
=================================

This module builds a rich *behavioural fingerprint* of a home by analysing
non-power sensors — the mundane signals of daily life:

  - Motion detectors (corridors, rooms)
  - Light switches (which rooms, what time)
  - Presence / occupancy sensors
  - Media players (TV on, music playing, which room)
  - Door and window sensors (arrivals, departures, ventilation patterns)
  - Person / device-tracker entities (who is home, when)
  - Weather / outdoor context (cold morning = heating, not unusual activity)

The key insight: **power alone lies**. A 2kW spike at 07:00 might be the
kettle (normal morning) or the hot tub heater coming on (anomalous). Knowing
that the kitchen motion sensor fired, the kitchen lights turned on, and the
person tracker shows someone home transforms this from ambiguous to clear.

Behavioural fingerprint per hour
---------------------------------
Each hourly snapshot contains:

  Power features (from main.py):
    total_power_w       — sum of all metered loads
    avg_temp_c          — mean indoor temperature
    sensor_changes      — total state-change count

  Activity features (this module):
    lights_on           — count of lights currently in ON state
    motion_events       — number of motion-trigger state changes this hour
    presence_count      — number of person entities reporting "home"
    media_active        — 1 if any media player is playing, else 0
    door_events         — door/window open/close events this hour
    outdoor_temp_c      — outdoor temperature (weather context)
    people_home_pct     — fraction of tracked people who are home (0–1)
    activity_diversity  — number of *distinct* sensor types active this hour

These are combined into a single feature vector per hour and fed to
the IsolationForest alongside the power features.

Why this matters
-----------------
- A home with no motion, no lights, and no people is in standby.
  Any power draw above baseline in that state is suspicious.
- A home with all people present, lights on, TV playing, doors opening
  is in full activity mode — high power is expected.
- Seasonal context (outdoor_temp_c) distinguishes "heater running" from
  "air con running" from "no heating reason but power is high".

All data comes from the HA long-term statistics API or the states API.
Nothing is stored locally except derived baselines (~10KB).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("habitus.activity")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ACTIVITY_BASELINE_PATH = os.path.join(DATA_DIR, "activity_baseline.json")
ACTIVITY_ANOMALIES_PATH = os.path.join(DATA_DIR, "activity_anomalies.json")

# ── Sensor classification ──────────────────────────────────────────────────────

#: Entity ID fragments that indicate a MOTION sensor
MOTION_PATTERNS: tuple[str, ...] = (
    "motion", "pir", "movement", "occupancy",
)

#: Fragments for LIGHT entities (domain prefix checked separately)
LIGHT_PATTERNS: tuple[str, ...] = (
    "light", "lamp", "bulb", "ceiling", "spot",
)

#: Fragments for PRESENCE / PERSON tracking
PRESENCE_PATTERNS: tuple[str, ...] = (
    "person.", "device_tracker.", "presence", "phone",
    "iphone", "android",
)

#: Fragments for MEDIA PLAYER activity
MEDIA_PATTERNS: tuple[str, ...] = (
    "media_player.", "tv", "speaker", "sonos", "spotify",
    "plex", "kodi", "chromecast", "firetv", "appletv",
)

#: Door / window sensors
DOOR_PATTERNS: tuple[str, ...] = (
    "door", "window", "gate", "garage", "entry",
)

#: Outdoor / weather context
WEATHER_PATTERNS: tuple[str, ...] = (
    "outdoor", "outside", "weather", "openweather",
    "met_office", "yr_", "accuweather",
)


def classify_entity(entity_id: str) -> str | None:
    """Return the behavioural category for an entity, or None if not relevant.

    Args:
        entity_id: Full HA entity ID, e.g. ``binary_sensor.hallway_motion``.

    Returns:
        One of ``"motion"``, ``"light"``, ``"presence"``, ``"media"``,
        ``"door"``, ``"weather"``, or ``None`` if not categorised.
    """
    eid = entity_id.lower()

    if any(p in eid for p in MOTION_PATTERNS) and "binary_sensor" in eid:
        return "motion"
    if eid.startswith("light.") or (
        "binary_sensor" in eid and any(p in eid for p in LIGHT_PATTERNS)
    ):
        return "light"
    if any(eid.startswith(p) for p in ("person.", "device_tracker.")):
        return "presence"
    if any(p in eid for p in PRESENCE_PATTERNS):
        return "presence"
    if eid.startswith("media_player.") or any(p in eid for p in MEDIA_PATTERNS):
        return "media"
    if "binary_sensor" in eid and any(p in eid for p in DOOR_PATTERNS):
        return "door"
    if any(p in eid for p in WEATHER_PATTERNS):
        return "weather"
    return None


def extract_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hourly activity feature vectors from raw state-change data.

    The input DataFrame comes from ``main.fetch_stats`` — it contains every
    hourly mean/sum value for every tracked entity.  We pivot and aggregate
    this into one row per hour with activity-level features.

    Args:
        df: Raw stats DataFrame with columns
            ``[entity_id, ts, mean, sum]``.

    Returns:
        DataFrame indexed by ``hour`` (UTC, floor-to-hour) with columns:

        - ``lights_on``        — average count of lights in ON state
        - ``motion_events``    — sum of binary state changes (0→1 transitions)
        - ``presence_count``   — average people-home count
        - ``media_active``     — max media activity flag (0/1)
        - ``door_events``      — sum of door open/close events
        - ``outdoor_temp_c``   — mean outdoor temperature
        - ``people_home_pct``  — fraction of tracked people home
        - ``activity_diversity``— distinct sensor categories active this hour
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["hour"] = pd.to_datetime(df["ts"], utc=True).dt.floor("h")
    df["category"] = df["entity_id"].apply(classify_entity)
    df["v"] = pd.to_numeric(df["mean"].fillna(df["sum"]), errors="coerce")

    active = df.dropna(subset=["category", "v"])

    hours = pd.DataFrame(
        {"hour": pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")}
    )

    def _agg(cat: str, fn: str) -> pd.Series:
        """Filter to category and aggregate by hour."""
        sub = active[active["category"] == cat]
        if sub.empty:
            return pd.Series(dtype=float)
        return getattr(sub.groupby("hour")["v"], fn)()

    # Lights: mean of binary ON states (binary sensor mean 0–1 = fraction on)
    lights_on = _agg("light", "mean").rename("lights_on")

    # Motion: sum of mean values (higher = more detections this hour)
    motion_events = _agg("motion", "sum").rename("motion_events")

    # Presence: sum (each person entity = 1 if home-ish)
    # person entity state is string ("home"/"not_home") — mean will be NaN
    # We use the count of non-zero means as a proxy
    presence_sub = active[active["category"] == "presence"]
    if not presence_sub.empty:
        presence_count = (
            presence_sub[presence_sub["v"] > 0.5]
            .groupby("hour")["entity_id"]
            .nunique()
            .rename("presence_count")
        )
        total_presence = presence_sub["entity_id"].nunique()
        people_home_pct = (
            presence_count / max(total_presence, 1)
        ).rename("people_home_pct")
    else:
        presence_count = pd.Series(dtype=float, name="presence_count")
        people_home_pct = pd.Series(dtype=float, name="people_home_pct")

    # Media: max value per hour (1 = something playing)
    media_active = _agg("media", "max").rename("media_active")

    # Doors: sum of events (each transition = activity)
    door_events = _agg("door", "sum").rename("door_events")

    # Outdoor temp: mean of weather sensors
    outdoor_temp = _agg("weather", "mean").rename("outdoor_temp_c")

    # Activity diversity: how many distinct sensor categories fired this hour
    diversity = (
        active[active["v"] > 0]
        .groupby("hour")["category"]
        .nunique()
        .rename("activity_diversity")
    )

    # Join all features
    result = hours.set_index("hour")
    for series in [
        lights_on, motion_events, presence_count, people_home_pct,
        media_active, door_events, outdoor_temp, diversity,
    ]:
        result = result.join(series, how="left")

    return result.fillna(0).reset_index()


def build_activity_baseline(activity_features: pd.DataFrame) -> dict[str, Any]:
    """Compute per-hour-of-day × day-of-week activity norms.

    For each time slot, stores the mean and standard deviation of each
    activity feature.  This is the "expected" activity level — deviations
    from this are scored as anomalies.

    Args:
        activity_features: Output of :func:`extract_activity_features`.

    Returns:
        Nested dict ``{"{hour}_{dow}": {feature: {mean, std}}}``.
        Saved to ``activity_baseline.json``.
    """
    if activity_features.empty:
        log.warning("No activity features — skipping baseline")
        return {}

    feats = activity_features.copy()
    feats["hour_of_day"] = feats["hour"].dt.hour
    feats["day_of_week"] = feats["hour"].dt.dayofweek

    activity_cols = [
        "lights_on", "motion_events", "presence_count", "people_home_pct",
        "media_active", "door_events", "outdoor_temp_c", "activity_diversity",
    ]

    baseline: dict[str, Any] = {}
    for (h, d), group in feats.groupby(["hour_of_day", "day_of_week"]):
        key = f"{h}_{d}"
        slot: dict[str, dict[str, float]] = {}
        for col in activity_cols:
            if col not in group.columns:
                continue
            vals = group[col].dropna().values
            if len(vals) < 3:
                continue
            slot[col] = {
                "mean": round(float(np.mean(vals)), 3),
                "std": round(float(np.std(vals)), 3),
                "n": len(vals),
            }
        if slot:
            baseline[key] = slot

    with open(ACTIVITY_BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)

    log.info(
        "Activity baseline saved — %d time slots, %d feature types",
        len(baseline),
        len(activity_cols),
    )
    return baseline


def score_activity_anomalies(current_states: dict[str, float] | None = None) -> dict:
    """Compare current activity levels against the stored baseline.

    Fetches live sensor states from HA if ``current_states`` is not provided.
    Computes z-scores for each activity feature in the current hour's slot.

    Args:
        current_states: Optional ``{entity_id: value}`` dict.  If not given,
            the HA REST API is called to fetch current states.

    Returns:
        Dict with ``anomalies`` list and summary fields.
    """
    if not os.path.exists(ACTIVITY_BASELINE_PATH):
        return {"anomalies": [], "summary": "No baseline yet"}

    with open(ACTIVITY_BASELINE_PATH) as f:
        baseline = json.load(f)

    now = datetime.datetime.now()
    key = f"{now.hour}_{now.weekday()}"

    if key not in baseline:
        return {"anomalies": [], "summary": f"No baseline for {key}"}

    slot = baseline[key]

    if current_states is None:
        current_states = _fetch_activity_states()

    # Derive current activity features from live states
    current = _derive_current_features(current_states)

    anomalies = []
    for feat, stats in slot.items():
        if stats["std"] < 0.01:
            continue
        val = current.get(feat, 0.0)
        z = abs(val - stats["mean"]) / max(stats["std"], 0.01)
        if z < 1.0:
            continue

        direction = "high" if val > stats["mean"] else "low"
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][now.weekday()]

        anomalies.append({
            "feature": feat,
            "label": _feature_label(feat),
            "current_value": round(val, 2),
            "baseline_mean": round(stats["mean"], 2),
            "baseline_std": round(stats["std"], 2),
            "z_score": round(z, 2),
            "direction": direction,
            "description": (
                f"{_feature_label(feat)} is {_fmt_feature(feat, val)} — "
                f"expected {_fmt_feature(feat, stats['mean'])} "
                f"±{_fmt_feature(feat, stats['std'])} "
                f"on {day_name} at {now.hour:02d}:00"
            ),
        })

    anomalies.sort(key=lambda x: x["z_score"], reverse=True)

    result = {
        "timestamp": now.isoformat(),
        "slot": key,
        "anomalies": anomalies,
        "current": current,
        "summary": (
            f"{len(anomalies)} activity anomalies at {now.hour:02d}:00 "
            f"({day_name})"
        ),
    }

    with open(ACTIVITY_ANOMALIES_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return result


def _fetch_activity_states() -> dict[str, float]:
    """Fetch current HA entity states relevant to activity scoring.

    Returns:
        Dict mapping entity_id to a numeric value (ON=1, OFF=0 for binary,
        float for numeric, home=1 for person entities).
    """
    import requests as req

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}
    result: dict[str, float] = {}

    try:
        r = req.get(f"{ha_url}/api/states", headers=headers, timeout=10)
        r.raise_for_status()
        for entity in r.json():
            eid = entity["entity_id"]
            cat = classify_entity(eid)
            if cat is None:
                continue
            state = entity["state"]
            try:
                val = float(state)
            except (ValueError, TypeError):
                # Map string states to numeric
                val = {
                    "on": 1.0, "off": 0.0,
                    "home": 1.0, "not_home": 0.0,
                    "playing": 1.0, "paused": 0.5,
                    "idle": 0.0, "standby": 0.0,
                    "unavailable": float("nan"),
                    "unknown": float("nan"),
                }.get(state.lower(), float("nan"))
            if not np.isnan(val):
                result[eid] = val
    except Exception as e:
        log.warning("Could not fetch activity states: %s", e)

    return result


def _derive_current_features(states: dict[str, float]) -> dict[str, float]:
    """Aggregate raw entity states into activity feature values.

    Args:
        states: Raw ``{entity_id: value}`` from :func:`_fetch_activity_states`.

    Returns:
        Dict with one value per activity feature column.
    """
    lights = [v for eid, v in states.items() if classify_entity(eid) == "light"]
    motion = [v for eid, v in states.items() if classify_entity(eid) == "motion"]
    presence = [v for eid, v in states.items() if classify_entity(eid) == "presence"]
    media = [v for eid, v in states.items() if classify_entity(eid) == "media"]
    doors = [v for eid, v in states.items() if classify_entity(eid) == "door"]
    weather = [v for eid, v in states.items() if classify_entity(eid) == "weather"]
    n_people = max(len(presence), 1)

    return {
        "lights_on": float(sum(1 for v in lights if v > 0.5)),
        "motion_events": float(sum(v for v in motion)),
        "presence_count": float(sum(1 for v in presence if v > 0.5)),
        "people_home_pct": sum(1 for v in presence if v > 0.5) / n_people,
        "media_active": float(max(media, default=0)),
        "door_events": float(sum(v for v in doors)),
        "outdoor_temp_c": float(np.mean(weather)) if weather else 0.0,
        "activity_diversity": float(
            len({
                classify_entity(eid)
                for eid, v in states.items()
                if v > 0.5 and classify_entity(eid) is not None
            })
        ),
    }


def _feature_label(feat: str) -> str:
    """Human-readable label for an activity feature."""
    return {
        "lights_on": "Lights on",
        "motion_events": "Motion activity",
        "presence_count": "People home",
        "people_home_pct": "Occupancy",
        "media_active": "Media playing",
        "door_events": "Door/window activity",
        "outdoor_temp_c": "Outdoor temperature",
        "activity_diversity": "Sensor diversity",
    }.get(feat, feat.replace("_", " ").title())


def _fmt_feature(feat: str, val: float) -> str:
    """Format a feature value with appropriate unit."""
    units = {
        "lights_on": " lights",
        "presence_count": " people",
        "outdoor_temp_c": "°C",
        "people_home_pct": "%",
    }
    unit = units.get(feat, "")
    if feat == "people_home_pct":
        return f"{val*100:.0f}{unit}"
    if feat in ("motion_events", "door_events", "activity_diversity"):
        return f"{val:.0f}{unit}"
    return f"{val:.1f}{unit}"


def get_activity_summary() -> dict[str, Any]:
    """Return a human-readable summary of current activity vs. baseline.

    Useful for the web UI overview card and for notification messages.

    Returns:
        Dict with ``status``, ``score``, ``highlights`` list, and raw ``anomalies``.
    """
    result = score_activity_anomalies()
    anomalies = result.get("anomalies", [])
    if not anomalies:
        return {"status": "normal", "score": 0, "highlights": [], "anomalies": []}

    top = anomalies[:3]
    score = int(min(100, sum(a["z_score"] * 20 for a in top[:3]) / 3))
    highlights = [a["description"] for a in top]

    return {
        "status": "anomaly" if score > 60 else "elevated",
        "score": score,
        "highlights": highlights,
        "anomalies": anomalies,
    }
