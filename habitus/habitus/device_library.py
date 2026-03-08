"""Device template library — learns device power profiles from named W sensors."""

from __future__ import annotations

import json
import logging
import os
import re

import pandas as pd

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DEVICE_LIBRARY_PATH = os.path.join(DATA_DIR, "device_library.json")

# Regex to strip common suffixes and extract a human-readable device name
# e.g. sensor.kettle_power → "Kettle"
#      sensor.dishwasher_electric_consumption_w → "Dishwasher"
#      sensor.living_room_tv_power → "Living Room TV"
_SUFFIX_STRIP = re.compile(
    r"[_\s]*(power|watt|watts|electric.*consumption.*w|consumption.*w|energy.*w|_w|_power|_watt)$",
    re.IGNORECASE,
)
_DOMAIN_STRIP = re.compile(r"^sensor\.", re.IGNORECASE)

# Keywords that suggest a sensor IS a per-device monitor (smart plug etc.)
_DEVICE_HINTS = re.compile(
    r"(kettle|fridge|freezer|dishwasher|washer|dryer|washing|tumble|oven|microwave|"
    r"toaster|coffee|tv|television|computer|pc|laptop|printer|charger|heater|fan|"
    r"pump|light|lamp|plug|socket|switch|aircon|ac_unit|boiler|hot.?water|"
    r"router|nas|server|camera|doorbell)",
    re.IGNORECASE,
)

# Keywords that suggest a sensor is a PHASE/TOTAL/AGGREGATE (NOT a per-device sensor)
_AGGREGATE_HINTS = re.compile(
    r"(total|combined|phase|l1|l2|l3|shore|grid|solar|inverter|mcu|charger.input|"
    r"mains|house|whole.home|whole.house|main.meter|smart.meter|net.energy)",
    re.IGNORECASE,
)

MAX_PLAUSIBLE_W = 25_000  # 25kW upper bound


def is_device_sensor(entity_id: str, name: str = "") -> bool:
    """Return True if this W sensor likely monitors a single device."""
    combined = f"{entity_id} {name}".lower()
    if _AGGREGATE_HINTS.search(combined):
        return False
    # Must look like a device (explicit hint keyword), OR be a generic plug/switch sensor
    # that's NOT an aggregate (e.g. sensor.plug_1_power)
    if _DEVICE_HINTS.search(combined):
        return True
    # Generic plug pattern: sensor.xxx_power, sensor.xxx_watt where xxx is a short slug
    slug = re.sub(r"^sensor\.", "", entity_id.lower())
    slug = _SUFFIX_STRIP.sub("", slug)
    # If slug is short (<30 chars) and not an aggregate → likely a plug
    return len(slug) < 30 and not _AGGREGATE_HINTS.search(slug)


def extract_device_name(entity_id: str) -> str:
    """Convert entity_id to human-readable device name."""
    slug = _DOMAIN_STRIP.sub("", entity_id)
    slug = _SUFFIX_STRIP.sub("", slug)
    slug = slug.replace("_", " ").strip()
    return slug.title()


def build_device_profile(series: pd.Series, entity_id: str) -> dict | None:
    """
    Given a time series of watt readings for a single device, extract:
    - off_w: idle/standby wattage
    - on_w: typical active wattage
    - step_w: transition size (on_w - off_w)
    - typical_on_min: median on-cycle duration in minutes
    - cycles_per_day: average daily on-cycles
    - daily_kwh: estimated daily energy use
    - confidence: 0-1 based on data quality
    """
    if series.empty or series.isna().all():
        return None
    vals = pd.to_numeric(series, errors="coerce").dropna().clip(lower=0, upper=MAX_PLAUSIBLE_W)
    if len(vals) < 24:  # need at least 24 hours
        return None

    # Bimodal distribution → find on/off peaks
    # Use percentiles as a fast heuristic
    p5 = float(vals.quantile(0.05))
    p95 = float(vals.quantile(0.95))
    p50 = float(vals.quantile(0.50))

    off_w = round(p5, 1)
    on_w = round(p95, 1)
    step_w = round(on_w - off_w, 1)

    if step_w < 5:  # too small to be meaningful
        return None

    # Detect on-cycles (values above midpoint)
    threshold = off_w + step_w * 0.4
    is_on = vals > threshold

    # Count cycles and estimate typical duration
    transitions = is_on.astype(int).diff().fillna(0)
    on_starts = int((transitions == 1).sum())

    # Calculate total hours from index if it's a DatetimeIndex
    if isinstance(vals.index, pd.DatetimeIndex):
        total_hours = (vals.index[-1] - vals.index[0]).total_seconds() / 3600
    else:
        total_hours = len(vals) / 60

    days = max(total_hours / 24, 1.0)

    on_fraction = float(is_on.mean())
    cycles_per_day = round(on_starts / days, 2)

    # Typical on-duration: total on-time / cycles
    if on_starts > 0:
        total_on_points = int(is_on.sum())
        avg_on_min = round((total_on_points / on_starts) * (total_hours * 60 / len(vals)), 1)
    else:
        avg_on_min = 0.0

    daily_kwh = round(on_fraction * on_w / 1000 * 24, 3)

    # Confidence: based on data volume and step size clarity
    confidence = min(1.0, (min(days, 30) / 30) * (min(step_w, 2000) / 2000 + 0.5))

    return {
        "entity_id": entity_id,
        "name": extract_device_name(entity_id),
        "off_w": off_w,
        "on_w": on_w,
        "step_w": step_w,
        "median_w": round(p50, 1),
        "typical_on_min": avg_on_min,
        "cycles_per_day": cycles_per_day,
        "daily_kwh": daily_kwh,
        "confidence": round(confidence, 2),
        "data_days": round(days, 0),
        "is_always_on": on_fraction > 0.95 and step_w < 50,
    }


def build_library_from_features(df: pd.DataFrame | None) -> list[dict]:
    """
    Build device template library from the training features DataFrame.
    df must have columns: entity_id, ts (or hour), mean (watt reading)
    Returns list of device profile dicts.
    """
    profiles: list[dict] = []

    if df is None or df.empty:
        return profiles

    # Find candidate device sensors in the data
    # df has entity_id column with the sensor IDs
    if "entity_id" not in df.columns:
        return profiles

    device_eids = [eid for eid in df["entity_id"].unique() if is_device_sensor(str(eid))]
    log.info(
        "Device library: found %d candidate device sensors out of %d total",
        len(device_eids),
        df["entity_id"].nunique(),
    )

    ts_col = "ts" if "ts" in df.columns else "hour"
    for eid in device_eids:
        sub = df[df["entity_id"] == eid].copy()
        if sub.empty:
            continue
        if ts_col not in sub.columns:
            continue
        sub = sub.set_index(ts_col).sort_index()
        if "mean" in sub.columns:
            series = pd.to_numeric(sub["mean"], errors="coerce").dropna()
        elif "total_power_w" in sub.columns:
            series = sub["total_power_w"].dropna()
        else:
            continue
        profile = build_device_profile(series, str(eid))
        if profile:
            profiles.append(profile)

    profiles.sort(key=lambda x: -x["daily_kwh"])
    log.info("Device library: built %d device profiles", len(profiles))
    return profiles


def save_library(profiles: list[dict]) -> None:
    """Persist device library to disk."""
    from .utils import atomic_write as _aw  # noqa: PLC0415

    _aw(DEVICE_LIBRARY_PATH, {"devices": profiles, "count": len(profiles)})


def load_library() -> list[dict]:
    """Load device library from disk."""
    try:
        with open(DEVICE_LIBRARY_PATH) as f:
            return json.load(f).get("devices", [])
    except Exception:
        return []


def match_wattage_to_device(step_w: float, tolerance: float = 0.20) -> dict | None:
    """Find the best-matching device template for a given wattage step."""
    devices = load_library()
    best = None
    best_score = float("inf")
    for d in devices:
        if d["step_w"] < 10:
            continue
        diff = abs(d["step_w"] - step_w) / max(d["step_w"], 1)
        if diff < tolerance and diff < best_score:
            best = d
            best_score = diff
    return best


def energy_breakdown() -> list[dict]:
    """Return per-device daily energy breakdown sorted by consumption."""
    devices = load_library()
    return [
        {
            "name": d["name"],
            "entity_id": d["entity_id"],
            "daily_kwh": d["daily_kwh"],
            "on_w": d["on_w"],
            "off_w": d["off_w"],
            "cycles_per_day": d["cycles_per_day"],
            "typical_on_min": d["typical_on_min"],
            "is_always_on": d.get("is_always_on", False),
            "confidence": d["confidence"],
        }
        for d in sorted(devices, key=lambda x: -x["daily_kwh"])
        if d.get("daily_kwh", 0) > 0
    ]
