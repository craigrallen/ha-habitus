"""Phantom Load Hunter — identify devices drawing power 24/7 unnecessarily."""

import json
import logging
import os
import re

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PHANTOM_PATH = os.path.join(DATA_DIR, "phantom_loads.json")
ENTITY_BASELINES_PATH = os.path.join(DATA_DIR, "entity_baselines.json")
WATT_ENTITIES_PATH = os.path.join(DATA_DIR, "watt_entities.json")

# Patterns that indicate a REAL instantaneous watt sensor (unit_of_measurement = W)
# Must end in _power, _w, or _watts — and not be a rate/speed/flow sensor
_WATT_SUFFIX = re.compile(
    r"(^|_)(power|watt|watts|active_power|apparent_power|reactive_power|current_power|"
    r"consumed_w|consumption_w|output_power|input_power|solar_power|load_power)$"
)

# Entity ID fragments that are definitely NOT power sensors despite containing keywords
_EXCLUDE_FRAGMENTS = (
    "write_rate", "read_rate", "transfer", "throughput",
    "kvah", "kvarh", "kvar",          # reactive/apparent energy
    "kwh", "kWh",                      # cumulative energy (not instantaneous)
    "energy", "consumed_energy",       # cumulative
    "cost", "price", "tariff",
    "speed", "rate", "bytes",
    "voltage", "_v_", "_v$",
    "current_a", "_a_", "ampere",
    "frequency", "_hz",
    "humidity", "temperature", "pressure",
    "battery", "rssi", "signal",
)

# Max realistic phantom load for a single device (anything above = sensor error)
MAX_PHANTOM_W = 500.0


def _is_watt_sensor(eid: str, units_map: dict) -> bool:
    """Return True only if this entity is confirmed to report watts."""
    # 1. Prefer unit_of_measurement from HA if available
    uom = units_map.get(eid, "")
    if uom:
        return uom == "W"
    # 2. Fallback: strict suffix matching on entity ID
    name = eid.split(".")[-1].lower()
    if any(frag in name for frag in _EXCLUDE_FRAGMENTS):
        return False
    return bool(_WATT_SUFFIX.search(name))


def cache_watt_entities(states: list) -> None:
    """Call this at startup with HA states to build a unit_of_measurement cache."""
    units = {
        s["entity_id"]: s.get("attributes", {}).get("unit_of_measurement", "")
        for s in states
        if s["entity_id"].startswith("sensor.")
    }
    try:
        with open(WATT_ENTITIES_PATH, "w") as f:
            json.dump(units, f)
    except Exception as e:
        log.warning("Could not cache watt entities: %s", e)


def _load_units_map() -> dict:
    try:
        with open(WATT_ENTITIES_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def find_phantom_loads(entity_baselines: dict = None, threshold_w: float = 2.0) -> list:
    """Find watt sensors that never drop to zero across all 24 hours."""
    if entity_baselines is None:
        if not os.path.exists(ENTITY_BASELINES_PATH):
            return []
        with open(ENTITY_BASELINES_PATH) as f:
            entity_baselines = json.load(f)
    if not entity_baselines:
        return []

    kwh_price = float(os.environ.get("HABITUS_KWH_PRICE", "0.20"))
    currency = os.environ.get("HABITUS_CURRENCY", "€")
    units_map = _load_units_map()
    results = []

    for eid, bl in entity_baselines.items():
        if not _is_watt_sensor(eid, units_map):
            continue

        # Collect mean power per hour
        hourly_means: dict[int, list[float]] = {}
        for key, val in bl.items():
            parts = key.split("_")
            if len(parts) < 2:
                continue
            try:
                h = int(parts[0])
            except ValueError:
                continue
            mean_val = val.get("mean", 0) if isinstance(val, dict) else float(val)
            hourly_means.setdefault(h, []).append(float(mean_val))

        if len(hourly_means) < 24:
            continue

        hour_avgs = {}
        for h in range(24):
            vals = hourly_means.get(h, [])
            if not vals:
                break
            hour_avgs[h] = sum(vals) / len(vals)

        if len(hour_avgs) < 24:
            continue

        min_power = min(hour_avgs.values())
        avg_power = sum(hour_avgs.values()) / 24

        # Skip: not always-on, or absurdly high (sensor error)
        if min_power < threshold_w:
            continue
        if avg_power > MAX_PHANTOM_W:
            log.debug("Skipping %s: avg %.1fW exceeds MAX_PHANTOM_W cap", eid, avg_power)
            continue

        kwh_year = avg_power * 8760 / 1000
        cost_year = round(kwh_year * kwh_price, 2)
        name = eid.split(".")[-1].replace("_", " ").title()

        results.append({
            "entity": eid,
            "name": name,
            "avg_phantom_w": round(avg_power, 1),
            "min_hourly_w": round(min_power, 1),
            "kwh_year": round(kwh_year, 1),
            "cost_year": cost_year,
            "currency": currency,
        })

    results.sort(key=lambda x: -x["avg_phantom_w"])
    return results


def _get_grid_annual_kwh() -> float | None:
    """Fetch actual annual grid consumption from HA long-term stats."""
    import requests as req
    import datetime, json as _json

    grid_entity = os.environ.get("HABITUS_ENERGY_GRID", "")
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    if not grid_entity or not token:
        return None
    try:
        end = datetime.datetime.now(datetime.timezone.utc)
        start = end - datetime.timedelta(days=365)
        r = req.get(
            f"{ha_url}/api/history/period/{start.isoformat()}",
            params={"filter_entity_id": grid_entity, "end_time": end.isoformat(), "minimal_response": "true"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not data[0]:
            return None
        states = [s for s in data[0] if s.get("state") not in ("unavailable", "unknown")]
        if len(states) < 2:
            return None
        first = float(states[0]["state"])
        last = float(states[-1]["state"])
        return max(0.0, last - first)
    except Exception as e:
        log.debug("Grid annual kWh lookup failed: %s", e)
        return None


def save(results: list) -> None:
    total_w = sum(r["avg_phantom_w"] for r in results)
    total_kwh = sum(r["kwh_year"] for r in results)
    kwh_price = float(os.environ.get("HABITUS_KWH_PRICE", "3.0"))
    currency = os.environ.get("HABITUS_CURRENCY", "kr")
    total_cost = round(total_kwh * kwh_price, 2)

    # Validate against actual grid consumption from Energy Dashboard
    grid_annual_kwh = _get_grid_annual_kwh()
    grid_annual_cost = round(grid_annual_kwh * kwh_price, 2) if grid_annual_kwh else None
    phantom_pct = round(100 * total_kwh / grid_annual_kwh, 1) if grid_annual_kwh and grid_annual_kwh > 0 else None

    # Sanity: if phantom total > grid total, something is wrong — cap and warn
    if grid_annual_kwh and total_kwh > grid_annual_kwh:
        log.warning(
            "Phantom kWh (%.0f) exceeds grid total (%.0f) — likely bad sensor data, capping",
            total_kwh, grid_annual_kwh,
        )
        # Rescale results proportionally
        scale = grid_annual_kwh * 0.5 / total_kwh  # assume max 50% could be phantom
        for r in results:
            r["kwh_year"] = round(r["kwh_year"] * scale, 1)
            r["cost_year"] = round(r["kwh_year"] * kwh_price, 2)
        total_kwh = sum(r["kwh_year"] for r in results)
        total_cost = round(total_kwh * kwh_price, 2)
        phantom_pct = round(100 * total_kwh / grid_annual_kwh, 1) if grid_annual_kwh else None

    payload = {
        "phantom_loads": results,
        "total_phantom_w": round(total_w, 1),
        "total_kwh_year": round(total_kwh, 1),
        "total_cost_year": total_cost,
        "currency": currency,
        "count": len(results),
        "grid_annual_kwh": round(grid_annual_kwh, 0) if grid_annual_kwh else None,
        "grid_annual_cost": grid_annual_cost,
        "phantom_pct_of_bill": phantom_pct,
    }
    try:
        with open(PHANTOM_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Phantom loads saved: %d devices found", len(results))
    except Exception as e:
        log.warning("Could not save phantom loads: %s", e)
    return payload


def load() -> dict:
    try:
        with open(PHANTOM_PATH) as f:
            return json.load(f)
    except Exception:
        return {}
