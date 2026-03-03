"""Phantom Load Hunter — identify devices drawing power 24/7 unnecessarily."""

import json
import logging
import os

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PHANTOM_PATH = os.path.join(DATA_DIR, "phantom_loads.json")
ENTITY_BASELINES_PATH = os.path.join(DATA_DIR, "entity_baselines.json")


def find_phantom_loads(entity_baselines: dict = None, threshold_w: float = 2.0) -> list:
    """Find entities that never drop to zero across all 24 hours.

    Args:
        entity_baselines: Dict from entity_baselines.json.
            If None, loads from disk.
        threshold_w: Minimum watts across all hours to flag as phantom.

    Returns:
        List of phantom load dicts sorted by annual waste descending.
    """
    if entity_baselines is None:
        if not os.path.exists(ENTITY_BASELINES_PATH):
            return []
        with open(ENTITY_BASELINES_PATH) as f:
            entity_baselines = json.load(f)

    if not entity_baselines:
        return []

    kwh_price = float(os.environ.get("HABITUS_KWH_PRICE", "0.20"))
    results = []

    for eid, bl in entity_baselines.items():
        e = eid.lower()
        if not any(k in e for k in ("_w", "watt", "power", "consumed")):
            continue

        # Collect mean power per hour (across all days-of-week)
        hourly_means: dict[int, list[float]] = {}
        for key, val in bl.items():
            parts = key.split("_")
            if len(parts) < 2:
                continue
            try:
                h = int(parts[0])
            except ValueError:
                continue
            hourly_means.setdefault(h, []).append(val.get("mean", 0))

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
        if min_power < threshold_w:
            continue

        avg_phantom = sum(hour_avgs.values()) / 24
        kwh_year = avg_phantom * 8760 / 1000
        cost_year = kwh_year * kwh_price
        name = eid.split(".")[-1].replace("_", " ").title()

        results.append({
            "entity": eid,
            "name": name,
            "avg_phantom_w": round(avg_phantom, 1),
            "min_hourly_w": round(min_power, 1),
            "kwh_year": round(kwh_year, 1),
            "cost_year_eur": round(cost_year, 2),
        })

    results.sort(key=lambda x: x["kwh_year"], reverse=True)
    return results


def save(results: list) -> None:
    """Save phantom load results to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PHANTOM_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Phantom loads saved: %d devices found", len(results))


def load() -> list:
    """Load phantom load results from disk."""
    if not os.path.exists(PHANTOM_PATH):
        return []
    try:
        with open(PHANTOM_PATH) as f:
            return json.load(f)
    except Exception:
        return []
