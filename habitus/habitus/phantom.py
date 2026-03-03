"""Phantom Load Hunter — uses grid kWh meter to identify baseline idle consumption.

Approach:
  - Pull hourly grid kWh deltas from the meter (no W sensors needed)
  - Quiet hours (02:00–05:00) = minimum baseline = phantom drain floor
  - Compare week-over-week / month-over-month periods (no price math)
  - Per-device phantom = device's kWh during those same quiet hours
"""

import json
import logging
import os
import datetime

import requests as req

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PHANTOM_PATH = os.path.join(DATA_DIR, "phantom_loads.json")

# Hours considered "idle" — everyone asleep, no active usage
IDLE_HOURS = {2, 3, 4}


def _fetch_hourly_kwh(entity_id: str, days: int = 60) -> list[dict]:
    """Return list of {hour: datetime, kwh_delta: float} from HA long-term stats."""
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    if not entity_id or not token:
        return []
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=days)
    try:
        r = req.get(
            f"{ha_url}/api/history/period/{start.isoformat()}",
            params={"filter_entity_id": entity_id, "end_time": end.isoformat(), "minimal_response": "true"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if r.status_code != 200:
            log.warning("Grid history fetch failed: %s", r.status_code)
            return []
        data = r.json()
        if not data or not data[0]:
            return []
        states = [
            s for s in data[0]
            if s.get("state") not in ("unavailable", "unknown", None)
        ]
        # Build hourly deltas from cumulative kWh
        rows = []
        for i in range(1, len(states)):
            try:
                prev_v = float(states[i-1]["state"])
                curr_v = float(states[i]["state"])
                delta = curr_v - prev_v
                if delta < 0 or delta > 1000:  # skip resets or wild values
                    continue
                ts = states[i].get("last_changed") or states[i].get("last_updated", "")
                dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                rows.append({"dt": dt, "kwh_delta": round(delta, 4)})
            except Exception:
                continue
        return rows
    except Exception as e:
        log.warning("Grid kWh fetch error: %s", e)
        return []


def _period_totals(rows: list[dict]) -> dict:
    """Return kWh totals for: this_week, last_week, this_month, last_month."""
    now = datetime.datetime.now(datetime.timezone.utc)
    week_start = now - datetime.timedelta(days=now.weekday(), hours=now.hour)
    last_week_start = week_start - datetime.timedelta(weeks=1)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month_end = month_start
    last_month_start = (month_start - datetime.timedelta(days=1)).replace(day=1)

    buckets = {
        "this_week": (week_start, now),
        "last_week": (last_week_start, week_start),
        "this_month": (month_start, now),
        "last_month": (last_month_start, last_month_end),
    }
    totals = {}
    for label, (s, e) in buckets.items():
        total = sum(r["kwh_delta"] for r in rows if s <= r["dt"] < e)
        totals[label] = round(total, 2)
    return totals


def _phantom_baseline(rows: list[dict]) -> dict:
    """Phantom = average kWh/hour during idle hours (02-05).
    
    Returns avg idle kWh/hour and extrapolated annual kWh.
    """
    idle_rows = [r for r in rows if r["dt"].hour in IDLE_HOURS]
    if not idle_rows:
        return {"avg_idle_kwh_per_hour": None, "phantom_kwh_year": None}
    avg = sum(r["kwh_delta"] for r in idle_rows) / len(idle_rows)
    annual = avg * 8760
    return {
        "avg_idle_kwh_per_hour": round(avg, 3),
        "phantom_kwh_year": round(annual, 1),
        "idle_hours_sampled": len(idle_rows),
    }


def run() -> dict:
    """Main entry point — fetch grid kWh, compute periods + phantom baseline."""
    grid_entity = os.environ.get("HABITUS_ENERGY_GRID", "")
    if not grid_entity:
        log.info("No grid entity configured — skipping phantom analysis")
        return {}

    log.info("Phantom: fetching grid kWh from %s", grid_entity)
    rows = _fetch_hourly_kwh(grid_entity, days=60)
    if len(rows) < 48:
        log.info("Not enough grid data (%d rows) for phantom analysis", len(rows))
        return {"reason": "insufficient_data", "rows": len(rows)}

    periods = _period_totals(rows)
    phantom = _phantom_baseline(rows)

    # Week-over-week change
    wow_delta = None
    wow_pct = None
    if periods.get("this_week") is not None and periods.get("last_week"):
        wow_delta = round(periods["this_week"] - periods["last_week"], 2)
        wow_pct = round(100 * wow_delta / periods["last_week"], 1) if periods["last_week"] else None

    # Month-over-month change
    mom_delta = None
    mom_pct = None
    if periods.get("this_month") is not None and periods.get("last_month"):
        mom_delta = round(periods["this_month"] - periods["last_month"], 2)
        mom_pct = round(100 * mom_delta / periods["last_month"], 1) if periods["last_month"] else None

    result = {
        "grid_entity": grid_entity,
        "periods": periods,
        "wow_delta_kwh": wow_delta,
        "wow_pct": wow_pct,
        "mom_delta_kwh": mom_delta,
        "mom_pct": mom_pct,
        "phantom": phantom,
        "idle_hours": sorted(IDLE_HOURS),
        "data_points": len(rows),
        "analysed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return result


def save(result: dict) -> None:
    try:
        with open(PHANTOM_PATH, "w") as f:
            json.dump(result, f, indent=2)
        if result.get("phantom", {}).get("phantom_kwh_year"):
            log.info(
                "Phantom baseline: %.3f kWh/idle-hour → %.0f kWh/year",
                result["phantom"]["avg_idle_kwh_per_hour"],
                result["phantom"]["phantom_kwh_year"],
            )
    except Exception as e:
        log.warning("Could not save phantom data: %s", e)


def load() -> dict:
    try:
        with open(PHANTOM_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# Legacy compat shim
def find_phantom_loads(*args, **kwargs) -> list:
    return []


def cache_watt_entities(*args, **kwargs) -> None:
    pass
