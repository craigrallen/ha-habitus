"""Phantom Load Hunter — uses HA Energy Dashboard statistics directly.

Uses recorder/statistics_during_period WebSocket API to get the same
monthly/hourly data the Energy Dashboard displays.
"""

import json
import logging
import os
import datetime
import asyncio

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PHANTOM_PATH = os.path.join(DATA_DIR, "phantom_loads.json")

# Hours considered "idle" — everyone asleep, no active usage
IDLE_HOURS = {2, 3, 4}


async def _fetch_statistics(entity_id: str, period: str = "month", months: int = 12) -> list[dict]:
    """Fetch statistics from HA using WebSocket API (same as Energy Dashboard)."""
    import websockets
    
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    if not entity_id or not token:
        return []
    
    ws_url = ha_url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=months * 31)
    
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "auth", "access_token": token}))
            auth_result = json.loads(await ws.recv())
            if auth_result.get("type") != "auth_ok":
                log.warning("WebSocket auth failed")
                return []
            
            await ws.send(json.dumps({
                "id": 1,
                "type": "recorder/statistics_during_period",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "statistic_ids": [entity_id],
                "period": period
            }))
            result = json.loads(await ws.recv())
            return result.get("result", {}).get(entity_id, [])
    except Exception as e:
        log.warning("Statistics fetch failed: %s", e)
        return []


async def _fetch_hourly_statistics(entity_id: str, days: int = 60) -> list[dict]:
    """Fetch hourly statistics for idle-hour analysis."""
    import websockets
    
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    if not entity_id or not token:
        return []
    
    ws_url = ha_url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=days)
    
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "auth", "access_token": token}))
            auth_result = json.loads(await ws.recv())
            if auth_result.get("type") != "auth_ok":
                return []
            
            await ws.send(json.dumps({
                "id": 1,
                "type": "recorder/statistics_during_period",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "statistic_ids": [entity_id],
                "period": "hour"
            }))
            result = json.loads(await ws.recv())
            return result.get("result", {}).get(entity_id, [])
    except Exception as e:
        log.warning("Hourly statistics fetch failed: %s", e)
        return []


def run() -> dict:
    """Main entry point — fetch stats from Energy Dashboard, compute phantom baseline."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # Already in an async context (Flask) — create a new thread with its own loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run_async())
            return future.result(timeout=60)
    else:
        return asyncio.run(_run_async())


async def _run_async() -> dict:
    grid_entity = os.environ.get("HABITUS_ENERGY_GRID", "")
    if not grid_entity:
        log.info("No grid entity configured — skipping phantom analysis")
        return {}

    log.info("Phantom: fetching Energy Dashboard stats for %s", grid_entity)
    
    # Get monthly stats (same data as Energy Dashboard shows)
    monthly = await _fetch_statistics(grid_entity, "month", 12)
    if not monthly:
        log.info("No monthly statistics available")
        return {"reason": "no_monthly_stats"}
    
    # Build monthly breakdown
    months_data = []
    for m in monthly:
        ts = datetime.datetime.fromtimestamp(m["start"] / 1000, datetime.timezone.utc)
        months_data.append({
            "month": ts.strftime("%Y-%m"),
            "kwh": round(m.get("change", 0), 1)
        })
    
    total_12mo = sum(m.get("change", 0) for m in monthly)
    
    # Period comparisons
    now = datetime.datetime.now(datetime.timezone.utc)
    current_month = now.strftime("%Y-%m")
    last_month = (now.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m")
    
    this_month_kwh = next((m["kwh"] for m in months_data if m["month"] == current_month), 0)
    last_month_kwh = next((m["kwh"] for m in months_data if m["month"] == last_month), 0)
    
    # Get hourly data for overnight baseline (last 30 days only — recent behavior)
    hourly = await _fetch_hourly_statistics(grid_entity, 30)
    phantom_info = {}
    if hourly:
        # Filter to idle hours and calculate average
        idle_changes = []
        for h in hourly:
            ts = datetime.datetime.fromtimestamp(h["start"] / 1000, datetime.timezone.utc)
            if ts.hour in IDLE_HOURS:
                change = h.get("change", 0)
                if change is not None and 0 <= change < 10:  # Sanity: max 10 kWh/hour
                    idle_changes.append(change)
        
        if idle_changes:
            avg_idle = sum(idle_changes) / len(idle_changes)
            phantom_info = {
                "avg_idle_kwh_per_hour": round(avg_idle, 3),
                "overnight_kwh_year": round(avg_idle * 8760, 0),
                "idle_hours_sampled": len(idle_changes),
            }
            log.info("Overnight baseline: %.3f kWh/idle-hour → %.0f kWh/year", avg_idle, avg_idle * 8760)
    
    result = {
        "grid_entity": grid_entity,
        "total_12mo_kwh": round(total_12mo, 1),
        "months": months_data,
        "this_month_kwh": this_month_kwh,
        "last_month_kwh": last_month_kwh,
        "mom_delta_kwh": round(this_month_kwh - last_month_kwh, 1) if last_month_kwh else None,
        "mom_pct": round(100 * (this_month_kwh - last_month_kwh) / last_month_kwh, 1) if last_month_kwh else None,
        "overnight_baseline": phantom_info,
        "idle_hours": sorted(IDLE_HOURS),
        "analysed_at": now.isoformat(),
    }
    return result


def save(result: dict) -> None:
    try:
        with open(PHANTOM_PATH, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        log.warning("Could not save phantom data: %s", e)


def load() -> dict:
    try:
        with open(PHANTOM_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# Legacy compat
def find_phantom_loads(*args, **kwargs) -> list:
    return []

def cache_watt_entities(*args, **kwargs) -> None:
    pass
