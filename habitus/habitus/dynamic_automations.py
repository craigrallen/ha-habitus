"""Dynamic automations that adapt as habits change.

Key capabilities:
- Detects behaviour drift: "evening routine shifted 30 min later this week"
- Calendar awareness: reads HA calendar entities for schedule context
- Adjusts automation timing based on ML-detected patterns
- Change point detection: spots when habits fundamentally shift
"""

import datetime
import json
import logging
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any

import numpy as np

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DYNAMIC_PATH = os.path.join(DATA_DIR, "dynamic_automations.json")
HA_DB = "/homeassistant/home-assistant_v2.db"

# Drift detection window
RECENT_DAYS = 7
BASELINE_DAYS = 30
# Minimum shift to report (minutes)
MIN_SHIFT_MINUTES = 15


def _get_routine_timings(entity_to_area: dict[str, str], days: int = 30) -> dict[str, list[dict]]:
    """Track when key events happen each day.

    Returns: {event_key: [{date, hour, minute, timestamp}, ...]}
    """
    if not os.path.exists(HA_DB):
        return {}

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    try:
        conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)
        rows = conn.execute("""
            SELECT sm.entity_id, s.state, s.last_changed_ts
            FROM states s
            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            WHERE s.last_changed_ts > ?
            AND (sm.entity_id LIKE 'light.%'
                 OR sm.entity_id LIKE 'switch.%'
                 OR sm.entity_id LIKE 'binary_sensor.%'
                 OR sm.entity_id LIKE 'media_player.%'
                 OR sm.entity_id LIKE 'person.%')
            AND s.state IN ('on', 'off', 'home', 'not_home', 'playing')
            ORDER BY s.last_changed_ts
        """, (cutoff_ts,)).fetchall()
        conn.close()
    except Exception as e:
        log.warning("dynamic: DB query failed: %s", e)
        return {}

    # Group by event key and day, keeping first occurrence per day
    daily_timings: dict[str, dict[str, dict]] = defaultdict(dict)

    for eid, state, ts in rows:
        event_key = f"{eid}:{state}"
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
        date_str = dt.strftime("%Y-%m-%d")

        if date_str not in daily_timings[event_key]:
            daily_timings[event_key][date_str] = {
                "date": date_str,
                "hour": dt.hour,
                "minute": dt.minute,
                "time_minutes": dt.hour * 60 + dt.minute,
                "timestamp": ts,
                "day_of_week": dt.weekday(),
            }

    # Convert to list format, filter to events with enough data
    result = {}
    for event_key, by_date in daily_timings.items():
        timings = sorted(by_date.values(), key=lambda t: t["date"])
        if len(timings) >= RECENT_DAYS:
            result[event_key] = timings

    return result


def _detect_timing_drift(timings: list[dict]) -> dict | None:
    """Detect if an event's timing has shifted recently vs baseline.

    Returns drift info if significant, None otherwise.
    """
    if len(timings) < RECENT_DAYS + 7:
        return None

    recent = timings[-RECENT_DAYS:]
    baseline = timings[:-RECENT_DAYS]

    recent_times = [t["time_minutes"] for t in recent]
    baseline_times = [t["time_minutes"] for t in baseline]

    recent_avg = np.mean(recent_times)
    baseline_avg = np.mean(baseline_times)
    shift_minutes = recent_avg - baseline_avg

    if abs(shift_minutes) < MIN_SHIFT_MINUTES:
        return None

    baseline_std = np.std(baseline_times) if len(baseline_times) > 1 else 30
    z_score = abs(shift_minutes) / max(baseline_std, 1)

    if z_score < 1.5:
        return None

    direction = "later" if shift_minutes > 0 else "earlier"

    return {
        "shift_minutes": round(shift_minutes),
        "direction": direction,
        "z_score": round(z_score, 2),
        "baseline_avg_time": f"{int(baseline_avg // 60):02d}:{int(baseline_avg % 60):02d}",
        "recent_avg_time": f"{int(recent_avg // 60):02d}:{int(recent_avg % 60):02d}",
        "confidence": min(95, round(z_score * 30)),
    }


def _get_calendar_events() -> list[dict]:
    """Read calendar events from HA calendar entities.

    Returns upcoming events that could affect automations.
    """
    import requests
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))

    if not token:
        return []

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        # Find calendar entities
        r = requests.post(
            f"{ha_url}/api/template",
            headers=headers,
            json={"template": "{{ states.calendar | map(attribute='entity_id') | list }}"},
            timeout=10,
        )
        if r.status_code != 200:
            return []

        calendar_ids = json.loads(r.text.replace("'", '"'))
        if not calendar_ids:
            return []

        # Get upcoming events
        now = datetime.datetime.now(datetime.UTC)
        end = now + datetime.timedelta(days=2)
        events = []

        for cal_id in calendar_ids[:5]:  # Cap at 5 calendars
            try:
                r = requests.get(
                    f"{ha_url}/api/calendars/{cal_id}",
                    headers=headers,
                    params={
                        "start": now.isoformat(),
                        "end": end.isoformat(),
                    },
                    timeout=10,
                )
                if r.status_code == 200:
                    for evt in r.json():
                        events.append({
                            "calendar": cal_id,
                            "summary": evt.get("summary", ""),
                            "start": evt.get("start", {}).get("dateTime", evt.get("start", {}).get("date", "")),
                            "end": evt.get("end", {}).get("dateTime", evt.get("end", {}).get("date", "")),
                        })
            except Exception:
                continue

        return events
    except Exception as e:
        log.debug("Calendar fetch failed: %s", e)
        return []


def run_dynamic_analysis(entity_to_area: dict[str, str], days: int = 30) -> dict[str, Any]:
    """Run full dynamic automation analysis."""
    timings = _get_routine_timings(entity_to_area, days=days)

    # Detect drift in all tracked events
    drifts = []
    for event_key, event_timings in timings.items():
        drift = _detect_timing_drift(event_timings)
        if drift:
            eid, state = event_key.rsplit(":", 1)
            name = eid.split(".")[-1].replace("_", " ").title()
            room = entity_to_area.get(eid, "")

            drift["event_key"] = event_key
            drift["entity_id"] = eid
            drift["state"] = state
            drift["name"] = name
            drift["room"] = room
            drift["description"] = (
                f"{name} ({state}) is happening {abs(drift['shift_minutes'])} min {drift['direction']} "
                f"this week (was {drift['baseline_avg_time']}, now {drift['recent_avg_time']})"
            )

            # Generate adaptive automation suggestion
            new_time = drift["recent_avg_time"]
            drift["suggested_automation"] = {
                "description": f"Shift automation for {name} to {new_time}",
                "yaml": f"""automation:
  alias: "Habitus Adaptive — {name} ({new_time})"
  description: "Auto-adjusted: {drift['description']}"
  trigger:
    - platform: time
      at: "{new_time}:00"
  action:
    - service: {eid.split('.')[0]}.turn_{state}
      target:
        entity_id: {eid}
""",
            }
            drifts.append(drift)

    drifts.sort(key=lambda d: -abs(d["shift_minutes"]))

    # Calendar context
    calendar_events = _get_calendar_events()
    calendar_insights = []
    for evt in calendar_events:
        start_str = evt.get("start", "")
        if "T" in start_str:
            try:
                start_dt = datetime.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if start_dt.hour < 8:
                    calendar_insights.append({
                        "event": evt["summary"],
                        "time": start_str,
                        "suggestion": f"Early event '{evt['summary']}' — consider shifting morning routine earlier",
                        "type": "early_start",
                    })
                elif start_dt.hour >= 20:
                    calendar_insights.append({
                        "event": evt["summary"],
                        "time": start_str,
                        "suggestion": f"Late event '{evt['summary']}' — evening routine may shift",
                        "type": "late_event",
                    })
            except (ValueError, TypeError):
                continue

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "tracked_events": len(timings),
        "detected_drifts": len(drifts),
        "drifts": drifts[:20],
        "calendar_insights": calendar_insights,
        "calendar_events_found": len(calendar_events),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DYNAMIC_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("dynamic: %d timing drifts detected, %d calendar insights", len(drifts), len(calendar_insights))
    return result
