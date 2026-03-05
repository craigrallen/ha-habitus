"""Pattern discovery and automation suggestion engine — v2.0."""

import datetime
import json
import logging
import os
from typing import Any

import pandas as pd

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PATTERNS_PATH = os.path.join(DATA_DIR, "patterns.json")
SUGGESTIONS_PATH = os.path.join(DATA_DIR, "suggestions.json")

NOTIFY = os.environ.get("HABITUS_NOTIFY_SERVICE", "notify.notify")
THRESHOLD = int(os.environ.get("HABITUS_ANOMALY_THRESHOLD", "70"))


def _has(stat_ids: list[str], *keywords: str) -> bool:
    """Check if any of the keywords appear in the tracked entity list."""
    joined = " ".join(stat_ids).lower()
    return any(k in joined for k in keywords)


def _pick_entity(
    stat_ids: list[str],
    domains: list[str] | tuple[str, ...],
    include_keywords: list[str] | tuple[str, ...] = (),
    exclude_keywords: list[str] | tuple[str, ...] = (),
) -> str | None:
    """Pick a best-effort entity from stat_ids by domain + keyword preference."""
    include = [k.lower() for k in include_keywords]
    exclude = [k.lower() for k in exclude_keywords]
    candidates: list[tuple[int, str]] = []
    for eid in stat_ids:
        if "." not in eid:
            continue
        domain, _obj = eid.split(".", 1)
        if domain not in domains:
            continue
        lower = eid.lower()
        if exclude and any(k in lower for k in exclude):
            continue
        score = 0
        if include:
            score += sum(4 for k in include if k in lower)
        if "living_room" in lower or "hallway" in lower:
            score += 2
        if "main" in lower or "total" in lower:
            score += 2
        candidates.append((score, eid))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates[0][1]


def _detect_profile(stat_ids: list[str]) -> dict[str, Any]:
    """Infer environment profile from detected entities/domains."""
    joined = " ".join(stat_ids).lower()
    boat_hits = sum(
        1
        for kw in (
            "bilge",
            "shore",
            "battery_soc",
            "house_battery",
            "inverter",
            "mastervolt",
            "mppt",
            "epever",
            "marine",
            "tank",
        )
        if kw in joined
    )
    home_hits = sum(
        1
        for kw in (
            "light.",
            "person.",
            "device_tracker",
            "binary_sensor",
            "climate.",
            "thermostat",
            "room",
            "door",
        )
        if kw in joined
    )
    profile = "boat" if boat_hits >= 2 and boat_hits >= home_hits else "home-default"
    return {
        "name": profile,
        "boat_signal": boat_hits,
        "home_signal": home_hits,
    }


def _max_consecutive_zeros(series: pd.Series) -> int:
    """Return the maximum number of consecutive zero-valued entries in a series.

    Args:
        series: Numeric Series (e.g. motion_events per hour).

    Returns:
        Integer count of the longest contiguous run of zero values.
    """
    max_run = 0
    current_run = 0
    for v in series:
        if v == 0:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run


def discover_patterns(features: pd.DataFrame) -> dict[str, Any]:
    patterns: dict[str, Any] = {}
    hourly = (
        features.groupby("hour_of_day")
        .agg(
            mean_power=("total_power_w", "mean"),
            std_power=("total_power_w", "std"),
            mean_temp=("avg_temp_c", "mean"),
            activity=("sensor_changes", "mean"),
        )
        .round(2)
    )

    night_power = float(hourly.loc[hourly.index.isin(range(1, 6)), "mean_power"].mean())
    if pd.isna(night_power):
        night_power = 50.0

    wakeup = next(
        (h for h in range(5, 12) if hourly.loc[h, "mean_power"] > night_power * 1.5), None
    )
    sleep_h = next(
        (h for h in range(23, 18, -1) if hourly.loc[h, "mean_power"] > night_power * 1.3), None
    )
    peak_hour = int(hourly["mean_power"].idxmax())
    peak_power = round(float(hourly["mean_power"].max()), 1)

    patterns["daily_routine"] = {
        "estimated_wakeup_hour": wakeup,
        "estimated_sleep_hour": sleep_h,
        "peak_usage_hour": peak_hour,
        "peak_usage_watts": peak_power,
        "night_baseline_watts": round(night_power, 1),
    }
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = (
        features.groupby("day_of_week")
        .agg(mean_power=("total_power_w", "mean"), activity=("sensor_changes", "mean"))
        .round(2)
    )
    patterns["weekly"] = {
        day_names[i]: {
            "mean_power_w": float(daily.loc[i, "mean_power"]),
            "activity": float(daily.loc[i, "activity"]),
        }
        for i in range(7)
        if i in daily.index
    }
    seasonal = (
        features.groupby("month")
        .agg(mean_power=("total_power_w", "mean"), mean_temp=("avg_temp_c", "mean"))
        .round(2)
    )
    mnames = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    patterns["seasonal"] = {
        mnames[m]: {
            "mean_power_w": float(seasonal.loc[m, "mean_power"]),
            "mean_temp_c": float(seasonal.loc[m, "mean_temp"]),
        }
        for m in seasonal.index
    }
    # ── MORNING LIGHTS PATTERN (weekday 06:00–07:59) ──────────────────────────
    morning_mask = (features["is_weekend"] == 0) & features["hour_of_day"].isin([6, 7])
    if morning_mask.any() and "lights_on" in features.columns:
        morning_lights_ratio = float((features.loc[morning_mask, "lights_on"] > 0).mean())
    else:
        morning_lights_ratio = 0.0
    patterns["morning_lights_pattern"] = {
        "lights_on_ratio": round(morning_lights_ratio, 3),
        "confidence": min(95, int(morning_lights_ratio * 100)),
    }

    # ── PEAK TARIFF POWER PATTERN (weekday 07:00–08:59, >800 W) ──────────────
    tariff_mask = (features["is_weekend"] == 0) & features["hour_of_day"].isin([7, 8])
    if tariff_mask.any():
        tariff_power = features.loc[tariff_mask, "total_power_w"]
        peak_tariff_ratio = float((tariff_power > 800).mean())
        mean_tariff_power = round(float(tariff_power.mean()), 1)
    else:
        peak_tariff_ratio = 0.0
        mean_tariff_power = 0.0
    patterns["peak_tariff_pattern"] = {
        "high_power_ratio": round(peak_tariff_ratio, 3),
        "mean_power_w": mean_tariff_power,
        "threshold_w": 800,
        "confidence": min(95, int(peak_tariff_ratio * 100)),
    }

    # ── VACANCY PATTERN (longest consecutive no-motion window) ────────────────
    if "motion_events" in features.columns:
        max_no_motion = _max_consecutive_zeros(features["motion_events"])
    else:
        max_no_motion = 0
    patterns["vacancy_pattern"] = {
        "max_no_motion_hours": max_no_motion,
        "extended_vacancy_detected": max_no_motion >= 12,
    }

    # ── BILGE TEMPERATURE BASELINE ─────────────────────────────────────────────
    temp_mean = round(float(features["avg_temp_c"].mean()), 2)
    temp_std = round(float(features["avg_temp_c"].std()), 2)
    patterns["bilge_temp_baseline"] = {
        "mean_c": temp_mean,
        "std_c": temp_std,
        "alert_threshold_c": round(temp_mean + 3.0, 2),
    }

    patterns["generated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    return patterns


def generate_suggestions(
    patterns: dict[str, Any], features: pd.DataFrame, stat_ids: list[str]
) -> list[dict[str, Any]]:
    suggestions = []
    routine = patterns.get("daily_routine", {})
    wakeup = routine.get("estimated_wakeup_hour")
    sleep_h = routine.get("estimated_sleep_hour")
    peak_h = routine.get("peak_usage_hour", 18)
    peak_w = routine.get("peak_usage_watts", 500)
    night_w = routine.get("night_baseline_watts", 100)

    has_bilge = _has(stat_ids, "bilge")
    has_battery = _has(stat_ids, "battery", "soc")
    has_shore = _has(stat_ids, "shore", "mains", "grid")
    has_solar = _has(stat_ids, "solar", "pv", "mppt", "epever", "scm")
    has_inverter = _has(stat_ids, "inverter", "mastervolt", "load")

    profile = _detect_profile(stat_ids)
    is_boat_profile = profile["name"] == "boat"

    motion_entity = _pick_entity(stat_ids, ["binary_sensor"], ["motion", "occupancy", "presence"])
    light_entity = _pick_entity(stat_ids, ["light"], ["living", "hall", "kitchen", "ceiling"])
    person_entity = _pick_entity(stat_ids, ["person", "device_tracker"])
    climate_entity = _pick_entity(stat_ids, ["climate"], ["heat", "thermostat", "hvac"])
    power_entity = _pick_entity(
        stat_ids,
        ["sensor"],
        ["total_load", "power", "consumption", "watt", "mastervolt", "load"],
        ["temperature", "humidity", "battery_soc"],
    ) or "sensor.mastervolt_total_load"

    # ── ROUTINE ───────────────────────────────────────────────────────────────
    if wakeup:
        suggestions.append(
            {
                "id": "morning_routine",
                "title": f"Morning Routine at {wakeup:02d}:00",
                "description": f"Power consistently rises above {int(night_w*1.5)}W around {wakeup:02d}:00 on weekdays — Habitus detected a recurring morning activity pattern.",
                "confidence": 85,
                "category": "routine",
                "applicable": True,
                "yaml": f"""automation:
  alias: "Habitus — Morning routine"
  description: "Auto-generated from observed {wakeup:02d}:00 wakeup pattern"
  trigger:
    - platform: time
      at: "{wakeup:02d}:00:00"
  condition:
    - condition: time
      weekday: [mon, tue, wed, thu, fri]
  action:
    - service: scene.turn_on
      target:
        entity_id: scene.morning  # replace with your scene""",
            }
        )

    if sleep_h:
        suggestions.append(
            {
                "id": "night_mode",
                "title": f"Night Mode at {sleep_h:02d}:00",
                "description": f"Power drops to near-baseline after {sleep_h:02d}:00 most nights. Suggest triggering night mode / reducing loads.",
                "confidence": 80,
                "category": "routine",
                "applicable": True,
                "yaml": f"""automation:
  alias: "Habitus — Night mode"
  description: "Auto-generated from observed {sleep_h:02d}:00 sleep pattern"
  trigger:
    - platform: time
      at: "{sleep_h:02d}:30:00"
  action:
    - service: scene.turn_on
      target:
        entity_id: scene.night  # replace with your scene""",
            }
        )

    suggestions.append(
        {
            "id": "weekend_mode",
            "title": "Weekend Mode",
            "description": "Weekend power profile differs significantly from weekdays — suggest separate scene/mode for Saturday/Sunday.",
            "confidence": 70,
            "category": "routine",
            "applicable": True,
            "yaml": """automation:
  alias: "Habitus — Weekend mode"
  trigger:
    - platform: time
      at: "09:00:00"
  condition:
    - condition: time
      weekday: [sat, sun]
  action:
    - service: scene.turn_on
      target:
        entity_id: scene.weekend  # replace with your scene""",
        }
    )

    if motion_entity and light_entity:
        suggestions.append(
            {
                "id": "occupancy_lights",
                "title": "Occupancy Lights",
                "description": "Motion/presence activity suggests an occupancy-based lighting routine for key rooms.",
                "confidence": 86,
                "category": "routine",
                "applicable": True,
                "entities": [motion_entity, light_entity],
                "yaml": f"""automation:
  alias: "Habitus — Occupancy lights"
  trigger:
    - platform: state
      entity_id: {motion_entity}
      to: "on"
  action:
    - service: light.turn_on
      target:
        entity_id: {light_entity}""",
            }
        )

    if person_entity:
        controllable = [
            e
            for e in stat_ids
            if e.startswith(("light.", "switch.", "media_player.", "climate.", "fan."))
        ][:8]
        if controllable:
            actions = "\n".join(
                [
                    "    - service: homeassistant.turn_off\n      target:\n        entity_id: " + eid
                    for eid in controllable
                ]
            )
            suggestions.append(
                {
                    "id": "away_mode",
                    "title": "Away Mode",
                    "description": "Presence patterns show clear home/away transitions. Turn off non-essential loads when leaving.",
                    "confidence": 84,
                    "category": "routine",
                    "applicable": True,
                    "entities": [person_entity] + controllable,
                    "yaml": f"""automation:
  alias: "Habitus — Away mode"
  trigger:
    - platform: state
      entity_id: {person_entity}
      to: "not_home"
      for:
        minutes: 10
  action:
{actions}""",
                }
            )

    if climate_entity and wakeup is not None:
        suggestions.append(
            {
                "id": "climate_preheat",
                "title": "Climate Preheat",
                "description": "Detected wake-up pattern and climate controls. Preheat shortly before normal morning activity.",
                "confidence": 82,
                "category": "energy",
                "applicable": True,
                "entities": [climate_entity],
                "yaml": f"""automation:
  alias: "Habitus — Climate preheat"
  trigger:
    - platform: time
      at: "{max(0, wakeup - 1):02d}:30:00"
  condition:
    - condition: time
      weekday: [mon, tue, wed, thu, fri]
  action:
    - service: climate.set_temperature
      target:
        entity_id: {climate_entity}
      data:
        temperature: 21""",
            }
        )

    # ── ENERGY ────────────────────────────────────────────────────────────────
    suggestions.append(
        {
            "id": "peak_power_alert",
            "title": f"High Power Alert (>{int(peak_w*1.3)}W)",
            "description": f"Peak normal usage is {int(peak_w)}W at {peak_h:02d}:00. This fires when load exceeds 130% of that for 10+ minutes.",
            "confidence": 90,
            "category": "energy",
            "applicable": has_inverter,
            "yaml": f"""automation:
  alias: "Habitus — High power alert"
  trigger:
    - platform: numeric_state
      entity_id: {power_entity}
      above: {int(peak_w*1.3)}
      for:
        minutes: 10
  action:
    - service: {NOTIFY}
      data:
        title: "⚡ High Power Usage"
        message: "Load has exceeded {int(peak_w*1.3)}W for 10+ minutes. Current: {{{{ states('{power_entity}') }}}}W" """,
        }
    )

    suggestions.append(
        {
            "id": "overnight_standby",
            "title": f"Overnight Standby Anomaly (>{int(night_w*1.4)}W)",
            "description": f"Overnight baseline is {int(night_w)}W. Fires between 01:00–05:00 if consumption rises above {int(night_w*1.4)}W — something unexpected is running.",
            "confidence": 85,
            "category": "energy",
            "applicable": True,
            "yaml": f"""automation:
  alias: "Habitus — Overnight power anomaly"
  trigger:
    - platform: numeric_state
      entity_id: {power_entity}
      above: {int(night_w*1.4)}
  condition:
    - condition: time
      after: "01:00:00"
      before: "05:00:00"
  action:
    - service: {NOTIFY}
      data:
        title: "🌙 Unusual Overnight Power"
        message: "Power is {{{{ states('{power_entity}') }}}}W at night — something may be left on." """,
        }
    )

    if has_solar and has_inverter:
        suggestions.append(
            {
                "id": "solar_export",
                "title": "Solar Surplus — Shift Loads",
                "description": "When solar production significantly exceeds current load, shift deferrable loads (water heating, charging) to maximise self-consumption.",
                "confidence": 78,
                "category": "energy",
                "applicable": True,
                "yaml": """automation:
  alias: "Habitus — Solar surplus load shifting"
  trigger:
    - platform: template
      value_template: >
        {{ (states('sensor.total_solar_production') | float(0)) >
           (states('sensor.mastervolt_total_load') | float(0)) * 1.3 }}
      for:
        minutes: 15
  action:
    - service: notify.notify
      data:
        title: "☀️ Solar Surplus"
        message: "Solar is generating more than you're using — good time to run high-load appliances." """,
            }
        )

    # ── BOAT / MARINE ─────────────────────────────────────────────────────────
    if has_battery:
        suggestions.append(
            {
                "id": "battery_protection",
                "title": "Battery SOC Protection Alert",
                "description": "Detected battery monitoring entities. Alert when state of charge drops to a critical level to prevent deep discharge.",
                "confidence": 95,
                "category": "boat",
                "applicable": True,
                "yaml": """automation:
  alias: "Habitus — Battery low alert"
  trigger:
    - platform: numeric_state
      entity_id: sensor.house_battery_energy_watts
      below: -500  # discharging at >500W and battery low
  condition:
    - condition: numeric_state
      entity_id: sensor.house_battery_soc  # adjust entity
      below: 20
  action:
    - service: """
                + NOTIFY
                + """
      data:
        title: "🔋 Battery Low"
        message: "House battery SOC is below 20%. Connect shore power or reduce loads." """,
            }
        )

    if has_bilge:
        suggestions.append(
            {
                "id": "bilge_anomaly",
                "title": "Bilge Pump Anomaly Alert",
                "description": "Bilge sensors detected. Alert if the bilge pump runs unexpectedly or bilge temperature spikes, which can indicate a leak or equipment issue.",
                "confidence": 95,
                "category": "boat",
                "applicable": True,
                "yaml": """automation:
  alias: "Habitus — Bilge anomaly"
  trigger:
    - platform: state
      entity_id: binary_sensor.bilge_pump_running  # adjust entity
      to: "on"
      for:
        minutes: 5
  action:
    - service: """
                + NOTIFY
                + """
      data:
        title: "⚠️ Bilge Pump Running"
        message: "Bilge pump has been running for 5+ minutes. Check for water ingress." """,
            }
        )

    if has_shore:
        suggestions.append(
            {
                "id": "shore_power_loss",
                "title": "Shore Power Loss Alert",
                "description": "Shore power entities detected. Alert immediately when shore power is lost so you can switch to battery/generator before discharge.",
                "confidence": 92,
                "category": "boat",
                "applicable": True,
                "yaml": """automation:
  alias: "Habitus — Shore power lost"
  trigger:
    - platform: numeric_state
      entity_id: sensor.shore_power_smart_meter_electric_consumption_w
      below: 10
      for:
        minutes: 2
  action:
    - service: """
                + NOTIFY
                + """
      data:
        title: "🔌 Shore Power Lost"
        message: "Shore power appears to have been disconnected. Running on battery." """,
            }
        )

    if has_inverter and has_solar:
        suggestions.append(
            {
                "id": "inverter_overload",
                "title": "Inverter Overload Predictor",
                "description": "Alert when total load is approaching inverter capacity limits, giving time to shed loads before an overload trip.",
                "confidence": 82,
                "category": "boat",
                "applicable": True,
                "yaml": f"""automation:
  alias: "Habitus — Inverter approaching overload"
  trigger:
    - platform: numeric_state
      entity_id: sensor.mastervolt_total_load
      above: {int(peak_w*1.5)}
      for:
        minutes: 3
  action:
    - service: {NOTIFY}
      data:
        title: "⚡ High Load Warning"
        message: "Load is {{{{ states('sensor.mastervolt_total_load') }}}}W — approaching inverter limits. Consider shedding loads." """,
            }
        )

    if is_boat_profile and person_entity:
        suggestions.append(
            {
                "id": "harbor_mode",
                "title": "Harbor Mode (Away Profile)",
                "description": "Automatically reduce non-essential loads when no presence is detected for extended periods — keeps standby power minimal while away.",
                "confidence": 72,
                "category": "boat",
                "applicable": True,
                "entities": [person_entity],
                "yaml": f"""automation:
  alias: "Habitus — Harbor mode"
  description: "Activates low-power profile when away for >2h"
  trigger:
    - platform: state
      entity_id: {person_entity}
      to: "not_home"
      for:
        hours: 2
  action:
    - service: scene.turn_on
      target:
        entity_id: scene.harbor_mode  # create this scene""",
            }
        )

    # ── ANOMALY ───────────────────────────────────────────────────────────────
    suggestions.append(
        {
            "id": "anomaly_alert",
            "title": f"Habitus Anomaly Alert (Score >{THRESHOLD})",
            "description": f"Send a notification when Habitus detects unusual behaviour scoring above {THRESHOLD}/100 for 5+ minutes.",
            "confidence": 95,
            "category": "anomaly",
            "applicable": True,
            "yaml": f"""automation:
  alias: "Habitus — Anomaly alert"
  trigger:
    - platform: state
      entity_id: binary_sensor.habitus_anomaly_detected
      to: "on"
      for:
        minutes: 5
  action:
    - service: {NOTIFY}
      data:
        title: "🧠 Habitus — Unusual Activity"
        message: >
          Habitus detected unusual home behaviour.
          Score: {{{{ states('sensor.habitus_anomaly_score') }}}}/100.
          Trained on {{{{ states('sensor.habitus_training_days') }}}} days of history.""",
        }
    )

    suggestions.append(
        {
            "id": "sensor_watchdog",
            "title": "Sensor Watchdog",
            "description": "Alert when a key sensor goes unavailable for more than 1 hour — catches connectivity issues, battery failures, or hardware faults early.",
            "confidence": 80,
            "category": "anomaly",
            "applicable": True,
            "yaml": """automation:
  alias: "Habitus — Sensor unavailable watchdog"
  trigger:
    - platform: state
      entity_id:
        - sensor.habitus_anomaly_score  # add your critical sensors
      to: "unavailable"
      for:
        hours: 1
  action:
    - service: """
            + NOTIFY
            + """
      data:
        title: "📡 Sensor Offline"
        message: "{{ trigger.entity_id }} has been unavailable for over 1 hour." """,
        }
    )

    suggestions.append(
        {
            "id": "daily_digest",
            "title": "Daily Energy Digest",
            "description": "Receive a morning summary of yesterday's energy usage, anomalies detected, and today's solar forecast.",
            "confidence": 88,
            "category": "anomaly",
            "applicable": True,
            "yaml": f"""automation:
  alias: "Habitus — Daily energy digest"
  trigger:
    - platform: time
      at: "08:00:00"
  action:
    - service: {NOTIFY}
      data:
        title: "📊 Daily Energy Digest"
        message: >
          Good morning! Habitus report:
          Anomaly score: {{{{ states('sensor.habitus_anomaly_score') }}}}/100
          Tracking {{{{ states('sensor.habitus_entity_count') }}}} sensors
          Model trained on {{{{ states('sensor.habitus_training_days') }}}} days of history.""",
        }
    )

    # ── DATA-DRIVEN PATTERNS (TASK-001) ───────────────────────────────────────
    morning_p = patterns.get("morning_lights_pattern", {})
    morning_ratio = morning_p.get("lights_on_ratio", 0.0)
    morning_conf = morning_p.get("confidence", int(morning_ratio * 100))
    suggestions.append(
        {
            "id": "morning_lights",
            "title": "Morning Lights Routine (06:30–07:15 weekdays)",
            "description": (
                f"Lights active in {morning_ratio:.0%} of observed weekday mornings "
                f"(06:00–08:00 window). Habitus suggests automating this routine."
            ),
            "confidence": morning_conf,
            "category": "routine",
            "applicable": morning_ratio > 0.3,
            "yaml": f"""automation:
  alias: "Habitus — Morning lights"
  description: "Observed in {morning_ratio:.0%} of weekday mornings"
  trigger:
    - platform: time
      at: "06:30:00"
  condition:
    - condition: time
      weekday: [mon, tue, wed, thu, fri]
  action:
    - service: light.turn_on
      target:
        area_id: living_room  # adjust to your area""",
        }
    )

    tariff_p = patterns.get("peak_tariff_pattern", {})
    tariff_ratio = tariff_p.get("high_power_ratio", 0.0)
    tariff_mean = tariff_p.get("mean_power_w", 0.0)
    tariff_conf = tariff_p.get("confidence", int(tariff_ratio * 100))
    suggestions.append(
        {
            "id": "peak_tariff_alert",
            "title": "Peak Tariff High Usage Alert (07:00–08:30 weekdays)",
            "description": (
                f"Power exceeded 800W during peak tariff hours in {tariff_ratio:.0%} of weekday "
                f"mornings. Mean peak load: {tariff_mean:.0f}W. Consider shifting loads earlier."
            ),
            "confidence": tariff_conf,
            "category": "energy",
            "applicable": tariff_ratio > 0.2,
            "yaml": f"""automation:
  alias: "Habitus — Peak tariff alert"
  trigger:
    - platform: numeric_state
      entity_id: {power_entity}
      above: 800
  condition:
    - condition: time
      after: "07:00:00"
      before: "08:30:00"
      weekday: [mon, tue, wed, thu, fri]
  action:
    - service: {NOTIFY}
      data:
        title: "\u26a1 Peak Tariff \u2014 High Usage"
        message: >
          Power is {{{{ states('{power_entity}') }}}}W during peak tariff hours
          (07:00\u201308:30). Consider delaying high-load appliances.""",
        }
    )

    vacancy_p = patterns.get("vacancy_pattern", {})
    max_no_motion = vacancy_p.get("max_no_motion_hours", 0)
    extended = vacancy_p.get("extended_vacancy_detected", False)
    vacancy_conf = min(90, int(max_no_motion / 24 * 80)) if extended else 40
    suggestions.append(
        {
            "id": "vacancy_security",
            "title": "Extended Vacancy Security Alert (No Motion >24h)",
            "description": (
                f"Longest unoccupied window detected: {max_no_motion}h. "
                "Alert if no motion in living areas for over 24 hours — possible security or "
                "sensor issue."
            ),
            "confidence": vacancy_conf,
            "category": "anomaly",
            "applicable": True,
            "yaml": f"""automation:
  alias: "Habitus \u2014 Extended vacancy alert"
  trigger:
    - platform: state
      entity_id: binary_sensor.hallway_motion  # adjust to your motion sensor
      to: "off"
      for:
        hours: 24
  action:
    - service: {NOTIFY}
      data:
        title: "\U0001f3e0 Extended Vacancy Detected"
        message: "No motion detected for 24+ hours (longest observed gap: {max_no_motion}h). Is everything okay?" """,
        }
    )

    bilge_p = patterns.get("bilge_temp_baseline", {})
    bilge_mean = bilge_p.get("mean_c", 12.0)
    bilge_threshold = bilge_p.get("alert_threshold_c", round(bilge_mean + 3.0, 2))
    if has_bilge:
        suggestions.append(
            {
                "id": "bilge_temp_anomaly",
                "title": f"Bilge Temperature Anomaly (>{bilge_threshold:.1f}\u00b0C)",
                "description": (
                    f"Bilge temp baseline: {bilge_mean:.1f}\u00b0C. Alert when bilge rises 3\u00b0C "
                    "above baseline — may indicate engine heat, poor ventilation, or fire risk."
                ),
                "confidence": 92,
                "category": "boat",
                "applicable": True,
                "yaml": f"""automation:
  alias: "Habitus \u2014 Bilge temperature anomaly"
  trigger:
    - platform: numeric_state
      entity_id: sensor.bilge_sensor_air_temperature  # adjust entity
      above: {bilge_threshold:.1f}
  action:
    - service: {NOTIFY}
      data:
        title: "\U0001f321 Bilge Temperature Alert"
        message: >
          Bilge temperature has exceeded {bilge_threshold:.1f}\u00b0C
          (baseline: {bilge_mean:.1f}\u00b0C). Check for heat source or ventilation issue.""",
            }
        )

    if has_shore and has_battery:
        suggestions.append(
            {
                "id": "shore_power_battery",
                "title": "Shore Power Loss with Low Battery Alert",
                "description": (
                    "Shore power entities and battery monitoring detected. Fires when shore power "
                    "drops near zero while battery SOC is below 40% — critical combined alert."
                ),
                "confidence": 95,
                "category": "boat",
                "applicable": True,
                "yaml": """automation:
  alias: "Habitus \u2014 Shore power lost with low battery"
  trigger:
    - platform: numeric_state
      entity_id: sensor.shore_power_smart_meter_electric_consumption_w  # adjust entity
      below: 10
      for:
        minutes: 2
  condition:
    - condition: numeric_state
      entity_id: sensor.house_battery_soc  # adjust entity
      below: 40
  action:
    - service: """
                + NOTIFY
                + """
      data:
        title: "\U0001f50c\U0001f50b Shore Power Lost \u2014 Battery Low"
        message: >
          Shore power disconnected and battery SOC is below 40%.
          Connect shore power or start generator immediately.""",
            }
        )

    # ── LOVELACE ──────────────────────────────────────────────────────────────
    suggestions.append(
        {
            "id": "lovelace_card",
            "title": "Lovelace Insights Card",
            "description": "Ready-to-paste Lovelace YAML for a Habitus insights card on any dashboard.",
            "confidence": 100,
            "category": "lovelace",
            "applicable": True,
            "yaml": """type: vertical-stack
cards:
  - type: horizontal-stack
    cards:
      - type: gauge
        entity: sensor.habitus_anomaly_score
        name: Anomaly Score
        min: 0
        max: 100
        needle: true
        segments:
          - from: 0
            color: "#4caf50"
          - from: 40
            color: "#ffb300"
          - from: 70
            color: "#f44336"
      - type: entity
        entity: binary_sensor.habitus_anomaly_detected
        name: Status
      - type: stat
        entity: sensor.habitus_training_days
        name: Training Days
      - type: stat
        entity: sensor.habitus_entity_count
        name: Sensors""",
        }
    )

    for s in suggestions:
        category = s.get("category", "")
        base_conf = int(s.get("confidence", 0))
        profile_boost = 0
        if profile["name"] == "home-default":
            if category in ("routine", "energy", "anomaly"):
                profile_boost += 10
            if category == "boat":
                profile_boost -= 35
                s["applicable"] = False
        elif profile["name"] == "boat" and category == "boat":
            profile_boost += 12

        s["profile"] = profile["name"]
        s["profile_boost"] = profile_boost
        s["rank_score"] = max(0, min(100, base_conf + profile_boost))
        s["generated_at"] = datetime.datetime.now(datetime.UTC).isoformat()

    suggestions.sort(key=lambda x: x.get("rank_score", x.get("confidence", 0)), reverse=True)
    return suggestions


def run(
    features: pd.DataFrame, stat_ids: list[str] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Discover patterns and write suggestions.json + patterns.json to DATA_DIR.

    Args:
        features: Hourly feature matrix from ``build_features``.
        stat_ids: Entity IDs tracked by this installation.

    Returns:
        Tuple of (patterns dict, suggestions list).
    """
    stat_ids = stat_ids or []
    patterns = discover_patterns(features)
    suggestions = generate_suggestions(patterns, features, stat_ids)
    with open(PATTERNS_PATH, "w") as f:
        json.dump(patterns, f, indent=2, default=str)
    with open(SUGGESTIONS_PATH, "w") as f:
        json.dump(suggestions, f, indent=2, default=str)
    log.info(
        f"Patterns saved — {len(suggestions)} suggestions ({sum(1 for s in suggestions if s['applicable'])} applicable)"
    )
    return patterns, suggestions
