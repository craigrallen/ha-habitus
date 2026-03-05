"""Seasonal Adaptation — generate season-specific automation suggestions.

Detects current season based on date and hemisphere.
Compares current usage patterns vs same-season history.
Generates season-specific suggestions with YAML.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

SEASONAL_SUGGESTIONS_PATH = os.path.join(DATA_DIR, "seasonal_suggestions.json")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")

HEMISPHERE = os.environ.get("HEMISPHERE", "north")


def get_current_season(date: datetime.date | None = None, hemisphere: str = "north") -> str:
    """Determine current season from date and hemisphere.

    Args:
        date: Date to check. Defaults to today.
        hemisphere: 'north' or 'south'.

    Returns:
        One of: 'winter', 'spring', 'summer', 'autumn'
    """
    if date is None:
        date = datetime.date.today()

    month = date.month

    if hemisphere.lower() == "south":
        # Southern hemisphere: seasons reversed
        if month in (12, 1, 2):
            return "summer"
        elif month in (3, 4, 5):
            return "autumn"
        elif month in (6, 7, 8):
            return "winter"
        else:
            return "spring"
    else:
        # Northern hemisphere
        if month in (12, 1, 2):
            return "winter"
        elif month in (3, 4, 5):
            return "spring"
        elif month in (6, 7, 8):
            return "summer"
        else:
            return "autumn"


def _generate_winter_suggestions(entities: list[str]) -> list[dict[str, Any]]:
    """Generate winter-specific automation suggestions."""
    suggestions = []

    # Heating preheat
    suggestions.append({
        "title": "Pre-heat home before morning",
        "seasonal_reason": "Winter mornings are cold — pre-heat saves comfort and energy vs reactive heating",
        "confidence": 0.85,
        "season": "winter",
        "generated_yaml": """automation:
  alias: "Habitus Seasonal — Winter morning pre-heat"
  description: "Pre-heat home 30 minutes before typical wake time in winter"
  trigger:
    - platform: time
      at: "06:30:00"
  condition:
    - condition: numeric_state
      entity_id: sensor.outdoor_temperature
      below: 5
  action:
    - service: climate.set_temperature
      target:
        entity_id: climate.living_room
      data:
        temperature: 21
    - service: notify.notify
      data:
        title: "🌡️ Habitus"
        message: "Pre-heating for cold morning (outdoor temp below 5°C)"
""",
    })

    # Early evening lighting
    suggestions.append({
        "title": "Activate evening lights earlier in winter",
        "seasonal_reason": "Winter days are short — sunset is earlier, so lights should come on sooner",
        "confidence": 0.80,
        "season": "winter",
        "generated_yaml": """automation:
  alias: "Habitus Seasonal — Winter early evening lights"
  description: "Turn on main lights at sunset in winter (earlier than summer)"
  trigger:
    - platform: sun
      event: sunset
      offset: "-00:30:00"
  condition:
    - condition: template
      value_template: "{{ now().month in [11, 12, 1, 2] }}"
  action:
    - service: light.turn_on
      target:
        entity_id: light.living_room
      data:
        brightness_pct: 80
        color_temp: 350
""",
    })

    # Frost alert
    suggestions.append({
        "title": "Frost alert automation",
        "seasonal_reason": "Protect pipes and plants from frost during winter nights",
        "confidence": 0.75,
        "season": "winter",
        "generated_yaml": """automation:
  alias: "Habitus Seasonal — Frost alert"
  description: "Alert when outdoor temperature approaches freezing"
  trigger:
    - platform: numeric_state
      entity_id: sensor.outdoor_temperature
      below: 2
  action:
    - service: notify.notify
      data:
        title: "🥶 Frost Warning"
        message: "Outdoor temperature is near freezing. Check pipes and outdoor plants."
""",
    })

    return suggestions


def _generate_summer_suggestions(entities: list[str]) -> list[dict[str, Any]]:
    """Generate summer-specific automation suggestions."""
    suggestions = []

    # Cooling/fan automation
    suggestions.append({
        "title": "Cooling automation for hot days",
        "seasonal_reason": "Summer heat peaks in the afternoon — cool before it gets too warm",
        "confidence": 0.82,
        "season": "summer",
        "generated_yaml": """automation:
  alias: "Habitus Seasonal — Summer cooling"
  description: "Activate cooling when indoor temperature exceeds threshold in summer"
  trigger:
    - platform: numeric_state
      entity_id: sensor.living_room_temperature
      above: 25
  condition:
    - condition: template
      value_template: "{{ now().month in [6, 7, 8] }}"
  action:
    - service: climate.set_temperature
      target:
        entity_id: climate.living_room
      data:
        hvac_mode: cool
        temperature: 23
""",
    })

    # Solar peak shift
    suggestions.append({
        "title": "Shift high-power tasks to solar peak hours",
        "seasonal_reason": "Summer solar production peaks midday — run appliances then to offset costs",
        "confidence": 0.78,
        "season": "summer",
        "generated_yaml": """automation:
  alias: "Habitus Seasonal — Solar peak appliance scheduling"
  description: "Notify to run dishwasher/washer during peak solar hours"
  trigger:
    - platform: time
      at: "11:00:00"
  condition:
    - condition: template
      value_template: "{{ now().month in [5, 6, 7, 8, 9] }}"
  action:
    - service: notify.notify
      data:
        title: "☀️ Solar Peak"
        message: "Good time to run high-power appliances (solar production peak)"
""",
    })

    # Late evening lighting
    suggestions.append({
        "title": "Later evening light schedule in summer",
        "seasonal_reason": "Summer evenings stay bright longer — delay automatic lights",
        "confidence": 0.77,
        "season": "summer",
        "generated_yaml": """automation:
  alias: "Habitus Seasonal — Summer late evening lights"
  description: "Only turn on main lights after true sunset in summer"
  trigger:
    - platform: sun
      event: sunset
      offset: "00:15:00"
  condition:
    - condition: template
      value_template: "{{ now().month in [5, 6, 7, 8] }}"
  action:
    - service: light.turn_on
      target:
        entity_id: light.living_room
      data:
        brightness_pct: 60
""",
    })

    return suggestions


def _generate_spring_suggestions(entities: list[str]) -> list[dict[str, Any]]:
    """Generate spring-specific automation suggestions."""
    return [
        {
            "title": "Turn off heating schedule (spring transition)",
            "seasonal_reason": "Spring temperatures rise — scheduled heating often no longer needed",
            "confidence": 0.72,
            "season": "spring",
            "generated_yaml": """automation:
  alias: "Habitus Seasonal — Spring heating off"
  description: "Disable scheduled heating when outdoor temp consistently above 15°C"
  trigger:
    - platform: numeric_state
      entity_id: sensor.outdoor_temperature
      above: 15
      for: "02:00:00"
  condition:
    - condition: template
      value_template: "{{ now().month in [3, 4, 5] }}"
  action:
    - service: climate.turn_off
      target:
        entity_id: climate.living_room
    - service: notify.notify
      data:
        title: "🌱 Spring Mode"
        message: "Outdoor temp > 15°C. Heating turned off automatically."
""",
        },
        {
            "title": "Window open detection — pause climate control",
            "seasonal_reason": "Spring is window-opening season — avoid wasting energy with open windows",
            "confidence": 0.80,
            "season": "spring",
            "generated_yaml": """automation:
  alias: "Habitus Seasonal — Window open climate pause"
  description: "Pause heating/cooling when windows are opened in spring/autumn"
  trigger:
    - platform: state
      entity_id: binary_sensor.living_room_window
      to: "on"
      for: "00:05:00"
  action:
    - service: climate.turn_off
      target:
        entity_id: climate.living_room
    - service: notify.notify
      data:
        title: "🪟 Window Open"
        message: "Climate control paused — window has been open for 5+ minutes"
""",
        },
    ]


def _generate_autumn_suggestions(entities: list[str]) -> list[dict[str, Any]]:
    """Generate autumn-specific automation suggestions."""
    return [
        {
            "title": "Resume heating schedule (autumn transition)",
            "seasonal_reason": "Autumn temperatures drop — time to re-enable scheduled heating",
            "confidence": 0.75,
            "season": "autumn",
            "generated_yaml": """automation:
  alias: "Habitus Seasonal — Autumn heating on"
  description: "Re-enable heating schedule when outdoor temp drops below 12°C"
  trigger:
    - platform: numeric_state
      entity_id: sensor.outdoor_temperature
      below: 12
      for: "03:00:00"
  condition:
    - condition: template
      value_template: "{{ now().month in [9, 10, 11] }}"
  action:
    - service: climate.set_temperature
      target:
        entity_id: climate.living_room
      data:
        temperature: 20
    - service: notify.notify
      data:
        title: "🍂 Autumn Mode"
        message: "Outdoor temp < 12°C. Heating schedule re-enabled."
""",
        },
        {
            "title": "Earlier evening lights (days getting shorter)",
            "seasonal_reason": "Autumn sunsets come earlier each week — adjust lighting schedule",
            "confidence": 0.78,
            "season": "autumn",
            "generated_yaml": """automation:
  alias: "Habitus Seasonal — Autumn evening lights"
  trigger:
    - platform: sun
      event: sunset
      offset: "-00:15:00"
  condition:
    - condition: template
      value_template: "{{ now().month in [9, 10, 11] }}"
  action:
    - service: light.turn_on
      target:
        entity_id: light.living_room
      data:
        brightness_pct: 70
""",
        },
    ]


def get_seasonal_suggestions(
    date: datetime.date | None = None,
    entities: list[str] | None = None,
    hemisphere: str | None = None,
) -> dict[str, Any]:
    """Get season-appropriate automation suggestions.

    Args:
        date: Date for season detection. Defaults to today.
        entities: Available entity IDs for personalization.
        hemisphere: 'north' or 'south'. Defaults to HEMISPHERE env var.

    Returns:
        Dict with season, suggestions list.
    """
    if hemisphere is None:
        hemisphere = HEMISPHERE
    if entities is None:
        entities = []

    season = get_current_season(date, hemisphere)

    generators = {
        "winter": _generate_winter_suggestions,
        "summer": _generate_summer_suggestions,
        "spring": _generate_spring_suggestions,
        "autumn": _generate_autumn_suggestions,
    }

    suggestions = generators.get(season, _generate_spring_suggestions)(entities)

    return {
        "season": season,
        "hemisphere": hemisphere,
        "date": (date or datetime.date.today()).isoformat(),
        "total": len(suggestions),
        "suggestions": suggestions,
    }


def run(
    date: datetime.date | None = None,
    entities: list[str] | None = None,
    hemisphere: str | None = None,
) -> dict[str, Any]:
    """Run seasonal adapter and save results."""
    result = get_seasonal_suggestions(date, entities, hemisphere)
    result["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()

    os.makedirs(os.environ.get("DATA_DIR", "/data"), exist_ok=True)
    with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "seasonal_suggestions.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("Seasonal adapter: season=%s, %d suggestions", result["season"], result["total"])
    return result


def load_seasonal_suggestions() -> dict[str, Any]:
    """Load cached seasonal suggestions."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "seasonal_suggestions.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "seasonal_suggestions.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {"season": "unknown", "suggestions": []}
