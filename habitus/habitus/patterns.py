"""
Pattern discovery and automation suggestion engine.
Runs after model training. Writes patterns.json and suggestions.json.
No raw data stored — everything derived from the feature matrix.
"""
import json, os, datetime
import pandas as pd
import numpy as np

DATA_DIR        = os.environ.get("DATA_DIR", "/data")
PATTERNS_PATH   = os.path.join(DATA_DIR, "patterns.json")
SUGGESTIONS_PATH = os.path.join(DATA_DIR, "suggestions.json")


def discover_patterns(features: pd.DataFrame) -> dict:
    """Extract recurring behavioral patterns from hourly feature matrix."""
    patterns = {}

    # ── Daily routine ─────────────────────────────────────────────────────────
    hourly = features.groupby('hour_of_day').agg(
        mean_power=('total_power_w', 'mean'),
        std_power=('total_power_w', 'std'),
        mean_temp=('avg_temp_c', 'mean'),
        activity=('sensor_changes', 'mean')
    ).round(1)

    # Find wakeup hour: first hour where power > 150% of night baseline
    night_power = hourly.loc[1:5, 'mean_power'].mean()
    wakeup = None
    for h in range(5, 12):
        if hourly.loc[h, 'mean_power'] > night_power * 1.5:
            wakeup = h
            break

    # Find sleep hour: last hour where power > 150% of night baseline before midnight
    sleep = None
    for h in range(23, 18, -1):
        if hourly.loc[h, 'mean_power'] > night_power * 1.3:
            sleep = h
            break

    # Peak usage hour
    peak_hour = int(hourly['mean_power'].idxmax())
    peak_power = round(float(hourly['mean_power'].max()))

    patterns['daily_routine'] = {
        'estimated_wakeup_hour': wakeup,
        'estimated_sleep_hour': sleep,
        'peak_usage_hour': peak_hour,
        'peak_usage_watts': peak_power,
        'night_baseline_watts': round(float(night_power), 1),
    }

    # ── Weekly pattern ────────────────────────────────────────────────────────
    daily = features.groupby('day_of_week').agg(
        mean_power=('total_power_w', 'mean'),
        activity=('sensor_changes', 'mean')
    ).round(1)
    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    patterns['weekly'] = {
        day_names[i]: {
            'mean_power_w': float(daily.loc[i, 'mean_power']),
            'activity': float(daily.loc[i, 'activity'])
        } for i in range(7) if i in daily.index
    }

    # ── Seasonal ──────────────────────────────────────────────────────────────
    seasonal = features.groupby('month').agg(
        mean_power=('total_power_w', 'mean'),
        mean_temp=('avg_temp_c', 'mean')
    ).round(1)
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    patterns['seasonal'] = {
        month_names[m]: {
            'mean_power_w': float(seasonal.loc[m, 'mean_power']),
            'mean_temp_c': float(seasonal.loc[m, 'mean_temp'])
        } for m in seasonal.index
    }

    patterns['generated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return patterns


def generate_suggestions(patterns: dict, features: pd.DataFrame) -> list:
    """Generate actionable automation suggestions based on discovered patterns."""
    suggestions = []
    routine = patterns.get('daily_routine', {})
    hourly = features.groupby('hour_of_day').agg(
        mean_power=('total_power_w', 'mean'),
    )
    night_power = routine.get('night_baseline_watts', 100)
    peak_hour   = routine.get('peak_usage_hour', 18)
    peak_power  = routine.get('peak_usage_watts', 500)
    wakeup      = routine.get('estimated_wakeup_hour')
    sleep_h     = routine.get('estimated_sleep_hour')

    # ── Morning routine ───────────────────────────────────────────────────────
    if wakeup:
        suggestions.append({
            "id": "morning_routine",
            "title": f"Morning Routine at {wakeup:02d}:00",
            "description": f"Power consistently rises above {int(night_power*1.5)}W around {wakeup:02d}:00 on weekdays — typical morning activity pattern detected.",
            "confidence": 85,
            "category": "routine",
            "yaml": f"""automation:
  alias: "Habitus — Morning routine"
  description: "Suggested by Habitus based on observed wake-up pattern"
  trigger:
    - platform: time
      at: "{wakeup:02d}:00:00"
  condition:
    - condition: time
      weekday: [mon, tue, wed, thu, fri]
  action:
    - service: scene.turn_on
      target:
        entity_id: scene.morning  # replace with your scene
"""
        })

    # ── Night mode ────────────────────────────────────────────────────────────
    if sleep_h:
        suggestions.append({
            "id": "night_mode",
            "title": f"Night Mode at {sleep_h:02d}:00",
            "description": f"Power drops to near-baseline after {sleep_h:02d}:00 most nights — suggest triggering night mode.",
            "confidence": 80,
            "category": "routine",
            "yaml": f"""automation:
  alias: "Habitus — Night mode"
  description: "Suggested by Habitus based on observed sleep pattern"
  trigger:
    - platform: time
      at: "{sleep_h:02d}:30:00"
  action:
    - service: scene.turn_on
      target:
        entity_id: scene.night  # replace with your scene
"""
        })

    # ── Peak usage alert ──────────────────────────────────────────────────────
    suggestions.append({
        "id": "peak_power_alert",
        "title": f"High Power Usage Alert (>{int(peak_power*1.3)}W)",
        "description": f"Peak normal usage is {peak_power}W at {peak_hour:02d}:00. Alert when power significantly exceeds this.",
        "confidence": 90,
        "category": "energy",
        "yaml": f"""automation:
  alias: "Habitus — High power alert"
  description: "Suggested by Habitus — triggers when power exceeds 130% of peak baseline"
  trigger:
    - platform: numeric_state
      entity_id: sensor.mastervolt_total_load
      above: {int(peak_power * 1.3)}
      for:
        minutes: 10
  action:
    - service: notify.notify
      data:
        title: "⚡ High Power Usage"
        message: "Current load exceeds {int(peak_power*1.3)}W for 10+ minutes."
"""
    })

    # ── Anomaly alert ─────────────────────────────────────────────────────────
    suggestions.append({
        "id": "anomaly_alert",
        "title": "Anomaly Detection Alert",
        "description": "Send a notification when Habitus detects unusual behaviour in your home.",
        "confidence": 95,
        "category": "anomaly",
        "yaml": """automation:
  alias: "Habitus — Anomaly alert"
  description: "Notifies when Habitus anomaly score exceeds 70"
  trigger:
    - platform: state
      entity_id: binary_sensor.habitus_anomaly_detected
      to: "on"
      for:
        minutes: 5
  action:
    - service: notify.notify
      data:
        title: "🧠 Habitus — Unusual Activity"
        message: >
          Habitus has detected unusual behaviour.
          Score: {{ states('sensor.habitus_anomaly_score') }}/100.
"""
    })

    # ── Idle device detection ─────────────────────────────────────────────────
    suggestions.append({
        "id": "standby_waste",
        "title": f"Overnight Standby Power ({int(night_power)}W baseline)",
        "description": f"Your overnight baseline is {int(night_power)}W. If it rises above {int(night_power*1.4)}W between 01:00–05:00, something unusual is running.",
        "confidence": 75,
        "category": "energy",
        "yaml": f"""automation:
  alias: "Habitus — Overnight power anomaly"
  description: "Suggested by Habitus — flags unusual overnight power draw"
  trigger:
    - platform: numeric_state
      entity_id: sensor.mastervolt_total_load
      above: {int(night_power * 1.4)}
  condition:
    - condition: time
      after: "01:00:00"
      before: "05:00:00"
  action:
    - service: notify.notify
      data:
        title: "🌙 Unusual Overnight Power"
        message: "Power is above {int(night_power*1.4)}W at night — something may have been left on."
"""
    })

    for s in suggestions:
        s['generated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return suggestions


def run(features: pd.DataFrame):
    patterns  = discover_patterns(features)
    suggestions = generate_suggestions(patterns, features)

    with open(PATTERNS_PATH,    'w') as f: json.dump(patterns,    f, indent=2)
    with open(SUGGESTIONS_PATH, 'w') as f: json.dump(suggestions, f, indent=2)

    import logging
    logging.getLogger('habitus').info(
        f"Patterns saved — {len(suggestions)} automation suggestions generated"
    )
    return patterns, suggestions
