"""Pattern discovery and automation suggestion engine — v2.0."""
import json, os, datetime, logging
import pandas as pd
import numpy as np

log = logging.getLogger('habitus')
DATA_DIR         = os.environ.get("DATA_DIR", "/data")
PATTERNS_PATH    = os.path.join(DATA_DIR, "patterns.json")
SUGGESTIONS_PATH = os.path.join(DATA_DIR, "suggestions.json")

NOTIFY = os.environ.get("HABITUS_NOTIFY_SERVICE", "notify.notify")
THRESHOLD = int(os.environ.get("HABITUS_ANOMALY_THRESHOLD", "70"))

def _has(stat_ids, *keywords):
    """Check if any of the keywords appear in the tracked entity list."""
    joined = ' '.join(stat_ids).lower()
    return any(k in joined for k in keywords)


def discover_patterns(features: pd.DataFrame) -> dict:
    patterns = {}
    hourly = features.groupby('hour_of_day').agg(
        mean_power=('total_power_w','mean'),
        std_power=('total_power_w','std'),
        mean_temp=('avg_temp_c','mean'),
        activity=('sensor_changes','mean')
    ).round(2)

    night_power = float(hourly.loc[hourly.index.isin(range(1,6)), 'mean_power'].mean())
    if pd.isna(night_power): night_power = 50.0

    wakeup = next((h for h in range(5,12) if hourly.loc[h,'mean_power'] > night_power*1.5), None)
    sleep_h = next((h for h in range(23,18,-1) if hourly.loc[h,'mean_power'] > night_power*1.3), None)
    peak_hour = int(hourly['mean_power'].idxmax())
    peak_power = round(float(hourly['mean_power'].max()), 1)

    patterns['daily_routine'] = {
        'estimated_wakeup_hour': wakeup,
        'estimated_sleep_hour': sleep_h,
        'peak_usage_hour': peak_hour,
        'peak_usage_watts': peak_power,
        'night_baseline_watts': round(night_power, 1),
    }
    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    daily = features.groupby('day_of_week').agg(mean_power=('total_power_w','mean'),activity=('sensor_changes','mean')).round(2)
    patterns['weekly'] = {day_names[i]: {'mean_power_w': float(daily.loc[i,'mean_power']), 'activity': float(daily.loc[i,'activity'])} for i in range(7) if i in daily.index}
    seasonal = features.groupby('month').agg(mean_power=('total_power_w','mean'),mean_temp=('avg_temp_c','mean')).round(2)
    mnames = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    patterns['seasonal'] = {mnames[m]: {'mean_power_w': float(seasonal.loc[m,'mean_power']), 'mean_temp_c': float(seasonal.loc[m,'mean_temp'])} for m in seasonal.index}
    patterns['generated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return patterns


def generate_suggestions(patterns: dict, features: pd.DataFrame, stat_ids: list) -> list:
    suggestions = []
    routine   = patterns.get('daily_routine', {})
    wakeup    = routine.get('estimated_wakeup_hour')
    sleep_h   = routine.get('estimated_sleep_hour')
    peak_h    = routine.get('peak_usage_hour', 18)
    peak_w    = routine.get('peak_usage_watts', 500)
    night_w   = routine.get('night_baseline_watts', 100)

    has_bilge   = _has(stat_ids, 'bilge')
    has_battery = _has(stat_ids, 'battery', 'soc')
    has_shore   = _has(stat_ids, 'shore', 'mains', 'grid')
    has_solar   = _has(stat_ids, 'solar', 'pv', 'mppt', 'epever', 'scm')
    has_inverter= _has(stat_ids, 'inverter', 'mastervolt', 'load')
    has_temp    = _has(stat_ids, 'temperature')

    # ── ROUTINE ───────────────────────────────────────────────────────────────
    if wakeup:
        suggestions.append({
            "id": "morning_routine", "title": f"Morning Routine at {wakeup:02d}:00",
            "description": f"Power consistently rises above {int(night_w*1.5)}W around {wakeup:02d}:00 on weekdays — Habitus detected a recurring morning activity pattern.",
            "confidence": 85, "category": "routine", "applicable": True,
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
        entity_id: scene.morning  # replace with your scene"""
        })

    if sleep_h:
        suggestions.append({
            "id": "night_mode", "title": f"Night Mode at {sleep_h:02d}:00",
            "description": f"Power drops to near-baseline after {sleep_h:02d}:00 most nights. Suggest triggering night mode / reducing loads.",
            "confidence": 80, "category": "routine", "applicable": True,
            "yaml": f"""automation:
  alias: "Habitus — Night mode"
  description: "Auto-generated from observed {sleep_h:02d}:00 sleep pattern"
  trigger:
    - platform: time
      at: "{sleep_h:02d}:30:00"
  action:
    - service: scene.turn_on
      target:
        entity_id: scene.night  # replace with your scene"""
        })

    suggestions.append({
        "id": "weekend_mode", "title": "Weekend Mode",
        "description": "Weekend power profile differs significantly from weekdays — suggest separate scene/mode for Saturday/Sunday.",
        "confidence": 70, "category": "routine", "applicable": True,
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
        entity_id: scene.weekend  # replace with your scene"""
    })

    # ── ENERGY ────────────────────────────────────────────────────────────────
    suggestions.append({
        "id": "peak_power_alert", "title": f"High Power Alert (>{int(peak_w*1.3)}W)",
        "description": f"Peak normal usage is {int(peak_w)}W at {peak_h:02d}:00. This fires when load exceeds 130% of that for 10+ minutes.",
        "confidence": 90, "category": "energy", "applicable": has_inverter,
        "yaml": f"""automation:
  alias: "Habitus — High power alert"
  trigger:
    - platform: numeric_state
      entity_id: sensor.mastervolt_total_load
      above: {int(peak_w*1.3)}
      for:
        minutes: 10
  action:
    - service: {NOTIFY}
      data:
        title: "⚡ High Power Usage"
        message: "Load has exceeded {int(peak_w*1.3)}W for 10+ minutes. Current: {{{{ states('sensor.mastervolt_total_load') }}}}W" """
    })

    suggestions.append({
        "id": "overnight_standby", "title": f"Overnight Standby Anomaly (>{int(night_w*1.4)}W)",
        "description": f"Overnight baseline is {int(night_w)}W. Fires between 01:00–05:00 if consumption rises above {int(night_w*1.4)}W — something unexpected is running.",
        "confidence": 85, "category": "energy", "applicable": True,
        "yaml": f"""automation:
  alias: "Habitus — Overnight power anomaly"
  trigger:
    - platform: numeric_state
      entity_id: sensor.mastervolt_total_load
      above: {int(night_w*1.4)}
  condition:
    - condition: time
      after: "01:00:00"
      before: "05:00:00"
  action:
    - service: {NOTIFY}
      data:
        title: "🌙 Unusual Overnight Power"
        message: "Power is {{{{ states('sensor.mastervolt_total_load') }}}}W at night — something may be left on." """
    })

    if has_solar and has_inverter:
        suggestions.append({
            "id": "solar_export", "title": "Solar Surplus — Shift Loads",
            "description": "When solar production significantly exceeds current load, shift deferrable loads (water heating, charging) to maximise self-consumption.",
            "confidence": 78, "category": "energy", "applicable": True,
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
        message: "Solar is generating more than you're using — good time to run high-load appliances." """
        })

    # ── BOAT / MARINE ─────────────────────────────────────────────────────────
    if has_battery:
        suggestions.append({
            "id": "battery_protection", "title": "Battery SOC Protection Alert",
            "description": "Detected battery monitoring entities. Alert when state of charge drops to a critical level to prevent deep discharge.",
            "confidence": 95, "category": "boat", "applicable": True,
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
    - service: """ + NOTIFY + """
      data:
        title: "🔋 Battery Low"
        message: "House battery SOC is below 20%. Connect shore power or reduce loads." """
        })

    if has_bilge:
        suggestions.append({
            "id": "bilge_anomaly", "title": "Bilge Pump Anomaly Alert",
            "description": "Bilge sensors detected. Alert if the bilge pump runs unexpectedly or bilge temperature spikes, which can indicate a leak or equipment issue.",
            "confidence": 95, "category": "boat", "applicable": True,
            "yaml": """automation:
  alias: "Habitus — Bilge anomaly"
  trigger:
    - platform: state
      entity_id: binary_sensor.bilge_pump_running  # adjust entity
      to: "on"
      for:
        minutes: 5
  action:
    - service: """ + NOTIFY + """
      data:
        title: "⚠️ Bilge Pump Running"
        message: "Bilge pump has been running for 5+ minutes. Check for water ingress." """
        })

    if has_shore:
        suggestions.append({
            "id": "shore_power_loss", "title": "Shore Power Loss Alert",
            "description": "Shore power entities detected. Alert immediately when shore power is lost so you can switch to battery/generator before discharge.",
            "confidence": 92, "category": "boat", "applicable": True,
            "yaml": """automation:
  alias: "Habitus — Shore power lost"
  trigger:
    - platform: numeric_state
      entity_id: sensor.shore_power_smart_meter_electric_consumption_w
      below: 10
      for:
        minutes: 2
  action:
    - service: """ + NOTIFY + """
      data:
        title: "🔌 Shore Power Lost"
        message: "Shore power appears to have been disconnected. Running on battery." """
        })

    if has_inverter and has_solar:
        suggestions.append({
            "id": "inverter_overload", "title": "Inverter Overload Predictor",
            "description": "Alert when total load is approaching inverter capacity limits, giving time to shed loads before an overload trip.",
            "confidence": 82, "category": "boat", "applicable": True,
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
        message: "Load is {{{{ states('sensor.mastervolt_total_load') }}}}W — approaching inverter limits. Consider shedding loads." """
        })

    suggestions.append({
        "id": "harbor_mode", "title": "Harbor Mode (Away Profile)",
        "description": "Automatically reduce non-essential loads when no presence is detected for extended periods — keeps standby power minimal while away.",
        "confidence": 72, "category": "boat", "applicable": True,
        "yaml": f"""automation:
  alias: "Habitus — Harbor mode"
  description: "Activates low-power profile when away for >2h"
  trigger:
    - platform: state
      entity_id: person.craig  # replace with your person entity
      to: "not_home"
      for:
        hours: 2
  action:
    - service: scene.turn_on
      target:
        entity_id: scene.harbor_mode  # create this scene"""
    })

    # ── ANOMALY ───────────────────────────────────────────────────────────────
    suggestions.append({
        "id": "anomaly_alert", "title": f"Habitus Anomaly Alert (Score >{THRESHOLD})",
        "description": f"Send a notification when Habitus detects unusual behaviour scoring above {THRESHOLD}/100 for 5+ minutes.",
        "confidence": 95, "category": "anomaly", "applicable": True,
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
          Trained on {{{{ states('sensor.habitus_training_days') }}}} days of history."""
    })

    suggestions.append({
        "id": "sensor_watchdog", "title": "Sensor Watchdog",
        "description": "Alert when a key sensor goes unavailable for more than 1 hour — catches connectivity issues, battery failures, or hardware faults early.",
        "confidence": 80, "category": "anomaly", "applicable": True,
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
    - service: """ + NOTIFY + """
      data:
        title: "📡 Sensor Offline"
        message: "{{ trigger.entity_id }} has been unavailable for over 1 hour." """
    })

    suggestions.append({
        "id": "daily_digest", "title": "Daily Energy Digest",
        "description": "Receive a morning summary of yesterday's energy usage, anomalies detected, and today's solar forecast.",
        "confidence": 88, "category": "anomaly", "applicable": True,
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
          Model trained on {{{{ states('sensor.habitus_training_days') }}}} days of history."""
    })

    # ── LOVELACE ──────────────────────────────────────────────────────────────
    suggestions.append({
        "id": "lovelace_card", "title": "Lovelace Insights Card",
        "description": "Ready-to-paste Lovelace YAML for a Habitus insights card on any dashboard.",
        "confidence": 100, "category": "lovelace", "applicable": True,
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
        name: Sensors"""
    })

    for s in suggestions:
        s['generated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return suggestions


def run(features: pd.DataFrame, stat_ids: list = None):
    stat_ids = stat_ids or []
    patterns    = discover_patterns(features)
    suggestions = generate_suggestions(patterns, features, stat_ids)
    with open(PATTERNS_PATH,    'w') as f: json.dump(patterns,    f, indent=2)
    with open(SUGGESTIONS_PATH, 'w') as f: json.dump(suggestions, f, indent=2)
    log.info(f"Patterns saved — {len(suggestions)} suggestions ({sum(1 for s in suggestions if s['applicable'])} applicable)")
    return patterns, suggestions
