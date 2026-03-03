"""Device training mode for appliance fingerprinting.

User flow:
1. Open Geek tab → Device Training
2. Select power entity (e.g., shore_power_smart_meter)
3. Click "Start Training" → system records baseline
4. User turns on device (e.g., hob)
5. System captures the power delta, shape, and duration
6. User names the device → fingerprint saved

This creates highly accurate signatures from the user's actual devices,
far better than generic wattage ranges.
"""

import datetime
import json
import logging
import os
import sqlite3
import time
from typing import Any

import numpy as np

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
CUSTOM_SIGNATURES_PATH = os.path.join(DATA_DIR, "custom_signatures.json")
TRAINING_SESSION_PATH = os.path.join(DATA_DIR, "training_session.json")
HA_DB = "/homeassistant/home-assistant_v2.db"


def _load_custom_signatures() -> list[dict]:
    try:
        if os.path.exists(CUSTOM_SIGNATURES_PATH):
            with open(CUSTOM_SIGNATURES_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_custom_signatures(sigs: list[dict]):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CUSTOM_SIGNATURES_PATH, "w") as f:
        json.dump(sigs, f, indent=2, default=str)


def start_training_session(power_entity: str) -> dict:
    """Start a device training session. Captures current baseline power."""
    import requests
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))

    # Get current power reading as baseline
    baseline_w = None
    try:
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"{ha_url}/api/states/{power_entity}", headers=headers, timeout=5)
        if r.status_code == 200:
            state = r.json().get("state", "0")
            baseline_w = float(state)
    except Exception as e:
        log.warning("device_trainer: couldn't read baseline: %s", e)

    session = {
        "status": "recording",
        "power_entity": power_entity,
        "baseline_w": baseline_w,
        "start_time": datetime.datetime.now(datetime.UTC).isoformat(),
        "start_ts": time.time(),
        "readings": [],
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRAINING_SESSION_PATH, "w") as f:
        json.dump(session, f, indent=2, default=str)

    log.info("device_trainer: started session, baseline=%.1fW on %s",
             baseline_w or 0, power_entity)
    return session


def stop_training_session(device_name: str, device_category: str = "custom") -> dict:
    """Stop training and analyse the captured power profile."""
    try:
        with open(TRAINING_SESSION_PATH) as f:
            session = json.load(f)
    except Exception:
        return {"error": "No active training session"}

    if session.get("status") != "recording":
        return {"error": "No active training session"}

    power_entity = session["power_entity"]
    start_ts = session["start_ts"]
    baseline_w = session.get("baseline_w", 0) or 0
    end_ts = time.time()
    duration_min = (end_ts - start_ts) / 60

    # Read power values during the training window from DB
    readings = []
    if os.path.exists(HA_DB):
        try:
            conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)
            rows = conn.execute("""
                SELECT s.state, s.last_changed_ts FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE sm.entity_id = ? AND s.last_changed_ts BETWEEN ? AND ?
                ORDER BY s.last_changed_ts
            """, (power_entity, start_ts, end_ts)).fetchall()
            conn.close()

            for state_val, ts in rows:
                try:
                    w = float(state_val)
                    if 0 <= w <= 25000:
                        readings.append({"ts": ts, "watts": w})
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            log.warning("device_trainer: DB read failed: %s", e)

    if not readings:
        # Fall back to current value via API
        return {"error": "No readings captured during training. Try a longer session."}

    # Analyse the profile
    watts = np.array([r["watts"] for r in readings])
    deltas = watts - baseline_w

    peak_w = float(np.max(watts))
    peak_delta = float(np.max(deltas))
    avg_w = float(np.mean(watts))
    avg_delta = float(np.mean(deltas))

    # Detect shape
    from .appliance_fingerprint import detect_power_shape
    shape = detect_power_shape(list(watts))

    # Inrush detection (first reading vs average)
    inrush_w = float(watts[0]) if len(watts) > 0 else 0
    has_inrush = inrush_w > avg_w * 1.3

    signature = {
        "name": device_name,
        "category": device_category,
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": "user_trained",
        "power_entity": power_entity,
        "baseline_w": round(baseline_w, 1),
        "peak_w": round(peak_w, 1),
        "peak_delta_w": round(peak_delta, 1),
        "avg_w": round(avg_w, 1),
        "avg_delta_w": round(avg_delta, 1),
        "min_w": round(float(np.min(watts)), 1),
        "duration_min": round(duration_min, 1),
        "shape": shape,
        "has_inrush": has_inrush,
        "inrush_w": round(inrush_w, 1),
        "readings_count": len(readings),
        "profile": [round(float(w), 1) for w in watts[::max(1, len(watts) // 50)]],  # Downsample to ~50 points
    }

    # Save to custom signatures
    sigs = _load_custom_signatures()
    sigs.append(signature)
    _save_custom_signatures(sigs)

    # Clear session
    session["status"] = "complete"
    with open(TRAINING_SESSION_PATH, "w") as f:
        json.dump(session, f, indent=2, default=str)

    log.info("device_trainer: captured '%s' — peak=%.0fW, avg=%.0fW, shape=%s, duration=%.1fmin",
             device_name, peak_w, avg_w, shape, duration_min)
    return signature


def get_training_status() -> dict:
    """Get current training session status."""
    try:
        if os.path.exists(TRAINING_SESSION_PATH):
            with open(TRAINING_SESSION_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"status": "idle"}


def list_custom_signatures() -> list[dict]:
    """List all user-trained device signatures."""
    return _load_custom_signatures()


def delete_signature(name: str) -> bool:
    """Delete a custom signature by name."""
    sigs = _load_custom_signatures()
    new_sigs = [s for s in sigs if s["name"] != name]
    if len(new_sigs) < len(sigs):
        _save_custom_signatures(new_sigs)
        return True
    return False
