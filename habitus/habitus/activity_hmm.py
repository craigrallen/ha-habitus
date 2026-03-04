"""Hidden Markov Model for activity state inference.

The home has hidden states: "sleeping", "cooking", "working", "watching TV", "away".
Observable signals: light states, power readings, motion, door opens.
HMM learns the transition probabilities automatically.

Predicts: "it's 17:30, Craig just opened front door → 85% chance 'evening relaxing' starts in 20 min"

Falls back to a simpler clustering approach if hmmlearn is unavailable.
"""

import datetime
import json
import logging
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from .ha_db import resolve_ha_db_path

import numpy as np

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
HMM_PATH = os.path.join(DATA_DIR, "activity_states.json")

# Number of hidden activity states to discover
N_STATES = 8
# Feature extraction: hourly windows
WINDOW_MINUTES = 60


def _build_observation_matrix(entity_to_area: dict[str, str], days: int = 30) -> tuple[np.ndarray, list[str]]:
    """Build observation matrix for HMM training.

    Each row = one hour window.
    Features: lights_on, switches_on, motion_events, media_active, climate_active,
              doors_open, people_home, power_level (binned), hour_of_day
    """
    db_path = resolve_ha_db_path()
    if not db_path:
        return np.array([]), []

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute("""
            SELECT sm.entity_id, s.state, s.last_changed_ts
            FROM states s
            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            WHERE s.last_changed_ts > ?
            AND s.state NOT IN ('unavailable', 'unknown', '')
            ORDER BY s.last_changed_ts
        """, (cutoff_ts,)).fetchall()
        conn.close()
    except Exception as e:
        log.warning("activity_hmm: DB query failed: %s", e)
        return np.array([]), []

    if not rows:
        return np.array([]), []

    # Bin events into hourly windows
    windows: dict[int, dict[str, int]] = defaultdict(lambda: {
        "lights_on": 0, "switches_on": 0, "motion": 0,
        "media": 0, "climate": 0, "doors": 0,
        "people_home": 0, "total_changes": 0,
    })

    for eid, state, ts in rows:
        window_key = int(ts // (WINDOW_MINUTES * 60))
        domain = eid.split(".")[0]
        eid_lower = eid.lower()
        w = windows[window_key]
        w["total_changes"] += 1

        if domain == "light" and state == "on":
            w["lights_on"] += 1
        elif domain == "switch" and state == "on":
            w["switches_on"] += 1
        elif domain == "binary_sensor" and state == "on" and any(k in eid_lower for k in ("motion", "pir", "occupancy")):
            w["motion"] += 1
        elif domain == "media_player" and state in ("playing", "on"):
            w["media"] += 1
        elif domain == "climate" and state in ("heat", "cool", "auto"):
            w["climate"] += 1
        elif domain == "binary_sensor" and state == "on" and any(k in eid_lower for k in ("door", "window")):
            w["doors"] += 1
        elif domain == "person" and state == "home":
            w["people_home"] += 1

    if not windows:
        return np.array([]), []

    # Convert to matrix
    sorted_keys = sorted(windows.keys())
    timestamps = []
    features = []

    for wk in sorted_keys:
        w = windows[wk]
        ts = wk * WINDOW_MINUTES * 60
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
        hour = dt.hour

        features.append([
            min(w["lights_on"], 10),
            min(w["switches_on"], 10),
            min(w["motion"], 20),
            min(w["media"], 5),
            min(w["climate"], 5),
            min(w["doors"], 10),
            min(w["people_home"], 5),
            min(w["total_changes"], 50),
            hour,
        ])
        timestamps.append(dt.isoformat())

    return np.array(features, dtype=np.float64), timestamps


# Activity state labels (assigned post-hoc based on feature patterns)
STATE_LABELS = {
    "sleeping": {"lights_on": 0, "motion": 0, "media": 0, "hour_range": (23, 7)},
    "away": {"lights_on": 0, "motion": 0, "people_home": 0},
    "morning_routine": {"motion": 2, "hour_range": (6, 10)},
    "cooking": {"switches_on": 2, "hour_range": (11, 21)},
    "working": {"lights_on": 1, "motion": 1, "media": 0, "hour_range": (8, 18)},
    "relaxing": {"media": 1, "lights_on": 1, "hour_range": (18, 23)},
    "active": {"motion": 3, "total_changes": 10},
    "idle": {},
}


def _label_state(centroid: np.ndarray) -> str:
    """Try to assign a human-readable label to a state centroid."""
    lights, switches, motion, media, climate, doors, people, changes, hour = centroid

    if motion < 1 and lights < 1 and media < 1 and (hour >= 23 or hour < 7):
        return "sleeping"
    if people < 0.5 and motion < 1:
        return "away"
    if media >= 1 and lights >= 1 and hour >= 17:
        return "relaxing"
    if switches >= 2 and (11 <= hour <= 21):
        return "cooking"
    if motion >= 3 and changes >= 10:
        return "active"
    if lights >= 1 and motion >= 1 and media < 1 and 8 <= hour <= 18:
        return "working"
    if 6 <= hour <= 10 and motion >= 1:
        return "morning_routine"
    return "idle"


def train_activity_model(entity_to_area: dict[str, str], days: int = 30) -> dict[str, Any]:
    """Train HMM on observation data to discover activity states."""
    X, timestamps = _build_observation_matrix(entity_to_area, days=days)
    if len(X) < 24:
        return {"states": [], "current_state": None}

    # Normalise features
    X_norm = X.copy()
    for i in range(X.shape[1] - 1):  # Don't normalise hour
        col_max = X_norm[:, i].max()
        if col_max > 0:
            X_norm[:, i] /= col_max

    try:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                            n_iter=100, random_state=42)
        model.fit(X_norm)
        state_sequence = model.predict(X_norm)
        use_hmm = True
        log.info("activity_hmm: trained GaussianHMM with %d states", N_STATES)
    except ImportError:
        log.warning("hmmlearn not available, falling back to KMeans clustering")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=N_STATES, random_state=42, n_init=10)
        state_sequence = km.fit_predict(X_norm)
        use_hmm = False

    # Compute state centroids and labels
    states_info = []
    for s in range(N_STATES):
        mask = state_sequence == s
        if not mask.any():
            continue
        centroid = X[mask].mean(axis=0)
        count = int(mask.sum())
        pct = round(count / len(X) * 100, 1)
        label = _label_state(centroid)

        # Typical hours for this state
        hours = X[mask, -1]  # hour column
        hour_counts = Counter(int(h) for h in hours)
        peak_hours = [h for h, _ in hour_counts.most_common(3)]

        states_info.append({
            "id": int(s),
            "label": label,
            "count": count,
            "percentage": pct,
            "peak_hours": peak_hours,
            "avg_lights": round(float(centroid[0]), 1),
            "avg_motion": round(float(centroid[2]), 1),
            "avg_media": round(float(centroid[3]), 1),
            "avg_climate": round(float(centroid[4]), 1),
            "avg_changes": round(float(centroid[7]), 1),
        })

    states_info.sort(key=lambda s: -s["count"])

    # Current state (last observation)
    current_state_id = int(state_sequence[-1]) if len(state_sequence) > 0 else None
    current_label = None
    if current_state_id is not None:
        for si in states_info:
            if si["id"] == current_state_id:
                current_label = si["label"]
                break

    # Transition matrix (state i → state j probability)
    trans_matrix = np.zeros((N_STATES, N_STATES))
    for i in range(len(state_sequence) - 1):
        trans_matrix[state_sequence[i], state_sequence[i + 1]] += 1
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_probs = trans_matrix / row_sums

    # Next-state predictions from current
    next_state_probs = []
    if current_state_id is not None:
        for j in range(N_STATES):
            prob = float(trans_probs[current_state_id, j])
            if prob > 0.05:
                label = "unknown"
                for si in states_info:
                    if si["id"] == j:
                        label = si["label"]
                        break
                next_state_probs.append({"state": label, "probability": round(prob, 3)})
        next_state_probs.sort(key=lambda x: -x["probability"])

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "method": "hmm" if use_hmm else "kmeans",
        "n_states": N_STATES,
        "total_windows": len(X),
        "states": states_info,
        "current_state": current_label,
        "next_state_predictions": next_state_probs[:5],
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HMM_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("activity_hmm: %d states discovered, current=%s", len(states_info), current_label)
    return result
