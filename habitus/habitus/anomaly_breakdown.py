"""Per-entity anomaly scoring — compares current values against per-entity baselines."""

import contextlib
import datetime
import json
import logging
import os

import numpy as np
import pandas as pd

from .sensor_classifier import ACCUMULATING, BINARY, classify_sensor

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
ENTITY_BASELINES_PATH = os.path.join(DATA_DIR, "entity_baselines.json")
ENTITY_ANOMALIES_PATH = os.path.join(DATA_DIR, "entity_anomalies.json")


def build_entity_baselines(df: pd.DataFrame):
    """Build per-entity baselines by hour-of-day × day-of-week. Saves to entity_baselines.json.

    For accumulating sensors (kWh, gas m³, water L), baselines are built on hourly
    delta (consumption rate) values rather than absolute values, since the absolute
    value of an accumulator grows monotonically and has no meaningful mean/std.

    Each baseline slot is tagged with ``baseline_type``:
    - ``"rate"`` — accumulating sensors; slot mean/std are over hourly deltas.
    - ``"absolute"`` — all other sensors; slot mean/std are over raw values.

    Existing ``_accumulating_state`` (previous readings for delta computation) is
    preserved across retraining cycles so scoring continuity is not disrupted.
    """
    if df.empty:
        return
    df = df.copy()
    df["v"] = pd.to_numeric(df["mean"].fillna(df["sum"]), errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["hour_of_day"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df = df.dropna(subset=["v"])

    baselines: dict = {}
    for eid, group in df.groupby("entity_id"):
        entity_bl: dict = {}

        # Classify sensor type before choosing baseline method
        history_vals: list[float] = group.sort_values("ts")["v"].dropna().tolist()
        sensor_type = classify_sensor(eid, history=history_vals)

        if sensor_type == ACCUMULATING:
            entity_bl = _build_rate_baseline(group)
        elif sensor_type == BINARY:
            entity_bl = _build_binary_baseline(group)
        else:
            for (h, d), g in group.groupby(["hour_of_day", "day_of_week"]):
                vals = g["v"].values
                if len(vals) < 3:
                    continue
                entity_bl[f"{h}_{d}"] = {
                    "mean": round(float(np.mean(vals)), 3),
                    "std": round(float(np.std(vals)), 3),
                    "n": len(vals),
                    "baseline_type": "absolute",
                }

        if entity_bl:
            entity_bl["_meta"] = {"sensor_type": sensor_type}
            baselines[eid] = entity_bl

    # Preserve accumulating state from any existing baselines file so that
    # scoring continuity is maintained across retraining cycles.
    if os.path.exists(ENTITY_BASELINES_PATH):
        with contextlib.suppress(json.JSONDecodeError, OSError):
            with open(ENTITY_BASELINES_PATH) as f:
                existing = json.load(f)
            if "_accumulating_state" in existing:
                baselines["_accumulating_state"] = existing["_accumulating_state"]

    with open(ENTITY_BASELINES_PATH, "w") as f:
        json.dump(baselines, f)
    log.info(f"Entity baselines saved for {len(baselines)} entities")


def _build_rate_baseline(group: pd.DataFrame) -> dict:
    """Build delta-based (rate) baseline slots for an accumulating sensor.

    Computes hourly deltas from the sorted time series, filters out negative
    deltas (meter resets / bad readings), then builds mean/std per
    hour-of-day × day-of-week slot.

    Args:
        group: DataFrame slice for a single entity with columns ``ts`` and ``v``.

    Returns:
        Dict of slot keys (``"h_d"``) → ``{mean, std, n, baseline_type: "rate"}``.
        Returns an empty dict when there is insufficient data.
    """
    group = group.sort_values("ts").dropna(subset=["v"])
    if len(group) < 2:
        return {}

    vals = group["v"].values.astype(float)
    ts_arr = pd.to_datetime(group["ts"].values)

    deltas = np.diff(vals)
    delta_ts = ts_arr[1:]

    # Exclude negative deltas (meter resets or bad readings)
    valid_mask = deltas >= 0
    deltas = deltas[valid_mask]
    delta_ts = delta_ts[valid_mask]

    if len(deltas) == 0:
        return {}

    delta_df = pd.DataFrame(
        {
            "delta": deltas,
            "hour_of_day": delta_ts.hour,
            "day_of_week": delta_ts.dayofweek,
        }
    )

    entity_bl: dict = {}
    for (h, d), g in delta_df.groupby(["hour_of_day", "day_of_week"]):
        slot_deltas = g["delta"].values
        if len(slot_deltas) < 3:
            continue
        entity_bl[f"{h}_{d}"] = {
            "mean": round(float(np.mean(slot_deltas)), 6),
            "std": round(float(np.std(slot_deltas)), 6),
            "n": int(len(slot_deltas)),
            "baseline_type": "rate",
        }

    return entity_bl


def _build_binary_baseline(group: pd.DataFrame) -> dict:
    """Build timing & frequency baseline slots for a binary (0/1) sensor.

    Computes per-slot ``on_fraction`` (fraction of hours the sensor is on),
    Bernoulli ``std``, and ``avg_transitions`` (mean number of state-change
    events entering that hour slot).  Also stores a global ``_binary_meta``
    entry with ``avg_duration_on`` — the average run length of consecutive
    'on' hours.

    Args:
        group: DataFrame slice for a single entity with columns ``ts`` and ``v``.

    Returns:
        Dict of slot keys (``"h_d"``) → binary baseline dicts, plus a
        ``_binary_meta`` key.  Returns an empty dict when there is insufficient
        data (fewer than 3 rows).
    """
    group = group.sort_values("ts").dropna(subset=["v"])
    if len(group) < 3:
        return {}

    vals = group["v"].values.astype(float)
    ts_arr = pd.to_datetime(group["ts"].values)
    h_arr = ts_arr.hour
    d_arr = ts_arr.dayofweek

    # Count transitions arriving at each hour slot: 1 if value changed from previous
    transition_counts: dict[str, list[float]] = {}
    for i in range(1, len(vals)):
        k = f"{int(h_arr[i])}_{int(d_arr[i])}"
        if k not in transition_counts:
            transition_counts[k] = []
        transition_counts[k].append(float(vals[i] != vals[i - 1]))

    df_slot = pd.DataFrame({"v": vals, "hour_of_day": h_arr, "day_of_week": d_arr})
    entity_bl: dict = {}
    for (h, d), g in df_slot.groupby(["hour_of_day", "day_of_week"]):
        if len(g) < 3:
            continue
        on_frac = float(np.mean(g["v"].values))
        bern_std = float(np.sqrt(max(on_frac * (1.0 - on_frac), 1e-4)))
        k = f"{h}_{d}"
        avg_trans = float(np.mean(transition_counts.get(k, [0.0])))
        entity_bl[k] = {
            "on_fraction": round(on_frac, 4),
            "std": round(bern_std, 4),
            "avg_transitions": round(avg_trans, 4),
            "n": int(len(g)),
            "baseline_type": "binary",
        }

    avg_dur = _compute_avg_on_duration(vals)
    entity_bl["_binary_meta"] = {"avg_duration_on": round(avg_dur, 4)}
    return entity_bl


def _compute_avg_on_duration(vals: np.ndarray) -> float:
    """Return the average run length (in steps/hours) of consecutive 'on' values.

    Args:
        vals: Ordered array of 0.0/1.0 values.

    Returns:
        Mean run length of '1' runs, or ``1.0`` when no 'on' run is found.
    """
    run_lengths: list[int] = []
    run = 0
    for v in vals:
        if v >= 0.5:
            run += 1
        elif run > 0:
            run_lengths.append(run)
            run = 0
    if run > 0:
        run_lengths.append(run)
    return float(np.mean(run_lengths)) if run_lengths else 1.0


def score_entities(current_states: dict | None = None) -> list:
    """Score each entity's current value against its baseline.

    For accumulating sensors (``baseline_type == "rate"``), the score is
    computed against the hourly delta (current − previous reading) rather than
    the raw value.  Previous readings are stored in ``_accumulating_state``
    inside ``entity_baselines.json`` and updated on every call.

    New accumulating entities — those with no stored previous reading — are
    bootstrapped on the first call (prev value saved, no score emitted).
    Entities within 24 h of their first recorded delta are also exempt from
    scoring to avoid false positives during the bootstrap period.

    Args:
        current_states: ``{entity_id: current_value}`` mapping.  When ``None``,
            values are fetched from the HA REST API.

    Returns:
        List of anomaly dicts sorted by ``z_score`` descending (max 20 entries).
    """
    if not os.path.exists(ENTITY_BASELINES_PATH):
        return []

    with open(ENTITY_BASELINES_PATH) as f:
        baselines = json.load(f)

    now = datetime.datetime.now()
    h, d = now.hour, now.weekday()
    key = f"{h}_{d}"

    # Fetch current states from HA if not provided (exclude metadata keys)
    if current_states is None:
        entity_ids = [k for k in baselines if not k.startswith("_")]
        current_states = _fetch_current_states(entity_ids)

    # Load per-entity accumulating state (stores prev readings for delta computation)
    acc_state: dict = baselines.get("_accumulating_state", {})
    acc_state_updated: dict = dict(acc_state)
    acc_state_changed = False

    # Load per-entity binary state (tracks transitions and on-duration)
    bin_state: dict = baselines.get("_binary_state", {})
    bin_state_updated: dict = dict(bin_state)
    bin_state_changed = False

    anomalies = []
    for eid, bl in baselines.items():
        if eid.startswith("_"):
            continue  # skip metadata keys (_z_score_run, _accumulating_state, …)
        if key not in bl:
            continue
        b = bl[key]
        baseline_type = b.get("baseline_type", "absolute")

        # Constant sensor guard (Bernoulli std near 0 ≡ always-off binary sensor)
        if b.get("std", 1.0) < 0.01:
            continue

        current = current_states.get(eid)
        if current is None:
            continue
        try:
            val = float(current)
        except (TypeError, ValueError):
            continue

        # ── Binary sensor: timing & frequency scoring ──────────────────────────
        if baseline_type == "binary":
            bin_state_changed = True
            current_val = 1 if val >= 0.5 else 0
            b_info = bin_state.get(eid, {})
            prev_raw = b_info.get("current_value")
            prev_bin: int | None = (
                None if prev_raw is None else (1 if float(prev_raw) >= 0.5 else 0)
            )

            is_transition = prev_bin is not None and prev_bin != current_val
            prev_h = b_info.get("hour")
            trans_count = 0 if prev_h != h else int(b_info.get("transitions_this_hour", 0))
            if is_transition:
                trans_count += 1

            state_start: str | None
            if current_val == 1:
                state_start = (
                    now.isoformat()
                    if (prev_bin is None or prev_bin == 0)
                    else b_info.get("state_start_ts", now.isoformat())
                )
            else:
                state_start = None

            bin_state_updated[eid] = {
                "current_value": float(current_val),
                "hour": h,
                "transitions_this_hour": trans_count,
                "state_start_ts": state_start,
            }

            on_frac = b.get("on_fraction", 0.5)
            bern_std = max(b.get("std", 0.5), 0.01)

            # Expected-state score: how unusual is being ON right now?
            z_expected = (1.0 - on_frac) / bern_std if current_val == 1 else 0.0

            # Transition-frequency score: flag if >3× baseline frequency
            avg_trans = b.get("avg_transitions", 0.0)
            z_freq = (
                trans_count / max(avg_trans, 0.1)
                if (avg_trans > 0 and trans_count > 3 * avg_trans)
                else 0.0
            )

            # Duration score: flag if on-duration >2× baseline
            avg_dur = float(bl.get("_binary_meta", {}).get("avg_duration_on", 1.0))
            z_dur = 0.0
            if current_val == 1 and state_start:
                start_dt = datetime.datetime.fromisoformat(state_start)
                dur_h = (now - start_dt).total_seconds() / 3600.0
                if dur_h > 2.0 * avg_dur:
                    z_dur = (dur_h - avg_dur) / max(avg_dur, 0.1)

            z = max(z_expected, z_freq, z_dur)
            if z < 0.5:
                continue

            name = eid.split(".")[-1].replace("_", " ").title()
            if z_freq > 0 and z_freq >= max(z_expected, z_dur):
                description = (
                    f"{name} has {trans_count} transitions this hour — "
                    f"baseline is {avg_trans:.1f}/h"
                )
            elif z_dur > 0 and z_dur >= z_expected and state_start:
                start_dt2 = datetime.datetime.fromisoformat(state_start)
                dur_h2 = (now - start_dt2).total_seconds() / 3600.0
                description = (
                    f"{name} has been on for {dur_h2:.1f}h — "
                    f"typical on-duration is {avg_dur:.1f}h"
                )
            else:
                description = (
                    f"{name} is active — baseline {_day(d)} {h:02d}:00 "
                    f"on-rate is {on_frac * 100:.0f}%"
                )

            anomalies.append(
                {
                    "entity_id": eid,
                    "name": name,
                    "current_value": float(current_val),
                    "baseline_mean": round(on_frac, 4),
                    "baseline_std": round(bern_std, 4),
                    "z_score": round(z, 2),
                    "unit": "",
                    "description": description,
                    "direction": "high",
                }
            )
            continue  # binary handled; skip gauge/rate logic below

        # ── Accumulating & absolute sensors: standard z-score scoring ──────────
        if baseline_type == "rate":
            prev_info = acc_state.get(eid)
            # Determine first_delta_ts — set on first encounter, preserved thereafter
            first_delta_ts = (
                now.isoformat()
                if prev_info is None
                else prev_info.get("first_delta_ts", now.isoformat())
            )
            acc_state_updated[eid] = {
                "prev_value": val,
                "prev_ts": now.isoformat(),
                "first_delta_ts": first_delta_ts,
            }
            acc_state_changed = True

            if prev_info is None:
                # Bootstrap: store prev value, no scoring on first encounter
                continue

            prev_value = prev_info.get("prev_value")
            if prev_value is None:
                continue

            # No scoring within first 24 h of delta accumulation
            first_dt = datetime.datetime.fromisoformat(first_delta_ts)
            if (now - first_dt).total_seconds() < 86400:
                continue

            delta = val - float(prev_value)
            if delta < 0:
                # Negative delta = meter reset or bad reading; skip scoring
                continue

            score_val = delta
        else:
            score_val = val

        z = abs(score_val - b["mean"]) / max(b["std"], 0.01)
        if z < 0.5:
            continue  # not interesting

        # Human-readable entity name
        name = eid.split(".")[-1].replace("_", " ").title()
        unit = _guess_unit(eid)

        if baseline_type == "rate":
            description = (
                f"{name} consumption rate is {_fmt(score_val, unit)}/h — "
                f"baseline {_day(d)} {h:02d}:00 is {_fmt(b['mean'], unit)}/h ±{_fmt(b['std'], unit)}/h"
            )
        else:
            description = (
                f"{name} is {_fmt(score_val, unit)} — "
                f"baseline {_day(d)} {h:02d}:00 is {_fmt(b['mean'], unit)} ±{_fmt(b['std'], unit)}"
            )

        anomalies.append(
            {
                "entity_id": eid,
                "name": name,
                "current_value": round(score_val, 6),
                "baseline_mean": round(b["mean"], 6),
                "baseline_std": round(b["std"], 6),
                "z_score": round(z, 2),
                "unit": unit,
                "description": description,
                "direction": "high" if score_val > b["mean"] else "low",
            }
        )

    # Persist updated state back to entity_baselines.json
    if acc_state_changed or bin_state_changed:
        if acc_state_changed:
            baselines["_accumulating_state"] = acc_state_updated
        if bin_state_changed:
            baselines["_binary_state"] = bin_state_updated
        with open(ENTITY_BASELINES_PATH, "w") as f:
            json.dump(baselines, f, indent=2)

    anomalies.sort(key=lambda x: x["z_score"], reverse=True)
    top = anomalies[:20]

    with open(ENTITY_ANOMALIES_PATH, "w") as f:
        json.dump({"timestamp": now.isoformat(), "anomalies": top}, f, indent=2)
    return top


def compute_breakdown(anomaly_score: float, current_states: dict | None = None) -> list:
    """Compute per-entity z-score breakdown for anomaly UI display.

    When anomaly_score > 40, identifies the top-5 most anomalous entities and
    persists a ``_z_score_run`` record to ``entity_baselines.json`` so the
    Lovelace card can display plain-English reasons without a second API call.

    Args:
        anomaly_score: Combined IsolationForest anomaly score (0–100).
        current_states: Optional ``{entity_id: current_value}`` mapping.
            When ``None``, values are fetched from the HA REST API.

    Returns:
        List of up to 5 anomaly dicts with keys: ``entity_id``, ``name``,
        ``current_value``, ``baseline_mean``, ``baseline_std``, ``z_score``,
        ``unit``, ``description``, ``direction``.  Returns ``[]`` when
        ``anomaly_score <= 40`` or no baselines are available.
    """
    if anomaly_score <= 40:
        return []

    all_anomalies = score_entities(current_states)
    top5 = all_anomalies[:5]

    # Persist z-score run into entity_baselines.json only if the file exists
    if not os.path.exists(ENTITY_BASELINES_PATH):
        return top5

    with open(ENTITY_BASELINES_PATH) as f:
        baselines = json.load(f)

    baselines["_z_score_run"] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "anomaly_score": anomaly_score,
        "top5": top5,
    }
    with open(ENTITY_BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)

    log.info(f"Anomaly breakdown computed: score={anomaly_score}, top5={len(top5)} entities")
    return top5


def _fetch_current_states(entity_ids: list) -> dict:
    import requests  # type: ignore[import-untyped]

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}
    result = {}
    try:
        r = requests.get(f"{ha_url}/api/states", headers=headers, timeout=10)
        if r.status_code == 200:
            for s in r.json():
                if s["entity_id"] in entity_ids:
                    with contextlib.suppress(ValueError, TypeError):

                        result[s["entity_id"]] = float(s["state"])
    except Exception as e:
        log.warning(f"Could not fetch current states: {e}")
    return result


def _guess_unit(eid):
    e = eid.lower()
    if "temperature" in e:
        return "°C"
    if "humidity" in e:
        return "%"
    if "_w" in e or "watt" in e or "power" in e:
        return "W"
    if "energy" in e or "kwh" in e:
        return "kWh"
    if "current" in e or "_a" in e:
        return "A"
    if "voltage" in e:
        return "V"
    if "pressure" in e:
        return "hPa"
    return ""


def _fmt(val, unit):
    return f"{val:.1f}{unit}" if unit else f"{val:.2f}"


def _day(d):
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]
