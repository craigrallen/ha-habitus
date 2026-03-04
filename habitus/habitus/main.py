"""
Habitus v2.0 — behavioral intelligence for Home Assistant.
HA is source of truth. /data stores only inference artifacts.
"""

import argparse
import asyncio
import contextlib
import datetime
import json
import logging
import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
import requests  # type: ignore[import-untyped]
import websockets

from . import activity as activity_engine
from . import anomaly_breakdown, automation_gap, automation_score, drift, phantom, seasonal
from . import (activity_hmm, appliance_fingerprint, automation_builder, conflict_detector,
                correlation_engine, dynamic_automations, energy_forecast, ha_areas,
                markov_chain, nilm_disaggregator, room_predictor, routine_predictor,
                scene_detector, sequence_miner)
from . import patterns as pattern_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
HA_WS_URL = (
    os.environ.get("HABITUS_HA_URL", "http://supervisor/core")
    .replace("http://", "ws://")
    .replace("https://", "wss://")
    + "/api/websocket"
)
HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_WS = os.environ.get("HA_WS", "ws://supervisor/core/api/websocket")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
NOTIFY_SVC = os.environ.get("HABITUS_NOTIFY_SERVICE", "notify.notify")
NOTIFY_ON = os.environ.get("HABITUS_NOTIFY_ON", "true").lower() == "true"
THRESHOLD = int(os.environ.get("HABITUS_ANOMALY_THRESHOLD", "70"))
DAILY_DIGEST = os.environ.get("HABITUS_DAILY_DIGEST", "false").lower() == "true"
DAILY_DIGEST_HOUR = int(os.environ.get("HABITUS_DAILY_DIGEST_HOUR", "8"))
# Minimum training days before anomaly scoring is trusted — assume normal until then
MIN_SCORING_DAYS = int(os.environ.get("HABITUS_MIN_SCORING_DAYS", "7"))
# Hard safety guard for very large history pulls (prevents web process OOM).
# Set <=0 to disable (not recommended).
FETCH_ROW_BUDGET = int(os.environ.get("HABITUS_FETCH_ROW_BUDGET", "1000000"))
FETCH_MIN_WINDOW_DAYS = int(os.environ.get("HABITUS_FETCH_MIN_WINDOW_DAYS", "7"))


def _fetch_row_budget() -> int:
    return int(os.environ.get("HABITUS_FETCH_ROW_BUDGET", str(FETCH_ROW_BUDGET)))


def _fetch_min_window_days() -> int:
    return int(os.environ.get("HABITUS_FETCH_MIN_WINDOW_DAYS", str(FETCH_MIN_WINDOW_DAYS)))


def contamination_for_days(days: int) -> float:
    """Return the IsolationForest contamination parameter for the given training age.

    Ramps from very conservative (0.005) during the early warmup phase to the
    full 0.05 once the model has 90+ days of history.  Low contamination during
    warmup avoids the excessive false-positive rate that occurs when the model
    has seen very little historical behaviour.

    Args:
        days: Training age in days (0 = no history yet).

    Returns:
        Contamination fraction in the range 0.005–0.05.
    """
    if days < 7:
        return 0.005
    if days < 14:
        return 0.01
    if days < 30:
        return 0.02
    if days < 90:
        return 0.04
    return 0.05


def contamination_tier_name(days: int) -> str:
    """Return a stable string identifier for the contamination tier at a given age.

    Tier names are stored in ``run_state.json`` so that a tier transition can be
    detected on the next run and the model retrained with the new contamination.

    Args:
        days: Training age in days.

    Returns:
        One of ``"warmup"``, ``"early"``, ``"growing"``, ``"mature"``, or
        ``"established"``.
    """
    if days < 7:
        return "warmup"
    if days < 14:
        return "early"
    if days < 30:
        return "growing"
    if days < 90:
        return "mature"
    return "established"


def should_retrain_for_tier_change(state: dict, current_days: int) -> bool:
    """Return True when the contamination tier has advanced and a retrain is needed.

    Only triggers when a previous tier is on record — first runs always train
    normally and store the initial tier without this check.

    Args:
        state: Current ``run_state.json`` contents.
        current_days: Training age in days (computed from available history).

    Returns:
        True if the stored tier differs from the tier implied by ``current_days``.
    """
    old_tier = str(state.get("contamination_tier", ""))
    if not old_tier:
        return False
    return old_tier != contamination_tier_name(current_days)


MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
SUGGESTIONS_PATH = os.path.join(DATA_DIR, "suggestions.json")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")
STATE_PATH = os.path.join(DATA_DIR, "run_state.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "progress.json")
DATA_QUALITY_PATH = os.path.join(DATA_DIR, "data_quality.json")
RESCAN_FLAG = os.path.join(DATA_DIR, ".rescan_requested")

FEATURE_COLS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "total_power_w",
    "avg_temp_c",
    "sensor_changes",
    # Activity features — provide richer context beyond power alone
    "lights_on",
    "motion_events",
    "presence_count",
    "people_home_pct",
    "media_active",
    "door_events",
    "outdoor_temp_c",
    "activity_diversity",
    # Grid energy delta — kWh proxy, cross-validates watt sensors
    "grid_kwh_w",
    # Water — usage proxy + leak detection
    "water_l_per_h",  # pump watts running = water flowing
    "water_leak",  # binary leak sensor
    # Gas — smart meter m³/h
    "gas_m3_per_h",
]

# Domains that produce useful behavioral time-series data
BEHAVIORAL_DOMAINS = {
    "sensor",
    "binary_sensor",
    "light",
    "switch",
    "climate",
    "media_player",
    "cover",
    "fan",
    "lock",
    "alarm_control_panel",
    "input_boolean",
    "input_number",
}

# Entity ID fragments that are useless noise — always skip these
SKIP = [
    "rssi",
    "signal_strength",
    "lqi",
    "link_quality",
    "fossil_fuel",
    "co2_intensity",
    "grid_fossil",
    "firmware",
    "software_version",
    "uptime",
    "battery_level",  # device battery % — not a behavioral signal
    # Phone/mobile data — monotonically increasing, not home behavior
    "_mobile_rx",
    "_mobile_tx",
    "_app_rx",
    "_app_tx",
    "_wifi_rx",
    "_wifi_tx",
    "steps_sensor",
    "step_sensor",
    "_bytes_",
    "data_received",
    "data_sent",
    "ip_address",
    "mac_address",
    "ssid",
    "latitude",
    "longitude",
    "update.",  # HA update entities
]

# Unit-of-measurement fragments to skip — purely diagnostic, not behavioral
SKIP_UNITS_IN_NAME = [
    "_db",
    "_dbm",
    "_lqi",
    "_rssi",
]


def is_behavioral(eid: str) -> bool:
    """Return True if this entity ID is worth tracking for behavioral analysis.

    Strategy: accept anything from a useful domain, then drop known-noise
    suffixes/prefixes.  This keeps ~5-10× more sensors than the old keyword
    allowlist while still excluding diagnostic junk.
    """
    e = eid.lower()
    domain = e.split(".")[0]

    # Must be a domain that produces time-series state values
    if domain not in BEHAVIORAL_DOMAINS:
        return False

    # Skip known noise patterns
    if any(k in e for k in SKIP):
        return False
    return not any(k in e for k in SKIP_UNITS_IN_NAME)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def set_progress(phase, done=0, total=0, rows=0, elapsed=0.0, eta=0.0):
    try:
        pct = round(done / total * 100) if total else 100
        with open(PROGRESS_PATH, "w") as f:
            json.dump(
                {
                    "running": True,
                    "phase": phase,
                    "done": done,
                    "total": total,
                    "pct": pct,
                    "rows": rows,
                    "elapsed_min": round(elapsed / 60, 1),
                    "eta_min": round(eta / 60, 1),
                },
                f,
            )
    except Exception:
        pass


def clear_progress():
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)


def clamp_fetch_window_by_row_budget(
    start_iso: str,
    end_iso: str,
    entity_count: int,
    row_budget: int | None = None,
    min_window_days: int | None = None,
) -> tuple[str, bool, dict[str, int | str]]:
    """Clamp history window to a deterministic max rows budget.

    Uses a conservative estimate for hourly statistics rows:
    ``estimated_rows ≈ entities × hours``.
    """
    budget = _fetch_row_budget() if row_budget is None else int(row_budget)
    min_days = _fetch_min_window_days() if min_window_days is None else int(min_window_days)
    info: dict[str, int | str] = {
        "requested_hours": 0,
        "max_hours": 0,
        "row_budget": budget,
        "entity_count": max(0, int(entity_count)),
    }

    if budget <= 0 or entity_count <= 0:
        return start_iso, False, info

    try:
        start_dt = pd.to_datetime(start_iso, utc=True)
        end_dt = pd.to_datetime(end_iso, utc=True)
    except Exception:
        return start_iso, False, info

    if end_dt <= start_dt:
        return start_iso, False, info

    requested_hours = max(1, int((end_dt - start_dt).total_seconds() // 3600))
    max_hours = max(int(min_days) * 24, max(1, budget // max(1, int(entity_count))))
    info["requested_hours"] = requested_hours
    info["max_hours"] = max_hours

    if requested_hours <= max_hours:
        return start_iso, False, info

    clamped_start = (end_dt - datetime.timedelta(hours=max_hours)).floor("h")
    clamped_start_iso = clamped_start.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return clamped_start_iso, True, info


# ── WebSocket ──────────────────────────────────────────────────────────────────
async def ws_connect():
    import websockets

    ws = await websockets.connect(HA_WS, max_size=50 * 1024 * 1024)
    await ws.recv()
    await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
    result = json.loads(await ws.recv())
    if result.get("type") != "auth_ok":
        raise RuntimeError(f"Auth failed: {result}")
    return ws


def _resolve_db_path() -> str | None:
    configured = os.environ.get("HABITUS_HA_DB", "").strip()
    candidates = [
        configured,
        "/homeassistant/home-assistant_v2.db",
        "/config/home-assistant_v2.db",
        "/mnt/data/supervisor/homeassistant/home-assistant_v2.db",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _sqlite_connect():
    if os.environ.get("HABITUS_FORCE_API", "").lower() == "true":
        return None
    db_path = _resolve_db_path()
    if not db_path:
        log.error("Recorder DB not found (checked HABITUS_HA_DB,/homeassistant,/config,/mnt/data)")
        return None
    try:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except Exception as e:
        log.error("Failed opening recorder DB (%s): %s", db_path, e)
        return None


async def get_stat_ids() -> tuple[list[str], int]:
    conn = _sqlite_connect()
    if not conn:
        raise RuntimeError("Recorder DB unavailable; direct SQL required")
    try:
        cur = conn.cursor()
        cur.execute("SELECT statistic_id FROM statistics_meta")
        all_ids = [r[0] for r in cur.fetchall()]
        conn.close()
        behavioral = [e for e in all_ids if is_behavioral(e)]
        log.info(
            "Found %d behavioral sensors with long-term stats (from %d total stats ids)",
            len(behavioral),
            len(all_ids),
        )
        return behavioral, len(all_ids)
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        raise RuntimeError(f"SQLite stats_meta read failed: {e}") from e


async def get_behavioral_entity_ids() -> list[str]:
    """Return behavioral entity ids from recorder metadata (includes non-stat entities)."""
    conn = _sqlite_connect()
    if not conn:
        raise RuntimeError("Recorder DB unavailable; direct SQL required")
    try:
        cur = conn.cursor()
        cur.execute("SELECT entity_id FROM states_meta")
        ids = [r[0] for r in cur.fetchall()]
        conn.close()
        return [e for e in ids if e and is_behavioral(e)]
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        raise RuntimeError(f"SQLite states_meta read failed: {e}") from e


def _state_to_numeric(state: object) -> float | None:
    """Best-effort conversion of HA state strings to numeric values."""
    if state is None:
        return None
    s = str(state).strip().lower()
    if s in {"unknown", "unavailable", "none", "null", ""}:
        return None
    try:
        return float(s)
    except Exception:
        pass
    on_states = {"on", "open", "opening", "home", "playing", "heat", "cool", "auto", "true", "armed"}
    off_states = {"off", "closed", "closing", "not_home", "idle", "standby", "false", "disarmed", "clear"}
    if s in on_states:
        return 1.0
    if s in off_states:
        return 0.0
    return None


def fetch_recent_raw_history(entity_ids: list[str], start_iso: str, end_iso: str) -> pd.DataFrame:
    """Fetch raw history for entities that lack long-term recorder statistics.

    Preferred path: direct SQLite read from recorder DB.
    Fallback path: HA /api/history/period (dev only).

    Includes one latest-state synthetic row for entities with no history rows in
    the requested range, so sparse/non-changing sensors still participate.
    """
    if not entity_ids:
        return pd.DataFrame()

    conn = _sqlite_connect()
    if conn:
        try:
            start_ts = pd.to_datetime(start_iso, utc=True).timestamp()
            end_ts = pd.to_datetime(end_iso, utc=True).timestamp()
            rows: list[dict] = []
            batch = int(os.environ.get("HABITUS_SQL_BATCH", "40"))
            cur = conn.cursor()
            for i in range(0, len(entity_ids), batch):
                chunk = entity_ids[i:i + batch]
                if not chunk:
                    continue
                ph = ",".join(["?"] * len(chunk))
                q = f"""
                    SELECT m.entity_id, s.last_updated_ts, s.state
                    FROM states s
                    JOIN states_meta m ON s.metadata_id = m.metadata_id
                    WHERE m.entity_id IN ({ph})
                      AND s.last_updated_ts >= ? AND s.last_updated_ts <= ?
                    ORDER BY s.last_updated_ts
                """
                cur.execute(q, [*chunk, start_ts, end_ts])
                for eid, ts, state in cur.fetchall():
                    val = _state_to_numeric(state)
                    if not eid or val is None:
                        continue
                    rows.append({"entity_id": eid, "ts": ts, "mean": val, "sum": None})

            # Ensure sparse/non-changing entities are still represented once
            seen = {r["entity_id"] for r in rows}
            missing = [e for e in entity_ids if e not in seen]
            if missing:
                for i in range(0, len(missing), batch):
                    chunk = missing[i:i + batch]
                    ph = ",".join(["?"] * len(chunk))
                    q_latest = f"""
                        SELECT m.entity_id, s.last_updated_ts AS ts, s.state
                        FROM states s
                        JOIN states_meta m ON s.metadata_id = m.metadata_id
                        JOIN (
                            SELECT m2.entity_id AS entity_id, MAX(s2.last_updated_ts) AS max_ts
                            FROM states s2
                            JOIN states_meta m2 ON s2.metadata_id = m2.metadata_id
                            WHERE m2.entity_id IN ({ph})
                            GROUP BY m2.entity_id
                        ) latest ON latest.entity_id = m.entity_id AND latest.max_ts = s.last_updated_ts
                    """
                    cur.execute(q_latest, chunk)
                    for eid, ts, state in cur.fetchall():
                        if ts is None:
                            continue
                        val = _state_to_numeric(state)
                        if not eid or val is None:
                            continue
                        # stamp at end window so it joins current-hour inference
                        rows.append({"entity_id": eid, "ts": end_ts, "mean": val, "sum": None})
            conn.close()
        except Exception as e:
            log.warning("SQLite raw history read failed: %s", e)
            try:
                conn.close()
            except Exception:
                pass
            rows = []

        if rows:
            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
            df = df.dropna(subset=["ts"])
            if df.empty:
                return pd.DataFrame()
            df["hour"] = df["ts"].dt.floor("h")
            out = df.groupby(["entity_id", "hour"], as_index=False)["mean"].mean()
            out = out.rename(columns={"hour": "ts"})
            out["sum"] = None
            log.info("Fetched raw short-term history for %d non-stat entities (%d hourly rows)", len(entity_ids), len(out))
            return out[["entity_id", "ts", "mean", "sum"]]

    # Fallback to API (dev only)
    if os.environ.get("HABITUS_FORCE_API", "").lower() != "true":
        return pd.DataFrame()

    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    rows: list[dict] = []
    batch_size = 25

    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i : i + batch_size]
        try:
            params = {
                "end_time": end_iso,
                "filter_entity_id": ",".join(batch),
                "minimal_response": "true",
                "no_attributes": "true",
                "significant_changes_only": "false",
            }
            r = requests.get(
                f"{HA_URL}/api/history/period/{start_iso}",
                headers=headers,
                params=params,
                timeout=60,
            )
            if r.status_code != 200:
                continue
            payload = r.json()
            if not isinstance(payload, list):
                continue
            for series in payload:
                if not isinstance(series, list):
                    continue
                for p in series:
                    if not isinstance(p, dict):
                        continue
                    eid = p.get("entity_id")
                    val = _state_to_numeric(p.get("state"))
                    if not eid or val is None:
                        continue
                    ts = p.get("last_changed") or p.get("last_updated")
                    if not ts:
                        continue
                    rows.append({"entity_id": eid, "ts": ts, "mean": val, "sum": None})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    if df.empty:
        return pd.DataFrame()

    df["hour"] = df["ts"].dt.floor("h")
    out = df.groupby(["entity_id", "hour"], as_index=False)["mean"].mean()
    out = out.rename(columns={"hour": "ts"})
    out["sum"] = None
    log.info("Fetched raw short-term history for %d non-stat entities (%d hourly rows)", len(entity_ids), len(out))
    return out[["entity_id", "ts", "mean", "sum"]]


def get_ha_entity_count() -> int:
    """Return total number of HA entities (recorder metadata)."""
    conn = _sqlite_connect()
    if not conn:
        return 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM states_meta")
        total = int(cur.fetchone()[0])
        conn.close()
        return total
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return 0


def fetch_stats_sqlite(entity_ids, start_iso, end_iso=None):
    """Fetch long-term statistics directly from HA SQLite DB.

    Preferred path for Supervisor app/add-on installs.
    Falls back to API path if DB is unavailable.
    """
    import time as _t

    if os.environ.get("HABITUS_FORCE_API", "").lower() == "true":
        return pd.DataFrame()

    db_path = _resolve_db_path()
    if not db_path:
        return pd.DataFrame()

    if end_iso is None:
        end_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:00:00+00:00")

    start_iso_eff, was_clamped, clamp_info = clamp_fetch_window_by_row_budget(
        start_iso,
        end_iso,
        len(entity_ids),
    )
    if was_clamped:
        log.warning(
            "Stats fetch window clamped (sqlite): budget=%s rows, entities=%s, requested=%sh, max=%sh, range=%s → %s",
            clamp_info["row_budget"],
            clamp_info["entity_count"],
            clamp_info["requested_hours"],
            clamp_info["max_hours"],
            start_iso,
            start_iso_eff,
        )

    try:
        start_ts = pd.to_datetime(start_iso_eff, utc=True).timestamp()
        end_ts = pd.to_datetime(end_iso, utc=True).timestamp()
    except Exception:
        return pd.DataFrame()

    rows = []
    total = len(entity_ids)
    done = 0
    t0 = _t.time()
    row_budget = max(0, int(_fetch_row_budget()))
    hit_budget = False

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()

        # Fast path: query in chunks of statistic_ids (much faster than per-sensor loop)
        batch = int(os.environ.get("HABITUS_SQL_BATCH", "100"))
        for i in range(0, len(entity_ids), batch):
            chunk = entity_ids[i : i + batch]
            if not chunk:
                continue
            ph = ",".join(["?"] * len(chunk))
            cur.execute(
                f"""
                SELECT sm.statistic_id AS entity_id, s.start_ts AS ts, s.mean AS mean, s.sum AS sum
                FROM statistics s
                JOIN statistics_meta sm ON s.metadata_id = sm.id
                WHERE sm.statistic_id IN ({ph})
                  AND s.start_ts >= ? AND s.start_ts <= ?
                ORDER BY sm.statistic_id, s.start_ts
                """,
                [*chunk, start_ts, end_ts],
            )
            batch_rows = cur.fetchall()

            if row_budget > 0:
                remaining = row_budget - len(rows)
                if remaining <= 0:
                    hit_budget = True
                    break
                if len(batch_rows) > remaining:
                    rows.extend(batch_rows[:remaining])
                    hit_budget = True
                else:
                    rows.extend(batch_rows)
            else:
                rows.extend(batch_rows)

            done += len(chunk)
            elapsed = _t.time() - t0
            eta = (elapsed / done) * (total - done) if done else 0
            set_progress("fetching", done, total, len(rows), elapsed, eta)
            log.info("  %d/%d sensors — %d rows", done, total, len(rows))
            if hit_budget:
                break

        conn.close()
    except Exception as e:
        log.warning("Direct SQLite stats read failed: %s", e)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    if hit_budget:
        log.warning(
            "Stats fetch row budget reached (sqlite): kept %d rows (budget=%d)",
            len(rows),
            row_budget,
        )
    df = pd.DataFrame(rows, columns=["entity_id", "ts", "mean", "sum"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    log.info("Fetched %d rows from SQLite | %s → %s", len(df), df["ts"].min().date(), df["ts"].max().date())
    return df


async def fetch_stats(entity_ids, start_iso, end_iso=None):
    """Fetch hourly statistics and return an aggregated DataFrame.

    Preferred path: direct SQLite reads.
    Fallback path: batched WebSocket API calls.
    """
    if end_iso is None:
        end_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:00:00+00:00")
    if not entity_ids:
        return pd.DataFrame()

    start_iso_eff, was_clamped, clamp_info = clamp_fetch_window_by_row_budget(
        start_iso,
        end_iso,
        len(entity_ids),
    )
    if was_clamped:
        log.warning(
            "Stats fetch window clamped: budget=%s rows, entities=%s, requested=%sh, max=%sh, range=%s → %s",
            clamp_info["row_budget"],
            clamp_info["entity_count"],
            clamp_info["requested_hours"],
            clamp_info["max_hours"],
            start_iso,
            start_iso_eff,
        )

    df_sql = fetch_stats_sqlite(entity_ids, start_iso_eff, end_iso)
    if os.environ.get("HABITUS_FORCE_API", "").lower() != "true":
        return df_sql

    all_rows = []
    done = 0
    total = len(entity_ids)
    import time as _t

    batch_size = int(os.environ.get("HABITUS_STATS_BATCH", "120"))
    max_retries = int(os.environ.get("HABITUS_STATS_RETRIES", "3"))
    row_budget = max(0, int(_fetch_row_budget()))
    hit_budget = False

    t0 = _t.time()
    for i in range(0, total, batch_size):
        batch = entity_ids[i:i + batch_size]
        ok = False

        for attempt in range(1, max_retries + 1):
            ws = None
            try:
                ws = await ws_connect()
                await ws.send(
                    json.dumps(
                        {
                            "id": 1,
                            "type": "recorder/statistics_during_period",
                            "start_time": start_iso_eff,
                            "end_time": end_iso,
                            "statistic_ids": batch,
                            "period": "hour",
                            "types": ["mean", "sum"],
                        }
                    )
                )
                result = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                payload = result.get("result", {}) or {}
                for sid in sorted(payload.keys()):
                    points = payload.get(sid) or []
                    for p in sorted(points, key=lambda x: x.get("start", 0)):
                        ts = p["start"]
                        if ts > 1e10:
                            ts /= 1000
                        all_rows.append(
                            {"entity_id": sid, "ts": ts, "mean": p.get("mean"), "sum": p.get("sum")}
                        )
                        if row_budget > 0 and len(all_rows) >= row_budget:
                            hit_budget = True
                            break
                    if hit_budget:
                        break
                ok = True
                break
            except Exception as e:
                if attempt >= max_retries:
                    log.warning(
                        "Batch %d-%d failed after %d attempts: %s",
                        i,
                        i + len(batch) - 1,
                        max_retries,
                        e,
                    )
                else:
                    await asyncio.sleep(min(5 * attempt, 15))
            finally:
                try:
                    if ws is not None:
                        await ws.close()
                except Exception:
                    pass

        done += len(batch)
        elapsed = _t.time() - t0
        eta = (elapsed / done) * (total - done) if done else 0
        set_progress("fetching", done, total, len(all_rows), elapsed, eta)
        if ok:
            log.info("  %d/%d sensors — %d rows", done, total, len(all_rows))
        if hit_budget:
            log.warning(
                "Stats fetch row budget reached (api): kept %d rows (budget=%d)",
                len(all_rows),
                row_budget,
            )
            break

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    log.info("Fetched %d rows | %s → %s", len(df), df["ts"].min().date(), df["ts"].max().date())
    return df


# ── Features ───────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build hourly feature matrix combining power and activity signals.

    Merges power metrics (total load, temperature, sensor change rate) with
    activity features (lights, motion, presence, media) so the model can
    distinguish a cold-morning heating spike from a genuine anomaly.

    Args:
        df: Raw stats DataFrame from ``fetch_stats`` with columns
            ``[entity_id, ts, mean, sum]``.

    Returns:
        DataFrame with one row per hour and all FEATURE_COLS populated.
    """
    df = df.copy()
    df["hour"] = df["ts"].dt.floor("h")
    hours = pd.DataFrame({"hour": pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")})
    hours["hour_of_day"] = hours["hour"].dt.hour
    hours["day_of_week"] = hours["hour"].dt.dayofweek
    hours["is_weekend"] = (hours["day_of_week"] >= 5).astype(int)
    hours["month"] = hours["hour"].dt.month
    _max_w = int(os.environ.get("HABITUS_MAX_POWER_KW", "25")) * 1000
    _power_entity = os.environ.get("HABITUS_POWER_ENTITY", "").strip()
    _energy_grid = os.environ.get("HABITUS_ENERGY_GRID", "").strip()
    _energy_rates = [e for e in os.environ.get("HABITUS_ENERGY_RATES", "").split(",") if e]

    grid_kwh_w = pd.Series(dtype=float, name="grid_kwh_w")  # init before branches
    if _power_entity:
        # Explicit override — use as-is (watts)
        power = df[df["entity_id"] == _power_entity].copy()
        power["v"] = pd.to_numeric(power["mean"], errors="coerce").clip(lower=0, upper=_max_w)
        total_power = power.groupby("hour")["v"].max().rename("total_power_w")
    elif _energy_grid:
        # Grid kWh entity from HA Energy Dashboard — convert hourly delta to W
        grid = df[df["entity_id"] == _energy_grid].copy()
        grid["v"] = pd.to_numeric(grid["mean"], errors="coerce").clip(lower=0)
        grid = grid.set_index("hour").sort_index()
        # kWh delta per hour × 1000 = average watts for that hour
        kwh_per_hour = grid["v"].diff().clip(lower=0, upper=_max_w / 1000)
        grid_kwh_w = (kwh_per_hour * 1000).rename("grid_kwh_w")
        total_power = grid_kwh_w.rename("total_power_w")
    elif _energy_rates:
        # Per-device watt sensors from Energy Dashboard — sum these (no overlap)
        power = df[df["entity_id"].isin(_energy_rates)].copy()
        power["v"] = pd.to_numeric(power["mean"], errors="coerce").clip(lower=0, upper=_max_w)
        total_power = power.groupby("hour")["v"].sum().rename("total_power_w")
    else:
        # Fallback: max of any power-like sensor (avoids double-counting)
        power = df[
            df["entity_id"].str.contains(
                "consumed_w|watt|power|inverter|load", case=False, na=False
            )
        ].copy()
        power["v"] = pd.to_numeric(power["mean"], errors="coerce").clip(lower=0, upper=_max_w)
        total_power = power.groupby("hour")["v"].max().rename("total_power_w")
    temp = df[df["entity_id"].str.contains("temperature", case=False, na=False)].copy()
    temp["v"] = pd.to_numeric(temp["mean"], errors="coerce")
    avg_temp = temp.groupby("hour")["v"].mean().rename("avg_temp_c")
    activity = df.groupby("hour").size().rename("sensor_changes")
    features = hours.set_index("hour")
    for s in [total_power, avg_temp, activity, grid_kwh_w]:
        features = features.join(s, how="left")
    # Merge activity features from activity engine
    try:
        act_features = activity_engine.extract_activity_features(df)
        if not act_features.empty:
            act_features = act_features.set_index("hour")
            features = features.join(act_features, how="left")
    except Exception as e:
        log.warning("Activity feature extraction failed: %s", e)
    # Water pump wattage — proxy for water usage (pump running = tap/appliance running)
    water_pump = df[
        df["entity_id"].str.contains(
            "waterpump.*_w$|water.*pump.*power|water.*consumption.*_w", case=False, na=False
        )
    ].copy()
    if not water_pump.empty:
        water_pump["v"] = pd.to_numeric(water_pump["mean"], errors="coerce").clip(
            lower=0, upper=5000
        )
        wp_series = water_pump.groupby("hour")["v"].max().rename("water_l_per_h")
        features = features.join(wp_series, how="left")

    # Water leak sensors — 1 if any leak detected in that hour
    leaks = df[
        df["entity_id"].str.contains("water_leak|bilge.*leak|leak.*detect", case=False, na=False)
    ].copy()
    if not leaks.empty:
        leak_col = "state" if "state" in leaks.columns else "mean"

        def _leak_to_float(v) -> float:
            if pd.isna(v):
                return 0.0
            if isinstance(v, str):
                return 1.0 if v.strip().lower() in {"on", "true", "wet", "detected", "1"} else 0.0
            with contextlib.suppress(TypeError, ValueError):
                return 1.0 if float(v) > 0 else 0.0
            return 0.0

        leaks["v"] = leaks[leak_col].map(_leak_to_float)
        leak_series = leaks.groupby("hour")["v"].max().rename("water_leak")
        features = features.join(leak_series, how="left")

    # Ensure all FEATURE_COLS exist (pad with zeros if activity extraction failed)
    for col in FEATURE_COLS:
        if col not in features.columns:
            features[col] = 0.0
    return features.fillna(0).reset_index()


# ── Model ──────────────────────────────────────────────────────────────────────
def train_model(features: pd.DataFrame, training_days: int = 0) -> tuple:
    """Train an IsolationForest model with contamination scaled to training age.

    Uses :func:`contamination_for_days` to select a conservative contamination
    value during the early learning phase and ramps up to the steady-state value
    once enough history has been accumulated.

    Args:
        features: Hourly feature matrix produced by :func:`build_features`.
        training_days: Age of the training data in days.  Defaults to 0
            (conservative warmup contamination) when not supplied.

    Returns:
        Tuple of ``(model, scaler)`` — a fitted IsolationForest and the
        StandardScaler used to normalise the training data.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = features[FEATURE_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    contamination = contamination_for_days(training_days)
    tier = contamination_tier_name(training_days)
    log.info(
        "Training IsolationForest: contamination=%.4f tier=%s days=%d",
        contamination,
        tier,
        training_days,
    )
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(Xs)
    days = round((features["hour"].max() - features["hour"].min()).total_seconds() / 86400)
    log.info(f"Main model trained on {len(X):,} snapshots ({days}d)")
    return model, scaler


def save_artifacts(model, scaler, features):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    baseline = {}
    for (h, d), g in features.groupby(["hour_of_day", "day_of_week"]):
        baseline[f"{h}_{d}"] = {
            "mean_power": round(float(g["total_power_w"].mean()), 1),
            "std_power": round(float(g["total_power_w"].std()), 1),
            "mean_temp": round(float(g["avg_temp_c"].mean()), 1),
            "n_samples": len(g),
        }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, default=str)
    log.info(f"Artifacts saved ({os.path.getsize(MODEL_PATH)//1024}KB model)")


def score_current(features):
    # During warmup grace period, never report anomaly — assume normal
    training_days = 0
    with contextlib.suppress(Exception):
        training_days = round(
            (features["hour"].max() - features["hour"].min()).total_seconds() / 86400
        )
    if training_days < MIN_SCORING_DAYS:
        log.info(
            "Warmup: only %d training days — score capped at 0 (need %d)",
            training_days,
            MIN_SCORING_DAYS,
        )
        return 0

    now = datetime.datetime.now(datetime.UTC).replace(minute=0, second=0, microsecond=0)
    row = features[features["hour"] == pd.Timestamp(now)]
    if not row.empty:
        X = row[FEATURE_COLS].values
    else:
        # Fallback: build a zero-padded row matching FEATURE_COLS exactly
        zeros = [0.0] * len(FEATURE_COLS)
        zeros[FEATURE_COLS.index("hour_of_day")] = now.hour
        zeros[FEATURE_COLS.index("day_of_week")] = now.weekday()
        zeros[FEATURE_COLS.index("is_weekend")] = int(now.weekday() >= 5)
        zeros[FEATURE_COLS.index("month")] = now.month
        X = np.array([zeros])
    score, model_used = seasonal.score_with_best_model(X)
    log.info(f"Scored with {model_used} model")
    return score


# ── Notify ────────────────────────────────────────────────────────────────────
def send_notification(title, message):
    notify_on = os.environ.get("HABITUS_NOTIFY_ON", "true").lower() == "true"
    if not notify_on:
        return
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    service = NOTIFY_SVC.replace(".", "/")
    try:
        r = requests.post(
            f"{HA_URL}/api/services/{service}",
            headers=headers,
            json={
                "title": title,
                "message": message,
                "data": {
                    "url": "/hassio/ingress/57582523_habitus",
                    "clickAction": "/hassio/ingress/57582523_habitus",
                },
            },
            timeout=5,
        )
        log.info(f"Notification sent via {NOTIFY_SVC}: {r.status_code}")
    except Exception as e:
        log.warning(f"Notification failed: {e}")


def format_digest_message(
    anomaly_score: int,
    entity_anomalies: list[dict],
    suggestions: list[dict],
    training_days: int,
    entity_count: int,
) -> str:
    """Format a plain-English daily digest message for Habitus.

    Args:
        anomaly_score: Current 0–100 anomaly score.
        entity_anomalies: Top anomalous entities with description strings.
        suggestions: List of suggested automations (each with a ``title`` key).
        training_days: Days of history the model was trained on.
        entity_count: Number of behavioral sensors being tracked.

    Returns:
        Formatted multi-line message string suitable for an HA notification body.
    """
    parts = [
        f"Anomaly score: {anomaly_score}/100 · {entity_count} sensors · {training_days}d history"
    ]
    if entity_anomalies:
        parts.append("\nTop anomalies:")
        for a in entity_anomalies[:3]:
            parts.append(f"  • {a.get('description', a.get('entity', ''))}")
    else:
        parts.append("\nNo anomalies detected — all systems normal.")
    if suggestions:
        n = len(suggestions)
        parts.append(f"\n{n} automation suggestion{'s' if n != 1 else ''} available in Habitus.")
    return "\n".join(parts)


def should_send_digest(
    state: dict,
    now: datetime.datetime | None = None,
) -> bool:
    """Return True if the daily digest should be sent now.

    The digest fires once per day at ``DAILY_DIGEST_HOUR`` (UTC) when
    ``DAILY_DIGEST`` is enabled and has not already fired today.

    Args:
        state: Current run state dict, may contain ``last_digest_date``.
        now: Override current time for testing. Defaults to ``datetime.now(UTC)``.

    Returns:
        True if a digest notification should be sent, False otherwise.
    """
    if not DAILY_DIGEST:
        return False
    if now is None:
        now = datetime.datetime.now(datetime.UTC)
    if now.hour != DAILY_DIGEST_HOUR:
        return False
    today = now.strftime("%Y-%m-%d")
    return state.get("last_digest_date") != today


def send_daily_digest(
    state: dict,
    anomaly_score: int,
    entity_anomalies: list[dict],
    suggestions: list[dict],
    training_days: int,
    entity_count: int,
    now: datetime.datetime | None = None,
) -> dict:
    """Send a daily summary digest notification and return an updated state dict.

    No-ops when ``DAILY_DIGEST`` is disabled, the digest hour has not arrived,
    or a digest has already been sent today.  The caller's ``state`` dict is
    never mutated; a shallow copy is returned when an update is required.

    Args:
        state: Current run state dict (will not be mutated).
        anomaly_score: Current 0–100 anomaly score.
        entity_anomalies: Top anomalous entities with description strings.
        suggestions: Suggested automation list (each with a ``title`` key).
        training_days: Days of history used for training.
        entity_count: Number of behavioral sensors tracked.
        now: Override current time for testing.

    Returns:
        Updated state dict with ``last_digest_date`` set when digest was sent,
        or the original ``state`` dict unchanged when no digest fired.
    """
    if now is None:
        now = datetime.datetime.now(datetime.UTC)
    if not should_send_digest(state, now):
        return state
    msg = format_digest_message(
        anomaly_score, entity_anomalies, suggestions, training_days, entity_count
    )
    send_notification("🧠 Habitus — Daily Digest", msg)
    updated = dict(state)
    updated["last_digest_date"] = now.strftime("%Y-%m-%d")
    return updated


# ── Publish ───────────────────────────────────────────────────────────────────
def publish(entity_id, state, attributes=None):
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(
            f"{HA_URL}/api/states/{entity_id}",
            headers=headers,
            json={"state": str(state), "attributes": attributes or {}},
            timeout=5,
        )
        if r.status_code not in (200, 201):
            log.error(f"Publish {entity_id}: {r.status_code}")
    except Exception as e:
        log.error(f"Publish {entity_id}: {e}")


async def get_energy_entities() -> dict:
    """Query HA Energy Dashboard config and return best power entities.

    Returns:
        dict with keys:
          - grid_kwh: grid consumption entity (kWh, cumulative)
          - grid_rate: grid consumption rate entity (W, instantaneous) if available
          - device_rates: list of per-device watt sensors from energy dashboard
    """
    try:
        async with websockets.connect(HA_WS_URL) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
            r = json.loads(await ws.recv())
            if r["type"] != "auth_ok":
                return {}
            await ws.send(json.dumps({"id": 1, "type": "energy/get_prefs"}))
            r = json.loads(await ws.recv())
            prefs = r.get("result", {})

        result = {}
        for src in prefs.get("energy_sources", []):
            if src.get("type") == "grid":
                flows = src.get("flow_from", [])
                if flows:
                    result["grid_kwh"] = flows[0].get("stat_energy_from")
                    # Pull price from Energy Dashboard config
                    price = flows[0].get("number_energy_price")
                    if price:
                        result["kwh_price"] = float(price)
                    elif flows[0].get("entity_energy_price"):
                        result["kwh_price_entity"] = flows[0]["entity_energy_price"]

        # Per-device watt-rate sensors (instantaneous W)
        result["device_rates"] = [
            d["stat_rate"] for d in prefs.get("device_consumption", []) if d.get("stat_rate")
        ]
        log.info(
            "Energy config: grid=%s, %d device rate sensors",
            result.get("grid_kwh"),
            len(result.get("device_rates", [])),
        )
        return result
    except Exception as e:
        log.warning("Could not fetch energy prefs: %s", e)
        return {}


def persistent_notification(notif_id: str, title: str, message: str) -> None:
    """Create or update a HA persistent notification (appears in bell icon everywhere)."""
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    try:
        # Clear old notification first
        requests.post(
            f"{HA_URL}/api/services/persistent_notification/dismiss",
            headers=headers,
            json={"notification_id": notif_id},
            timeout=5,
        )
        requests.post(
            f"{HA_URL}/api/services/persistent_notification/create",
            headers=headers,
            json={
                "notification_id": notif_id,
                "title": title,
                "message": message,
            },
            timeout=5,
        )
        log.info(f"Persistent notification updated: {notif_id}")
    except Exception as e:
        log.warning(f"Persistent notification failed: {e}")


def publish_dashboard_entities(
    anomaly_score: int,
    entity_anomalies: list,
    suggestions: list,
) -> None:
    """Publish text sensors + persistent notifications so anomalies/suggestions
    surface automatically on any HA dashboard without user interaction."""
    # 1. Text sensors (usable in markdown/entities cards)
    top_anomaly = entity_anomalies[0]["description"] if entity_anomalies else "None detected"
    publish(
        "sensor.habitus_top_anomaly",
        top_anomaly[:255],
        {"friendly_name": "Habitus Top Anomaly", "icon": "mdi:alert-circle-outline"},
    )
    for i, s in enumerate(suggestions[:3], 1):
        publish(
            f"sensor.habitus_suggestion_{i}",
            s.get("title", "")[:255],
            {
                "friendly_name": f"Habitus Suggestion {i}",
                "description": s.get("description", ""),
                "icon": "mdi:lightbulb-outline",
            },
        )

    # 2. Persistent notification — always visible in HA bell icon
    if anomaly_score >= 90:  # Only notify for genuinely critical anomalies
        lines = [f"**Score: {anomaly_score}/100**"]
        if entity_anomalies:
            lines.append("\n**Anomalies:**")
            for a in entity_anomalies[:3]:
                lines.append(f"- {a['description']}")
        if suggestions:
            lines.append("\n**Suggested automations:**")
            for s in suggestions[:3]:
                lines.append(f"- {s.get('title','')}")
        lines.append("\n[Open Habitus](/hassio/ingress/57582523_habitus)")
        persistent_notification(
            "habitus_anomaly",
            "🧠 Habitus — Unusual Activity",
            "\n".join(lines),
        )
    else:
        # Score normal — clear any previous anomaly notification
        headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
        with contextlib.suppress(Exception):
            requests.post(
                f"{HA_URL}/api/services/persistent_notification/dismiss",
                headers=headers,
                json={"notification_id": "habitus_anomaly"},
                timeout=5,
            )

    # 2b. Immediate leak alert — doesn't wait for anomaly score
    try:
        headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
        leak_states = requests.get(f"{HA_URL}/api/states", headers=headers, timeout=5).json()
        active_leaks = [
            s["entity_id"].replace("binary_sensor.", "").replace("_", " ")
            for s in leak_states
            if s["entity_id"].startswith("binary_sensor.")
            and any(
                k in s["entity_id"].lower()
                for k in ("leak_detected", "water_leak", "flood", "moisture")
            )
            and str(s.get("state", "")).lower() in ("on", "wet", "detected", "true")
        ]
        if active_leaks:
            persistent_notification(
                "habitus_leak",
                "🚨 Habitus — Water Leak Detected",
                "**Active leak sensors:**\n"
                + "\n".join(f"- {leak}" for leak in active_leaks)
                + "\n\nCheck immediately!",
            )
            log.warning("LEAK ALERT: %s", active_leaks)
        else:
            # Clear old leak notification if all clear
            requests.post(
                f"{HA_URL}/api/services/persistent_notification/dismiss",
                headers=headers,
                json={"notification_id": "habitus_leak"},
                timeout=5,
            )
    except Exception as e:
        log.warning("Leak check failed: %s", e)

    # 3. Always update suggestions notification so user sees new ideas
    if suggestions:
        # Suggestions available via Habitus UI — no persistent notification (too intrusive)
        pass


# ── Main ──────────────────────────────────────────────────────────────────────
async def run(days_history: int, mode: str = "full") -> None:
    """Main Habitus run loop — fetch, train (if needed), score, publish.

    In ``overnight`` schedule mode, daytime invocations pass ``mode='score'``
    to skip the expensive training step and only re-score the current hour
    using the existing model.  Training is deferred to the overnight window
    (default 02:00) when the home is typically idle.

    In ``continuous`` mode every run does a full retrain.

    Args:
        days_history: Maximum history window.  HA returns whatever it has.
        mode: ``"full"`` to fetch + train + score, ``"score"`` to score only.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    now_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:00:00+00:00")
    state = load_state()

    # Adaptive contamination — force retrain when training age crosses a tier boundary
    _stored_days = state.get("training_days", 0)
    if should_retrain_for_tier_change(state, _stored_days):
        log.info(
            "Contamination tier advanced: %s → %s (%d days) — forcing full retrain",
            state.get("contamination_tier"),
            contamination_tier_name(_stored_days),
            _stored_days,
        )
        mode = "full"

    if mode == "full":
        send_notification(
            "🧠 Habitus — Training started",
            "Fetching sensor history and retraining behavioral model. "
            "The app stays usable — data appears progressively as each phase completes.",
        )

    # ── Score-only mode ────────────────────────────────────────────────────────
    # Used in overnight schedule during daytime hours.
    # Re-scores current state using the existing trained model — fast (<5s).
    # Skips all data fetching and retraining.
    if mode == "score":
        if not os.path.exists(MODEL_PATH):
            log.info("No model yet — falling back to full run")
            mode = "full"
        else:
            log.info("Score-only mode — using existing model")
            # Run conflict detection on every score cycle (lightweight, real-time)
            try:
                conflicts = conflict_detector.detect_conflicts()
                conflict_detector.save_conflicts(conflicts)
            except Exception as e:
                log.warning("Conflict detection failed: %s", e)
            now = datetime.datetime.now(datetime.UTC).replace(minute=0, second=0, microsecond=0)
            X = np.array(
                [
                    [
                        now.hour,
                        now.weekday(),
                        int(now.weekday() >= 5),
                        now.month,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ]
            )
            anomaly_score, _ = seasonal.score_with_best_model(X)
            entity_anomalies = anomaly_breakdown.score_entities()
            weighted_entity_score = anomaly_breakdown.compute_weighted_score(entity_anomalies)
            if weighted_entity_score > 0:
                log.info("Confidence-weighted entity score: %.3f", weighted_entity_score)
            activity_summary = activity_engine.get_activity_summary()
            training_days = state.get("training_days", 0)
            entity_count = state.get("entity_count", 0)
            top_anomaly = entity_anomalies[0]["description"] if entity_anomalies else None
            state.update(
                {
                    "last_score": now_iso,
                    "anomaly_score": anomaly_score,
                    "mode": "score",
                    "requested_days": days_history,
                }
            )
            save_state(state)
            is_anomalous = anomaly_score > THRESHOLD
            log.info(
                "Score-only: %d/100 (%s)",
                anomaly_score,
                "⚠ ANOMALY" if is_anomalous else "✓ normal",
            )
            if is_anomalous:
                msg_parts = [f"Score: {anomaly_score}/100"]
                if top_anomaly:
                    msg_parts.append(top_anomaly)
                send_notification("🧠 Habitus — Unusual Activity", "\n".join(msg_parts))
            state = send_daily_digest(
                state, anomaly_score, entity_anomalies, [], training_days, entity_count
            )
            save_state(state)
            _publish_sensors(anomaly_score, is_anomalous, training_days, entity_count)
            await _register_lovelace_card()
            return

    # Load user-overridden settings from state.json (persist across restarts)
    _state_path = os.path.join(DATA_DIR, "state.json")
    try:
        with open(_state_path) as _sf:
            _saved = json.load(_sf).get("user_settings", {})
        if _saved.get("power_entity") and not os.environ.get("HABITUS_POWER_ENTITY"):
            os.environ["HABITUS_POWER_ENTITY"] = _saved["power_entity"]
            log.info("Loaded saved power entity from settings: %s", _saved["power_entity"])
        if "days_history" in _saved:
            try:
                days_history = int(_saved.get("days_history"))
                os.environ["HABITUS_DAYS"] = str(days_history)
                log.info("Loaded saved days_history from settings: %s", days_history)
            except Exception:
                pass
        if "notify_on_anomaly" in _saved:
            os.environ["HABITUS_NOTIFY_ON"] = "true" if bool(_saved.get("notify_on_anomaly")) else "false"
            log.info("Loaded saved notify setting from settings: %s", os.environ["HABITUS_NOTIFY_ON"])
    except Exception:
        pass

    # Auto-detect energy entities from HA Energy Dashboard
    if not os.environ.get("HABITUS_POWER_ENTITY") and not os.environ.get("HABITUS_ENERGY_GRID"):
        energy = await get_energy_entities()
        if energy.get("grid_kwh"):
            os.environ["HABITUS_ENERGY_GRID"] = energy["grid_kwh"]
            log.info("Using Energy Dashboard grid entity: %s", energy["grid_kwh"])
        if energy.get("kwh_price"):
            os.environ["HABITUS_KWH_PRICE"] = str(energy["kwh_price"])
            log.info(
                "Energy Dashboard price: %.2f %s/kWh",
                energy["kwh_price"],
                os.environ.get("HABITUS_CURRENCY", "kr"),
            )
        # Pull electricity price + currency directly from Energy Dashboard
        if energy.get("kwh_price") and not os.environ.get("HABITUS_KWH_PRICE_LOCKED"):
            os.environ["HABITUS_KWH_PRICE"] = str(energy["kwh_price"])
            log.info(
                "Energy Dashboard price: %s %s/kWh",
                energy["kwh_price"],
                os.environ.get("HABITUS_CURRENCY", "kr"),
            )
            # Prefer a real-time watt sensor over kWh delta — look for _w companion
            # e.g. sensor.foo_electric_consumption_kwh → sensor.foo_electric_consumption_w
            kwh_id = energy["grid_kwh"]
            watt_candidate = kwh_id.replace("_kwh", "_w").replace("_energy", "_power")
            try:
                headers = {"Authorization": f"Bearer {HA_TOKEN}"}
                r = requests.get(
                    f"{HA_URL}/api/states/{watt_candidate}", headers=headers, timeout=5
                )
                if r.status_code == 200:
                    uom = r.json().get("attributes", {}).get("unit_of_measurement", "")
                    if uom == "W":
                        os.environ["HABITUS_POWER_ENTITY"] = watt_candidate
                        log.info(
                            "Auto-detected watt sensor companion: %s (preferred over kWh delta)",
                            watt_candidate,
                        )
            except Exception as e:
                log.debug("Watt companion probe failed: %s", e)
        if energy.get("device_rates"):
            os.environ["HABITUS_ENERGY_RATES"] = ",".join(energy["device_rates"])
        if energy.get("gas"):
            os.environ["HABITUS_GAS_ENTITIES"] = ",".join(energy["gas"])
            log.info("Gas meters: %s", energy["gas"])
        if energy.get("water"):
            os.environ["HABITUS_WATER_ENTITIES"] = ",".join(energy["water"])
            log.info("Water meters: %s", energy["water"])

    stat_ids, total_stat_ids = await get_stat_ids()
    behavioral_entity_ids = await get_behavioral_entity_ids()
    stat_id_set = set(stat_ids)
    non_stat_ids = [e for e in behavioral_entity_ids if e not in stat_id_set]
    tracked_entity_count = len(stat_ids) + len(non_stat_ids)

    if not stat_ids and not non_stat_ids:
        log.error("No behavioral sensors found")
        return

    total_entities = get_ha_entity_count()
    log.info(
        "Behavioral coverage: %d with long-term stats + %d raw-only entities",
        len(stat_ids),
        len(non_stat_ids),
    )

    if state.get("data_to") and os.path.exists(MODEL_PATH):
        # Incremental
        fetch_from = state["data_to"]
        log.info(f"Incremental: {fetch_from} → {now_iso}")
        df_new = await fetch_stats(stat_ids, fetch_from, now_iso)
        if df_new.empty or len(df_new) < 24:
            log.info("Not enough new data — scoring with existing model")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            anomaly_score = seasonal.score_with_best_model(
                np.array(
                    [
                        [
                            datetime.datetime.now().hour,
                            datetime.datetime.now().weekday(),
                            int(datetime.datetime.now().weekday() >= 5),
                            datetime.datetime.now().month,
                            0,
                            0,
                            0,
                        ]
                    ]
                )
            )[0]
            training_days = state.get("training_days", 0)
            entity_count = state.get("entity_count", tracked_entity_count)
        else:
            cap = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days_history)
            cap_iso = cap.strftime("%Y-%m-%dT%H:00:00+00:00")
            saved = state.get("data_from", "2000-01-01T00:00:00+00:00")
            full_from = saved if saved > cap_iso else cap_iso
            log.info(f"Retraining {days_history}d window {full_from} → {now_iso}")
            set_progress("fetching", 0, len(stat_ids), 0, 0, 0)
            df_stats = await fetch_stats(stat_ids, full_from, now_iso) if stat_ids else pd.DataFrame()

            raw_max_days = int(os.environ.get("HABITUS_RAW_MAX_DAYS", "3650"))
            raw_days = min(days_history, raw_max_days)
            raw_from = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=raw_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            raw_to = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            df_raw = fetch_recent_raw_history(non_stat_ids, raw_from, raw_to)

            frames = [x for x in (df_stats, df_raw) if x is not None and not x.empty]
            if not frames:
                log.warning("No data")
                return
            df = pd.concat(frames, ignore_index=True)
            unique_entity_count = int(df["entity_id"].nunique())

            set_progress("building_baselines", len(stat_ids), len(stat_ids), len(df), 0, 0)
            log.info("Building entity baselines...")
            anomaly_breakdown.build_entity_baselines(df)
            try:
                log.info("Building feature matrix...")
                features = build_features(df)
                log.info("Feature matrix built: %d rows", len(features))
            except Exception as e:
                import traceback, sys
                log.error("CRASH in build_features: %s", e)
                traceback.print_exc()
                raise
            del df
            set_progress("training", len(stat_ids), len(stat_ids), len(features), 0, 0)
            log.info(f"Training IsolationForest on {len(features):,} rows...")
            _train_days = round(
                (features["hour"].max() - features["hour"].min()).total_seconds() / 86400
            )
            model, scaler = train_model(features, _train_days)
            save_artifacts(model, scaler, features)
            # Partial score after training — baseline + initial anomaly data visible
            _partial_score = score_current(features)
            training_days = _train_days
            _warming_up = training_days < MIN_SCORING_DAYS
            state.update(
                {
                    "phase": "warming_up" if _warming_up else "model_ready",
                    "anomaly_score": _partial_score,
                    "training_days": _train_days,
                    "warming_up": training_days < MIN_SCORING_DAYS,
                    "warmup_days_remaining": max(0, MIN_SCORING_DAYS - training_days),
                    "contamination_tier": contamination_tier_name(_train_days),
                }
            )
            save_state(state)
            log.info(f"Model ready — preliminary score {_partial_score}/100")
            # Only train seasonal models with enough data (need ≥180d for all seasons)
            _days_of_data = (
                (features["hour"].max() - features["hour"].min()).days if not features.empty else 0
            )
            if _days_of_data >= 180:
                set_progress("seasonal_training", len(stat_ids), len(stat_ids), len(features), 0, 0)
                log.info("Training seasonal models (%d days of data)...", _days_of_data)
                seasonal.train_seasonal_models(features)
            else:
                log.info(
                    "Skipping seasonal models — only %d days of data (need ≥180)", _days_of_data
                )
            set_progress("pattern_analysis", len(stat_ids), len(stat_ids), len(features), 0, 0)
            log.info("Discovering patterns...")
            pattern_engine.run(features, stat_ids)

            # ── Novel ML features (incremental) ──
            log.info("Running phantom load detection...")
            phantom_results = phantom.run()
            phantom.save(phantom_results)

            drift_data = drift.detect_drift(features)
            drift.save(drift_data)

            try:
                auto_scores = await automation_score.score_all(HA_URL, HA_TOKEN)
                automation_score.save(auto_scores)
            except Exception as e:
                log.warning("Automation scoring failed: %s", e)
                auto_scores = []

            try:
                if os.path.exists(SUGGESTIONS_PATH):
                    with open(SUGGESTIONS_PATH) as _f:
                        suggestions_for_gap = json.loads(_f.read())
                else:
                    suggestions_for_gap = []
                gaps = await automation_gap.analyse(
                    HA_URL, HA_TOKEN, suggestions_for_gap, auto_scores
                )
                automation_gap.save(gaps)
            except Exception as e:
                log.warning("Automation gap analysis failed: %s", e)

            # Scene detection and smart automation suggestions
            try:
                try:
                    ha_areas.fetch_areas()
                except Exception as e:
                    log.warning("HA area fetch failed: %s", e)
                log.info("Running scene detection...")
                scenes = scene_detector.detect_scenes(days=min(days_history, 30))
                scene_detector.save(scenes)
                log.info("Detected %d implicit scenes", len(scenes))

                log.info("Running room prediction...")
                try:
                    area_data = ha_areas._load_cache()
                    e2a = area_data.get("entity_to_area", {})
                    if e2a:
                        room_predictor.run_room_prediction(e2a, days=min(days_history, 30))
                except Exception as e:
                    log.warning("Room prediction failed: %s", e)

                log.info("Running routine prediction...")
                try:
                    routine_predictor.run_routine_prediction(days=min(days_history, 30))
                except Exception as e:
                    log.warning("Routine prediction failed: %s", e)

                log.info("Running deep correlation analysis...")
                try:
                    area_data2 = ha_areas._load_cache()
                    e2a2 = area_data2.get("entity_to_area", {})
                    correlation_engine.run_correlation_analysis(e2a2, days=min(days_history, 30))
                except Exception as e:
                    log.warning("Correlation analysis failed: %s", e)

                log.info("Running sequence mining (PrefixSpan)...")
                try:
                    area_data3 = ha_areas._load_cache()
                    e2a3 = area_data3.get("entity_to_area", {})
                    sequence_miner.mine_sequences(e2a3, days=min(days_history, 30))
                except Exception as e:
                    log.warning("Sequence mining failed: %s", e)

                log.info("Running Markov chain next-action model...")
                try:
                    area_data4 = ha_areas._load_cache()
                    e2a4 = area_data4.get("entity_to_area", {})
                    markov_chain.build_markov_model(e2a4, days=min(days_history, 30))
                except Exception as e:
                    log.warning("Markov chain failed: %s", e)

                log.info("Running HMM activity state model...")
                try:
                    area_data5 = ha_areas._load_cache()
                    e2a5 = area_data5.get("entity_to_area", {})
                    activity_hmm.train_activity_model(e2a5, days=min(days_history, 30))
                except Exception as e:
                    log.warning("HMM activity model failed: %s", e)

                log.info("Running energy forecast...")
                try:
                    energy_forecast.run_energy_forecast(days_history=min(days_history, 90))
                except Exception as e:
                    log.warning("Energy forecast failed: %s", e)

                log.info("Running dynamic automation analysis...")
                try:
                    area_data6 = ha_areas._load_cache()
                    e2a6 = area_data6.get("entity_to_area", {})
                    dynamic_automations.run_dynamic_analysis(e2a6, days=min(days_history, 30))
                except Exception as e:
                    log.warning("Dynamic automation analysis failed: %s", e)

                log.info("Running appliance fingerprinting...")
                try:
                    appliance_fingerprint.run_fingerprinting(days=min(days_history, 30))
                except Exception as e:
                    log.warning("Appliance fingerprinting failed: %s", e)

                log.info("Running NILM disaggregation...")
                try:
                    nilm_disaggregator.run_disaggregation(days=min(days_history, 7))
                except Exception as e:
                    log.warning("NILM disaggregation failed: %s", e)

                log.info("Building smart automation suggestions...")
                if os.path.exists(SUGGESTIONS_PATH):
                    with open(SUGGESTIONS_PATH) as _f:
                        existing_sug = json.loads(_f.read())
                else:
                    existing_sug = []
                patterns_data = {}
                if os.path.exists(os.path.join(DATA_DIR, "patterns.json")):
                    with open(os.path.join(DATA_DIR, "patterns.json")) as _f:
                        patterns_data = json.loads(_f.read())
                smart_sug = automation_builder.generate_smart_suggestions(
                    scenes, patterns_data, existing_sug
                )
                automation_builder.save_smart_suggestions(smart_sug)
            except Exception as e:
                log.warning("Scene detection / automation builder failed: %s", e)

            anomaly_score = score_current(features)
            training_days = round(
                (features["hour"].max() - features["hour"].min()).total_seconds() / 86400
            )
            entity_count = tracked_entity_count
    else:
        # First run — all history
        cap = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days_history)
        full_from = cap.strftime("%Y-%m-%dT%H:00:00+00:00")
        log.info(f"First run: fetching last {days_history} days from {full_from}")
        set_progress("fetching", 0, len(stat_ids), 0, 0, 0)
        df_stats = await fetch_stats(stat_ids, full_from, now_iso) if stat_ids else pd.DataFrame()

        raw_max_days = int(os.environ.get("HABITUS_RAW_MAX_DAYS", "3650"))
        raw_days = min(days_history, raw_max_days)
        raw_from = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=raw_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw_to = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        df_raw = fetch_recent_raw_history(non_stat_ids, raw_from, raw_to)

        frames = [x for x in (df_stats, df_raw) if x is not None and not x.empty]
        if not frames:
            log.warning("No data returned")
            return
        df = pd.concat(frames, ignore_index=True)
        unique_entity_count = int(df["entity_id"].nunique())

        set_progress("building_baselines", len(stat_ids), len(stat_ids), len(df), 0, 0)
        log.info("Building entity baselines...")
        anomaly_breakdown.build_entity_baselines(df)
        activity_engine.build_activity_baseline(activity_engine.extract_activity_features(df))
        # Partial state write — unblocks UI baseline tab
        state.update({"phase": "baselines_ready", "entity_count": tracked_entity_count})
        save_state(state)
        features = build_features(df)
        del df
        set_progress("training", len(stat_ids), len(stat_ids), len(features), 0, 0)
        log.info(f"Training IsolationForest on {len(features):,} rows...")
        _train_days = round(
            (features["hour"].max() - features["hour"].min()).total_seconds() / 86400
        )
        model, scaler = train_model(features, _train_days)
        save_artifacts(model, scaler, features)
        _partial_score = score_current(features)
        state.update(
            {
                "phase": "model_ready",
                "anomaly_score": _partial_score,
                "training_days": _train_days,
                "contamination_tier": contamination_tier_name(_train_days),
            }
        )
        save_state(state)
        log.info(f"Model ready — preliminary score {_partial_score}/100")
        # Only train seasonal models with enough data (need ≥180d for all seasons)
        _days_of_data = (
            (features["hour"].max() - features["hour"].min()).days if not features.empty else 0
        )
        if _days_of_data >= 180:
            set_progress("seasonal_training", len(stat_ids), len(stat_ids), len(features), 0, 0)
            log.info("Training seasonal models (%d days of data)...", _days_of_data)
            seasonal.train_seasonal_models(features)
        else:
            log.info("Skipping seasonal models — only %d days of data (need ≥180)", _days_of_data)
        set_progress("pattern_analysis", len(stat_ids), len(stat_ids), len(features), 0, 0)
        log.info("Discovering patterns...")
        pattern_engine.run(features, stat_ids)

        # ── Novel ML features ──
        log.info("Running phantom load detection...")
        phantom_results = phantom.run()
        phantom.save(phantom_results)

        log.info("Running routine drift detection...")
        drift_data = drift.detect_drift(features)
        drift.save(drift_data)

        log.info("Scoring automation effectiveness...")
        try:
            auto_scores = await automation_score.score_all(HA_URL, HA_TOKEN)
            automation_score.save(auto_scores)
        except Exception as e:
            log.warning("Automation scoring failed: %s", e)
            auto_scores = []

        try:
            if os.path.exists(SUGGESTIONS_PATH):
                with open(SUGGESTIONS_PATH) as _f:
                    suggestions_for_gap = json.loads(_f.read())
            else:
                suggestions_for_gap = []
            gaps = await automation_gap.analyse(HA_URL, HA_TOKEN, suggestions_for_gap, auto_scores)
            automation_gap.save(gaps)
        except Exception as e:
            log.warning("Automation gap analysis failed: %s", e)

        # Scene detection and smart automation suggestions
        try:
            try:
                ha_areas.fetch_areas()
            except Exception as e:
                log.warning("HA area fetch failed: %s", e)
            log.info("Running scene detection...")
            scenes = scene_detector.detect_scenes(days=min(days_history, 30))
            scene_detector.save(scenes)
            log.info("Detected %d implicit scenes", len(scenes))

            log.info("Running room prediction...")
            try:
                area_data = ha_areas._load_cache()
                e2a = area_data.get("entity_to_area", {})
                if e2a:
                    room_predictor.run_room_prediction(e2a, days=min(days_history, 30))
            except Exception as e:
                log.warning("Room prediction failed: %s", e)

            log.info("Running routine prediction...")
            try:
                routine_predictor.run_routine_prediction(days=min(days_history, 30))
            except Exception as e:
                log.warning("Routine prediction failed: %s", e)

            log.info("Running deep correlation analysis...")
            try:
                area_data2 = ha_areas._load_cache()
                e2a2 = area_data2.get("entity_to_area", {})
                correlation_engine.run_correlation_analysis(e2a2, days=min(days_history, 30))
            except Exception as e:
                log.warning("Correlation analysis failed: %s", e)

            log.info("Running sequence mining (PrefixSpan)...")
            try:
                e2a_sm = ha_areas._load_cache().get("entity_to_area", {})
                sequence_miner.mine_sequences(e2a_sm, days=min(days_history, 30))
            except Exception as e:
                log.warning("Sequence mining failed: %s", e)

            log.info("Running Markov chain next-action model...")
            try:
                e2a_mk = ha_areas._load_cache().get("entity_to_area", {})
                markov_chain.build_markov_model(e2a_mk, days=min(days_history, 30))
            except Exception as e:
                log.warning("Markov chain failed: %s", e)

            log.info("Running HMM activity state model...")
            try:
                e2a_hm = ha_areas._load_cache().get("entity_to_area", {})
                activity_hmm.train_activity_model(e2a_hm, days=min(days_history, 30))
            except Exception as e:
                log.warning("HMM activity model failed: %s", e)

            log.info("Running energy forecast...")
            try:
                energy_forecast.run_energy_forecast(days_history=min(days_history, 90))
            except Exception as e:
                log.warning("Energy forecast failed: %s", e)

            log.info("Running dynamic automation analysis...")
            try:
                e2a_da = ha_areas._load_cache().get("entity_to_area", {})
                dynamic_automations.run_dynamic_analysis(e2a_da, days=min(days_history, 30))
            except Exception as e:
                log.warning("Dynamic automation analysis failed: %s", e)

            log.info("Running appliance fingerprinting...")
            try:
                appliance_fingerprint.run_fingerprinting(days=min(days_history, 30))
            except Exception as e:
                log.warning("Appliance fingerprinting failed: %s", e)

            log.info("Running NILM disaggregation...")
            try:
                nilm_disaggregator.run_disaggregation(days=min(days_history, 7))
            except Exception as e:
                log.warning("NILM disaggregation failed: %s", e)

            log.info("Building smart automation suggestions...")
            if os.path.exists(SUGGESTIONS_PATH):
                with open(SUGGESTIONS_PATH) as _f:
                    existing_sug = json.loads(_f.read())
            else:
                existing_sug = []
            patterns_data = {}
            if os.path.exists(os.path.join(DATA_DIR, "patterns.json")):
                with open(os.path.join(DATA_DIR, "patterns.json")) as _f:
                    patterns_data = json.loads(_f.read())
            smart_sug = automation_builder.generate_smart_suggestions(
                scenes, patterns_data, existing_sug
            )
            automation_builder.save_smart_suggestions(smart_sug)
        except Exception as e:
            log.warning("Scene detection / automation builder failed: %s", e)

        anomaly_score = score_current(features)
        training_days = round(
            (features["hour"].max() - features["hour"].min()).total_seconds() / 86400
        )
        entity_count = tracked_entity_count
        state["data_from"] = full_from
        # Record actual first data point date (not the query start)
        # actual_start logic moved earlier (before del df)
        # if not df.empty and "hour" in df.columns:
        #     actual_start = df["hour"].min().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        #     state["data_from"] = actual_start

    # Per-entity and activity scoring
    entity_anomalies = anomaly_breakdown.score_entities()
    weighted_entity_score = anomaly_breakdown.compute_weighted_score(entity_anomalies)
    if weighted_entity_score > 0:
        log.info("Confidence-weighted entity score: %.3f", weighted_entity_score)
    activity_summary = activity_engine.get_activity_summary()
    top_anomaly = entity_anomalies[0]["description"] if entity_anomalies else None

    state.update(
        {
            "last_run": now_iso,
            "version": os.environ.get("HABITUS_VERSION", os.environ.get("BUILD_VERSION", "?")),
            "max_power_kw": int(os.environ.get("HABITUS_MAX_POWER_KW", "25")),
            "data_to": now_iso,
            "training_days": training_days,
            "requested_days": days_history,
            "entity_count": entity_count,
            "total_stat_ids": total_stat_ids,
            "total_entities": total_entities,
            "anomaly_score": anomaly_score,
            "top_anomaly": top_anomaly,
            "seasonal_models": seasonal.seasonal_status(),
            "contamination_tier": contamination_tier_name(training_days),
        }
    )
    save_state(state)
    clear_progress()

    send_notification(
        "🧠 Habitus — Training complete",
        f"Model trained on {training_days} days · {entity_count} sensors · "
        f"Anomaly score: {anomaly_score}/100",
    )
    is_anomalous = anomaly_score > THRESHOLD
    log.info(f"Score: {anomaly_score}/100 ({'⚠ ANOMALY' if is_anomalous else '✓ normal'})")
    if is_anomalous:
        msg_parts = [f"Score: {anomaly_score}/100"]
        if top_anomaly:
            msg_parts.append(top_anomaly)
        if activity_summary.get("highlights"):
            msg_parts.append(activity_summary["highlights"][0])
        send_notification("🧠 Habitus — Unusual Activity", "\n".join(msg_parts))

    _publish_sensors(anomaly_score, is_anomalous, training_days, entity_count)
    await _register_lovelace_card()

    # Surface anomalies and suggestions on the HA dashboard automatically
    try:
        if os.path.exists(SUGGESTIONS_PATH):
            with open(SUGGESTIONS_PATH) as _f:
                suggestions = json.loads(_f.read())
        else:
            suggestions = []
    except Exception:
        suggestions = []
    publish_dashboard_entities(anomaly_score, entity_anomalies, suggestions)
    state = send_daily_digest(
        state, anomaly_score, entity_anomalies, suggestions, training_days, entity_count
    )
    save_state(state)

    log.info("Done.")


def _publish_sensors(
    anomaly_score: int,
    is_anomalous: bool,
    training_days: int,
    entity_count: int,
) -> None:
    """Publish the 4 Habitus sensor entities to HA.

    Called from both full and score-only run modes.

    Args:
        anomaly_score: Current 0–100 anomaly score.
        is_anomalous: Whether score exceeds the configured threshold.
        training_days: Days of history the model was trained on.
        entity_count: Number of behavioral sensors being tracked.
    """
    # Respect warmup — never publish a high score during learning
    if anomaly_score > 0 and training_days < MIN_SCORING_DAYS:
        anomaly_score = 0
        is_anomalous = False
    publish(
        "sensor.habitus_anomaly_score",
        anomaly_score,
        {
            "friendly_name": "Habitus Anomaly Score",
            "unit_of_measurement": "",
            "icon": "mdi:brain",
            "state_class": "measurement",
        },
    )
    publish(
        "binary_sensor.habitus_anomaly_detected",
        "on" if is_anomalous else "off",
        {"friendly_name": "Habitus Anomaly", "device_class": "problem"},
    )
    publish(
        "sensor.habitus_training_days",
        training_days,
        {
            "friendly_name": "Habitus Training Days",
            "unit_of_measurement": "days",
            "icon": "mdi:calendar-range",
        },
    )
    publish(
        "sensor.habitus_entity_count",
        entity_count,
        {
            "friendly_name": "Habitus Tracked Sensors",
            "unit_of_measurement": "sensors",
            "icon": "mdi:chip",
        },
    )


async def _register_lovelace_card():
    """Register Habitus Lovelace card resource and inject into default dashboard."""
    import json

    import websockets

    resource_url = "/local/habitus/habitus-card.js"

    try:
        ws = await websockets.connect(HA_WS, max_size=50 * 1024 * 1024)
        # Auth
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
        auth_resp = json.loads(await ws.recv())
        if auth_resp.get("type") != "auth_ok":
            return

        msg_id = 100

        # Check existing resources
        msg_id += 1
        await ws.send(json.dumps({"id": msg_id, "type": "lovelace/resources"}))
        res_resp = json.loads(await ws.recv())
        resources = res_resp.get("result", []) if res_resp.get("success") else []

        already_registered = any(r.get("url", "").startswith(resource_url) for r in resources)

        if not already_registered:
            msg_id += 1
            await ws.send(
                json.dumps(
                    {
                        "id": msg_id,
                        "type": "lovelace/resources/create",
                        "res_type": "module",
                        "url": resource_url + "?v=2.52.0",
                    }
                )
            )
            await ws.recv()

        # Try to inject card into default dashboard view 0
        msg_id += 1
        await ws.send(json.dumps({"id": msg_id, "type": "lovelace/config"}))
        dash_resp = json.loads(await ws.recv())
        dash = dash_resp.get("result", {}) if dash_resp.get("success") else {}

        if dash and "views" in dash and len(dash["views"]) > 0:
            view = dash["views"][0]
            cards = view.get("cards", [])
            has_habitus = any("habitus-card" in (c.get("type", "") or "") for c in cards)
            if not has_habitus:
                cards.insert(0, {"type": "custom:habitus-card"})
                view["cards"] = cards
                msg_id += 1
                await ws.send(
                    json.dumps(
                        {
                            "id": msg_id,
                            "type": "lovelace/config/save",
                            "config": dash,
                        }
                    )
                )
                await ws.recv()

        await ws.close()
    except Exception as exc:
        import logging

        logging.getLogger("habitus").warning("Lovelace card registration failed: %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habitus ML engine — trains behavioral model and scores current state."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Maximum history window in days (HA limits to what it has)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "score"],
        default="full",
        help=(
            "full: fetch new data, retrain model, score, publish. "
            "score: skip training, only score current state and publish. "
            "In overnight schedule, daytime runs use 'score' to avoid "
            "resource-intensive training during active hours."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run(days_history=args.days, mode=args.mode))
