"""Habitus v2.1 — polished web UI."""

import json
import os
import re

import yaml as _yaml  # type: ignore[import-untyped]
from flask import Flask, jsonify, render_template, request

from . import trainer as _trainer

DATA_DIR = os.environ.get("DATA_DIR", "/data")
STATE_PATH = os.path.join(DATA_DIR, "run_state.json")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")
PATTERNS_PATH = os.path.join(DATA_DIR, "patterns.json")
SUGGESTIONS_PATH = os.path.join(DATA_DIR, "suggestions.json")
ANOMALIES_PATH = os.path.join(DATA_DIR, "entity_anomalies.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "progress.json")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
RESCAN_FLAG = os.path.join(DATA_DIR, ".rescan_requested")
PHANTOM_PATH = os.path.join(DATA_DIR, "phantom_loads.json")
DRIFT_PATH = os.path.join(DATA_DIR, "drift.json")
AUTO_SCORES_PATH = os.path.join(DATA_DIR, "automation_scores.json")
GAP_PATH = os.path.join(DATA_DIR, "automation_gap.json")
DATA_QUALITY_PATH = os.path.join(DATA_DIR, "data_quality.json")

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=_TEMPLATE_DIR)


def _load_page() -> str:
    """Return the index.html template content (used by tests for string assertions)."""
    _tmpl = os.path.join(_TEMPLATE_DIR, "index.html")
    try:
        with open(_tmpl) as _f:
            return _f.read()
    except OSError:
        return ""


PAGE: str = _load_page()


def _normalize_automation_id(entity_id_or_alias: str) -> str:
    """Normalize an automation entity_id or alias to a plain ASCII slug.

    Strips the ``automation.`` prefix, replaces unicode dashes with spaces,
    lowercases, removes non-alphanumeric characters, and collapses runs of
    spaces/underscores to a single underscore.
    """
    s = entity_id_or_alias.strip()
    prefix = "automation."
    if s.lower().startswith(prefix):
        s = s[len(prefix) :]
    s = s.replace("\u2014", " ").replace("\u2013", " ")  # em-dash, en-dash
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s_]", "", s)
    s = re.sub(r"[\s_]+", "_", s.strip())
    return s.strip("_")


def _unique_alias_id(alias: str, existing_ids: set[str]) -> str:
    """Return a unique slug for *alias*, appending ``_2``, ``_3``, etc. on collision."""
    base = _normalize_automation_id(alias)
    if base not in existing_ids:
        return base
    n = 2
    while f"{base}_{n}" in existing_ids:
        n += 1
    return f"{base}_{n}"


def _read(path: str, default: object = None) -> object:
    """Read and parse a JSON file, returning *default* on any file/parse error."""
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return default


@app.route("/")
@app.route("/ingress")
@app.route("/ingress/")
def index():
    schedule = os.environ.get("HABITUS_SCHEDULE", "overnight")
    train_time = os.environ.get("HABITUS_TRAIN_TIME", "02:00")
    return render_template("index.html", schedule=schedule, train_time=train_time)


@app.route("/api/state")
@app.route("/ingress/api/state")
def api_state():
    return jsonify(_read(STATE_PATH) or {})


@app.route("/api/baseline")
@app.route("/ingress/api/baseline")
def api_baseline():
    return jsonify(_read(BASELINE_PATH) or {})


@app.route("/api/progress")
@app.route("/ingress/api/progress")
def api_progress():
    """Return training progress with stale-lock recovery and metric normalisation."""
    import time as _time

    state = _read(STATE_PATH) or {}

    # --- File missing → synthesise idle payload from state ---
    if not os.path.exists(PROGRESS_PATH):
        payload: dict = {
            "running": False,
            "phase": "idle",
            "pct": 100,
            "done": 0,
            "total": 0,
            "rows": 0,
        }
        if state.get("last_run"):
            payload["last_run"] = state["last_run"]
        if state.get("last_completed_progress"):
            payload["last_completed_progress"] = state["last_completed_progress"]
        return jsonify(payload)

    raw = _read(PROGRESS_PATH) or {}

    def _sint(v: object) -> int:
        try:
            return int(float(str(v)))
        except (ValueError, TypeError):
            return 0

    def _sfloat(v: object) -> float:
        try:
            return float(str(v))
        except (ValueError, TypeError):
            return 0.0

    # --- Stale-lock / dead-trainer recovery ---
    if raw.get("running"):
        age = _time.time() - os.path.getmtime(PROGRESS_PATH)
        stale_sec = int(os.environ.get("HABITUS_PROGRESS_STALE_SEC", "300"))
        dead_grace_sec = int(os.environ.get("HABITUS_PROGRESS_DEAD_GRACE_SEC", "60"))
        pct_val = min(100, max(0, _sint(raw.get("pct", 0))))

        should_recover = False
        if age > stale_sec:
            should_recover = True
        elif not _trainer.is_running() and (pct_val >= 100 or age > dead_grace_sec):
            # Immediate: fetch phase finished (pct=100) but trainer never started
            should_recover = True

        if should_recover:
            recovered: dict = {
                "running": False,
                "phase": "idle",
                "stale_recovered": True,
                "rows": max(0, _sint(raw.get("rows", 0))),
                "done": _sint(raw.get("done", 0)),
                "total": _sint(raw.get("total", 0)),
            }
            if state.get("last_run"):
                recovered["last_run"] = state["last_run"]
            return jsonify(recovered)

    # --- Normalise and clamp metrics ---
    running = bool(raw.get("running"))
    phase = str(raw.get("phase") or "fetching")
    # A running job cannot be in "idle" phase
    if running and phase == "idle":
        phase = "fetching"

    pct = min(100, max(0, _sint(raw.get("pct", 0))))
    done = _sint(raw.get("done", 0))
    total = _sint(raw.get("total", 0))
    if total > 0 and done > total:
        done = total
    rows = max(0, _sint(raw.get("rows", 0)))
    elapsed_min = _sfloat(raw.get("elapsed_min", 0))
    eta_min = _sfloat(raw.get("eta_min", 0))

    out: dict = {
        "running": running,
        "phase": phase,
        "pct": pct,
        "done": done,
        "total": total,
        "rows": rows,
        "elapsed_min": elapsed_min,
        "eta_min": eta_min,
    }
    for key in ("progressive_window", "last_run"):
        if key in raw:
            out[key] = raw[key]
    return jsonify(out)


@app.route("/api/patterns")
@app.route("/ingress/api/patterns")
def api_patterns():
    return jsonify(_read(PATTERNS_PATH) or {})


@app.route("/api/suggestions")
@app.route("/ingress/api/suggestions")
def api_suggestions():
    return jsonify(_read(SUGGESTIONS_PATH) or [])


@app.route("/api/anomalies")
@app.route("/ingress/api/anomalies")
def api_anomalies():
    data = _read(ANOMALIES_PATH) or {}
    try:
        from . import feedback

        suppressed = feedback.get_suppressed_entities()
        if suppressed and isinstance(data.get("anomalies"), list):
            data["anomalies"] = [
                a for a in data.get("anomalies", []) if a.get("entity_id") not in suppressed
            ]
    except Exception:
        pass
    return jsonify(data)


@app.route("/api/anomaly_breakdown")
@app.route("/ingress/api/anomaly_breakdown")
def api_anomaly_breakdown():
    """Return per-entity anomaly breakdown with confidence weights.

    Reads ``entity_anomalies.json`` and returns the full breakdown including
    the confidence-weighted global score and per-entity ``confidence`` /
    ``confidence_label`` fields for UI display.
    """
    data = _read(ANOMALIES_PATH) or {}
    try:
        from . import feedback

        suppressed = feedback.get_suppressed_entities()
        if suppressed and isinstance(data.get("anomalies"), list):
            data["anomalies"] = [
                a for a in data.get("anomalies", []) if a.get("entity_id") not in suppressed
            ]
    except Exception:
        pass
    return jsonify(data)


@app.route("/api/sensor_health")
@app.route("/ingress/api/sensor_health")
def api_sensor_health():
    """Return sensor data-quality issues from ``data_quality.json``.

    Reports impossible-value detections (negative power, out-of-range temperature,
    humidity clamps, value jumps, stuck sensors) so they can be surfaced in the
    UI as a separate Sensor Health section rather than behavioral anomalies.
    """
    return jsonify(_read(DATA_QUALITY_PATH) or [])


@app.route("/api/anomaly_feedback", methods=["GET", "POST"])
@app.route("/ingress/api/anomaly_feedback", methods=["GET", "POST"])
def api_anomaly_feedback():
    from . import feedback

    if request.method == "GET":
        return jsonify(feedback.get_feedback_stats())

    data = request.get_json(silent=True) or {}
    anomaly_id = (
        str(data.get("anomaly_id", "")).strip()
        or str(data.get("entity_id", "")).strip()
        or "unknown"
    )
    action = str(data.get("action", "")).strip()
    entity_id = str(data.get("entity_id", "")).strip()
    score = float(data.get("score", 0) or 0)
    details = str(data.get("details", "")).strip()

    if action not in {"confirmed", "dismissed", "false_positive"}:
        return jsonify({"ok": False, "error": "invalid action"}), 400

    entry = feedback.record_feedback(
        anomaly_id=anomaly_id,
        action=action,
        entity_id=entity_id,
        score=score,
        details=details,
    )
    return jsonify({"ok": True, "entry": entry, "stats": feedback.get_feedback_stats()})


@app.route("/api/rescan", methods=["POST"])
@app.route("/ingress/api/rescan", methods=["POST"])
@app.route("/api/full_train", methods=["POST"])
@app.route("/ingress/api/full_train", methods=["POST"])
def api_full_train():
    """Trigger a full training run via the trainer manager."""
    days = int(os.environ.get("HABITUS_DAYS", "365"))
    started = _trainer.start(days, mode="full")
    if not started:
        return jsonify({"ok": False, "error": "Training already running"}), 409
    return jsonify({"ok": True, "message": f"Full training started ({days}d)"})


@app.route("/api/training_status")
@app.route("/ingress/api/training_status")
def api_training_status():
    return jsonify({"running": _trainer.is_running()})


@app.route("/api/power_sensors")
@app.route("/ingress/api/power_sensors")
def api_power_sensors():
    """Return all watt sensors from HA, plus current selection."""
    import requests as req  # type: ignore[import-untyped]

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    current = os.environ.get("HABITUS_POWER_ENTITY", "")
    try:
        r = req.get(f"{ha_url}/api/states", headers={"Authorization": f"Bearer {token}"}, timeout=8)
        sensors = []
        for s in r.json():
            eid = s["entity_id"]
            uom = s["attributes"].get("unit_of_measurement", "")
            if uom == "W" and eid.startswith("sensor."):
                try:
                    val = float(s["state"])
                    sensors.append(
                        {
                            "entity_id": eid,
                            "name": s["attributes"].get("friendly_name", eid),
                            "current_w": round(val, 1),
                        }
                    )
                except Exception:
                    pass
        sensors.sort(key=lambda x: -x["current_w"])
        return jsonify({"sensors": sensors, "selected": current, "auto_detected": current})
    except Exception as e:
        return jsonify({"error": str(e), "sensors": [], "selected": current})


@app.route("/api/settings", methods=["GET", "POST"])
@app.route("/ingress/api/settings", methods=["GET", "POST"])
def api_settings():
    """Get or update user-overridable settings (persisted to state.json)."""
    state_path = os.path.join(os.environ.get("DATA_DIR", "/data"), "state.json")
    try:
        with open(state_path) as f:
            state = json.load(f)
    except Exception:
        state = {}
    settings = state.get("user_settings", {})

    if request.method == "POST":
        data = request.get_json() or {}
        if "power_entity" in data:
            settings["power_entity"] = data["power_entity"]
            os.environ["HABITUS_POWER_ENTITY"] = data["power_entity"]
        state["user_settings"] = settings
        try:
            with open(state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})
        return jsonify({"ok": True, "settings": settings})

    return jsonify({"settings": settings})


@app.route("/api/add_automation", methods=["POST"])
@app.route("/ingress/api/add_automation", methods=["POST"])
def api_add_automation():
    """Create a new automation in Home Assistant from YAML, with unique ID generation."""
    import requests as req

    data = request.get_json() or {}
    yaml_str = (data.get("yaml") or "").strip()

    if not yaml_str:
        return jsonify({"ok": False, "error": "yaml is required"}), 400

    try:
        parsed = _yaml.safe_load(yaml_str)
    except _yaml.YAMLError as exc:
        return jsonify({"ok": False, "error": f"invalid YAML: {exc}"}), 400

    if not isinstance(parsed, dict):
        return jsonify({"ok": False, "error": "invalid YAML: expected a mapping"}), 400

    auto = parsed.get("automation", parsed)
    if not isinstance(auto, dict):
        return jsonify({"ok": False, "error": "invalid YAML: automation must be a mapping"}), 400

    alias = str(auto.get("alias") or "").strip()
    if not alias:
        return jsonify({"ok": False, "error": "automation alias is required"}), 400

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")

    # Fetch existing automations to detect ID collisions
    existing_ids: set[str] = set()
    try:
        r_states = req.get(
            f"{ha_url}/api/states",
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        for s in r_states.json():
            eid = s.get("entity_id", "")
            if eid.startswith("automation."):
                existing_ids.add(_normalize_automation_id(eid))
    except Exception:
        pass

    automation_id = _unique_alias_id(alias, existing_ids)
    try:
        r = req.post(
            f"{ha_url}/api/config/automation/config/{automation_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=auto,
            timeout=10,
        )
        if r.status_code in (200, 201, 204):
            return jsonify({"ok": True, "automation_id": automation_id})
        return jsonify({"ok": False, "error": f"HA {r.status_code}: {r.text}"}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"failed to add automation: {exc}"}), 500


@app.route("/api/remove_automation", methods=["POST"])
@app.route("/ingress/api/remove_automation", methods=["POST"])
def api_remove_automation():
    """Remove an automation from Home Assistant by entity_id or alias."""
    import requests as req

    data = request.get_json() or {}
    entity_id = str(data.get("entity_id") or "").strip()
    alias = str(data.get("alias") or "").strip()

    automation_id = _normalize_automation_id(entity_id or alias)
    if not automation_id:
        return jsonify({"ok": False, "error": "entity_id or alias is required"}), 400

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")

    try:
        r = req.delete(
            f"{ha_url}/api/config/automation/config/{automation_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code in (200, 204):
            return jsonify({"ok": True, "automation_id": automation_id})
        if r.status_code == 404:
            return jsonify({"ok": False, "error": "automation not found"}), 404
        return jsonify({"ok": False, "error": f"HA {r.status_code}: {r.text}"}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"failed to remove automation: {exc}"}), 500


@app.route("/api/phantom")
@app.route("/ingress/api/phantom")
def api_phantom():
    return jsonify(_read(PHANTOM_PATH) or [])


@app.route("/api/drift")
@app.route("/ingress/api/drift")
def api_drift():
    return jsonify(_read(DRIFT_PATH) or {})


@app.route("/api/automation_scores")
@app.route("/ingress/api/automation_scores")
def api_automation_scores():
    return jsonify(_read(AUTO_SCORES_PATH) or [])


@app.route("/api/automation_gap")
@app.route("/ingress/api/automation_gap")
def api_automation_gap():
    return jsonify(_read(GAP_PATH) or {})


@app.route("/api/insights")
@app.route("/ingress/api/insights")
def api_insights():
    """Return energy insights: peak hours, top consumers, waste, solar ratio."""
    from . import insights as _ins  # noqa: PLC0415

    return jsonify(_ins.compute_insights())


@app.route("/api/ha_automations")
@app.route("/ingress/api/ha_automations")
def api_ha_automations():
    """Return list of automation entity IDs + aliases currently in HA."""
    import requests as req  # type: ignore[import-untyped]

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    try:
        r = req.get(
            f"{ha_url}/api/states",
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        autos = []
        for s in r.json():
            if s["entity_id"].startswith("automation."):
                autos.append(
                    {
                        "entity_id": s["entity_id"],
                        "alias": s.get("attributes", {}).get("friendly_name", ""),
                        "state": s.get("state", "off"),
                    }
                )
        return jsonify({"automations": autos})
    except Exception as e:
        return jsonify({"automations": [], "error": str(e)})


# ── Anomaly Feedback ─────────────────────────────────────────────────────────


@app.route("/api/feedback", methods=["POST"])
@app.route("/ingress/api/feedback", methods=["POST"])
def api_feedback():
    """Record user feedback on an anomaly (dismiss or confirm)."""
    from . import feedback as _fb  # noqa: PLC0415

    data = request.get_json() or {}
    entry = _fb.record_feedback(
        anomaly_id=data.get("anomaly_id", ""),
        action=data.get("action", "dismissed"),
        entity_id=data.get("entity_id", ""),
        score=float(data.get("score", 0)),
        details=data.get("details", ""),
    )
    return jsonify({"ok": True, "entry": entry})


@app.route("/api/feedback/stats")
@app.route("/ingress/api/feedback/stats")
def api_feedback_stats():
    """Return feedback statistics for model tuning."""
    from . import feedback as _fb  # noqa: PLC0415

    return jsonify(_fb.get_feedback_stats())


@app.route("/api/feedback/reset", methods=["POST"])
@app.route("/ingress/api/feedback/reset", methods=["POST"])
def api_feedback_reset():
    """Reset all anomaly feedback to defaults."""
    feedback_path = os.path.join(DATA_DIR, "anomaly_feedback.json")
    if os.path.exists(feedback_path):
        os.remove(feedback_path)
    return jsonify({"ok": True, "message": "All feedback reset"})


# ── New Feature Endpoints ────────────────────────────────────────────────────


@app.route("/api/ignore_list")
@app.route("/ingress/api/ignore_list")
def api_ignore_list():
    from . import entity_ignore as _ign  # noqa: PLC0415

    return jsonify(_ign.get_ignore_list())


@app.route("/api/ignore_entity", methods=["POST"])
@app.route("/ingress/api/ignore_entity", methods=["POST"])
def api_ignore_entity():
    from . import entity_ignore as _ign  # noqa: PLC0415

    data = request.get_json() or {}
    eid = data.get("entity_id", "")
    if not eid:
        return jsonify({"ok": False, "error": "entity_id required"}), 400
    _ign.add_entity(eid, data.get("reason", ""))
    return jsonify({"ok": True})


@app.route("/api/unignore_entity", methods=["POST"])
@app.route("/ingress/api/unignore_entity", methods=["POST"])
def api_unignore_entity():
    from . import entity_ignore as _ign  # noqa: PLC0415

    data = request.get_json() or {}
    eid = data.get("entity_id", "")
    if not eid:
        return jsonify({"ok": False, "error": "entity_id required"}), 400
    _ign.remove_entity(eid)
    return jsonify({"ok": True})


@app.route("/api/export/<dataset>")
@app.route("/ingress/api/export/<dataset>")
def api_export(dataset: str):
    from flask import Response

    from . import csv_export as _csv  # noqa: PLC0415

    if dataset not in _csv.AVAILABLE_EXPORTS:
        return jsonify({"error": f"Unknown dataset: {dataset}"}), 404
    filename, export_fn = _csv.AVAILABLE_EXPORTS[dataset]
    return Response(
        export_fn(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/explain")
@app.route("/ingress/api/explain")
def api_explain():
    from . import nl_explanations as _nl  # noqa: PLC0415

    anomaly_data = _read(ANOMALIES_PATH) or {}
    anomalies = anomaly_data.get("anomalies", [])
    score = anomaly_data.get("weighted_score", anomaly_data.get("score", 0))
    baselines = _read(os.path.join(DATA_DIR, "entity_baselines.json")) or {}
    total = len([k for k in baselines if not k.startswith("_")])
    return jsonify(
        {
            "overall": _nl.explain_overall_score(int(score), anomalies, total),
            "score": int(score),
            "entities": [
                {"entity_id": a.get("entity_id", ""), "explanation": _nl.explain_entity_anomaly(a)}
                for a in anomalies[:10]
            ],
            "drift": _nl.explain_drift(_read(DRIFT_PATH) or {}),
        }
    )


@app.route("/api/vacation")
@app.route("/ingress/api/vacation")
def api_vacation():
    from . import vacation_mode as _vac  # noqa: PLC0415

    return jsonify(_vac.get_state())


@app.route("/api/vacation/toggle", methods=["POST"])
@app.route("/ingress/api/vacation/toggle", methods=["POST"])
def api_vacation_toggle():
    from . import vacation_mode as _vac  # noqa: PLC0415

    state = _vac.deactivate() if _vac.is_active() else _vac.activate(manual=True)
    return jsonify({"ok": True, "state": state})


@app.route("/api/device_health")
@app.route("/ingress/api/device_health")
def api_device_health():
    from . import device_health as _dh  # noqa: PLC0415

    return jsonify(_dh.get_health_report())


@app.route("/api/weekly_report")
@app.route("/ingress/api/weekly_report")
def api_weekly_report():
    from . import weekly_report as _wr  # noqa: PLC0415

    existing = _read(os.path.join(DATA_DIR, "weekly_report.json"))
    if existing:
        return jsonify(existing)
    return jsonify(_wr.generate_report())


@app.route("/api/sleep")
@app.route("/ingress/api/sleep")
def api_sleep():
    from . import sleep_quality as _sl  # noqa: PLC0415

    return jsonify(_sl.load())


@app.route("/api/comfort")
@app.route("/ingress/api/comfort")
def api_comfort():
    from . import comfort_score as _co  # noqa: PLC0415

    return jsonify(_co.load())


@app.route("/api/nilm_breakdown")
@app.route("/ingress/api/nilm_breakdown")
def api_nilm_breakdown():
    return jsonify(_read(os.path.join(DATA_DIR, "nilm_seq2point.json")) or {})


@app.route("/api/room_thresholds")
@app.route("/ingress/api/room_thresholds")
def api_room_thresholds():
    from . import room_thresholds as _rt  # noqa: PLC0415

    baselines = _read(os.path.join(DATA_DIR, "entity_baselines.json")) or {}
    return jsonify(_rt.get_zone_map([k for k in baselines if not k.startswith("_")]))


@app.route("/api/room_thresholds/override", methods=["POST"])
@app.route("/ingress/api/room_thresholds/override", methods=["POST"])
def api_room_threshold_override():
    from . import room_thresholds as _rt  # noqa: PLC0415

    data = request.get_json() or {}
    eid = data.get("entity_id", "")
    if not eid:
        return jsonify({"ok": False, "error": "entity_id required"}), 400
    mult = data.get("multiplier")
    if mult is not None:
        _rt.set_override(eid, float(mult))
    else:
        _rt.remove_override(eid)
    return jsonify({"ok": True})


@app.route("/api/notification_action", methods=["POST"])
@app.route("/ingress/api/notification_action", methods=["POST"])
def api_notification_action():
    from . import actionable_notifications as _an  # noqa: PLC0415

    data = request.get_json() or {}
    return jsonify(_an.handle_action(data.get("action", ""), data.get("entity_id", "")))


@app.route("/api/snooze", methods=["POST"])
@app.route("/ingress/api/snooze", methods=["POST"])
def api_snooze():
    from . import actionable_notifications as _an  # noqa: PLC0415

    data = request.get_json() or {}
    return jsonify(_an.snooze(int(data.get("hours", 24))))


@app.route("/api/unsnooze", methods=["POST"])
@app.route("/ingress/api/unsnooze", methods=["POST"])
def api_unsnooze():
    from . import actionable_notifications as _an  # noqa: PLC0415

    return jsonify(_an.unsnooze())


@app.route("/api/energy_budget", methods=["GET", "POST"])
@app.route("/ingress/api/energy_budget", methods=["GET", "POST"])
def api_energy_budget():
    from . import energy_budget as _eb  # noqa: PLC0415

    if request.method == "POST":
        data = request.get_json() or {}
        return jsonify(_eb.set_budget(data.get("monthly_kwh"), data.get("monthly_cost")))
    return jsonify(_eb.get_budget_status())


@app.route("/api/benchmarks")
@app.route("/ingress/api/benchmarks")
def api_benchmarks():
    from . import community_benchmarks as _cb  # noqa: PLC0415

    return jsonify(_cb.run())


def start_web(port=8099):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
