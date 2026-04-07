"""Habitus v2.1 — polished web UI."""

import json
import os

import yaml as _yaml  # type: ignore[import-untyped]
from flask import Flask, jsonify, render_template, request

from habitus import trainer as _trainer

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
    return jsonify(_read(PROGRESS_PATH) or {})


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
    return jsonify(_read(ANOMALIES_PATH) or {})


@app.route("/api/anomaly_breakdown")
@app.route("/ingress/api/anomaly_breakdown")
def api_anomaly_breakdown():
    """Return per-entity anomaly breakdown with confidence weights.

    Reads ``entity_anomalies.json`` and returns the full breakdown including
    the confidence-weighted global score and per-entity ``confidence`` /
    ``confidence_label`` fields for UI display.
    """
    data = _read(ANOMALIES_PATH) or {}
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


@app.route("/api/rescan", methods=["POST"])
@app.route("/ingress/api/rescan", methods=["POST"])
@app.route("/api/full_train", methods=["POST"])
@app.route("/ingress/api/full_train", methods=["POST"])
def api_full_train():
    """Trigger a full 365-day training run without progressive steps."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from habitus.main import run

    def do_train():
        asyncio.run(run(days_history=365, mode="full"))

    with ThreadPoolExecutor() as pool:
        pool.submit(do_train)
    return jsonify({"ok": True, "message": "Full 365d training started"})


def api_rescan():
    try:
        import glob as _glob  # noqa: PLC0415

        for p in _glob.glob(os.path.join(DATA_DIR, "*.pkl")) + [
            STATE_PATH,
            BASELINE_PATH,
            PATTERNS_PATH,
            SUGGESTIONS_PATH,
        ]:
            if os.path.exists(p):
                os.remove(p)
        days = int(os.environ.get("HABITUS_DAYS", "365"))
        # Use progressive training: 30d → 60d → 90d → 180d → max
        from habitus import progressive as _prog  # noqa: PLC0415

        if _prog.is_expanding():
            return jsonify({"ok": False, "error": "Progressive training already running"}), 409
        _prog.start_progressive(max_days=days)
        return jsonify({"ok": True, "started": True, "progressive": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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
    import requests as req

    data = request.get_json()
    yaml_str = data.get("yaml", "")
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    try:
        parsed = _yaml.safe_load(yaml_str)
        auto = parsed.get("automation", parsed)
        alias = (
            auto.get("alias", "habitus_auto")
            .lower()
            .replace(" ", "_")
            .replace("—", "")
            .replace("–", "")
        )
        r = req.post(
            f"{ha_url}/api/config/automation/config/{alias}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=auto,
            timeout=10,
        )
        if r.status_code in (200, 201, 204):
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": f"HA {r.status_code}"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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
    from habitus import insights as _ins  # noqa: PLC0415

    return jsonify(_ins.compute_insights())


# ── New Feature Endpoints ────────────────────────────────────────────────────


@app.route("/api/ignore_list")
@app.route("/ingress/api/ignore_list")
def api_ignore_list():
    from habitus import entity_ignore as _ign  # noqa: PLC0415

    return jsonify(_ign.get_ignore_list())


@app.route("/api/ignore_entity", methods=["POST"])
@app.route("/ingress/api/ignore_entity", methods=["POST"])
def api_ignore_entity():
    from habitus import entity_ignore as _ign  # noqa: PLC0415

    data = request.get_json() or {}
    eid = data.get("entity_id", "")
    if not eid:
        return jsonify({"ok": False, "error": "entity_id required"}), 400
    _ign.add_entity(eid, data.get("reason", ""))
    return jsonify({"ok": True})


@app.route("/api/unignore_entity", methods=["POST"])
@app.route("/ingress/api/unignore_entity", methods=["POST"])
def api_unignore_entity():
    from habitus import entity_ignore as _ign  # noqa: PLC0415

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

    from habitus import csv_export as _csv  # noqa: PLC0415

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
    from habitus import nl_explanations as _nl  # noqa: PLC0415

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
    from habitus import vacation_mode as _vac  # noqa: PLC0415

    return jsonify(_vac.get_state())


@app.route("/api/vacation/toggle", methods=["POST"])
@app.route("/ingress/api/vacation/toggle", methods=["POST"])
def api_vacation_toggle():
    from habitus import vacation_mode as _vac  # noqa: PLC0415

    state = _vac.deactivate() if _vac.is_active() else _vac.activate(manual=True)
    return jsonify({"ok": True, "state": state})


@app.route("/api/device_health")
@app.route("/ingress/api/device_health")
def api_device_health():
    from habitus import device_health as _dh  # noqa: PLC0415

    return jsonify(_dh.get_health_report())


@app.route("/api/weekly_report")
@app.route("/ingress/api/weekly_report")
def api_weekly_report():
    from habitus import weekly_report as _wr  # noqa: PLC0415

    existing = _read(os.path.join(DATA_DIR, "weekly_report.json"))
    if existing:
        return jsonify(existing)
    return jsonify(_wr.generate_report())


@app.route("/api/sleep")
@app.route("/ingress/api/sleep")
def api_sleep():
    from habitus import sleep_quality as _sl  # noqa: PLC0415

    return jsonify(_sl.load())


@app.route("/api/comfort")
@app.route("/ingress/api/comfort")
def api_comfort():
    from habitus import comfort_score as _co  # noqa: PLC0415

    return jsonify(_co.load())


@app.route("/api/nilm_breakdown")
@app.route("/ingress/api/nilm_breakdown")
def api_nilm_breakdown():
    return jsonify(_read(os.path.join(DATA_DIR, "nilm_seq2point.json")) or {})


@app.route("/api/room_thresholds")
@app.route("/ingress/api/room_thresholds")
def api_room_thresholds():
    from habitus import room_thresholds as _rt  # noqa: PLC0415

    baselines = _read(os.path.join(DATA_DIR, "entity_baselines.json")) or {}
    return jsonify(_rt.get_zone_map([k for k in baselines if not k.startswith("_")]))


@app.route("/api/room_thresholds/override", methods=["POST"])
@app.route("/ingress/api/room_thresholds/override", methods=["POST"])
def api_room_threshold_override():
    from habitus import room_thresholds as _rt  # noqa: PLC0415

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
    from habitus import actionable_notifications as _an  # noqa: PLC0415

    data = request.get_json() or {}
    return jsonify(_an.handle_action(data.get("action", ""), data.get("entity_id", "")))


@app.route("/api/snooze", methods=["POST"])
@app.route("/ingress/api/snooze", methods=["POST"])
def api_snooze():
    from habitus import actionable_notifications as _an  # noqa: PLC0415

    data = request.get_json() or {}
    return jsonify(_an.snooze(int(data.get("hours", 24))))


@app.route("/api/unsnooze", methods=["POST"])
@app.route("/ingress/api/unsnooze", methods=["POST"])
def api_unsnooze():
    from habitus import actionable_notifications as _an  # noqa: PLC0415

    return jsonify(_an.unsnooze())


@app.route("/api/energy_budget")
@app.route("/ingress/api/energy_budget")
def api_energy_budget():
    from habitus import energy_budget as _eb  # noqa: PLC0415

    return jsonify(_eb.get_budget_status())


@app.route("/api/energy_budget", methods=["POST"])
@app.route("/ingress/api/energy_budget", methods=["POST"])
def api_set_energy_budget():
    from habitus import energy_budget as _eb  # noqa: PLC0415

    data = request.get_json() or {}
    return jsonify(_eb.set_budget(data.get("monthly_kwh"), data.get("monthly_cost")))


@app.route("/api/benchmarks")
@app.route("/ingress/api/benchmarks")
def api_benchmarks():
    from habitus import community_benchmarks as _cb  # noqa: PLC0415

    return jsonify(_cb.run())


def start_web(port=8099):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
