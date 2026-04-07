"""Weekly Intelligence Report — compiles and sends a weekly digest.

Aggregates anomaly history, energy data, pattern changes, sensor health,
and automation suggestions into a formatted summary delivered via HA
notification service every Monday morning.
"""

import datetime
import json
import logging
import os
from typing import Any

import requests  # type: ignore[import-untyped]

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
REPORT_PATH = os.path.join(DATA_DIR, "weekly_report.json")


def _load_json(path: str, default: Any = None) -> Any:
    """Read a JSON file, returning *default* on any error."""
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return default


def generate_report() -> dict[str, Any]:
    """Compile the weekly intelligence report from existing data files.

    Reads anomaly history, patterns, suggestions, drift, phantom loads,
    entity baselines, and data quality to produce a single summary dict.

    Returns:
        Report dict with keys: ``week_ending``, ``anomaly_summary``,
        ``energy_summary``, ``pattern_changes``, ``automation_health``,
        ``sensor_health``, ``highlights``, ``generated_at``.
    """
    now = datetime.datetime.now(datetime.UTC)
    week_ago = now - datetime.timedelta(days=7)

    # ── Anomaly history ──────────────────────────────────────────────────
    history = _load_json(os.path.join(DATA_DIR, "anomaly_history.json"), [])
    week_events = [
        e
        for e in history
        if datetime.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) > week_ago
    ]
    scores = [e.get("anomaly_score", 0) for e in week_events]

    entity_counts: dict[str, int] = {}
    for ev in week_events:
        for ent in ev.get("entities", []):
            eid = ent.get("entity_id", "")
            if eid:
                entity_counts[eid] = entity_counts.get(eid, 0) + 1
    top_entity = max(entity_counts, key=lambda k: entity_counts[k]) if entity_counts else "none"

    anomaly_summary = {
        "total_events": len(week_events),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "peak_score": max(scores, default=0),
        "top_entity": top_entity,
        "top_entity_count": entity_counts.get(top_entity, 0),
    }

    # ── Energy summary ───────────────────────────────────────────────────
    phantom = _load_json(os.path.join(DATA_DIR, "phantom_loads.json"), {})
    overnight = phantom.get("overnight_baseline", {})
    energy_summary = {
        "this_month_kwh": phantom.get("this_month_kwh", 0),
        "overnight_baseline_w": round(overnight.get("avg_idle_kwh_per_hour", 0) * 1000, 1),
        "overnight_annual_kwh": overnight.get("overnight_kwh_year", 0),
    }

    # ── Pattern changes ──────────────────────────────────────────────────
    drift_data = _load_json(os.path.join(DATA_DIR, "drift.json"), {})
    drifts = drift_data.get("drifts", [])
    sig_drifts = [d for d in drifts if d.get("significant")]
    pattern_changes = {
        "drift_detected": len(sig_drifts) > 0,
        "drift_count": len(sig_drifts),
        "drift_summary": drift_data.get("summary", "No drift detected"),
    }

    # ── Automation health ────────────────────────────────────────────────
    suggestions = _load_json(os.path.join(DATA_DIR, "suggestions.json"), [])
    applicable = [s for s in suggestions if s.get("applicable")]
    gap_data = _load_json(os.path.join(DATA_DIR, "automation_gap.json"), {})
    gaps = gap_data.get("gaps", [])
    missing = [g for g in gaps if g.get("status") == "missing"]
    automation_health = {
        "total_suggestions": len(suggestions),
        "applicable_count": len(applicable),
        "missing_automations": len(missing),
        "gap_summary": gap_data.get("summary", ""),
    }

    # ── Sensor health ────────────────────────────────────────────────────
    baselines = _load_json(os.path.join(DATA_DIR, "entity_baselines.json"), {})
    quality_issues = _load_json(os.path.join(DATA_DIR, "data_quality.json"), [])
    stuck = [i for i in quality_issues if i.get("issue") == "stuck"]
    out_of_range = [
        i
        for i in quality_issues
        if i.get("issue") in ("temp_out_of_range", "humidity_out_of_range")
    ]
    sensor_health = {
        "total_entities": len(baselines),
        "issues_count": len(quality_issues),
        "stuck_count": len(stuck),
        "out_of_range_count": len(out_of_range),
    }

    # ── Highlights ───────────────────────────────────────────────────────
    highlights: list[str] = []
    n_events = int(anomaly_summary["total_events"])
    if n_events == 0:
        highlights.append("Clean week — no anomalies detected across all sensors.")
    elif n_events <= 3:
        highlights.append(f"Quiet week with only {n_events} anomaly events.")
    else:
        highlights.append(
            f"{anomaly_summary['total_events']} anomaly events this week "
            f"(avg score: {anomaly_summary['avg_score']})."
        )

    if top_entity != "none" and entity_counts.get(top_entity, 0) >= 3:
        friendly = top_entity.split(".")[-1].replace("_", " ").title()
        highlights.append(
            f"Most recurring anomaly: {friendly} " f"({entity_counts[top_entity]} events)."
        )

    if sig_drifts:
        highlights.append(f"Routine drift detected: {drift_data.get('summary', '')}.")

    if missing:
        highlights.append(
            f"{len(missing)} automation gap{'' if len(missing) == 1 else 's'} "
            "found — Habitus has suggestions ready to deploy."
        )

    if stuck:
        highlights.append(
            f"{len(stuck)} sensor{'' if len(stuck) == 1 else 's'} "
            "appear stuck (unchanged for 24+ hours)."
        )

    if not highlights:
        highlights.append("All systems nominal.")

    state = _load_json(os.path.join(DATA_DIR, "run_state.json"), {})
    return {
        "week_ending": now.strftime("%Y-%m-%d"),
        "training_days": state.get("training_days", 0),
        "anomaly_summary": anomaly_summary,
        "energy_summary": energy_summary,
        "pattern_changes": pattern_changes,
        "automation_health": automation_health,
        "sensor_health": sensor_health,
        "highlights": highlights,
        "generated_at": now.isoformat(),
    }


def format_report_message(report: dict[str, Any]) -> str:
    """Format the report dict into a human-readable notification message.

    Args:
        report: Report dict from :func:`generate_report`.

    Returns:
        Multi-line string suitable for HA notification body.
    """
    anom = report.get("anomaly_summary", {})
    energy = report.get("energy_summary", {})
    patterns = report.get("pattern_changes", {})
    auto = report.get("automation_health", {})
    health = report.get("sensor_health", {})
    highlights = report.get("highlights", [])

    lines = [
        f"Habitus Weekly Report — week ending {report.get('week_ending', '?')}",
        "",
    ]

    # Highlights
    for h in highlights:
        lines.append(f"  - {h}")
    lines.append("")

    # Anomalies
    lines.append(
        f"Anomalies: {anom.get('total_events', 0)} events, "
        f"avg {anom.get('avg_score', 0)}, peak {anom.get('peak_score', 0)}"
    )

    # Energy
    if energy.get("overnight_baseline_w"):
        lines.append(
            f"Overnight baseline: {energy['overnight_baseline_w']}W "
            f"({energy.get('overnight_annual_kwh', 0):.0f} kWh/year)"
        )

    # Drift
    if patterns.get("drift_detected"):
        lines.append(f"Routine drift: {patterns.get('drift_summary', '')}")

    # Automations
    if auto.get("missing_automations"):
        lines.append(
            f"Automation gaps: {auto['missing_automations']} missing "
            f"(of {auto.get('applicable_count', 0)} suggestions)"
        )

    # Sensor health
    if health.get("issues_count"):
        lines.append(
            f"Sensor issues: {health['issues_count']} "
            f"({health.get('stuck_count', 0)} stuck, "
            f"{health.get('out_of_range_count', 0)} out of range)"
        )

    lines.append(
        f"\nTracking {health.get('total_entities', 0)} entities, "
        f"trained on {report.get('training_days', '?')} days."
    )
    return "\n".join(lines)


def should_send_report(state: dict[str, Any]) -> bool:
    """Check whether it is time to send the weekly report.

    Sends once per week on Monday between 08:00–08:59 UTC.

    Args:
        state: Dict with optional ``last_weekly_report`` ISO date string.

    Returns:
        True if today is Monday, hour is 8, and the report hasn't been
        sent today.
    """
    now = datetime.datetime.now(datetime.UTC)
    if now.weekday() != 0 or now.hour != 8:
        return False
    last = state.get("last_weekly_report", "")
    if last:
        today = now.strftime("%Y-%m-%d")
        if last.startswith(today):
            return False
    return True


def send_weekly_report(ha_url: str, ha_token: str, notify_service: str) -> dict[str, Any]:
    """Generate, format, send, and persist the weekly report.

    Args:
        ha_url: Home Assistant base URL.
        ha_token: Long-lived access token.
        notify_service: HA notify service (e.g. ``"notify.notify"``).

    Returns:
        The generated report dict (also saved to ``weekly_report.json``).
    """
    report = generate_report()
    message = format_report_message(report)

    # Send via HA REST
    try:
        requests.post(
            f"{ha_url}/api/services/{notify_service.replace('.', '/')}",
            headers={"Authorization": f"Bearer {ha_token}"},
            json={"title": "Habitus Weekly Report", "message": message},
            timeout=10,
        )
        log.info("Weekly report sent via %s", notify_service)
    except requests.RequestException as e:
        log.warning("Failed to send weekly report: %s", e)

    # Persist
    atomic_write(REPORT_PATH, report)
    log.info(
        "Weekly report generated: %d anomaly events, %d highlights",
        report["anomaly_summary"]["total_events"],
        len(report["highlights"]),
    )
    return report
