"""Habitus v2.1 — polished web UI."""

import json
import os
import re

import yaml as _yaml  # type: ignore[import-untyped]
from flask import Flask, jsonify, render_template_string, request

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
SCENES_PATH = os.path.join(DATA_DIR, "scenes.json")
SCENE_ANALYSIS_PATH = os.path.join(DATA_DIR, "scene_analysis.json")
SMART_SUGGESTIONS_PATH = os.path.join(DATA_DIR, "smart_suggestions.json")
HA_AUTOMATIONS_PATH = os.path.join(DATA_DIR, "ha_automations.json")

app = Flask(__name__)


def _read(path, default=None):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


_ACTIVE_PROGRESS_PHASES = {
    "fetching",
    "building_baselines",
    "training",
    "seasonal_training",
    "pattern_analysis",
    "post_analysis",
    "scoring",
}


def _int_or_default(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _float_or_default(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_automation_id(text: str) -> str:
    value = (text or "").strip().lower()
    value = value.replace("automation.", "", 1)
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _extract_automation_from_yaml(yaml_str: str) -> tuple[dict | None, str | None]:
    try:
        parsed = _yaml.safe_load(yaml_str)
    except Exception as e:
        return None, f"invalid YAML: {e}"

    if parsed is None:
        return None, "empty YAML"

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        return None, "YAML must decode to a mapping"

    auto = parsed.get("automation", parsed)
    if isinstance(auto, list):
        auto = auto[0] if auto else {}
    if not isinstance(auto, dict):
        return None, "automation payload must be an object"

    alias = (auto.get("alias") or "").strip()
    if not alias:
        return None, "automation alias is required"

    trigger = auto.get("trigger")
    action = auto.get("action")
    if trigger is None or action is None:
        return None, "automation must include trigger and action"

    if isinstance(trigger, dict):
        auto["trigger"] = [trigger]
    elif not isinstance(trigger, list):
        return None, "trigger must be a list or object"

    if isinstance(action, dict):
        auto["action"] = [action]
    elif not isinstance(action, list):
        return None, "action must be a list or object"

    auto.setdefault("mode", "single")
    return auto, None


def _existing_automation_ids(ha_url: str, token: str) -> set[str]:
    import requests as req

    ids: set[str] = set()
    try:
        r = req.get(
            f"{ha_url}/api/states",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code != 200:
            return ids
        for entry in r.json():
            eid = entry.get("entity_id", "")
            if isinstance(eid, str) and eid.startswith("automation."):
                ids.add(_normalize_automation_id(eid))
    except Exception:
        return ids
    return ids


def _unique_alias_id(alias: str, existing: set[str]) -> str:
    base = _normalize_automation_id(alias) or "habitus_automation"
    candidate = base
    idx = 2
    while candidate in existing:
        candidate = f"{base}_{idx}"
        idx += 1
    return candidate


def _normalize_progress_payload(progress: dict | None, state: dict | None) -> dict:
    p = dict(progress or {})
    st = state if isinstance(state, dict) else {}

    running = bool(p.get("running"))
    phase = str(p.get("phase") or "").strip()
    if running:
        if phase not in _ACTIVE_PROGRESS_PHASES:
            phase = "fetching"
    elif phase in _ACTIVE_PROGRESS_PHASES or not phase:
        phase = "idle"

    done = max(0, _int_or_default(p.get("done"), 0))
    total = max(0, _int_or_default(p.get("total"), 0))
    if total > 0 and done > total:
        done = total
    rows = max(0, _int_or_default(p.get("rows"), 0))

    has_recent = bool(st.get("last_run")) or rows > 0 or done > 0 or total > 0
    default_pct = 0 if running else (100 if has_recent else 0)
    pct = max(0, min(100, _int_or_default(p.get("pct"), default_pct)))

    payload = {
        "running": running,
        "phase": phase,
        "done": done,
        "total": total,
        "pct": pct,
        "rows": rows,
        "elapsed_min": max(0.0, _float_or_default(p.get("elapsed_min"), 0.0)),
        "eta_min": max(0.0, _float_or_default(p.get("eta_min"), 0.0)),
    }

    if p.get("stale_recovered"):
        payload["stale_recovered"] = True
    if "stale_age_sec" in p:
        payload["stale_age_sec"] = max(0, _int_or_default(p.get("stale_age_sec"), 0))

    last_run = p.get("last_run") or st.get("last_run")
    if last_run:
        payload["last_run"] = last_run

    last_completed = p.get("last_completed_progress") or st.get("last_completed_progress")
    if isinstance(last_completed, dict) and last_completed:
        payload["last_completed_progress"] = last_completed

    return payload


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Habitus</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg:     #0e1117;
  --bg2:    #151820;
  --card:   #1a1e28;
  --card2:  #1f2433;
  --border: #252a38;
  --border2:#2e3446;
  --accent: #4fc3f7;
  --accent2:#0288d1;
  --green:  #43d98b;
  --amber:  #ffb547;
  --red:    #ff5f6d;
  --purple: #b39ddb;
  --text:   #e8ecf4;
  --text2:  #a0a8bc;
  --text3:  #606880;
  --radius: 14px;
  --radius-sm: 8px;
}
[data-theme="light"] {
  --bg:     #f8f9fb;
  --bg2:    #ffffff;
  --card:   #ffffff;
  --card2:  #f1f3f7;
  --border: #e2e5eb;
  --border2:#d0d4dc;
  --accent: #0288d1;
  --accent2:#01579b;
  --green:  #2e7d32;
  --amber:  #f57c00;
  --red:    #d32f2f;
  --purple: #7b1fa2;
  --text:   #1a1a1a;
  --text2:  #555555;
  --text3:  #888888;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 14px;
  max-width: 1040px;
  margin: 0 auto;
  -webkit-font-smoothing: antialiased;
}

/* ── Header ── */
.header {
  padding: 22px 24px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.header-left { display: flex; align-items: center; gap: 14px; }
.logo { width: 40px; height: 40px; border-radius: 10px; }
.header h1 { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.02em; }
.header .version { font-size: 0.72rem; color: var(--text3); background: var(--card2); padding: 2px 8px; border-radius: 20px; border: 1px solid var(--border); margin-left: 8px; }
.header-right { display: flex; align-items: center; gap: 10px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.ok  { background: var(--green); box-shadow: 0 0 8px var(--green); }
.status-dot.warn { background: var(--amber); box-shadow: 0 0 8px var(--amber); }
.status-dot.bad  { background: var(--red); box-shadow: 0 0 8px var(--red); }
.status-label { font-size: 0.78rem; color: var(--text2); }
.last-run { font-size: 0.72rem; color: var(--text3); }

/* ── Nav ── */
nav {
  display: flex;
  gap: 0;
  padding: 16px 24px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 22px;
  overflow-x: auto;
  scrollbar-width: none;
}
nav::-webkit-scrollbar { display: none; }
nav button {
  background: none;
  border: none;
  color: var(--text3);
  padding: 8px 16px;
  cursor: pointer;
  font-size: 0.82rem;
  font-weight: 500;
  font-family: inherit;
  border-bottom: 2px solid transparent;
  white-space: nowrap;
  transition: color .15s;
  letter-spacing: -0.01em;
}
nav button:hover { color: var(--text2); }
nav button.active { color: var(--accent); border-bottom-color: var(--accent); }

/* ── Tabs ── */
.tab { display: none; padding: 0 24px 32px; }
.tab.active { display: block; }

/* ── Metric grid ── */
.metrics {
  display: grid;
  grid-template-columns: 240px 1fr 1fr 1fr;
  gap: 12px;
  margin-bottom: 18px;
}
@media (max-width: 700px) {
  .metrics { grid-template-columns: 1fr 1fr; }
  .score-card { grid-column: span 2; }
}

/* ── Score card ── */
.score-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}
.score-card::before {
  content: '';
  position: absolute;
  top: -40px; right: -40px;
  width: 140px; height: 140px;
  border-radius: 50%;
  opacity: 0.06;
}
.score-card.sn::before { background: var(--green); }
.score-card.sa::before { background: var(--amber); }
.score-card.sr::before { background: var(--red); }

/* SVG Gauge */
.gauge-wrap { position: relative; width: 120px; height: 70px; }
.gauge-wrap svg { width: 120px; height: 70px; }
.gauge-num {
  position: absolute;
  bottom: 0; left: 50%;
  transform: translateX(-50%);
  font-size: 1.8rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
}
.gauge-label { font-size: 0.72rem; color: var(--text3); margin-top: 8px; text-transform: uppercase; letter-spacing: .08em; }
.gauge-status { margin-top: 10px; }

/* ── Stat card ── */
.stat-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.stat-card .slabel { font-size: 0.72rem; color: var(--text3); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px; }
.stat-card .sval { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.03em; color: var(--text); }
.stat-card .sunit { font-size: 0.75rem; color: var(--text3); margin-left: 3px; font-weight: 400; }
.stat-card .ssub { font-size: 0.72rem; color: var(--text3); margin-top: 6px; }

/* ── Section ── */
.sec {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  margin-bottom: 12px;
}
.sec-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.sec-header h2 {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: .09em;
}
.sec-sub { font-size: 0.72rem; color: var(--text3); }

/* ── Two col ── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
@media (max-width: 600px) { .two-col { grid-template-columns: 1fr; } }

/* ── Table ── */
table { width: 100%; border-collapse: collapse; }
th {
  text-align: left;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: .07em;
  padding: 6px 10px 10px;
  border-bottom: 1px solid var(--border);
}
td { padding: 9px 10px; border-bottom: 1px solid var(--border); font-size: 0.83rem; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg2); }


/* ── Responsive tables ── */
.table-wrap { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
.table-wrap table { min-width: 720px; }
#tab-breakdown .table-wrap table { min-width: 860px; }

@media (max-width: 700px) {
  .tab { padding: 0 10px 20px; }
  .sec { padding: 12px; }
  .sec-header { margin-bottom: 10px; }
  th { font-size: 0.66rem; padding: 6px 8px 8px; }
  td { font-size: 0.78rem; padding: 8px; }

  /* Keep cards/tables on-screen on phone without clipping */
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .table-wrap table { min-width: 0; width: 100%; table-layout: fixed; }
  .table-wrap th, .table-wrap td { white-space: normal; word-break: break-word; }

  /* Breakdown table: hide secondary columns on mobile, keep core signal */
  #tab-breakdown th, #tab-breakdown td { white-space: normal; word-break: break-word; }
  #tab-breakdown .bd-col-current,
  #tab-breakdown .bd-col-baseline,
  #tab-breakdown .bd-col-confidence {
    display: none;
  }
  #tab-breakdown .bd-col-sensor { width: 58%; }
  #tab-breakdown .bd-col-deviation { width: 24%; }
  #tab-breakdown .bd-col-bar { width: 18%; }

  /* Automation health: reduce columns on mobile to avoid viewport cut-off */
  .auto-health-table th:nth-child(3), .auto-health-table td:nth-child(3),
  .auto-health-table th:nth-child(5), .auto-health-table td:nth-child(5) {
    display: none;
  }
}

/* ── Bar ── */
.bar-wrap { background: var(--border); border-radius: 3px; height: 4px; }
.train-banner { display:none; position:fixed; bottom:0; left:0; right:0; background:var(--card2);
  border-top:1px solid var(--border2); padding:10px 18px; font-size:.8rem; color:var(--text2);
  z-index:999; display:flex; align-items:center; gap:10px; }
.train-banner .spin { width:14px;height:14px;border:2px solid var(--border2);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;flex-shrink:0; }
#train-banner { display:none; }
.bar { height: 4px; border-radius: 3px; background: var(--accent); transition: width .5s cubic-bezier(.4,0,.2,1); }
.bar.green { background: var(--green); }
.bar.amber { background: var(--amber); }
.bar.red   { background: var(--red); }

/* ── Badge ── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 9px;
  border-radius: 20px;
  font-size: 0.71rem;
  font-weight: 600;
  letter-spacing: .02em;
}
.b-ok     { background: rgba(67,217,139,.12); color: var(--green); border: 1px solid rgba(67,217,139,.2); }
.b-warn   { background: rgba(255,181,71,.12); color: var(--amber); border: 1px solid rgba(255,181,71,.2); }
.b-alert  { background: rgba(255,95,109,.12); color: var(--red);   border: 1px solid rgba(255,95,109,.2); }
.b-info   { background: rgba(79,195,247,.12); color: var(--accent);border: 1px solid rgba(79,195,247,.2); }
.b-purple { background: rgba(179,157,219,.12);color: var(--purple);border: 1px solid rgba(179,157,219,.2); }
.b-muted  { background: var(--bg2); color: var(--text3); border: 1px solid var(--border); }
.b-boat   { background: rgba(128,203,196,.12); color: #80cbc4; border: 1px solid rgba(128,203,196,.2); }

/* ── Btn ── */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 7px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 600;
  font-family: inherit;
  background: var(--card2);
  color: var(--text2);
  transition: all .15s;
  white-space: nowrap;
}
.btn:hover { background: var(--border2); color: var(--text); border-color: var(--border2); }
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn-accent { background: rgba(79,195,247,.1); color: var(--accent); border-color: rgba(79,195,247,.25); }
.btn-accent:hover { background: rgba(79,195,247,.18); }
.btn-danger { background: rgba(255,95,109,.08); color: var(--red); border-color: rgba(255,95,109,.2); }
.btn-danger:hover { background: rgba(255,95,109,.15); }
.btn-success { background: rgba(67,217,139,.08); color: var(--green); border-color: rgba(67,217,139,.2); }
.btn-success:hover { background: rgba(67,217,139,.15); }

/* ── Suggestion card ── */
.sug {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 10px;
  transition: border-color .15s;
}
.sug:hover { border-color: var(--border2); }
.sug.na { opacity: .4; }
.sug-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px; }
.sug h3 { font-size: 0.88rem; font-weight: 600; letter-spacing: -0.01em; }
.sug .desc { color: var(--text2); font-size: 0.8rem; line-height: 1.6; margin-bottom: 12px; }
.sug pre {
  background: #090c12;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  font-size: 0.72rem;
  color: #7ec8e3;
  font-family: 'SF Mono', 'Fira Code', monospace;
  overflow-x: auto;
  margin-bottom: 10px;
  white-space: pre;
  max-height: 200px;
  overflow-y: auto;
  line-height: 1.6;
}
.sug-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.conf-bar { display: flex; align-items: center; gap: 6px; }
.conf-dots { display: flex; gap: 3px; }
.conf-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--border2); }
.conf-dot.filled { background: var(--accent); }

/* ── Filter tabs ── */
.ftabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
.ftab {
  padding: 5px 14px;
  border-radius: 20px;
  border: 1px solid var(--border);
  background: none;
  color: var(--text3);
  font-size: 0.78rem;
  font-weight: 500;
  font-family: inherit;
  cursor: pointer;
  transition: all .15s;
}
.ftab:hover { color: var(--text2); border-color: var(--border2); }
.ftab.active { background: var(--accent); color: #111; border-color: var(--accent); font-weight: 600; }

/* ── Pattern summary ── */
.pat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
.pat-item { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 12px 14px; }
.pat-item .pi-label { font-size: 0.7rem; color: var(--text3); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 4px; }
.pat-item .pi-val { font-size: 1rem; font-weight: 700; letter-spacing: -0.02em; }

/* ── Anomaly list ── */
.anom-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
  gap: 10px;
}
.anom-item:last-child { border-bottom: none; }
.anom-name { font-size: 0.83rem; font-weight: 500; }
.anom-sub { font-size: 0.72rem; color: var(--text3); margin-top: 2px; }
.anom-score { font-size: 0.88rem; font-weight: 700; white-space: nowrap; }

/* ── Season cards ── */
.season-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
@media (max-width: 500px) { .season-grid { grid-template-columns: 1fr 1fr; } }
.season-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 12px; text-align: center; }
.season-card .s-icon { font-size: 1.4rem; margin-bottom: 4px; }
.season-card .s-name { font-size: 0.72rem; font-weight: 600; color: var(--text2); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }

/* ── Progress overlay ── */
.prog-overlay {
  position: fixed; inset: 0;
  background: rgba(14,17,23,.9);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.prog-box {
  background: var(--card);
  border: 1px solid var(--border2);
  border-radius: 18px;
  padding: 32px 36px;
  max-width: 420px;
  width: 90%;
  text-align: center;
  box-shadow: 0 24px 64px rgba(0,0,0,.5);
}
.prog-icon { font-size: 2.5rem; margin-bottom: 12px; }
.prog-title { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 6px; }
.prog-desc { color: var(--text2); font-size: 0.82rem; line-height: 1.6; margin-bottom: 20px; }
.prog-bar-wrap { background: var(--border); border-radius: 6px; height: 6px; overflow: hidden; margin-bottom: 8px; }
.prog-bar { height: 6px; border-radius: 6px; background: linear-gradient(90deg, var(--accent2), var(--accent)); transition: width 1s cubic-bezier(.4,0,.2,1); }
.prog-meta { font-size: 0.75rem; color: var(--text3); }
.prog-steps { display: flex; justify-content: center; gap: 6px; margin-top: 16px; flex-wrap: wrap; }
.prog-step { font-size: 0.7rem; padding: 3px 10px; border-radius: 20px; background: var(--bg2); color: var(--text3); border: 1px solid var(--border); }
.prog-step.active { background: rgba(79,195,247,.12); color: var(--accent); border-color: rgba(79,195,247,.25); }
.prog-step.done { background: rgba(67,217,139,.08); color: var(--green); border-color: rgba(67,217,139,.2); }

/* ── Toast ── */
.toast {
  position: fixed; bottom: 20px; right: 20px;
  background: var(--card2);
  border: 1px solid var(--border2);
  border-radius: 10px;
  padding: 10px 16px;
  font-size: 0.8rem;
  opacity: 0;
  transition: opacity .25s, transform .25s;
  transform: translateY(8px);
  pointer-events: none;
  max-width: 280px;
  z-index: 200;
  box-shadow: 0 8px 24px rgba(0,0,0,.3);
}
.toast.show { opacity: 1; transform: translateY(0); }
.toast.ts { border-color: rgba(67,217,139,.3); color: var(--green); }
.toast.te { border-color: rgba(255,95,109,.3); color: var(--red); }

/* ── Settings ── */
pre.raw {
  font-size: 0.75rem;
  color: var(--text3);
  white-space: pre-wrap;
  word-break: break-all;
  font-family: 'SF Mono','Fira Code',monospace;
  line-height: 1.6;
}
.about-links { display: flex; flex-direction: column; gap: 10px; }
.about-link {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 14px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  text-decoration: none;
  color: var(--text2);
  font-size: 0.83rem;
  transition: border-color .15s, color .15s;
}
.about-link:hover { border-color: var(--border2); color: var(--text); }
.about-link .al-icon { font-size: 1.1rem; }
.divider { height: 1px; background: var(--border); margin: 16px 0; }
.warn-box {
  background: rgba(255,181,71,.06);
  border: 1px solid rgba(255,181,71,.2);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  font-size: 0.8rem;
  color: var(--amber);
  margin-bottom: 12px;
  display: none;
  line-height: 1.5;
}

/* ── Loading skeleton / shimmer ─────────────────────────────── */
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
.skeleton {
  background: linear-gradient(90deg,
    var(--bg2) 25%,
    rgba(255,255,255,0.06) 50%,
    var(--bg2) 75%);
  background-size: 800px 100%;
  animation: shimmer 1.4s infinite linear;
  border-radius: 6px;
  min-height: 16px;
}
.skeleton-line {
  height: 14px;
  border-radius: 4px;
  margin: 6px 0;
}
.skeleton-block {
  height: 60px;
  border-radius: 8px;
  margin: 8px 0;
}

/* ── Collapsible sections ────────────────────────────────────── */
.sec-header.collapsible {
  cursor: pointer;
  user-select: none;
}
.sec-header.collapsible::after {
  content: ' ▾';
  font-size: .75rem;
  color: var(--text3);
  margin-left: 6px;
}
.sec-header.collapsible.collapsed::after {
  content: ' ▸';
}
.sec-body { overflow: hidden; transition: max-height 0.28s ease, opacity 0.2s; }
.sec-body.collapsed { max-height: 0 !important; opacity: 0; pointer-events: none; }

/* ── Insight chips ───────────────────────────────────────────── */
.insight-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: .8rem;
  font-weight: 500;
  cursor: pointer;
  transition: opacity .15s;
  text-decoration: none;
  border: 1px solid transparent;
}
.insight-chip:hover { opacity: .8; }
.chip-danger { background: rgba(239,68,68,.15); color: #ef4444; border-color: rgba(239,68,68,.3); }
.chip-warn   { background: rgba(245,158,11,.15); color: var(--amber); border-color: rgba(245,158,11,.3); }
.chip-ok     { background: rgba(34,197,94,.15); color: var(--green); border-color: rgba(34,197,94,.3); }
.chip-info   { background: rgba(79,195,247,.15); color: #4fc3f7; border-color: rgba(79,195,247,.3); }
.chip-muted  { background: var(--bg2); color: var(--text2); border-color: var(--border); }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>Habitus <span class="version" id="hdr-version">v—</span></h1>
    <div style="font-size:.78rem;color:var(--text3)">Built in Europe. Private by architecture.</div>
  </div>
  <div class="header-right">
    <button id="theme-toggle" onclick="toggleTheme()" title="Toggle theme" style="background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:4px 10px;color:var(--text);cursor:pointer;font-size:1rem;margin-right:12px">🌙</button>
    <div class="status-dot" id="hdr-dot"></div>
    <span class="status-label" id="hdr-label">Loading</span>
    <span class="last-run" id="hdr-lr"></span>
  </div>
</div>

<nav>
  <button class="active" onclick="gotoTab('overview',this)">Overview</button>
  <button onclick="gotoTab('breakdown',this)">Anomaly Breakdown</button>
  <button onclick="gotoTab('smarthome',this)">Smart Home</button>
  <button onclick="gotoTab('energy',this)">Energy & Patterns</button>
  <button onclick="gotoTab('settings',this)">Settings</button>
  <button onclick="gotoTab('geek',this)">🔬 Geek</button>
</nav>

<!-- OVERVIEW -->
<div id="tab-overview" class="tab active">
  <div class="metrics">
    <div class="score-card card" id="score-card">
      <div class="gauge-wrap">
        <svg viewBox="0 0 120 70">
          <path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke="#252a38" stroke-width="8" stroke-linecap="round"/>
          <path id="gauge-arc" d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke="#4fc3f7" stroke-width="8" stroke-linecap="round" stroke-dasharray="157" stroke-dashoffset="157" style="transition:stroke-dashoffset 1s cubic-bezier(.4,0,.2,1),stroke .5s"/>
        </svg>
        <div class="gauge-num" id="gauge-num">—</div>
      </div>
      <div class="gauge-label">Anomaly Score</div>
      <div class="gauge-status" id="gauge-badge"></div>
    </div>
    <div class="stat-card">
      <div class="slabel">Training Data</div>
      <div class="sval" id="tdays">—<span class="sunit">days</span></div>
      <div class="ssub" id="tdays-range"></div>
    </div>
    <div class="stat-card">
      <div class="slabel">Tracked Sensors</div>
      <div class="sval" id="ecount">—<span class="sunit">sensors</span></div>
      <div class="ssub" id="ecount-sub"></div>
    </div>
    <div class="stat-card">
      <div class="slabel">Season Model</div>
      <div class="sval" id="season-active" style="font-size:1.1rem">—</div>
      <div class="ssub" id="season-sub"></div>
    </div>
  </div>

  <!-- ── Insights Summary Card ──────────────────────────────── -->
  <div class="sec" id="insights-summary-card" style="margin-bottom:16px">
    <div class="sec-header">
      <h2>🔍 Insights</h2>
      <span class="sec-sub" id="insights-sub">Click any chip to jump to that section</span>
    </div>
    <div id="insights-chips" style="display:flex;flex-wrap:wrap;gap:8px;padding:4px 0 8px">
      <div style="color:var(--text3);font-size:.82rem">Loading insights...</div>
    </div>
  </div>

  <div class="two-col">
    <div class="sec">
      <div class="sec-header"><h2>Patterns Discovered</h2></div>
      <div id="pat-summary"><div style="color:var(--text3)">Loading...</div></div>
    </div>
    <div class="sec">
      <div class="sec-header"><h2>Top Anomalies Now</h2></div>
      <div id="top-anomalies"><div style="color:var(--text3)">Loading...</div></div>
    </div>
  </div>

  <div class="sec">
    <div class="sec-header"><h2>Hourly Power Baseline</h2><span class="sec-sub">Learned normal consumption per hour</span></div>
    <div class="table-wrap"><table>
      <thead><tr><th>Time</th><th>Expected</th><th>±Variance</th><th style="width:120px"></th></tr></thead>
      <tbody id="bl-table"><tr><td colspan="4" style="color:var(--text3);padding:16px">Loading baselines...</td></tr></tbody>
    </table></div>
  </div>
</div>

<!-- ANOMALY BREAKDOWN -->
<div id="tab-breakdown" class="tab">
  <div class="sec">
    <div class="sec-header">
      <h2>Per-Entity Anomaly Scores</h2>
      <span class="sec-sub" id="bd-ts"></span>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th class="bd-col-sensor">Sensor</th><th class="bd-col-current">Current</th><th class="bd-col-baseline">Baseline</th><th class="bd-col-deviation">Deviation</th><th class="bd-col-confidence">Confidence</th><th class="bd-col-bar" style="width:90px"></th></tr></thead>
      <tbody id="bd-table"><tr><td colspan="6" style="color:var(--text3);padding:16px">No entity data yet.</td></tr></tbody>
    </table></div>
  </div>
</div>

<!-- SMART HOME (merged Automations + Insights) -->
<div id="tab-smarthome" class="tab">

  <!-- Active Conflicts -->
  <div class="sec" id="conflicts-section" style="display:none">
    <div class="sec-header"><h2>⚠️ Active Conflicts</h2><span class="sec-sub">Things that don't make sense right now</span></div>
    <div id="conflicts-list"></div>
  </div>

  <!-- Energy Forecast -->
  <div class="sec" id="forecast-section" style="display:none">
    <div class="sec-header"><h2>⚡ Energy Forecast</h2><span class="sec-sub">Weather-aware 7-day prediction</span></div>
    <div id="forecast-summary" style="margin-bottom:8px"></div>
    <div id="forecast-days"></div>
  </div>

  <!-- Behaviour Drift -->
  <div class="sec" id="drift-section" style="display:none">
    <div class="sec-header"><h2>📈 Behaviour Drift</h2><span class="sec-sub">Your habits are shifting</span></div>
    <div id="drift-list"></div>
  </div>

  <!-- Room Predictions -->
  <div class="sec" id="predictions-section" style="display:none">
    <div class="sec-header"><h2>🧠 Room Predictions</h2><span class="sec-sub">What you'll probably want when you enter a room</span></div>
    <div id="predictions-list"></div>
  </div>

  <!-- Discovered Scenes -->
  <div class="sec">
    <div class="sec-header"><h2>🎬 Discovered Scenes</h2><span class="sec-sub">Entity groups that activate together</span></div>
    <div id="scenes-list"><div style="color:var(--text3);padding:12px">Loading scenes...</div></div>
  </div>

  <!-- Scene Improvements -->
  <div class="sec" style="margin-top:16px" id="scene-improvements-sec">
    <div class="sec-header"><h2>🔍 Scene Improvements</h2><span class="sec-sub">Missing entities and smart trigger suggestions for your HA scenes</span></div>
    <div id="scene-improvements-list"><div style="color:var(--text3);padding:12px">Loading scene analysis...</div></div>
  </div>

  <!-- Suggested Automations -->
  <div class="sec" style="margin-top:16px">
    <div class="sec-header"><h2>💡 Suggested Automations</h2><span class="sec-sub">Patterns we detected in your usage</span></div>
    <div class="ftabs">
      <button class="ftab active" onclick="filterSug('all',this)">All</button>
      <button class="ftab" onclick="filterSug('scene',this)">🎬 Scenes</button>
      <button class="ftab" onclick="filterSug('routine',this)">🔁 Routine</button>
      <button class="ftab" onclick="filterSug('energy',this)">⚡ Energy</button>
      <button class="ftab" onclick="filterSug('boat',this)">⛵ Boat</button>
      <button class="ftab" onclick="filterSug('anomaly',this)">🧠 Anomaly</button>
      <button class="ftab" onclick="filterSug('lovelace',this)">🃏 Lovelace</button>
    </div>
    <div id="sug-list"><div style="color:var(--text3);padding:12px">Loading suggestions...</div></div>
  </div>

  <!-- Your HA Automations -->
  <div class="sec" style="margin-top:16px">
    <div class="sec-header"><h2>🤖 Your Automations</h2><span class="sec-sub">Existing automations from Home Assistant</span></div>
    <div id="ha-automations-list"><div class="skeleton skeleton-block"></div></div>
  </div>

  <!-- Entity Picker -->
  <div class="sec" style="margin-top:16px">
    <div class="sec-header"><h2>🔍 Entity Picker</h2><span class="sec-sub">Find entities for your automations</span></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
      <input type="text" id="entity-search" placeholder="Search entities (e.g. light, kitchen)..."
        style="flex:1;min-width:200px;background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem"
        oninput="filterEntities()">
      <select id="entity-domain-filter" onchange="filterEntities()"
        style="background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem">
        <option value="">All domains</option>
        <option value="light">💡 Lights</option>
        <option value="switch">🔌 Switches</option>
        <option value="media_player">🔊 Media</option>
        <option value="climate">🌡️ Climate</option>
        <option value="cover">🪟 Covers</option>
        <option value="fan">🌀 Fans</option>
        <option value="scene">🎬 Scenes</option>
        <option value="automation">🤖 Automations</option>
      </select>
    </div>
    <div id="entity-results" style="max-height:300px;overflow-y:auto"></div>
  </div>

  <!-- Insights (moved from old tab) -->
  <div class="sec" style="margin-top:16px">
    <div class="sec-header"><h2>🔌 Appliance Detection</h2><span class="sec-sub">Devices identified by power signature</span></div>
    <div id="appliance-list"><div class="skeleton skeleton-block"></div></div>
  </div>
  <div class="sec" style="margin-top:12px">
    <div class="sec-header"><h2>🕐 Predicted Routines</h2><span class="sec-sub">Recurring activities detected from sensor patterns</span></div>
    <div id="routines-list"><div class="skeleton skeleton-block"></div></div>
  </div>
  <div class="sec" style="margin-top:12px">
    <div class="sec-header"><h2>📊 Phantom Loads</h2><span class="sec-sub">Devices drawing power 24/7</span></div>
    <div id="phantom-list"><div class="skeleton skeleton-block"></div></div>
  </div>
  <div class="sec" style="margin-top:12px">
    <div class="sec-header"><h2>📈 Routine Drift</h2><span class="sec-sub">Changes in daily patterns</span></div>
    <div id="drift-info"><div class="skeleton skeleton-block"></div></div>
  </div>
  <div class="sec" style="margin-top:12px">
    <div class="sec-header"><h2>⚡ Automation Health</h2><span class="sec-sub">How well your automations work</span></div>
    <div id="auto-scores"><div class="skeleton skeleton-block"></div></div>
  </div>
  <div class="sec" style="margin-top:12px">
    <div class="sec-header"><h2>🔧 Automation Gaps</h2><span class="sec-sub">Suggestions vs existing automations</span></div>
    <div id="auto-gap"><div class="skeleton skeleton-block"></div></div>
  </div>

</div>

<!-- ======= NEW FEATURE SECTIONS ======= -->

<!-- ONBOARDING WIZARD (full-screen modal) -->
<div id="onboarding-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.8);z-index:9999;align-items:center;justify-content:center">
  <div style="background:var(--card);border-radius:16px;padding:32px;max-width:560px;width:90%;max-height:90vh;overflow-y:auto">
    <div id="onboarding-progress" style="display:flex;gap:8px;margin-bottom:24px">
      <div class="ob-step" style="flex:1;height:4px;background:var(--accent);border-radius:2px"></div>
      <div class="ob-step" style="flex:1;height:4px;background:var(--border);border-radius:2px"></div>
      <div class="ob-step" style="flex:1;height:4px;background:var(--border);border-radius:2px"></div>
      <div class="ob-step" style="flex:1;height:4px;background:var(--border);border-radius:2px"></div>
      <div class="ob-step" style="flex:1;height:4px;background:var(--border);border-radius:2px"></div>
      <div class="ob-step" style="flex:1;height:4px;background:var(--border);border-radius:2px"></div>
    </div>
    <div id="onboarding-content">
      <h2 style="margin:0 0 12px">👋 Welcome to Habitus</h2>
      <p style="color:var(--text2);margin:0 0 24px">Habitus learns your home's behaviour and suggests automations — privately, locally, without cloud.</p>
      <button onclick="obNext()" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:12px 24px;font-size:1rem;cursor:pointer;width:100%">Get Started →</button>
      <button onclick="obSkip()" style="background:none;color:var(--text3);border:none;padding:8px;cursor:pointer;width:100%;margin-top:8px;font-size:.85rem">Skip setup</button>
    </div>
  </div>
</div>

<!-- AUTOMATION HEALTH -->
<div id="automation-health-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>🏥 Automation Health</h2><span class="sec-sub">Dead, stale, and over-triggering automations</span></div>
  <div id="automation-health-list"><div class="skeleton skeleton-block"></div></div>
</div>

<!-- AUTOMATION CONFLICTS (inter-automation) -->
<div id="automation-conflicts-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>⚔️ Automation Conflicts</h2><span class="sec-sub">Automations that fight each other</span></div>
  <div id="automation-conflicts-list"><div class="skeleton skeleton-block"></div></div>
</div>

<!-- DETECTED ROUTINES (routine builder) -->
<div id="detected-routines-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>🔄 Detected Routines</h2><span class="sec-sub">Temporal sequences → chainable automations</span></div>
  <div id="detected-routines-list"><div class="skeleton skeleton-block"></div></div>
</div>

<!-- GUEST MODE -->
<div id="guest-mode-section" class="sec" style="margin-top:12px">
  <div id="guest-mode-banner" style="display:none;background:var(--accent);color:#fff;border-radius:8px;padding:12px 16px;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between">
    <span>👥 Guests may be present — <strong id="guest-prob-text">probability 0%</strong></span>
    <button onclick="activateGuestMode()" style="background:#fff;color:var(--accent);border:none;border-radius:6px;padding:6px 12px;cursor:pointer;font-weight:600">Activate Guest Mode</button>
  </div>
  <div class="sec-header"><h2>👥 Guest Mode</h2><span class="sec-sub">Unusual activity pattern detection</span></div>
  <div id="guest-mode-details"><div class="skeleton skeleton-block"></div></div>
</div>

<!-- SEASONAL SUGGESTIONS -->
<div id="seasonal-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>🌿 Seasonal Suggestions</h2><span class="sec-sub">Automations tuned to the current season</span><span id="season-badge" style="margin-left:8px;background:var(--accent);color:#fff;border-radius:12px;padding:2px 10px;font-size:.75rem"></span></div>
  <div id="seasonal-list"><div class="skeleton skeleton-block"></div></div>
</div>

<!-- BATTERY STATUS -->
<div id="battery-status-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>🔋 Battery Status</h2><span class="sec-sub">Device batteries sorted by urgency</span></div>
  <div id="battery-status-list"><div class="skeleton skeleton-block"></div></div>
</div>

<!-- INTEGRATION HEALTH -->
<div id="integration-health-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>🔌 Integration Health</h2><span class="sec-sub">Stale entities and integration scores</span><span id="health-score-badge" style="margin-left:8px;font-size:.8rem;color:var(--text3)"></span></div>
  <div id="integration-health-list"><div class="skeleton skeleton-block"></div></div>
</div>

<!-- NL AUTOMATION CREATOR -->
<div id="nl-automation-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>✍️ Describe an Automation</h2><span class="sec-sub">Write in plain English — Habitus generates the YAML</span></div>
  <div style="display:flex;gap:8px;margin-bottom:12px">
    <input type="text" id="nl-input" placeholder='e.g. "Turn off lights at 11pm" or "When motion detected turn on hallway light"'
      style="flex:1;background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-size:.9rem"
      oninput="nlPreview()" onkeydown="if(event.key==='Enter')nlPreview()">
    <button onclick="nlPreview()" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px 16px;cursor:pointer">Parse</button>
  </div>
  <div id="nl-preview" style="display:none">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <span id="nl-confidence" style="font-size:.8rem;color:var(--text3)"></span>
      <button id="nl-add-btn" onclick="nlAddToHA()" style="background:var(--success,#4caf50);color:#fff;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:.85rem">Add to HA</button>
    </div>
    <div id="nl-clarifications" style="color:var(--warn,#ff9800);font-size:.82rem;margin-bottom:8px"></div>
    <pre id="nl-yaml" style="background:var(--card2);border-radius:8px;padding:12px;font-size:.78rem;overflow-x:auto;max-height:300px;overflow-y:auto;white-space:pre-wrap"></pre>
  </div>
</div>

<!-- DASHBOARD GENERATOR -->
<div id="dashboard-generator-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>🎨 Dashboard Generator</h2><span class="sec-sub">Optimised Lovelace dashboard from your usage patterns</span></div>
  <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
    <button onclick="generateDashboard()" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px 16px;cursor:pointer">Generate Dashboard</button>
    <button onclick="copyDashboardYaml()" id="copy-dashboard-btn" style="display:none;background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 16px;cursor:pointer">Copy YAML</button>
    <button onclick="applyDashboard()" id="apply-dashboard-btn" style="display:none;background:var(--success,#4caf50);color:#fff;border:none;border-radius:8px;padding:8px 16px;cursor:pointer">Apply to HA</button>
  </div>
  <div id="dashboard-preview"><div style="color:var(--text3);font-size:.85rem">Click "Generate Dashboard" to create a Lovelace config from your usage patterns.</div></div>
</div>

<!-- RECENT ACTIVITY (changelog) -->
<div id="changelog-section" class="sec" style="margin-top:12px">
  <div class="sec-header"><h2>📋 Recent Activity</h2><span class="sec-sub">Automation changes and events</span></div>
  <div id="changelog-list"><div class="skeleton skeleton-block"></div></div>
</div>

<!-- ENERGY & PATTERNS -->
<div id="tab-energy" class="tab">
  <!-- Energy + Weather History -->
  <div class="sec" id="energy-weather-section">
    <div class="sec-header">
      <h2>📊 Energy vs Weather</h2>
      <select id="ew-range" onchange="loadEnergyWeather()" style="background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:.8rem">
        <option value="30">30 days</option>
        <option value="90" selected>90 days</option>
        <option value="180">6 months</option>
        <option value="365">1 year</option>
      </select>
    </div>
    <div id="ew-chart" style="margin-bottom:8px"></div>
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch">
      <table id="ew-table" style="width:100%;font-size:.75rem;border-collapse:collapse">
        <thead><tr style="border-bottom:1px solid var(--border)">
          <th style="text-align:left;padding:4px">Date</th>
          <th style="text-align:right;padding:4px">kWh</th>
          <th style="text-align:right;padding:4px">Avg °C</th>
          <th style="text-align:right;padding:4px">Min °C</th>
          <th style="text-align:right;padding:4px">Max °C</th>
          <th style="text-align:left;padding:4px">Bar</th>
        </tr></thead>
        <tbody id="ew-body"></tbody>
      </table>
    </div>
  </div>

  <!-- NILM Disaggregation -->
  <div class="sec" id="nilm-section" style="display:none">
    <div class="sec-header">
      <h2>🔌 Power Disaggregation (NILM)</h2>
      <button class="btn btn-accent" onclick="runNilm()" style="font-size:.75rem">⚡ Re-analyse</button>
    </div>
    <div id="nilm-current" style="margin-bottom:8px"></div>
    <div id="nilm-appliances"></div>
    <div id="nilm-energy" style="margin-top:8px"></div>
  </div>
  <div class="two-col">
    <div class="sec">
      <div class="sec-header"><h2>Weekly Profile</h2></div>
      <table>
        <thead><tr><th>Day</th><th>Avg Power</th><th style="width:100px"></th></tr></thead>
        <tbody id="wk-table"></tbody>
      </table>
    </div>
    <div class="sec">
      <div class="sec-header"><h2>Seasonal Models</h2></div>
      <div class="season-grid" id="season-cards"></div>
    </div>
  </div>
  <div class="sec">
    <div class="sec-header"><h2>Monthly Overview</h2><span class="sec-sub">Average power & temperature per month</span></div>
    <table>
      <thead><tr><th>Month</th><th>Avg Power</th><th>Avg Temp</th><th style="width:120px"></th></tr></thead>
      <tbody id="mo-table"></tbody>
    </table>
  </div>
</div>



<!-- SETTINGS -->
<div id="tab-settings" class="tab">
  <div class="sec">
    <div class="sec-header">
      <h2>Model Management</h2>
      <div style="display:flex;gap:8px">
        <button class="btn btn-accent" onclick="doRefresh()">↻ Refresh</button>
        <button class="btn btn-danger" onclick="confirmRescan()" id="btn-rescan">🔄 Full Rescan</button>
      </div>
    </div>
    <div class="warn-box" id="rescan-warn">
      ⚠️ This will delete the existing model and re-train from scratch using all available HA history. Takes 20–40 minutes depending on data volume.
    </div>

    <div style="margin-top:12px;padding:10px;border:1px solid var(--border);border-radius:10px;background:var(--bg2)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div style="font-size:.8rem;font-weight:600;color:var(--text2)">Training Progress</div>
        <div id="train-log-status" style="font-size:.75rem;color:var(--text3)">Idle</div>
      </div>
      <div class="bar-wrap" style="height:8px;margin-bottom:8px"><div id="train-log-bar" class="bar" style="width:0%"></div></div>
      <div id="train-log-meta" style="font-size:.74rem;color:var(--text3);margin-bottom:8px">No active training run</div>
      <pre id="train-log-lines" class="raw" style="max-height:170px;overflow:auto;margin:0">Waiting for training events…</pre>
    </div>

    <pre class="raw" id="raw-state">Loading...</pre>
  </div>
  <div class="sec" style="margin-top:12px">
    <div class="sec-header"><h2>Power Source</h2></div>
    <p style="color:var(--text3);font-size:.8rem;margin:0 0 12px">Habitus auto-detects your main power sensor. Override it here if the wrong one was selected.</p>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <select id="power-sensor-select" style="flex:1;min-width:200px;background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem">
        <option value="">Loading sensors…</option>
      </select>
      <button class="btn btn-accent" onclick="savePowerSensor()" style="white-space:nowrap">Save &amp; Retrain</button>
    </div>
    <div id="power-sensor-status" style="margin-top:8px;font-size:.78rem;color:var(--text3)"></div>
  </div>

  <script>
  (function(){
    async function loadPowerSensors(){
      const sel = document.getElementById('power-sensor-select');
      const status = document.getElementById('power-sensor-status');
      try {
        const r = await fetch('api/power_sensors');
        const d = await r.json();
        sel.innerHTML = '';
        if (d.sensors.length === 0) {
          sel.innerHTML = '<option value="">No watt sensors found</option>';
          return;
        }
        d.sensors.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.entity_id;
          opt.textContent = `${s.name} (${s.current_w}W)`;
          if (s.entity_id === d.selected) opt.selected = true;
          sel.appendChild(opt);
        });
        if (d.selected) {
          status.textContent = `Auto-detected: ${d.selected}`;
        }
      } catch(e) {
        sel.innerHTML = '<option value="">Error loading sensors</option>';
      }
    }
    window.savePowerSensor = async function(){
      const sel = document.getElementById('power-sensor-select');
      const status = document.getElementById('power-sensor-status');
      const eid = sel.value;
      if (!eid) return;
      status.textContent = 'Saving…';
      try {
        const r = await fetch('api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({power_entity: eid})});
        const d = await r.json();
        if (d.ok) {
          status.textContent = `✓ Saved: ${eid} — triggering retrain…`;
          status.style.color = 'var(--green)';
          setTimeout(async () => {
            await fetch('api/rescan', {method:'POST'});
            status.textContent = `✓ Saved: ${eid} — retraining started`;
          }, 500);
        } else {
          status.textContent = `Error: ${d.error}`;
          status.style.color = 'var(--red)';
        }
      } catch(e) {
        status.textContent = `Error: ${e}`;
        status.style.color = 'var(--red)';
      }
    };

    async function loadNotificationSetting(){
      const btn = document.getElementById('notify-toggle-btn');
      const st = document.getElementById('notify-toggle-status');
      if (!btn || !st) return;
      try {
        const r = await fetch('api/settings');
        const d = await r.json();
        const on = !!(d.settings && d.settings.notify_on_anomaly);
        btn.textContent = on ? 'Disable notifications' : 'Enable notifications';
        btn.className = on ? 'btn btn-danger' : 'btn btn-accent';
        st.textContent = on ? 'Notifications are ON' : 'Notifications are OFF';
      } catch(e) {
        btn.textContent = 'Toggle notifications';
        st.textContent = 'Failed to load setting';
      }
    }

    window.toggleHabitusNotifications = async function(){
      const btn = document.getElementById('notify-toggle-btn');
      const st = document.getElementById('notify-toggle-status');
      if (!btn || !st) return;
      btn.disabled = true;
      try {
        const currentOn = (btn.textContent || '').toLowerCase().includes('disable');
        const next = !currentOn;
        const r = await fetch('api/settings', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({notify_on_anomaly: next})
        });
        const d = await r.json();
        if (d.ok) {
          await loadNotificationSetting();
          st.textContent = next ? 'Notifications enabled' : 'Notifications disabled';
        } else {
          st.textContent = `Error: ${d.error || 'failed'}`;
        }
      } catch(e) {
        st.textContent = `Error: ${e}`;
      }
      btn.disabled = false;
    };

    loadPowerSensors();
    loadNotificationSetting();
  })();
  </script>

  <div class="sec" style="margin-top:12px">
    <div class="sec-header"><h2>Notifications</h2></div>
    <p style="color:var(--text3);font-size:.8rem;margin:0 0 12px">Control whether Habitus sends anomaly/training notifications.</p>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <button id="notify-toggle-btn" class="btn btn-accent" onclick="toggleHabitusNotifications()">Toggle notifications</button>
      <span id="notify-toggle-status" style="font-size:.78rem;color:var(--text3)"></span>
    </div>
  </div>

  <div class="sec" style="margin-top:12px">
    <div class="sec-header"><h2>About</h2></div>
    <div class="about-links">
      <a class="about-link" href="https://github.com/craigrallen/ha-habitus" target="_blank">
        <span class="al-icon">📦</span>
        <div><div style="font-weight:600;color:var(--text)">GitHub Repository</div><div style="font-size:.72rem;color:var(--text3)">craigrallen/ha-habitus</div></div>
      </a>
      <a class="about-link" href="https://buymeacoffee.com/craigrallen" target="_blank">
        <span class="al-icon">☕</span>
        <div><div style="font-weight:600;color:var(--amber)">Buy Me a Coffee</div><div style="font-size:.72rem;color:var(--text3)">If Habitus is useful, consider supporting it</div></div>
      </a>
    </div>
  </div>
</div>

<!-- GEEK TAB -->
<div id="tab-geek" class="tab">
  <!-- Activity States (HMM) -->
  <div class="sec" id="hmm-section" style="display:none">
    <div class="sec-header"><h2>🧩 Activity States (HMM)</h2><span class="sec-sub">Hidden Markov Model — what the home is "doing"</span></div>
    <div id="hmm-current" style="margin-bottom:8px"></div>
    <div id="hmm-states"></div>
  </div>

  <!-- Routine Sequences -->
  <div class="sec" id="sequences-section" style="display:none">
    <div class="sec-header"><h2>🔄 Routine Sequences (PrefixSpan)</h2><span class="sec-sub">Ordered event flows mined from history</span></div>
    <div id="sequences-list"></div>
  </div>

  <!-- Next-Action Predictions (Markov) -->
  <div class="sec" id="markov-section" style="display:none">
    <div class="sec-header"><h2>🎯 Next-Action Predictions (Markov)</h2><span class="sec-sub">Transition probabilities between actions</span></div>
    <div id="markov-list"></div>
  </div>

  <!-- Deep Correlations -->
  <div class="sec" id="correlations-section" style="display:none">
    <div class="sec-header"><h2>🔗 Deep Correlations</h2><span class="sec-sub">Statistically significant entity relationships</span></div>
    <div id="corr-stats" style="font-size:.8rem;color:var(--text3);margin-bottom:8px"></div>
    <div id="correlations-list"></div>
  </div>

  <!-- Device Training Mode -->
  <div class="sec">
    <div class="sec-header"><h2>🎓 Device Training</h2><span class="sec-sub">Teach Habitus your appliances by turning them on</span></div>
    <div id="training-ui">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px">
        <select id="train-power-entity" style="flex:1;min-width:200px;background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem">
          <option value="">Select power sensor...</option>
        </select>
      </div>
      <div id="train-idle">
        <p style="font-size:.82rem;color:var(--text3);margin:0 0 8px">1. Select your main power sensor<br>2. Click Start → turn on ONE device → Click Stop<br>3. Name it → fingerprint saved</p>
        <button class="btn btn-accent" onclick="startTraining()">▶️ Start Recording</button>
      </div>
      <div id="train-recording" style="display:none">
        <div class="card" style="padding:12px;border-left:3px solid var(--amber);margin-bottom:8px">
          <b>🔴 Recording...</b> Turn on your device now.
          <div style="font-size:.8rem;color:var(--text3)" id="train-baseline">Baseline: --</div>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="text" id="train-device-name" placeholder="Device name (e.g., Hob Ring 1)" style="flex:1;background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem">
          <button class="btn btn-accent" onclick="stopTraining()">⏹ Stop & Save</button>
        </div>
      </div>
    </div>

    <!-- Custom signatures list -->
    <div id="custom-sigs" style="margin-top:12px"></div>
  </div>

  <!-- Anomaly Feedback Stats -->
  <div class="sec">
    <div class="sec-header"><h2>📊 Anomaly Feedback</h2><span class="sec-sub">Your confirmations improve the model</span></div>
    <div id="feedback-stats"></div>
    <div style="margin-top:8px">
      <label style="font-size:.82rem;color:var(--text3);cursor:pointer">
        <input type="checkbox" id="sharing-toggle" onchange="toggleSharing(this.checked)">
        Share anonymised anomaly data to improve Habitus for everyone
      </label>
      <div style="font-size:.72rem;color:var(--text3);margin-top:4px">Only entity domains, scores, and feedback actions are shared. No names, IPs, or identifying info.</div>
    </div>
  </div>

  <!-- History Depth -->
  <div class="sec">
    <div class="sec-header"><h2>📅 Analysis History Depth</h2></div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <select id="history-depth" style="background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.85rem" onchange="saveHistoryDepth()">
        <option value="30">30 days</option>
        <option value="90">90 days</option>
        <option value="180">6 months</option>
        <option value="365">1 year</option>
        <option value="730">2 years</option>
        <option value="1095">3 years</option>
        <option value="3650">All history (max 10 years)</option>
      </select>
      <span style="font-size:.78rem;color:var(--text3)">More history = longer training but richer patterns</span>
    </div>
  </div>
</div>

<!-- PROGRESS OVERLAY -->
<div id="prog-overlay" class="prog-overlay" style="display:none">
  <div class="prog-box">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <span id="prog-icon">📡</span>
      <span id="prog-title" style="font-weight:600">Training</span>
      <button onclick="document.getElementById('prog-overlay').style.display='none'" style="margin-left:auto;background:none;border:none;color:var(--text3);cursor:pointer;font-size:1.1rem">×</button>
    </div>
    <div id="prog-desc" style="color:var(--text2);font-size:.78rem;margin-bottom:6px">Fetching...</div>
    <div class="prog-bar-wrap" style="height:4px;background:var(--card2);border-radius:2px;overflow:hidden">
      <div class="prog-bar" id="prog-bar" style="width:0%;height:100%;background:var(--accent);transition:width .3s"></div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let allSuggestions = [];
let haAutomationMap = new Map();
let currentFilter = 'all';
const phaseOrder = ['fetching','building_baselines','training','seasonal_training','pattern_analysis'];

function loadDismissedAnomalies(){
  try { return new Set(JSON.parse(localStorage.getItem('habitus-dismissed-anomalies') || '[]')); }
  catch(_) { return new Set(); }
}
function saveDismissedAnomalies(setObj){
  localStorage.setItem('habitus-dismissed-anomalies', JSON.stringify(Array.from(setObj)));
}
function anomalyKey(a){
  return (a && (a.entity_id || a.name || a.description || '')).toString();
}
function dismissAnomaly(anomalyId, entityId, score){
  const key = (entityId || anomalyId || '').toString();
  const dismissed = loadDismissedAnomalies();
  dismissed.add(key);
  saveDismissedAnomalies(dismissed);
  if (anomalyId || entityId) {
    anomalyFeedback(anomalyId || key, 'dismissed', entityId || key, score || 0);
  }
  load();
}

function gotoTab(id, btn) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  btn.classList.add('active');
}

function toast(msg, type='ts') {
  const el = document.getElementById('toast');
  el.textContent = msg; el.className = 'toast show ' + type;
  setTimeout(()=>el.className='toast', 3000);
}

function filterSug(cat, btn) {
  currentFilter = cat;
  document.querySelectorAll('.ftab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  renderSuggestions();
}

function confDots(n) {
  return Array.from({length:5},(_,i)=>`<div class="conf-dot${i < Math.round(n/20)?` filled`:''}"></div>`).join('');
}

function fmtW(w,cap){cap=cap||25000;if(!isFinite(w)||w<0||w>cap)return'—';if(w>=1000)return(w/1000).toFixed(1)+'kW';return Math.round(w)+'W';}

function normalizeAutomationId(s=''){
  return (s||'')
    .toString()
    .toLowerCase()
    .replace(/^automation\./,'')
    .replace(/[^a-z0-9_]+/g,'_')
    .replace(/_+/g,'_')
    .replace(/^_|_$/g,'');
}

function extractAliasFromYaml(yaml=''){
  const m = (yaml||'').match(/^\s*alias\s*:\s*["']?([^"'\n]+)["']?/m);
  return m ? m[1].trim() : '';
}

function buildAutomationMap(haAutos=[]){
  const map = new Map();
  (haAutos||[]).forEach(a => {
    const byAlias = normalizeAutomationId(a.alias || '');
    const byEntity = normalizeAutomationId(a.entity_id || '');
    if (byAlias) map.set(byAlias, a);
    if (byEntity) map.set(byEntity, a);
  });
  return map;
}

function renderSuggestions() {
  const list = currentFilter === 'all' ? allSuggestions : allSuggestions.filter(s=>s.category===currentFilter);
  if (!list.length) {
    document.getElementById('sug-list').innerHTML = '<div style="color:var(--text3);padding:16px">No suggestions in this category.</div>';
    return;
  }
  const catBadge = {routine:['b-info','🔁'],energy:['b-warn','⚡'],boat:['b-boat','⛵'],anomaly:['b-alert','🧠'],lovelace:['b-purple','🃏'],scene:['b-purple','🎬']};
  document.getElementById('sug-list').innerHTML = list.map(s => {
    const [bc, ic] = catBadge[s.category] || ['b-muted','•'];
    const overlapHtml = s.overlap_automation ? `<div style="font-size:.75rem;color:var(--amber);margin:4px 0">⚠ Possible overlap with: ${s.overlap_automation}</div>` : '';
    const entitiesHtml = (s.entities && s.entities.length) ? `<div style="margin:6px 0;display:flex;flex-wrap:wrap;gap:3px">${s.entities.map(e=>`<span class="badge b-info" style="font-size:.68rem;margin:1px">${e}</span>`).join('')}</div>` : '';
    const timeHtml = s.time_pattern ? `<div style="font-size:.72rem;color:var(--text3)">⏰ ${(s.time_pattern.peak_hour||'').toString().padStart(2,'0')}:00 · ${s.time_pattern.days||'daily'}</div>` : '';
    const whyHtml = s.why_suggested ? `<div style="font-size:.78rem;color:var(--text2);margin:6px 0">🧠 Why now: ${s.why_suggested}</div>` : '';
    const confidenceWhyHtml = s.confidence_rationale ? `<div style="font-size:.74rem;color:var(--text3)">${s.confidence_rationale}</div>` : '';
    const benefitHtml = s.expected_benefit ? `<div style="font-size:.74rem;color:var(--green)">Expected benefit: ${s.expected_benefit}</div>` : '';
    const costBadge = (s.cost_estimate && s.cost_estimate.monthly_saving_eur > 0)
      ? `<span class="badge b-ok" style="background:var(--green);color:#fff;font-size:.72rem">Saves ~€${s.cost_estimate.monthly_saving_eur.toFixed(1)}/month</span>`
      : '';
    const badgeStyleByName = {
      'high-confidence':'b-ok',
      'solid-confidence':'b-info',
      'exploratory':'b-warn',
      'high-relevance':'b-purple',
      'relevant':'b-info',
      'low-relevance':'b-muted',
      'high-usefulness':'b-ok',
      'needs-entities':'b-alert',
    };
    const statusBadgesHtml = Array.isArray(s.status_badges) && s.status_badges.length
      ? `<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:6px">${s.status_badges.map(b=>`<span class="badge ${badgeStyleByName[b]||'b-muted'}">${b.replace(/-/g,' ')}</span>`).join('')}</div>`
      : '';

    const alias = extractAliasFromYaml(s.yaml || '') || s.title || '';
    const key = normalizeAutomationId(alias);
    const existing = key ? haAutomationMap.get(key) : null;
    const existsBadge = s.category !== 'lovelace'
      ? (existing
          ? `<span class="badge b-ok" title="${existing.entity_id || ''}">in HA</span>`
          : `<span class="badge b-muted">not in HA</span>`)
      : '';

    return `
    <div class="sug ${s.applicable===false?'na':''}">
      <div class="sug-head">
        <h3>${s.title}</h3>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
          <span class="badge ${bc}">${ic} ${s.category}</span>
          ${existsBadge}
          ${costBadge}
        </div>
      </div>
      <div class="sug-meta">
        <div class="conf-bar"><div class="conf-dots">${confDots(s.confidence)}</div><span style="font-size:.72rem;color:var(--text3)">${s.confidence}% confidence</span></div>
        ${s.applicable===false?'<span class="badge b-alert">⚠ Entity not in your system</span>':''}
      </div>
      <div class="desc">${s.description}</div>
      ${whyHtml}
      ${confidenceWhyHtml}
      ${benefitHtml}
      ${statusBadgesHtml}
      ${entitiesHtml}
      ${timeHtml}
      ${overlapHtml}
      <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap">
        <button class="btn btn-accent" onclick="copyYaml('${s.id}')">📋 Copy YAML</button>
        ${s.category!=='lovelace' ? (!existing
          ? `<button class="btn btn-success" id="add-${s.id}" onclick="addToHA('${s.id}')">+ Add to HA</button>`
          : `<button class="btn btn-danger" id="remove-${s.id}" onclick="removeFromHA('${(existing.entity_id || '').replace(/'/g, "\\'")}', '${s.id}')">🗑 Remove from HA</button>`) : ''}
        <button class="btn" style="color:var(--text3);border-color:var(--border)" onclick="dismissSuggestion('${s.id}')" title="Dismiss — won't suppress permanently, but records feedback">✕ Dismiss</button>
      </div>
      <details><summary style="cursor:pointer;color:var(--accent);font-size:.82rem;margin:6px 0">Show YAML</summary>
        <pre id="yaml-${s.id}">${(s.yaml||'').trim()}</pre>
      </details>
    </div>`;
  }).join('');
}

function copyYaml(id) {
  navigator.clipboard.writeText(document.getElementById('yaml-'+id).textContent)
    .then(()=>toast('Copied to clipboard ✓'));
}

async function recordFeedback(suggestionId, action) {
  if (!suggestionId) return;
  try {
    await fetch('api/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({suggestion_id: suggestionId, action})
    });
  } catch(e) { /* non-fatal */ }
}

async function dismissSuggestion(suggestionId) {
  await recordFeedback(suggestionId, 'dismiss');
  const el = document.getElementById('sug-card-'+suggestionId);
  if (el) { el.style.opacity = '0.4'; el.style.pointerEvents = 'none'; }
  toast('Suggestion dismissed');
}

async function addToHA(id) {
  const btn = document.getElementById('add-'+id);
  const yaml = document.getElementById('yaml-'+id).textContent;
  btn.disabled = true; btn.textContent = 'Adding…';
  try {
    const r = await fetch('api/add_automation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yaml})});
    const d = await r.json();
    if (d.ok) {
      btn.textContent = '✓ Added';
      toast('Automation added ✓');
      recordFeedback(id, 'add');
      setTimeout(()=>load(), 500);
    } else {
      btn.disabled=false; btn.textContent='+ Add to HA'; toast('Failed: '+(d.error||'?'),'te');
    }
  } catch(e) { btn.disabled=false; btn.textContent='+ Add to HA'; toast('Network error','te'); }
}

async function addGapToHA(yamlId, btnId) {
  const btn = document.getElementById(btnId);
  const yaml = (document.getElementById(yamlId)?.textContent || '').trim();
  if (!yaml) { toast('Missing YAML','te'); return; }
  if (btn) { btn.disabled = true; btn.textContent = 'Adding…'; }
  try {
    const r = await fetch('api/add_automation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yaml})});
    const d = await r.json();
    if (d.ok) {
      if (btn) btn.textContent = '✓ Added';
      toast('Automation added ✓');
      setTimeout(()=>load(), 500);
    } else {
      if (btn) { btn.disabled = false; btn.textContent = '+ Add to HA'; }
      toast('Failed: '+(d.error||'?'),'te');
    }
  } catch(e) {
    if (btn) { btn.disabled = false; btn.textContent = '+ Add to HA'; }
    toast('Network error','te');
  }
}

async function removeFromHA(entityId, suggestionId='') {
  const btn = suggestionId ? document.getElementById('remove-'+suggestionId) : null;
  if (btn) { btn.disabled = true; btn.textContent = 'Removing…'; }
  try {
    const r = await fetch('api/remove_automation',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({entity_id:entityId})
    });
    const d = await r.json();
    if (d.ok) {
      toast('Automation removed ✓');
      if (suggestionId) recordFeedback(suggestionId, 'remove');
      load();
    } else {
      if (btn) { btn.disabled = false; btn.textContent = '🗑 Remove from HA'; }
      toast('Remove failed: '+(d.error||'?'),'te');
    }
  } catch(e) {
    if (btn) { btn.disabled = false; btn.textContent = '🗑 Remove from HA'; }
    toast('Network error','te');
  }
}

function setGauge(score) {
  const arc = document.getElementById('gauge-arc');
  const num = document.getElementById('gauge-num');
  const card = document.getElementById('score-card');
  const full = 157;
  const offset = full - (full * Math.min(score,100) / 100);
  arc.style.strokeDashoffset = offset;
  num.textContent = score;
  const color = score>=70 ? 'var(--red)' : score>=40 ? 'var(--amber)' : 'var(--green)';
  arc.style.stroke = color;
  num.style.color = color;
  card.className = 'score-card ' + (score>=70?'sr':score>=40?'sa':'sn');
}

function updateTrainingLog(progress, state) {
  const statusEl = document.getElementById('train-log-status');
  const barEl = document.getElementById('train-log-bar');
  const metaEl = document.getElementById('train-log-meta');
  const linesEl = document.getElementById('train-log-lines');
  if (!statusEl || !barEl || !metaEl || !linesEl) return;

  const phase = (progress && progress.phase) || (state && state.phase) || 'idle';
  const pct = Math.max(0, Math.min(100, (progress && progress.pct) || 0));
  const done = (progress && progress.done) || 0;
  const total = (progress && progress.total) || 0;
  const rows = (progress && progress.rows) || 0;
  const running = !!(progress && progress.running);
  const lastRunIso = (state && state.last_run) || (progress && progress.last_run) || null;
  const lastCompleted = (progress && progress.last_completed_progress)
    || (state && state.last_completed_progress)
    || null;
  const completedDone = Number((lastCompleted && lastCompleted.done) || done || 0);
  const completedTotal = Number((lastCompleted && lastCompleted.total) || total || 0);
  const completedRows = Number((lastCompleted && lastCompleted.rows) || rows || 0);
  const completedPhase = (lastCompleted && lastCompleted.phase) || 'complete';
  const hasRecentProgress = !!lastRunIso || !!lastCompleted || rows > 0 || done > 0 || total > 0;

  if (running) {
    statusEl.textContent = `Running · ${phase}`;
    barEl.style.width = `${pct}%`;
    barEl.style.opacity = '1';
    metaEl.textContent = `${pct}% · ${done}/${total || '?'} sensors · ${rows.toLocaleString()} rows`;
  } else if (hasRecentProgress) {
    statusEl.textContent = 'Last run complete';
    barEl.style.width = '100%';
    barEl.style.opacity = '0.35';
    const when = lastRunIso ? new Date(lastRunIso).toLocaleString() : 'recently';
    const sensorsLabel = completedTotal > 0 ? `${completedDone}/${completedTotal}` : `${completedDone}`;
    metaEl.textContent = `Last run: ${when} · ${completedPhase} · ${sensorsLabel} sensors · ${completedRows.toLocaleString()} rows`;
  } else {
    statusEl.textContent = 'Idle';
    barEl.style.width = '0%';
    barEl.style.opacity = '0.2';
    metaEl.textContent = 'No training run recorded yet';
  }

  if (!window._trainLog) window._trainLog = [];
  const stamp = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  const lineKey = running
    ? `run:${phase}:${pct}:${done}:${total}:${rows}`
    : hasRecentProgress
      ? `last:${lastRunIso || 'none'}:${completedPhase}:${completedDone}:${completedTotal}:${completedRows}`
      : 'idle';

  if (window._trainLogLastKey !== lineKey) {
    window._trainLogLastKey = lineKey;
    const line = running
      ? `[${stamp}] ${phase} · ${pct}% · ${done}/${total || '?'} sensors · ${rows.toLocaleString()} rows`
      : hasRecentProgress
        ? `[${stamp}] last run complete · ${completedPhase} · ${(lastRunIso ? new Date(lastRunIso).toLocaleString() : 'recently')} · ${completedRows.toLocaleString()} rows`
        : `[${stamp}] waiting for first run`;
    window._trainLog.push(line);
    if (window._trainLog.length > 120) window._trainLog = window._trainLog.slice(-120);
    linesEl.textContent = window._trainLog.join('\n');
    linesEl.scrollTop = linesEl.scrollHeight;
  }
}

async function load() {
  const [state, baseline, progress, patterns, suggestions, anomalies, phantomData, driftData, autoScores, gapData, scenesData, smartSuggestions, haAutomations, sceneAnalysis] = await Promise.all([
    fetch('api/state').then(r=>r.json()).catch(()=>({})),
    fetch('api/baseline').then(r=>r.json()).catch(()=>({})),
    fetch('api/progress').then(r=>r.json()).catch(()=>({})),
    fetch('api/patterns').then(r=>r.json()).catch(()=>({})),
    fetch('api/suggestions').then(r=>r.json()).catch(()=>([])),
    fetch('api/anomalies').then(r=>r.json()).catch(()=>({})),
    fetch('api/phantom').then(r=>r.json()).catch(()=>([])),
    fetch('api/drift').then(r=>r.json()).catch(()=>({})),
    fetch('api/automation_scores').then(r=>r.json()).catch(()=>([])),
    fetch('api/automation_gap').then(r=>r.json()).catch(()=>({})),
    fetch('api/scenes').then(r=>r.json()).catch(()=>({scenes:[]})),
    fetch('api/smart_suggestions').then(r=>r.json()).catch(()=>({suggestions:[]})),
    fetch('api/ha_automations').then(r=>r.json()).catch(()=>({automations:[]})),
    fetch('api/scene_analysis').then(r=>r.json()).catch(()=>({suggestions:[]})),
  ]);

  // Always update visible training status + log
  updateTrainingLog(progress, state);

  // Progress overlay — only block UI if no data at all yet
  const hasData = state.last_run || state.phase === 'baselines_ready' || state.phase === 'model_ready';
  if (progress && progress.running && !hasData) {
    document.getElementById('prog-overlay').style.display = 'flex';
    const phase = progress.phase||'fetching';
    const icons = {fetching:'📡',building_baselines:'🗄️',training:'🧠',seasonal_training:'🌱',pattern_analysis:'🔍'};
    const titles = {fetching:'Fetching sensor history',building_baselines:'Building entity baselines',training:'Training anomaly model',seasonal_training:'Training seasonal models',pattern_analysis:'Analysing patterns'};
    document.getElementById('prog-icon').textContent = icons[phase]||'⏳';
    document.getElementById('prog-title').textContent = titles[phase]||phase;
    const rows=(progress.rows||0).toLocaleString();
    const d=progress.done||0, t=progress.total||'?';
    const desc = {
      fetching: progress.progressive_window
        ? `${d} of ${t} sensors · ${rows} rows · window: last ${progress.progressive_window} days`
        : `${d} of ${t} sensors · ${rows} data points loaded`,
      building_baselines: `Computing per-entity baselines from ${rows} rows`,
      training:           `Fitting anomaly model on ${rows} snapshots`,
      seasonal_training:  `Training seasonal models (winter/spring/summer/autumn)`,
      pattern_analysis:   `Discovering daily & weekly patterns`,
      scoring:            `Scoring current state`,
    }[phase] || phase;
    document.getElementById('prog-desc').textContent = desc;
    // Animated row ticker during fetch
    if(phase==='fetching'){
      document.getElementById('prog-rows').textContent = rows+' rows';
    }
    document.getElementById('prog-bar').style.width = (progress.pct||0)+'%';
    const eta = progress.eta_min>0 ? `~${progress.eta_min} min remaining` : '';
    const el  = progress.elapsed_min ? `${progress.elapsed_min} min elapsed` : '';
    document.getElementById('prog-meta').textContent = [el,eta].filter(Boolean).join(' · ');
    // Step indicators
    const phaseIdx = phaseOrder.indexOf(phase);
    phaseOrder.forEach((p,i)=>{
      const el = document.getElementById('ps-'+p);
      if (!el) return;
      el.className = 'prog-step' + (i < phaseIdx ? ' done' : i===phaseIdx ? ' active' : '');
    });
    document.getElementById('hdr-dot').className = 'status-dot warn';
    const winLabel = progress.progressive_window ? ` (${progress.progressive_window}d window)` : '';
    if(document.getElementById('hdr-version'))document.getElementById('hdr-version').textContent='v'+(progress.version||state.version||'—');
  document.getElementById('hdr-label').textContent = `Training ${progress.pct||0}%${winLabel}`;
    if (!hasData) return;  // only block if no data yet
  }
  document.getElementById('prog-overlay').style.display = 'none';

  // Version badge
  if (document.getElementById('hdr-version')) {
    document.getElementById('hdr-version').textContent = 'v' + (state.version || '—');
  }

  // Header — treat as normal during warmup grace period
  const warmingUp = state.warming_up || state.phase === 'warming_up';
  const warmupDaysLeft = state.warmup_days_remaining ?? 0;
  const score = warmingUp ? 0 : (state.anomaly_score ?? 0);
  document.getElementById('hdr-dot').className = 'status-dot ' + (score>=70?'bad':score>=40?'warn':'ok');
  document.getElementById('hdr-label').textContent = score>=70?'Anomaly Detected':score>=40?'Elevated':'Normal';
  const lr = state.last_run ? new Date(state.last_run).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '';
  document.getElementById('hdr-lr').textContent = lr ? `Updated ${lr}` : '';

  // Score gauge
  setGauge(score);
  const bstr = score>=70?`<span class="badge b-alert">⚠ Anomaly</span>`:score>=40?`<span class="badge b-warn">Elevated</span>`:`<span class="badge b-ok">✓ Normal</span>`;
  document.getElementById('gauge-badge').innerHTML = bstr;

  // Stats
  const td = state.training_days??'—';
  const requestedDays = state.requested_days ?? null;
  if (requestedDays && td !== '—') {
    document.getElementById('tdays').innerHTML = `${td}<span class="sunit">d actual</span> <span class="sunit">/ ${requestedDays}d cfg</span>`;
  } else {
    document.getElementById('tdays').innerHTML = td+'<span class="sunit">days</span>';
  }
  const df=state.data_from?new Date(state.data_from).toLocaleDateString():'?';
  const dt=state.data_to?new Date(state.data_to).toLocaleDateString():'?';
  document.getElementById('tdays-range').textContent = td!=='—'
    ? `${df} → ${dt}` + (requestedDays ? ` · configured ${requestedDays}d` : '')
    : '';

  document.getElementById('ecount').innerHTML = (state.entity_count??'—')+'<span class="sunit">sensors</span>';
  const totalEntities = state.total_entities || null;
  const totalStatIds = state.total_stat_ids || null;
  const extraParts = [];
  if (totalStatIds) extraParts.push(`${totalStatIds} long-term stats`);
  if (totalEntities) extraParts.push(`${totalEntities} HA entities`);
  document.getElementById('ecount-sub').textContent = extraParts.length ? `from ${extraParts.join(' · ')}` : '';

  // Season
  const sm = state.seasonal_models||{};
  const now = new Date(); const m = now.getMonth()+1;
  const curSeason = m<=2||m===12?'winter':m<=5?'spring':m<=8?'summer':'autumn';
  const sIcons = {winter:'❄️',spring:'🌱',summer:'☀️',autumn:'🍂'};
  const trained = sm[curSeason];
  document.getElementById('season-active').textContent = (sIcons[curSeason]||'') + ' ' + curSeason.charAt(0).toUpperCase()+curSeason.slice(1);
  document.getElementById('season-sub').innerHTML = trained?'<span class="badge b-ok">Seasonal model</span>':'<span class="badge b-muted">Using main model</span>';

  // Patterns summary
  const r = patterns.daily_routine||{};
  document.getElementById('pat-summary').innerHTML = r.peak_usage_hour != null ? `
    <div class="pat-grid">
      <div class="pat-item"><div class="pi-label">Est. Wakeup</div><div class="pi-val">${r.estimated_wakeup_hour!=null?r.estimated_wakeup_hour+':00':'—'}</div></div>
      <div class="pat-item"><div class="pi-label">Est. Sleep</div><div class="pi-val">${r.estimated_sleep_hour!=null?r.estimated_sleep_hour+':00':'—'}</div></div>
      <div class="pat-item"><div class="pi-label">Peak Hour</div><div class="pi-val">${r.peak_usage_hour}:00 · ${r.peak_usage_watts}W</div></div>
      <div class="pat-item"><div class="pi-label">Night Base</div><div class="pi-val">${fmtW(r.night_baseline_watts)}</div></div>
    </div>` : '<div style="color:var(--text3);font-size:.82rem">No patterns yet — baseline first, then flow.</div>';

  // Top anomalies (dismissable)
  const dismissedAnomalies = loadDismissedAnomalies();
  const ents = (anomalies.anomalies||[])
    .filter(a => !dismissedAnomalies.has(anomalyKey(a)))
    .slice(0,5);
  document.getElementById('top-anomalies').innerHTML = ents.length
    ? ents.map(e => `
      <div class="anom-item">
        <div>
          <div class="anom-name">${e.name}</div>
          <div class="anom-sub">${e.current_value}${e.unit} vs ${e.baseline_mean}${e.unit} expected</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="anom-score" style="color:${e.z_score>=3?'var(--red)':e.z_score>=1.5?'var(--amber)':'var(--green)'}">${e.z_score}σ</div>
          <button class="btn" style="padding:3px 8px;font-size:.72rem" onclick="dismissAnomaly('${(e.id || e.entity_id || e.name || '').replace(/'/g, "\\'")}', '${(e.entity_id || e.name || '').replace(/'/g, "\\'")}', ${e.z_score || 0})">Dismiss</button>
        </div>
      </div>`).join('')
    : '<div style="color:var(--text3);font-size:.82rem">No anomalies right now.</div>';

  // Baseline table
  const byH={};
  for(const[k,v] of Object.entries(baseline)){const h=parseInt(k.split('_')[0],10);if(!byH[h]){byH[h]={sum:0,n:0,std:0}}byH[h].sum+=v.mean_power;byH[h].n++;byH[h].std=Math.max(byH[h].std,v.std_power)}
  const mxP=Math.max(...Object.values(byH).map(v=>v.sum/v.n),1);
  const nowH=new Date().getHours();
  document.getElementById('bl-table').innerHTML=Array.from({length:24},(_,h)=>{
    const b=byH[h]||{sum:0,n:1,std:0};
    const mean=Math.round(b.sum/b.n),std=Math.round(b.std),pct=Math.round(mean/mxP*100);
    const pd=h<6?'Night':h<12?'Morning':h<18?'Afternoon':'Evening';
    const cur=h===nowH?` style="background:rgba(79,195,247,.04)"`:''
    return`<tr${cur}><td><span style="font-weight:${h===nowH?700:400}">${String(h).padStart(2,'0')}:00</span> <span style="color:var(--text3);font-size:.7rem">${pd}</span>${h===nowH?' <span class="badge b-info" style="padding:1px 6px;font-size:.66rem">now</span>':''}</td><td>${fmtW(mean)}</td><td style="color:var(--text3)">±${fmtW(std)}</td><td><div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div></td></tr>`;
  }).join('');

  // Anomaly breakdown
  const allE = anomalies.anomalies||[];
  const bts = anomalies.timestamp ? 'as of '+new Date(anomalies.timestamp).toLocaleString() : '';
  document.getElementById('bd-ts').textContent = bts;
  document.getElementById('bd-table').innerHTML = allE.length
    ? allE.map(e=>{
        const z=e.z_score,pct=Math.min(100,Math.round(z/5*100));
        const col=z>=3?'var(--red)':z>=1.5?'var(--amber)':'var(--green)';
        const bc=z>=3?'b-alert':z>=1.5?'b-warn':'b-ok';
        const confLabel=e.confidence_label||'';
        return`<tr>
          <td class="bd-col-sensor"><div style="font-weight:500">${e.name}</div><div style="font-size:.7rem;color:var(--text3)">${e.entity_id}</div></td>
          <td class="bd-col-current" style="font-weight:600">${e.current_value}${e.unit}</td>
          <td class="bd-col-baseline" style="color:var(--text3)">${e.baseline_mean}${e.unit} <span style="color:var(--text3);font-size:.7rem">±${e.baseline_std}${e.unit}</span></td>
          <td class="bd-col-deviation"><span class="badge ${bc}">${z}σ</span></td>
          <td class="bd-col-confidence" style="font-size:.75rem;color:var(--text3)">${confLabel}</td>
          <td class="bd-col-bar"><div class="bar-wrap"><div class="bar" style="width:${pct}%;background:${col}"></div></div></td>
        </tr>`;}).join('')
    : '<tr><td colspan="6" style="color:var(--text3);padding:16px">No entity anomalies detected.</td></tr>';

  // Smart Suggestions (prefer smart_suggestions which include scenes; fallback to legacy)
  const smartSugList = (smartSuggestions && smartSuggestions.suggestions && smartSuggestions.suggestions.length)
    ? smartSuggestions.suggestions : suggestions;
  allSuggestions = smartSugList;

  // ── Activity States (HMM) ──
  fetch('api/activity_states').then(r=>r.json()).catch(()=>({states:[]})).then(h => {
    const sec = document.getElementById('hmm-section');
    if (!h.states || h.states.length === 0) { sec.style.display='none'; return; }
    sec.style.display='';
    const icons = {sleeping:'😴',away:'🚶',cooking:'🍳',working:'💻',relaxing:'📺',morning_routine:'☀️',active:'🏃',idle:'🏠'};
    document.getElementById('hmm-current').innerHTML = `<div class="card" style="padding:12px;border-left:3px solid var(--accent)">
      <b>Current state:</b> ${icons[h.current_state]||'❓'} <b>${(h.current_state||'unknown').replace('_',' ')}</b>
      (${h.method}) · ${h.total_windows} hourly windows analysed
      ${h.next_state_predictions ? '<br>Next likely: ' + h.next_state_predictions.slice(0,3).map(p => `${icons[p.state]||''} ${p.state} (${Math.round(p.probability*100)}%)`).join(', ') : ''}
    </div>`;
    document.getElementById('hmm-states').innerHTML = h.states.map(s => `
      <div class="card" style="padding:8px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center">
        <div>${icons[s.label]||'❓'} <b>${s.label.replace('_',' ')}</b></div>
        <div style="font-size:.8rem;color:var(--text3)">${s.percentage}% of time · peaks: ${s.peak_hours.map(h=>h+':00').join(', ')} · ${s.avg_motion.toFixed(0)} motion · ${s.avg_lights.toFixed(0)} lights</div>
      </div>
    `).join('');
  });

  // ── Energy Forecast ──
  fetch('api/energy_forecast').then(r=>r.json()).catch(()=>({forecast:[]})).then(ef => {
    const sec = document.getElementById('forecast-section');
    if (!ef.forecast || ef.forecast.length === 0) { sec.style.display='none'; return; }
    sec.style.display='';
    const trendIcon = ef.trend === 'up' ? '📈' : ef.trend === 'down' ? '📉' : '➡️';
    const tc = ef.temperature_correlation;
    document.getElementById('forecast-summary').innerHTML = `<div class="card" style="padding:12px">
      ${trendIcon} Weekly forecast: <b>${ef.weekly_total_kwh} kWh</b> (recent avg: ${ef.recent_weekly_avg_kwh} kWh)
      · Model: ${ef.model_type} · ${ef.training_days} days training
      ${tc ? `<br>🌡️ Temp correlation: ${tc.coefficient} (${tc.interpretation})` : ''}
    </div>`;
    document.getElementById('forecast-days').innerHTML = `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:6px">
      ${ef.forecast.map(d => `<div class="card" style="padding:8px;text-align:center">
        <div style="font-size:.75rem;color:var(--text3)">${d.day_name}</div>
        <div style="font-size:1.2rem;font-weight:bold">${d.predicted_kwh}</div>
        <div style="font-size:.7rem;color:var(--text3)">±${d.uncertainty_kwh} kWh</div>
        <div style="font-size:.65rem;color:var(--text3)">${d.low_kwh}–${d.high_kwh}</div>
      </div>`).join('')}
    </div>`;
  });

  // ── Behaviour Drift ──
  fetch('api/dynamic').then(r=>r.json()).catch(()=>({drifts:[]})).then(da => {
    const sec = document.getElementById('drift-section');
    if (!da.drifts || da.drifts.length === 0) { sec.style.display='none'; return; }
    sec.style.display='';
    document.getElementById('drift-list').innerHTML = da.drifts.slice(0,10).map(d => `
      <div class="card" style="padding:10px;margin-bottom:6px;border-left:3px solid ${d.shift_minutes>0?'var(--accent)':'var(--blue)'}">
        <div style="display:flex;justify-content:space-between">
          <b>${d.name}</b> <span style="font-size:.75rem">${d.room||''}</span>
        </div>
        <div style="font-size:.82rem;color:var(--text3)">${d.description}</div>
        <div style="font-size:.72rem;color:var(--accent);margin-top:2px">${d.confidence}% confident · suggestion: shift to ${d.recent_avg_time}</div>
      </div>
    `).join('');
  });

  // ── Routine Sequences ──
  fetch('api/sequences').then(r=>r.json()).catch(()=>({sequences:[]})).then(sq => {
    const sec = document.getElementById('sequences-section');
    if (!sq.sequences || sq.sequences.length === 0) { sec.style.display='none'; return; }
    sec.style.display='';
    document.getElementById('sequences-list').innerHTML = sq.sequences.slice(0,15).map(s => `
      <div class="card" style="padding:10px;margin-bottom:6px">
        <div style="display:flex;justify-content:space-between">
          <div style="font-size:.85rem"><b>${s.description}</b></div>
          <span style="font-size:.72rem;color:var(--accent)">${s.support}× (${s.frequency_pct}%)</span>
        </div>
        <div style="font-size:.72rem;color:var(--text3);margin-top:2px">${s.rooms.join(', ')} · ${s.length} steps · ${s.category}</div>
      </div>
    `).join('');
  });

  // ── Markov Predictions ──
  fetch('api/markov').then(r=>r.json()).catch(()=>({predictions:[]})).then(mk => {
    const sec = document.getElementById('markov-section');
    if (!mk.predictions || mk.predictions.length === 0) { sec.style.display='none'; return; }
    sec.style.display='';
    document.getElementById('markov-list').innerHTML = mk.predictions.slice(0,15).map(p => `
      <div class="card" style="padding:10px;margin-bottom:6px">
        <div style="font-size:.85rem">${p.description}</div>
        <div style="font-size:.78rem;color:var(--accent);margin-top:4px">💬 "${p.suggestion}"</div>
        <div style="font-size:.7rem;color:var(--text3);margin-top:2px">${p.total_observations} observations · ${Math.round(p.probability*100)}% probability</div>
      </div>
    `).join('');
  });

  // ── Deep Correlations ──
  fetch('api/correlations').then(r=>r.json()).catch(()=>({suggestions:[]})).then(cd => {
    const el = document.getElementById('correlations-list');
    const sec = document.getElementById('correlations-section');
    const stats = document.getElementById('corr-stats');
    if (!cd.suggestions || cd.suggestions.length === 0) { sec.style.display='none'; return; }
    sec.style.display='';
    const s = cd.stats || {};
    stats.textContent = `${cd.total_correlations} correlations found · ${cd.actionable_suggestions} actionable · ${s.entities_analysed || '?'} entities · ${s.total_events || '?'} events analysed`;
    const catIcons = {trigger_action:'⚡',room_routine:'🏠',cross_room_routine:'🚪→🚪',presence_driven:'👤',climate_response:'🌡️',general:'🔗'};
    el.innerHTML = cd.suggestions.slice(0,20).map(s => {
      const c = s.correlation || {};
      const icon = catIcons[s.category] || '🔗';
      const rooms = [c.room_a, c.room_b].filter(Boolean);
      const roomTag = rooms.length ? `<span style="font-size:.72rem;background:var(--card2);padding:1px 6px;border-radius:3px;margin-left:6px">${rooms.join(' → ')}</span>` : '';
      return `<div class="card" style="padding:12px;margin-bottom:8px;border-left:3px solid ${s.lift>5?'var(--red)':s.lift>3?'var(--accent)':'var(--text3)'}">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div>${icon} <b>${c.name_a}</b> → <b>${c.name_b}</b>${roomTag}</div>
          <span style="font-size:.75rem;color:var(--accent)">${s.confidence}% · ${s.lift}× lift</span>
        </div>
        <div style="font-size:.82rem;color:var(--text3);margin-top:4px">${s.description}</div>
        <div style="font-size:.72rem;color:var(--text3);margin-top:2px">${c.cooccurrences} co-occurrences out of ${c.total_a} events</div>
        ${s.yaml ? `<details style="margin-top:6px"><summary style="cursor:pointer;color:var(--accent);font-size:.8rem">Automation YAML</summary><pre style="font-size:.72rem;margin-top:4px">${s.yaml}</pre></details>` : ''}
      </div>`;
    }).join('');
  });

  // ── Room Predictions ──
  fetch('api/room_predictions').then(r=>r.json()).catch(()=>({automations:[]})).then(rp => {
    const el = document.getElementById('predictions-list');
    const sec = document.getElementById('predictions-section');
    if (!rp.automations || rp.automations.length === 0) { sec.style.display='none'; return; }
    sec.style.display='';
    el.innerHTML = rp.automations.map(p => `
      <div class="card" style="padding:12px;margin-bottom:8px;border-left:3px solid var(--accent)">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div>🧠 <b>${p.room}</b> — ${p.time_window} (${p.day_type})</div>
          <span style="font-size:.75rem;color:var(--accent)">${p.confidence}% · ${p.entry_count} observations</span>
        </div>
        <div style="font-size:.82rem;color:var(--text3);margin-top:4px">${p.description}</div>
        <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px">
          ${p.actions.map(a => `<span style="background:var(--card2);padding:2px 8px;border-radius:4px;font-size:.78rem">${a.name} → <b>${a.state}</b> (${Math.round(a.probability*100)}%)</span>`).join('')}
        </div>
        <details style="margin-top:8px"><summary style="cursor:pointer;color:var(--accent);font-size:.8rem">Notification automation YAML</summary><pre style="font-size:.72rem;margin-top:4px">${p.yaml}</pre></details>
      </div>
    `).join('');
  });

  // ── Active Conflicts ──
  fetch('api/conflicts').then(r=>r.json()).catch(()=>({conflicts:[]})).then(cd => {
    const el = document.getElementById('conflicts-list');
    const sec = document.getElementById('conflicts-section');
    // Also populate the automation-conflicts-list section
    const acListEl = document.getElementById('automation-conflicts-list');
    if (acListEl) {
      if (!cd.conflicts || cd.conflicts.length === 0) {
        acListEl.innerHTML = '<div style="color:var(--text3);padding:12px;font-size:.82rem">✓ No conflicts detected — your automations are working well together.</div>';
      } else {
        const sevColors = {critical:'#ff4444',high:'#ff8800',medium:'#ffbb33',low:'var(--text3)'};
        acListEl.innerHTML = cd.conflicts.map(c => `
          <div class="card" style="padding:12px;margin-bottom:8px;border-left:3px solid ${sevColors[c.severity]||'var(--border)'}">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <div><span style="font-size:1.2rem">${c.icon||'⚠️'}</span> <b>${c.title||'Conflict'}</b></div>
              <span style="font-size:.75rem;color:${sevColors[c.severity]};text-transform:uppercase">${c.severity||''}</span>
            </div>
            <div style="font-size:.82rem;color:var(--text3);margin-top:4px">${c.description||''}</div>
            <div style="font-size:.82rem;margin-top:6px">💡 ${c.suggestion||''}</div>
          </div>`).join('');
      }
    }
    if (!cd.conflicts || cd.conflicts.length === 0) { sec.style.display='none'; return; }
    sec.style.display='';
    const sevColors = {critical:'#ff4444',high:'#ff8800',medium:'#ffbb33',low:'var(--text3)'};
    el.innerHTML = cd.conflicts.map(c => `
      <div class="card" style="padding:12px;margin-bottom:8px;border-left:3px solid ${sevColors[c.severity]||'var(--border)'}">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div><span style="font-size:1.2rem">${c.icon||'⚠️'}</span> <b>${c.title}</b></div>
          <span style="font-size:.75rem;color:${sevColors[c.severity]};text-transform:uppercase">${c.severity}</span>
        </div>
        <div style="font-size:.82rem;color:var(--text3);margin-top:4px">${c.description}</div>
        <div style="font-size:.82rem;margin-top:6px">💡 ${c.suggestion}</div>
        ${c.est_waste_w ? `<div style="font-size:.75rem;color:var(--text3);margin-top:4px">~${c.est_waste_w}W estimated waste</div>` : ''}
        ${c.yaml ? `<details style="margin-top:8px"><summary style="cursor:pointer;color:var(--accent);font-size:.8rem">Fix automation YAML</summary><pre style="font-size:.72rem;margin-top:4px">${c.yaml}</pre></details>` : ''}
      </div>
    `).join('');
  });

  // Discovered Scenes
  const scenes = (scenesData && scenesData.scenes) ? scenesData.scenes : [];
  if (scenes.length) {
    document.getElementById('scenes-list').innerHTML = scenes.map((sc, idx) => {
      const entList = sc.entities.map(e => `<span class="badge b-info" style="margin:2px;font-size:.72rem">${e}</span>`).join('');
      const tp = sc.time_pattern || {};
      return `
      <div class="sug" style="margin-bottom:12px;border-left:3px solid var(--purple)">
        <div class="sug-head">
          <h3>🎬 ${sc.name}</h3>
          <span class="badge b-purple" style="background:var(--purple);color:#fff">${sc.confidence}% confidence</span>
        </div>
        <div class="desc">${sc.description || ''}</div>
        <div style="margin:8px 0;display:flex;flex-wrap:wrap;gap:4px">${entList}</div>
        <div style="font-size:.75rem;color:var(--text3);margin-bottom:8px">
          ⏰ Peak: ${(tp.peak_hour||18).toString().padStart(2,'0')}:00 · ${tp.days||'daily'} · ${sc.occurrences||0} occurrences
        </div>
        ${sc.scene_yaml ? `<details><summary style="cursor:pointer;color:var(--accent);font-size:.82rem;margin-bottom:4px">Show Scene YAML</summary><pre id="scene-yaml-${idx}" style="font-size:.72rem">${(sc.scene_yaml||'').trim()}</pre><button class="btn btn-accent" style="margin-top:4px" onclick="navigator.clipboard.writeText(document.getElementById('scene-yaml-${idx}').textContent).then(()=>toast('Copied ✓'))">📋 Copy</button></details>` : ''}
      </div>`;
    }).join('');
  } else {
    document.getElementById('scenes-list').innerHTML = '<div style="color:var(--text3);padding:12px;font-size:.82rem">No implicit scenes yet. Give it a few days — signal comes before noise drops.</div>';
  }

  // Scene Improvements
  const sceneImprovements = (sceneAnalysis && sceneAnalysis.suggestions) ? sceneAnalysis.suggestions : [];
  const siEl = document.getElementById('scene-improvements-list');
  if (siEl) {
    if (sceneImprovements.length) {
      siEl.innerHTML = sceneImprovements.map((si, idx) => {
        const missingHtml = (si.missing_entities || []).map(me => `
          <div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid var(--border)">
            <span class="badge b-muted" style="font-size:.68rem">${me.confidence_pct}%</span>
            <span style="font-size:.82rem;font-weight:500">${me.entity_id}</span>
            <span style="font-size:.72rem;color:var(--text3);flex:1">${me.rationale || ''}</span>
          </div>`).join('');
        const triggerHtml = (si.suggested_triggers || []).map((tr, ti) => `
          <div style="margin-top:8px">
            <div style="font-size:.8rem;color:var(--text2);margin-bottom:4px">${tr.description || ''}</div>
            ${tr.trigger_yaml ? `<details><summary style="cursor:pointer;color:var(--accent);font-size:.78rem">Show trigger YAML</summary><pre id="si-yaml-${idx}-${ti}" style="font-size:.72rem">${(tr.trigger_yaml||'').trim()}</pre><button class="btn btn-accent" style="margin-top:4px;padding:2px 8px;font-size:.72rem" onclick="addToHA(document.getElementById('si-yaml-${idx}-${ti}').textContent)">+ Add to HA</button></details>` : ''}
          </div>`).join('');
        const scoreColor = si.improvement_score >= 70 ? 'var(--accent)' : si.improvement_score >= 40 ? 'var(--warn)' : 'var(--text3)';
        return `<div class="sug" style="margin-bottom:12px;border-left:3px solid ${scoreColor}">
          <div class="sug-head">
            <h3>🔍 ${si.scene_name || si.scene_id}</h3>
            <span class="badge" style="background:${scoreColor};color:#fff;font-size:.72rem">${si.improvement_score}/100</span>
          </div>
          <div class="desc" style="font-size:.8rem;color:var(--text2);margin-bottom:8px">${si.why_suggested || ''}</div>
          ${(si.scene_entities||[]).length ? `<div style="font-size:.74rem;color:var(--text3);margin-bottom:6px">Current entities: ${si.scene_entities.join(', ')}</div>` : ''}
          ${missingHtml ? `<div style="margin-bottom:8px"><div style="font-size:.78rem;font-weight:600;color:var(--text2);margin-bottom:4px">Missing entities:</div>${missingHtml}</div>` : ''}
          ${triggerHtml ? `<div><div style="font-size:.78rem;font-weight:600;color:var(--text2);margin-bottom:4px">Suggested triggers:</div>${triggerHtml}</div>` : ''}
        </div>`;
      }).join('');
    } else {
      siEl.innerHTML = '<div style="color:var(--text3);padding:12px;font-size:.82rem">No scene improvements yet — run a full training first, or add some HA scenes.</div>';
    }
  }

  // HA Automations
  const haAutos = (haAutomations && haAutomations.automations) ? haAutomations.automations : [];
  haAutomationMap = buildAutomationMap(haAutos);
  renderSuggestions();
  if (haAutos.length) {
    const enabled = haAutos.filter(a => a.current_state === 'on');
    const disabled = haAutos.filter(a => a.current_state !== 'on');
    let haHtml = `<div style="margin-bottom:10px;font-size:.82rem;color:var(--text2)">${haAutos.length} automations (${enabled.length} enabled, ${disabled.length} disabled)</div>`;
    haHtml += enabled.slice(0, 20).map(a => {
      const triggered = a.last_triggered ? new Date(a.last_triggered).toLocaleDateString() : 'never';
      return `<div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border)">
        <span class="badge b-ok" style="font-size:.68rem">on</span>
        <div style="flex:1;font-weight:500;font-size:.82rem">${a.alias}</div>
        <div style="font-size:.72rem;color:var(--text3)">Last: ${triggered}</div>
        <button class="btn btn-danger" style="padding:2px 8px;font-size:.72rem" onclick="removeFromHA('${(a.entity_id||'').replace(/'/g, "\\'")}')">Remove</button>
      </div>`;
    }).join('');
    if (disabled.length) {
      haHtml += `<details style="margin-top:10px"><summary style="cursor:pointer;color:var(--text3);font-size:.82rem">Disabled automations (${disabled.length})</summary><div style="margin-top:6px">`;
      haHtml += disabled.map(a => `<div style="display:flex;align-items:center;gap:8px;padding:4px 0;opacity:.8">
        <span class="badge b-muted" style="font-size:.68rem">off</span>
        <div style="flex:1;font-size:.82rem">${a.alias}</div>
        <button class="btn btn-danger" style="padding:2px 8px;font-size:.72rem" onclick="removeFromHA('${(a.entity_id||'').replace(/'/g, "\\'")}')">Remove</button>
      </div>`).join('');
      haHtml += '</div></details>';
    }
    if (haAutos.length > 20) {
      haHtml += `<div style="font-size:.75rem;color:var(--text3);margin-top:8px">Showing first 20 of ${enabled.length} enabled automations</div>`;
    }
    document.getElementById('ha-automations-list').innerHTML = haHtml;
  } else {
    document.getElementById('ha-automations-list').innerHTML = '<div style="color:var(--text3);padding:12px">No automations loaded yet. Next run should bring them through clean.</div>';
  }

  // Weekly
  const wk=patterns.weekly||{};
  const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const mxW=Math.max(...Object.values(wk).map(v=>v.mean_power_w),1);
  document.getElementById('wk-table').innerHTML=days.map(d=>{
    const v=wk[d]||{mean_power_w:0};
    const p=Math.round(v.mean_power_w/mxW*100);
    const we=d==='Sat'||d==='Sun';
    return`<tr><td>${d}${we?` <span style="color:var(--text3);font-size:.7rem">weekend</span>`:''}</td><td style="font-weight:500">${fmtW(v.mean_power_w)}</td><td><div class="bar-wrap"><div class="bar" style="width:${p}%;background:${we?'var(--purple)':'var(--accent)'}"></div></div></td></tr>`;
  }).join('');

  // Monthly
  const se=patterns.seasonal||{};
  const mos=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const mxM=Math.max(...Object.values(se).map(v=>v.mean_power_w),1);
  document.getElementById('mo-table').innerHTML=mos.filter(m=>se[m]).map(m=>{
    const v=se[m],p=Math.round(v.mean_power_w/mxM*100);
    return`<tr><td>${m}</td><td style="font-weight:500">${fmtW(v.mean_power_w)}</td><td style="color:var(--text3)">${v.mean_temp_c}°C</td><td><div class="bar-wrap"><div class="bar" style="width:${p}%"></div></div></td></tr>`;
  }).join('');

  // Season cards
  const seasons=[['winter','❄️',' Winter'],['spring','🌱',' Spring'],['summer','☀️',' Summer'],['autumn','🍂',' Autumn']];
  document.getElementById('season-cards').innerHTML=seasons.map(([s,ic,nm])=>`
    <div class="season-card">
      <div class="s-icon">${ic}</div>
      <div class="s-name">${nm}</div>
      ${sm[s]?'<span class="badge b-ok">Trained</span>':'<span class="badge b-muted">No data</span>'}
    </div>`).join('');

  // Insights — Energy Dashboard stats (same data HA shows)
  const pd = phantomData && !Array.isArray(phantomData) ? phantomData : {};
  const months = pd.months || [];
  const phantom = pd.overnight_baseline || {};
  const total12mo = pd.total_12mo_kwh || 0;
  const momPct = pd.mom_pct;

  function kwhBar(val, max) {
    const pct = max > 0 ? Math.round(val/max*100) : 0;
    return `<div class="bar-wrap" style="flex:1"><div class="bar accent" style="width:${pct}%"></div></div>`;
  }

  let phantomHtml = '';

  // 12-month total header
  if (total12mo) {
    phantomHtml += `<div style="background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:14px">
      <div style="font-size:.78rem;color:var(--text3);margin-bottom:6px">LAST 12 MONTHS (from Energy Dashboard)</div>
      <div style="display:flex;align-items:baseline;gap:8px">
        <span style="font-size:1.6rem;font-weight:700;color:var(--text)">${total12mo.toLocaleString()}</span>
        <span style="color:var(--text3)">kWh total grid consumption</span>
      </div>
    </div>`;
  }

  // Phantom baseline card
  if (phantom.overnight_kwh_year) {
    const phantomPct = total12mo > 0 ? Math.round(100 * phantom.overnight_kwh_year / total12mo) : null;
    phantomHtml += `<div style="background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:14px">
      <div style="font-size:.78rem;color:var(--text3);margin-bottom:6px">OVERNIGHT BASELINE (02:00–05:00)</div>
      <div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap">
        <span style="font-size:1.4rem;font-weight:700;color:var(--amber)">${phantom.avg_idle_kwh_per_hour}</span>
        <span style="color:var(--text3)">kWh/hour</span>
        <span style="color:var(--text2)">→ ${Math.round(phantom.overnight_kwh_year).toLocaleString()} kWh/year</span>
        ${phantomPct ? `<span style="color:var(--amber)">(${phantomPct}% of usage)</span>` : ''}
      </div>
      <div style="font-size:.75rem;color:var(--text3);margin-top:4px">Typical overnight draw when on shore power</div>
    </div>`;
  }

  // Day-normalized comparison
  const cmp = pd.same_days_comparison || {};
  const avgDaily = pd.this_month_avg_daily;
  const lastAvgDaily = pd.last_month_avg_daily;
  if (cmp.days) {
    const arrow = cmp.delta_pct > 0
      ? `<span style="color:var(--red)">↑ ${cmp.delta_pct}%</span>`
      : cmp.delta_pct < 0
      ? `<span style="color:var(--green)">↓ ${Math.abs(cmp.delta_pct)}%</span>`
      : '';
    phantomHtml += `<div style="background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:14px">
      <div style="font-size:.78rem;color:var(--text3);margin-bottom:8px">SAME-PERIOD COMPARISON (first ${cmp.days} days)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div>
          <div style="font-size:.72rem;color:var(--text3)">This month</div>
          <div style="font-size:1.2rem;font-weight:700;color:var(--text)">${cmp.this_month_first_n} kWh</div>
          <div style="font-size:.72rem;color:var(--text3)">${avgDaily} kWh/day avg</div>
        </div>
        <div>
          <div style="font-size:.72rem;color:var(--text3)">Last month (first ${cmp.days} days)</div>
          <div style="font-size:1.2rem;font-weight:700;color:var(--text)">${cmp.last_month_first_n} kWh</div>
          <div style="font-size:.72rem;color:var(--text3)">${lastAvgDaily} kWh/day avg</div>
        </div>
      </div>
      <div style="margin-top:8px;font-size:.82rem">${arrow} ${cmp.delta_kwh > 0 ? '+' : ''}${cmp.delta_kwh} kWh</div>
    </div>`;
  }

  // Monthly breakdown table
  if (months.length) {
    const maxMonth = Math.max(...months.map(m=>m.kwh), 1);
    phantomHtml += `<div style="font-size:.82rem;font-weight:600;color:var(--text2);margin:12px 0 8px">Monthly Usage</div>
    <table>
      <thead><tr><th>Month</th><th>kWh</th><th style="width:150px"></th></tr></thead>
      <tbody>${months.slice().reverse().map(m => `
        <tr>
          <td style="font-weight:500">${m.month}</td>
          <td style="font-weight:600;color:var(--text)">${m.kwh}</td>
          <td>${kwhBar(m.kwh, maxMonth)}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
  } else if (!phantom.overnight_kwh_year) {
    phantomHtml = '<div style="color:var(--text3);padding:12px">No Energy Dashboard stats available yet.</div>';
  }
  document.getElementById('phantom-list').innerHTML = phantomHtml;

  // ── Appliance Fingerprints ──
  fetch('api/appliance_fingerprints').then(r=>r.json()).catch(()=>({appliances:[],recent_events:[],total_events:0})).then(fp => {
    const el = document.getElementById('appliance-list');
    if (!fp.appliances || fp.appliances.length === 0) {
      el.innerHTML = '<div style="color:var(--text3);padding:12px">No appliance signatures detected yet. Runs after next training cycle.</div>';
      return;
    }
    let html = `<div style="color:var(--text3);font-size:.78rem;margin-bottom:8px">${fp.total_events} power events detected, ${fp.identification_rate}% identified from ${fp.entities_scanned} sensors</div>`;
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px">';
    fp.appliances.forEach(a => {
      const peakStr = a.peak_hours ? a.peak_hours.map(h=>`${h}:00`).join(', ') : '—';
      html += `<div class="card" style="padding:12px">
        <div style="font-size:1.3rem;margin-bottom:4px">${a.icon} ${a.appliance.replace(/_/g,' ')}</div>
        <div style="font-size:.82rem;color:var(--text3)">
          <div><b>${a.avg_power_w}W</b> avg · ${a.event_count} events</div>
          <div>~${a.avg_duration_min} min each · ${a.daily_avg}/day</div>
          <div>~${a.est_monthly_kwh} kWh/month</div>
          <div style="margin-top:4px">Peak: ${peakStr}</div>
        </div>
      </div>`;
    });
    html += '</div>';
    if (fp.recent_events && fp.recent_events.length > 0) {
      html += '<details style="margin-top:12px"><summary style="cursor:pointer;color:var(--accent);font-size:.85rem">Recent events (' + fp.recent_events.length + ')</summary>';
      html += '<div class="table-wrap"><table><thead><tr><th>Time</th><th>Appliance</th><th>Power</th><th>Duration</th><th>Confidence</th></tr></thead><tbody>';
      fp.recent_events.slice(0,15).forEach(e => {
        const t = new Date(e.start).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
        html += `<tr><td>${t}</td><td>${e.icon||'❓'} ${(e.appliance||'unknown').replace(/_/g,' ')}</td><td>${e.power_w}W</td><td>${e.duration_min}min</td><td>${e.confidence||0}%</td></tr>`;
      });
      html += '</tbody></table></div></details>';
    }
    el.innerHTML = html;
  });

  // ── Energy + Weather History ──
  function loadEnergyWeather(){
    const days = document.getElementById('ew-range').value;
    fetch(`api/energy_weather_history?days=${days}`).then(r=>r.json()).catch(()=>({days:[]})).then(ew => {
      const body = document.getElementById('ew-body');
      const chart = document.getElementById('ew-chart');
      if (!ew.days || ew.days.length === 0) { body.innerHTML='<tr><td colspan="6" style="color:var(--text3);padding:8px">No data yet. Needs energy + temperature sensors.</td></tr>'; return; }

      const maxKwh = Math.max(...ew.days.map(d => d.kwh));
      const maxTemp = Math.max(...ew.days.filter(d=>d.avg_temp!==null).map(d=>d.avg_temp), 1);
      const minTemp = Math.min(...ew.days.filter(d=>d.avg_temp!==null).map(d=>d.avg_temp), 0);

      // Mini chart — dual bar (energy=blue, temp=amber)
      const chartW = Math.min(ew.days.length, 90);
      const recent = ew.days.slice(-chartW);
      chart.innerHTML = `<div style="display:flex;align-items:flex-end;height:80px;gap:1px;margin-bottom:4px">
        ${recent.map(d => {
          const kwPct = maxKwh > 0 ? d.kwh / maxKwh * 100 : 0;
          const tempPct = d.avg_temp !== null && (maxTemp - minTemp) > 0 ? (d.avg_temp - minTemp) / (maxTemp - minTemp) * 100 : 0;
          const isWeekend = d.is_weekend;
          return `<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:1px;height:100%;justify-content:flex-end" title="${d.date}: ${d.kwh}kWh, ${d.avg_temp!==null?d.avg_temp+'°C':'?'}">
            <div style="width:100%;background:${isWeekend?'var(--amber)':'var(--accent)'};height:${kwPct}%;min-height:1px;border-radius:1px 1px 0 0;opacity:0.7"></div>
          </div>`;
        }).join('')}
      </div>
      <div style="display:flex;justify-content:space-between;font-size:.65rem;color:var(--text3)">
        <span>${recent[0]?.date||''}</span>
        <span style="color:var(--accent)">■ kWh (amber=weekend)</span>
        <span>${recent[recent.length-1]?.date||''}</span>
      </div>`;

      // Table
      body.innerHTML = ew.days.slice().reverse().map(d => {
        const barW = maxKwh > 0 ? d.kwh / maxKwh * 100 : 0;
        const tempColor = d.avg_temp !== null ? (d.avg_temp < 5 ? 'var(--blue)' : d.avg_temp > 20 ? 'var(--red)' : 'var(--text3)') : 'var(--text3)';
        return `<tr style="border-bottom:1px solid var(--border);${d.is_weekend?'background:rgba(255,255,255,0.02)':''}">
          <td style="padding:3px 4px;white-space:nowrap">${d.day_name} ${d.date}</td>
          <td style="padding:3px 4px;text-align:right;font-weight:600">${d.kwh}</td>
          <td style="padding:3px 4px;text-align:right;color:${tempColor}">${d.avg_temp!==null?d.avg_temp:'—'}</td>
          <td style="padding:3px 4px;text-align:right;color:${tempColor}">${d.min_temp!==null?d.min_temp:'—'}</td>
          <td style="padding:3px 4px;text-align:right;color:${tempColor}">${d.max_temp!==null?d.max_temp:'—'}</td>
          <td style="padding:3px 4px;width:120px"><div style="height:10px;border-radius:2px;background:var(--accent);width:${barW}%;opacity:0.6"></div></td>
        </tr>`;
      }).join('');
    });
  }
  loadEnergyWeather();

  // ── NILM Disaggregation ──
  fetch('api/nilm').then(r=>r.json()).catch(()=>({breakdown:[]})).then(n => {
    const sec = document.getElementById('nilm-section');
    if (!n.current_breakdown || n.current_breakdown.length === 0) {
      if (n.discovered_appliances && n.discovered_appliances.length > 0) sec.style.display='';
      else { sec.style.display='none'; return; }
    }
    sec.style.display='';

    // Current breakdown (pie-like bar)
    if (n.current_breakdown && n.current_breakdown.length > 0) {
      const total = n.current_total_w || 0;
      const colors = ['var(--accent)','var(--blue)','var(--amber)','var(--red)','#8b5cf6','#10b981','#f59e0b','#ef4444','#6366f1','#ec4899'];
      document.getElementById('nilm-current').innerHTML = `
        <div class="card" style="padding:12px">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <b>Current: ${total}W</b>
            <span style="font-size:.75rem;color:var(--text3)">${n.readings_count||0} readings · ${n.edges_detected||0} edges · ${n.events_paired||0} events</span>
          </div>
          <div style="display:flex;height:24px;border-radius:4px;overflow:hidden;margin-bottom:8px">
            ${n.current_breakdown.map((b,i) => `<div style="width:${total>0?b.estimated_w/total*100:0}%;background:${colors[i%colors.length]}" title="${b.appliance}: ${b.estimated_w}W"></div>`).join('')}
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;font-size:.78rem">
            ${n.current_breakdown.map((b,i) => `<span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${colors[i%colors.length]}"></span> ${b.icon} ${b.appliance}: ${b.estimated_w}W</span>`).join('')}
          </div>
        </div>`;
    }

    // Discovered appliances
    if (n.discovered_appliances && n.discovered_appliances.length > 0) {
      document.getElementById('nilm-appliances').innerHTML =
        '<div style="font-size:.82rem;margin-bottom:4px"><b>Discovered Appliance Slots</b></div>' +
        n.discovered_appliances.map(a => `
          <div class="card" style="padding:8px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center">
            <div>${a.icon} <b>${a.appliance}</b> (~${a.centroid_w}W) ${a.source==='user_trained'?'<span style="font-size:.65rem;background:var(--accent);color:#000;padding:1px 4px;border-radius:3px">trained</span>':''}</div>
            <div style="font-size:.75rem;color:var(--text3)">${a.event_count} events · ${a.avg_duration_min}min avg · ${a.total_kwh} kWh · ${a.match_confidence}% match</div>
          </div>
        `).join('');
    }

    // 24h energy breakdown
    if (n.energy_24h && n.energy_24h.length > 0) {
      document.getElementById('nilm-energy').innerHTML = `
        <div style="font-size:.82rem;margin-bottom:4px"><b>Last 24h Energy by Appliance</b> (${n.total_kwh_24h} kWh total)</div>
        ${n.energy_24h.map(e => `<div style="display:flex;justify-content:space-between;font-size:.8rem;padding:2px 0"><span>${e.appliance}</span><b>${e.kwh_24h} kWh</b></div>`).join('')}`;
    }
  });

  // ── Device Training ──
  fetch('api/power_sensors').then(r=>r.json()).catch(()=>({sensors:[]})).then(ps => {
    const sel = document.getElementById('train-power-entity');
    if (ps.sensors) ps.sensors.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.entity_id || s;
      opt.textContent = s.entity_id || s;
      sel.appendChild(opt);
    });
  });
  fetch('api/custom_signatures').then(r=>r.json()).catch(()=>({signatures:[]})).then(cs => {
    const el = document.getElementById('custom-sigs');
    if (!cs.signatures || cs.signatures.length === 0) { el.innerHTML='<div style="color:var(--text3);font-size:.8rem">No custom signatures yet. Train your first device above.</div>'; return; }
    el.innerHTML = '<div style="font-size:.85rem;margin-bottom:6px"><b>Trained Devices</b></div>' + cs.signatures.map(s => `
      <div class="card" style="padding:8px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center">
        <div><b>${s.name}</b> — ${s.peak_delta_w}W peak, ${s.avg_delta_w}W avg, ${s.shape}, ${s.duration_min}min</div>
        <button class="btn" style="font-size:.7rem;padding:2px 6px" onclick="deleteSignature('${s.name}')">🗑</button>
      </div>
    `).join('');
  });

  // ── Anomaly Feedback Stats ──
  fetch('api/feedback_stats').then(r=>r.json()).catch(()=>({stats:{}})).then(fb => {
    const el = document.getElementById('feedback-stats');
    const s = fb.stats || {};
    el.innerHTML = `<div class="card" style="padding:10px">
      Confirmed: <b>${s.confirmed||0}</b> · Dismissed: <b>${s.dismissed||0}</b> · Total: ${s.total||0}
      ${fb.false_positive_entities && fb.false_positive_entities.length ? '<br><span style="font-size:.78rem;color:var(--amber)">Auto-widened bands for: ' + fb.false_positive_entities.join(', ') + '</span>' : ''}
    </div>`;
    if (fb.sharing_enabled) document.getElementById('sharing-toggle').checked = true;
  });

  // ── History Depth ──
  fetch('api/settings').then(r=>r.json()).catch(()=>({settings:{}})).then(s => {
    const d = (s.settings||{}).days_history || 30;
    document.getElementById('history-depth').value = String(d);
  });

  // ── Predicted Routines ──
  fetch('api/routines').then(r=>r.json()).catch(()=>({routines:[]})).then(rd => {
    const el = document.getElementById('routines-list');
    if (!rd.routines || rd.routines.length === 0) {
      el.innerHTML = '<div style="color:var(--text3);padding:12px">No recurring routines detected yet. Needs humidity/temperature sensor data.</div>';
      return;
    }
    el.innerHTML = rd.routines.map(r => `
      <div class="card" style="padding:12px;margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div><span style="font-size:1.3rem">${r.icon}</span> <b>${r.activity.replace(/_/g,' ')} — ${r.room.replace(/_/g,' ')}</b></div>
          <span style="font-size:.75rem;color:var(--accent)">${r.confidence}% confident</span>
        </div>
        <div style="font-size:.85rem;margin-top:6px">
          ⏰ Usually at <b>${r.typical_time}</b> (${r.day_pattern}) · ${r.events_per_day}/day · ~${r.avg_duration_min} min each
        </div>
        <div style="font-size:.85rem;margin-top:4px;color:var(--accent)">
          💡 Suggestion: ${r.suggestion}
        </div>
        ${r.yaml ? `<details style="margin-top:8px"><summary style="cursor:pointer;color:var(--accent);font-size:.8rem">Pre-heat automation YAML</summary><pre style="font-size:.72rem;margin-top:4px">${r.yaml}</pre></details>` : ''}
      </div>
    `).join('');
  });

  // Insights — Routine Drift
  if (driftData && driftData.drifts && driftData.drifts.length) {
    const sig = driftData.drifts.filter(d=>d.significant);
    document.getElementById('drift-info').innerHTML = `
      <div style="margin-bottom:14px;font-size:.88rem;font-weight:600;color:${sig.length?'var(--amber)':'var(--green)'}">${driftData.summary || 'No significant drift'}</div>
      <div class="pat-grid">${driftData.drifts.map(d=>`
        <div class="pat-item">
          <div class="pi-label">${d.metric.replace(/_/g,' ')}</div>
          <div class="pi-val" style="color:${d.significant?'var(--amber)':'var(--text)'}">${d.diff_min!=null?(d.diff_min>0?'+':'')+d.diff_min+' min':d.diff+' '+d.unit}</div>
          ${d.direction?`<div style="font-size:.72rem;color:var(--text3)">${d.direction}</div>`:''}
        </div>`).join('')}
      </div>`;
  } else {
    const reason = driftData.reason || 'Not enough data yet';
    document.getElementById('drift-info').innerHTML = `<div style="color:var(--text3);padding:12px">${reason}</div>`;
  }

  // Insights — Automation Health
  if (autoScores && autoScores.length) {
    document.getElementById('auto-scores').innerHTML = `
      <div class="table-wrap">
      <table class="auto-health-table">
        <thead><tr><th>Automation</th><th>Triggers/7d</th><th>Overrides</th><th>Score</th><th style="width:80px"></th></tr></thead>
        <tbody>${autoScores.map(a=>{
          const col = a.score>=70?'var(--green)':a.score>=40?'var(--amber)':'var(--red)';
          const bc = a.score>=70?'b-ok':a.score>=40?'b-warn':'b-alert';
          const emoji = a.score>=70?'🟢':a.score>=40?'🟡':'🔴';
          return `<tr>
            <td><div style="font-weight:500">${a.name}</div><div style="font-size:.7rem;color:var(--text3)">${a.entity_id}</div></td>
            <td>${a.triggers_7d}</td>
            <td style="color:${a.overrides?'var(--amber)':'var(--text3)'}">${a.overrides}</td>
            <td><span class="badge ${bc}">${emoji} ${a.score}/100</span></td>
            <td><div class="bar-wrap"><div class="bar" style="width:${a.score}%;background:${col}"></div></div></td>
          </tr>`;}).join('')}
        </tbody>
      </table>
      </div>`;
  } else {
    document.getElementById('auto-scores').innerHTML = '<div style="color:var(--text3);padding:12px">No automations scored yet.</div>';
  }
  // Insights — Automation Gaps
  const gaps = (gapData && gapData.gaps) ? gapData.gaps : [];
  if (gaps.length) {
    const missing = gaps.filter(g=>g.status==='missing');
    const poor = gaps.filter(g=>g.status==='exists_poor');
    const disabled = gaps.filter(g=>g.status==='exists_disabled');
    const working = gaps.filter(g=>g.status==='exists_working');
    let html = '';
    if (gapData.summary) html += `<div style="margin-bottom:14px;font-size:.82rem;color:var(--text2)">${gapData.summary}</div>`;
    if (missing.length) {
      html += `<div style="font-weight:600;margin-bottom:8px;color:var(--accent)">Missing automations (${missing.length})</div>`;
      missing.forEach((g,i)=>{
        const yid = 'gap-yaml-'+i;
        const bid = 'gap-add-'+i;
        const costBadge = (g.cost_estimate && g.cost_estimate.monthly_saving_eur > 0)
          ? `<span class="badge b-ok" style="background:var(--green);color:#fff;font-size:.72rem">Saves ~€${g.cost_estimate.monthly_saving_eur.toFixed(1)}/month</span>`
          : '';
        html += `<div class="sug" style="margin-bottom:10px">
          <div class="sug-head"><h3>${g.suggestion}</h3><div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap"><span class="badge b-alert">missing</span>${costBadge}</div></div>
          ${g.ha_automation_yaml ? `<pre id="${yid}" style="font-size:.72rem">${g.ha_automation_yaml}</pre>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
              <button class="btn btn-accent" onclick="navigator.clipboard.writeText(document.getElementById('${yid}').textContent).then(()=>toast('Copied ✓'))">📋 Copy YAML</button>
              <button class="btn btn-success" id="${bid}" onclick="addGapToHA('${yid}','${bid}')">+ Add to HA</button>
            </div>` : ''}
        </div>`;
      });
    }
    if (poor.length) {
      html += `<div style="font-weight:600;margin:14px 0 8px;color:var(--amber)">Improvable automations (${poor.length})</div>`;
      poor.forEach(g=>{
        html += `<div class="sug" style="margin-bottom:10px;border-left:3px solid var(--amber)">
          <div class="sug-head"><h3>${g.suggestion}</h3><span class="badge b-warn">exists — poor</span></div>
          <div class="desc" style="color:var(--amber)">${g.improvement||''}</div>
        </div>`;
      });
    }
    if (disabled.length) {
      html += `<div style="font-weight:600;margin:14px 0 8px;color:var(--text2)">Disabled automations (${disabled.length})</div>`;
      disabled.forEach(g=>{
        html += `<div class="sug" style="margin-bottom:10px">
          <div class="sug-head"><h3>${g.suggestion}</h3><span class="badge b-muted">disabled</span></div>
          <div class="desc">${g.improvement||''}</div>
        </div>`;
      });
    }
    if (working.length) {
      html += `<details style="margin-top:14px"><summary style="cursor:pointer;color:var(--text3);font-size:.82rem">Working automations (${working.length}) — collapsed</summary><div style="margin-top:8px">`;
      working.forEach(g=>{
        html += `<div class="sug" style="margin-bottom:6px;opacity:.7">
          <div class="sug-head"><h3>${g.suggestion}</h3><span class="badge b-ok">working</span></div>
        </div>`;
      });
      html += '</div></details>';
    }
    document.getElementById('auto-gap').innerHTML = html;
  } else {
    document.getElementById('auto-gap').innerHTML = '<div style="color:var(--text3);padding:12px">No gap analysis yet — run Habitus first.</div>';
  }


  // Settings
  document.getElementById('raw-state').textContent = JSON.stringify(state,null,2);

  // ── 5a) Unified Insights Summary ────────────────────────────────────────
  (async () => {
    const [conflictsData, healthData, batteryData, gapD, guestData] = await Promise.all([
      fetch('api/conflicts').then(r=>r.json()).catch(()=>({})),
      fetch('api/automation_health').then(r=>r.json()).catch(()=>({})),
      fetch('api/battery_status').then(r=>r.json()).catch(()=>({})),
      Promise.resolve(gapData),
      fetch('api/guest_mode').then(r=>r.json()).catch(()=>({})),
    ]);

    const chips = [];

    // Conflicts
    const nConflicts = conflictsData?.count ?? (conflictsData?.conflicts?.length ?? 0);
    if (nConflicts > 0) {
      chips.push(`<a href="#" class="insight-chip chip-danger" onclick="gotoTab('tab-smarthome',null);return false;" title="Jump to Conflicts">⚡ ${nConflicts} conflict${nConflicts!==1?'s':''}</a>`);
    }

    // Dead automations
    const deadCount = (healthData?.automations || []).filter(a => a.status === 'dead').length;
    if (deadCount > 0) {
      chips.push(`<a href="#" class="insight-chip chip-warn" onclick="gotoTab('tab-smarthome',null);return false;" title="Jump to Automation Health">💤 ${deadCount} dead automation${deadCount!==1?'s':''}</a>`);
    }

    // Battery alerts
    const batteryAlerts = (batteryData?.batteries || []).filter(b => b.level_pct !== null && b.level_pct < 20).length;
    if (batteryAlerts > 0) {
      chips.push(`<a href="#" class="insight-chip chip-danger" onclick="gotoTab('tab-smarthome',null);return false;" title="Jump to Battery Status">🔋 ${batteryAlerts} battery alert${batteryAlerts!==1?'s':''}</a>`);
    }

    // Missing automation gaps
    const missingGaps = (gapD?.gaps || []).filter(g => g.status === 'missing').length;
    if (missingGaps > 0) {
      chips.push(`<a href="#" class="insight-chip chip-info" onclick="gotoTab('tab-smarthome',null);return false;" title="Jump to Automation Gaps">🔧 ${missingGaps} missing automation${missingGaps!==1?'s':''}</a>`);
    }

    // Cost saving potential
    const totalSaving = [...(gapD?.gaps||[]), ...(Array.isArray(smartSuggestions?.suggestions) ? smartSuggestions.suggestions : [])]
      .reduce((sum, item) => sum + (item?.cost_estimate?.monthly_saving_eur || 0), 0);
    if (totalSaving > 0) {
      chips.push(`<a href="#" class="insight-chip chip-ok" onclick="gotoTab('tab-smarthome',null);return false;" title="Potential monthly energy savings">€${totalSaving.toFixed(1)}/month savings</a>`);
    }

    // Guest mode
    const guestProb = guestData?.guest_probability ?? 0;
    if (guestProb > 0.4) {
      chips.push(`<span class="insight-chip chip-info">👥 Guests may be present (${Math.round(guestProb*100)}%)</span>`);
    }

    const container = document.getElementById('insights-chips');
    if (container) {
      if (chips.length) {
        container.innerHTML = chips.join('');
      } else {
        container.innerHTML = '<span style="color:var(--green);font-size:.82rem">✓ No issues detected</span>';
      }
    }
  })();

  // ── 5d) Section collapse/expand (backed by localStorage) ─────────────────
  document.querySelectorAll('.sec-header').forEach(hdr => {
    const sec = hdr.closest('.sec');
    if (!sec) return;
    const body = sec.querySelector('.sec-body, :scope > :not(.sec-header)');
    if (!body) return;

    // Make the next sibling a .sec-body if not already
    const sectionId = hdr.querySelector('h2')?.textContent?.trim().replace(/[^a-z0-9]/gi,'_').toLowerCase();
    if (!sectionId) return;
    const storageKey = 'habitus_collapsed_' + sectionId;

    hdr.classList.add('collapsible');
    // Wrap content in a .sec-body div if it isn't already
    const secBody = (() => {
      let el = hdr.nextElementSibling;
      if (el && !el.classList.contains('sec-body')) {
        const wrapper = document.createElement('div');
        wrapper.className = 'sec-body';
        while (hdr.nextElementSibling) {
          wrapper.appendChild(hdr.nextElementSibling);
        }
        sec.appendChild(wrapper);
        return wrapper;
      }
      return el;
    })();
    if (!secBody) return;

    const collapsed = localStorage.getItem(storageKey) === '1';
    if (collapsed) {
      hdr.classList.add('collapsed');
      secBody.classList.add('collapsed');
      secBody.style.maxHeight = '0';
    } else {
      secBody.style.maxHeight = secBody.scrollHeight + 'px';
    }

    hdr.addEventListener('click', () => {
      const isNowCollapsed = !hdr.classList.contains('collapsed');
      hdr.classList.toggle('collapsed', isNowCollapsed);
      secBody.classList.toggle('collapsed', isNowCollapsed);
      secBody.style.maxHeight = isNowCollapsed ? '0' : secBody.scrollHeight + 'px';
      localStorage.setItem(storageKey, isNowCollapsed ? '1' : '0');
    });
  });

  // Schedule info
  const schedule = '{{ schedule }}' || state.schedule || 'overnight';
  const trainTime = '{{ train_time }}' || state.train_time || '02:00';
  const lastScore = state.last_score ? new Date(state.last_score).toLocaleString() : 'never';
  const lastTrain = state.last_run  ? new Date(state.last_run).toLocaleString()   : 'never';
  const modeLabels = {
    overnight: `<span class="badge b-info">🌙 Overnight</span>`,
    continuous: `<span class="badge b-warn">🔄 Continuous</span>`,
  };
  document.getElementById('schedule-info').innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px">
      <div class="pat-item">
        <div class="pi-label">Mode</div>
        <div style="margin-top:4px">${modeLabels[schedule] || schedule}</div>
      </div>
      <div class="pat-item">
        <div class="pi-label">Train Time</div>
        <div class="pi-val">${schedule === 'overnight' ? trainTime : 'every run'}</div>
      </div>
      <div class="pat-item">
        <div class="pi-label">Last Full Train</div>
        <div style="font-size:.78rem;color:var(--text2);margin-top:2px">${lastTrain}</div>
      </div>
      <div class="pat-item">
        <div class="pi-label">Last Score</div>
        <div style="font-size:.78rem;color:var(--text2);margin-top:2px">${lastScore}</div>
      </div>
    </div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:12px;font-size:.8rem;color:var(--text2);line-height:1.6">
      ${schedule === 'overnight'
        ? `🌙 <strong style="color:var(--text)">Overnight mode</strong> — model retrains once per day at <strong>${trainTime}</strong> when your home is idle. During the day, Habitus scores every ${state.scan_interval_hours || 6}h using the existing model (fast, no resource usage).`
        : `🔄 <strong style="color:var(--text)">Continuous mode</strong> — model retrains on every scan cycle. More responsive to pattern changes but uses more CPU. Not recommended for low-powered devices.`
      }
      <br><br>Change this in <strong>Settings → Add-on → Configuration</strong> in Home Assistant.
    </div>`;
}

function runNilm(){
  toast('Running NILM disaggregation...');
  fetch('api/nilm/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({days:7})})
    .then(r=>r.json()).then(d=>{
      if(d.error){toast(d.error,'err');}else{toast(`NILM: ${d.appliance_slots} appliances found, ${d.current_total_w}W current`);location.reload();}
    }).catch(()=>toast('NILM failed','err'));
}
function startTraining(){
  const entity = document.getElementById('train-power-entity').value;
  if(!entity){toast('Select a power sensor first','err');return;}
  fetch('api/training/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({power_entity:entity})})
    .then(r=>r.json()).then(d=>{
      document.getElementById('train-idle').style.display='none';
      document.getElementById('train-recording').style.display='';
      document.getElementById('train-baseline').textContent='Baseline: '+(d.baseline_w||0).toFixed(0)+'W';
      toast('Recording started — turn on your device now');
    }).catch(()=>toast('Failed to start training','err'));
}
function stopTraining(){
  const name = document.getElementById('train-device-name').value;
  if(!name){toast('Enter a device name','err');return;}
  fetch('api/training/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({device_name:name})})
    .then(r=>r.json()).then(d=>{
      document.getElementById('train-idle').style.display='';
      document.getElementById('train-recording').style.display='none';
      if(d.error){toast(d.error,'err');}else{toast(`Saved: ${d.name} — ${d.peak_delta_w}W peak, ${d.shape} shape`);}
    }).catch(()=>toast('Failed to save','err'));
}
function deleteSignature(name){
  fetch('api/custom_signatures/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})
    .then(r=>r.json()).then(()=>{toast('Deleted');location.reload();});
}
function toggleSharing(enabled){
  fetch('api/sharing',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled})})
    .then(()=>toast(enabled?'Anonymous sharing enabled':'Sharing disabled'));
}
function saveHistoryDepth(){
  const days = parseInt(document.getElementById('history-depth').value);
  fetch('api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({days_history:days})})
    .then(()=>toast(`History depth set to ${days} days. Retrain to apply.`));
}
function anomalyFeedback(id, action, entityId, score){
  fetch('api/anomaly_feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({anomaly_id:id,action,entity_id:entityId,score})})
    .then(r=>r.json()).then(d=>{
      toast(action==='confirmed'?'✅ Confirmed — model will learn':'❌ Dismissed — widening normal band');
      const btn = document.getElementById('fb-'+id);
      if(btn) btn.innerHTML='<span style="color:var(--text3)">'+( action==='confirmed'?'✅':'❌')+'</span>';
    });
}
function doRefresh(){load();toast('Refreshed ✓');}

function confirmRescan(){
  document.getElementById('rescan-warn').style.display='block';
  const btn=document.getElementById('btn-rescan');
  btn.textContent='⚠ Confirm Rescan';
  btn.onclick=doRescan;
}
async function doRescan(){
  const btn=document.getElementById('btn-rescan');
  btn.disabled=true; btn.textContent='Triggered…';
  document.getElementById('rescan-warn').style.display='none';
  const r=await fetch('api/rescan',{method:'POST'});
  const d=await r.json();
  if(d.ok) toast('Full rescan started ✓');
  else toast('Failed: '+(d.error||'?'),'te');
  btn.disabled=false; btn.textContent='🔄 Full Rescan';
  btn.onclick=confirmRescan;
}

// Entity picker
let allEntities = [];
async function loadEntities() {
  try {
    const r = await fetch('api/entities');
    allEntities = await r.json();
    filterEntities();
  } catch(e) { console.warn('Entity load failed', e); }
}
function filterEntities() {
  const q = (document.getElementById('entity-search').value || '').toLowerCase();
  const domain = document.getElementById('entity-domain-filter').value;
  const filtered = allEntities.filter(e => {
    if (domain && !e.entity_id.startsWith(domain + '.')) return false;
    if (q && !e.entity_id.includes(q) && !(e.name||'').toLowerCase().includes(q)) return false;
    return true;
  }).slice(0, 50);
  const el = document.getElementById('entity-results');
  if (!filtered.length) {
    el.innerHTML = '<div style="color:var(--text3);font-size:.82rem;padding:8px">No entities found.</div>';
    return;
  }
  el.innerHTML = filtered.map(e => `
    <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border)">
      <span style="font-size:.72rem;color:var(--accent);min-width:50px">${e.entity_id.split('.')[0]}</span>
      <div style="flex:1">
        <div style="font-weight:500;font-size:.82rem">${e.name || e.entity_id}</div>
        <div style="font-size:.7rem;color:var(--text3)">${e.entity_id}</div>
      </div>
      <div style="font-size:.78rem;color:var(--text2)">${e.state || ''}</div>
      <button class="btn btn-accent" style="font-size:.7rem;padding:2px 8px" onclick="navigator.clipboard.writeText('${e.entity_id}').then(()=>toast('Copied: ${e.entity_id}'))">📋</button>
    </div>`).join('');
}
loadEntities();

// ══ New Feature Loaders ══════════════════════════════════════

async function loadAutomationHealth() {
  const data = await fetch('api/automation_health').then(r=>r.json()).catch(()=>({automations:[]}));
  const el = document.getElementById('automation-health-list');
  if (!el) return;
  const autos = data.automations || [];
  if (!autos.length) { el.innerHTML = '<div style="color:var(--text3);padding:12px">No automation health data yet. Run a full train first.</div>'; return; }
  const statusColors = {healthy:'#4caf50',stale:'#ff9800',dead:'#f44336',over_triggering:'#9c27b0'};
  el.innerHTML = autos.slice(0,20).map(a => `
    <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)">
      <span style="background:${statusColors[a.status]||'#999'};color:#fff;border-radius:12px;padding:2px 10px;font-size:.75rem;white-space:nowrap">${a.status}</span>
      <div style="flex:1">
        <div style="font-size:.9rem">${a.alias}</div>
        <div style="color:var(--text3);font-size:.78rem">${a.recommendation}</div>
      </div>
      ${a.status==='dead'?`<button onclick="disableAutomation('${a.entity_id}')" style="background:#f44336;color:#fff;border:none;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:.8rem">Disable</button>`:''}
    </div>`).join('');
}

async function disableAutomation(entityId) {
  if (!confirm('Disable ' + entityId + '?')) return;
  const r = await fetch('api/automation_health/disable', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({entity_id:entityId})}).then(r=>r.json()).catch(()=>({}));
  if (r.success) { toast('Disabled ' + entityId); loadAutomationHealth(); }
  else toast('Error: ' + (r.error||'unknown'), 'err');
}

async function loadDetectedRoutines() {
  const data = await fetch('api/routine_builder').then(r=>r.json()).catch(()=>({routines:[]}));
  const el = document.getElementById('detected-routines-list');
  if (!el) return;
  const routines = data.routines || [];
  if (!routines.length) { el.innerHTML = '<div style="color:var(--text3);padding:12px">No routines detected yet. Run a full train with sufficient history.</div>'; return; }
  el.innerHTML = routines.slice(0,10).map(r => {
    const steps = (r.steps||[]).map(s=>s.entity_id).join(' → ');
    return `<div style="padding:10px 0;border-bottom:1px solid var(--border)">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <span style="background:var(--accent);color:#fff;border-radius:12px;padding:2px 10px;font-size:.75rem">${r.time_cluster}</span>
        <span style="color:var(--text3);font-size:.78rem">${r.frequency_days} days · ${Math.round(r.confidence*100)}% confidence</span>
      </div>
      <div style="font-size:.85rem;color:var(--text2)">${steps}</div>
      <button onclick="showYaml('routine-yaml-${r.time_cluster}-${r.frequency_days}')" style="background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:3px 10px;font-size:.78rem;cursor:pointer;margin-top:4px">View YAML</button>
      <button onclick="addRoutineToHA('${encodeURIComponent(r.generated_yaml||'')}')" style="background:var(--success,#4caf50);color:#fff;border:none;border-radius:6px;padding:3px 10px;font-size:.78rem;cursor:pointer;margin-top:4px;margin-left:4px">Create as Automation</button>
      <pre id="routine-yaml-${r.time_cluster}-${r.frequency_days}" style="display:none;background:var(--card2);padding:8px;border-radius:6px;font-size:.75rem;margin-top:6px;overflow-x:auto;white-space:pre-wrap">${(r.generated_yaml||'').replace(/</g,'&lt;')}</pre>
    </div>`;
  }).join('');
}

async function loadGuestMode() {
  const data = await fetch('api/guest_mode').then(r=>r.json()).catch(()=>({guest_probability:0}));
  const banner = document.getElementById('guest-mode-banner');
  const details = document.getElementById('guest-mode-details');
  if (!banner || !details) return;
  const prob = data.guest_probability || 0;
  const pct = Math.round(prob * 100);
  if (prob > 0.5 && banner) { banner.style.display = 'flex'; document.getElementById('guest-prob-text').textContent = 'probability ' + pct + '%'; }
  const factors = data.factors || {};
  const factorHtml = Object.entries(factors).map(([k,v])=>`<div style="display:flex;justify-content:space-between;font-size:.83rem;padding:3px 0"><span>${k.replace(/_/g,' ')}</span><span style="color:var(--accent)">${Math.round(v*100)}%</span></div>`).join('');
  details.innerHTML = `<div style="margin-bottom:12px"><div style="font-size:1.2rem;font-weight:600;margin-bottom:4px">Guest Probability: ${pct}%</div><div style="background:var(--border);height:8px;border-radius:4px"><div style="background:var(--accent);height:100%;width:${pct}%;border-radius:4px;transition:.3s"></div></div></div>${factorHtml||'<div style="color:var(--text3)">Normal activity patterns detected.</div>'}`;
}

async function activateGuestMode() {
  const data = await fetch('api/guest_mode').then(r=>r.json()).catch(()=>({}));
  const suggestions = data.suggestions || [];
  if (!suggestions.length) { toast('No guest mode suggestions available'); return; }
  toast('Guest Mode suggestions ready — see seasonal/suggestions section');
}

async function loadSeasonalSuggestions() {
  const data = await fetch('api/seasonal_suggestions').then(r=>r.json()).catch(()=>({suggestions:[]}));
  const el = document.getElementById('seasonal-list');
  const badge = document.getElementById('season-badge');
  if (!el) return;
  if (badge) badge.textContent = (data.season||'').toUpperCase();
  const sugs = data.suggestions || [];
  if (!sugs.length) { el.innerHTML = '<div style="color:var(--text3);padding:12px">No seasonal suggestions yet.</div>'; return; }
  el.innerHTML = sugs.map(s => `
    <div style="padding:10px 0;border-bottom:1px solid var(--border)">
      <div style="font-weight:600;margin-bottom:2px">${s.title}</div>
      <div style="color:var(--text3);font-size:.82rem;margin-bottom:4px">${s.seasonal_reason}</div>
      <div style="color:var(--text3);font-size:.78rem">Confidence: ${Math.round((s.confidence||0)*100)}%</div>
    </div>`).join('');
}

async function loadBatteryStatus() {
  const data = await fetch('api/battery_status').then(r=>r.json()).catch(()=>({batteries:[]}));
  const el = document.getElementById('battery-status-list');
  if (!el) return;
  const batteries = data.batteries || [];
  if (!batteries.length) { el.innerHTML = '<div style="color:var(--text3);padding:12px">No battery sensors found.</div>'; return; }
  const colors = {critical:'#f44336',low:'#ff9800',ok:'#4caf50'};
  el.innerHTML = batteries.slice(0,15).map(b => `
    <div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border)">
      <div style="flex:1">
        <div style="font-size:.88rem">${b.friendly_name||b.entity_id}</div>
        <div style="color:var(--text3);font-size:.75rem">${b.area||''} ${b.estimated_days_remaining?'· ~'+b.estimated_days_remaining+'d remaining':''}</div>
      </div>
      <div style="width:80px">
        <div style="background:var(--border);height:6px;border-radius:3px">
          <div style="background:${colors[b.alert]};height:100%;width:${b.level}%;border-radius:3px"></div>
        </div>
        <div style="text-align:right;font-size:.75rem;margin-top:2px;color:${colors[b.alert]}">${b.level}%</div>
      </div>
      <span style="background:${colors[b.alert]};color:#fff;border-radius:8px;padding:2px 8px;font-size:.72rem">${b.alert}</span>
    </div>`).join('');
}

async function loadIntegrationHealth() {
  const data = await fetch('api/integration_health').then(r=>r.json()).catch(()=>({overall_score:0}));
  const el = document.getElementById('integration-health-list');
  const badge = document.getElementById('health-score-badge');
  if (!el) return;
  const score = data.overall_score || 0;
  if (badge) badge.textContent = 'Overall: ' + score + '%';
  const scores = Object.values(data.integration_scores||{}).sort((a,b)=>a.score-b.score);
  const stale = (data.stale_entities||[]).slice(0,10);
  const unavail = (data.unavailable_entities||[]).slice(0,10);
  el.innerHTML = `
    <div style="margin-bottom:12px">
      <div style="font-weight:600;margin-bottom:8px">Domain Scores</div>
      ${scores.slice(0,8).map(s=>`<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px"><span style="width:120px;font-size:.82rem">${s.domain}</span><div style="flex:1;background:var(--border);height:6px;border-radius:3px"><div style="background:${s.score>90?'#4caf50':s.score>70?'#ff9800':'#f44336'};height:100%;width:${s.score}%;border-radius:3px"></div></div><span style="font-size:.78rem;color:var(--text3)">${s.score}%</span></div>`).join('')}
    </div>
    ${stale.length?`<div><div style="font-weight:600;margin-bottom:6px">⚠️ Stale Entities (${data.stale_count})</div>${stale.map(e=>`<div style="font-size:.82rem;padding:3px 0;color:var(--text2)">${e.entity_id} <span style="color:var(--text3)">(${e.age_hours?e.age_hours+'h ago':'unknown'})</span></div>`).join('')}</div>`:''}
    ${unavail.length?`<div style="margin-top:8px"><div style="font-weight:600;margin-bottom:6px">❌ Unavailable (${data.unavailable_count})</div>${unavail.map(e=>`<div style="font-size:.82rem;padding:3px 0;color:var(--text2)">${e.entity_id}</div>`).join('')}</div>`:''}`;
}

async function loadChangelog() {
  const data = await fetch('api/changelog?limit=15').then(r=>r.json()).catch(()=>({entries:[]}));
  const el = document.getElementById('changelog-list');
  if (!el) return;
  const entries = data.entries || [];
  if (!entries.length) { el.innerHTML = '<div style="color:var(--text3);padding:12px">No activity logged yet.</div>'; return; }
  el.innerHTML = entries.map(e => {
    const ts = e.timestamp ? new Date(e.timestamp).toLocaleString() : '';
    return `<div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--border)">
      <span style="font-size:1.1rem">${e.icon||'📝'}</span>
      <div style="flex:1"><div style="font-size:.87rem">${e.description||e.alias||''}</div><div style="color:var(--text3);font-size:.75rem">${ts}</div></div>
    </div>`;
  }).join('');
}

let _nlDebounce = null;
async function nlPreview() {
  clearTimeout(_nlDebounce);
  _nlDebounce = setTimeout(async () => {
    const text = document.getElementById('nl-input')?.value || '';
    if (!text.trim()) { document.getElementById('nl-preview').style.display='none'; return; }
    const data = await fetch('api/nl_automation', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})}).then(r=>r.json()).catch(()=>({}));
    const preview = document.getElementById('nl-preview');
    if (!preview) return;
    preview.style.display = 'block';
    document.getElementById('nl-confidence').textContent = 'Confidence: ' + Math.round((data.confidence||0)*100) + '%';
    const clarifications = data.clarification_needed || [];
    document.getElementById('nl-clarifications').textContent = clarifications.join(' | ');
    document.getElementById('nl-yaml').textContent = data.generated_yaml || '';
    document.getElementById('nl-add-btn').style.display = data.generated_yaml ? '' : 'none';
    window._nlYaml = data.generated_yaml;
  }, 300);
}

async function nlAddToHA() {
  const yaml = window._nlYaml;
  if (!yaml) { toast('No automation to add', 'err'); return; }
  const data = await fetch('api/automations/add', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yaml})}).then(r=>r.json()).catch(()=>({}));
  if (data.status==='success'||data.ok) toast('Automation added!');
  else toast('Error: '+(data.error||'Could not add'), 'err');
}

async function generateDashboard() {
  const el = document.getElementById('dashboard-preview');
  if (el) el.innerHTML = '<div style="color:var(--text3)">Generating...</div>';
  const data = await fetch('api/dashboard_yaml').then(r=>r.json()).catch(()=>({yaml:''}));
  if (!el) return;
  const yaml = data.yaml || '';
  window._dashboardYaml = yaml;
  document.getElementById('copy-dashboard-btn').style.display = yaml ? '' : 'none';
  document.getElementById('apply-dashboard-btn').style.display = yaml ? '' : 'none';
  el.innerHTML = yaml ? `<pre style="background:var(--card2);border-radius:8px;padding:12px;font-size:.78rem;overflow-x:auto;max-height:400px;overflow-y:auto;white-space:pre-wrap">${yaml.replace(/</g,'&lt;')}</pre>` : '<div style="color:var(--text3)">Failed to generate dashboard.</div>';
}

function copyDashboardYaml() {
  navigator.clipboard.writeText(window._dashboardYaml||'').then(()=>toast('Dashboard YAML copied!'));
}

async function applyDashboard() {
  if (!confirm('Apply generated dashboard to HA? This will replace your existing dashboard.')) return;
  const r = await fetch('api/lovelace/config', {method:'POST'}).then(r=>r.json()).catch(()=>({}));
  if (r.success) toast('Dashboard applied to HA!');
  else toast('Error: '+(r.error||'unknown'), 'err');
}

// Onboarding
let _obStep = 0;
async function checkOnboarding() {
  const status = await fetch('api/onboarding/status').then(r=>r.json()).catch(()=>({complete:true}));
  if (!status.complete) {
    document.getElementById('onboarding-modal').style.display = 'flex';
  }
}

function obNext() { _obStep++; if (_obStep >= 6) obComplete(); }
function obSkip() { fetch('api/onboarding/complete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({skipped:true})}); document.getElementById('onboarding-modal').style.display = 'none'; }
function obComplete() {
  fetch('api/onboarding/complete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({skipped:false})});
  document.getElementById('onboarding-modal').style.display = 'none';
  toast('Setup complete! Running initial train...');
  fullTrain();
}

function addRoutineToHA(yamlEncoded) {
  const yaml = decodeURIComponent(yamlEncoded);
  fetch('api/automations/add', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yaml})})
    .then(r=>r.json()).then(d=>{
      if(d.status==='success'||d.ok) toast('Routine added to HA!');
      else toast('Error: '+(d.error||'could not add'),'err');
    }).catch(()=>toast('Network error','err'));
}

function showYaml(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// Load all new feature sections
async function loadNewFeatures() {
  await Promise.all([
    loadAutomationHealth(),
    loadDetectedRoutines(),
    loadGuestMode(),
    loadSeasonalSuggestions(),
    loadBatteryStatus(),
    loadIntegrationHealth(),
    loadChangelog(),
  ]);
}

load();
loadNewFeatures();
checkOnboarding();
// Fast-poll during training, normal refresh otherwise
let _pollTimer = null;
function schedulePoll() {
  if (_pollTimer) clearTimeout(_pollTimer);
  const interval = (window._isTraining) ? 3000 : 30000;
  _pollTimer = setTimeout(() => { load(); schedulePoll(); }, interval);
}
schedulePoll();

// Theme toggle
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('habitus-theme', next);
  document.getElementById('theme-toggle').textContent = next === 'dark' ? '🌙' : '☀️';
}
(function initTheme() {
  const saved = localStorage.getItem('habitus-theme');
  if (saved === 'light') document.documentElement.setAttribute('data-theme', 'light');
  setTimeout(() => {
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = (saved === 'light') ? '☀️' : '🌙';
  }, 100);
})();

</script>
  <div id="train-banner" class="train-banner">
    <div class="spin"></div>
    <span id="train-banner-text">Training in background…</span>
    <span id="train-banner-pct" style="margin-left:auto;color:var(--accent);font-weight:600"></span>
  </div>
</body>
</html>"""


@app.route("/")
@app.route("/ingress")
@app.route("/ingress/")
def index():
    schedule = os.environ.get("HABITUS_SCHEDULE", "overnight")
    train_time = os.environ.get("HABITUS_TRAIN_TIME", "02:00")
    page = PAGE.replace("'{{ schedule }}'", f"'{schedule}'").replace(
        "'{{ train_time }}'", f"'{train_time}'"
    )
    return render_template_string(page)


@app.route("/api/state")
@app.route("/ingress/api/state")
def api_state():
    import os
    data = _read(STATE_PATH) or {}
    data["version"] = os.environ.get("HABITUS_VERSION", "?")
    return jsonify(data)


@app.route("/api/baseline")
@app.route("/ingress/api/baseline")
def api_baseline():
    return jsonify(_read(BASELINE_PATH) or {})


@app.route("/api/progress")
@app.route("/ingress/api/progress")
def api_progress():
    """Return training progress with stale-run auto-recovery.

    If progress.json says running but the trainer is not actually running,
    or progress has not updated for a long time, mark it idle so the UI
    doesn't stay blocked forever.
    """
    p = _read(PROGRESS_PATH) or {}
    try:
        if p.get("running"):
            recover = False
            reason = None
            age = 0
            trainer_running = _trainer.is_running()

            try:
                done_i = int(p.get("done", 0) or 0)
                total_i = int(p.get("total", 0) or 0)
                pct_i = int(float(p.get("pct", 0) or 0))
            except Exception:
                done_i, total_i, pct_i = 0, 0, 0

            # If fetch is already reported complete but trainer is not running,
            # recover immediately (don't wait for grace timeout).
            if (not trainer_running) and total_i > 0 and done_i >= total_i and pct_i >= 100:
                recover = True
                reason = "completed_no_trainer"

            if os.path.exists(PROGRESS_PATH):
                import time as _t

                stale_sec = int(os.environ.get("HABITUS_PROGRESS_STALE_SEC", "600"))
                dead_grace_sec = int(os.environ.get("HABITUS_PROGRESS_DEAD_GRACE_SEC", "120"))
                age = int(_t.time() - os.path.getmtime(PROGRESS_PATH))

                # Hard timeout always wins.
                if age > stale_sec:
                    recover = True
                    reason = "stale_timeout"
                # If trainer watchdog says not running, only recover after grace period.
                elif (not trainer_running) and age > dead_grace_sec:
                    recover = True
                    reason = "trainer_not_running"

            if recover:
                p["running"] = False
                p["phase"] = "idle"
                p["stale_recovered"] = True
                p["stale_reason"] = reason
                p["stale_age_sec"] = age
                try:
                    with open(PROGRESS_PATH, "w") as f:
                        json.dump(p, f)
                except Exception:
                    pass
    except Exception:
        pass

    # Enrich and normalize to a stable progress contract for the UI.
    st = _read(STATE_PATH) or {}
    return jsonify(_normalize_progress_payload(p, st))


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


@app.route("/api/full_train", methods=["POST"])
@app.route("/ingress/api/full_train", methods=["POST"])
def api_full_train():
    """Trigger a full training run using configured history depth."""

    # Resolve days from user settings first, then env default
    days = int(os.environ.get("HABITUS_DAYS", "365"))
    try:
        st = _read(STATE_PATH) or {}
        us = st.get("user_settings", {}) if isinstance(st, dict) else {}
        cfg_days = int(us.get("days_history", days))
        if 7 <= cfg_days <= 3650:
            days = cfg_days
    except Exception:
        pass

    # Use central trainer manager so running-state and exception logging are consistent.
    if not _trainer.start(days=days, mode="full"):
        return jsonify({"ok": False, "error": "Training already running"}), 409

    return jsonify({"ok": True, "message": f"Full {days}d training started"})


@app.route("/api/rescan", methods=["POST"])
@app.route("/ingress/api/rescan", methods=["POST"])
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

        p = _read(PROGRESS_PATH) or {}
        if p.get("running"):
            stale_sec = int(os.environ.get("HABITUS_PROGRESS_STALE_SEC", "600"))
            age = 0
            if os.path.exists(PROGRESS_PATH):
                import time as _t

                age = int(_t.time() - os.path.getmtime(PROGRESS_PATH))
            if age <= stale_sec:
                return jsonify({"ok": False, "error": "Training already running"}), 409

            # Stale progress file from a dead run; clear it so rescan can proceed.
            p["running"] = False
            p["phase"] = "idle"
            p["stale_recovered"] = True
            p["stale_reason"] = "stale_timeout"
            p["stale_age_sec"] = age
            try:
                with open(PROGRESS_PATH, "w") as f:
                    json.dump(p, f)
            except Exception:
                pass
        if _prog.is_expanding():
            return jsonify({"ok": False, "error": "Progressive training already running"}), 409
        _prog.start_progressive(max_days=days)
        return jsonify({"ok": True, "started": True, "progressive": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/training_status")
@app.route("/ingress/api/training_status")
def api_training_status():
    state = _read(STATE_PATH) or {}
    return jsonify({
        "running": _trainer.is_running(),
        "ha_reachable": state.get("ha_reachable", None),
    })


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
        if "days_history" in data:
            try:
                days = int(data["days_history"])
                if 7 <= days <= 3650:
                    settings["days_history"] = days
                    os.environ["HABITUS_DAYS"] = str(days)
            except (ValueError, TypeError):
                pass
        if "anonymous_sharing" in data:
            settings["anonymous_sharing"] = bool(data["anonymous_sharing"])
        if "notify_on_anomaly" in data:
            notify_on = bool(data["notify_on_anomaly"])
            settings["notify_on_anomaly"] = notify_on
            os.environ["HABITUS_NOTIFY_ON"] = "true" if notify_on else "false"
        state["user_settings"] = settings
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, default=str)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})
        return jsonify({"ok": True, "settings": settings})

    if "notify_on_anomaly" not in settings:
        settings["notify_on_anomaly"] = os.environ.get("HABITUS_NOTIFY_ON", "true").lower() == "true"
    return jsonify({"settings": settings})


@app.route("/api/add_automation", methods=["POST"])
@app.route("/ingress/api/add_automation", methods=["POST"])
def api_add_automation():
    import requests as req

    data = request.get_json() or {}
    yaml_str = (data.get("yaml", "") or "").strip()
    if not yaml_str:
        return jsonify({"ok": False, "error": "missing yaml payload"}), 400

    auto, err = _extract_automation_from_yaml(yaml_str)
    if err or not auto:
        return jsonify({"ok": False, "error": err or "invalid automation YAML"}), 400

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    existing = _existing_automation_ids(ha_url, token)
    alias_id = _unique_alias_id(str(auto.get("alias", "habitus_automation")), existing)

    try:
        r = req.post(
            f"{ha_url}/api/config/automation/config/{alias_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=auto,
            timeout=10,
        )
        if r.status_code in (200, 201, 204):
            return jsonify({"ok": True, "automation_id": alias_id, "alias": auto.get("alias", "")})

        detail = ""
        try:
            detail = (r.text or "")[:240]
        except Exception:
            detail = ""
        msg = f"HA {r.status_code}"
        if detail:
            msg += f": {detail}"
        return jsonify({"ok": False, "error": msg, "automation_id": alias_id}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to add automation: {e}"}), 500


@app.route("/api/remove_automation", methods=["POST"])
@app.route("/ingress/api/remove_automation", methods=["POST"])
def api_remove_automation():
    import requests as req

    data = request.get_json() or {}
    entity_id = (data.get("entity_id") or "").strip()
    alias = (data.get("alias") or "").strip()

    automation_id = _normalize_automation_id(entity_id or alias)
    if not automation_id:
        return jsonify({"ok": False, "error": "missing entity_id/alias"}), 400

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", "")

    try:
        r = req.delete(
            f"{ha_url}/api/config/automation/config/{automation_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code in (200, 201, 204):
            return jsonify({"ok": True, "automation_id": automation_id})
        if r.status_code == 404:
            return jsonify({"ok": False, "error": f"automation not found: {automation_id}"}), 404
        detail = ""
        try:
            detail = (r.text or "")[:240]
        except Exception:
            detail = ""
        msg = f"HA {r.status_code}"
        if detail:
            msg += f": {detail}"
        return jsonify({"ok": False, "error": msg}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to remove automation: {e}"}), 500


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


@app.route("/api/scenes")
@app.route("/ingress/api/scenes")
def api_scenes():
    """Return discovered implicit scenes."""
    return jsonify(_read(SCENES_PATH) or {"scenes": [], "count": 0})


@app.route("/api/scene_analysis")
@app.route("/ingress/api/scene_analysis")
def api_scene_analysis():
    """Return scene improvement suggestions (cached; regenerated on full train)."""
    force = request.args.get("force", "").lower() in ("1", "true", "yes")
    cached = _read(SCENE_ANALYSIS_PATH)
    if cached is not None and not force:
        return jsonify({"suggestions": cached, "count": len(cached)})
    try:
        from . import scene_analysis as _sa
        results = _sa.run_scene_analysis(force=True)
        return jsonify({"suggestions": results, "count": len(results)})
    except Exception as e:
        log.warning("api_scene_analysis failed: %s", e)
        return jsonify({"suggestions": [], "count": 0, "error": str(e)})


@app.route("/api/ha_automations")
@app.route("/ingress/api/ha_automations")
def api_ha_automations():
    """Return cached HA automations."""
    return jsonify(_read(HA_AUTOMATIONS_PATH) or {"automations": [], "count": 0})


@app.route("/api/smart_suggestions")
@app.route("/ingress/api/smart_suggestions")
def api_smart_suggestions():
    """Return merged smart suggestions with confidence, YAML, and overlap info.

    Applies feedback adjustments (boost/suppress) before returning.
    """
    data = _read(SMART_SUGGESTIONS_PATH) or {"suggestions": [], "count": 0}
    suggestions = data.get("suggestions", [])
    if suggestions:
        try:
            from . import suggestion_feedback as _sf
            suggestions = _sf.apply_feedback_to_suggestions(list(suggestions))
            data["suggestions"] = suggestions
            data["count"] = len(suggestions)
        except Exception as e:
            log.warning("suggestion_feedback apply failed: %s", e)
    return jsonify(data)


@app.route("/api/energy_weather_history")
@app.route("/ingress/api/energy_weather_history")
def api_energy_weather_history():
    from .energy_forecast import get_energy_weather_history
    days = request.args.get("days", 90, type=int)
    return jsonify({"days": get_energy_weather_history(days=days)})

@app.route("/api/nilm")
@app.route("/ingress/api/nilm")
def api_nilm():
    return jsonify(_read(os.path.join(DATA_DIR, "nilm_disaggregation.json")) or {"breakdown": [], "discovered_appliances": []})

@app.route("/api/nilm/run", methods=["POST"])
@app.route("/ingress/api/nilm/run", methods=["POST"])
def api_nilm_run():
    from .nilm_disaggregator import run_disaggregation
    data = request.get_json() or {}
    result = run_disaggregation(
        power_entity=data.get("power_entity", ""),
        days=data.get("days", 7),
    )
    return jsonify(result)

@app.route("/api/feedback", methods=["POST"])
@app.route("/ingress/api/feedback", methods=["POST"])
def api_feedback():
    """Record user feedback on a suggestion (add / dismiss / remove)."""
    from . import suggestion_feedback as _sf
    data = request.get_json() or {}
    suggestion_id = (data.get("suggestion_id") or "").strip()
    action = (data.get("action") or "").strip()
    if not suggestion_id:
        return jsonify({"ok": False, "error": "suggestion_id is required"}), 400
    try:
        entry = _sf.record_feedback(suggestion_id, action)
        summary = _sf.get_feedback_summary()
        return jsonify({"ok": True, "entry": entry, "summary": summary})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"feedback error: {e}"}), 500


@app.route("/api/anomaly_feedback", methods=["POST"])
@app.route("/ingress/api/anomaly_feedback", methods=["POST"])
def api_anomaly_feedback():
    """Record user feedback on an anomaly."""
    from .feedback import record_feedback, get_feedback_stats
    data = request.get_json() or {}
    action = data.get("action", "dismissed")
    entry = record_feedback(
        anomaly_id=data.get("anomaly_id", ""),
        action=action,
        entity_id=data.get("entity_id", ""),
        score=data.get("score", 0),
        details=data.get("details", ""),
    )
    return jsonify({"ok": True, "entry": entry, "stats": get_feedback_stats()["stats"]})

@app.route("/api/feedback_stats")
@app.route("/ingress/api/feedback_stats")
def api_feedback_stats():
    from .feedback import get_feedback_stats
    return jsonify(get_feedback_stats())

@app.route("/api/anonymous_export")
@app.route("/ingress/api/anonymous_export")
def api_anonymous_export():
    from .feedback import get_anonymous_export
    data = get_anonymous_export()
    if data is None:
        return jsonify({"error": "Anonymous sharing is disabled"}), 403
    return jsonify(data)

@app.route("/api/sharing", methods=["POST"])
@app.route("/ingress/api/sharing", methods=["POST"])
def api_sharing():
    from .feedback import set_sharing
    data = request.get_json() or {}
    set_sharing(data.get("enabled", False))
    return jsonify({"ok": True})

@app.route("/api/training/start", methods=["POST"])
@app.route("/ingress/api/training/start", methods=["POST"])
def api_training_start():
    from .device_trainer import start_training_session
    data = request.get_json() or {}
    entity = data.get("power_entity", "")
    if not entity:
        return jsonify({"error": "power_entity required"}), 400
    return jsonify(start_training_session(entity))

@app.route("/api/training/stop", methods=["POST"])
@app.route("/ingress/api/training/stop", methods=["POST"])
def api_training_stop():
    from .device_trainer import stop_training_session
    data = request.get_json() or {}
    name = data.get("device_name", "Unknown Device")
    category = data.get("category", "custom")
    return jsonify(stop_training_session(name, category))

@app.route("/api/device_training/status")
@app.route("/ingress/api/device_training/status")
def api_device_training_status():
    from .device_trainer import get_training_status
    return jsonify(get_training_status())

@app.route("/api/custom_signatures")
@app.route("/ingress/api/custom_signatures")
def api_custom_signatures():
    from .device_trainer import list_custom_signatures
    return jsonify({"signatures": list_custom_signatures()})

@app.route("/api/custom_signatures/delete", methods=["POST"])
@app.route("/ingress/api/custom_signatures/delete", methods=["POST"])
def api_delete_signature():
    from .device_trainer import delete_signature
    data = request.get_json() or {}
    ok = delete_signature(data.get("name", ""))
    return jsonify({"ok": ok})

@app.route("/api/sequences")
@app.route("/ingress/api/sequences")
def api_sequences():
    return jsonify(_read(os.path.join(DATA_DIR, "sequences.json")) or {"sequences": []})

@app.route("/api/markov")
@app.route("/ingress/api/markov")
def api_markov():
    return jsonify(_read(os.path.join(DATA_DIR, "markov_model.json")) or {"predictions": []})

@app.route("/api/activity_states")
@app.route("/ingress/api/activity_states")
def api_activity_states():
    return jsonify(_read(os.path.join(DATA_DIR, "activity_states.json")) or {"states": []})

@app.route("/api/energy_forecast")
@app.route("/ingress/api/energy_forecast")
def api_energy_forecast():
    return jsonify(_read(os.path.join(DATA_DIR, "energy_forecast.json")) or {"forecast": []})

@app.route("/api/dynamic")
@app.route("/ingress/api/dynamic")
def api_dynamic():
    return jsonify(_read(os.path.join(DATA_DIR, "dynamic_automations.json")) or {"drifts": []})

@app.route("/api/correlations")
@app.route("/ingress/api/correlations")
def api_correlations():
    """Return deep correlation analysis results."""
    corr_path = os.path.join(DATA_DIR, "correlations.json")
    return jsonify(_read(corr_path) or {"correlations": [], "suggestions": [], "total_correlations": 0})


@app.route("/api/room_predictions")
@app.route("/ingress/api/room_predictions")
def api_room_predictions():
    """Return room-based predictive automations."""
    rp_path = os.path.join(DATA_DIR, "room_predictions.json")
    return jsonify(_read(rp_path) or {"automations": [], "prediction_count": 0})


@app.route("/api/routines")
@app.route("/ingress/api/routines")
def api_routines():
    """Return detected routines (shower times, cooking patterns, etc)."""
    r_path = os.path.join(DATA_DIR, "routines.json")
    return jsonify(_read(r_path) or {"routines": [], "total_events": 0})


@app.route("/api/conflicts")
@app.route("/ingress/api/conflicts")
def api_conflicts():
    """Return detected cross-domain conflicts (window+heating, nobody home, etc)."""
    c_path = os.path.join(DATA_DIR, "conflicts.json")
    return jsonify(_read(c_path) or {"conflicts": [], "count": 0, "total_est_waste_w": 0})


@app.route("/api/appliance_fingerprints")
@app.route("/ingress/api/appliance_fingerprints")
def api_appliance_fingerprints():
    """Return detected appliance power signatures."""
    fp_path = os.path.join(DATA_DIR, "appliance_fingerprints.json")
    return jsonify(_read(fp_path) or {"appliances": [], "recent_events": [], "total_events": 0})


@app.route("/api/entities")
@app.route("/ingress/api/entities")
def api_entities():
    """Return all HA entities for the entity picker widget."""
    import requests as req  # type: ignore[import-untyped]

    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    try:
        r = req.get(f"{ha_url}/api/states", headers={"Authorization": f"Bearer {token}"}, timeout=10)
        if r.status_code != 200:
            return jsonify([])
        entities = []
        for s in r.json():
            eid = s["entity_id"]
            domain = eid.split(".")[0]
            if domain in ("light", "switch", "media_player", "climate", "cover",
                          "fan", "scene", "automation", "input_boolean", "script"):
                entities.append({
                    "entity_id": eid,
                    "name": s["attributes"].get("friendly_name", eid),
                    "state": s["state"],
                    "domain": domain,
                })
        entities.sort(key=lambda x: x["entity_id"])
        return jsonify(entities)
    except Exception as e:
        return jsonify([])


@app.route("/api/automation_health")
@app.route("/ingress/api/automation_health")
def api_automation_health():
    """Return automation health report (dead/stale/over-triggering)."""
    from . import automation_health
    return jsonify(automation_health.load_health())


@app.route("/api/automation_health/disable", methods=["POST"])
@app.route("/ingress/api/automation_health/disable", methods=["POST"])
def api_disable_automation():
    """Disable a HA automation by entity_id."""
    import requests as req  # type: ignore[import-untyped]
    data = request.get_json(silent=True) or {}
    entity_id = data.get("entity_id", "")
    if not entity_id:
        return jsonify({"error": "entity_id required"}), 400
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    try:
        r = req.post(
            f"{ha_url}/api/services/automation/turn_off",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"entity_id": entity_id},
            timeout=10,
        )
        return jsonify({"success": r.status_code in (200, 201, 204)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/routine_builder")
@app.route("/ingress/api/routine_builder")
def api_routine_builder():
    """Return detected routines from routine_builder."""
    from . import routine_builder
    return jsonify(routine_builder.load_routines())


@app.route("/api/guest_mode")
@app.route("/ingress/api/guest_mode")
def api_guest_mode():
    """Return guest mode analysis."""
    from . import guest_mode
    return jsonify(guest_mode.load_guest_mode())


@app.route("/api/seasonal_suggestions")
@app.route("/ingress/api/seasonal_suggestions")
def api_seasonal_suggestions():
    """Return seasonal automation suggestions."""
    from . import seasonal_adapter
    return jsonify(seasonal_adapter.load_seasonal_suggestions())


@app.route("/api/battery_status")
@app.route("/ingress/api/battery_status")
def api_battery_status():
    """Return battery watchdog status."""
    from . import battery_watchdog
    return jsonify(battery_watchdog.load_battery_status())


@app.route("/api/integration_health")
@app.route("/ingress/api/integration_health")
def api_integration_health():
    """Return integration health report."""
    from . import integration_health
    return jsonify(integration_health.load_integration_health())


@app.route("/api/nl_automation", methods=["POST"])
@app.route("/ingress/api/nl_automation", methods=["POST"])
def api_nl_automation():
    """Parse natural language automation description and return intent + YAML."""
    from . import nl_automation
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text required"}), 400
    intent = nl_automation.parse_intent(text)
    return jsonify(intent)


@app.route("/api/dashboard_yaml")
@app.route("/ingress/api/dashboard_yaml")
def api_dashboard_yaml():
    """Return generated Lovelace dashboard YAML."""
    from . import dashboard_generator
    yaml_str = dashboard_generator.load_dashboard_yaml()
    if not yaml_str:
        result = dashboard_generator.run()
        yaml_str = result.get("yaml", "")
    return jsonify({"yaml": yaml_str})


@app.route("/api/lovelace/config", methods=["POST"])
@app.route("/ingress/api/lovelace/config", methods=["POST"])
def api_apply_lovelace():
    """Apply generated dashboard YAML to HA (POST to Lovelace config endpoint)."""
    import requests as req  # type: ignore[import-untyped]
    import yaml as _yaml
    from . import dashboard_generator
    yaml_str = dashboard_generator.load_dashboard_yaml()
    if not yaml_str:
        return jsonify({"error": "No dashboard YAML generated yet"}), 400
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    try:
        config = _yaml.safe_load(yaml_str)
        r = req.post(
            f"{ha_url}/api/lovelace/config",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=config,
            timeout=15,
        )
        return jsonify({"success": r.status_code in (200, 201)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/changelog")
@app.route("/ingress/api/changelog")
def api_changelog():
    """Return automation changelog entries."""
    from . import changelog
    limit = request.args.get("limit", type=int)
    entries = changelog.load_changelog(limit=limit)
    return jsonify({"entries": entries, "total": len(entries)})


@app.route("/api/onboarding/status")
@app.route("/ingress/api/onboarding/status")
def api_onboarding_status():
    """Return onboarding status."""
    from . import onboarding
    return jsonify(onboarding.get_status())


@app.route("/api/onboarding/complete", methods=["POST"])
@app.route("/ingress/api/onboarding/complete", methods=["POST"])
def api_onboarding_complete():
    """Mark onboarding as complete."""
    from . import onboarding
    data = request.get_json(silent=True) or {}
    record = onboarding.complete_onboarding(
        tariff=data.get("tariff"),
        tariff_peak=data.get("tariff_peak"),
        tariff_offpeak=data.get("tariff_offpeak"),
        notification_prefs=data.get("notification_prefs"),
        skipped=data.get("skipped", False),
    )
    return jsonify({"success": True, "record": record})


def start_web(port=8099):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
