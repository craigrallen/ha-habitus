"""Habitus v2.1 — polished web UI."""

import json
import os

import yaml as _yaml
from flask import Flask, jsonify, render_template_string, request

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

app = Flask(__name__)


def _read(path, default=None):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


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
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>Habitus <span class="version">v2.20.0</span></h1>
  </div>
  <div class="header-right">
    <div class="status-dot" id="hdr-dot"></div>
    <span class="status-label" id="hdr-label">Loading</span>
    <span class="last-run" id="hdr-lr"></span>
  </div>
</div>

<nav>
  <button class="active" onclick="gotoTab('overview',this)">Overview</button>
  <button onclick="gotoTab('breakdown',this)">Anomaly Breakdown</button>
  <button onclick="gotoTab('automations',this)">Automations</button>
  <button onclick="gotoTab('energy',this)">Energy & Patterns</button>
  <button onclick="gotoTab('settings',this)">Settings</button>
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
    <table>
      <thead><tr><th>Time</th><th>Expected</th><th>±Variance</th><th style="width:120px"></th></tr></thead>
      <tbody id="bl-table"><tr><td colspan="4" style="color:var(--text3);padding:16px">Loading baselines...</td></tr></tbody>
    </table>
  </div>
</div>

<!-- ANOMALY BREAKDOWN -->
<div id="tab-breakdown" class="tab">
  <div class="sec">
    <div class="sec-header">
      <h2>Per-Entity Anomaly Scores</h2>
      <span class="sec-sub" id="bd-ts"></span>
    </div>
    <table>
      <thead><tr><th>Sensor</th><th>Current</th><th>Baseline</th><th>Deviation</th><th style="width:90px"></th></tr></thead>
      <tbody id="bd-table"><tr><td colspan="5" style="color:var(--text3);padding:16px">No entity data yet.</td></tr></tbody>
    </table>
  </div>
</div>

<!-- AUTOMATIONS -->
<div id="tab-automations" class="tab">
  <div class="ftabs">
    <button class="ftab active" onclick="filterSug('all',this)">All</button>
    <button class="ftab" onclick="filterSug('routine',this)">🔁 Routine</button>
    <button class="ftab" onclick="filterSug('energy',this)">⚡ Energy</button>
    <button class="ftab" onclick="filterSug('boat',this)">⛵ Boat</button>
    <button class="ftab" onclick="filterSug('anomaly',this)">🧠 Anomaly</button>
    <button class="ftab" onclick="filterSug('lovelace',this)">🃏 Lovelace</button>
  </div>
  <div id="sug-list"><div style="color:var(--text3);padding:12px">Loading suggestions...</div></div>
</div>

<!-- ENERGY & PATTERNS -->
<div id="tab-energy" class="tab">
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
    <pre class="raw" id="raw-state">Loading...</pre>
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

<!-- PROGRESS OVERLAY -->
<div id="prog-overlay" class="prog-overlay" style="display:none">
  <div class="prog-box">
    <div class="prog-icon" id="prog-icon">📡</div>
    <div class="prog-title" id="prog-title">Training in progress</div>
    <div class="prog-desc" id="prog-desc">Fetching sensor history from Home Assistant</div>
    <div class="prog-bar-wrap"><div class="prog-bar" id="prog-bar" style="width:0%"></div></div>
    <div class="prog-meta" id="prog-meta"></div>
    <div class="prog-steps">
      <div class="prog-step" id="ps-fetching">📡 Fetch</div>
      <div class="prog-step" id="ps-building_baselines">🗄️ Baselines</div>
      <div class="prog-step" id="ps-training">🧠 Train</div>
      <div class="prog-step" id="ps-seasonal_training">🌱 Seasonal</div>
      <div class="prog-step" id="ps-pattern_analysis">🔍 Patterns</div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let allSuggestions = [];
let currentFilter = 'all';
const phaseOrder = ['fetching','building_baselines','training','seasonal_training','pattern_analysis'];

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

function renderSuggestions() {
  const list = currentFilter === 'all' ? allSuggestions : allSuggestions.filter(s=>s.category===currentFilter);
  if (!list.length) {
    document.getElementById('sug-list').innerHTML = '<div style="color:var(--text3);padding:16px">No suggestions in this category.</div>';
    return;
  }
  const catBadge = {routine:['b-info','🔁'],energy:['b-warn','⚡'],boat:['b-boat','⛵'],anomaly:['b-alert','🧠'],lovelace:['b-purple','🃏']};
  document.getElementById('sug-list').innerHTML = list.map(s => {
    const [bc, ic] = catBadge[s.category] || ['b-muted','•'];
    return `
    <div class="sug ${s.applicable===false?'na':''}">
      <div class="sug-head">
        <h3>${s.title}</h3>
        <span class="badge ${bc}">${ic} ${s.category}</span>
      </div>
      <div class="sug-meta">
        <div class="conf-bar"><div class="conf-dots">${confDots(s.confidence)}</div><span style="font-size:.72rem;color:var(--text3)">${s.confidence}% confidence</span></div>
        ${s.applicable===false?'<span class="badge b-alert">⚠ Entity not in your system</span>':''}
      </div>
      <div class="desc">${s.description}</div>
      <pre id="yaml-${s.id}">${(s.yaml||'').trim()}</pre>
      <div style="display:flex;gap:8px">
        <button class="btn btn-accent" onclick="copyYaml('${s.id}')">📋 Copy YAML</button>
        ${s.category!=='lovelace'?`<button class="btn btn-success" id="add-${s.id}" onclick="addToHA('${s.id}')">+ Add to HA</button>`:''}
      </div>
    </div>`;
  }).join('');
}

function copyYaml(id) {
  navigator.clipboard.writeText(document.getElementById('yaml-'+id).textContent)
    .then(()=>toast('Copied to clipboard ✓'));
}

async function addToHA(id) {
  const btn = document.getElementById('add-'+id);
  const yaml = document.getElementById('yaml-'+id).textContent;
  btn.disabled = true; btn.textContent = 'Adding…';
  try {
    const r = await fetch('api/add_automation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yaml})});
    const d = await r.json();
    if (d.ok) { btn.textContent = '✓ Added'; toast('Automation added ✓'); }
    else { btn.disabled=false; btn.textContent='+ Add to HA'; toast('Failed: '+(d.error||'?'),'te'); }
  } catch(e) { btn.disabled=false; btn.textContent='+ Add to HA'; toast('Network error','te'); }
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

async function load() {
  const [state, baseline, progress, patterns, suggestions, anomalies] = await Promise.all([
    fetch('api/state').then(r=>r.json()).catch(()=>({})),
    fetch('api/baseline').then(r=>r.json()).catch(()=>({})),
    fetch('api/progress').then(r=>r.json()).catch(()=>({})),
    fetch('api/patterns').then(r=>r.json()).catch(()=>({})),
    fetch('api/suggestions').then(r=>r.json()).catch(()=>([])),
    fetch('api/anomalies').then(r=>r.json()).catch(()=>({})),
  ]);

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
    document.getElementById('hdr-label').textContent = `Training ${progress.pct||0}%${winLabel}`;
    if (!hasData) return;  // only block if no data yet
  }
  document.getElementById('prog-overlay').style.display = 'none';

  // Header
  const score = state.anomaly_score ?? 0;
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
  document.getElementById('tdays').innerHTML = td+'<span class="sunit">days</span>';
  const df=state.data_from?new Date(state.data_from).toLocaleDateString():'?';
  const dt=state.data_to?new Date(state.data_to).toLocaleDateString():'?';
  document.getElementById('tdays-range').textContent = td!=='—' ? `${df} → ${dt}` : '';

  document.getElementById('ecount').innerHTML = (state.entity_count??'—')+'<span class="sunit">sensors</span>';

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
      <div class="pat-item"><div class="pi-label">Night Base</div><div class="pi-val">${r.night_baseline_watts}W</div></div>
    </div>` : '<div style="color:var(--text3);font-size:.82rem">No patterns yet — run Habitus first.</div>';

  // Top anomalies
  const ents = (anomalies.anomalies||[]).slice(0,5);
  document.getElementById('top-anomalies').innerHTML = ents.length
    ? ents.map(e => `
      <div class="anom-item">
        <div>
          <div class="anom-name">${e.name}</div>
          <div class="anom-sub">${e.current_value}${e.unit} vs ${e.baseline_mean}${e.unit} expected</div>
        </div>
        <div class="anom-score" style="color:${e.z_score>=3?'var(--red)':e.z_score>=1.5?'var(--amber)':'var(--green)'}">${e.z_score}σ</div>
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
    return`<tr${cur}><td><span style="font-weight:${h===nowH?700:400}">${String(h).padStart(2,'0')}:00</span> <span style="color:var(--text3);font-size:.7rem">${pd}</span>${h===nowH?' <span class="badge b-info" style="padding:1px 6px;font-size:.66rem">now</span>':''}</td><td>${mean}W</td><td style="color:var(--text3)">±${std}W</td><td><div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div></td></tr>`;
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
        return`<tr>
          <td><div style="font-weight:500">${e.name}</div><div style="font-size:.7rem;color:var(--text3)">${e.entity_id}</div></td>
          <td style="font-weight:600">${e.current_value}${e.unit}</td>
          <td style="color:var(--text3)">${e.baseline_mean}${e.unit} <span style="color:var(--text3);font-size:.7rem">±${e.baseline_std}${e.unit}</span></td>
          <td><span class="badge ${bc}">${z}σ</span></td>
          <td><div class="bar-wrap"><div class="bar" style="width:${pct}%;background:${col}"></div></div></td>
        </tr>`;}).join('')
    : '<tr><td colspan="5" style="color:var(--text3);padding:16px">No entity anomalies detected.</td></tr>';

  // Suggestions
  allSuggestions = suggestions;
  renderSuggestions();

  // Weekly
  const wk=patterns.weekly||{};
  const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const mxW=Math.max(...Object.values(wk).map(v=>v.mean_power_w),1);
  document.getElementById('wk-table').innerHTML=days.map(d=>{
    const v=wk[d]||{mean_power_w:0};
    const p=Math.round(v.mean_power_w/mxW*100);
    const we=d==='Sat'||d==='Sun';
    return`<tr><td>${d}${we?` <span style="color:var(--text3);font-size:.7rem">weekend</span>`:''}</td><td style="font-weight:500">${Math.round(v.mean_power_w)}W</td><td><div class="bar-wrap"><div class="bar" style="width:${p}%;background:${we?'var(--purple)':'var(--accent)'}"></div></div></td></tr>`;
  }).join('');

  // Monthly
  const se=patterns.seasonal||{};
  const mos=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const mxM=Math.max(...Object.values(se).map(v=>v.mean_power_w),1);
  document.getElementById('mo-table').innerHTML=mos.filter(m=>se[m]).map(m=>{
    const v=se[m],p=Math.round(v.mean_power_w/mxM*100);
    return`<tr><td>${m}</td><td style="font-weight:500">${Math.round(v.mean_power_w)}W</td><td style="color:var(--text3)">${v.mean_temp_c}°C</td><td><div class="bar-wrap"><div class="bar" style="width:${p}%"></div></div></td></tr>`;
  }).join('');

  // Season cards
  const seasons=[['winter','❄️',' Winter'],['spring','🌱',' Spring'],['summer','☀️',' Summer'],['autumn','🍂',' Autumn']];
  document.getElementById('season-cards').innerHTML=seasons.map(([s,ic,nm])=>`
    <div class="season-card">
      <div class="s-icon">${ic}</div>
      <div class="s-name">${nm}</div>
      ${sm[s]?'<span class="badge b-ok">Trained</span>':'<span class="badge b-muted">No data</span>'}
    </div>`).join('');

  // Settings
  document.getElementById('raw-state').textContent = JSON.stringify(state,null,2);

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

load();
// Fast-poll during training, normal refresh otherwise
let _pollTimer = null;
function schedulePoll() {
  if (_pollTimer) clearTimeout(_pollTimer);
  const interval = (window._isTraining) ? 3000 : 30000;
  _pollTimer = setTimeout(() => { load(); schedulePoll(); }, interval);
}
schedulePoll();
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


@app.route("/api/rescan", methods=["POST"])
@app.route("/ingress/api/rescan", methods=["POST"])
def api_rescan():
    try:
        import glob as _glob  # noqa: PLC0415
        for p in _glob.glob(os.path.join(DATA_DIR, "*.pkl")) + [STATE_PATH, BASELINE_PATH, PATTERNS_PATH, SUGGESTIONS_PATH]:
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


def start_web(port=8099):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
