"""Habitus ingress web UI — insights dashboard served inside HA."""
import json, os, pickle, datetime
from flask import Flask, render_template_string, jsonify

DATA_DIR = os.environ.get("DATA_DIR", "/data")
STATE_PATH = os.path.join(DATA_DIR, "run_state.json")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Habitus</title>
<style>
  :root { --bg: #111318; --card: #1c1f26; --accent: #4fc3f7; --green: #4caf50; --amber: #ffb300; --red: #f44336; --text: #e0e0e0; --muted: #888; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: var(--card); border-radius: 12px; padding: 18px; }
  .card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card .unit { font-size: 0.8rem; color: var(--muted); margin-left: 4px; }
  .score-normal { color: var(--green); }
  .score-amber  { color: var(--amber); }
  .score-red    { color: var(--red); }
  .section { background: var(--card); border-radius: 12px; padding: 18px; margin-bottom: 16px; }
  .section h2 { font-size: 0.9rem; font-weight: 600; color: var(--accent); margin-bottom: 14px; text-transform: uppercase; letter-spacing: .06em; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 4px 8px; border-bottom: 1px solid #2a2d35; }
  td { padding: 6px 8px; border-bottom: 1px solid #1e2028; }
  .bar-wrap { background: #2a2d35; border-radius: 4px; height: 6px; margin-top: 4px; }
  .bar { height: 6px; border-radius: 4px; background: var(--accent); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
  .badge-ok  { background: #1b3a1f; color: var(--green); }
  .badge-warn { background: #3a2e00; color: var(--amber); }
  .badge-alert { background: #3a1a1a; color: var(--red); }
  .last-run { color: var(--muted); font-size: 0.78rem; margin-top: 4px; }
  .spinner { color: var(--muted); font-size: 0.9rem; padding: 40px; text-align: center; }
</style>
</head>
<body>
<h1>🧠 Habitus</h1>
<p class="subtitle">Behavioral intelligence for your home — <span id="status-text">loading...</span></p>

<div class="grid" id="metrics">
  <div class="card"><div class="label">Anomaly Score</div><div class="value" id="score">—</div></div>
  <div class="card"><div class="label">Status</div><div class="value" id="badge-wrap" style="font-size:1rem;padding-top:6px">—</div></div>
  <div class="card"><div class="label">Training Data</div><div class="value" id="training-days">—<span class="unit">days</span></div></div>
  <div class="card"><div class="label">Tracked Sensors</div><div class="value" id="entity-count">—<span class="unit">sensors</span></div></div>
</div>

<div class="section">
  <h2>Hourly Power Baselines</h2>
  <table>
    <thead><tr><th>Hour</th><th>Expected Power</th><th>Typical Range</th><th></th></tr></thead>
    <tbody id="baseline-table"><tr><td colspan="4" class="spinner">Loading baselines...</td></tr></tbody>
  </table>
</div>

<div class="section">
  <h2>Run State</h2>
  <pre id="run-state" style="font-size:0.8rem;color:var(--muted);white-space:pre-wrap">Loading...</pre>
</div>

<script>
async function load() {
  const [state, baseline] = await Promise.all([
    fetch('api/state').then(r => r.json()),
    fetch('api/baseline').then(r => r.json())
  ]);

  const score = state.anomaly_score ?? 0;
  const el = document.getElementById('score');
  el.textContent = score;
  el.className = 'value ' + (score >= 70 ? 'score-red' : score >= 40 ? 'score-amber' : 'score-normal');

  const badge = score >= 70 ? '<span class="badge badge-alert">⚠ Anomaly</span>'
              : score >= 40 ? '<span class="badge badge-warn">Elevated</span>'
              :               '<span class="badge badge-ok">Normal</span>';
  document.getElementById('badge-wrap').innerHTML = badge;
  document.getElementById('training-days').innerHTML = (state.training_days ?? '—') + '<span class="unit">days</span>';
  document.getElementById('entity-count').innerHTML = (state.entity_count ?? '—') + '<span class="unit">sensors</span>';
  
  const lastRun = state.last_run ? new Date(state.last_run).toLocaleString() : 'never';
  const dataFrom = state.data_from ? new Date(state.data_from).toLocaleDateString() : '?';
  const dateTo   = state.data_to   ? new Date(state.data_to).toLocaleDateString()   : '?';
  document.getElementById('status-text').textContent = `last run ${lastRun} · data ${dataFrom} → ${dateTo}`;
  document.getElementById('run-state').textContent = JSON.stringify(state, null, 2);

  // Baselines — show by hour of day (averaged across weekdays)
  const byHour = {};
  for (const [key, v] of Object.entries(baseline)) {
    const [h] = key.split('_');
    if (!byHour[h]) byHour[h] = {sum: 0, n: 0, std: 0};
    byHour[h].sum += v.mean_power;
    byHour[h].n++;
    byHour[h].std = Math.max(byHour[h].std, v.std_power);
  }
  const maxPow = Math.max(...Object.values(byHour).map(v => v.sum / v.n));
  const rows = Array.from({length: 24}, (_, h) => {
    const b = byHour[h] || {sum: 0, n: 1, std: 0};
    const mean = Math.round(b.sum / b.n);
    const std  = Math.round(b.std);
    const pct  = maxPow > 0 ? Math.round((mean / maxPow) * 100) : 0;
    const period = h < 6 ? 'Night' : h < 12 ? 'Morning' : h < 18 ? 'Afternoon' : 'Evening';
    return `<tr>
      <td>${String(h).padStart(2,'0')}:00 <span style="color:var(--muted);font-size:0.75rem">${period}</span></td>
      <td>${mean}W</td>
      <td>±${std}W</td>
      <td style="width:120px"><div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div></td>
    </tr>`;
  });
  document.getElementById('baseline-table').innerHTML = rows.join('');
}
load();
</script>
</body>
</html>
"""

@app.route('/')
@app.route('/ingress')
@app.route('/ingress/')
def index():
    return render_template_string(PAGE)

@app.route('/api/state')
@app.route('/ingress/api/state')
def api_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route('/api/baseline')
@app.route('/ingress/api/baseline')
def api_baseline():
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return jsonify(json.load(f))
    return jsonify({})

def start_web(port=8099):
    app.run(host='0.0.0.0', port=port, debug=False)
