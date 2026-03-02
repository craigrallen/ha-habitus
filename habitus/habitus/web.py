"""Habitus ingress web UI."""
import json, os, subprocess, threading
from flask import Flask, render_template_string, jsonify, request

DATA_DIR      = os.environ.get("DATA_DIR", "/data")
STATE_PATH    = os.path.join(DATA_DIR, "run_state.json")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")
MODEL_PATH    = os.path.join(DATA_DIR, "model.pkl")
RESCAN_FLAG   = os.path.join(DATA_DIR, ".rescan_requested")

app = Flask(__name__)
_scan_lock = threading.Lock()
_scan_status = {"running": False, "message": ""}

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Habitus</title>
<style>
  :root{--bg:#111318;--card:#1c1f26;--accent:#4fc3f7;--green:#4caf50;--amber:#ffb300;--red:#f44336;--text:#e0e0e0;--muted:#888}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:20px;max-width:900px;margin:0 auto}
  h1{font-size:1.4rem;font-weight:600;margin-bottom:4px}
  .subtitle{color:var(--muted);font-size:0.85rem;margin-bottom:24px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:24px}
  .card{background:var(--card);border-radius:12px;padding:18px}
  .card .label{font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}
  .card .value{font-size:2rem;font-weight:700}
  .card .unit{font-size:0.8rem;color:var(--muted);margin-left:4px}
  .score-normal{color:var(--green)}.score-amber{color:var(--amber)}.score-red{color:var(--red)}
  .section{background:var(--card);border-radius:12px;padding:18px;margin-bottom:16px}
  .section-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
  .section h2{font-size:0.9rem;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
  table{width:100%;border-collapse:collapse;font-size:0.85rem}
  th{text-align:left;color:var(--muted);font-weight:500;padding:4px 8px;border-bottom:1px solid #2a2d35}
  td{padding:6px 8px;border-bottom:1px solid #1e2028}
  .bar-wrap{background:#2a2d35;border-radius:4px;height:6px;margin-top:4px}
  .bar{height:6px;border-radius:4px;background:var(--accent)}
  .badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:0.75rem;font-weight:600}
  .badge-ok{background:#1b3a1f;color:var(--green)}.badge-warn{background:#3a2e00;color:var(--amber)}.badge-alert{background:#3a1a1a;color:var(--red)}
  .btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:8px;border:none;cursor:pointer;font-size:0.85rem;font-weight:600;transition:opacity .15s}
  .btn:hover{opacity:.85}.btn:disabled{opacity:.4;cursor:not-allowed}
  .btn-danger{background:#3a1a1a;color:var(--red);border:1px solid #5a2020}
  .btn-primary{background:#1a2a3a;color:var(--accent);border:1px solid #1e4060}
  .toast{position:fixed;bottom:24px;right:24px;background:#1c1f26;border:1px solid #2a2d35;border-radius:10px;padding:12px 18px;font-size:0.85rem;opacity:0;transition:opacity .3s;pointer-events:none;max-width:320px}
  .toast.show{opacity:1}
  .toast.success{border-color:var(--green);color:var(--green)}
  .toast.error{border-color:var(--red);color:var(--red)}
  .scan-bar{height:3px;background:var(--accent);border-radius:2px;width:0%;transition:width 30s linear;margin-top:12px;display:none}
  pre{font-size:0.8rem;color:var(--muted);white-space:pre-wrap;word-break:break-all}
</style>
</head>
<body>
<h1>🧠 Habitus</h1>
<p class="subtitle" id="subtitle">Loading...</p>

<div class="grid">
  <div class="card"><div class="label">Anomaly Score</div><div class="value" id="score">—</div></div>
  <div class="card"><div class="label">Status</div><div style="padding-top:6px" id="badge-wrap">—</div></div>
  <div class="card"><div class="label">Training Data</div><div class="value" id="training-days">—<span class="unit">days</span></div></div>
  <div class="card"><div class="label">Tracked Sensors</div><div class="value" id="entity-count">—<span class="unit">sensors</span></div></div>
</div>

<div class="section">
  <div class="section-header">
    <h2>Hourly Power Baselines</h2>
  </div>
  <table>
    <thead><tr><th>Hour</th><th>Expected Power</th><th>±Typical Variance</th><th></th></tr></thead>
    <tbody id="baseline-table"><tr><td colspan="4" style="color:var(--muted);padding:16px">Loading...</td></tr></tbody>
  </table>
</div>

<div class="section">
  <div class="section-header">
    <h2>Model & Run State</h2>
    <div style="display:flex;gap:8px">
      <button class="btn btn-primary" onclick="refresh()" id="btn-refresh">↻ Refresh</button>
      <button class="btn btn-danger" onclick="confirmRescan()" id="btn-rescan">🔄 Full Rescan</button>
    </div>
  </div>
  <div id="scan-info" style="font-size:0.82rem;color:var(--muted);margin-bottom:12px;display:none">
    ⚠️ A full rescan will delete the existing model and re-train from scratch using all available history. This may take 10–30 minutes depending on your data volume.
  </div>
  <div class="scan-bar" id="scan-bar"></div>
  <pre id="run-state">Loading...</pre>
</div>

<div class="toast" id="toast"></div>

<script>
function toast(msg, type='success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show ' + type;
  setTimeout(() => el.className = 'toast', 3500);
}

async function load() {
  const progress = await fetch('api/progress').then(r=>r.json()).catch(()=>({}));
  if (progress.running) {
    document.getElementById('subtitle').textContent = '⏳ Training in progress...';
    const pct = progress.pct || 0;
    const done = progress.done || 0;
    const total = progress.total || '?';
    document.getElementById('score').textContent = '…';
    document.getElementById('badge-wrap').innerHTML = `<span class="badge badge-warn">Training ${pct}%</span>`;
    document.getElementById('training-days').innerHTML = '—<span class="unit">days</span>';
    document.getElementById('entity-count').innerHTML = `${done}<span class="unit"> / ${total}</span>`;
    document.getElementById('baseline-table').innerHTML = `<tr><td colspan="4" style="color:var(--muted);padding:20px 8px">
      ⏳ Fetching history from ${total} sensors (${done} done, ${pct}%)…<br>
      <span style="font-size:0.78rem;margin-top:6px;display:block">First run fetches your full HA history — this takes 20–40 minutes. The page refreshes automatically.</span>
      <div class="bar-wrap" style="margin-top:12px"><div class="bar" style="width:${pct}%"></div></div>
    </td></tr>`;
    document.getElementById('run-state').textContent = JSON.stringify(progress, null, 2);
    return;
  }
  const [state, baseline] = await Promise.all([
    fetch('api/state').then(r => r.json()).catch(() => ({})),
    fetch('api/baseline').then(r => r.json()).catch(() => ({}))
  ]);

  const score = state.anomaly_score ?? 0;
  const el = document.getElementById('score');
  el.textContent = score;
  el.className = 'value ' + (score >= 70 ? 'score-red' : score >= 40 ? 'score-amber' : 'score-normal');

  const badge = score >= 70 ? '<span class="badge badge-alert">⚠ Anomaly</span>'
              : score >= 40 ? '<span class="badge badge-warn">Elevated</span>'
              :               '<span class="badge badge-ok">✓ Normal</span>';
  document.getElementById('badge-wrap').innerHTML = badge;
  document.getElementById('training-days').innerHTML = (state.training_days ?? '—') + '<span class="unit">days</span>';
  document.getElementById('entity-count').innerHTML = (state.entity_count ?? '—') + '<span class="unit">sensors</span>';

  const lastRun  = state.last_run  ? new Date(state.last_run).toLocaleString()  : 'never';
  const dataFrom = state.data_from ? new Date(state.data_from).toLocaleDateString() : '?';
  const dateTo   = state.data_to   ? new Date(state.data_to).toLocaleDateString()   : '?';
  document.getElementById('subtitle').textContent = `Last run: ${lastRun} · Data window: ${dataFrom} → ${dateTo}`;
  document.getElementById('run-state').textContent = JSON.stringify(state, null, 2);

  const byHour = {};
  for (const [key, v] of Object.entries(baseline)) {
    const [h] = key.split('_');
    if (!byHour[h]) byHour[h] = {sum:0,n:0,std:0};
    byHour[h].sum += v.mean_power; byHour[h].n++;
    byHour[h].std = Math.max(byHour[h].std, v.std_power);
  }
  const maxPow = Math.max(...Object.values(byHour).map(v => v.sum/v.n), 1);
  const rows = Array.from({length:24},(_,h) => {
    const b = byHour[h] || {sum:0,n:1,std:0};
    const mean = Math.round(b.sum/b.n), std = Math.round(b.std);
    const pct  = Math.round((mean/maxPow)*100);
    const period = h<6?'Night':h<12?'Morning':h<18?'Afternoon':'Evening';
    return `<tr><td>${String(h).padStart(2,'0')}:00 <span style="color:var(--muted);font-size:.75rem">${period}</span></td><td>${mean}W</td><td>±${std}W</td><td style="width:120px"><div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div></td></tr>`;
  });
  document.getElementById('baseline-table').innerHTML = rows.join('');
}

function refresh() { load(); toast('Refreshed'); }

function confirmRescan() {
  document.getElementById('scan-info').style.display = 'block';
  const btn = document.getElementById('btn-rescan');
  btn.textContent = '⚠ Confirm Rescan';
  btn.onclick = doRescan;
}

async function doRescan() {
  const btn = document.getElementById('btn-rescan');
  btn.disabled = true;
  btn.textContent = '🔄 Scanning...';
  document.getElementById('scan-info').style.display = 'none';

  const bar = document.getElementById('scan-bar');
  bar.style.display = 'block';
  bar.style.width = '0%';
  setTimeout(() => bar.style.width = '95%', 100);

  try {
    const r = await fetch('api/rescan', {method:'POST'});
    const d = await r.json();
    if (d.ok) {
      toast('Full rescan started — may take 10–30 min. Refresh in a while.', 'success');
    } else {
      toast('Failed: ' + (d.error || 'unknown'), 'error');
    }
  } catch(e) {
    toast('Request failed: ' + e, 'error');
  }

  setTimeout(() => { bar.style.display='none'; bar.style.width='0%'; }, 2000);
  btn.disabled = false;
  btn.textContent = '🔄 Full Rescan';
  btn.onclick = confirmRescan;
}

load();
setInterval(load, 60000);
</script>
</body>
</html>
"""

@app.route('/')
@app.route('/ingress')
@app.route('/ingress/')
def index():
    return render_template_string(PAGE)

@app.route('/api/progress')
@app.route('/ingress/api/progress')
def api_progress():
    progress_path = os.path.join(DATA_DIR, 'progress.json')
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return jsonify(json.load(f))
    return jsonify({})

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

@app.route('/api/rescan', methods=['POST'])
@app.route('/ingress/api/rescan', methods=['POST'])
def api_rescan():
    """Delete model artifacts and flag for full rescan on next main loop iteration."""
    try:
        # Wipe model artifacts so next run does a full retrain
        for path in [MODEL_PATH, STATE_PATH, BASELINE_PATH]:
            if os.path.exists(path):
                os.remove(path)
        # Write flag file (main loop checks this too)
        with open(RESCAN_FLAG, 'w') as f:
            f.write('1')
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

def start_web(port=8099):
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
