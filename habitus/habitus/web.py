"""Habitus ingress web UI — v1.0"""
import json, os
from flask import Flask, render_template_string, jsonify, request

DATA_DIR         = os.environ.get("DATA_DIR", "/data")
STATE_PATH       = os.path.join(DATA_DIR, "run_state.json")
BASELINE_PATH    = os.path.join(DATA_DIR, "baseline.json")
MODEL_PATH       = os.path.join(DATA_DIR, "model.pkl")
PATTERNS_PATH    = os.path.join(DATA_DIR, "patterns.json")
SUGGESTIONS_PATH = os.path.join(DATA_DIR, "suggestions.json")
PROGRESS_PATH    = os.path.join(DATA_DIR, "progress.json")
RESCAN_FLAG      = os.path.join(DATA_DIR, ".rescan_requested")

app = Flask(__name__)

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Habitus</title>
<style>
:root{--bg:#111318;--card:#1c1f26;--accent:#4fc3f7;--green:#4caf50;--amber:#ffb300;--red:#f44336;--text:#e0e0e0;--muted:#888;--border:#2a2d35}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:0;max-width:960px;margin:0 auto}
header{padding:20px 20px 0;display:flex;align-items:center;gap:12px;margin-bottom:4px}
header h1{font-size:1.3rem;font-weight:700}
.subtitle{padding:0 20px 16px;color:var(--muted);font-size:0.82rem}
nav{display:flex;gap:2px;padding:0 20px 0;border-bottom:1px solid var(--border);margin-bottom:20px}
nav button{background:none;border:none;color:var(--muted);padding:10px 16px;cursor:pointer;font-size:0.85rem;border-bottom:2px solid transparent;transition:all .15s}
nav button.active{color:var(--accent);border-bottom-color:var(--accent)}
nav button:hover{color:var(--text)}
.tab{display:none;padding:0 20px 24px}
.tab.active{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--card);border-radius:10px;padding:16px}
.card .label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.card .value{font-size:1.8rem;font-weight:700}
.card .unit{font-size:0.75rem;color:var(--muted);margin-left:3px}
.score-normal{color:var(--green)}.score-amber{color:var(--amber)}.score-red{color:var(--red)}
.section{background:var(--card);border-radius:10px;padding:16px;margin-bottom:14px}
.section-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.section h2{font-size:0.82rem;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
table{width:100%;border-collapse:collapse;font-size:0.83rem}
th{text-align:left;color:var(--muted);font-weight:500;padding:4px 8px;border-bottom:1px solid var(--border)}
td{padding:6px 8px;border-bottom:1px solid #1e2028}
.bar-wrap{background:#2a2d35;border-radius:4px;height:5px;margin-top:3px}
.bar{height:5px;border-radius:4px;background:var(--accent);transition:width .5s ease}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600}
.badge-ok{background:#1b3a1f;color:var(--green)}.badge-warn{background:#3a2e00;color:var(--amber)}.badge-alert{background:#3a1a1a;color:var(--red)}.badge-info{background:#1a2a3a;color:var(--accent)}
.btn{display:inline-flex;align-items:center;gap:5px;padding:7px 14px;border-radius:7px;border:none;cursor:pointer;font-size:0.82rem;font-weight:600;transition:opacity .15s}
.btn:hover{opacity:.85}.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-danger{background:#3a1a1a;color:var(--red);border:1px solid #5a2020}
.btn-primary{background:#1a2a3a;color:var(--accent);border:1px solid #1e4060}
.btn-success{background:#1b3a1f;color:var(--green);border:1px solid #2a5a2f}
.suggestion{background:#13161d;border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:12px}
.suggestion-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px}
.suggestion h3{font-size:0.9rem;font-weight:600;color:var(--text)}
.suggestion .desc{color:var(--muted);font-size:0.82rem;line-height:1.5;margin-bottom:12px}
.suggestion pre{background:#0d1017;border-radius:6px;padding:12px;font-size:0.75rem;color:#a8d8f0;overflow-x:auto;margin-bottom:10px;white-space:pre-wrap}
.confidence{font-size:0.72rem;color:var(--muted)}
.toast{position:fixed;bottom:20px;right:20px;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 16px;font-size:0.82rem;opacity:0;transition:opacity .3s;pointer-events:none}
.toast.show{opacity:1}.toast.success{border-color:var(--green);color:var(--green)}.toast.error{border-color:var(--red);color:var(--red)}
.progress-section{background:var(--card);border-radius:10px;padding:20px;margin-bottom:14px}
pre.state{font-size:0.78rem;color:var(--muted);white-space:pre-wrap;word-break:break-all}
.cat-tag{font-size:0.7rem;padding:1px 6px;border-radius:4px;background:#1a2a3a;color:var(--accent);margin-left:6px}
</style>
</head>
<body>
<header>
  <div>
    <h1>🧠 Habitus</h1>
  </div>
</header>
<p class="subtitle" id="subtitle">Loading...</p>

<nav>
  <button class="active" onclick="showTab('overview',this)">Overview</button>
  <button onclick="showTab('automations',this)">Proposed Automations</button>
  <button onclick="showTab('energy',this)">Energy Patterns</button>
  <button onclick="showTab('settings',this)">Settings</button>
</nav>

<!-- OVERVIEW TAB -->
<div id="tab-overview" class="tab active">
  <div id="progress-section" style="display:none" class="progress-section">
    <div id="progress-content"></div>
    <div class="bar-wrap" style="height:8px;margin-top:12px"><div class="bar" id="progress-bar" style="width:0%"></div></div>
    <div style="color:var(--muted);font-size:0.75rem;margin-top:6px">Page refreshes every 30 seconds.</div>
  </div>
  <div id="main-content">
    <div class="grid">
      <div class="card"><div class="label">Anomaly Score</div><div class="value" id="score">—</div></div>
      <div class="card"><div class="label">Status</div><div style="padding-top:4px" id="badge-wrap">—</div></div>
      <div class="card"><div class="label">Training Data</div><div class="value" id="training-days">—<span class="unit">days</span></div></div>
      <div class="card"><div class="label">Tracked Sensors</div><div class="value" id="entity-count">—<span class="unit">sensors</span></div></div>
    </div>
    <div class="section">
      <div class="section-header"><h2>Hourly Power Baselines</h2></div>
      <table>
        <thead><tr><th>Hour</th><th>Expected Power</th><th>±Variance</th><th></th></tr></thead>
        <tbody id="baseline-table"><tr><td colspan="4" style="color:var(--muted);padding:16px">Loading...</td></tr></tbody>
      </table>
    </div>
    <div class="section" id="patterns-section" style="display:none">
      <div class="section-header"><h2>Discovered Patterns</h2></div>
      <div id="patterns-content"></div>
    </div>
  </div>
</div>

<!-- AUTOMATIONS TAB -->
<div id="tab-automations" class="tab">
  <div class="section">
    <div class="section-header">
      <h2>Proposed Automations</h2>
      <span style="color:var(--muted);font-size:0.78rem">Generated from your home's patterns</span>
    </div>
    <div id="suggestions-list"><div style="color:var(--muted);padding:12px">Loading suggestions...</div></div>
  </div>
</div>

<!-- ENERGY TAB -->
<div id="tab-energy" class="tab">
  <div class="section">
    <div class="section-header"><h2>Weekly Power Profile</h2></div>
    <table>
      <thead><tr><th>Day</th><th>Avg Power</th><th>Activity</th><th></th></tr></thead>
      <tbody id="weekly-table"><tr><td colspan="4" style="color:var(--muted);padding:16px">Loading...</td></tr></tbody>
    </table>
  </div>
  <div class="section">
    <div class="section-header"><h2>Monthly Overview</h2></div>
    <table>
      <thead><tr><th>Month</th><th>Avg Power</th><th>Avg Temp</th><th></th></tr></thead>
      <tbody id="monthly-table"><tr><td colspan="4" style="color:var(--muted);padding:16px">Loading...</td></tr></tbody>
    </table>
  </div>
</div>

<!-- SETTINGS TAB -->
<div id="tab-settings" class="tab">
  <div class="section">
    <div class="section-header">
      <h2>Model Management</h2>
      <div style="display:flex;gap:8px">
        <button class="btn btn-primary" onclick="load()">↻ Refresh</button>
        <button class="btn btn-danger" onclick="confirmRescan()" id="btn-rescan">🔄 Full Rescan</button>
      </div>
    </div>
    <div id="scan-info" style="display:none;color:var(--amber);font-size:0.82rem;padding:10px;background:#3a2e0030;border-radius:6px;margin-bottom:12px">
      ⚠️ This will delete the existing model and re-train from scratch using all available history. Takes 20–40 minutes.
    </div>
    <pre class="state" id="run-state">Loading...</pre>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let currentTab = 'overview';
function showTab(id, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  btn.classList.add('active');
  currentTab = id;
}

function toast(msg, type='success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show ' + type;
  setTimeout(() => el.className='toast', 3000);
}

function copyYaml(id) {
  const pre = document.getElementById('yaml-'+id);
  navigator.clipboard.writeText(pre.textContent).then(() => toast('Copied to clipboard ✓'));
}

async function addToHA(yaml, id) {
  const btn = document.getElementById('add-btn-'+id);
  btn.disabled = true; btn.textContent = 'Adding…';
  const r = await fetch('api/add_automation', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({yaml})
  });
  const d = await r.json();
  if (d.ok) { btn.textContent = '✓ Added'; toast('Automation added to HA ✓'); }
  else { btn.disabled=false; btn.textContent='Add to HA'; toast('Failed: '+d.error, 'error'); }
}

async function load() {
  const [state, baseline, progress, patterns, suggestions] = await Promise.all([
    fetch('api/state').then(r=>r.json()).catch(()=>({})),
    fetch('api/baseline').then(r=>r.json()).catch(()=>({})),
    fetch('api/progress').then(r=>r.json()).catch(()=>({})),
    fetch('api/patterns').then(r=>r.json()).catch(()=>({})),
    fetch('api/suggestions').then(r=>r.json()).catch(()=>([])),
  ]);

  // Progress state
  if (progress.running) {
    document.getElementById('progress-section').style.display = 'block';
    document.getElementById('main-content').style.opacity = '0.3';
    const pct = progress.pct||0, done=progress.done||0, total=progress.total||'?';
    const rows = (progress.rows||0).toLocaleString();
    const eta = progress.eta_min > 0 ? ` · ~${progress.eta_min} min remaining` : '';
    const elapsed = progress.elapsed_min ? `${progress.elapsed_min} min elapsed` : '';
    const isTraining = progress.phase === 'training';
    document.getElementById('subtitle').textContent = isTraining
      ? `🧠 Training model on ${rows} hourly snapshots…`
      : `📡 Fetching history: ${done}/${total} sensors (${pct}%) · ${elapsed}${eta}`;
    document.getElementById('progress-content').innerHTML = isTraining
      ? `<div style="color:var(--text);margin-bottom:8px">🧠 <strong>Training model</strong> — fitting ${rows} hourly data points</div>`
      : `<div style="color:var(--text);margin-bottom:8px">📡 <strong>Fetching sensor history</strong> — ${done} of ${total} sensors complete</div>
         <div style="color:var(--muted);font-size:0.82rem">${rows} hourly data points loaded${eta ? ' · ' + eta.trim() : ''}</div>`;
    document.getElementById('progress-bar').style.width = pct + '%';
    return;
  } else {
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('main-content').style.opacity = '1';
  }

  // Main stats
  const score = state.anomaly_score ?? 0;
  const el = document.getElementById('score');
  el.textContent = score;
  el.className = 'value '+(score>=70?'score-red':score>=40?'score-amber':'score-normal');
  const badge = score>=70?'<span class="badge badge-alert">⚠ Anomaly</span>'
              : score>=40?'<span class="badge badge-warn">Elevated</span>'
              :           '<span class="badge badge-ok">✓ Normal</span>';
  document.getElementById('badge-wrap').innerHTML = badge;
  document.getElementById('training-days').innerHTML=(state.training_days??'—')+'<span class="unit">days</span>';
  document.getElementById('entity-count').innerHTML=(state.entity_count??'—')+'<span class="unit">sensors</span>';
  const lastRun=state.last_run?new Date(state.last_run).toLocaleString():'never';
  const from=state.data_from?new Date(state.data_from).toLocaleDateString():'?';
  const to=state.data_to?new Date(state.data_to).toLocaleDateString():'?';
  document.getElementById('subtitle').textContent=`Last run: ${lastRun} · Data: ${from} → ${to}`;
  document.getElementById('run-state').textContent=JSON.stringify(state,null,2);

  // Baselines
  const byHour={};
  for(const[k,v] of Object.entries(baseline)){const[h]=k.split('_');if(!byHour[h]){byHour[h]={sum:0,n:0,std:0}}byHour[h].sum+=v.mean_power;byHour[h].n++;byHour[h].std=Math.max(byHour[h].std,v.std_power)}
  const maxP=Math.max(...Object.values(byHour).map(v=>v.sum/v.n),1);
  document.getElementById('baseline-table').innerHTML=Array.from({length:24},(_,h)=>{
    const b=byHour[h]||{sum:0,n:1,std:0};
    const m=Math.round(b.sum/b.n),s=Math.round(b.std),pct=Math.round(m/maxP*100);
    const p=h<6?'Night':h<12?'Morning':h<18?'Afternoon':'Evening';
    return`<tr><td>${String(h).padStart(2,'0')}:00 <span style="color:var(--muted);font-size:.72rem">${p}</span></td><td>${m}W</td><td>±${s}W</td><td style="width:100px"><div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div></td></tr>`;
  }).join('');

  // Patterns summary
  const routine = patterns.daily_routine;
  if (routine) {
    document.getElementById('patterns-section').style.display = 'block';
    const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    document.getElementById('patterns-content').innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;font-size:0.83rem">
        <div><span style="color:var(--muted)">Est. wakeup</span><br><strong>${routine.estimated_wakeup_hour!=null?routine.estimated_wakeup_hour+':00':'unknown'}</strong></div>
        <div><span style="color:var(--muted)">Est. sleep</span><br><strong>${routine.estimated_sleep_hour!=null?routine.estimated_sleep_hour+':00':'unknown'}</strong></div>
        <div><span style="color:var(--muted)">Peak usage</span><br><strong>${routine.peak_usage_hour}:00 — ${routine.peak_usage_watts}W</strong></div>
        <div><span style="color:var(--muted)">Night baseline</span><br><strong>${routine.night_baseline_watts}W</strong></div>
      </div>`;
  }

  // Suggestions
  if (suggestions.length) {
    const catColors={routine:'badge-info',energy:'badge-warn',anomaly:'badge-alert'};
    document.getElementById('suggestions-list').innerHTML = suggestions.map(s=>`
      <div class="suggestion">
        <div class="suggestion-header">
          <div>
            <h3>${s.title} <span class="cat-tag">${s.category}</span></h3>
            <div class="confidence" style="margin-top:3px">Confidence: ${s.confidence}%</div>
          </div>
        </div>
        <div class="desc">${s.description}</div>
        <pre id="yaml-${s.id}">${s.yaml.trim()}</pre>
        <div style="display:flex;gap:8px">
          <button class="btn btn-primary" onclick="copyYaml('${s.id}')">📋 Copy YAML</button>
          <button class="btn btn-success" id="add-btn-${s.id}" onclick="addToHA(document.getElementById('yaml-${s.id}').textContent,'${s.id}')">+ Add to HA</button>
        </div>
      </div>`).join('');
  } else {
    document.getElementById('suggestions-list').innerHTML = '<div style="color:var(--muted);padding:12px">No suggestions yet — run Habitus first to generate automation ideas.</div>';
  }

  // Weekly
  const weekly=patterns.weekly||{};
  const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const maxW=Math.max(...Object.values(weekly).map(v=>v.mean_power_w),1);
  document.getElementById('weekly-table').innerHTML=days.map(d=>{
    const v=weekly[d]||{mean_power_w:0,activity:0};
    const pct=Math.round(v.mean_power_w/maxW*100);
    const we=d==='Sat'||d==='Sun';
    return`<tr><td>${d}${we?' <span style="color:var(--muted);font-size:.7rem">weekend</span>':''}</td><td>${Math.round(v.mean_power_w)}W</td><td>${Math.round(v.activity)} changes/h</td><td style="width:120px"><div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div></td></tr>`;
  }).join('');

  // Monthly
  const seasonal=patterns.seasonal||{};
  const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const maxM=Math.max(...Object.values(seasonal).map(v=>v.mean_power_w),1);
  document.getElementById('monthly-table').innerHTML=months.filter(m=>seasonal[m]).map(m=>{
    const v=seasonal[m];
    const pct=Math.round(v.mean_power_w/maxM*100);
    return`<tr><td>${m}</td><td>${Math.round(v.mean_power_w)}W</td><td>${v.mean_temp_c}°C</td><td style="width:120px"><div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div></td></tr>`;
  }).join('');
}

function confirmRescan() {
  const info=document.getElementById('scan-info');
  const btn=document.getElementById('btn-rescan');
  info.style.display='block';
  btn.textContent='⚠ Confirm Rescan';
  btn.onclick=doRescan;
}
async function doRescan() {
  document.getElementById('btn-rescan').disabled=true;
  document.getElementById('btn-rescan').textContent='Triggered…';
  document.getElementById('scan-info').style.display='none';
  const r=await fetch('api/rescan',{method:'POST'});
  const d=await r.json();
  if(d.ok){toast('Full rescan started — check back in 20–30 min','success');}
  else{toast('Failed: '+(d.error||'unknown'),'error');}
  document.getElementById('btn-rescan').disabled=false;
  document.getElementById('btn-rescan').textContent='🔄 Full Rescan';
  document.getElementById('btn-rescan').onclick=confirmRescan;
}

load();
setInterval(load, 30000);
</script>
</body>
</html>"""

@app.route('/') 
@app.route('/ingress')
@app.route('/ingress/')
def index(): return render_template_string(PAGE)

def _read(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

@app.route('/api/state'); @app.route('/ingress/api/state')
def api_state(): return jsonify(_read(STATE_PATH) or {})

@app.route('/api/baseline'); @app.route('/ingress/api/baseline')
def api_baseline(): return jsonify(_read(BASELINE_PATH) or {})

@app.route('/api/progress'); @app.route('/ingress/api/progress')
def api_progress(): return jsonify(_read(PROGRESS_PATH) or {})

@app.route('/api/patterns'); @app.route('/ingress/api/patterns')
def api_patterns(): return jsonify(_read(PATTERNS_PATH) or {})

@app.route('/api/suggestions'); @app.route('/ingress/api/suggestions')
def api_suggestions(): return jsonify(_read(SUGGESTIONS_PATH) or [])

@app.route('/api/rescan', methods=['POST']); @app.route('/ingress/api/rescan', methods=['POST'])
def api_rescan():
    try:
        for p in [MODEL_PATH, STATE_PATH, BASELINE_PATH, PATTERNS_PATH, SUGGESTIONS_PATH]:
            if os.path.exists(p): os.remove(p)
        open(RESCAN_FLAG,'w').write('1')
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/add_automation', methods=['POST']); @app.route('/ingress/api/add_automation', methods=['POST'])
def api_add_automation():
    import yaml, requests as req
    data = request.get_json()
    yaml_str = data.get('yaml','')
    ha_url   = os.environ.get('HA_URL','http://supervisor/core')
    token    = os.environ.get('SUPERVISOR_TOKEN','')
    try:
        parsed = yaml.safe_load(yaml_str)
        auto   = parsed.get('automation', parsed)
        r = req.post(f"{ha_url}/api/config/automation/config/{auto.get('alias','habitus_auto').lower().replace(' ','_')}",
                     headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"},
                     json=auto, timeout=10)
        if r.status_code in (200,201): return jsonify({"ok":True})
        return jsonify({"ok":False,"error":f"HA returned {r.status_code}"}), 400
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 500

def start_web(port=8099):
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
