"""Habitus v2.0 web UI — 5 tabs."""
import json, os, yaml as _yaml
from flask import Flask, render_template_string, jsonify, request

DATA_DIR          = os.environ.get("DATA_DIR", "/data")
STATE_PATH        = os.path.join(DATA_DIR, "run_state.json")
BASELINE_PATH     = os.path.join(DATA_DIR, "baseline.json")
PATTERNS_PATH     = os.path.join(DATA_DIR, "patterns.json")
SUGGESTIONS_PATH  = os.path.join(DATA_DIR, "suggestions.json")
ANOMALIES_PATH    = os.path.join(DATA_DIR, "entity_anomalies.json")
PROGRESS_PATH     = os.path.join(DATA_DIR, "progress.json")
MODEL_PATH        = os.path.join(DATA_DIR, "model.pkl")
RESCAN_FLAG       = os.path.join(DATA_DIR, ".rescan_requested")

app = Flask(__name__)

def _read(path, default=None):
    try:
        if os.path.exists(path):
            with open(path) as f: return json.load(f)
    except Exception: pass
    return default

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Habitus</title>
<style>
:root{--bg:#111318;--card:#1c1f26;--accent:#4fc3f7;--green:#4caf50;--amber:#ffb300;--red:#f44336;--text:#e0e0e0;--muted:#888;--border:#2a2d35}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:980px;margin:0 auto}
header{padding:18px 20px 4px;display:flex;align-items:center;gap:10px}
header h1{font-size:1.25rem;font-weight:700}
.sub{padding:0 20px 14px;color:var(--muted);font-size:0.8rem}
nav{display:flex;gap:2px;padding:0 20px;border-bottom:1px solid var(--border);margin-bottom:18px;overflow-x:auto}
nav button{background:none;border:none;color:var(--muted);padding:9px 14px;cursor:pointer;font-size:0.83rem;border-bottom:2px solid transparent;white-space:nowrap}
nav button.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab{display:none;padding:0 20px 28px}.tab.active{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--card);border-radius:10px;padding:14px}
.lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.val{font-size:1.7rem;font-weight:700}.unit{font-size:.72rem;color:var(--muted);margin-left:2px}
.sn{color:var(--green)}.sa{color:var(--amber)}.sr{color:var(--red)}
.sec{background:var(--card);border-radius:10px;padding:14px;margin-bottom:12px}
.sh{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.sh h2{font-size:.8rem;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{text-align:left;color:var(--muted);font-weight:500;padding:4px 8px;border-bottom:1px solid var(--border)}
td{padding:6px 8px;border-bottom:1px solid #1e2028}
.bw{background:#2a2d35;border-radius:3px;height:4px;margin-top:3px}.bx{height:4px;border-radius:3px;background:var(--accent);transition:width .4s}
.badge{display:inline-block;padding:2px 7px;border-radius:20px;font-size:.7rem;font-weight:600}
.bok{background:#1b3a1f;color:var(--green)}.bwa{background:#3a2e00;color:var(--amber)}.bal{background:#3a1a1a;color:var(--red)}.binf{background:#1a2a3a;color:var(--accent)}.bbo{background:#1a2a2a;color:#80cbc4}.blo{background:#2a2a1a;color:#fff176}
.btn{display:inline-flex;align-items:center;gap:4px;padding:6px 13px;border-radius:7px;border:none;cursor:pointer;font-size:.8rem;font-weight:600;transition:opacity .15s}
.btn:hover{opacity:.85}.btn:disabled{opacity:.4;cursor:not-allowed}
.btp{background:#1a2a3a;color:var(--accent);border:1px solid #1e4060}
.btd{background:#3a1a1a;color:var(--red);border:1px solid #5a2020}
.bts{background:#1b3a1f;color:var(--green);border:1px solid #2a5a2f}
.sug{background:#13161d;border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:10px}
.sug.na{opacity:.45}
.sug h3{font-size:.88rem;font-weight:600;margin-bottom:4px}
.sug .desc{color:var(--muted);font-size:.8rem;line-height:1.5;margin-bottom:10px}
.sug pre{background:#0d1017;border-radius:6px;padding:10px;font-size:.72rem;color:#a8d8f0;overflow-x:auto;margin-bottom:8px;white-space:pre-wrap;max-height:220px;overflow-y:auto}
.ftabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.ftab{padding:4px 12px;border-radius:20px;border:1px solid var(--border);background:none;color:var(--muted);font-size:.78rem;cursor:pointer}
.ftab.active{background:var(--accent);color:#111;border-color:var(--accent)}
.pov{position:fixed;top:0;left:0;right:0;bottom:0;background:#111318ee;display:flex;align-items:center;justify-content:center;z-index:100}
.pbox{background:var(--card);border-radius:14px;padding:28px;max-width:440px;width:90%;text-align:center}
.pbox h2{font-size:1.1rem;margin-bottom:8px}
.pbox .pdesc{color:var(--muted);font-size:.83rem;margin-bottom:16px;line-height:1.5}
.pbw{background:#2a2d35;border-radius:4px;height:8px;margin-bottom:8px;overflow:hidden}
.pbx{height:8px;border-radius:4px;background:var(--accent);transition:width 1s ease}
.toast{position:fixed;bottom:20px;right:20px;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-size:.8rem;opacity:0;transition:opacity .3s;pointer-events:none;max-width:300px;z-index:200}
.toast.show{opacity:1}.toast.ts{border-color:var(--green);color:var(--green)}.toast.te{border-color:var(--red);color:var(--red)}
pre.raw{font-size:.75rem;color:var(--muted);white-space:pre-wrap;word-break:break-all}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:600px){.row2{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div><h1>🧠 Habitus</h1></div></header>
<p class="sub" id="sub">Loading...</p>

<nav>
  <button class="active" onclick="tab('overview',this)">Overview</button>
  <button onclick="tab('breakdown',this)">Anomaly Breakdown</button>
  <button onclick="tab('automations',this)">Automations</button>
  <button onclick="tab('energy',this)">Energy & Patterns</button>
  <button onclick="tab('settings',this)">Settings</button>
</nav>

<!-- OVERVIEW -->
<div id="tab-overview" class="tab active">
  <div class="grid">
    <div class="card"><div class="lbl">Anomaly Score</div><div class="val" id="score">—</div></div>
    <div class="card"><div class="lbl">Status</div><div style="padding-top:4px" id="badge"></div></div>
    <div class="card"><div class="lbl">Training Data</div><div class="val" id="tdays">—<span class="unit">days</span></div></div>
    <div class="card"><div class="lbl">Tracked Sensors</div><div class="val" id="ecount">—<span class="unit">sensors</span></div></div>
  </div>
  <div class="row2">
    <div class="sec">
      <div class="sh"><h2>Patterns Discovered</h2></div>
      <div id="pat-summary" style="font-size:.83rem;color:var(--muted)">Loading...</div>
    </div>
    <div class="sec">
      <div class="sh"><h2>Top Anomalies Now</h2></div>
      <div id="top-anomalies" style="font-size:.82rem">Loading...</div>
    </div>
  </div>
  <div class="sec">
    <div class="sh"><h2>Hourly Power Baseline</h2></div>
    <table><thead><tr><th>Hour</th><th>Expected</th><th>±Variance</th><th></th></tr></thead>
    <tbody id="bl-table"><tr><td colspan="4" style="color:var(--muted);padding:14px">Loading...</td></tr></tbody></table>
  </div>
</div>

<!-- ANOMALY BREAKDOWN -->
<div id="tab-breakdown" class="tab">
  <div class="sec">
    <div class="sh"><h2>Entity Anomaly Scores</h2><span style="color:var(--muted);font-size:.77rem" id="bd-ts"></span></div>
    <table><thead><tr><th>Sensor</th><th>Current</th><th>Baseline</th><th>Deviation</th><th></th></tr></thead>
    <tbody id="bd-table"><tr><td colspan="5" style="color:var(--muted);padding:14px">No data yet — run Habitus first.</td></tr></tbody></table>
  </div>
</div>

<!-- AUTOMATIONS -->
<div id="tab-automations" class="tab">
  <div class="ftabs">
    <button class="ftab active" onclick="filterSug('all',this)">All</button>
    <button class="ftab" onclick="filterSug('routine',this)">Routine</button>
    <button class="ftab" onclick="filterSug('energy',this)">Energy</button>
    <button class="ftab" onclick="filterSug('boat',this)">Boat</button>
    <button class="ftab" onclick="filterSug('anomaly',this)">Anomaly</button>
    <button class="ftab" onclick="filterSug('lovelace',this)">Lovelace</button>
  </div>
  <div id="sug-list"><div style="color:var(--muted);padding:12px">Loading suggestions...</div></div>
</div>

<!-- ENERGY & PATTERNS -->
<div id="tab-energy" class="tab">
  <div class="row2">
    <div class="sec">
      <div class="sh"><h2>Weekly Profile</h2></div>
      <table><thead><tr><th>Day</th><th>Avg Power</th><th></th></tr></thead>
      <tbody id="wk-table"></tbody></table>
    </div>
    <div class="sec">
      <div class="sh"><h2>Seasonal Models</h2></div>
      <div id="season-status" style="font-size:.83rem"></div>
    </div>
  </div>
  <div class="sec">
    <div class="sh"><h2>Monthly Overview</h2></div>
    <table><thead><tr><th>Month</th><th>Avg Power</th><th>Avg Temp</th><th></th></tr></thead>
    <tbody id="mo-table"></tbody></table>
  </div>
</div>

<!-- SETTINGS -->
<div id="tab-settings" class="tab">
  <div class="sec">
    <div class="sh">
      <h2>Model Management</h2>
      <div style="display:flex;gap:8px">
        <button class="btn btp" onclick="doRefresh()">↻ Refresh</button>
        <button class="btn btd" onclick="confirmRescan()" id="btn-rescan">🔄 Full Rescan</button>
      </div>
    </div>
    <div id="rescan-warn" style="display:none;color:var(--amber);font-size:.8rem;padding:10px;background:#3a2e0030;border-radius:6px;margin-bottom:10px">
      ⚠️ Deletes the existing model and retrains from scratch — takes 20–40 minutes.
    </div>
    <pre class="raw" id="raw-state">Loading...</pre>
  </div>
  <div class="sec">
    <div class="sh"><h2>About</h2></div>
    <div style="font-size:.83rem;color:var(--muted);line-height:1.8">
      <div>📦 <a href="https://github.com/craigrallen/ha-habitus" style="color:var(--accent)" target="_blank">github.com/craigrallen/ha-habitus</a></div>
      <div>☕ <a href="https://buymeacoffee.com/craigrallen" style="color:var(--amber)" target="_blank">buymeacoffee.com/craigrallen</a></div>
    </div>
  </div>
</div>

<!-- PROGRESS OVERLAY -->
<div id="prog-overlay" class="pov" style="display:none">
  <div class="pbox">
    <div style="font-size:2rem;margin-bottom:10px" id="prog-icon">📡</div>
    <h2 id="prog-title">Training in progress...</h2>
    <p class="pdesc" id="prog-desc">Fetching sensor history from Home Assistant</p>
    <div class="pbw"><div class="pbx" id="prog-bar" style="width:0%"></div></div>
    <div style="color:var(--muted);font-size:.75rem;margin-top:6px" id="prog-eta"></div>
    <div style="color:var(--muted);font-size:.72rem;margin-top:4px">Page refreshes every 30 seconds</div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let allSuggestions = [];
let currentFilter = 'all';

function tab(id, btn) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  btn.classList.add('active');
}

function toast(msg, type='ts') {
  const el=document.getElementById('toast');
  el.textContent=msg; el.className='toast show '+type;
  setTimeout(()=>el.className='toast',3200);
}

function filterSug(cat, btn) {
  currentFilter = cat;
  document.querySelectorAll('.ftab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  renderSuggestions();
}

function renderSuggestions() {
  const list = currentFilter==='all' ? allSuggestions : allSuggestions.filter(s=>s.category===currentFilter);
  if (!list.length) { document.getElementById('sug-list').innerHTML='<div style="color:var(--muted);padding:12px">No suggestions in this category.</div>'; return; }
  const catBadge = {routine:'binf',energy:'bwa',boat:'bbo',anomaly:'bal',lovelace:'blo'};
  document.getElementById('sug-list').innerHTML = list.map(s=>`
    <div class="sug ${s.applicable===false?'na':''}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
        <div>
          <h3>${s.title} <span class="badge ${catBadge[s.category]||'binf'}">${s.category}</span></h3>
          <div style="color:var(--muted);font-size:.7rem;margin-top:2px">Confidence: ${s.confidence}%${s.applicable===false?' · <span style="color:var(--red)">Entity not found in your system</span>':''}</div>
        </div>
      </div>
      <div class="desc">${s.description}</div>
      <pre id="yaml-${s.id}">${(s.yaml||'').trim()}</pre>
      <div style="display:flex;gap:7px">
        <button class="btn btp" onclick="copy('yaml-${s.id}')">📋 Copy YAML</button>
        ${s.category!=='lovelace'?`<button class="btn bts" id="add-${s.id}" onclick="addToHA('${s.id}')">+ Add to HA</button>`:''}
      </div>
    </div>`).join('');
}

function copy(id) {
  navigator.clipboard.writeText(document.getElementById(id).textContent).then(()=>toast('Copied ✓'));
}

async function addToHA(id) {
  const btn = document.getElementById('add-'+id);
  const yaml = document.getElementById('yaml-'+id).textContent;
  btn.disabled=true; btn.textContent='Adding…';
  const r = await fetch('api/add_automation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yaml})});
  const d = await r.json();
  if(d.ok){btn.textContent='✓ Added';toast('Automation added to HA ✓');}
  else{btn.disabled=false;btn.textContent='+ Add to HA';toast('Failed: '+(d.error||'unknown'),'te');}
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

  // Progress overlay
  if (progress && progress.running) {
    document.getElementById('prog-overlay').style.display='flex';
    const phase = progress.phase||'fetching';
    const icons = {fetching:'📡',building_baselines:'🗄️',training:'🧠',seasonal_training:'🌱',pattern_analysis:'🔍'};
    const titles = {fetching:'Fetching sensor history',building_baselines:'Building entity baselines',training:'Training anomaly model',seasonal_training:'Training seasonal models',pattern_analysis:'Analysing patterns'};
    const pct = progress.pct||0;
    document.getElementById('prog-icon').textContent = icons[phase]||'⏳';
    document.getElementById('prog-title').textContent = titles[phase]||phase;
    document.getElementById('prog-bar').style.width = pct+'%';
    const rows=(progress.rows||0).toLocaleString();
    const desc = phase==='fetching'
      ? `${progress.done||0} of ${progress.total||'?'} sensors queried · ${rows} data points loaded`
      : phase==='training' ? `Fitting Isolation Forest on ${rows} hourly snapshots`
      : phase==='seasonal_training' ? 'Building separate models per season (winter/spring/summer/autumn)'
      : phase==='building_baselines' ? `Computing per-entity normal ranges for ${rows} data points`
      : `Generating automation suggestions from discovered patterns`;
    document.getElementById('prog-desc').textContent = desc;
    const eta = progress.eta_min > 0 ? `~${progress.eta_min} min remaining` : '';
    const el = progress.elapsed_min ? `${progress.elapsed_min} min elapsed` : '';
    document.getElementById('prog-eta').textContent = [el,eta].filter(Boolean).join(' · ');
    document.getElementById('sub').textContent = `⏳ ${titles[phase]||phase}… ${pct}%`;
    return;
  } else {
    document.getElementById('prog-overlay').style.display='none';
  }

  // Overview
  const score = state.anomaly_score??0;
  const el=document.getElementById('score');
  el.textContent=score;
  el.className='val '+(score>=70?'sr':score>=40?'sa':'sn');
  const bstr = score>=70?'<span class="badge bal">⚠ Anomaly</span>':score>=40?'<span class="badge bwa">Elevated</span>':'<span class="badge bok">✓ Normal</span>';
  document.getElementById('badge').innerHTML=bstr;
  document.getElementById('tdays').innerHTML=(state.training_days??'—')+'<span class="unit">days</span>';
  document.getElementById('ecount').innerHTML=(state.entity_count??'—')+'<span class="unit">sensors</span>';
  const lr=state.last_run?new Date(state.last_run).toLocaleString():'never';
  const df=state.data_from?new Date(state.data_from).toLocaleDateString():'?';
  const dt=state.data_to?new Date(state.data_to).toLocaleDateString():'?';
  document.getElementById('sub').textContent=`Last run: ${lr} · Data: ${df} → ${dt}`;
  document.getElementById('raw-state').textContent=JSON.stringify(state,null,2);

  // Patterns summary
  const r=patterns.daily_routine||{};
  document.getElementById('pat-summary').innerHTML = r.peak_usage_hour!=null ? `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div><span style="color:var(--muted)">Est. wakeup</span><br><strong>${r.estimated_wakeup_hour!=null?r.estimated_wakeup_hour+':00':'unknown'}</strong></div>
      <div><span style="color:var(--muted)">Est. sleep</span><br><strong>${r.estimated_sleep_hour!=null?r.estimated_sleep_hour+':00':'unknown'}</strong></div>
      <div><span style="color:var(--muted)">Peak hour</span><br><strong>${r.peak_usage_hour}:00 — ${r.peak_usage_watts}W</strong></div>
      <div><span style="color:var(--muted)">Night base</span><br><strong>${r.night_baseline_watts}W</strong></div>
    </div>` : '<div style="color:var(--muted)">No patterns yet — run Habitus first.</div>';

  // Top anomalies
  const ents = (anomalies.anomalies||[]).slice(0,5);
  document.getElementById('top-anomalies').innerHTML = ents.length
    ? ents.map(e=>`<div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid var(--border)">
        <div style="display:flex;justify-content:space-between">
          <span style="color:var(--text)">${e.name}</span>
          <span class="${e.z_score>=3?'sr':e.z_score>=1.5?'sa':'sn'}">${e.z_score}σ</span>
        </div>
        <div style="color:var(--muted);font-size:.75rem;margin-top:2px">${e.current_value}${e.unit} vs baseline ${e.baseline_mean}${e.unit}</div>
      </div>`).join('')
    : '<div style="color:var(--muted)">No anomalies detected right now.</div>';

  // Baseline table
  const byH={};
  for(const[k,v] of Object.entries(baseline)){const[h]=k.split('_');if(!byH[h]){byH[h]={sum:0,n:0,std:0}}byH[h].sum+=v.mean_power;byH[h].n++;byH[h].std=Math.max(byH[h].std,v.std_power)}
  const mxP=Math.max(...Object.values(byH).map(v=>v.sum/v.n),1);
  document.getElementById('bl-table').innerHTML=Array.from({length:24},(_,h)=>{
    const b=byH[h]||{sum:0,n:1,std:0};const m=Math.round(b.sum/b.n),s=Math.round(b.std),p=Math.round(m/mxP*100);
    const pd=h<6?'Night':h<12?'Morning':h<18?'Afternoon':'Evening';
    return`<tr><td>${String(h).padStart(2,'0')}:00 <span style="color:var(--muted);font-size:.7rem">${pd}</span></td><td>${m}W</td><td>±${s}W</td><td style="width:90px"><div class="bw"><div class="bx" style="width:${p}%"></div></div></td></tr>`;
  }).join('');

  // Anomaly breakdown
  const allEnts = anomalies.anomalies||[];
  const bts = anomalies.timestamp ? new Date(anomalies.timestamp).toLocaleString() : '';
  document.getElementById('bd-ts').textContent = bts ? `as of ${bts}` : '';
  document.getElementById('bd-table').innerHTML = allEnts.length
    ? allEnts.map(e=>{
        const z=e.z_score; const pct=Math.min(100,Math.round(z/5*100));
        const cls=z>=3?'sr':z>=1.5?'sa':'sn';
        return`<tr><td><div>${e.name}</div><div style="color:var(--muted);font-size:.7rem">${e.entity_id}</div></td><td>${e.current_value}${e.unit}</td><td>${e.baseline_mean}${e.unit} ±${e.baseline_std}${e.unit}</td><td class="${cls}">${z}σ</td><td style="width:80px"><div class="bw"><div class="bx" style="width:${pct}%;background:${z>=3?'var(--red)':z>=1.5?'var(--amber)':'var(--green)'}"></div></div></td></tr>`;
      }).join('')
    : '<tr><td colspan="5" style="color:var(--muted);padding:14px">No entity anomalies detected.</td></tr>';

  // Suggestions
  allSuggestions = suggestions;
  renderSuggestions();

  // Weekly
  const wk=patterns.weekly||{};const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const mxW=Math.max(...Object.values(wk).map(v=>v.mean_power_w),1);
  document.getElementById('wk-table').innerHTML=days.map(d=>{
    const v=wk[d]||{mean_power_w:0,activity:0};const p=Math.round(v.mean_power_w/mxW*100);const we=d==='Sat'||d==='Sun';
    return`<tr><td>${d}${we?' <span style="color:var(--muted);font-size:.68rem">wknd</span>':''}</td><td>${Math.round(v.mean_power_w)}W</td><td style="width:100px"><div class="bw"><div class="bx" style="width:${p}%"></div></div></td></tr>`;
  }).join('');

  // Monthly
  const se=patterns.seasonal||{};const mos=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const mxM=Math.max(...Object.values(se).map(v=>v.mean_power_w),1);
  document.getElementById('mo-table').innerHTML=mos.filter(m=>se[m]).map(m=>{
    const v=se[m];const p=Math.round(v.mean_power_w/mxM*100);
    return`<tr><td>${m}</td><td>${Math.round(v.mean_power_w)}W</td><td>${v.mean_temp_c}°C</td><td style="width:90px"><div class="bw"><div class="bx" style="width:${p}%"></div></div></td></tr>`;
  }).join('');

  // Seasonal models
  const sm=state.seasonal_models||{};const seasons=['winter','spring','summer','autumn'];const icons={winter:'❄️',spring:'🌱',summer:'☀️',autumn:'🍂'};
  document.getElementById('season-status').innerHTML=seasons.map(s=>`
    <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border);font-size:.83rem">
      <span>${icons[s]} ${s.charAt(0).toUpperCase()+s.slice(1)}</span>
      ${sm[s]?'<span class="badge bok">trained</span>':'<span class="badge" style="background:#222;color:var(--muted)">no data yet</span>'}
    </div>`).join('');
}

function doRefresh(){load();toast('Refreshed');}

function confirmRescan(){
  document.getElementById('rescan-warn').style.display='block';
  const btn=document.getElementById('btn-rescan');
  btn.textContent='⚠ Confirm Rescan';btn.onclick=doRescan;
}
async function doRescan(){
  const btn=document.getElementById('btn-rescan');
  btn.disabled=true;btn.textContent='Triggered…';
  document.getElementById('rescan-warn').style.display='none';
  const r=await fetch('api/rescan',{method:'POST'});
  const d=await r.json();
  if(d.ok)toast('Full rescan started — check back in ~30 min ✓');
  else toast('Failed: '+(d.error||'?'),'te');
  btn.disabled=false;btn.textContent='🔄 Full Rescan';btn.onclick=confirmRescan;
}

load();
setInterval(load, 30000);
</script>
</body>
</html>"""

@app.route('/'); @app.route('/ingress'); @app.route('/ingress/')
def index(): return render_template_string(PAGE)

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

@app.route('/api/anomalies'); @app.route('/ingress/api/anomalies')
def api_anomalies(): return jsonify(_read(ANOMALIES_PATH) or {})

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
    import requests as req
    data = request.get_json()
    yaml_str = data.get('yaml','')
    ha_url = os.environ.get('HA_URL','http://supervisor/core')
    token  = os.environ.get('SUPERVISOR_TOKEN','')
    try:
        parsed = _yaml.safe_load(yaml_str)
        auto   = parsed.get('automation', parsed)
        alias  = auto.get('alias','habitus_auto').lower().replace(' ','_').replace('—','')
        r = req.post(f"{ha_url}/api/config/automation/config/{alias}",
                     headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"},
                     json=auto, timeout=10)
        if r.status_code in (200,201,204): return jsonify({"ok":True})
        return jsonify({"ok":False,"error":f"HA {r.status_code}: {r.text[:100]}"}),400
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

def start_web(port=8099):
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
