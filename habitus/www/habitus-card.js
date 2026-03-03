/**
 * Habitus Custom Lovelace Cards
 * 4 card types: Pulse, Chip, Detail Panel, Timeline
 * Self-contained, no external dependencies
 */

const HABITUS_INGRESS = '/api/hassio_ingress/57582523_habitus';
const HABITUS_COLORS = {
  bg: '#0f1117', card: '#1a1d27', card2: '#242736',
  accent: '#6c8ebf', green: '#4caf50', amber: '#ffb300',
  red: '#f44336', text: '#f0f4ff', text2: '#b8c0d8',
  text3: '#7a8299', purple: '#9c7fd4', border: '#2d3148',
};

function _hsc(score) {
  const s = parseInt(score, 10);
  if (isNaN(s)) return HABITUS_COLORS.text3;
  if (s >= 70) return HABITUS_COLORS.red;
  if (s >= 40) return HABITUS_COLORS.amber;
  return HABITUS_COLORS.green;
}

function _hsl(score) {
  const s = parseInt(score, 10);
  if (isNaN(s)) return '\u2014';
  if (s >= 70) return 'Anomaly';
  if (s >= 40) return 'Elevated';
  return 'Normal';
}

function _hes(hass, id, fb) {
  if (fb === undefined) fb = '\u2014';
  const e = hass && hass.states && hass.states[id];
  return e ? e.state : fb;
}

function _hlc(hass, id) {
  const e = hass && hass.states && hass.states[id];
  if (!e || !e.last_changed) return '';
  try {
    return new Date(e.last_changed).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch(x) { return ''; }
}

const _HF = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

/* Card 1: habitus-card - Pulse */
class HabitusCard extends HTMLElement {
  static getConfigElement() { return document.createElement('div'); }
  static getStubConfig() { return {}; }
  constructor() { super(); this.attachShadow({ mode: 'open' }); this._config = {}; }
  setConfig(c) { this._config = c; }
  set hass(h) { this._hass = h; this._render(); }
  _render() {
    const h = this._hass;
    const score = _hes(h, 'sensor.habitus_anomaly_score');
    const anom = _hes(h, 'binary_sensor.habitus_anomaly_detected') === 'on';
    const top = _hes(h, 'sensor.habitus_top_anomaly');
    const time = _hlc(h, 'sensor.habitus_anomaly_score');
    const color = _hsc(score);
    const label = _hsl(score);
    const s = parseInt(score, 10);
    const pulseCSS = anom ? `
      @keyframes pulse-ring { 0% { transform: scale(1); opacity: 0.6; } 100% { transform: scale(1.8); opacity: 0; } }
      .pr { animation: pulse-ring 1.5s ease-out infinite; }
      .pr2 { animation: pulse-ring 1.5s ease-out 0.5s infinite; }
    ` : '';
    this.shadowRoot.innerHTML = `<style>
      :host{display:block}*{box-sizing:border-box}
      .c{background:rgba(26,29,39,0.85);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid ${HABITUS_COLORS.border};border-radius:16px;padding:28px 20px 20px;font-family:${_HF};color:${HABITUS_COLORS.text};cursor:pointer;transition:transform .2s,box-shadow .2s;text-align:center;position:relative;overflow:hidden}
      .c:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(0,0,0,.4)}
      .cw{position:relative;width:120px;height:120px;margin:0 auto 16px;display:flex;align-items:center;justify-content:center}
      .ci{width:100px;height:100px;border-radius:50%;background:radial-gradient(circle at 40% 35%,${color}44,${color}18);border:3px solid ${color};display:flex;align-items:center;justify-content:center;flex-direction:column;position:relative;z-index:2;transition:border-color .6s,background .6s}
      .sc{font-size:32px;font-weight:700;line-height:1;color:${color};transition:color .6s}
      .sl{font-size:11px;color:${HABITUS_COLORS.text3};text-transform:uppercase;letter-spacing:1px;margin-top:2px}
      .pr,.pr2{position:absolute;top:10px;left:10px;width:100px;height:100px;border-radius:50%;border:2px solid ${color}55;z-index:1}
      ${pulseCSS}
      .at{font-size:13px;color:${HABITUS_COLORS.text2};margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
      .tm{font-size:11px;color:${HABITUS_COLORS.text3}}
      .gl{position:absolute;top:-40px;left:50%;transform:translateX(-50%);width:160px;height:160px;background:radial-gradient(circle,${color}15,transparent 70%);pointer-events:none}
    </style>
    <div class="c"><div class="gl"></div><div class="cw">
      ${anom?'<div class="pr"></div><div class="pr2"></div>':''}
      <div class="ci"><div class="sc">${isNaN(s)?'\u2014':s}</div><div class="sl">${label}</div></div>
    </div>
    <div class="at">${top!=='\u2014'&&top!=='unavailable'?top:'No anomalies detected'}</div>
    <div class="tm">${time?'Updated '+time:''}</div></div>`;
    this.shadowRoot.querySelector('.c').onclick=()=>window.open(HABITUS_INGRESS,'_blank');
  }
  getCardSize() { return 3; }
}

/* Card 2: habitus-card-minimal - Chip */
class HabitusCardMinimal extends HTMLElement {
  static getConfigElement() { return document.createElement('div'); }
  static getStubConfig() { return {}; }
  constructor() { super(); this.attachShadow({ mode: 'open' }); this._config = {}; this._exp = false; }
  setConfig(c) { this._config = c; }
  set hass(h) { this._hass = h; this._render(); }
  _render() {
    const h = this._hass;
    const score = _hes(h, 'sensor.habitus_anomaly_score');
    const color = _hsc(score);
    const label = _hsl(score);
    const top = _hes(h, 'sensor.habitus_top_anomaly');
    const sug = _hes(h, 'sensor.habitus_suggestion_1');
    const ex = this._exp;
    this.shadowRoot.innerHTML = `<style>
      :host{display:block}*{box-sizing:border-box}
      .ch{background:${HABITUS_COLORS.card};border-radius:24px;padding:8px 16px;font-family:${_HF};color:${HABITUS_COLORS.text};cursor:pointer;display:flex;align-items:center;gap:10px;transition:all .3s;min-height:40px;flex-wrap:wrap}
      .ch:hover{background:${HABITUS_COLORS.card2}}
      .ic{font-size:18px;filter:drop-shadow(0 0 4px ${color}88);transition:filter .3s}
      .lb{font-size:13px;font-weight:500;flex:1;color:${HABITUS_COLORS.text2}}
      .bd{font-size:11px;font-weight:600;background:${color}22;color:${color};padding:2px 8px;border-radius:10px;letter-spacing:.5px}
      .dt{width:100%;font-size:12px;color:${HABITUS_COLORS.text3};padding:6px 0 2px 28px;overflow:hidden;max-height:${ex?'80px':'0'};opacity:${ex?'1':'0'};transition:max-height .3s,opacity .3s}
      .dl{margin-bottom:3px}
    </style>
    <div class="ch"><span class="ic">\uD83E\uDDE0</span><span class="lb">${label}</span><span class="bd">${score==='\u2014'?'\u2014':score+'/100'}</span>
    <div class="dt">${top!=='\u2014'&&top!=='unavailable'?'<div class="dl">\u26A1 '+top+'</div>':''}${sug!=='\u2014'&&sug!=='unavailable'?'<div class="dl">\uD83D\uDCA1 '+sug+'</div>':''}</div></div>`;
    this.shadowRoot.querySelector('.ch').onclick=()=>{this._exp=!this._exp;this._render();};
  }
  getCardSize() { return 1; }
}

/* Card 3: habitus-card-detail - Intelligence Panel
 * Fetches /api/anomalies and /api/suggestions to display:
 *   - Animated anomaly score gauge (0-100)
 *   - Top 3 anomaly reasons (entity + reason string)
 *   - Latest suggested automation (title + confidence %)
 *   - Last updated timestamp
 */
class HabitusCardDetail extends HTMLElement {
  static getConfigElement() { return document.createElement('div'); }
  static getStubConfig() { return {}; }
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._config = {};
    this._anomalies = null;  // top-3 anomaly reason dicts
    this._suggestion = null; // top suggestion with confidence
    this._fetchDone = false;
    this._fetchErr = false;
  }
  setConfig(c) { this._config = c; }
  set hass(h) {
    this._hass = h;
    if (!this._fetchDone) this._fetchData();
    this._render();
  }
  async _fetchData() {
    this._fetchDone = true;
    try {
      const [ar, sr] = await Promise.all([
        fetch(HABITUS_INGRESS + '/api/anomalies'),
        fetch(HABITUS_INGRESS + '/api/suggestions'),
      ]);
      if (ar.ok) {
        const ad = await ar.json();
        this._anomalies = (ad.anomalies || []).slice(0, 3);
      }
      if (sr.ok) {
        const sd = await sr.json();
        this._suggestion = Array.isArray(sd) && sd.length ? sd[0] : null;
      }
    } catch(e) {
      this._fetchErr = true;
    }
    this._render();
  }
  _render() {
    const h = this._hass;
    const score = _hes(h, 'sensor.habitus_anomaly_score');
    const s = parseInt(score,10)||0;
    const color = _hsc(score);
    const label = _hsl(score);
    const days = _hes(h, 'sensor.habitus_training_days');
    const sens = _hes(h, 'sensor.habitus_entity_count');
    const time = _hlc(h, 'sensor.habitus_anomaly_score');
    const pct = Math.min(s, 100);

    // Top-3 anomaly reasons
    let anomalyRows = '';
    if (this._anomalies && this._anomalies.length) {
      anomalyRows = this._anomalies.map((a, i) => {
        const name = a.name || (a.entity_id || '').split('.').pop().replace(/_/g,' ');
        const desc = a.description || name;
        const dir = a.direction === 'high' ? '\u2191' : (a.direction === 'low' ? '\u2193' : '');
        return `<div class="ar"><span class="an">${i+1}. ${name} ${dir}</span><span class="ad">${desc}</span></div>`;
      }).join('');
    } else if (this._fetchDone && !this._fetchErr) {
      const fallback = _hes(h, 'sensor.habitus_top_anomaly');
      anomalyRows = fallback !== '\u2014' && fallback !== 'unavailable'
        ? `<div class="ar"><span class="ad">${fallback}</span></div>`
        : `<div class="ar nm">No anomalies detected</div>`;
    } else {
      anomalyRows = `<div class="ar nm">\u2014</div>`;
    }

    // Latest suggestion with confidence
    let sugRow = '';
    if (this._suggestion) {
      const title = this._suggestion.title || '';
      const conf = this._suggestion.confidence != null ? `${this._suggestion.confidence}%` : '';
      sugRow = `<div class="sg"><span class="st">${title}</span>${conf ? `<span class="sc2">${conf}</span>` : ''}</div>`;
    } else {
      const fallback = _hes(h, 'sensor.habitus_suggestion_1');
      sugRow = fallback !== '\u2014' && fallback !== 'unavailable'
        ? `<div class="sg"><span class="st">${fallback}</span></div>`
        : `<div class="sg nm">No suggestions yet</div>`;
    }

    this.shadowRoot.innerHTML = `<style>
      :host{display:block}*{box-sizing:border-box}
      .p{background:${HABITUS_COLORS.card};border:1px solid ${HABITUS_COLORS.border};border-radius:16px;padding:20px;font-family:${_HF};color:${HABITUS_COLORS.text}}
      .hd{display:flex;align-items:center;gap:14px;margin-bottom:16px}
      .ga{width:44px;height:44px;border-radius:50%;background:conic-gradient(${color} ${pct*3.6}deg,${HABITUS_COLORS.border} ${pct*3.6}deg);display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .6s}
      .gi{width:34px;height:34px;border-radius:50%;background:${HABITUS_COLORS.card};display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:${color}}
      .ti{flex:1;font-size:16px;font-weight:600}.stb2{display:inline-block;font-size:10px;font-weight:600;background:${color}22;color:${color};padding:2px 8px;border-radius:8px;margin-left:6px}
      .tm{font-size:11px;color:${HABITUS_COLORS.text3};margin-left:auto}
      .sh{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:${HABITUS_COLORS.text3};margin:14px 0 6px;font-weight:600}
      .ar{margin-bottom:8px;line-height:1.5}
      .an{display:block;font-size:12px;font-weight:600;color:${HABITUS_COLORS.text2};margin-bottom:1px}
      .ad{font-size:12px;color:${HABITUS_COLORS.text3}}
      .nm{font-size:12px;color:${HABITUS_COLORS.text3}}
      .sg{display:flex;align-items:center;gap:8px;margin-top:4px}
      .st{font-size:13px;color:${HABITUS_COLORS.text2};flex:1}
      .sc2{font-size:11px;font-weight:600;background:${HABITUS_COLORS.green}22;color:${HABITUS_COLORS.green};padding:2px 8px;border-radius:8px;flex-shrink:0}
      .ft{display:flex;justify-content:center;gap:20px;font-size:11px;color:${HABITUS_COLORS.text3};padding-top:12px;margin-top:12px;border-top:1px solid ${HABITUS_COLORS.border}}
      .ab{display:inline-block;background:${HABITUS_COLORS.accent}22;color:${HABITUS_COLORS.accent};border:none;border-radius:6px;padding:5px 12px;margin-top:12px;font-size:12px;font-weight:600;cursor:pointer;font-family:${_HF};width:100%;text-align:center}
      .ab:hover{background:${HABITUS_COLORS.accent}44}
    </style>
    <div class="p">
      <div class="hd">
        <div class="ga"><div class="gi">${s}</div></div>
        <div class="ti">Home Intelligence<span class="stb2">${label}</span></div>
        <div class="tm">${time ? 'Updated ' + time : ''}</div>
      </div>
      <div class="sh">\u26a1 Top Anomaly Reasons</div>
      ${anomalyRows}
      <div class="sh">\uD83D\uDCA1 Latest Suggestion</div>
      ${sugRow}
      <button class="ab" id="ab">Open Habitus Dashboard</button>
      <div class="ft">
        <span>\uD83D\uDCC5 ${days!=='\u2014'?days+' days trained':'\u2014'}</span>
        <span>\uD83D\uDCE1 ${sens!=='\u2014'?sens+' sensors':'\u2014'}</span>
      </div>
    </div>`;
    this.shadowRoot.getElementById('ab').onclick = (e) => {
      e.stopPropagation();
      window.open(HABITUS_INGRESS, '_blank');
    };
  }
  getCardSize() { return 5; }
}

/* Card 4: habitus-card-graph - Timeline */
class HabitusCardGraph extends HTMLElement {
  static getConfigElement() { return document.createElement('div'); }
  static getStubConfig() { return {}; }
  constructor() { super(); this.attachShadow({ mode: 'open' }); this._config = {}; this._bl = null; this._ld = true; this._err = false; }
  setConfig(c) { this._config = c; }
  set hass(h) { this._hass = h; if (!this._bl && !this._ft) this._fetch(); this._render(); }
  async _fetch() {
    this._ft = true;
    try {
      const r = await fetch(HABITUS_INGRESS + '/api/baseline');
      if (!r.ok) throw new Error(r.status);
      const d = await r.json();
      this._bl = this._parse(d); this._ld = false;
    } catch(e) {
      this._err = true; this._ld = false;
      setTimeout(()=>{this._ft=false;this._bl=null;this._err=false;this._ld=true;},30000);
    }
    this._render();
  }
  _parse(d) {
    if (Array.isArray(d) && d.length >= 24) return d.slice(0,24).map(v=>typeof v==='number'?v:(v&&v.value!=null?v.value:(v&&v.power!=null?v.power:0)));
    if (d && d.hourly && Array.isArray(d.hourly)) return d.hourly.slice(0,24).map(v=>typeof v==='number'?v:(v&&v.value!=null?v.value:0));
    if (d && typeof d==='object') { const a=[]; for(let i=0;i<24;i++){const k=String(i).padStart(2,'0');a.push(d[k]!=null?d[k]:(d[i]!=null?d[i]:0));} return a; }
    return new Array(24).fill(0);
  }
  _hc(v,mx) {
    if(mx===0) return HABITUS_COLORS.card2;
    const r=v/mx;
    if(r>0.75) return HABITUS_COLORS.red;
    if(r>0.4) return HABITUS_COLORS.green;
    return HABITUS_COLORS.accent;
  }
  _render() {
    const h = this._hass;
    const score = _hes(h, 'sensor.habitus_anomaly_score');
    const color = _hsc(score);
    const label = _hsl(score);
    const now = new Date().getHours();
    let hm = '';
    if (this._ld) {
      hm = `<div class="ld"><div class="sp"></div></div>`;
    } else if (this._err || !this._bl) {
      hm = `<div class="ld" style="color:${HABITUS_COLORS.text3}">Baseline unavailable</div>`;
    } else {
      const mx = Math.max(...this._bl, 1);
      const bars = this._bl.map((v,i)=>{
        const c=this._hc(v,mx);const op=Math.max(0.15,v/mx);const cu=i===now;
        return `<div class="br${cu?' cu':''}" style="background:${c};opacity:${op}" title="${i}:00 \u2014 ${v.toFixed(1)}">${cu?'<div class="mk">\u25BC</div>':''}</div>`;
      }).join('');
      hm = `<div class="hm">${bars}</div><div class="ax">${[0,3,6,9,12,15,18,21].map(x=>'<span>'+x+'</span>').join('')}</div>`;
    }
    this.shadowRoot.innerHTML = `<style>
      :host{display:block}*{box-sizing:border-box}
      .cd{background:${HABITUS_COLORS.card};border:1px solid ${HABITUS_COLORS.border};border-radius:16px;padding:20px;font-family:${_HF};color:${HABITUS_COLORS.text}}
      .tt{font-size:13px;font-weight:600;margin-bottom:16px;color:${HABITUS_COLORS.text2}}
      .hm{display:flex;gap:3px;height:48px;align-items:flex-end;margin-bottom:6px}
      .br{flex:1;min-width:0;border-radius:4px 4px 2px 2px;height:100%;transition:opacity .4s,background .4s;position:relative}
      .cu{opacity:1!important;box-shadow:0 0 8px rgba(255,255,255,.2)}
      .mk{position:absolute;top:-14px;left:50%;transform:translateX(-50%);font-size:10px;color:${HABITUS_COLORS.text}}
      .ax{display:flex;justify-content:space-between;font-size:10px;color:${HABITUS_COLORS.text3};padding:0 1px;margin-bottom:14px}
      .bt{display:flex;align-items:center;gap:12px;padding-top:12px;border-top:1px solid ${HABITUS_COLORS.border}}
      .sv{font-size:24px;font-weight:700;color:${color}}
      .bg{font-size:11px;font-weight:600;background:${color}22;color:${color};padding:3px 10px;border-radius:10px}
      .ld{height:48px;display:flex;align-items:center;justify-content:center;margin-bottom:20px}
      .sp{width:20px;height:20px;border:2px solid ${HABITUS_COLORS.border};border-top-color:${HABITUS_COLORS.accent};border-radius:50%;animation:spin .8s linear infinite}
      @keyframes spin{to{transform:rotate(360deg)}}
    </style>
    <div class="cd"><div class="tt">24h Activity Timeline</div>${hm}
    <div class="bt"><div class="sv">${isNaN(parseInt(score,10))?'\u2014':parseInt(score,10)}</div><div class="bg">${label}</div></div></div>`;
  }
  getCardSize() { return 3; }
}

customElements.define('habitus-card', HabitusCard);
customElements.define('habitus-card-minimal', HabitusCardMinimal);
customElements.define('habitus-card-detail', HabitusCardDetail);
customElements.define('habitus-card-graph', HabitusCardGraph);

window.customCards = window.customCards || [];
window.customCards.push(
  { type: 'habitus-card', name: 'Habitus Pulse', description: 'Animated anomaly score with glassmorphism design' },
  { type: 'habitus-card-minimal', name: 'Habitus Chip', description: 'Compact pill-style status indicator' },
  { type: 'habitus-card-detail', name: 'Habitus Intelligence Panel', description: 'Full-width panel: top-3 anomaly reasons, latest suggestion with confidence' },
  { type: 'habitus-card-graph', name: 'Habitus Timeline', description: '24-hour activity heatmap' },
);
