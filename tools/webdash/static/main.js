let ws;
let chart;
const gpuLabels = [];
const gpuMemUsed = [];
const gpuUtil = [];

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    console.log('WS connected');
    setInterval(() => ws.send(JSON.stringify({type:'gpu'})), 1000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'gpu') {
      updateGPU(msg.data);
    } else if (msg.type === 'log') {
      appendLog(msg.line);
      tryParseResult(msg.line);
    } else if (msg.type === 'done') {
      setRunEnabled(true);
    }
  };
  ws.onclose = () => console.log('WS closed');
}

function setRunEnabled(v){
  const btn = document.getElementById('run');
  btn.disabled = !v;
}

function appendLog(line){
  const out = document.getElementById('output');
  out.textContent += line + '\n';
  out.scrollTop = out.scrollHeight;
}

function tryParseResult(line){
  // Try to parse final JSON-line from bench script to update summary
  if (!line.startsWith('{') || !line.endsWith('}')) return;
  try {
    const data = JSON.parse(line);
  renderSummaryTable(data);
  } catch {}
}

function updateGPU(info){
  const tbl = document.getElementById('gpu-table');
  tbl.innerHTML = '';
  if (!info || !info.nvml){
    tbl.innerHTML = '<div>NVML</div><div>Không khả dụng</div>';
    return;
  }
  const kv = [
    ['Tên GPU', info.name],
    ['VRAM tổng (MB)', info.mem_total_mb],
    ['VRAM dùng (MB)', info.mem_used_mb],
    ['GPU util (%)', info.gpu_util],
    ['MEM util (%)', info.mem_util],
    ['Nhiệt độ (°C)', info.temp_c],
  ];
  for (const [k,v] of kv){
    const a = document.createElement('div'); a.textContent = k;
    const b = document.createElement('div'); b.textContent = v;
    tbl.appendChild(a); tbl.appendChild(b);
  }
  // chart
  if (!chart) initChart();
  const ts = new Date().toLocaleTimeString();
  gpuLabels.push(ts); if (gpuLabels.length>60) gpuLabels.shift();
  gpuMemUsed.push(info.mem_used_mb); if (gpuMemUsed.length>60) gpuMemUsed.shift();
  gpuUtil.push(info.gpu_util); if (gpuUtil.length>60) gpuUtil.shift();
  chart.data.labels = [...gpuLabels];
  chart.data.datasets[0].data = [...gpuMemUsed];
  chart.data.datasets[1].data = [...gpuUtil];
  chart.update('none');
}

function initChart(){
  const ctx = document.getElementById('chart-gpu');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {label:'VRAM dùng (MB)', data:[], borderColor:'#58a6ff', backgroundColor:'rgba(88,166,255,0.2)', yAxisID:'y'},
        {label:'GPU util (%)', data:[], borderColor:'#3fb950', backgroundColor:'rgba(63,185,80,0.2)', yAxisID:'y1'},
      ]
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        y: { type: 'linear', position:'left' },
        y1:{ type: 'linear', position:'right', grid:{ drawOnChartArea:false } }
      }
    }
  });
}

function runBench(){
  setRunEnabled(false);
  document.getElementById('output').textContent = '';
  const msg = {
    type: 'bench',
    data: {
      preset: document.getElementById('preset').value,
      device: document.getElementById('device').value,
      strategy: document.getElementById('strategy').value || undefined,
      tokens: parseInt(document.getElementById('tokens').value || '64'),
      paging_window: parseInt(document.getElementById('paging').value || '0') || undefined,
      stream: document.getElementById('stream').checked,
    }
  };
  ws.send(JSON.stringify(msg));
}

function renderSummaryTable(data){
  const wrap = document.getElementById('summary-table');
  const v = data.vanilla || {};
  const p = data.pylittle || {};
  const dev = data.devices || {};
  const strategy = data.strategy || null;
  const vram = data.vram_delta_mb ?? 'n/a';

  // Build a readable table
  const rows = [
    ['Thiết bị', `Vanilla: ${dev.vanilla ?? 'n/a'} | PyLittle: ${dev.pylittle ?? 'n/a'}`],
    v.latency_s != null || p.latency_s != null
      ? ['Latency (s)', `Vanilla: ${fmt(v.latency_s)} | PyLittle: ${fmt(p.latency_s)} | Speedup: ${fmt(data.speedup_x)}`]
      : ['Throughput (chars/s)', `Vanilla: ${fmt(v.throughput_chars_s)} | PyLittle: ${fmt(p.throughput_chars_s)}`],
    ['VRAM Δ (MB)', `${vram}`],
    ['Độ dài output', `Vanilla len: ${v.len ?? 'n/a'} | PyLittle len: ${p.len ?? 'n/a'}`],
    ['Tokens', `Generated: ${orNA(v.tokens_generated)} | Requested: ${orNA(p.tokens_requested)}`],
    ['Chiến lược', strategy ? JSON.stringify(strategy) : '(none)'],
  ];

  wrap.innerHTML = `<table>
    <thead><tr><th>Mục</th><th>Giá trị</th></tr></thead>
    <tbody>
      ${rows.map(([k,val]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(val))}</td></tr>`).join('')}
    </tbody>
  </table>`;
}

function fmt(x){ return (x==null) ? 'n/a' : (typeof x==='number' ? (Math.round(x*1000)/1000) : x); }
function orNA(x){ return (x==null||Number.isNaN(x)) ? 'n/a' : x; }
function escapeHtml(s){ return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c])); }

window.addEventListener('load', () => {
  connectWS();
  document.getElementById('run').addEventListener('click', runBench);
});
