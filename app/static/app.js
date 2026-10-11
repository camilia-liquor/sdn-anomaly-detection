/* 实时检测大屏 + 迷你 SOC 控制台前端逻辑：纯原生 JS + canvas，无任何外部依赖（断网可演示） */
"use strict";

const S = {
  pos: 0,
  n: 0,
  playing: true,
  speed: 10,
  thr: 0.5,
  overview: null,
  buf: [],            // {p, attack, alarm} 最近若干窗口，用于画曲线
  tableRows: [],
  alarms: 0,
  falseAlarms: 0,
  attackRows: 0,
  decided: 0,
  correct: 0,
  lastAlarmWin: -1,
  alarmRun: 0,
  confirm: 1,
  segments: [],
  timer: null,
};

/* ------------------------------------------------------------------ 常量 */
const MAXBUF = 600;
const MAXROWS = 60;

const STATUS_REFRESH_MS = 3000;   // 健康状态条刷新周期
const LEDGER_REFRESH_MS = 2000;   // 审计台账刷新周期
const REVIEW_REFRESH_MS = 3000;   // 复核台（待研判队列 / 观察池）刷新周期
const FEED_BATCH_MAX = 32;        // 单次推给在线检测接口的最大流记录数
const ATTACK_HOLD_MS = 4000;      // 攻击态保持时长（无新判定后自动回落）
const DISPOSAL_HOLD_MS = 30000;   // 处置标记保持时长（真实系统里封禁是持续的）
const TOPO_FRAME_MS = 33;         // 拓扑动画约 30fps：肉眼流畅、空转 CPU 占用低

const FONT_CN = '"Microsoft YaHei", "PingFang SC", system-ui, sans-serif';
const FONT_MONO = '"Cascadia Mono", Consolas, Menlo, monospace';

/* ------------------------------------------------------------------ 迷你 SOC 控制台状态 */
const SOC = {
  policy: { threshold: 0.5, confirm_windows: 3, action: "observe", dry_run: true },
  whitelist: [],
  detector: { ready: null, reason: "", down: false },
  stats: null,
  online: [],            // 服务端权威台账（GET /api/alerts）
  local: [],             // 回放兜底台账（在线检测服务不可用时的降级展示）
  seen: new Set(),
  feed: { enabled: true, busy: false, sent: 0, sampled: 0, failures: 0, lastError: "", off: false },
  lastWin: null,
  attacker: null,        // {host, cls, p, ip}
  attackerIP: null,
  mode: "online",        // online | replay
  lastLedgerAt: 0,
  toastTimer: null,
};

/* 演示拓扑：2 台交换机 + 6 台主机（主机编址 10.0.0.1~10.0.0.6）。
   数据集本身没有 IP 字段，因此攻击源按「判定类别 → 固定演示地址」映射，
   同一类别始终落到同一台主机，方便评委对照；在线接口回填的 Src IP 也是这个地址，
   因此拓扑、审计台账、在线统计三处的源 IP 完全一致、可互相追溯。 */
const HOSTS = [
  { id: "h1", ip: "10.0.0.1", sw: 0, role: "管理终端" },
  { id: "h2", ip: "10.0.0.2", sw: 0, role: "办公主机" },
  { id: "h3", ip: "10.0.0.3", sw: 0, role: "数据服务" },
  { id: "h4", ip: "10.0.0.4", sw: 1, role: "Web 服务" },
  { id: "h5", ip: "10.0.0.5", sw: 1, role: "接入主机" },
  { id: "h6", ip: "10.0.0.6", sw: 1, role: "接入主机" },
];
HOSTS.forEach((h) => {
  h.attackUntil = 0; h.disposalUntil = 0; h.cls = ""; h.p = 0;
  h.action = ""; h.executed = false; h.srcIP = ""; h.win = null;
});
const HOST_DEFAULT = 4;   // 取不到末位时的兜底主机：h5

const CLASS_TO_IP = {
  DDoS: "10.0.0.5", DoS: "10.0.0.6", Botnet: "10.0.0.4",
  BruteForce: "10.0.0.3", Probe: "10.0.0.2", WebAttack: "10.0.0.1", U2R: "10.0.0.1",
};
const DEFAULT_ATTACK_IP = "10.0.0.5";

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* ------------------------------------------------------------------ 启动 */
async function boot() {
  let ov = null;
  try {
    ov = await fetchJSON("/api/overview");
  } catch (e) {
    ov = null;
  }
  // 先起 SOC 控制台（状态条 / 策略 / 台账 / 拓扑），保证任何一路失败都不影响其它面板
  bootSocConsole();
  // 漂移与未知态势面板：独立取数，失败只降级本面板，不影响回放与其它面板
  loadNovelty(DRIFT.bins);
  if (!ov || !ov.ok) {
    showError((ov && ov.reason) || "演示数据未就绪");
    return;
  }
  S.overview = ov;
  S.n = ov.n_rows;
  S.thr = ov.threshold || 0.5;
  S.confirm = ov.confirm_windows || 1;
  S.segments = ov.segments || [];
  $("threshold").value = S.thr;
  $("threshold-val").textContent = S.thr.toFixed(2);

  buildTableHead(ov.display_features);
  buildInjectButtons(ov.classes, S.segments);
  renderMetrics(ov.metrics);
  loadFigures();
  setStatus("运行中", "");
  S.timer = setInterval(tick, 400);
  tick();
}

function showError(msg) {
  setStatus("未就绪", "error");
  const box = document.createElement("div");
  box.className = "error-box";
  box.innerHTML = "<b>演示数据未就绪</b><br>" + msg +
    "<br><br>请先在项目根目录执行：<code>python run_all.py --smoke</code>（快速）或 <code>python run_all.py</code>（完整）" +
    "<br><br>提示：健康状态条、策略控制台与攻击拓扑仍会尝试连接在线检测接口，可在没有回放数据时照常演示降级行为。";
  document.querySelector(".layout").prepend(box);
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " → " + r.status);
  return r.json();
}

/* ------------------------------------------------------------------ 主循环 */
async function tick() {
  if (!S.playing || S.pos >= S.n) {
    if (S.pos >= S.n) setStatus("回放结束", "paused");
    return;
  }
  const count = Math.max(1, Math.min(S.speed, S.n - S.pos));
  let data;
  try {
    data = await fetchJSON(`/api/stream?pos=${S.pos}&count=${count}&threshold=${S.thr}`);
  } catch (e) {
    setStatus("请求失败", "error");
    return;
  }
  applyBatch(data);
  S.pos = data.end;
  driftSyncCursor();          // 漂移面板游标跟随回放位置（只在跨格时重绘）
  if (S.pos >= S.n) setStatus("回放结束", "paused");
}

function applyBatch(data) {
  const rows = data.rows;
  for (const r of rows) {
    if (r.attack) S.attackRows++;
    if (r.win) {
      S.buf.push({ p: r.win.p, attack: r.attack, alarm: r.win.alarm, cls: r.win.pred, idx: r.win.idx });
      if (S.buf.length > MAXBUF) S.buf.shift();
      S.decided++;
      if (r.win.pred === r.true) S.correct++;
      // 告警确认（与后端在线评估同一口径）：要求连续 N 个窗口都超阈值才真正告警。
      // N 来自后端 metrics 的 online.confirm_windows，因此界面统计与报告数字一致。
      const needConfirm = S.confirm || 1;
      S.alarmRun = r.win.alarm ? (S.alarmRun || 0) + 1 : 0;
      if (S.alarmRun >= needConfirm && r.win.idx !== S.lastAlarmWin) {
        S.lastAlarmWin = r.win.idx;
        S.alarms++;
        if (r.win.true_attack === false) S.falseAlarms++;
        pushAlarm(r);
        pushLocalLedger(r);
      } else if (!r.win.alarm) {
        S.lastAlarmWin = -1;
      }
    }
  }
  S.tableRows = rows.concat(S.tableRows).slice(0, MAXROWS);
  renderTable(S.tableRows);
  const last = rows[rows.length - 1];
  renderGauge(last);
  renderStats();
  drawChart();
  $("progress-bar").style.width = (100 * S.pos / S.n).toFixed(2) + "%";
  $("pos-hint").textContent = `位置 ${S.pos} / ${S.n}（已回放 ${(100 * S.pos / S.n).toFixed(1)}%） · ` +
    `时间轴 ${last ? last.t.toFixed(1) : "0"} 秒 · 窗口长度 ${S.overview.window_size}`;
  const sbPos = $("sb-pos");
  if (sbPos) sbPos.textContent = `${S.pos.toLocaleString()} / ${S.n.toLocaleString()}`;

  // 拓扑着色（真实窗口判定）+ 把真实窗口推给在线检测接口
  topoNoteFromRows(rows);
  feedRows(rows);
}

/* ------------------------------------------------------------------ 渲染 */
function renderGauge(row) {
  if (!row) return;
  const p = row.win ? row.win.p : null;
  const val = $("risk-val");
  if (p === null) {
    val.textContent = "--";
    val.classList.remove("alarm");
    $("risk-label").textContent = "窗口尚未填满";
    return;
  }
  val.textContent = p.toFixed(3);
  val.classList.toggle("alarm", row.win.alarm);
  $("risk-label").textContent = row.win.alarm
    ? `⚠ 告警：判定为 ${row.win.pred}`
    : `正常 · 模型判定 ${row.win.pred}`;

  const probs = row.win.probs || {};
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  $("prob-bars").innerHTML = sorted.map(([name, v], i) =>
    `<div class="pb ${i === 0 ? "top" : ""}">
       <span class="name">${name}</span>
       <span class="track"><span class="fill" style="width:${(100 * v).toFixed(1)}%"></span></span>
       <span class="val">${v.toFixed(3)}</span>
     </div>`).join("");

  $("win-meta").innerHTML = `
    <span>当前窗口</span><b>#${row.win.idx}（覆盖第 ${row.win.idx >= 0 ? "" : ""}${S.overview.window_size} 条流）</b>
    <span>真实类别</span><b style="color:${row.attack ? "#fca5a5" : "#86efac"}">${row.true}</b>
    <span>判定结果</span><b>${row.win.pred}</b>
    <span>异常概率</span><b>${row.win.p.toFixed(4)}</b>`;
}

function renderTable(rows) {
  const tbody = $("flow-body");
  tbody.innerHTML = rows.map((r) => {
    const alarm = r.win && r.win.alarm;
    const cls = alarm ? "alarm" : (r.attack ? "attack" : "");
    const feats = (r.feat || []).map((v) => `<td>${fmtNum(v)}</td>`).join("");
    const pred = r.win
      ? `<span class="tag ${r.win.alarm ? "no" : "ok"}">${r.win.pred}</span>`
      : `<span class="tag n">-</span>`;
    const truth = `<span class="tag ${r.attack ? "no" : "ok"}">${r.true}</span>`;
    const pTxt = r.win ? r.win.p.toFixed(3) : "-";
    return `<tr class="${cls}">
      <td>${r.i}</td><td>${r.t.toFixed(2)}</td>${feats}
      <td>${truth}</td><td>${pred}</td><td>${pTxt}</td></tr>`;
  }).join("");
}

function buildTableHead(features) {
  $("flow-head").innerHTML = "<tr>" +
    `<th class="l">#</th><th class="l">时间(s)</th>` +
    features.map((f) => `<th title="${f}">${shortName(f)}</th>`).join("") +
    `<th class="l">真实</th><th class="l">预测</th><th>p(异常)</th></tr>`;
}

function renderStats() {
  const acc = S.decided ? (S.correct / S.decided) : 0;
  const items = [
    ["已回放流记录", S.pos.toLocaleString()],
    ["已判定窗口", S.decided.toLocaleString()],
    ["告警次数", S.alarms.toString()],
    ["其中误报", S.falseAlarms.toString()],
    ["真实异常记录", S.attackRows.toString()],
    ["窗口判定准确率", (100 * acc).toFixed(2) + "%"],
  ];
  $("stats").innerHTML = items.map(([k, v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

function pushAlarm(row) {
  const log = $("alarm-log");
  if (log.querySelector(".log-empty")) log.innerHTML = "";
  const div = document.createElement("div");
  div.className = "log-row";
  const truth = row.win.true || row.true;
  const ok = row.win.true_attack !== false;
  div.innerHTML = `<span class="t">${row.t.toFixed(2)}s</span>
    <span class="c">${row.win.pred}</span>
    <span class="d">p=${row.win.p.toFixed(3)} · 窗口真值 ${truth}${ok ? "（命中）" : "（误报）"}</span>`;
  log.prepend(div);
  while (log.children.length > 60) log.removeChild(log.lastChild);
}

function buildInjectButtons(classes, segments) {
  const have = new Set(segments.map((s) => s.class));
  $("inject-buttons").innerHTML = classes.filter((c) => c !== "Normal").map((c) => {
    const disabled = have.has(c) ? "" : "disabled style='opacity:.4'";
    return `<button data-cls="${c}" ${disabled}>▶ 注入 ${c}</button>`;
  }).join("");
  $("inject-buttons").querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cls = btn.dataset.cls;
      const r = await fetchJSON(`/api/first_segment_of/${encodeURIComponent(cls)}`);
      if (r.error) return;
      jumpTo(r.pos, `注入 ${cls}（真实片段 #${r.index}）`);
    });
  });
}

async function jumpTo(pos, note) {
  S.pos = pos;
  S.lastAlarmWin = -1;
  await fetchJSON(`/api/stream?pos=${Math.max(0, pos - 1)}&count=1&threshold=${S.thr}`); // 预热
  setStatus(`运行中 · ${note}`, "");
  if (!S.playing) togglePlay();
  feedPrime(pos);   // 先把跳转点之前的窗口推给在线服务，避免缓冲区里混着旧位置的数据
  tick();
}

/* ------------------------------------------------------------------ 曲线 */
function drawChart() {
  const cv = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = 220;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 38, r: 10, t: 10, b: 18 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const n = S.buf.length;

  // 网格
  ctx.strokeStyle = "rgba(139,160,192,.18)";
  ctx.fillStyle = "rgba(139,160,192,.75)";
  ctx.font = "10px Consolas, monospace";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ih * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText((1 - i / 4).toFixed(2), 6, y + 3);
  }
  if (n < 2) return;

  const X = (i) => pad.l + iw * i / Math.max(n - 1, 1);
  const Y = (p) => pad.t + ih * (1 - Math.max(0, Math.min(1, p)));

  // 真实攻击区间底色
  let start = -1;
  for (let i = 0; i < n; i++) {
    if (S.buf[i].attack && start < 0) start = i;
    if ((!S.buf[i].attack || i === n - 1) && start >= 0) {
      const end = S.buf[i].attack ? i : i - 1;
      ctx.fillStyle = "rgba(245,158,11,.16)";
      ctx.fillRect(X(start), pad.t, Math.max(X(end) - X(start), 1), ih);
      start = -1;
    }
  }
  // 告警区间底色
  start = -1;
  for (let i = 0; i < n; i++) {
    if (S.buf[i].alarm && start < 0) start = i;
    if ((!S.buf[i].alarm || i === n - 1) && start >= 0) {
      const end = S.buf[i].alarm ? i : i - 1;
      ctx.fillStyle = "rgba(239,68,68,.22)";
      ctx.fillRect(X(start), pad.t, Math.max(X(end) - X(start), 1), ih);
      start = -1;
    }
  }
  // 阈值线
  ctx.strokeStyle = "rgba(239,68,68,.85)";
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, Y(S.thr)); ctx.lineTo(w - pad.r, Y(S.thr)); ctx.stroke();
  ctx.setLineDash([]);

  // 概率曲线
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 1.6; ctx.beginPath();
  S.buf.forEach((d, i) => { const x = X(i), y = Y(d.p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = "rgba(56,189,248,.75)";
  ctx.fillText("p(异常)", pad.l + 6, pad.t + 11);
}

/* ------------------------------------------------------------------ 指标与图表 */
function renderMetrics(m) {
  if (!m) { $("metrics").innerHTML = "<div class='hint'>暂无指标（先运行 run_all.py）</div>"; return; }
  const mc = m.multiclass || {}, bn = m.binary || {}, on = m.online || {};
  const cards = [
    ["多分类准确率", pct(mc.accuracy)],
    ["多分类宏 F1", f4(mc.macro_f1)],
    ["Kappa", f4(mc.cohen_kappa)],
    ["二分类 F1", f4(bn.f1)],
    ["二分类 ROC-AUC", f4(bn.roc_auc)],
    ["TPR@1%FPR", f4(bn["tpr_at_fpr_0.01"])],
    ["攻击片段召回", on.segment_recall === undefined ? "-" : pct(on.segment_recall)],
    ["平均检测时延", on.mean_detection_delay_sec == null ? "-" : on.mean_detection_delay_sec.toFixed(1) + " s"],
    ["误报率", on.false_alarms_per_hour == null ? "-" : on.false_alarms_per_hour.toFixed(2) + " 次/时"],
    ["模型参数量", (m.model && m.model.n_params ? m.model.n_params.toLocaleString() : "-")],
  ];
  $("metrics").innerHTML = cards.map(([k, v]) =>
    `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  $("metrics-source").textContent = m.model && m.model.variant ? "模型：" + m.model.variant : "";
}

async function loadFigures() {
  const names = await fetchJSON("/api/figures");
  $("gallery").innerHTML = names.map((n) =>
    `<figure data-src="/figures/${n}"><img loading="lazy" src="/figures/${n}" alt="${n}" />
     <figcaption>${n.replace(/\.png$/, "")}</figcaption></figure>`).join("");
  $("gallery").querySelectorAll("figure").forEach((f) => {
    f.addEventListener("click", () => {
      $("overlay-img").src = f.dataset.src;
      $("overlay").classList.add("show");
    });
  });
}

$("overlay").addEventListener("click", () => $("overlay").classList.remove("show"));

/* ------------------------------------------------------------------ 交互 */
function togglePlay() {
  S.playing = !S.playing;
  $("btn-play").textContent = S.playing ? "暂停" : "继续";
  setStatus(S.playing ? "运行中" : "已暂停", S.playing ? "" : "paused");
}
function setStatus(text, cls) {
  $("status-text").textContent = text;
  $("status-pill").className = "pill " + (cls || "");
}

function fmtNum(v) {
  if (v === null || v === undefined || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(3);
}
const f4 = (v) => (v === undefined || v === null || isNaN(v) ? "-" : Number(v).toFixed(4));
const pct = (v) => (v === undefined || v === null || isNaN(v) ? "-" : (100 * v).toFixed(2) + "%");
const shortName = (f) => (f.length > 13 ? f.slice(0, 12) + "…" : f);

$("btn-play").addEventListener("click", togglePlay);
$("btn-step").addEventListener("click", () => {
  if (S.playing) togglePlay();
  tick();
});
$("btn-reset").addEventListener("click", () => {
  S.pos = 0; S.buf = []; S.tableRows = []; S.alarms = 0; S.falseAlarms = 0;
  S.attackRows = 0; S.decided = 0; S.correct = 0; S.lastAlarmWin = -1;
  $("alarm-log").innerHTML = "<div class='log-empty'>暂无告警</div>";
  HOSTS.forEach((h) => { h.attackUntil = 0; h.disposalUntil = 0; h.cls = ""; h.action = ""; });
  SOC.lastWin = null; SOC.attacker = null;
  renderStats(); drawChart();
  driftResetCursor();         // 游标回到第一格
  if (!S.playing) togglePlay();
});
$("speed").addEventListener("change", (e) => { S.speed = Number(e.target.value); });
$("threshold").addEventListener("input", (e) => {
  S.thr = Number(e.target.value);
  $("threshold-val").textContent = S.thr.toFixed(2);
  // 就地重算告警状态（不必重跑模型，前端只是把概率跟新阈值比一次）
  S.buf = S.buf.map((d) => ({ ...d, alarm: d.p >= S.thr }));
  S.alarms = S.buf.filter((d) => d.alarm).length;
  renderStats(); drawChart();
});
window.addEventListener("resize", () => { drawChart(); driftScheduleDraw(); });

/* ==================================================================================
   迷你 SOC 控制台
   ================================================================================== */

/* ---------------------------------------------------------------- 5) 复核台 */
/* 未知攻击与漏报的出口：待研判队列（未知形态告警）+ 观察池（差一点到阈值 / 形态新颖但未确认）。
   这两个接口是"未知 → 已知"闭环的入口；结论回流后与窗口特征按 review_id 关联即为训练样本。 */
const RV = { queue: [], obs: [] };

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = null;
  try { data = await r.json(); } catch (e) { data = null; }
  // 服务端对非法输入返回 400 + {ok:false,error}：把原因原样透出，不要吞成"未知错误"
  if (!r.ok || (data && data.ok === false)) {
    throw new Error((data && data.error) || (url + " → " + r.status));
  }
  return data;
}

function rvToast(msg, isErr) {
  const t = $("rv-toast");
  if (!t) return;
  t.textContent = msg;
  t.className = "toast show" + (isErr ? " err" : "");
  if (RV.timer) clearTimeout(RV.timer);
  RV.timer = setTimeout(() => { t.className = "toast" + (isErr ? " err" : ""); }, 4000);
}

function rvEsc(v) {
  return String(v === null || v === undefined ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function rvEvidence(ev) {
  // 证据只含定位字段（五元组/协议/标签）：首条做"源 → 目的"摘要，全部挂 title 供悬停查看
  const rows = Array.isArray(ev) ? ev : [];
  if (!rows.length) return { text: "-", title: "该条目没有随附流证据（证据条数可能被配成 0）" };
  const r0 = rows[0] || {};
  const src = r0["Src IP"] || r0["Source IP"] || "?";
  const dst = r0["Dst IP"] || r0["Destination IP"] || "?";
  const dp = r0["Dst Port"] || r0["Destination Port"];
  return {
    text: src + " → " + dst + (dp ? ":" + dp : ""),
    title: "窗口内 " + rows.length + " 条流证据：\n" + JSON.stringify(rows, null, 1).slice(0, 1200),
  };
}

function rvNum(v, nd) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(nd === undefined ? 4 : nd) : "-";
}

function fillReviewClasses() {
  const sel = $("rv-class");
  if (!sel || sel.options.length) return;
  const names = (SOC.stats && SOC.stats.class_names) || [];
  const list = names.length ? names
    : ["DDoS", "DoS", "Probe", "BruteForce", "WebAttack", "Botnet", "U2R"];
  sel.innerHTML = list.map((c) => '<option value="' + rvEsc(c) + '">' + rvEsc(c) + "</option>").join("");
}

function renderReview() {
  fillReviewClasses();
  const q = RV.queue || [], obs = RV.obs || [];
  const tb = $("rv-body");
  if (tb) {
    tb.innerHTML = q.length ? q.map((it) => {
      const ev = rvEvidence(it.evidence);
      const done = it.status === "labeled";
      const ops = done
        ? '<span class="hint">已回流：' + rvEsc(it.verdict || "") + "</span>"
        : '<button class="btn" data-act="attack" data-id="' + it.review_id + '">确认为攻击</button>' +
          '<button class="btn" data-act="normal" data-id="' + it.review_id + '">判为正常</button>';
      return "<tr>" +
        "<td class='l'>" + rvEsc(it.review_id) + "</td>" +
        "<td class='l'>" + rvEsc(it.window_index) + "</td>" +
        "<td class='l'>" + rvEsc(it.pred_class) + "</td>" +
        "<td>" + rvNum(it.novelty, 3) + "</td>" +
        "<td class='l' title=\"" + rvEsc(ev.title) + "\">" + rvEsc(ev.text) + "</td>" +
        "<td class='l'>" + rvEsc(it.action) + "</td>" +
        "<td class='l'>" + rvEsc(it.why) + "</td>" +
        "<td class='l'>" + (done ? "已研判" : "待研判") + "</td>" +
        "<td class='l'>" + ops + "</td>" +
        "</tr>";
    }).join("")
      : '<tr><td colspan="9" class="hint">暂无待研判项 —— 没有出现"未知形态"告警' +
        '（未知标记阈值默认为 0=关闭，需按标定 SOP 在安静时段标定后开启）</td></tr>';
  }
  const obTb = $("rv-obs-body");
  if (obTb) {
    obTb.innerHTML = obs.length ? obs.map((it, i) => {
      const ev = rvEvidence(it.evidence);
      return "<tr>" +
        "<td class='l'>" + rvEsc(it.window_index) + "</td>" +
        "<td>" + rvNum(it.p_attack, 4) + "</td>" +
        "<td class='l'>" + rvEsc(it.pred_class) + "</td>" +
        "<td>" + rvNum(it.novelty, 3) + "</td>" +
        "<td class='l' title=\"" + rvEsc(ev.title) + "\">" + rvEsc(ev.text) + "</td>" +
        "<td class='l'>" + rvEsc(it.why) + "</td>" +
        "<td class='l'><button class='btn' data-act='promote' data-idx='" + i + "'>提升进队列</button></td>" +
        "</tr>";
    }).join("") : '<tr><td colspan="7" class="hint">观察池为空</td></tr>';
  }
  const st = SOC.stats || {};
  const pick = (k) => (st[k] === undefined ? "-" : st[k]);
  const counts = $("rv-counts");
  if (counts) {
    counts.textContent = "待研判 " + pick("review_pending") + " · 已回流 " + pick("review_labeled") +
      " · 确认为攻击 " + pick("review_confirmed_attack") + " · 观察池 " + pick("observations") +
      " · 证据 " + pick("review_evidence_total") + " 条";
  }
  const meta = $("rv-meta");
  if (meta) meta.textContent = "队列 " + q.length + " 条（最多显示 50）";
  const om = $("rv-obs-meta");
  if (om) om.textContent = "观察池 " + obs.length + " 条（最多显示 50）";
}

async function pollReview() {
  try {
    const [q, ob] = await Promise.all([
      fetchJSON("/api/review?limit=50"),
      fetchJSON("/api/observations?limit=50"),
    ]);
    RV.queue = Array.isArray(q) ? q : [];
    RV.obs = Array.isArray(ob) ? ob : [];
    renderReview();
  } catch (e) {
    const meta = $("rv-meta");
    if (meta) meta.textContent = "复核接口不可用：" + e.message;
  }
}

async function submitVerdict(id, verdict) {
  try {
    await postJSON("/api/review", { review_id: id, verdict: verdict });
    rvToast("已回流：" + verdict + "（#" + id + "）", false);
    pollReview();
  } catch (e) {
    rvToast("提交失败：" + e.message, true);
  }
}

async function promoteObservation(index) {
  try {
    const res = await postJSON("/api/review/promote", { index: index });
    const rid = res && res.item ? res.item.review_id : "?";
    rvToast("已提升进待研判队列（#" + rid + "）", false);
    pollReview();
  } catch (e) {
    rvToast("提升失败：" + e.message, true);
  }
}

function wireReview() {
  const tb = $("rv-body");
  if (tb) {
    tb.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-act]");
      if (!btn) return;
      const id = Number(btn.dataset.id);
      if (btn.dataset.act === "normal") submitVerdict(id, "normal");
      else if (btn.dataset.act === "attack") {
        const sel = $("rv-class");
        submitVerdict(id, "attack:" + ((sel && sel.value) ? sel.value : "Unknown"));
      }
    });
  }
  const obTb = $("rv-obs-body");
  if (obTb) {
    obTb.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-act='promote']");
      if (btn) promoteObservation(Number(btn.dataset.idx));
    });
  }
  const rf = $("btn-rv-refresh");
  if (rf) rf.addEventListener("click", () => { pollReview(); rvToast("已刷新", false); });
}

/* ---------------------------------------------------------------- 启动控制台 */
function bootSocConsole() {
  wireConsole();
  wireReview();
  startTopoLoop();
  loadPolicy();
  pollStatus();
  pollAlerts();
  pollReview();
  setInterval(pollStatus, STATUS_REFRESH_MS);
  setInterval(pollAlerts, LEDGER_REFRESH_MS);
  setInterval(pollReview, REVIEW_REFRESH_MS);
}

/* ---------------------------------------------------------------- 3) 服务健康状态条 */
async function pollStatus() {
  let snap = null, down = false, reason = "";
  try {
    snap = await fetchJSON("/api/stats");
    SOC.stats = snap;
    SOC.detector.ready = !!snap.ready;
    reason = snap.reason || "";
  } catch (e) {
    down = true;
    SOC.detector.ready = false;
    reason = "";
  }
  // /api/health 也带检测服务状态（新增字段，不影响既有 ok/reason）；
  // 它失败不影响状态条：只作为交叉校验。
  try {
    const hp = await fetchJSON("/api/health");
    if (typeof hp.detector_ready === "boolean") {
      SOC.detector.ready = hp.detector_ready;
      if (!hp.detector_ready && hp.detector_reason) reason = hp.detector_reason;
    }
  } catch (e) { /* 忽略：健康检查不可用不该阻塞界面 */ }
  SOC.detector.down = down;
  SOC.detector.reason = reason || "";
  SOC.mode = SOC.detector.ready ? "online" : "replay";
  renderStatusBar();
  renderLedger();
  renderPolicyEcho();
}

function renderStatusBar() {
  const det = SOC.detector;
  const box = $("sb-detector");
  const val = $("sb-detector-v");
  if (!box || !val) return;
  if (det.down) {
    box.className = "sb bad";
    val.textContent = "检测服务未启动（在线接口不可用）";
  } else if (det.ready) {
    box.className = "sb ok";
    val.textContent = "就绪 · " + ((SOC.stats && SOC.stats.infer_path) || "在线");
  } else {
    box.className = "sb warn";
    val.textContent = "未就绪：" + (det.reason || "缺少模型或预处理产物");
  }
  const st = SOC.stats || {};
  const lat = st.batch_latency_ms || {};
  $("sb-rows").textContent = st.rows === undefined ? "-" : Number(st.rows).toLocaleString();
  $("sb-windows").textContent = st.windows === undefined ? "-" : Number(st.windows).toLocaleString();
  $("sb-alarms").textContent = st.alarms_confirmed === undefined ? "-" : Number(st.alarms_confirmed).toLocaleString();
  $("sb-latency").textContent = lat.p50 === undefined ? "-" : Number(lat.p50).toFixed(1) + " ms";
  const f = SOC.feed;
  if (f.off) $("sb-feed").textContent = "已停止（" + (f.lastError || "在线接口不可用") + "）";
  else if (!f.enabled) $("sb-feed").textContent = "已关闭";
  else if (!det.ready) $("sb-feed").textContent = "不可用（服务未就绪）";
  else $("sb-feed").textContent = "已推送 " + f.sent.toLocaleString() + " 行" + (f.sampled > 0 ? "（抽样 " + f.sampled + "）" : "");
}

/* ---------------------------------------------------------------- 2) 处置控制台 */
function wireConsole() {
  const sl = $("p-threshold");
  if (sl) sl.addEventListener("input", () => { $("p-threshold-val").textContent = Number(sl.value).toFixed(2); });
  const btn = $("btn-apply-policy");
  if (btn) btn.addEventListener("click", applyPolicy);
  const feed = $("p-feed");
  if (feed) feed.addEventListener("change", () => {
    SOC.feed.enabled = feed.checked;
    SOC.feed.off = false;
    SOC.feed.failures = 0;
    renderStatusBar();
  });
  const dry = $("p-dryrun");
  if (dry) dry.addEventListener("change", () => { /* 真正生效要按「应用策略」 */ });
}

function parseWhitelist(txt) {
  return String(txt || "").split(/[,，\s;；]+/).map((s) => s.trim()).filter(Boolean);
}

async function loadPolicy() {
  try {
    const p = await fetchJSON("/api/policy");
    if (p && typeof p === "object" && "threshold" in p) {
      SOC.policy = Object.assign({}, SOC.policy, p);
      fillPolicyForm(SOC.policy);
      renderPolicyEcho();
    }
  } catch (e) {
    setPolicyBadge("读取策略失败", "bad");
  }
}

function fillPolicyForm(p) {
  if ($("p-threshold")) { $("p-threshold").value = Number(p.threshold).toFixed(2); $("p-threshold-val").textContent = Number(p.threshold).toFixed(2); }
  if ($("p-confirm")) $("p-confirm").value = String(p.confirm_windows);
  if ($("p-action")) $("p-action").value = String(p.action);
  if ($("p-dryrun")) $("p-dryrun").checked = !!p.dry_run;
}

function setPolicyBadge(text, cls) {
  const b = $("policy-state");
  if (!b) return;
  b.textContent = text;
  b.className = "badge " + (cls || "");
}

function renderPolicyEcho(extra) {
  const box = $("policy-echo-box");
  if (!box) return;
  const p = SOC.policy;
  const wl = SOC.whitelist.length ? SOC.whitelist.join(", ") : "（空）";
  box.innerHTML =
    `服务端回显策略：阈值 <b>${Number(p.threshold).toFixed(2)}</b> · 连续确认 <b>${p.confirm_windows}</b> 窗口 · ` +
    `动作 <b>${p.action}</b> · 演练模式 <b>${p.dry_run ? "开启（只出建议）" : "关闭（真正下发）"}</b><br>` +
    `白名单（本页已提交）：<b>${wl}</b>`;
  if (extra) box.innerHTML += "<br>" + extra;
  if (SOC.detector.ready === false) setPolicyBadge("在线服务未就绪", "warn");
  else setPolicyBadge(p.dry_run ? "演练模式" : "已启用下发", p.dry_run ? "warn" : "ok");
}

async function applyPolicy() {
  const body = {
    threshold: Number($("p-threshold").value),
    confirm_windows: Number($("p-confirm").value),
    action: $("p-action").value,
    whitelist: parseWhitelist($("p-whitelist").value),
    dry_run: $("p-dryrun").checked,   // 必须是真布尔：服务端对 "false" 字符串会判为 True
  };
  try {
    const r = await fetch("/api/policy", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error("/api/policy → " + r.status);
    const pol = await r.json();
    if (!pol || typeof pol !== "object" || !("threshold" in pol)) throw new Error("返回内容异常");
    SOC.policy = Object.assign({}, SOC.policy, pol);
    SOC.whitelist = body.whitelist;
    fillPolicyForm(SOC.policy);
    renderPolicyEcho(`下发时间 <b>${new Date().toLocaleTimeString()}</b>`);
    toast("✓ 策略已生效（服务端回显一致）", false);
  } catch (e) {
    toast("✗ 策略未生效：" + (e.message || e), true);
    setPolicyBadge("下发失败", "bad");
  }
}

function toast(msg, isErr) {
  const t = $("policy-toast");
  if (!t) return;
  t.textContent = msg;
  t.className = "toast show" + (isErr ? " err" : "");
  if (SOC.toastTimer) clearTimeout(SOC.toastTimer);
  SOC.toastTimer = setTimeout(() => { t.className = "toast" + (isErr ? " err" : ""); }, 4000);
}

/* ---------------------------------------------------------------- 台账（在线 / 回放兜底） */
function ledgerKey(a) {
  return [a.time, a.window_index, a.src_ip, a.pred_class, a.action].join("|");
}

async function pollAlerts() {
  try {
    const list = await fetchJSON("/api/alerts?limit=20");
    SOC.online = (Array.isArray(list) ? list : []).map((a) => Object.assign({}, a, { _src: "online" }));
    SOC.online.forEach((a) => SOC.seen.add(ledgerKey(a)));
    SOC.lastLedgerAt = Date.now();
    // 反转成「由旧到新」再打标，保证最后落笔的标记属于最新一条告警
    applyAlertsToTopo(SOC.online.slice(0, 6).slice().reverse());
  } catch (e) {
    // 在线台账取不到：保留已有内容，界面靠状态条提示服务状态
  }
  renderLedger();
}

function ingestAlerts(list) {
  if (!Array.isArray(list) || !list.length) return;
  let added = 0;
  list.forEach((a) => {
    const k = ledgerKey(a);
    if (SOC.seen.has(k)) return;
    SOC.seen.add(k);
    SOC.online.unshift(Object.assign({}, a, { _src: "online", _fresh: true }));
    added++;
  });
  if (!added) return;
  SOC.online = SOC.online.slice(0, 60);
  applyAlertsToTopo(list);
  renderLedger();
}

/** 回放兜底台账：在线检测服务不可用时，用回放的真实判定显示「谁、何时、依据什么」，
 *  只是没有「处置执行」信息（那必须由在线服务给出），因此来源列会明确标注。 */
function pushLocalLedger(row) {
  const ip = CLASS_TO_IP[row.win.pred] || DEFAULT_ATTACK_IP;
  const p = SOC.policy;
  SOC.local.unshift({
    time: Date.now() / 1000,
    window_index: row.win.idx,
    p_attack: row.win.p,
    pred_class: row.win.pred,
    src_ip: ip,
    dst_ip: "10.0.0.1",
    dst_port: 80,
    action: p.action,
    executed: !p.dry_run,
    rows_covered: [Math.max(0, row.i - ((S.overview && S.overview.window_size) || 16) + 1), row.i],
    ground_truth: row.win.true || row.true || null,
    _src: "replay",
  });
  SOC.local = SOC.local.slice(0, 60);
  if (SOC.mode !== "online") renderLedger();
}

function fmtClock(sec) {
  try {
    return new Date(Number(sec) * 1000).toLocaleTimeString();
  } catch (e) { return "-"; }
}

function renderLedger() {
  const tbody = $("ledger-body");
  if (!tbody) return;
  const online = SOC.mode === "online";
  const rows = (online ? SOC.online : SOC.local).slice(0, 20);
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="l" style="color:#8ba0c0;padding:12px">
      ${online
        ? "暂无告警。点击左侧「注入 DDoS / DoS / …」后，在线检测服务的确认告警会实时出现在这里。"
        : "在线检测服务未就绪：台账将在服务可用后自动切换为服务端审计记录。"}</td></tr>`;
  } else {
    tbody.innerHTML = rows.map((a) => {
      const act = String(a.action || "-");
      const cls = act === "block" ? "block" : (act === "throttle" ? "throttle" : "observe");
      const conf = Number(a.p_attack);
      const bar = clamp(conf * 100, 0, 100).toFixed(0);
      const rc = Array.isArray(a.rows_covered) && a.rows_covered.length === 2
        ? `第 ${a.rows_covered[0]}~${a.rows_covered[1]} 条流` : "—";
      const src = a._src === "replay"
        ? '<span class="chip replay">回放兜底</span>'
        : '<span class="chip online">在线服务</span>';
      const exec = a.executed
        ? '<span class="chip exec">✔ 已执行</span>'
        : '<span class="chip dry">演练·未下发</span>';
      return `<tr class="${a._fresh ? "fresh" : ""}">
        <td class="l mono">${fmtClock(a.time)}</td>
        <td class="l mono">${a.src_ip || "—"}</td>
        <td class="l">${a.pred_class || "—"}</td>
        <td title="p(异常)=${isFinite(conf) ? conf.toFixed(4) : "-"}">
          <span class="mini-bar"><i style="width:${bar}%"></i></span>${isFinite(conf) ? conf.toFixed(3) : "-"}</td>
        <td class="l"><span class="chip ${cls}">${act}</span></td>
        <td class="l">${exec}</td>
        <td class="l mono" title="触发窗口 #${a.window_index} 覆盖 ${rc}">#${a.window_index} <span style="color:#8ba0c0">(${rc})</span></td>
        <td class="l">${src}</td></tr>`;
    }).join("");
  }
  const meta = $("ledger-meta");
  if (meta) {
    const src = online ? "在线检测服务（/api/alerts）" : "回放兜底（在线检测服务未就绪）";
    meta.textContent = `来源：${src} · 共 ${rows.length} 条 · 更新于 ${new Date().toLocaleTimeString()}`;
  }
  const note = $("ledger-note");
  if (note) {
    note.innerHTML = online
      ? "台账来自在线检测服务的审计记录（<code>/api/alerts</code>）：每条含时间、源 IP、判定类别、置信度、处置动作、是否执行与触发窗口（服务端窗口序号 + 其覆盖的流记录区间），可逐条回溯。"
      : "在线检测服务未就绪（<code>/api/detect</code> 不可用）：下表为<b>回放判定兜底</b>，判定类别与置信度仍来自模型真实预测，但「处置动作/是否执行」取自本页策略设置，仅供参考。";
  }
  // 清掉一次性高亮标记
  SOC.online.forEach((a) => { a._fresh = false; });
}

/* ---------------------------------------------------------------- 1) 攻击拓扑动画 */
function hostIndexForIP(ip) {
  const m = /(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})/.exec(String(ip || ""));
  if (!m) return HOST_DEFAULT;              // 拿不到 IP：落到演示攻击源 h5
  const last = Number(m[4]);
  if (!last || last > HOSTS.length) return HOST_DEFAULT;  // 超出主机范围：循环用 h5
  return last - 1;
}

/** 真实窗口判定 → 主机着色（攻击源变红并脉冲） */
function topoNoteFromRows(rows) {
  let lastWin = null;
  for (const r of rows) {
    if (!r.win) continue;
    lastWin = { cls: r.win.pred, p: r.win.p, alarm: r.win.alarm, idx: r.win.idx, i: r.i, truth: r.true };
    if (!r.win.alarm || r.win.pred === "Normal") continue;
    const ip = CLASS_TO_IP[r.win.pred] || DEFAULT_ATTACK_IP;
    const h = HOSTS[hostIndexForIP(ip)];
    h.cls = r.win.pred;
    h.p = r.win.p;
    h.srcIP = h.ip;
    h.win = r.win.idx;
    h.attackUntil = nowMs() + ATTACK_HOLD_MS;
    SOC.attacker = { host: h.id, cls: r.win.pred, p: r.win.p, ip: h.ip, win: r.win.idx };
    SOC.attackerIP = h.ip;
  }
  if (lastWin) SOC.lastWin = lastWin;
}

/** 真实告警（含处置动作）→ 主机标记（阻断/限速/仅观察） */
function applyAlertsToTopo(list) {
  let touched = false;
  for (const a of list) {
    if (!a || !a.src_ip) continue;
    const h = HOSTS[hostIndexForIP(a.src_ip)];
    h.action = String(a.action || "observe");
    h.executed = !!a.executed;
    h.cls = a.pred_class || h.cls;
    h.p = Number(a.p_attack) || h.p;
    h.srcIP = a.src_ip;
    h.disposalUntil = nowMs() + DISPOSAL_HOLD_MS;
    if (!h.attackUntil) h.attackUntil = nowMs() + ATTACK_HOLD_MS;
    SOC.attacker = { host: h.id, cls: h.cls, p: h.p, ip: h.ip, win: a.window_index };
    touched = true;
  }
  return touched;
}

const nowMs = () => (window.performance && performance.now ? performance.now() : Date.now());

const TOPO = { t: 0, phase: {}, last: 0 };

function startTopoLoop() {
  const loop = (ts) => {
    requestAnimationFrame(loop);
    if (document.hidden) { TOPO.last = ts; return; }
    if (ts - TOPO.last < TOPO_FRAME_MS) return;      // 限帧：约 30fps，空转 CPU 占用低
    const dt = Math.min((ts - TOPO.last) / 1000, 0.12);
    TOPO.last = ts;
    try { drawTopo(dt); } catch (e) { /* 拓扑绘制异常不能影响主流程 */ }
  };
  requestAnimationFrame(loop);
}

function topoBuild(w, h) {
  const ctrl = { x: w / 2, y: 44, w: clamp(w * 0.27, 208, 330), h: 64 };
  const sw = [
    { x: w * 0.28, y: h * 0.50, w: 118, h: 42, name: "交换机 s1" },
    { x: w * 0.72, y: h * 0.50, w: 118, h: 42, name: "交换机 s2" },
  ];
  const hostW = 78, hostH = 52, hostY = h - 40 - hostH / 2;
  const spread = clamp(w * 0.085, 80, 128);
  const xs = [sw[0].x - spread, sw[0].x, sw[0].x + spread, sw[1].x - spread, sw[1].x, sw[1].x + spread];
  HOSTS.forEach((host, i) => {
    host.w = hostW; host.h = hostH;
    host.x = clamp(xs[i], hostW / 2 + 8, w - hostW / 2 - 8);
    host.y = hostY;
  });
  const busY = (ctrl.y + ctrl.h / 2 + sw[0].y - sw[0].h / 2) / 2;
  return { ctrl, sw, busY, hostW, hostH };
}

function polyLen(pts) {
  const segs = [];
  let total = 0;
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i][0] - pts[i - 1][0], dy = pts[i][1] - pts[i - 1][1];
    const l = Math.hypot(dx, dy) || 1e-6;
    segs.push(l); total += l;
  }
  return { segs, total };
}

function pointAt(pts, geo, t) {
  let d = clamp(t, 0, 1) * geo.total;
  for (let i = 0; i < geo.segs.length; i++) {
    if (d <= geo.segs[i]) {
      const k = d / geo.segs[i];
      const x0 = pts[i][0], y0 = pts[i][1], x1 = pts[i + 1][0], y1 = pts[i + 1][1];
      return { x: x0 + (x1 - x0) * k, y: y0 + (y1 - y0) * k, ang: Math.atan2(y1 - y0, x1 - x0) };
    }
    d -= geo.segs[i];
  }
  const n = pts.length - 1;
  return { x: pts[n][0], y: pts[n][1], ang: 0 };
}

function strokePath(ctx, pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}

function drawTopo(dt) {
  const cv = $("topo");
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(340, cv.clientWidth || 900);
  const h = 340;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  TOPO.t += dt;
  const T = nowMs();
  const playing = S.playing;

  const G = topoBuild(w, h);
  const { ctrl, sw, busY } = G;

  // 背景光晕（控制器为中心）
  const gl = ctx.createRadialGradient(ctrl.x, sw[0].y - 10, 10, ctrl.x, sw[0].y - 10, w * 0.6);
  gl.addColorStop(0, "rgba(56,189,248,.07)");
  gl.addColorStop(1, "rgba(56,189,248,0)");
  ctx.fillStyle = gl;
  ctx.fillRect(0, 0, w, h);

  // ---------------- 主机状态汇总 ----------------
  const swAtk = [false, false];
  HOSTS.forEach((host) => {
    if (T < host.attackUntil) swAtk[host.sw] = true;
  });
  HOSTS.forEach((host) => {
    // 暂停时冻结衰减，方便停下来讲解现场状态：只保留「最后一台攻击源」，其余按正常节奏回落
    if (!playing && host.disposalUntil > 0) host.disposalUntil = T + DISPOSAL_HOLD_MS;
    if (!playing && SOC.attacker && SOC.attacker.host === host.id && host.attackUntil > 0) {
      host.attackUntil = T + ATTACK_HOLD_MS;
    }
  });

  // ---------------- 连线 ----------------
  const links = [];
  sw.forEach((s, i) => {
    const pts = [[s.x, s.y - s.h / 2], [s.x, busY], [ctrl.x, busY], [ctrl.x, ctrl.y + ctrl.h / 2]];
    links.push({ id: "c-s" + i, pts, geo: polyLen(pts), kind: "trunk", sw: i });
  });
  HOSTS.forEach((host, i) => {
    const s = sw[host.sw];
    const midY = (host.y - host.h / 2 + s.y + s.h / 2) / 2;
    const pts = [[host.x, host.y - host.h / 2], [host.x, midY], [s.x, midY], [s.x, s.y + s.h / 2]];
    links.push({ id: "h" + i, pts, geo: polyLen(pts), kind: "host", host: i });
  });

  // 攻击态路径（威胁光带）
  const attacker = SOC.attacker ? HOSTS.find((x) => x.id === SOC.attacker.host) : null;
  const attacking = attacker && (T < attacker.attackUntil);
  if (attacking) {
    const hostLink = links.find((l) => l.kind === "host" && l.host === HOSTS.indexOf(attacker));
    const trunk = links.find((l) => l.kind === "trunk" && l.sw === attacker.sw);
    ctx.save();
    ctx.lineWidth = 7;
    ctx.strokeStyle = "rgba(239,68,68,.22)";
    ctx.shadowColor = "rgba(239,68,68,.55)";
    ctx.shadowBlur = 14;
    if (hostLink) strokePath(ctx, hostLink.pts);
    if (trunk) strokePath(ctx, trunk.pts);
    ctx.restore();
    ctx.save();
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = "rgba(252,165,165,.95)";
    ctx.setLineDash([9, 7]);
    ctx.lineDashOffset = -(TOPO.t * 70) % 1000;
    if (hostLink) strokePath(ctx, hostLink.pts);
    if (trunk) strokePath(ctx, trunk.pts);
    ctx.restore();
  }

  // 普通连线
  ctx.lineWidth = 1.4;
  links.forEach((l) => {
    const hot = l.kind === "trunk" ? swAtk[l.sw] : (T < HOSTS[l.host].attackUntil);
    ctx.strokeStyle = hot ? "rgba(248,113,113,.45)" : "rgba(139,160,192,.32)";
    strokePath(ctx, l.pts);
  });

  // ---------------- 粒子（流速/密度随状态变化） ----------------
  links.forEach((l) => {
    const hot = l.kind === "trunk" ? swAtk[l.sw] : (T < HOSTS[l.host].attackUntil);
    const blocked = l.kind === "host" && HOSTS[l.host].disposalUntil > T && HOSTS[l.host].action === "block" && HOSTS[l.host].executed;
    const n = hot ? (l.kind === "trunk" ? 14 : 9) : (l.kind === "trunk" ? 5 : 3);
    const spd = hot ? 0.55 : 0.11;
    let ph = TOPO.phase[l.id] || 0;
    ph = (ph + spd * dt) % 1;
    TOPO.phase[l.id] = ph;
    if (blocked) return;                       // 已阻断：该链路不再放行流量
    for (let i = 0; i < n; i++) {
      const t = (ph + i / n) % 1;
      const pt = pointAt(l.pts, l.geo, t);
      const a = 0.35 + 0.45 * Math.sin(Math.PI * t);
      ctx.beginPath();
      ctx.fillStyle = hot ? `rgba(255,120,120,${a})` : `rgba(125,211,252,${a})`;
      ctx.arc(pt.x, pt.y, hot ? 2.4 : 1.7, 0, Math.PI * 2);
      ctx.fill();
      if (hot) {
        ctx.beginPath();
        ctx.fillStyle = "rgba(239,68,68,.18)";
        ctx.arc(pt.x, pt.y, 5.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });

  // 被阻断链路：在链路上画红色禁止标记
  HOSTS.forEach((host, i) => {
    const blocked = host.disposalUntil > T && host.action === "block" && host.executed;
    if (!blocked) return;
    const l = links.find((x) => x.kind === "host" && x.host === i);
    if (!l) return;
    for (const t of [0.35, 0.62]) {
      const pt = pointAt(l.pts, l.geo, t);
      drawProhibit(ctx, pt.x, pt.y, 7, "rgba(239,68,68,.85)", 2);
    }
  });

  // ---------------- 控制器节点 ----------------
  drawCtrlNode(ctx, ctrl, T);

  // ---------------- 交换机节点 ----------------
  sw.forEach((s, i) => {
    const hot = swAtk[i];
    ctx.save();
    roundRect(ctx, s.x - s.w / 2, s.y - s.h / 2, s.w, s.h, 9);
    ctx.fillStyle = hot ? "rgba(239,68,68,.16)" : "rgba(30,45,74,.85)";
    ctx.fill();
    ctx.lineWidth = hot ? 2 : 1.4;
    ctx.strokeStyle = hot ? "#ef4444" : "#3b82f6";
    ctx.stroke();
    ctx.restore();
    for (let k = 0; k < 4; k++) {   // 端口刻线
      const px = s.x - s.w / 2 + 14 + k * ((s.w - 28) / 3);
      ctx.beginPath();
      ctx.moveTo(px, s.y + s.h / 2);
      ctx.lineTo(px, s.y + s.h / 2 + 5);
      ctx.strokeStyle = hot ? "rgba(248,113,113,.8)" : "rgba(125,211,252,.6)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.font = `600 12px ${FONT_CN}`;
    ctx.fillStyle = hot ? "#fecaca" : "#bfdbfe";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(s.name, s.x, s.y - 3);
    ctx.font = `10px ${FONT_CN}`;
    ctx.fillStyle = "rgba(139,160,192,.9)";
    ctx.fillText("OpenFlow 1.3", s.x, s.y + 11);
    // 攻击计数徽标
    const cnt = HOSTS.filter((x) => x.sw === i && (T < x.attackUntil)).length;
    if (cnt > 0) {
      ctx.beginPath();
      ctx.fillStyle = "#ef4444";
      ctx.arc(s.x + s.w / 2 - 6, s.y - s.h / 2 + 2, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = `600 11px ${FONT_MONO}`;
      ctx.fillText(String(cnt), s.x + s.w / 2 - 6, s.y - s.h / 2 + 3);
    }
  });

  // ---------------- 主机节点 ----------------
  HOSTS.forEach((host, i) => {
    const atk = T < host.attackUntil;
    const disp = host.disposalUntil > T;
    const blocked = disp && host.action === "block" && host.executed;
    const suggestBlock = disp && host.action === "block" && !host.executed;
    const throttle = disp && host.action === "throttle";
    const observe = disp && host.action === "observe";
    const white = SOC.whitelist.indexOf(host.ip) >= 0;
    const x = host.x - host.w / 2, y = host.y - host.h / 2;

    // 脉冲光圈
    if (atk) {
      const phase = (TOPO.t % 1.4) / 1.4;
      for (const off of [0, 0.5]) {
        const ph = (phase + off) % 1;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(239,68,68,${(1 - ph) * 0.55})`;
        ctx.lineWidth = 2;
        roundRect(ctx, x - ph * 20, y - ph * 16, host.w + ph * 40, host.h + ph * 32, 12);
        ctx.stroke();
      }
    }
    // 节点主体
    ctx.save();
    if (atk) { ctx.shadowColor = "rgba(239,68,68,.75)"; ctx.shadowBlur = 18; }
    else if (white) { ctx.shadowColor = "rgba(129,140,248,.5)"; ctx.shadowBlur = 10; }
    roundRect(ctx, x, y, host.w, host.h, 10);
    const g = ctx.createLinearGradient(x, y, x, y + host.h);
    if (blocked || suggestBlock) { g.addColorStop(0, "rgba(71,85,105,.95)"); g.addColorStop(1, "rgba(51,65,85,.95)"); }
    else if (atk) { g.addColorStop(0, "rgba(127,29,29,.95)"); g.addColorStop(1, "rgba(69,10,10,.95)"); }
    else { g.addColorStop(0, "rgba(13,37,64,.95)"); g.addColorStop(1, "rgba(12,24,44,.95)"); }
    ctx.fillStyle = g;
    ctx.fill();
    ctx.restore();
    ctx.lineWidth = atk ? 2.2 : 1.4;
    ctx.strokeStyle = blocked || suggestBlock ? "#94a3b8"
      : atk ? "#ef4444" : (throttle ? "#f59e0b" : (white ? "#818cf8" : "#38bdf8"));
    if (suggestBlock) ctx.setLineDash([5, 4]);
    roundRect(ctx, x, y, host.w, host.h, 10);
    ctx.stroke();
    ctx.setLineDash([]);

    // 主机名 + IP
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `600 13px ${FONT_CN}`;
    ctx.fillStyle = blocked ? "#cbd5e1" : (atk ? "#fecaca" : "#e0f2fe");
    ctx.fillText(host.id, host.x, host.y - 9);
    ctx.font = `10.5px ${FONT_MONO}`;
    ctx.fillStyle = blocked ? "#94a3b8" : "rgba(186,230,253,.9)";
    ctx.fillText(host.ip, host.x, host.y + 6);
    ctx.font = `9.5px ${FONT_CN}`;
    ctx.fillStyle = "rgba(139,160,192,.85)";
    ctx.fillText(atk ? "攻击源" : host.role, host.x, host.y + 19);

    // 处置标记
    if (blocked) {
      drawProhibit(ctx, x + host.w - 2, y + 3, 13, "#ef4444", 3);
      drawTag(ctx, host.x, y + host.h + 11, "已阻断", "rgba(239,68,68,.9)", "#fff");
    } else if (suggestBlock) {
      drawProhibit(ctx, x + host.w - 2, y + 3, 13, "rgba(248,113,113,.85)", 2);
      drawTag(ctx, host.x, y + host.h + 11, "建议阻断·演练", "rgba(120,53,15,.9)", "#fed7aa");
    } else if (throttle) {
      ctx.beginPath();
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 3;
      ctx.arc(x + host.w - 4, y + 4, 9, 0, Math.PI * 2);
      ctx.stroke();
      drawTag(ctx, host.x, y + host.h + 11, "限速中", "rgba(120,53,15,.9)", "#fde68a");
    } else if (white) {
      if (observe) drawBang(ctx, x + host.w - 4, y + 4);
      drawTag(ctx, host.x, y + host.h + 11, "白名单·不处置", "rgba(49,46,129,.9)", "#c7d2fe");
    } else if (observe) {
      drawBang(ctx, x + host.w - 4, y + 4);
      drawTag(ctx, host.x, y + host.h + 11, "仅观察", "rgba(124,45,18,.9)", "#fed7aa");
    }

    // 攻击浮标（判定类别 + 概率）
    if (atk) {
      const label = `${host.cls || "Attack"} · p=${Number(host.p || 0).toFixed(3)}`;
      ctx.font = `600 11.5px ${FONT_CN}`;
      const tw = ctx.measureText(label).width + 16;
      const bx = clamp(host.x - tw / 2, 6, w - tw - 6);
      const by = y - 26;
      ctx.save();
      ctx.shadowColor = "rgba(239,68,68,.6)";
      ctx.shadowBlur = 12;
      roundRect(ctx, bx, by, tw, 19, 9);
      ctx.fillStyle = "rgba(127,29,29,.96)";
      ctx.fill();
      ctx.restore();
      roundRect(ctx, bx, by, tw, 19, 9);
      ctx.strokeStyle = "rgba(252,165,165,.9)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#fee2e2";
      ctx.font = `600 11.5px ${FONT_CN}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, bx + tw / 2, by + 10);
    }
  });

  drawTopoHint(attacking ? attacker : null, T);
}

function drawCtrlNode(ctx, ctrl, T) {
  const p = SOC.policy;
  const x = ctrl.x - ctrl.w / 2, y = ctrl.y - ctrl.h / 2;
  ctx.save();
  ctx.shadowColor = "rgba(56,189,248,.45)";
  ctx.shadowBlur = 16;
  roundRect(ctx, x, y, ctrl.w, ctrl.h, 12);
  const g = ctx.createLinearGradient(x, y, x, y + ctrl.h);
  g.addColorStop(0, "rgba(12,74,110,.95)");
  g.addColorStop(1, "rgba(8,47,73,.95)");
  ctx.fillStyle = g;
  ctx.fill();
  ctx.restore();
  roundRect(ctx, x, y, ctrl.w, ctrl.h, 12);
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `600 13.5px ${FONT_CN}`;
  ctx.fillStyle = "#e0f2fe";
  ctx.fillText("控制器 Ryu", ctrl.x, y + 15);

  const wl = SOC.whitelist.length;
  ctx.font = `10.5px ${FONT_MONO}`;
  ctx.fillStyle = "rgba(186,230,253,.95)";
  ctx.fillText(`阈值 ${Number(p.threshold).toFixed(2)} · 确认 ${p.confirm_windows} 窗口 · ${wl} 条白名单`, ctrl.x, y + 34);

  const act = String(p.action);
  const actColor = act === "block" ? "#fca5a5" : (act === "throttle" ? "#fcd34d" : "#cbd5e1");
  ctx.font = `600 11px ${FONT_CN}`;
  ctx.fillStyle = actColor;
  ctx.fillText(`策略动作 ${act}${p.dry_run ? " · 演练模式（dry_run）" : " · 已启用真实下发"}`, ctrl.x, y + 51);

  // 状态点（就绪 / 未就绪）
  const det = SOC.detector;
  const dc = det.down ? "#ef4444" : (det.ready ? "#22c55e" : "#f59e0b");
  ctx.beginPath();
  ctx.fillStyle = dc;
  ctx.shadowColor = dc;
  ctx.shadowBlur = 10;
  ctx.arc(x + 12, y + 12, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
}

function drawProhibit(ctx, cx, cy, r, color, lw) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(15,23,42,.85)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.stroke();
  ctx.beginPath();
  const k = r * 0.7071;
  ctx.moveTo(cx - k, cy + k);
  ctx.lineTo(cx + k, cy - k);
  ctx.stroke();
  ctx.restore();
}

function drawBang(ctx, cx, cy) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, 10, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(251,146,60,.95)";
  ctx.fill();
  ctx.fillStyle = "#1f2937";
  ctx.font = `700 13px ${FONT_MONO}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("!", cx, cy + 0.5);
  ctx.restore();
}

function drawTag(ctx, cx, cy, text, bg, fg) {
  ctx.save();
  ctx.font = `600 10.5px ${FONT_CN}`;
  const tw = ctx.measureText(text).width + 14;
  roundRect(ctx, cx - tw / 2, cy - 8, tw, 16, 8);
  ctx.fillStyle = bg;
  ctx.fill();
  ctx.fillStyle = fg;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, cx, cy + 1);
  ctx.restore();
}

function drawTopoHint(attacker, T) {
  const el = $("topo-hint");
  if (!el) return;
  const w = SOC.lastWin;
  const p = SOC.policy;
  const parts = [];
  parts.push(`数据来源：<code>/api/stream</code> 的窗口判定给主机着色，<code>/api/alerts</code> 的处置动作给节点打标`);
  if (!w) {
    parts.push("等待回放数据…");
  } else {
    const st = w.alarm ? "告警" : "正常";
    parts.push(`当前窗口 #${w.idx}：${st}（${w.cls}，p=${Number(w.p).toFixed(3)}）`);
  }
  if (attacker) {
    parts.push(`攻击源 <b>${attacker.ip}</b>（${attacker.id}）判定 ${attacker.cls} p=${Number(attacker.p).toFixed(3)}`);
  }
  // 处置摘要优先显示「正在攻击的那台主机」，避免被别的历史标记抢占
  let acted = (attacker ? HOSTS[hostIndexForIP(attacker.ip)] : null);
  if (!acted || acted.disposalUntil <= nowMs()) {
    const marked = HOSTS.filter((h) => h.disposalUntil > nowMs());
    acted = marked.sort((a, b) => b.disposalUntil - a.disposalUntil)[0];
  }
  if (acted) {
    const ex = acted.action === "block" ? (acted.executed ? "已阻断" : "建议阻断（演练）")
      : acted.action === "throttle" ? "限速" : "仅观察";
    parts.push(`处置：<b>${acted.action}</b> → ${ex}（源 ${acted.srcIP || acted.ip}）`);
  }
  parts.push(`当前策略：阈值 ${Number(p.threshold).toFixed(2)} / 确认 ${p.confirm_windows} / ${p.action}${p.dry_run ? " / 演练" : ""}`);
  el.innerHTML = parts.join(" · ");
}

/* ------------------------------------------------------------------ 回放 → 在线检测服务 */
async function fetchReplayRows(pos, count) {
  const d = await fetchJSON(`/api/replay_rows?pos=${Math.max(0, Math.floor(pos))}&count=${Math.max(1, Math.floor(count))}`);
  if (!d || !d.ok || !Array.isArray(d.rows) || !d.rows.length) {
    throw new Error((d && d.error) || "/api/replay_rows 无数据");
  }
  return d.rows;
}

async function postDetect(rows, srcIP) {
  const r = await fetch("/api/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ meta: { src_ip: srcIP, dst_ip: "10.0.0.1", dst_port: 80 }, rows }),
  });
  if (r.status === 503) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j.error || "在线检测服务未就绪");
  }
  if (!r.ok) throw new Error("/api/detect → " + r.status);
  const det = await r.json();
  if (det && det.ok === false) throw new Error(det.error || "在线推流失败");
  return det;
}

function feedRows(rows) {
  const f = SOC.feed;
  if (!f.enabled || f.off || !SOC.detector.ready || !rows || !rows.length) return;
  const keep = rows.slice(-FEED_BATCH_MAX);
  f.sampled += rows.length - keep.length;   // 高倍速回放：按批内最近若干条推送
  feedIndices(keep[0].i, keep.length);
}

async function feedIndices(pos, count) {
  const f = SOC.feed;
  if (f.busy) return;                        // 上一批还在路上，避免请求堆积
  f.busy = true;
  try {
    const rows = await fetchReplayRows(pos, count);
    const det = await postDetect(rows, SOC.attackerIP || DEFAULT_ATTACK_IP);
    f.sent += rows.length;
    f.failures = 0;
    if (det.stats && det.stats.ready) { SOC.stats = det.stats; renderStatusBar(); }
    ingestAlerts(det.alerts);
  } catch (e) {
    f.lastError = String((e && e.message) || e);
    f.failures++;
    if (f.failures >= 3) {                   // 连续失败：自动停止推送并如实提示（不影响其它面板）
      f.off = true;
      f.enabled = false;
      const cb = $("p-feed");
      if (cb) cb.checked = false;
    }
    renderStatusBar();
  } finally {
    f.busy = false;
  }
}

/** 跳转（注入攻击）时先把跳转点之前的窗口推过去，保证在线服务的滑动缓冲一致 */
async function feedPrime(pos) {
  const f = SOC.feed;
  if (!f.enabled || f.off || !SOC.detector.ready) return;
  const win = (S.overview && S.overview.window_size) || 16;
  try {
    const rows = await fetchReplayRows(Math.max(0, pos - win), win);
    const det = await postDetect(rows, SOC.attackerIP || DEFAULT_ATTACK_IP);
    f.sent += rows.length;
    if (det.stats && det.stats.ready) { SOC.stats = det.stats; renderStatusBar(); }
    ingestAlerts(det.alerts);
    f.failures = 0;
  } catch (e) { /* 预热失败不提示：正常推流阶段会给出状态 */ }
}

/* ==================================================================================
   漂移与未知态势面板
   ----------------------------------------------------------------------------------
   要回答的问题：**模型对当前这段流量有多陌生？**（未知/新颖性占比随时间的变化）
   数据全部来自后端 /api/novelty 一次算好的分箱结果，前端只做绘制，
   不逐帧遍历窗口、不重算模型 —— 因此回放时开销可以忽略。

   上一格 canvas：每格滑窗里被判为「未知形态」的窗口占比（柱）+ 标定参考线（水平虚线）
   下一格 canvas：同一时间轴上，各类别被判为攻击的窗口占该格的比例（堆叠）
   两者共用一条「当前回放位置」游标，并可一键跳到未知占比最高的一格。
   ================================================================================== */
const DRIFT = {
  ok: false, reason: "", bins: 60, data: null,
  curBin: -1,      // 游标所在格
  scan: 0,         // 单调扫描指针（避免每次回放都从头找）
  hover: -1,       // 鼠标悬停的格
  pending: false,  // requestAnimationFrame 节流标志
  xnote: "",       // 静态说明文案（悬停时临时替换）
};

/* 类别配色：沿用大屏的深色 SOC 调色板（不引入任何外部资源） */
const DRIFT_COLORS = {
  Botnet: "#a78bfa", BruteForce: "#f472b6", DDoS: "#ef4444", DoS: "#f59e0b",
  Probe: "#22d3ee", U2R: "#84cc16", WebAttack: "#60a5fa",
};
const DRIFT_OTHER_COLOR = "#94a3b8";

const fmtDur = (s) => {
  s = Math.max(0, Number(s) || 0);
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = Math.floor(s % 60);
  return hh > 0 ? `${hh}h${String(mm).padStart(2, "0")}m` : `${mm}m${String(ss).padStart(2, "0")}s`;
};

/** 把上界抬到 1/2/2.5/5/10 × 10^k，让纵轴刻度好看 */
function niceCeil(v) {
  if (!isFinite(v) || v <= 0) return 0.01;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / p;
  const m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
  return m * p;
}

/* ---------------------------------------------------------------- 取数 */
async function loadNovelty(bins) {
  const want = Math.round(clamp(Number(bins) || DRIFT.bins || 60, 8, 240));
  try {
    const d = await fetchJSON("/api/novelty?bins=" + want);
    DRIFT.data = d || null;
    DRIFT.ok = !!(d && d.ok);
    DRIFT.reason = (d && d.reason) || "";
  } catch (e) {
    DRIFT.data = null;
    DRIFT.ok = false;
    DRIFT.reason = "接口不可用（" + (e.message || e) + "）";
  }
  DRIFT.bins = (DRIFT.data && DRIFT.data.bins) || want;
  DRIFT.curBin = -1;
  DRIFT.scan = 0;
  DRIFT.hover = -1;
  renderDriftChrome();
  driftScheduleDraw();
}

/* ---------------------------------------------------------------- 面板外框文案 */
function renderDriftChrome() {
  const body = $("drift-body"), fb = $("drift-fallback"), chips = $("drift-chips");
  const btn = $("btn-drift-peak"), concl = $("drift-conclusion");
  if (!body || !fb || !chips) return;
  const d = DRIFT.data;

  // ---------- 降级：没有标定文件 / 没有窗口级缓存 → 如实说明，不画曲线 ----------
  if (!DRIFT.ok) {
    body.classList.add("hide");
    fb.className = "drift-fallback show";
    // 后端 reason 也以「未标定，面板不可用」开头，这里去掉前缀避免同一句话出现两遍
    const why = String(DRIFT.reason || "").replace(/^未标定，面板不可用[：:]\s*/, "");
    fb.innerHTML = `<b>未标定，面板不可用</b><br>${why || "缺少窗口级预测缓存或标定阈值"}` +
      (d && d.hint ? `<br><span style="color:#8ba0c0">${d.hint}</span>` : "");
    chips.innerHTML = `<span class="drift-chip warn">漂移面板<b>未标定</b></span>`;
    if (btn) { btn.disabled = true; btn.title = "标定阈值不可用时无法定位未知占比最高处"; }
    const pk = $("drift-peak-hint");
    if (pk) pk.textContent = "";
    if (concl) {
      concl.textContent = "面板只在「窗口级预测缓存 + 未知标记标定文件」两样产物齐备时才绘制曲线；" +
        "缺任意一样都如实降级，不伪造参考线。";
    }
    return;
  }

  // ---------- 正常 ----------
  body.classList.remove("hide");
  fb.className = "drift-fallback";
  fb.innerHTML = "";
  const peak = d.peak || {};
  const glob = Number(d.unknown_ratio_global);
  const budget = Number(d.budget_line);
  const ratios = d.unknown_ratio || [];
  const nOver = ratios.filter((v) => (Number(v) || 0) > budget).length;
  chips.innerHTML = [
    `<span class="drift-chip">当前全局未知占比<b>${(100 * glob).toFixed(2)}%</b></span>`,
    `<span class="drift-chip" title="熵是新奇性分数：8 类概率分布越平，越像「模型没见过这种形态」">熵阈值（标定）<b>${Number(d.threshold).toFixed(4)}</b></span>`,
    `<span class="drift-chip warn" title="标定误报预算：正常流量的未知率应低于它；局部未知率越过它就需要人工复核">标定参考线<b>${(100 * budget).toFixed(1)}%</b></span>`,
    `<span class="drift-chip${nOver ? " hot" : ""}" title="未知占比越过标定参考线的格数（这些时段值得人工复核）">越过参考线的格<b>${nOver}/${d.bins}</b></span>`,
    `<span class="drift-chip" id="drift-cur-chip">当前回放格未知占比<b id="drift-cur-ratio">—</b></span>`,
    `<span class="drift-chip" title="本次曲线用到的判定窗口总数">判定窗口<b>${Number(d.n_windows).toLocaleString()}</b></span>`,
    `<span class="drift-chip" title="窗口级预测缓存（回放路径，不需要 TensorFlow）">数据源<b>${d.source || "-"}</b></span>`,
  ].join("");

  if (btn) {
    btn.disabled = !(peak && isFinite(Number(peak.row)));
    btn.title = "把回放位置跳到未知占比最高那一格的起点";
  }
  const pk = $("drift-peak-hint");
  if (pk && peak) {
    pk.innerHTML = `峰值：第 <b>${Number(peak.bin) + 1}/${d.bins}</b> 格 · 未知占比 <b>${(100 * peak.ratio).toFixed(2)}%</b>` +
      `（参考线 ${(100 * budget).toFixed(1)}%）· 判定窗口 #${peak.w_start}~#${peak.w_end} · 回放位置 <b>${Number(peak.row).toLocaleString()}</b>` +
      (peak.top_class ? ` · 该格判定 ${peak.top_class} ${peak.top_class_count} 个窗口` : "");
  }
  if (concl && d.conclusion) concl.textContent = d.conclusion;

  // 攻击类别图例（堆叠色块顺序与绘制顺序一致）
  const lg = $("drift-legend");
  if (lg) {
    lg.innerHTML = `<span class="lg" style="color:#8ba0c0">堆叠：模型判定类别占该格窗口比例</span>` +
      (d.attack_classes || []).map((c) =>
        `<span class="lg"><i class="sw" style="background:${DRIFT_COLORS[c] || DRIFT_OTHER_COLOR}"></i>${c}</span>`).join("");
  }
  const xn = $("drift-xnote");
  const x = d.x || {};
  const rows = (x.row_start || []);
  DRIFT.xnote = `横轴：回放流位置 ${rows.length ? rows[0].toLocaleString() : 0} – ` +
    `${rows.length ? (Number(x.row_end[rows.length - 1]) || 0).toLocaleString() : 0}` +
    `（共 ${d.bins} 格，每格约 ${Math.max(1, Math.round(Number(d.n_windows) / d.bins))} 个判定窗口，` +
    `合计 ${Number(d.n_windows).toLocaleString()} 个窗口；时长 ${fmtDur(x.t_end ? x.t_end[x.t_end.length - 1] : 0)}）· ` +
    `纵轴左：未知占比 · 点任意一格可跳转` +
    (d.mapped === false ? " · ⚠ 窗口→回放位置映射为估算值" : "");
  if (xn) xn.textContent = DRIFT.xnote;
  updateDriftCurChip();
}

function updateDriftCurChip() {
  const el = $("drift-cur-ratio");
  if (!el) return;
  const d = DRIFT.data;
  const chip = $("drift-cur-chip");
  if (!DRIFT.ok || !d || DRIFT.curBin < 0) {
    el.textContent = "—";
    if (chip) chip.className = "drift-chip";
    return;
  }
  const v = Number((d.unknown_ratio || [])[DRIFT.curBin]) || 0;
  el.textContent = (100 * v).toFixed(2) + "%";
  const budget = Number(d.budget_line);
  if (chip) chip.className = "drift-chip" + (v > budget ? " hot" : (v > 0 ? " warn" : ""));
  const x = d.x || {};
  if (chip && x.row_start) {
    chip.title = `第 ${DRIFT.curBin + 1}/${d.bins} 格 · 判定窗口 #${x.w_start[DRIFT.curBin]}~#${x.w_end[DRIFT.curBin]}` +
      ` · 回放位置 ${x.row_start[DRIFT.curBin]}~${x.row_end[DRIFT.curBin]}`;
  }
}

/* ---------------------------------------------------------------- 回放位置 → 格 */
/** 当前回放位置落在第几格（单调扫描：只在越界时回退，摊还 O(1)） */
function driftBinForPos(pos) {
  const d = DRIFT.data;
  const rs = (d && d.x && d.x.row_start) || [];
  if (!rs.length) return -1;
  let i = clamp(DRIFT.scan, 0, rs.length - 1);
  if (rs[i] > pos) i = 0;
  while (i + 1 < rs.length && rs[i + 1] <= pos) i++;
  while (i > 0 && rs[i] > pos) i--;
  DRIFT.scan = i;
  return i;
}

/** 回放推进时调用：只有「走到另一格」才更新 DOM 与重绘（不是每帧） */
function driftSyncCursor() {
  if (!DRIFT.ok) return;
  const bin = driftBinForPos(S.pos);
  if (bin === DRIFT.curBin) return;
  DRIFT.curBin = bin;
  updateDriftCurChip();
  driftScheduleDraw();
}

function driftResetCursor() {
  DRIFT.scan = 0;
  DRIFT.curBin = driftBinForPos(S.pos);
  updateDriftCurChip();
  driftScheduleDraw();
}

/* ---------------------------------------------------------------- 绘制 */
function driftScheduleDraw() {
  if (DRIFT.pending) return;
  DRIFT.pending = true;
  requestAnimationFrame(() => { DRIFT.pending = false; drawDrift(); });
}

/** 统一的 dpr 画布准备：返回逻辑尺寸的 2D 上下文 */
function driftCtx(cv, hCss) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(320, cv.clientWidth || 900);
  // 用元素实际高度（canvas 有 1px 边框，clientHeight 比 CSS height 小 2px），避免底部被裁掉
  const h = Math.max(60, cv.clientHeight || hCss);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawDrift() {
  if (!DRIFT.ok || !DRIFT.data) return;
  try { drawDriftRatio(); } catch (e) { /* 面板绘制异常不影响回放主流程 */ }
  try { drawDriftStack(); } catch (e) { /* 同上 */ }
}

function drawDriftRatio() {
  const cv = $("drift");
  if (!cv) return;
  const d = DRIFT.data;
  const { ctx, w, h } = driftCtx(cv, 200);
  const pad = { l: 46, r: 12, t: 16, b: 16 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const bins = Math.max(1, Number(d.bins) || 60);
  const ratio = d.unknown_ratio || [];
  const peak = d.peak || {};
  const budget = Number(d.budget_line) || 0;
  let maxR = budget;
  for (const v of ratio) maxR = Math.max(maxR, Number(v) || 0);
  const yMax = Math.max(niceCeil(maxR * 1.28), 0.01);

  const X = (i) => pad.l + iw * (i + 0.5) / bins;
  const Y = (v) => pad.t + ih * (1 - clamp((Number(v) || 0) / yMax, 0, 1));
  const bw = clamp(iw / bins - 2.4, 2, 26);

  // 网格 + 纵轴刻度
  ctx.font = "10px " + FONT_MONO;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 1;
  for (let k = 0; k <= 3; k++) {
    const v = yMax * k / 3, y = Y(v);
    ctx.strokeStyle = k === 0 ? "rgba(139,160,192,.35)" : "rgba(139,160,192,.15)";
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = "rgba(139,160,192,.75)";
    ctx.fillText((100 * v).toFixed(yMax < 0.05 ? 2 : 1) + "%", 5, y + 0.5);
  }

  // 未知占比柱：颜色分三档（越暖＝该格未知占比越高）；
  // 注意「一格 ≈ 59 个判定窗口」，所以单个未知窗口就会让该格达到 1.7%——
  // 因此颜色按「相对全局峰值」分档，标定参考线另外用水平虚线标注（它是一条刻度，不是序列）。
  const peakV = Number(peak.ratio) || 0;
  const scale = Math.max(peakV, budget * 2, 1e-6);
  // 峰格先铺一条淡红竖带（不用描边：满高描边会被误读成第二条游标线）
  if (isFinite(Number(peak.bin)) && peakV > 0) {
    ctx.fillStyle = "rgba(239,68,68,.10)";
    ctx.fillRect(X(Number(peak.bin)) - bw / 2 - 1, pad.t, bw + 2, ih);
  }
  for (let i = 0; i < bins; i++) {
    const v = Number(ratio[i]) || 0;
    const x = X(i) - bw / 2;
    const y = Y(v);
    const base = pad.t + ih;
    if (v <= 0) {                                    // 0 也要留一道基座，避免看着像缺数据
      ctx.fillStyle = "rgba(139,160,192,.3)";
      ctx.fillRect(x, base - 1.5, bw, 1.5);
      continue;
    }
    const u = v / scale;
    const g = ctx.createLinearGradient(0, y, 0, base);
    if (i === Number(peak.bin) || u >= 0.66) { g.addColorStop(0, "#fca5a5"); g.addColorStop(1, "rgba(239,68,68,.3)"); }
    else if (u >= 0.33) { g.addColorStop(0, "#fcd34d"); g.addColorStop(1, "rgba(245,158,11,.26)"); }
    else { g.addColorStop(0, "#7dd3fc"); g.addColorStop(1, "rgba(56,189,248,.2)"); }
    ctx.fillStyle = g;
    ctx.fillRect(x, y, bw, Math.max(base - y, 2));
  }

  // 标定参考线（水平虚线：它不是时间序列，而是一条「刻度」）
  const yb = Y(budget);
  ctx.strokeStyle = "rgba(245,158,11,.9)";
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(pad.l, yb); ctx.lineTo(w - pad.r, yb); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(252,211,77,.95)";
  ctx.font = "10.5px " + FONT_CN;
  ctx.textAlign = "right";
  ctx.fillText(`标定参考线 ${(100 * budget).toFixed(1)}%（熵阈值 ${Number(d.threshold).toFixed(3)}）`, w - pad.r - 4, yb - 6);
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(139,160,192,.8)";
  ctx.fillText("未知占比（每格被判「未知形态」的窗口比例）", pad.l + 4, pad.t + 5);

  // 峰格标注
  if (isFinite(Number(peak.ratio)) && Number(peak.ratio) > 0) {
    const px = clamp(X(Number(peak.bin)), pad.l + 48, w - pad.r - 48);
    const py = Math.max(Y(peak.ratio) - 8, pad.t + 14);
    ctx.fillStyle = "#fecaca";
    ctx.font = "600 10.5px " + FONT_CN;
    ctx.textAlign = "center";
    ctx.fillText(`峰值 ${(100 * Number(peak.ratio)).toFixed(1)}%`, px, py);
    ctx.beginPath();
    ctx.moveTo(px, py + 5); ctx.lineTo(px - 4, py + 10); ctx.lineTo(px + 4, py + 10);
    ctx.closePath();
    ctx.fillStyle = "#ef4444";
    ctx.fill();
    ctx.textAlign = "left";
  }

  // 鼠标悬停格的竖线（比游标浅，避免混淆）
  if (DRIFT.hover >= 0 && DRIFT.hover !== DRIFT.curBin) {
    const hx = X(DRIFT.hover);
    ctx.strokeStyle = "rgba(139,160,192,.45)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(hx, pad.t); ctx.lineTo(hx, pad.t + ih); ctx.stroke();
  }

  // 回放游标
  drawDriftCursor(ctx, X, pad, ih, bins, true);
}

function drawDriftCursor(ctx, X, pad, ih, bins, withHead) {
  if (DRIFT.curBin < 0 || DRIFT.curBin >= bins) return;
  const cx = X(DRIFT.curBin);
  ctx.strokeStyle = "rgba(230,237,247,.92)";
  ctx.lineWidth = 1.3;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, pad.t + ih); ctx.stroke();
  ctx.setLineDash([]);
  if (withHead) {
    ctx.beginPath();
    ctx.moveTo(cx, pad.t + 2); ctx.lineTo(cx - 4.5, pad.t - 5); ctx.lineTo(cx + 4.5, pad.t - 5);
    ctx.closePath();
    ctx.fillStyle = "#e6edf7";
    ctx.fill();
  }
}

function drawDriftStack() {
  const cv = $("drift-stack");
  if (!cv) return;
  const d = DRIFT.data;
  const { ctx, w, h } = driftCtx(cv, 140);
  const pad = { l: 46, r: 12, t: 10, b: 20 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const bins = Math.max(1, Number(d.bins) || 60);
  const classes = d.attack_classes || [];
  const share = d.attack_share || {};
  const x = d.x || {};

  // 纵轴上界：所有格里最高的「攻击占比合计」
  let maxS = 0;
  for (let i = 0; i < bins; i++) {
    let s = 0;
    for (const c of classes) s += Number((share[c] || [])[i]) || 0;
    maxS = Math.max(maxS, s);
  }
  const yMax = clamp(niceCeil(Math.max(maxS * 1.15, 0.05)), 0.05, 1);

  const X = (i) => pad.l + iw * (i + 0.5) / bins;
  const bw = clamp(iw / bins - 2.4, 2, 26);
  const base = pad.t + ih;

  ctx.font = "10px " + FONT_MONO;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 1;
  for (let k = 0; k <= 2; k++) {
    const v = yMax * k / 2, y = base - ih * v / yMax;
    ctx.strokeStyle = k === 0 ? "rgba(139,160,192,.35)" : "rgba(139,160,192,.15)";
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = "rgba(139,160,192,.75)";
    ctx.fillText((100 * v).toFixed(0) + "%", 5, y + 0.5);
  }
  ctx.fillStyle = "rgba(139,160,192,.8)";
  ctx.font = "10.5px " + FONT_CN;
  ctx.fillText("攻击构成（各类别被判为攻击的窗口占该格比例，堆叠）", pad.l + 4, pad.t + 6);

  // 堆叠柱
  for (let i = 0; i < bins; i++) {
    let acc = 0;
    const px = X(i) - bw / 2;
    for (const c of classes) {
      const v = Number((share[c] || [])[i]) || 0;
      if (v <= 0) continue;
      const hgt = Math.max(ih * v / yMax, 1.2);
      ctx.fillStyle = DRIFT_COLORS[c] || DRIFT_OTHER_COLOR;
      ctx.globalAlpha = i === DRIFT.hover ? 1 : 0.88;
      ctx.fillRect(px, base - acc - hgt, bw, hgt);
      ctx.globalAlpha = 1;
      acc += hgt;
    }
    if (acc <= 0) {
      ctx.fillStyle = "rgba(139,160,192,.3)";
      ctx.fillRect(px, base - 1.5, bw, 1.5);
    }
  }

  // 时间刻度（用回放流的相对时间，评委更容易对上"攻击开始那一刻"）
  if (x.t_end && x.t_end.length) {
    ctx.fillStyle = "rgba(139,160,192,.75)";
    ctx.font = "10px " + FONT_MONO;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const ticks = Math.min(7, bins);
    for (let k = 0; k < ticks; k++) {
      const i = Math.round((bins - 1) * k / Math.max(ticks - 1, 1));
      ctx.fillText(fmtDur(Number(x.t_end[i]) || 0), clamp(X(i), pad.l + 14, w - pad.r - 14), base + 5);
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
  }

  drawDriftCursor(ctx, X, pad, ih, bins, false);
}

/* ---------------------------------------------------------------- 交互 */
function driftBinAtX(clientX, cv) {
  cv = cv || $("drift");
  if (!cv || !DRIFT.ok) return -1;
  const rect = cv.getBoundingClientRect();
  const pad = { l: 46, r: 12 };
  const iw = rect.width - pad.l - pad.r;
  if (iw <= 0) return -1;
  const bins = Math.max(1, Number(DRIFT.data.bins) || 60);
  const k = Math.floor((clientX - rect.left - pad.l) / (iw / bins));
  return clamp(k, 0, bins - 1);
}

function driftHoverNote(bin) {
  const d = DRIFT.data;
  const x = d.x || {};
  const ratio = Number((d.unknown_ratio || [])[bin]) || 0;
  const cnt = Number((d.unknown_count || [])[bin]) || 0;
  let nAtk = 0, top = "", topN = 0;
  for (const c of (d.attack_classes || [])) {
    const v = Number((d.attack_counts[c] || [])[bin]) || 0;
    nAtk += v;
    if (v > topN) { topN = v; top = c; }
  }
  const nWin = (x.w_end && x.w_start) ? (x.w_end[bin] - x.w_start[bin] + 1) : 0;
  return `第 ${bin + 1}/${d.bins} 格 · 未知占比 <b>${(100 * ratio).toFixed(2)}%</b>（${cnt}/${nWin} 个窗口）· ` +
    `判定攻击 ${nAtk} 个窗口${top ? `（主要是 ${top} ${topN} 个）` : ""} · ` +
    `判定窗口 #${x.w_start ? x.w_start[bin] : "-"}~#${x.w_end ? x.w_end[bin] : "-"} · ` +
    `回放位置 <b>${x.row_start ? Number(x.row_start[bin]).toLocaleString() : "-"}</b>` +
    `（${fmtDur(x.t_start ? x.t_start[bin] : 0)}）· 点击跳到这一格`;
}

async function jumpToNoveltyPeak() {
  const d = DRIFT.data;
  if (!DRIFT.ok || !d || !d.peak || !isFinite(Number(d.peak.row))) return;
  const p = d.peak;
  const top = p.top_class ? `，该格判定 ${p.top_class} ${p.top_class_count} 个窗口` : "";
  await jumpTo(Number(p.row),
    `跳到未知占比最高处（第 ${Number(p.bin) + 1}/${d.bins} 格 · ${(100 * Number(p.ratio)).toFixed(2)}%${top}）`);
}

async function jumpToDriftBin(bin) {
  const x = DRIFT.data && DRIFT.data.x;
  if (!x || !x.row_start || bin < 0 || bin >= x.row_start.length) return;
  await jumpTo(Number(x.row_start[bin]),
    `跳到第 ${bin + 1}/${DRIFT.data.bins} 格（回放位置 ${Number(x.row_start[bin]).toLocaleString()}）`);
}

const driftPeakBtn = $("btn-drift-peak");
if (driftPeakBtn) driftPeakBtn.addEventListener("click", () => { jumpToNoveltyPeak(); });

// 分箱粒度：改一格包含多少个判定窗口（格越大越平滑，格越小越容易看出突然漂移）
const driftBinsSel = $("drift-bins");
if (driftBinsSel) {
  driftBinsSel.addEventListener("change", () => {
    DRIFT.bins = Math.round(clamp(Number(driftBinsSel.value) || 60, 8, 240));
    loadNovelty(DRIFT.bins);
  });
}

const driftCv = $("drift");
if (driftCv) {
  driftCv.addEventListener("mousemove", (e) => {
    const bin = driftBinAtX(e.clientX);
    if (bin === DRIFT.hover) return;
    DRIFT.hover = bin;
    const xn = $("drift-xnote");
    if (xn) xn.innerHTML = bin >= 0 ? driftHoverNote(bin) : DRIFT.xnote;
    driftScheduleDraw();
  });
  driftCv.addEventListener("mouseleave", () => {
    DRIFT.hover = -1;
    const xn = $("drift-xnote");
    if (xn) xn.textContent = DRIFT.xnote;
    driftScheduleDraw();
  });
  driftCv.addEventListener("click", (e) => {
    const bin = driftBinAtX(e.clientX);
    if (bin >= 0) jumpToDriftBin(bin);
  });
}
const driftStackCv = $("drift-stack");
if (driftStackCv) {
  driftStackCv.addEventListener("click", (e) => {
    const bin = driftBinAtX(e.clientX, driftStackCv);   // 两张图共用同一 x 轴几何
    if (bin >= 0) jumpToDriftBin(bin);
  });
}

/* ------------------------------------------------------------------ 入口 */
boot().catch((e) => { console.error(e); showError(String(e)); });
