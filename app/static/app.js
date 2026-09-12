/* 实时检测大屏前端逻辑：纯原生 JS + canvas，无任何外部依赖（断网可演示） */
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

const $ = (id) => document.getElementById(id);
const MAXBUF = 600;
const MAXROWS = 60;

/* ------------------------------------------------------------------ 启动 */
async function boot() {
  const ov = await fetchJSON("/api/overview");
  S.overview = ov;
  if (!ov.ok) {
    showError(ov.reason || "演示数据未就绪");
    return;
  }
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
    "<br><br>请先在项目根目录执行：<code>python run_all.py --smoke</code>（快速）或 <code>python run_all.py</code>（完整）";
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
  renderStats(); drawChart();
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
window.addEventListener("resize", drawChart);

boot().catch((e) => { console.error(e); showError(String(e)); });
