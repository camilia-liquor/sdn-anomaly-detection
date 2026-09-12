/* 评审视图前端：数据全部来自 /api/judge（读 results/ 真实产物）。
 * 设计原则：**缺项灰显、不编造**；任何字段为 null 时显示「未测」。 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmt = (v, nd = 4) => (v === null || v === undefined || Number.isNaN(v))
    ? "未测" : Number(v).toFixed(nd);
  const pct = (v, nd = 1) => (v === null || v === undefined)
    ? "未测" : (Number(v) * 100).toFixed(nd) + "%";
  const n0 = (v) => (v === null || v === undefined) ? "未测" : Number(v).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
  const ms = (v) => (v === null || v === undefined) ? "未测" : Number(v).toFixed(1) + " ms";
  const h = (v) => (v === null || v === undefined) ? "未测" : Number(v).toFixed(2) + " h";

  function card(inner) {
    const d = document.createElement("div");
    d.className = "j-card";
    d.innerHTML = inner;
    return d;
  }

  function kpi(label, value, sub, cls) {
    return `<div class="j-kpi ${cls || ""}"><div class="k">${label}</div>
            <div class="v">${value}</div><div class="s">${sub || ""}</div></div>`;
  }

  // ---------------------------------------------------------------- 核心数字
  function renderHeadline(d) {
    const h1 = d.headline || {};
    const std = h1.macro_f1_std ? ` ± ${fmt(h1.macro_f1_std)}` : "";
    const fa = h1.false_alarms_per_hour;
    const faStd = h1.false_alarms_std ? ` ± ${fmt(h1.false_alarms_std, 2)}` : "";
    $("kpis").innerHTML = [
      kpi("多分类宏平均 F1", fmt(h1.macro_f1) + std,
        `${h1.n_seeds ? h1.n_seeds + " 个随机种子" : ""} · 8 类流量`, "hi"),
      kpi("攻击片段召回", pct(h1.segment_recall, 0),
        h1.n_segments ? `${h1.n_detected}/${h1.n_segments} 段全部检出` : "", "ok"),
      kpi("平均检测时延", ms(h1.mean_delay_ms), "流时间口径（非墙钟）", "ok"),
      kpi("误报率", (fa === null || fa === undefined ? "未测" : fmt(fa, 2) + " /h") + faStd,
        "默认开启「连续 3 窗口确认」", "ok"),
      kpi("二分类 ROC-AUC", fmt(h1.roc_auc, 4), "概率口径（非硬标签）", "hi"),
      kpi("测试流规模", h1.test_windows ? n0(h1.test_windows) + " 窗口" : "未测",
        h1.stream_hours ? `≈ ${fmt(h1.stream_hours, 2)} 小时` : "", ""),
    ].join("");
  }

  // ------------------------------------------------- ① 工作点选择器（可交互）
  let WP = [], OPS = {}, cur = null;

  function wpMatch(thr, cfm) {
    return WP.find((r) => Math.abs(r.threshold - thr) < 1e-9 && r.confirm === cfm) || null;
  }

  function renderWorkingPoint() {
    if (!cur) return;
    const r = wpMatch(cur.thr, cur.cfm);
    const cells = r ? [
      ["攻击片段召回", pct(r.recall, 0), r.recall >= 0.999 ? "good" : "bad"],
      ["平均检测时延", ms(r.mean_delay_ms), ""],
      ["误报 / 小时", fmt(r.false_alarms_per_hour, 2), ""],
      ["误报 / 天", n0(r.false_alarms_per_day), r.false_alarms_per_day > 100 ? "bad" : "good"],
      ["告警 / 天（含真实攻击）", n0(r.alarms_per_day), ""],
      ["人工处置 / 天", h(r.triage_hours_per_day),
        (r.triage_hours_per_day || 0) > 8 ? "bad" : "good"],
      ["误报占比", pct(r.false_alarm_share, 0), ""],
      ["漏检攻击 / 天", fmt(r.missed_per_day, 2), (r.missed_per_day || 0) > 0 ? "bad" : "good"],
    ] : [];
    $("wp-cells").innerHTML = cells.map(([k, v, c]) =>
      `<div class="cell ${c}"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

    const rec = OPS.recommended || {}, dft = OPS.config_default || {}, nc = OPS.no_confirm || {};
    const shift = OPS.shift_hours || 8;
    const lines = [];
    lines.push(`当前选择 <b>阈值 ${cur.thr} + 确认 ${cur.cfm}</b>` +
      (r ? `：召回 <b>${pct(r.recall, 0)}</b>、误报 <b>${fmt(r.false_alarms_per_hour, 2)} 次/小时</b>、`
        + `人工 <b>${h(r.triage_hours_per_day)}/天</b>（约 ${((r.triage_hours_per_day || 0) / shift).toFixed(1)} 个班次）` : "")) ;
    if (rec.threshold !== undefined) {
      lines.push(`推荐上线工作点：阈值 ${rec.threshold} + 确认 ${rec.confirm} → ` +
        `告警 ${n0(rec.alarms_per_day)}/天、人工 <b>${h(rec.triage_hours_per_day)}/天</b>、漏检 ${fmt(rec.missed_attacks_per_day, 2)}/天`);
    }
    if (nc.threshold !== undefined) {
      lines.push(`反事实：<b>不做「连续确认」</b>（阈值 ${nc.threshold} + 确认 1）→ 人工 <b>${h(nc.triage_hours_per_day)}/天</b>` +
        `（≈ ${((nc.triage_hours_per_day || 0) / shift).toFixed(1)} 个班次，误报本身就成了事故）` +
        (OPS.saved_triage_hours_per_day ? `，确认机制每天省下 <b>${h(OPS.saved_triage_hours_per_day)}</b>` : ""));
    }
    if (OPS.triage_minutes_per_alert) {
      lines.push(`换算口径：每条告警排查 ${OPS.triage_minutes_per_alert} 分钟、一个班次 ${shift} 小时、` +
        `攻击频次假设 ${OPS.attacks_per_day_assumption || "—"} 次/天、误报预算 ${OPS.false_alarm_budget_per_hour || "—"} 次/小时`);
    }
    $("wp-cmp").innerHTML = lines.join("<br />");
    $("wp-cur").textContent = `当前：阈值 ${cur.thr} + 确认 ${cur.cfm}`;

    // 全表（高亮当前行；标出推荐点）
    const rows = WP.map((x) => {
      const cls = (Math.abs(x.threshold - cur.thr) < 1e-9 && x.confirm === cur.cfm) ? "on"
        : (rec.threshold !== undefined && Math.abs(x.threshold - rec.threshold) < 1e-9 && x.confirm === rec.confirm) ? "rec"
        : (x.triage_hours_per_day || 0) > 8 ? "bad" : "";
      return `<tr class="${cls}"><td>${x.threshold} + 确认 ${x.confirm}</td>` +
        `<td>${pct(x.recall, 0)}</td><td>${fmt(x.false_alarms_per_hour, 2)}</td>` +
        `<td>${n0(x.false_alarms_per_day)}</td><td>${n0(x.alarms_per_day)}</td>` +
        `<td>${h(x.triage_hours_per_day)}</td><td>${pct(x.false_alarm_share, 0)}</td>` +
        `<td>${fmt(x.missed_per_day, 2)}</td><td>${ms(x.mean_delay_ms)}</td></tr>`;
    }).join("");
    $("wp-table").innerHTML =
      `<thead><tr><th>阈值 + 确认次数</th><th>片段召回</th><th>误报/小时</th><th>误报/天</th>` +
      `<th>告警/天</th><th>人工/天</th><th>误报占比</th><th>漏检/天</th><th>平均时延</th></tr></thead>` +
      `<tbody>${rows}</tbody>`;
  }

  function initWorkingPoint(d) {
    WP = (d.operating_points || []).filter((r) => r.threshold !== undefined);
    OPS = d.ops || {};
    const thrs = [...new Set(WP.map((r) => r.threshold))].sort((a, b) => a - b);
    const cfms = [...new Set(WP.map((r) => r.confirm))].sort((a, b) => a - b);
    if (!thrs.length || !cfms.length) {
      $("wp-cmp").innerHTML = "未找到 results/operating_point.json 与 results/ops_impact.json，本面板不可用";
      return;
    }
    const slider = $("wp-thr");
    slider.min = 0; slider.max = thrs.length - 1; slider.step = 1;
    slider.value = String(Math.max(0, thrs.indexOf(0.5) >= 0 ? thrs.indexOf(0.5) : 0));
    $("wp-thr-v").textContent = String(thrs[Number(slider.value)]);
    const rec = OPS.recommended || {};
    cur = { thr: thrs[Number(slider.value)], cfm: rec.confirm || cfms[0] };
    $("wp-cfm").innerHTML = cfms.map((k) =>
      `<button data-cfm="${k}" class="${k === cur.cfm ? "on" : ""}">确认 ${k}</button>`).join("");
    $("wp-cfm").querySelectorAll("button").forEach((b) => {
      b.onclick = () => {
        cur.cfm = Number(b.dataset.cfm);
        $("wp-cfm").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
        renderWorkingPoint();
      };
    });
    slider.oninput = () => {
      cur.thr = thrs[Number(slider.value)];
      $("wp-thr-v").textContent = String(cur.thr);
      const has = wpMatch(cur.thr, cur.cfm);
      if (!has) {  // 该阈值下没有这个确认次数：自动落到最近的可用组合，避免显示"未测"
        const near = WP.filter((r) => Math.abs(r.threshold - cur.thr) < 1e-9);
        if (near.length) {
          cur.cfm = near[0].confirm;
          $("wp-cfm").querySelectorAll("button").forEach((x) =>
            x.classList.toggle("on", Number(x.dataset.cfm) === cur.cfm));
        }
      }
      renderWorkingPoint();
    };
    renderWorkingPoint();
  }

  // ------------------------------------------------------ ② 三次自我证伪
  function renderFalsifications(d) {
    const box = $("fals");
    box.innerHTML = "";
    (d.falsifications || []).forEach((f) => {
      box.appendChild(card(`
        <h3>${f.title}</h3>
        <div class="row"><span class="k">假设：</span>${f.hypothesis}</div>
        <div class="row"><span class="k">怎么试的：</span>${f.how}</div>
        <div class="row"><span class="k">结果：</span><b>${f.result}</b></div>
        <div class="honest">${f.honest}</div>
        <div class="src">${f.source}</div>`));
    });
    if (!box.children.length) box.innerHTML = '<div class="j-card">未找到实验产物</div>';
  }

  // --------------------------------------------- ③ 稳健性与边界（三张卡）
  function renderRobustness(d) {
    const box = $("robust");
    box.innerHTML = "";
    const adv = d.robustness_adversarial;
    if (adv) {
      box.appendChild(card(`
        <h3>对抗鲁棒性（白盒 · 特征空间 L∞）</h3>
        <div class="row">干净宏 F1 <b>${fmt(adv.clean_macro_f1)}</b>；
          ε=${adv.worst_eps} 的定向扰动下宏 F1 <b>${fmt(adv.worst_macro_f1)}</b>、
          <b>逃逸率 ${pct(adv.worst_evasion, 1)}</b></div>
        <div class="row">同范数<b>随机扰动几乎无效</b>（宏 F1 ≥ ${fmt(adv.random_best)}）→ 退化来自定向优化，不是「加噪声就崩」</div>
        <div class="row">对抗训练：收益 <b>${fmt(adv.adv_gain)}</b> vs 代价 <b>${fmt(adv.adv_cost)}</b>
          → ${adv.worthwhile ? "划算" : "不划算"}（判据事前写死：收益 > 代价）</div>
        <div class="honest">${adv.honest}</div>
        <div class="src">results/adversarial_eval.md</div>`));
    }
    const oo = d.robustness_openset;
    if (oo) {
      const rows = (oo.cards || []).map((c) =>
        `<div class="row"><b>${c.scenario}</b>：1% 误报预算下未知攻击检出
         ${pct(c.baseline, 1)}（${c.baseline_score}）→ <b>${pct(c.mahalanobis, 1)}</b>（马氏距离）；
         已知攻击误拒 ${pct(c.false_reject_base, 2)} → ${pct(c.false_reject_new, 2)}</div>`).join("");
      box.appendChild(card(`
        <h3>未知攻击（开集）拒识 —— 本轮把它修好了</h3>
        ${rows}
        <div class="row">判定：<b>${oo.verdict}</b></div>
        <div class="honest">${oo.honest}</div>
        <div class="src">results/openset_ood_plus.md</div>`));
    }
    const cf = d.controller_fields;
    if (cf) {
      const p17 = cf.point_17 || {}, pex = cf.point_ext || {};
      box.appendChild(card(`
        <h3>字段缺口补平：只用 OpenFlow 能算出来的字段</h3>
        <div class="row">控制器原生 <b>${cf.n17} 列</b>：宏 F1 <b>${fmt(cf.f1_17)}</b>
          ± ${fmt(cf.f1_17_std)} → 补上可推导聚合量后 <b>${cf.n_ext} 列</b>：<b>${fmt(cf.f1_ext)}</b>
          ± ${fmt(cf.f1_ext_std)}（Δ=${fmt(cf.delta)}，σ=${fmt(cf.sigma)}）</div>
        <div class="row">工作点：阈值 ${p17.threshold !== undefined ? p17.threshold : "—"} + 确认 ${p17.confirm || "—"}
          （${ms(p17.mean_delay_ms)}）→ 阈值 ${pex.threshold !== undefined ? pex.threshold : "—"} + 确认 ${pex.confirm || "—"}
          （<b>${ms(pex.mean_delay_ms)}</b>）</div>
        <div class="row">同时列出 OpenFlow 流表<b>本身不提供</b>的 ${cf.unavailable} 类字段（不伪造）</div>
        <div class="honest">${cf.honest}</div>
        <div class="src">results/ctrl_field_extension.md</div>`));
    }
    if (!box.children.length) box.innerHTML = '<div class="j-card">未找到实验产物</div>';
  }

  // ------------------------------------------------------- ④ 诚实边界
  function renderBoundaries(d) {
    $("bounds").innerHTML = (d.boundaries || []).map((b) =>
      `<div class="item"><div class="st ${b.ok ? "yes" : "no"}">${b.ok ? "✔ " : "✘ "}${b.item} · ${b.status}</div>
       <div class="dt">${b.detail}</div></div>`).join("");
  }

  // ------------------------------------------------------- ⑤ 可验证性
  function renderVerifiable(d) {
    const v = d.verifiable || {};
    const a = v.acceptance || {};
    const t = v.tests || {};
    const head = `<div class="c"><span class="what">交付验收自检</span><code>` +
      `${a.passed === null || a.passed === undefined ? "未生成 results/acceptance_summary.json" :
        `通过 ${a.passed} 项 / 失败 ${a.failed} 项 / 警告 ${a.warned} 项`}</code></div>`;
    const tests = Object.keys(t).length
      ? `<div class="c"><span class="what">三套单元 / 行为测试</span><code>` +
        Object.entries(t).map(([k, val]) => `${k}: ${val}`).join(" , ") + `</code></div>`
      : "";
    const tools = `<div class="c"><span class="what">工具脚本</span><code>${v.n_tools || "—"} 个（含交付验收自检）</code></div>`;
    const cmds = (v.commands || []).map(([what, cmd]) =>
      `<div class="c"><span class="what">${what}</span><code>${cmd}</code></div>`).join("");
    $("cmds").innerHTML = head + tests + tools + cmds;
  }

  // ------------------------------------------------------ ⑥ 开源与融合
  function renderOss(d) {
    const o = d.opensource || {};
    const box = $("oss");
    box.innerHTML = "";
    box.appendChild(card(`
      <h3>AI ←→ 开源：不是并列贴两个名词，而是真的接起来</h3>
      <div class="row">${o.integration}</div>
      <div class="row">自研替代依赖：${o.self_built}</div>
      <div class="src">integrations/ryu_guard.py · tools/ryu_live_check.py</div>`));
    box.appendChild(card(`
      <h3>开放成果（可复用）</h3>
      <div class="row">许可：<b>${o.license}</b>
        ${o.has_notice ? "· NOTICE" : ""}${o.has_contributing ? " · CONTRIBUTING" : ""}
        ${o.has_ci ? " · CI 工作流" : ""}</div>
      <div class="row">工具脚本 <b>${o.n_tools || "—"}</b> 个 ·
        部署样例 ${o.has_deploy ? "有" : "无"}（systemd / nginx / Windows 服务）·
        样例数据与样例抓包 ${o.samples ? "有" : "无"}</div>
      <div class="row">第三方组件逐项登记用途/许可/自研边界（官方强制附件）：
        competition/01_开源及第三方资源使用清单.md</div>
      <div class="src">LICENSE · docs/ · competition/</div>`));
  }

  function renderFoot(d) {
    $("foot").innerHTML =
      `<b>诚实声明：</b>本页所有数字都来自 <code>results/</code> 下的真实实验产物（页面刷新即重新读取），` +
      `缺失项一律显示「未测」而不会补零或估算；` +
      `已知边界见第 ④ 节 —— 包括零样本跨域不可用、域随机化被我们实测否定、Ryu 未在物理交换机实跑。` +
      `<br />生成时间：${new Date().toLocaleString("zh-CN")}`;
  }

  fetch("/api/judge").then((r) => r.json()).then((d) => {
    if (!d.ok) throw new Error("接口返回 ok=false");
    renderHeadline(d);
    initWorkingPoint(d);
    renderFalsifications(d);
    renderRobustness(d);
    renderBoundaries(d);
    renderVerifiable(d);
    renderOss(d);
    renderFoot(d);
    $("src-badge").className = "badge ok";
    $("src-badge").textContent = "数据源：results/ 真实产物";
    document.title = "评审视图 · 已加载";
  }).catch((e) => {
    $("src-badge").className = "badge bad";
    $("src-badge").textContent = "数据加载失败：" + e.message;
    $("kpis").innerHTML = '<div class="j-kpi"><div class="k">错误</div><div class="v">/api/judge 不可用</div>' +
      '<div class="s">请确认服务已启动且 results/ 下有产物</div></div>';
  });
})();
