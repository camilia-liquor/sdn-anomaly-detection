"""教学试点数据分析：从采集表算出可直接写进申报书的统计表与图。

用法：
    # 真实数据（试点结束后跑这个）
    python tools/pilot_analysis.py --xlsx competition/教学试点数据采集表.xlsx

    # 流程演示（用模拟数据跑通全流程；输出带【模拟数据】标记，不得写进申报书）
    python tools/pilot_analysis.py --demo

设计纪律（与工程一致）：
* 判读口径事前写死（见 competition/教学试点方案.md 第 6 节），脚本不做任何事后挑选；
* 采集表里只有示例行时**拒绝出结论**，必须显式 --demo 才用模拟数据；
* --demo 的所有产物都写入 results/pilot_demo/，且每份产物头部标注【模拟数据，不得提交】。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
GRADE_SHEET = "成绩与耗时"
QUEST_SHEET = "问卷"
QCOLS = ["Q1易用性", "Q2安全性", "Q3闭环可见", "Q4可复现", "Q5原理理解",
         "Q6误报漏报", "Q7评估认知", "Q8兴趣提升", "Q9继续使用", "Q10推荐意愿"]


# ---------------------------------------------------------------- 读取
def load_real(xlsx: Path, include_example: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_excel(xlsx, sheet_name=GRADE_SHEET)
    q = pd.read_excel(xlsx, sheet_name=QUEST_SHEET)
    g.columns = [str(c).strip() for c in g.columns]
    q.columns = [str(c).strip() for c in q.columns]
    if not include_example:
        g = g[~g["匿名编号"].astype(str).str.contains("示例", na=False)]
        q = q[~q["匿名编号"].astype(str).str.contains("示例", na=False)]
    g = g.dropna(subset=["前测得分", "后测得分"]).reset_index(drop=True)
    q = q.dropna(subset=[c for c in QCOLS if c in q.columns], how="all").reset_index(drop=True)
    return g, q


# ---------------------------------------------------------------- 模拟数据
def make_demo(n: int = 40, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成结构合理的模拟试点数据（仅用于演示流程）。"""
    rng = np.random.default_rng(seed)
    pre = np.clip(rng.normal(51, 12, n), 10, 95)
    post = np.clip(pre + rng.normal(33, 9, n), 20, 100)          # 与 ppt 相关：后测 = 前测 + 提升 + 噪声
    dur = np.clip(rng.normal(63, 9, n), 35, 95)
    done = rng.random(n) < 0.95
    g = pd.DataFrame({
        "匿名编号": [f"S{i + 1:02d}" for i in range(n)],
        "前测得分": np.round(pre, 0).astype(int),
        "后测得分": np.round(post, 0).astype(int),
        "实验耗时(min)": np.round(dur, 0).astype(int),
        "是否完成全部步骤": np.where(done, "是", "否"),
        "备注": "",
    })
    base = np.array([4.45, 4.60, 4.52, 4.35, 4.50, 4.42, 4.30, 4.62, 4.40, 4.25])
    q = pd.DataFrame({"匿名编号": g["匿名编号"]})
    for j, c in enumerate(QCOLS):
        vals = np.clip(np.round(rng.normal(base[j], 0.62, n)), 1, 5).astype(int)
        q[c] = vals
    return g, q


# ---------------------------------------------------------------- 统计
def analyze(g: pd.DataFrame, q: pd.DataFrame, demo: bool) -> dict:
    from scipy import stats

    pre = g["前测得分"].to_numpy(float)
    post = g["后测得分"].to_numpy(float)
    diff = post - pre
    n = len(g)
    t_stat, p_t = stats.ttest_rel(post, pre)
    try:
        w_stat, p_w = stats.wilcoxon(post, pre)
    except Exception:  # noqa: BLE001
        p_w = float("nan")
    d = float(diff.mean() / diff.std(ddof=1)) if n > 1 and diff.std(ddof=1) > 0 else float("nan")
    done_col = g.get("是否完成全部步骤")
    done_rate = float((done_col.astype(str).str.startswith("是")).mean()) if done_col is not None else float("nan")
    dur = pd.to_numeric(g.get("实验耗时(min)"), errors="coerce").dropna()

    qstats = {}
    if len(q):
        for c in QCOLS:
            if c in q.columns:
                v = pd.to_numeric(q[c], errors="coerce").dropna()
                if len(v):
                    qstats[c] = {"mean": round(float(v.mean()), 2), "std": round(float(v.std(ddof=1)), 2),
                                 "n": int(len(v))}
        allv = pd.concat([pd.to_numeric(q[c], errors="coerce") for c in QCOLS if c in q.columns]).dropna()
        qstats["__overall__"] = {"mean": round(float(allv.mean()), 2), "std": round(float(allv.std(ddof=1)), 2),
                                 "n": int(len(allv))}

    return {
        "demo": demo, "n": int(n),
        "pre": {"mean": round(float(pre.mean()), 1), "std": round(float(pre.std(ddof=1)), 1)},
        "post": {"mean": round(float(post.mean()), 1), "std": round(float(post.std(ddof=1)), 1)},
        "gain": round(float(diff.mean()), 1),
        "paired_t": {"t": round(float(t_stat), 3), "p": float(p_t)},
        "wilcoxon_p": float(p_w) if p_w == p_w else None,
        "cohen_d": round(d, 2),
        "completion_rate": round(done_rate, 3),
        "duration_min": {"mean": round(float(dur.mean()), 1), "median": round(float(dur.median()), 1),
                         "min": int(dur.min()), "max": int(dur.max())} if len(dur) else None,
        "questionnaire": qstats,
    }


def judge(r: dict) -> list[tuple[str, str, str]]:
    """按事前约定的合格线判读（口径写死在 教学试点方案.md 第 6 节）。"""
    out = []
    out.append(("知识点掌握提升", f"{r['gain']} 分", "✅ 达标（≥15）" if r["gain"] >= 15 else "❌ 未达标（<15）"))
    p = r["paired_t"]["p"]
    out.append(("统计显著性", f"p = {p:.3g}", "✅ 显著（p<0.05）" if p < 0.05 else "❌ 不显著"))
    d = r["cohen_d"]
    out.append(("效应量 d", f"{d}", "✅ 大效应（≥0.8）" if d >= 0.8 else ("△ 中等" if d >= 0.5 else "❌ 小效应")))
    cr = r["completion_rate"]
    out.append(("实验完成率", f"{cr:.1%}", "✅ 达标（≥90%）" if cr >= 0.9 else "❌ 未达标"))
    qo = r["questionnaire"].get("__overall__", {}).get("mean")
    out.append(("问卷总均分", f"{qo}", "✅ 达标（≥4.0）" if (qo or 0) >= 4.0 else "❌ 未达标"))
    q9 = r["questionnaire"].get("Q9继续使用", {}).get("mean")
    out.append(("继续使用意愿", f"{q9}", "✅ 达标（≥4.0）" if (q9 or 0) >= 4.0 else "❌ 未达标"))
    return out


# ---------------------------------------------------------------- 产物
def make_figure(g: pd.DataFrame, q: pd.DataFrame, out: Path, demo: bool) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    out.mkdir(parents=True, exist_ok=True)
    made = []

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    pre, post = g["前测得分"].to_numpy(float), g["后测得分"].to_numpy(float)
    ax[0].bar(["前测", "后测"], [pre.mean(), post.mean()], color=["#8fb8e0", "#2f6fb5"], width=0.55)
    for i, v in enumerate([pre.mean(), post.mean()]):
        ax[0].text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=11)
    ax[0].set_ylabel("平均得分")
    ax[0].set_title("前测 / 后测平均得分" + ("（模拟数据）" if demo else ""))
    ax[0].set_ylim(0, 105)
    for i in range(len(pre)):
        ax[1].plot([0, 1], [pre[i], post[i]], color="#bbbbbb", lw=0.8, zorder=1)
    ax[1].scatter(np.zeros(len(pre)), pre, color="#8fb8e0", zorder=2, s=14)
    ax[1].scatter(np.ones(len(post)), post, color="#2f6fb5", zorder=2, s=14)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["前测", "后测"])
    ax[1].set_title("逐人配对变化" + ("（模拟数据）" if demo else ""))
    fig.tight_layout()
    p1 = out / "pilot_pre_post.png"
    fig.savefig(p1, dpi=140); plt.close(fig); made.append(str(p1.relative_to(ROOT)))

    if len(q):
        means = [pd.to_numeric(q[c], errors="coerce").mean() for c in QCOLS if c in q.columns]
        labels = [c.split("Q")[-1][:6] for c in QCOLS if c in q.columns]
        fig, ax = plt.subplots(figsize=(11, 4.0))
        ax.bar(range(len(means)), means, color="#4a8fd4")
        ax.axhline(4.0, color="#d9534f", ls="--", lw=1.2, label="合格线 4.0")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 5.2); ax.set_ylabel("均分（5 级）")
        ax.set_title("满意度问卷各题均分" + ("（模拟数据）" if demo else ""))
        ax.legend()
        fig.tight_layout()
        p2 = out / "pilot_questionnaire.png"
        fig.savefig(p2, dpi=140); plt.close(fig); made.append(str(p2.relative_to(ROOT)))
    return made


def write_report(r: dict, figs: list[str], out: Path, demo: bool) -> Path:
    L = []
    tag = "【模拟数据，不得提交】\n\n" if demo else ""
    L.append(tag + "# 教学试点效果分析\n")
    if demo:
        L.append("> ⚠️ **本文件由 `--demo` 生成，数据为模拟值，仅用于演示流程与格式。**"
                 "正式提交前必须用真实采集数据重跑，并删除本提示。\n")
    L.append(f"- 有效样本数：**{r['n']}** 人")
    L.append(f"- 前测：**{r['pre']['mean']} ± {r['pre']['std']}**　后测：**{r['post']['mean']} ± {r['post']['std']}**"
             f"　提升：**{r['gain']} 分**")
    L.append(f"- 配对 t 检验：t = {r['paired_t']['t']}，p = {r['paired_t']['p']:.3g}"
             + (f"；Wilcoxon p = {r['wilcoxon_p']:.3g}" if r.get("wilcoxon_p") else ""))
    L.append(f"- 效应量 Cohen's d = **{r['cohen_d']}**")
    L.append(f"- 实验完成率：**{r['completion_rate']:.1%}**")
    if r.get("duration_min"):
        dm = r["duration_min"]
        L.append(f"- 实验耗时：均值 **{dm['mean']} min**（中位 {dm['median']}，区间 {dm['min']}~{dm['max']}）")
    L.append("\n## 满意度问卷（5 级）\n")
    L.append("| 题项 | 均分 | 标准差 | 人数 |")
    L.append("|---|---|---|---|")
    for c in QCOLS:
        if c in r["questionnaire"]:
            v = r["questionnaire"][c]
            L.append(f"| {c} | {v['mean']} | {v['std']} | {v['n']} |")
    ov = r["questionnaire"].get("__overall__")
    if ov:
        L.append(f"| **总均分** | **{ov['mean']}** | {ov['std']} | {ov['n']} |")
    L.append("\n## 按事前口径判读\n")
    L.append("| 指标 | 实测 | 判读 |")
    L.append("|---|---|---|")
    for a, b, c in judge(r):
        L.append(f"| {a} | {b} | {c} |")
    L.append("\n## 图表\n")
    for f in figs:
        L.append(f"- `{f}`")
    L.append("\n> 判读口径在试点前确定（见 `competition/教学试点方案.md` 第 6 节），未做事后调整。")
    p = out / "pilot_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def write_demo_extras(g: pd.DataFrame, q: pd.DataFrame, r: dict, out: Path) -> list[str]:
    """--demo 专用：把"模拟课堂"还原成可自看的三份材料（采集表已填版 / 课堂记录 / 申报书段落预览）。

    这些产物**只用于自己预览流程与格式**，文件头均标注【模拟数据·仅供自看】，不得进任何提交材料。
    """
    out.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    banner = "【模拟数据·仅供自看，禁止写入任何提交材料】"

    # ① 已填写的采集表（xlsx）
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws0 = wb.active
        ws0.title = "说明"
        ws0["A1"] = banner
        ws0["A3"] = "本工作簿由 tools/pilot_analysis.py --demo 生成，数据为模拟值，仅用于预览填表格式。"
        ws0["A4"] = "正式采集请使用 competition/教学试点数据采集表.xlsx（该表只有空行与示例行）。"

        ws1 = wb.create_sheet(GRADE_SHEET)
        ws1.append([banner])
        ws1.append(["匿名编号", "前测得分", "后测得分", "实验耗时(min)", "是否完成全部步骤", "备注"])
        for _, row in g.iterrows():
            ws1.append([row["匿名编号"], row["前测得分"], row["后测得分"],
                        row["实验耗时(min)"], row["是否完成全部步骤"], ""])

        ws2 = wb.create_sheet(QUEST_SHEET)
        ws2.append([banner])
        ws2.append(["匿名编号"] + QCOLS)
        for _, row in q.iterrows():
            ws2.append([row["匿名编号"]] + [int(row[c]) for c in QCOLS if c in q.columns])

        ws3 = wb.create_sheet("汇总看板")
        ws3.append([banner])
        ws3.append(["指标", "数值"])
        for k, v in [("有效样本数", r["n"]),
                     ("前测均分", f'{r["pre"]["mean"]} ± {r["pre"]["std"]}'),
                     ("后测均分", f'{r["post"]["mean"]} ± {r["post"]["std"]}'),
                     ("提升幅度", f'{r["gain"]} 分'),
                     ("配对 t 检验 p", f'{r["paired_t"]["p"]:.3g}'),
                     ("效应量 Cohen's d", r["cohen_d"]),
                     ("实验完成率", f'{r["completion_rate"]:.1%}'),
                     ("平均实验耗时", f'{r["duration_min"]["mean"]} min' if r.get("duration_min") else "—"),
                     ("问卷总均分", r["questionnaire"].get("__overall__", {}).get("mean")),
                     ("继续使用意愿", r["questionnaire"].get("Q9继续使用", {}).get("mean"))]:
            ws3.append([k, str(v)])
        p = out / "教学试点数据采集表_模拟填写版.xlsx"
        wb.save(p)
        made.append(str(p.relative_to(ROOT)))
    except Exception as e:  # noqa: BLE001
        print(f"  (模拟采集表生成失败：{type(e).__name__}: {e})")

    # ② 课堂还原记录
    import numpy as np
    dur = g["实验耗时(min)"].to_numpy(int)
    done = g["是否完成全部步骤"].astype(str).str.startswith("是").to_numpy()
    L = [banner, "", "# 模拟课堂还原记录（90 分钟）", "",
         "> 本文是 `--demo` 生成的**模拟教学过程**，用于预览真实试点时应该记录哪些内容。", "",
         "## 时间线", "",
         "| 时段 | 环节 | 现场情况（模拟） |", "|---|---|---|",
         f"| 0–10 min | 前测 | {r['n']} 人参加，有效卷 {r['n']} 份；均分 **{r['pre']['mean']}**（SD {r['pre']['std']}） |",
         "| 10–25 min | 讲解与演示 | 讲解 SDN 三层架构与流表统计；演示一次完整检测 |",
         "| 25–45 min | 实验一·观察正常流量 | 18 组中 17 组一次启动成功，1 组因端口占用重试一次（约 2 分钟） |",
         "| 45–70 min | 实验二·发起攻击并观察闭环 | 各组发起 SYN Flood / UDP Flood；首个告警平均约 12 s 内出现；阻断后流量回落 |",
         "| 70–80 min | 实验三·读报告、讲边界 | 有学生追问\"为什么同域 99% 到真实网络就不行\"，引出目标域标定 |",
         f"| 80–90 min | 后测 + 问卷 | 后测均分 **{r['post']['mean']}**（提升 {r['gain']} 分）；问卷总均分 "
         f"**{r['questionnaire'].get('__overall__', {}).get('mean')}** |",
         "", "## 逐人记录（模拟）", "",
         "| 匿名编号 | 前测 | 后测 | 提升 | 耗时(min) | 完成 |", "|---|---|---|---|---|---|"]
    for _, row in g.iterrows():
        L.append(f"| {row['匿名编号']} | {row['前测得分']} | {row['后测得分']} | "
                 f"{int(row['后测得分']) - int(row['前测得分'])} | {row['实验耗时(min)']} | {row['是否完成全部步骤']} |")
    L += ["", "## 现场观察（模拟）", "",
          f"- 完成率 **{r['completion_rate']:.1%}**（{int(done.sum())}/{len(done)}）",
          f"- 平均耗时 **{dur.mean():.1f} min**（最短 {dur.min()}，最长 {dur.max()}）",
          "- 常见卡点：拓扑启动等待时间（约 1 分钟）；有 3 组第一次没找到告警面板位置",
          "- 教学亮点：实验三的\"误报/漏报 + 跨环境为什么失效\"讨论最活跃，多名学生主动提问",
          "", "## 反馈摘录（模拟）", "",
          "1. \"以前只在课本上看过入侵检测，这次自己发攻击、看它被拦住，印象深多了。\"",
          "2. \"原来准确率 99% 不代表能用——老师讲的标定那部分最颠覆我的认知。\"",
          "3. \"在虚拟机里随便打，不用担心把网络搞坏，这点很关键。\"",
          "4. （教师）\"平台把攻击到阻断的全过程可视化了，课堂上不用再靠讲。\"",
          "5. \"建议增加一个可以自己改阈值、观察误报变化的功能。\"",
          "", "> ⚠️ 以上全部为模拟内容，仅用于预览记录格式；真实试点须逐项据实填写。"]
    p2 = out / "模拟课堂记录.md"
    p2.write_text("\n".join(L) + "\n", encoding="utf-8")
    made.append(str(p2.relative_to(ROOT)))

    # ③ 申报书段落预览（若这些是真数据，会长什么样）
    qs = r["questionnaire"]
    L2 = [banner, "", "# 申报书「（七）应用效果」4/5 段落预览", "",
          "> 下面是**假设以上数据为真实采集**时，可直接粘进申报书的写法。**仅为预览，禁止提交。**", "",
          "**4. 教学试点数据**", "",
          f"- 试点形式：小规模课堂试点（《网络攻防》实验课），有效样本 **{r['n']} 人**；",
          f"- 知识点掌握：前测 **{r['pre']['mean']} ± {r['pre']['std']}** → 后测 "
          f"**{r['post']['mean']} ± {r['post']['std']}**，平均提升 **{r['gain']} 分**；"
          f"配对 t 检验 p = {r['paired_t']['p']:.2g}，Cohen's d = {r['cohen_d']}（大效应）；",
          f"- 实验完成率 **{r['completion_rate']:.1%}**，平均完成时间 "
          f"**{r['duration_min']['mean']} 分钟**；",
          f"- 平台满意度总均分 **{qs.get('__overall__', {}).get('mean')}**（5 级），"
          f"\"愿意继续使用\" **{qs.get('Q9继续使用', {}).get('mean')}**、"
          f"\"愿意推荐\" **{qs.get('Q10推荐意愿', {}).get('mean')}**；",
          "- 所有数据匿名化处理（仅保留匿名编号），自愿参与并可随时退出；", "",
          "**5. 用户反馈摘录**", "", "（引用「反馈摘录（模拟）」中的 3~5 条）", "",
          "> 真实提交时：把\"小规模课堂试点\"换成实际形式（课堂/远程体验）、人数、时间，并附现场照片与签到记录。"]
    p3 = out / "申报书段落预览.md"
    p3.write_text("\n".join(L2) + "\n", encoding="utf-8")
    made.append(str(p3.relative_to(ROOT)))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="教学试点数据分析")
    ap.add_argument("--xlsx", default=str(ROOT / "competition" / "教学试点数据采集表.xlsx"))
    ap.add_argument("--demo", action="store_true", help="用模拟数据演示全流程（产物进 results/pilot_demo/）")
    ap.add_argument("--include-example", action="store_true", help="把示例行也计入（仅调试用）")
    args = ap.parse_args(argv)

    if args.demo:
        g, q = make_demo()
        out = ROOT / "results" / "pilot_demo"
        print("⚠️  DEMO 模式：使用模拟数据，产物写入 results/pilot_demo/，不得提交。")
    else:
        xlsx = Path(args.xlsx)
        if not xlsx.exists():
            print(f"找不到采集表：{xlsx}")
            return 2
        g, q = load_real(xlsx, args.include_example)
        out = ROOT / "results" / "pilot"
        if len(g) == 0:
            print("❌ 采集表里没有实测数据（只剩示例行）。")
            print("   请先按 competition/教学试点方案.md 完成试点并录入数据；")
            print("   若只想看流程与格式，运行：python tools/pilot_analysis.py --demo")
            return 3

    r = analyze(g, q, args.demo)
    figs = make_figure(g, q, (ROOT / "results" / "figures") if not args.demo else out / "figures", args.demo)
    rep = write_report(r, figs, out, args.demo)
    (out / "pilot_summary.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n有效样本 {r['n']}｜前测 {r['pre']['mean']} → 后测 {r['post']['mean']}（+{r['gain']}）"
          f"｜p={r['paired_t']['p']:.3g}｜d={r['cohen_d']}｜完成率 {r['completion_rate']:.1%}")
    for a, b, c in judge(r):
        print(f"  {a}: {b}  {c}")
    print(f"\n产物：{rep}")
    print(f"      {out / 'pilot_summary.json'}")
    for f in figs:
        print(f"      {ROOT / f}")
    if args.demo:
        extras = write_demo_extras(g, q, r, out)
        print("\n模拟课堂附加产物（仅供自看）：")
        for f in extras:
            print(f"      {ROOT / f}")
        print("\n⚠️  以上全部为模拟数据。产物只落在 results/pilot_demo/，禁止写入任何提交材料；")
        print("    真实试点完成后，改用不带 --demo 的命令重新生成 results/pilot/。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
