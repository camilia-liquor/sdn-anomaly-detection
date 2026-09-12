"""逐类攻击的**在线告警指标**：不只是"总体召回 100%"，而是每一类各自多少。

为什么需要它
------------
`results/summary_*.md` 给的是**所有攻击片段合在一起**的在线指标（片段召回、平均时延、误报率），
`results/per_class_*.csv` 给的是**窗口级**的逐类精度/召回/F1。两者之间缺一张表：
**每一类攻击在"在线告警"口径下各自表现如何**——哪类报得快、哪类要等更久、有没有整段漏掉的类型、
误报发生在哪类攻击附近。运维排班和阈值调优看的正是这张表。

本工具直接复用 `src/evaluate.py` 的口径（同一套片段定义、同一套确认规则、同一套误报判定），
只是把统计**按类别拆开**，因此与 `results/summary_*.md` 的总体数字天然自洽（本文件会做一致性自检）。

用法::

    python tools/per_class_online.py                                   # 默认主模型 seed 0
    python tools/per_class_online.py --tag cnn_bilstm_main_s1
    python tools/per_class_online.py --threshold 0.7 --confirm 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.evaluate import confirm_alarms  # noqa: E402
from src.utils import ARTIFACT_DIR, FIGURE_DIR, RESULT_DIR, get_logger, save_json  # noqa: E402

LOGGER = get_logger("cida.per_class_online")


def load_stream(tag: str, processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """取测试流（标签与时间）与窗口级在线概率（告警用）。

    返回 ``(labels, times, p_attack, win_starts)``——保持平铺返回，别用嵌套元组（容易解包错）。
    """
    npz = np.load(processed_dir / "tensors.npz", allow_pickle=True)
    labels = np.asarray(npz["test_stream_labels"]).astype(str)
    times = np.asarray(npz["test_stream_time"], dtype=np.float64)
    scores_path = ARTIFACT_DIR / f"online_scores_{tag}.npz"
    if not scores_path.exists():
        raise SystemExit(f"找不到在线概率缓存：{scores_path}\n请先运行 python run_all.py --tag {tag}")
    z = np.load(scores_path, allow_pickle=True)
    p_attack = np.asarray(z["p_attack"], dtype=np.float64)
    win_starts = np.asarray(z["starts"], dtype=np.int64)
    return labels, times, p_attack, win_starts


def analyze(tag: str, processed_dir: Path, cfg: Config, threshold: float | None,
            confirm: int | None) -> dict:
    labels, times, p_attack, win_starts = load_stream(tag, processed_dir)
    n = len(labels)
    T = int(cfg.window_size)
    thr = float(cfg.report_threshold if threshold is None else threshold)
    n_confirm = int(confirm if confirm is not None else cfg.online_confirm_windows)
    min_len = int(max(T, getattr(cfg, "online_min_segment_rows", 10)))

    is_attack = labels != "Normal"

    # ---- 片段：两种切法都要算 ----
    # ① **逐类片段**（本表主体）：连续且**同类**的非 Normal 行算一段 → 能回答"每类各自表现"；
    # ② **合并片段**（官方口径，`src/evaluate.py`）：连续的非 Normal 行算一段，
    #    **相邻的不同类攻击会被合并**（例如 DDoS 紧接 DoS 只算一段）→ `summary_*.md` 里的 24 段就是这个。
    # 两者是同一批数据的两种切法，数量不同是正常的（本工具会各自对账，避免误判为 bug）。
    raw_segments: list[tuple[int, int, str]] = []
    merged_raw: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if is_attack[i]:
            j = i
            while j + 1 < n and is_attack[j + 1]:
                j += 1
            merged_raw.append((i, j))
            # 在这段连续攻击区间内部，再按类别切成若干个逐类片段
            k = i
            while k <= j:
                m = k
                while m + 1 <= j and labels[m + 1] == labels[k]:
                    m += 1
                raw_segments.append((k, m, str(labels[k])))
                k = m + 1
            i = j + 1
        else:
            i += 1
    segments = [(a, b, c) for a, b, c in raw_segments if (b - a + 1) >= min_len]
    n_ignored = len(raw_segments) - len(segments)
    merged_segments = [(a, b) for a, b in merged_raw if (b - a + 1) >= min_len]
    n_merged_ignored = len(merged_raw) - len(merged_segments)

    alarmed_raw = p_attack >= thr
    alarmed = confirm_alarms(alarmed_raw, n_confirm)
    span_sec = max(float(times[-1] - times[0]), 1e-6)
    hours = span_sec / 3600.0

    # ---- 每类的片段 / 检出 / 时延 ----
    per: dict[str, dict] = defaultdict(lambda: {"segments": 0, "detected": 0, "missed_lengths": [],
                                                "delays_rows": [], "delays_sec": [],
                                                "alarms_inside": 0, "segment_lengths": []})
    alarm_in_seg: list[int | None] = [None] * len(segments)
    for k, start in enumerate(win_starts):
        if not alarmed[k]:
            continue
        end = int(start) + T - 1
        for si, (s0, s1, _cls) in enumerate(segments):
            if not (end < s0 or int(start) > s1):
                per[segments[si][2]]["alarms_inside"] += 1
                if alarm_in_seg[si] is None:
                    alarm_in_seg[si] = k
                break

    for si, (s0, s1, cls) in enumerate(segments):
        d = per[cls]
        d["segments"] += 1
        d["segment_lengths"].append(int(s1 - s0 + 1))
        if alarm_in_seg[si] is None:
            d["missed_lengths"].append(int(s1 - s0 + 1))
            continue
        d["detected"] += 1
        end_row = int(win_starts[alarm_in_seg[si]] + T - 1)
        d["delays_rows"].append(int(end_row - s0))
        d["delays_sec"].append(float(times[min(end_row, n - 1)] - times[s0]))

    # ---- 误报：窗口内没有任何攻击行 ----
    false_total = 0
    false_nearest: Counter = Counter()
    for k, start in enumerate(win_starts):
        if not alarmed[k]:
            continue
        end = int(start) + T - 1
        if is_attack[max(int(start), 0): min(end + 1, n)].any():
            continue
        false_total += 1
        # 归因：把这次误报记到"时间上最近的攻击片段"所属类别上，便于定位是哪类攻击的工作点偏松
        best, best_dist = None, None
        for s0, s1, cls in segments:
            dist = 0 if (s0 <= end and int(start) <= s1) else min(abs(int(start) - s1), abs(s0 - end))
            if best_dist is None or dist < best_dist:
                best, best_dist = cls, dist
        false_nearest[best or "（无攻击片段）"] += 1

    classes = sorted(per.keys())
    rows = []
    for cls in classes:
        d = per[cls]
        rows.append({
            "class": cls,
            "n_segments": int(d["segments"]),
            "n_detected": int(d["detected"]),
            "segment_recall": float(d["detected"] / max(d["segments"], 1)),
            "n_missed": int(d["segments"] - d["detected"]),
            "missed_lengths": list(d["missed_lengths"]),
            "seg_len_min": int(min(d["segment_lengths"])) if d["segment_lengths"] else 0,
            "seg_len_median": float(np.median(d["segment_lengths"])) if d["segment_lengths"] else 0.0,
            "seg_len_max": int(max(d["segment_lengths"])) if d["segment_lengths"] else 0,
            "mean_delay_ms": float(np.mean(d["delays_sec"]) * 1000.0) if d["delays_sec"] else None,
            "median_delay_ms": float(np.median(d["delays_sec"]) * 1000.0) if d["delays_sec"] else None,
            "p95_delay_ms": float(np.percentile(d["delays_sec"], 95) * 1000.0) if d["delays_sec"] else None,
            "mean_delay_rows": float(np.mean(d["delays_rows"])) if d["delays_rows"] else None,
            "alarms_inside": int(d["alarms_inside"]),
            "alarms_per_segment": float(d["alarms_inside"] / max(d["segments"], 1)),
        })

    # ---- 合并片段口径的检出统计（用于与 summary_*.md / metrics_*.json 对账）----
    merged_detected = 0
    for s0, s1 in merged_segments:
        for k, start in enumerate(win_starts):
            if not alarmed[k]:
                continue
            end = int(start) + T - 1
            if not (end < s0 or int(start) > s1):
                merged_detected += 1
                break

    return {
        "tag": tag,
        "threshold": thr,
        "confirm_windows": n_confirm,
        "window_size": T,
        "min_segment_rows": min_len,
        "stream_rows": int(n),
        "stream_hours": round(hours, 4),
        "totals": {
            "n_raw_segments": len(raw_segments),
            "n_ignored_short": int(n_ignored),
            "n_segments": len(segments),
            "n_detected": int(sum(1 for x in alarm_in_seg if x is not None)),
            "false_alarms": int(false_total),
            "false_alarms_per_hour": float(false_total / hours) if hours > 0 else None,
            "n_alarmed_windows": int(alarmed.sum()),
        },
        "merged_segments": {
            "definition": "连续的非 Normal 行算一段（相邻不同类攻击合并）——与 src/evaluate.py 同口径",
            "n_raw": len(merged_raw),
            "n_ignored_short": int(n_merged_ignored),
            "n_segments": len(merged_segments),
            "n_detected": int(merged_detected),
            "segment_recall": float(merged_detected / max(len(merged_segments), 1)),
        },
        "per_class": rows,
        "false_alarm_nearest_class": dict(false_nearest),
    }


def _bar(rows: list[dict], out_png: Path) -> Path | None:
    try:
        from src.utils import setup_matplotlib

        cjk = setup_matplotlib(logger=LOGGER)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib 不可用，跳过绘图：%s", exc)
        return None
    _L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    cls = [r["class"] for r in rows]
    mean_ms = [r["mean_delay_ms"] or 0.0 for r in rows]
    p95 = [r["p95_delay_ms"] or 0.0 for r in rows]
    x = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(x - 0.2, mean_ms, width=0.4, label=_L("平均检测时延", "mean delay"))
    ax.bar(x + 0.2, p95, width=0.4, label=_L("P95 检测时延", "P95 delay"))
    for xi, r in zip(x, rows):
        rec = r["segment_recall"] * 100
        ax.text(xi, max(r["p95_delay_ms"] or 0.0, r["mean_delay_ms"] or 0.0) * 1.05 + 0.5,
                _L(f"召回 {rec:.0f}%\n{r['n_detected']}/{r['n_segments']}",
                   f"recall {rec:.0f}%\n{r['n_detected']}/{r['n_segments']}"),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cls, rotation=20, ha="right")
    ax.set_ylabel(_L("检测时延（毫秒，流时间）", "detection delay (ms, flow time)"))
    ax.set_title(_L("每类攻击的在线检测时延与片段召回", "Per-class online delay and segment recall"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def write_md(res: dict, out_md: Path, fig: Path | None, summary_json: Path | None) -> None:
    t = res["totals"]
    rows = sorted(res["per_class"], key=lambda r: (r["mean_delay_ms"] is None, r["mean_delay_ms"] or 0.0))
    lines: list[str] = []
    lines.append("# 逐类攻击的在线告警指标（哪类报得快、哪类要等更久）")
    lines.append("")
    lines.append(f"模型 `{res['tag']}` · 窗口 T={res['window_size']} · 阈值 {res['threshold']} · "
                 f"连续确认 {res['confirm_windows']} 窗口 · 片段长度下限 {res['min_segment_rows']} 行 · "
                 f"测试流 {res['stream_rows']:,} 条 / {res['stream_hours']:.2f} 小时")
    lines.append("")
    lines.append("> 确认规则、误报判定、片段长度下限与 `src/evaluate.py::online_alarm_evaluation` **完全一致**；"
                 "唯一的差别是**切段粒度**：本表按**类别**切（回答「每类各自表现」），"
                 "而 `summary_*.md` 按**是否攻击**切（相邻不同类攻击合并）。两种切法都在本文件里给出并对账。")
    lines.append("")
    lines.append("## 一、逐类明细")
    lines.append("")
    lines.append("| 攻击类型 | 片段数 | 检出 | 片段召回 | **平均时延(ms)** | 中位 | P95 | 折合流记录 | "
                 "片段内告警 | 每一段平均告警 | 片段长度(最小/中位/最大) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        delay = "—" if r["mean_delay_ms"] is None else f"**{r['mean_delay_ms']:.2f}**"
        med = "—" if r["median_delay_ms"] is None else f"{r['median_delay_ms']:.2f}"
        p95 = "—" if r["p95_delay_ms"] is None else f"{r['p95_delay_ms']:.2f}"
        drows = "—" if r["mean_delay_rows"] is None else f"{r['mean_delay_rows']:.1f}"
        lines.append(f"| **{r['class']}** | {r['n_segments']} | {r['n_detected']} | "
                     f"{r['segment_recall'] * 100:.0f}% | {delay} | {med} | {p95} | {drows} | "
                     f"{r['alarms_inside']} | {r['alarms_per_segment']:.1f} | "
                     f"{r['seg_len_min']}/{r['seg_len_median']:.0f}/{r['seg_len_max']} |")
    lines.append("")
    total_line = (f"合计：**逐类片段 {t['n_detected']}/{t['n_segments']} 全部检出**" if t["n_detected"] == t["n_segments"]
                  else f"合计：逐类片段 {t['n_detected']}/{t['n_segments']}")
    total_line += (f"；误报 **{t['false_alarms']}** 次"
                   + (f"（{t['false_alarms_per_hour']:.2f} 次/小时）" if t["false_alarms_per_hour"] else "")
                   + f"；另有 {t['n_ignored_short']} 个过短片段（< {res['min_segment_rows']} 行）按口径不计入分母。")
    lines.append(total_line)
    lines.append("")

    missed = [r for r in res["per_class"] if r["n_missed"]]
    if missed:
        lines.append("**漏掉的片段**：")
        for r in missed:
            lines.append(f"* {r['class']}：漏掉 {r['n_missed']} 段，长度分别为 {r['missed_lengths']}")
        lines.append("")
    else:
        lines.append("**没有漏掉任何片段**（逐类均为 100% 片段召回）。")
        lines.append("")

    if res["false_alarm_nearest_class"]:
        lines.append("## 二、误报发生在哪类攻击附近")
        lines.append("")
        lines.append("把每次误报归因到「时间上最近的攻击片段」所属类别，用于判断是哪类攻击的工作点偏松：")
        lines.append("")
        lines.append("| 最近的攻击类别 | 误报次数 |")
        lines.append("|---|---|")
        for cls, cnt in sorted(res["false_alarm_nearest_class"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {cls} | {cnt} |")
        lines.append("")
    else:
        lines.append("## 二、误报")
        lines.append("")
        lines.append("本工作点下**没有任何误报**。")
        lines.append("")

    lines.append("## 三、怎么用这张表")
    lines.append("")
    lines.append("* **排班与响应**：平均时延最长的那一类，决定了值班人员的响应节奏；")
    lines.append("* **阈值调优**：若某类误报集中（见第二节），可以在工作点表里单独看它附近的片段，"
                 "判断是否要收紧阈值或提高确认次数；")
    lines.append("* **与窗口级指标互补**：`results/per_class_*.csv` 回答「每一类的分类精度」，"
                 "本表回答「每一类**什么时候**被报出来」——两个问题不一样。")
    lines.append("")
    lines.append("## 四、口径与自洽性")
    lines.append("")
    mg = res.get("merged_segments") or {}
    lines.append("* **两种切法都在本文件里**，它们不是矛盾而是同一批数据的两种切法：")
    lines.append(f"  * **逐类片段**（本表主体）：连续且**同类**的非 Normal 行算一段 → 共 **{res['totals']['n_segments']}** 段，"
                 "回答「每一类各自表现如何」；")
    lines.append(f"  * **合并片段**（`src/evaluate.py` 口径）：连续的非 Normal 行算一段，"
                 f"**相邻的不同类攻击会被合并** → 共 **{mg.get('n_segments', '?')}** 段"
                 f"（检出 {mg.get('n_detected', '?')}，召回 "
                 f"{(mg.get('segment_recall') or 0) * 100:.0f}%），`results/summary_*.md` 里的段数就是这个。")
    lines.append("* 本工具会用**合并口径**与 `results/metrics_<tag>.json` 自动对账（见 `.json` 的 `consistency` 字段），"
                 "确保没有算错口径；一致说明两种切法都建立在同一批告警之上。")
    lines.append("* 时延是**流时间**口径（数据集真实时间戳差），不是墙上时间。")
    lines.append("* 过短片段（长度 < 片段下限）不进入分母——这是本工程一贯口径（见 `docs/01` 的问题清单）。")
    lines.append("")
    if fig:
        try:
            rel = fig.relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(fig)
        lines.append(f"图：`{rel}`")
        lines.append("")
    lines.append("> 本文件由 `tools/per_class_online.py` 生成，可复现。")
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="逐类攻击的在线告警指标")
    ap.add_argument("--tag", default="cnn_bilstm_main_s0")
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--confirm", type=int, default=None)
    ap.add_argument("--out-md", default=str(RESULT_DIR / "per_class_online.md"))
    ap.add_argument("--out-json", default=str(RESULT_DIR / "per_class_online.json"))
    ap.add_argument("--figure", default=str(FIGURE_DIR / "per_class_online_delay.png"))
    args = ap.parse_args(argv)

    cfg = Config()
    processed = Path(args.processed_dir) if args.processed_dir else ROOT / "data" / "processed"
    res = analyze(args.tag, processed, cfg, args.threshold, args.confirm)

    # 与总体在线指标对账：用**合并片段口径**（与 src/evaluate.py 相同）比对，才是同一个东西。
    cons: dict = {}
    mpath = RESULT_DIR / f"metrics_{args.tag}.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8")).get("online") or {}
        mg = res["merged_segments"]
        cons = {
            "source": mpath.name,
            "note": "对账用的是「合并片段」口径（连续非 Normal 行为一段），与 src/evaluate.py 一致；"
                    "本表主体是更细的「逐类片段」口径，段数天然更多。",
            "merged_n_attack_segments": [mg["n_segments"], m.get("n_attack_segments")],
            "merged_n_segments_detected": [mg["n_detected"], m.get("n_segments_detected")],
            "false_alarms": [res["totals"]["false_alarms"], m.get("false_alarms")],
            "match": (mg["n_segments"] == m.get("n_attack_segments")
                      and mg["n_detected"] == m.get("n_segments_detected")
                      and res["totals"]["false_alarms"] == m.get("false_alarms")),
        }
        if cons["match"]:
            LOGGER.info("与 %s 对账一致（合并片段口径）：片段 %d/%d、检出 %d、误报 %d",
                        mpath.name, mg["n_segments"], m.get("n_attack_segments"),
                        mg["n_detected"], res["totals"]["false_alarms"])
        else:
            LOGGER.warning("与 %s 对账不一致：%s", mpath.name, cons)
    res["consistency"] = cons

    fig = _bar(res["per_class"], Path(args.figure))
    write_md(res, Path(args.out_md), fig, mpath if mpath.exists() else None)
    save_json(res, Path(args.out_json))

    for r in sorted(res["per_class"], key=lambda x: (x["mean_delay_ms"] is None, x["mean_delay_ms"] or 0)):
        LOGGER.info("%-11s 片段 %2d/%2d 召回 %3.0f%% 平均时延 %s",
                    r["class"], r["n_detected"], r["n_segments"], r["segment_recall"] * 100,
                    "—" if r["mean_delay_ms"] is None else f"{r['mean_delay_ms']:.2f} ms")
    LOGGER.info("报告 → %s", args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
