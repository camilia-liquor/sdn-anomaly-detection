"""部署工作点选择器：把「判定阈值 × 窗口长度 × 告警确认次数」三旋钮量化成一张 Pareto 表。

为什么这是一个值得单独做的工具
------------------------------
模型给出的是概率，"能不能上线"取决于**误报预算**：安全运营团队能接受每小时几次误报，
决定了阈值该定多高、窗口该多长、要不要连续确认。这三件事相互耦合，
论文里通常只报一个固定阈值下的 F1，无法回答"我们该把旋钮拧到哪里"。

本工具直接复用评估阶段缓存的在线概率（`artifacts/online_scores_*.npz`），
**无需重新训练或推理**，就能枚举所有工作点并给出：
  * 攻击片段召回率（漏报风险）
  * 平均 / P95 检测时延（响应速度）
  * 误报次数与误报率（次/小时，运维负担）

并画出 Pareto 前沿，标出"在给定误报预算下召回最高的点"。

用法：
    python tools/operating_point.py                          # 主模型，默认网格
    python tools/operating_point.py --tag cnn_bilstm_main_s0 --budget 5
    python tools/operating_point.py --thresholds 0.3,0.5,0.7,0.9 --confirms 1,2,3,5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData  # noqa: E402
from src.evaluate import online_alarm_evaluation  # noqa: E402
from src.utils import ARTIFACT_DIR, FIGURE_DIR, RESULT_DIR, ensure_dirs, get_logger, markdown_table, save_json, save_text, set_seed  # noqa: E402

LOGGER = get_logger("cida.oppoint")


def load_scores(tag: str) -> tuple[np.ndarray, np.ndarray]:
    path = ARTIFACT_DIR / f"online_scores_{tag}.npz"
    if not path.exists():
        raise SystemExit(
            f"找不到 {path.name}。请先运行一次评估以生成在线概率缓存：\n"
            f"  python run_all.py --tag {tag.split('_s')[0]} --variants cnn_bilstm --seeds 0 --no-figures"
        )
    z = np.load(path)
    return np.asarray(z["p_attack"], dtype=float), np.asarray(z["starts"], dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="部署工作点选择器（阈值 × 窗口 × 确认次数）")
    ap.add_argument("--tag", default="cnn_bilstm_main_s0", help="使用哪个模型的在线概率缓存")
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.9", help="候选判定阈值")
    ap.add_argument("--confirms", default="1,2,3,5", help="候选告警确认窗口数")
    ap.add_argument("--budget", type=float, default=10.0, help="误报预算（次/小时），用于标出推荐工作点")
    ap.add_argument("--out", default="operating_point", help="输出文件名前缀")
    args = ap.parse_args(argv)

    ensure_dirs()
    cfg = Config()
    set_seed(cfg.seed)
    data = PreparedData.load()
    p_attack, starts = load_scores(args.tag)
    LOGGER.info("载入在线概率：%d 个窗口（窗口长度 T=%d，在线步长=%d）", len(p_attack), cfg.window_size, cfg.online_stride)

    thresholds = [float(x) for x in str(args.thresholds).split(",") if x.strip()]
    confirms = [int(x) for x in str(args.confirms).split(",") if x.strip()]

    rows, raw = [], []
    for thr in thresholds:
        for cf in confirms:
            m = online_alarm_evaluation(data, p_attack, starts, cfg, threshold=thr, confirm=cf)
            rec = float(m["segment_recall"] or 0.0)
            fa = m["false_alarms_per_hour"]
            delay = m["mean_detection_delay_sec"]
            raw.append({"threshold": thr, "confirm": cf, **{k: m[k] for k in (
                "segment_recall", "n_segments_detected", "n_attack_segments", "false_alarms",
                "false_alarms_per_hour", "mean_detection_delay_sec", "p95_detection_delay_sec",
                "n_alarmed_windows", "n_raw_alarmed_windows", "alarm_rate")}})
            rows.append([
                f"{thr:.2f}", str(cf),
                f"{rec:.3f}", f"{m['n_segments_detected']}/{m['n_attack_segments']}",
                f"{m['false_alarms']}", f"{fa:.2f}" if fa is not None else "-",
                f"{delay*1000:.1f}" if delay is not None else "-",
                f"{(m['p95_detection_delay_sec'] or 0)*1000:.1f}",
                f"{m['n_alarmed_windows']}",
            ])

    table = markdown_table(
        ["阈值", "确认次数", "片段召回", "检出/总片段", "误报次数", "误报/小时", "平均时延(ms)", "P95时延(ms)", "告警窗口数"],
        rows,
    )
    print("\n=== 检测工作点全表（窗口 T=%d）===" % cfg.window_size)
    print(table)

    # ---- 推荐工作点：在误报预算内召回最高（并列时取时延最小）----
    feasible = [r for r in raw if (r["false_alarms_per_hour"] or 0) <= args.budget and r["segment_recall"] >= 0.999]
    best = min(feasible, key=lambda r: (r["mean_detection_delay_sec"] or 1e9)) if feasible else None
    if best:
        msg = (f"在误报预算 {args.budget:g} 次/小时内且片段召回 100% 的工作点中，"
               f"**阈值 {best['threshold']:.2f} + 连续确认 {best['confirm']} 个窗口** 时延最小："
               f"平均 {(best['mean_detection_delay_sec'] or 0)*1000:.1f} ms、"
               f"误报 {best['false_alarms_per_hour']:.2f} 次/小时。")
    else:
        msg = f"在误报预算 {args.budget:g} 次/小时内没有召回 100% 的工作点，需要放宽预算或接受漏报。"
    print("\n" + msg)
    LOGGER.info(msg.replace("**", ""))

    # ---- 落盘 ----
    save_json({"tag": args.tag, "window_size": cfg.window_size, "budget": args.budget,
               "table": raw, "recommended": best}, RESULT_DIR / f"{args.out}.json")
    text = (
        f"# 检测工作点选择（模型 `{args.tag}`，窗口 T={cfg.window_size}，在线步长 {cfg.online_stride}）\n\n"
        "三旋钮的含义：**阈值**越高越保守；**窗口**越长上下文越多；**确认次数**要求连续多个窗口都超阈值才报警。\n"
        "三者共同决定「召回–时延–误报」的权衡，下表是全枚举结果（无需重训即可得到）：\n\n"
        f"{table}\n\n## 推荐工作点\n\n{msg}\n\n"
        "> 读法：先按业务能接受的**误报预算**（次/小时）划一条线，再在这条线以上挑**召回最高、时延最小**的组合。\n"
        "> 本表由 `tools/operating_point.py` 生成，可复现。\n"
    )
    save_text(text, RESULT_DIR / f"{args.out}.md")

    # ---- Pareto 图 ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from src.visualize import init as viz_init

        viz_init(cfg)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for cf in confirms:
            pts = [r for r in raw if r["confirm"] == cf and r["false_alarms_per_hour"] is not None]
            xs = [p["false_alarms_per_hour"] for p in pts]
            axes[0].plot(xs, [p["segment_recall"] for p in pts], marker="o", label=f"确认 {cf} 个窗口")
            axes[1].plot(xs, [(p["mean_detection_delay_sec"] or 0) * 1000 for p in pts],
                         marker="o", label=f"确认 {cf} 个窗口")
        axes[0].axvline(args.budget, ls="--", c="red", lw=1, label=f"误报预算 {args.budget:g}/h")
        axes[1].axvline(args.budget, ls="--", c="red", lw=1, label=f"误报预算 {args.budget:g}/h")
        axes[0].set_xlabel("误报率（次/小时）"); axes[0].set_ylabel("攻击片段召回率")
        axes[0].set_title("召回 vs 误报：选择工作点")
        axes[1].set_xlabel("误报率（次/小时）"); axes[1].set_ylabel("平均检测时延（ms）")
        axes[1].set_title("时延 vs 误报：确认次数越多误报越低、时延略增")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.tight_layout()
        out_png = FIGURE_DIR / f"10_operating_point_{args.tag}.png"
        fig.savefig(out_png)
        plt.close(fig)
        LOGGER.info("Pareto 图已保存 → %s", out_png)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Pareto 图生成失败：%s", exc)

    LOGGER.info("工作点报告已保存 → %s", RESULT_DIR / f"{args.out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
