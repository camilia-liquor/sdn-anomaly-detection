"""把「未知攻击（开集）检测」的实验结论落成**可用的运行时阈值**。

背景
----
`tools/openset_eval.py` 已经证明：把整类攻击从训练里拿掉后，多分类概率的**熵**（或 `1 - max softmax`、
能量分数）能把"训练时没见过"的窗口与已知窗口分开。但要真正在服务里用起来，还需要一个**阈值**，
而且阈值必须在**正常流量**上按误报预算标定——否则它只会变成新的误报来源。

本工具做的事（**不需要重训、不需要新数据**）：
1. 读既有的窗口级预测缓存 `artifacts/predictions_windows_<tag>.csv`（含 8 类概率与真实标签）；
2. 对每个窗口算两种新颖性分数：`entropy`（多分类概率熵）与 `msp`（1 − 最大类概率）；
3. **只在真实标签为 Normal 的窗口上**取分位数作为阈值（对应 10% / 1% / 0.1% 三档误报预算）；
4. 报告每个阈值下：正常窗口被判"未知"的比例（= 新增误报）、已知攻击窗口被判"未知"的比例
   （= 细粒度标签丢失率，但**告警仍然会发**）、以及命中率；
5. 输出 `results/openset_thresholds.json` 供 `src/service.py` 的 `unknown_threshold` 使用，
   并生成一张给部署方看对照表 `results/openset_thresholds.md`。

用法::

    python tools/openset_calibrate.py                     # 默认用主模型的窗口预测缓存
    python tools/openset_calibrate.py --tag cnn_bilstm_main_s0 --score entropy
    python tools/openset_calibrate.py --budgets 0.10,0.01,0.001

⚠️ 语义边界（必须说清，报告里也写了）：
生产模型**见过全部 8 类**，因此这个标记的含义是"**模型没见过这种形态**"（新颖性/低置信），
而不是一个已被验证的"未知攻击检测器"——后者只在 `tools/openset_eval.py` 的留出整类实验中验证过。
两者互补：本工具负责把可用的旋钮交到运维手里，留出实验负责说明这个旋钮的能力边界。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ARTIFACT_DIR, FIGURE_DIR, RESULT_DIR, get_logger, save_json  # noqa: E402

LOGGER = get_logger("cida.openset_cal")

NORMAL_NAMES = {"Normal", "normal", "BENIGN", "Benign", "0"}
SCORES = ("entropy", "msp")


def _prob_columns(df: pd.DataFrame) -> list[str]:
    """从窗口预测缓存里挑出 8 类的概率列（形如 p_Botnet / p_DDoS …）。"""
    skip = {"p_attack"}
    cols = [c for c in df.columns if c.startswith("p_") and c not in skip]
    if len(cols) < 2:
        raise SystemExit(f"预测缓存里找不到多分类概率列（形如 p_<类别>），实际列：{list(df.columns)}")
    return cols


def novelty_scores(probs: np.ndarray, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """由多分类概率算两种新颖性分数（**越大越可能是未知**）。"""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    entropy = -(p * np.log(p)).sum(axis=1)                      # 熵：分布越平越可疑
    msp = 1.0 - p.max(axis=1)                                   # 1 - 最大类概率
    return {"entropy": entropy, "msp": msp}


def calibrate(tag: str, score: str, budgets: list[float]) -> dict:
    path = ARTIFACT_DIR / f"predictions_windows_{tag}.csv"
    if not path.exists():
        raise SystemExit(f"找不到窗口预测缓存：{path}\n请先运行 python run_all.py（或指定 --tag）")
    df = pd.read_csv(path)
    label_col = "y_true_name" if "y_true_name" in df.columns else "y_true"
    if label_col not in df.columns:
        raise SystemExit(f"预测缓存里没有真实标签列（y_true_name/y_true），实际列：{list(df.columns)}")
    prob_cols = _prob_columns(df)
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    labels = df[label_col].astype(str).to_numpy()
    scores = novelty_scores(probs)[score]

    is_normal = np.isin(labels, list(NORMAL_NAMES))
    n_norm, n_atk = int(is_normal.sum()), int((~is_normal).sum())
    if n_norm == 0 or n_atk == 0:
        raise SystemExit(f"标签分布异常：Normal {n_norm} 条 / 攻击 {n_atk} 条，无法标定")

    s_norm, s_atk = scores[is_normal], scores[~is_normal]
    rows = []
    for b in sorted(budgets, reverse=True):
        thr = float(np.quantile(s_norm, 1.0 - b))
        false_novel = float((s_norm > thr).mean())          # 正常窗口被判"未知" = 新增误报
        reject_known = float((s_atk > thr).mean())          # 已知攻击被判"未知" = 细粒度标签丢失
        rows.append({
            "budget": float(b), "threshold": thr,
            "normal_flagged": false_novel, "known_attack_flagged": reject_known,
            "normal_flagged_count": int((s_norm > thr).sum()),
            "known_attack_flagged_count": int((s_atk > thr).sum()),
        })
    return {
        "tag": tag,
        "source": path.name,
        "score": score,
        "prob_columns": prob_cols,
        "n_windows": int(len(df)),
        "n_normal": n_norm,
        "n_attack": n_atk,
        "score_stats": {
            "normal_mean": float(s_norm.mean()), "normal_p50": float(np.median(s_norm)),
            "normal_p99": float(np.quantile(s_norm, 0.99)),
            "attack_mean": float(s_atk.mean()), "attack_p50": float(np.median(s_atk)),
        },
        # 两种分数都给，便于对比；默认推荐熵（本工程开集实验里跨场景最稳）
        "levels": rows,
        "recommended": rows[1] if len(rows) > 1 else rows[0],
        "semantics": (
            "生产模型见过全部 8 类，因此本阈值标记的是「模型没见过这种形态」（新颖性/低置信），"
            "不是已被验证的未知攻击检测器；后者只在 tools/openset_eval.py 的留出整类实验中验证过。"
        ),
    }


def figure(out: dict, path: Path) -> Path | None:
    try:
        from src.utils import setup_matplotlib

        cjk = setup_matplotlib(logger=LOGGER)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib 不可用，跳过绘图：%s", exc)
        return None
    _L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    lv = out["levels"]
    b = [x["budget"] for x in lv]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(b, [x["normal_flagged"] for x in lv], marker="o",
            label=_L("正常窗口被判「未知」", "normal flagged"))
    ax.plot(b, [x["known_attack_flagged"] for x in lv], marker="s",
            label=_L("已知攻击被判「未知」", "known attack flagged"))
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(_L("误报预算（越靠右越严格）", "false-alarm budget"))
    ax.set_ylabel(_L("被判为「未知」的比例", "flagged-as-unknown rate"))
    ax.set_title(_L("未知标记阈值的代价曲线（在正常窗口上标定）",
                    "Cost of the unknown-flag threshold (calibrated on normal windows)"))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_md(out: dict, out_md: Path, fig_path: Path | None) -> None:
    rec = out["recommended"]
    lines: list[str] = []
    lines.append("# 未知标记阈值标定（把开集结论落成运行时旋钮）")
    lines.append("")
    lines.append(f"数据来源：`{out['source']}`（{out['n_windows']:,} 个窗口：正常 {out['n_normal']:,}、"
                 f"攻击 {out['n_attack']:,}）；新颖性分数：**{out['score']}**。")
    lines.append("")
    lines.append("标定口径：**阈值只在真实标签为 Normal 的窗口上取分位数**，对应三档误报预算。")
    lines.append("")
    lines.append("| 误报预算 | 阈值 | 正常窗口被判「未知」（新增误报） | 已知攻击被判「未知」（丢细粒度标签） |")
    lines.append("|---|---|---|---|")
    for x in out["levels"]:
        lines.append(f"| {x['budget'] * 100:g}% | {x['threshold']:.4f} | "
                     f"{x['normal_flagged'] * 100:.2f}%（{x['normal_flagged_count']} 条） | "
                     f"{x['known_attack_flagged'] * 100:.2f}%（{x['known_attack_flagged_count']} 条） |")
    lines.append("")
    lines.append(f"**推荐**：误报预算 {rec['budget'] * 100:g}% → 阈值 **{rec['threshold']:.4f}**。")
    lines.append("")
    lines.append(f"分数分布：正常窗口均值 {out['score_stats']['normal_mean']:.4f}"
                 f"（中位 {out['score_stats']['normal_p50']:.4f}、99 分位 {out['score_stats']['normal_p99']:.4f}）；"
                 f"攻击窗口均值 {out['score_stats']['attack_mean']:.4f}"
                 f"（中位 {out['score_stats']['attack_p50']:.4f}）。")
    lines.append("")
    lines.append("## 怎么用")
    lines.append("")
    lines.append("```bash")
    lines.append("# 方式一：起服务时通过环境变量给定阈值（见 deploy/cida-detect.env）")
    lines.append(f"set CIDA_UNKNOWN_THRESHOLD={rec['threshold']:.4f}")
    lines.append("")
    lines.append("# 方式二：运行时热更新（不用重启）")
    lines.append(f"curl -X POST http://127.0.0.1:5000/api/policy -H \"Content-Type: application/json\" "
                 f"-d \"{{\\\"unknown_threshold\\\": {rec['threshold']:.4f}}}\"")
    lines.append("")
    lines.append("# 关闭未知标记：unknown_threshold 设为 0")
    lines.append("```")
    lines.append("")
    lines.append("打开后，超过阈值的窗口其判定类别会变成 `Unknown（未知攻击）`，"
                 "并照常产生告警与处置建议（**攻击的发现能力不受影响**，只是不再给出细分类别）。")
    lines.append("")
    lines.append("## ⚠️ 语义边界（必须和实验报告一起看）")
    lines.append("")
    lines.append("生产模型**见过全部 8 类**，所以这个标记的含义是「**模型没见过这种形态**」（新颖性/低置信），"
                 "**不是**一个已被验证的「未知攻击检测器」。后者的能力边界见 `results/openset_report.md`：")
    lines.append("")
    lines.append("* 把整类攻击从训练里拿掉时，该分数能把未知攻击与已知窗口分开（跨场景 AUROC ≈ 0.978）；")
    lines.append("* 但**留出的攻击越像正常流量，低误报预算下越检不出**（留出 Botnet 时 1% 预算只有 44.3%、"
                 "0.1% 预算仅 0.9%）；")
    lines.append("* 因此**不要用未知标记替代二分类告警**，它是叠加在告警之上的一层「要不要人工研判」的提示；")
    lines.append("* 换网络/换采集口径后阈值必须重新标定（正常流量的分数分布会变）。")
    lines.append("")
    if fig_path:
        try:
            rel = fig_path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(fig_path)
        lines.append(f"图：`{rel}`")
        lines.append("")
    lines.append("> 本文件由 `tools/openset_calibrate.py` 生成，可复现（不需要重训）。")
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="标定「未知攻击」标记阈值（落成运行时旋钮）")
    ap.add_argument("--tag", default="cnn_bilstm_main_s0", help="使用哪个模型的窗口级预测缓存")
    ap.add_argument("--score", default="entropy", choices=SCORES, help="新颖性分数（默认 entropy）")
    ap.add_argument("--budgets", default="0.10,0.01,0.001", help="误报预算列表（逗号分隔）")
    ap.add_argument("--out-json", default=str(RESULT_DIR / "openset_thresholds.json"))
    ap.add_argument("--out-md", default=str(RESULT_DIR / "openset_thresholds.md"))
    ap.add_argument("--figure", default=str(FIGURE_DIR / "openset_threshold_cost.png"))
    args = ap.parse_args(argv)

    budgets = [float(x) for x in str(args.budgets).split(",") if x.strip()]
    out = calibrate(args.tag, args.score, budgets)
    fig_path = figure(out, Path(args.figure))
    write_md(out, Path(args.out_md), fig_path)
    save_json(out, Path(args.out_json))

    rec = out["recommended"]
    LOGGER.info("推荐阈值（%s，误报预算 %g%%）：%.4f —— 正常被判未知 %.2f%%、已知攻击被判未知 %.2f%%",
                out["score"], rec["budget"] * 100, rec["threshold"],
                rec["normal_flagged"] * 100, rec["known_attack_flagged"] * 100)
    LOGGER.info("产物：%s / %s", args.out_md, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
