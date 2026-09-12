"""错误分析：混淆对、逐类错误、置信度分布、错例的特征画像。

为什么单独做这一节
------------------
评审规则在"验证方法"里明确列出可用 **错误分析**；只报一个总准确率无法说明模型"错在哪里、为什么错"。
本工具复用评估阶段缓存的窗口级预测（`artifacts/predictions_windows_*.csv`）
与测试张量（`data/processed/tensors.npz`），**无需重新推理**，直接产出：

1. **混淆对排行**：最常见的"真实类 → 预测类"错误组合及其中位置信度；
2. **逐类错误率**：哪些攻击类最难识别；
3. **置信度分布**：正确/错误样本的置信度对比（判断模型是否"错得还很自信"）；
4. **错例特征画像**：对最主要的混淆对，找出让模型犯错的**具体特征**（z 分数差异最大的前若干维）；
5. 一张三联图（混淆对条形图 / 置信度直方图 / 错例特征偏移）。

用法：
    python tools/error_analysis.py
    python tools/error_analysis.py --tag cnn_bilstm_main_s0 --top 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR,
    FIGURE_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    ensure_dirs,
    get_logger,
    load_json,
    markdown_table,
    resolve_dir,
    save_json,
    save_text,
    set_seed,
)

LOGGER = get_logger("cida.errors")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="错误分析（混淆对 / 逐类错误 / 置信度 / 错例特征）")
    ap.add_argument("--tag", default="cnn_bilstm_main_s0")
    ap.add_argument("--top", type=int, default=10, help="展示多少个混淆对")
    ap.add_argument("--top-features", type=int, default=8, help="错例特征画像展示多少个特征")
    args = ap.parse_args(argv)

    ensure_dirs()
    cfg = Config()
    set_seed(cfg.seed)

    csv = ARTIFACT_DIR / f"predictions_windows_{args.tag}.csv"
    if not csv.exists():
        raise SystemExit(f"找不到 {csv.name}，请先运行一次评估（run_all.py 或 tools/reevaluate.py --tag ...）")
    df = pd.read_csv(csv)
    data = PreparedData.load()
    meta = load_json(resolve_dir(getattr(cfg, "processed_dir", ""), PROCESSED_DIR) / "meta.json")
    feats = list(meta["feature_names"])
    class_names = list(meta["class_names"])

    aligned = len(df) == len(data.X_test)
    LOGGER.info("窗口级预测 %d 条（与测试张量对齐：%s）", len(df), aligned)

    # ---------- 1) 混淆对 ----------
    wrong = df[df["correct"] == 0]
    pairs = (wrong.groupby(["y_true_name", "y_pred_name"])
             .agg(n=("correct", "size"), conf=("p_attack", "median"))
             .reset_index().sort_values("n", ascending=False))
    total_wrong = max(len(wrong), 1)
    rows = [[r["y_true_name"], r["y_pred_name"], int(r["n"]),
             f"{r['n']/total_wrong:.1%}", f"{r['conf']:.3f}"] for _, r in pairs.head(args.top).iterrows()]
    pair_table = markdown_table(["真实类别", "被误判为", "次数", "占全部错误比例", "异常概率中位数"], rows)

    # ---------- 2) 逐类错误率 ----------
    per_class = []
    for c in class_names:
        sub = df[df["y_true_name"] == c]
        if len(sub) == 0:
            continue
        err = 1 - sub["correct"].mean()
        top_pair = ""
        if err > 0:
            vc = sub[sub["correct"] == 0]["y_pred_name"].value_counts()
            top_pair = f"{vc.index[0]}×{int(vc.iloc[0])}" if len(vc) else ""
        per_class.append([c, int(len(sub)), f"{err:.3%}", top_pair])
    per_class.sort(key=lambda x: -float(x[2].rstrip("%")))
    class_table = markdown_table(["类别", "测试窗口数", "错误率", "主要误判去向"], per_class)

    # ---------- 3) 置信度分布 ----------
    conf_ok = df[df["correct"] == 1]["p_attack"]
    conf_bad = wrong["p_attack"]
    conf_lines = markdown_table(
        ["样本", "数量", "异常概率均值", "异常概率中位数", "P90"],
        [["预测正确", f"{len(conf_ok)}", f"{conf_ok.mean():.4f}", f"{conf_ok.median():.4f}",
          f"{np.percentile(conf_ok, 90):.4f}" if len(conf_ok) else "-"],
         ["预测错误", f"{len(conf_bad)}", f"{conf_bad.mean():.4f}" if len(conf_bad) else "-",
          f"{conf_bad.median():.4f}" if len(conf_bad) else "-",
          f"{np.percentile(conf_bad, 90):.4f}" if len(conf_bad) else "-"]],
    )

    # ---------- 4) 错例特征画像（针对最主要的混淆对）----------
    feat_lines, feat_json = "", {}
    if aligned and len(pairs):
        t, p = pairs.iloc[0]["y_true_name"], pairs.iloc[0]["y_pred_name"]
        X = np.asarray(data.X_test, dtype=np.float32)
        mean_feat = X.mean(axis=(0, 1))                     # 每个特征的全局均值（z 空间）
        mask_err = (df["y_true_name"] == t) & (df["y_pred_name"] == p)
        mask_ok = (df["y_true_name"] == t) & (df["correct"] == 1)
        if mask_err.sum() >= 3 and mask_ok.sum() >= 3:
            v_err = X[mask_err.to_numpy()].mean(axis=(0, 1))
            v_ok = X[mask_ok.to_numpy()].mean(axis=(0, 1))
            delta = v_err - v_ok
            order = np.argsort(-np.abs(delta))[: args.top_features]
            feat_json = {"main_confusion_pair": [t, p], "n_error": int(mask_err.sum()),
                         "n_correct": int(mask_ok.sum()),
                         "top_features": [{"feature": feats[i], "delta_z": float(delta[i]),
                                           "z_error": float(v_err[i]), "z_correct": float(v_ok[i])} for i in order]}
            feat_lines = markdown_table(
                ["特征", "错例均值(z)", "同类正确样本均值(z)", "差异"],
                [[feats[i], f"{v_err[i]:+.2f}", f"{v_ok[i]:+.2f}", f"{delta[i]:+.2f}"] for i in order],
            )

    # ---------- 落盘 ----------
    text = [
        f"# 错误分析（模型 `{args.tag}`）",
        "",
        f"- 测试窗口 {len(df)} 个，错误 {len(wrong)} 个（错误率 {len(wrong)/max(len(df),1):.2%}）",
        "- 数据来源：评估阶段缓存的窗口级预测 + 测试张量（无需重新推理，可复现）",
        "",
        "## 1. 混淆对排行（模型最常把什么错认成什么）",
        "",
        pair_table,
        "",
        "## 2. 逐类错误率",
        "",
        class_table,
        "",
        "## 3. 置信度分布：错的那部分，模型有多「自信」？",
        "",
        conf_lines,
        "",
    ]
    if feat_lines:
        text += [
            f"## 4. 错例特征画像（主要混淆对：{feat_json['main_confusion_pair'][0]} → "
            f"{feat_json['main_confusion_pair'][1]}，错例 {feat_json['n_error']} 个 vs 同类正确 {feat_json['n_correct']} 个）",
            "",
            "下表给出「错例相对同类正确样本」在哪些特征上偏移最大（z 分数空间，已标准化）：",
            "",
            feat_lines,
            "",
            "> 解读：差异大的特征就是模型混淆的来源，也是**下一步做特征工程/加样本**的着力点。",
            "",
        ]
    text.append("> 本报告由 `tools/error_analysis.py` 生成，可复现。")
    save_text("\n".join(text), RESULT_DIR / "error_analysis.md")
    save_json({"tag": args.tag, "n_windows": int(len(df)), "n_wrong": int(len(wrong)),
               "per_class": per_class, "top_pairs": pairs.head(args.top).to_dict("records"),
               "feature_profile": feat_json}, RESULT_DIR / "error_analysis.json")
    print("\n".join(text[:24]))
    LOGGER.info("错误分析报告已保存 → %s", RESULT_DIR / "error_analysis.md")

    # ---------- 图 ----------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from src.visualize import init as viz_init

        viz_init(cfg)
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

        ax = axes[0]
        top = pairs.head(args.top).iloc[::-1]
        ax.barh([f"{a} → {b}" for a, b in zip(top["y_true_name"], top["y_pred_name"])], top["n"], color="#ef4444")
        ax.set_xlabel("错误窗口数")
        ax.set_title("混淆对排行（真实 → 预测）")

        ax = axes[1]
        bins = np.linspace(0, 1, 21)
        if len(conf_ok):
            ax.hist(conf_ok, bins=bins, alpha=0.7, label="预测正确", color="#22c55e")
        if len(conf_bad):
            ax.hist(conf_bad, bins=bins, alpha=0.7, label="预测错误", color="#ef4444")
        ax.set_xlabel("异常概率 p"); ax.set_ylabel("窗口数")
        ax.set_title("置信度分布"); ax.legend()

        ax = axes[2]
        if feat_json:
            fl = feat_json["top_features"][::-1]
            ax.barh([f["feature"][:22] for f in fl], [f["delta_z"] for f in fl],
                    color=["#2563eb" if f["delta_z"] > 0 else "#f59e0b" for f in fl])
            ax.axvline(0, c="gray", lw=1)
            ax.set_xlabel("错例 − 同类正确 的特征均值差（z）")
            ax.set_title("错例特征画像：模型被哪些特征带偏")
        fig.tight_layout()
        out_png = FIGURE_DIR / f"11_error_analysis_{args.tag}.png"
        fig.savefig(out_png)
        plt.close(fig)
        LOGGER.info("错误分析图已保存 → %s", out_png)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("错误分析图生成失败：%s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
