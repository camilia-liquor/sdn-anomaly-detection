"""可视化成图：论文 / 答辩 / 演示界面三处复用的统一图表。

产出（默认保存到 results/figures/）
-----------------------------------
* ``01_class_distribution``  数据类别分布（体现真实的不平衡）
* ``02_feature_profiles``    关键流特征在各类上的分布（说明仿真数据的物理合理性）
* ``03_training_curves``     训练/验证损失与准确率、学习率变化
* ``04_confusion_matrix``    混淆矩阵（原始计数 + 行归一化）
* ``05_multiclass_roc_pr``   多分类 OvR 的 ROC / PR 曲线
* ``06_binary_detection``    异常检测工作点：ROC、PR、阈值-F1 权衡
* ``07_online_timeline``     连续测试流上的真实攻击区间 vs 模型告警（最有说服力的一张）
* ``08_attention``           注意力热力图（模型盯住了窗口里的哪几条流）
* ``09_ablation``            消融实验对比
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data_pipeline import BINARY_NORMAL, PreparedData
from .utils import FIGURE_DIR, get_logger, setup_matplotlib

LOGGER = get_logger("cida.viz")
_ZH = True  # 是否用中文标注（取决于系统是否有中文字体）


def _L(zh: str, en: str) -> str:
    return zh if _ZH else en


def _finish(fig, path: Path, logger=LOGGER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("图表已保存：%s", path.name)
    return path


def init(cfg: Config | None = None) -> bool:
    global _ZH
    _ZH = setup_matplotlib(chinese=True, logger=LOGGER)
    return _ZH


# ======================================================================================
# 1. 数据画像
# ======================================================================================
def plot_class_distribution(data: PreparedData, out_dir: Path | None = None, tag: str = "") -> Path:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or FIGURE_DIR)
    names = data.class_names
    train_counts = [data.train_class_counts.get(n, 0) for n in names]
    test_counts = [int((np.asarray(data.test_stream_labels) == n).sum()) if data.test_stream_labels is not None else 0 for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.bar(x - 0.2, train_counts, 0.4, label=_L("训练集（窗口）", "Train (windows)"), color="#2563eb")
    ax.bar(x + 0.2, test_counts, 0.4, label=_L("测试流（流记录）", "Test stream (flows)"), color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(_L("样本数", "Count"))
    ax.set_title(_L("类别分布（攻击类天然稀少，需类别加权）", "Class distribution (naturally imbalanced)"))
    ax.legend()
    ax.set_yscale("log")

    ax = axes[1]
    tot = sum(train_counts) or 1
    ax.pie(train_counts, labels=names, autopct=lambda p: f"{p:.1f}%", startangle=90,
           colors=plt.cm.tab10(np.linspace(0, 1, len(names))))
    ax.set_title(_L(f"训练窗口占比（共 {tot:,} 个窗口）", f"Train share (total {tot:,})"))
    fig.tight_layout()
    return _finish(fig, out_dir / "01_class_distribution.png")


def plot_feature_profiles(cfg: Config, out_dir: Path | None = None, max_features: int = 6, tag: str = "") -> Path:
    """关键特征在各类流量上的分布（箱线图），说明仿真数据的物理合理性。"""
    import matplotlib.pyplot as plt

    from .data_pipeline import clean_dataframe, load_capture

    out_dir = Path(out_dir or FIGURE_DIR)
    df, _ = load_capture(cfg)
    df, feats, _, mc_col = clean_dataframe(df, cfg)
    # 选"物理含义明确、区分度大"的若干特征
    preferred = [
        "Flow Packets/s", "Flow Duration", "SYN Flag Count", "Idle Mean",
        "packet_count", "flow_duration_sec", "Total Fwd Packets", "Bwd IAT Mean",
    ]
    chosen = [c for c in preferred if c in feats][:max_features]
    if not chosen:
        chosen = feats[:max_features]
    classes = [c for c in df[mc_col].astype(str).unique()]
    classes = sorted(classes, key=lambda c: (c != BINARY_NORMAL, c))

    n = len(chosen)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), squeeze=False)
    for i, feat in enumerate(chosen):
        ax = axes[i // cols][i % cols]
        series = [df.loc[df[mc_col].astype(str) == c, feat].replace([np.inf, -np.inf], np.nan).dropna().to_numpy() for c in classes]
        try:
            ax.boxplot(series, labels=classes, showfliers=False)
        except TypeError:  # matplotlib >= 3.9
            ax.boxplot(series, tick_labels=classes, showfliers=False)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.set_yscale("symlog")
    fig.suptitle(_L("关键流特征在不同流量类型上的分布（物理量自洽）",
                    "Key flow features by traffic class"), fontsize=12)
    fig.tight_layout()
    return _finish(fig, out_dir / "02_feature_profiles.png")


# ======================================================================================
# 2. 训练过程
# ======================================================================================
def plot_training_curves(history: dict, out_dir: Path | None = None, tag: str = "") -> Path:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or FIGURE_DIR)
    epochs = np.arange(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    ax.plot(epochs, history.get("loss", []), label=_L("训练损失", "train loss"), color="#2563eb")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label=_L("验证损失", "val loss"), color="#dc2626")
        best = int(np.argmin(history["val_loss"]))
        ax.axvline(best + 1, ls="--", c="gray", lw=1, label=_L(f"最优 epoch={best + 1}", f"best epoch={best + 1}"))
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.set_title(_L("损失曲线", "Loss"))

    ax = axes[1]
    tr_acc = history.get("mc_accuracy", history.get("accuracy", []))
    va_acc = history.get("val_mc_accuracy", history.get("val_accuracy", []))
    if tr_acc:
        ax.plot(epochs, tr_acc, label=_L("训练准确率", "train acc"), color="#2563eb")
    if va_acc:
        ax.plot(epochs, va_acc, label=_L("验证准确率", "val acc"), color="#dc2626")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(); ax.set_title(_L("准确率", "Accuracy"))
    if va_acc:
        ax.set_ylim(max(0.0, min(va_acc) - 0.05), 1.005)

    ax = axes[2]
    if "lr" in history:
        ax.plot(epochs, history["lr"], color="#059669", marker="o", ms=3)
        ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title(_L("学习率调度（验证损失停滞则衰减）", "LR schedule"))

    fig.suptitle(_L(f"训练过程：{tag}", f"Training curves: {tag}"), fontsize=13)
    fig.tight_layout()
    return _finish(fig, out_dir / f"03_training_curves_{tag}.png" if tag else out_dir / "03_training_curves.png")


# ======================================================================================
# 3. 混淆矩阵 / ROC / PR
# ======================================================================================
def plot_confusion_matrix(report: dict, class_names: list[str], out_dir: Path | None = None, tag: str = "") -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir or FIGURE_DIR)
    cm = np.asarray(report["multiclass"]["confusion_matrix"])
    cmn = np.asarray(report["multiclass"]["confusion_matrix_normalized"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(_L("混淆矩阵（计数）", "Confusion matrix (counts)"))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="rocket_r", cbar=False, ax=axes[1], vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(_L("混淆矩阵（按真实类归一化）", "Confusion matrix (row-normalized)"))
    for ax in axes:
        ax.set_xlabel(_L("预测类别", "Predicted")); ax.set_ylabel(_L("真实类别", "True"))
    fig.suptitle(_L(f"分类结果：{tag}", f"Confusion: {tag}"), fontsize=13)
    fig.tight_layout()
    return _finish(fig, out_dir / f"04_confusion_matrix_{tag}.png" if tag else out_dir / "04_confusion_matrix.png")


def plot_multiclass_roc_pr(data: PreparedData, report: dict, out_dir: Path | None = None, tag: str = "") -> Path | None:
    """多分类 OvR 的 ROC 与 PR 曲线（需要在内存中有概率矩阵，故从 report 缓存读取）。"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    probs = report.get("_p_mc")
    if probs is None:
        return None
    probs = np.asarray(probs)
    out_dir = Path(out_dir or FIGURE_DIR)
    y = np.asarray(data.y_test_mc)
    names = data.class_names
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, name in enumerate(names):
        yt = (y == i).astype(int)
        if yt.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yt, probs[:, i])
        axes[0].plot(fpr, tpr, color=cmap[i], lw=1.6, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(yt, probs[:, i])
        axes[1].plot(rec, prec, color=cmap[i], lw=1.6, label=f"{name} (AP={auc(rec[::-1], prec[::-1]):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel(_L("误报率 FPR", "FPR")); axes[0].set_ylabel(_L("召回率 TPR", "TPR"))
    axes[0].set_title(_L(f"多分类 ROC（OvR） 宏平均 AUC={report['multiclass'].get('macro_roc_auc_ovr', float('nan')):.4f}",
                         "Multiclass ROC (OvR)"))
    axes[1].set_xlabel(_L("召回率 Recall", "Recall")); axes[1].set_ylabel(_L("精确率 Precision", "Precision"))
    axes[1].set_title(_L("多分类 PR 曲线", "Multiclass PR"))
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    return _finish(fig, out_dir / f"05_multiclass_roc_pr_{tag}.png" if tag else out_dir / "05_multiclass_roc_pr.png")


def plot_binary_detection(report: dict, out_dir: Path | None = None, tag: str = "") -> Path | None:
    """异常检测工作点：ROC + PR + 阈值-F1 权衡。"""
    import matplotlib.pyplot as plt

    cached = report.get("_binary_curve")
    if cached is None:
        return None
    y_true, scores = cached
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    out_dir = Path(out_dir or FIGURE_DIR)
    fpr, tpr, thr = roc_curve(y_true, scores)
    prec, rec, thr2 = precision_recall_curve(y_true, scores)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC={auc(fpr, tpr):.4f}")
    for target in (0.01, 0.001):
        idx = np.searchsorted(fpr, target)
        idx = min(idx, len(tpr) - 1)
        ax.scatter([fpr[idx]], [tpr[idx]], color="#dc2626", zorder=5)
        ax.annotate(f"TPR@{target:.1%}FPR={tpr[idx]:.3f}", (fpr[idx], tpr[idx]),
                    textcoords="offset points", xytext=(8, -10), fontsize=8)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel(_L("误报率 FPR", "FPR")); ax.set_ylabel(_L("检测率 TPR", "TPR"))
    ax.set_title(_L("异常检测 ROC", "Detection ROC")); ax.legend(fontsize=9)

    ax = axes[1]
    ap = auc(rec, prec)
    ax.plot(rec, prec, color="#059669", lw=2, label=f"PR-AUC={ap:.4f}")
    b = report["binary"]
    ax.scatter([b["recall"]], [b["precision"]], color="#dc2626", zorder=5,
               label=_L(f"阈值 {b['threshold']:.2f}", f"thr {b['threshold']:.2f}"))
    ax.set_xlabel(_L("召回率 Recall", "Recall")); ax.set_ylabel(_L("精确率 Precision", "Precision"))
    ax.set_title(_L("异常检测 PR 曲线", "Detection PR")); ax.legend(fontsize=9)

    ax = axes[2]
    f1s = np.divide(2 * prec[:-1] * rec[:-1], prec[:-1] + rec[:-1],
                    out=np.zeros_like(prec[:-1]), where=(prec[:-1] + rec[:-1]) > 0)
    ax.plot(thr2, f1s, color="#7c3aed", lw=2)
    if len(f1s):
        k = int(np.argmax(f1s))
        ax.axvline(thr2[k], ls="--", c="gray", lw=1)
        ax.annotate(f"best F1={f1s[k]:.4f}\nthr={thr2[k]:.3f}", (thr2[k], f1s[k]),
                    textcoords="offset points", xytext=(10, -14), fontsize=9)
    ax.set_xlabel(_L("判定阈值", "threshold")); ax.set_ylabel("F1")
    ax.set_title(_L("阈值-F1 权衡（现场调参依据）", "Threshold vs F1"))
    fig.suptitle(_L(f"二分类异常检测工作点：{tag}", f"Binary detection: {tag}"), fontsize=13)
    fig.tight_layout()
    return _finish(fig, out_dir / f"06_binary_detection_{tag}.png" if tag else out_dir / "06_binary_detection.png")


# ======================================================================================
# 4. 在线检测时间轴（最有说服力的一张图）
# ======================================================================================
def plot_online_timeline(data: PreparedData, pred_csv: Path, cfg: Config, out_dir: Path | None = None, tag: str = "") -> Path | None:
    """连续测试流的"真实标签条带 / 异常概率 / 预测条带"三联图。

    条带用 imshow 渲染（几万条流记录也能秒出图，不会因为画几万个 axvspan 卡死）。
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    pred_csv = Path(pred_csv)
    if not pred_csv.exists() or data.test_stream_labels is None:
        return None
    df = pd.read_csv(pred_csv)
    out_dir = Path(out_dir or FIGURE_DIR)
    labels = np.asarray(data.test_stream_labels).astype(str)
    times = np.asarray(data.test_stream_time, dtype=np.float64)
    n = len(labels)
    t_rel = times - times[0]
    t_win = df["t_end"].to_numpy() - times[0]

    classes = sorted(set(labels), key=lambda c: (c != BINARY_NORMAL, c))
    color_map = {c: plt.cm.tab10(i % 10) for i, c in enumerate(classes)}
    color_map[BINARY_NORMAL] = (0.82, 0.85, 0.89, 1.0)
    code_of = {c: i for i, c in enumerate(classes)}
    cmap = ListedColormap([color_map[c] for c in classes])

    codes = np.array([code_of.get(l, 0) for l in labels], dtype=np.int16)
    x_end = float(max(t_rel[-1], 1.0))

    fig, axes = plt.subplots(3, 1, figsize=(15, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2.4, 1]})
    # (a) 真实标签条带
    ax = axes[0]
    ax.imshow(codes[None, :], aspect="auto", cmap=cmap, interpolation="nearest",
              extent=[0, x_end, 0, 1], vmin=0, vmax=len(classes) - 1)
    ax.set_yticks([]); ax.set_ylabel(_L("真实流量", "Truth"))
    ax.set_title(_L("连续测试流：真实攻击区间 vs 模型检测结果", "Continuous test stream: truth vs detection"))
    handles = [Patch(facecolor=color_map[c], label=c) for c in classes]
    ax.legend(handles=handles, ncol=min(len(classes), 8), fontsize=8, loc="lower left",
              bbox_to_anchor=(0, 1.02), frameon=False)

    # (b) 异常概率曲线 + 阈值 + 告警区间
    ax = axes[1]
    ax.plot(t_win, df["p_attack"], color="#2563eb", lw=1.3, label=_L("异常概率", "p(attack)"))
    ax.axhline(cfg.report_threshold, ls="--", c="#dc2626", lw=1,
               label=_L(f"告警阈值 {cfg.report_threshold}", f"threshold {cfg.report_threshold}"))
    ax.fill_between(t_win, 0, df["p_attack"], where=(df["p_attack"] >= cfg.report_threshold),
                    color="#dc2626", alpha=0.25, label=_L("告警区间", "alarm"))
    ax.set_ylabel(_L("异常概率", "p(attack)")); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(_L("在线检测：滑窗异常概率与告警", "Online detection probability"))

    # (c) 预测类别条带
    ax = axes[2]
    pred_codes = np.full(n, -1, dtype=np.int16)
    starts = df["win_start"].to_numpy()
    ends = df["win_end"].to_numpy()
    preds = df["pred_class"].astype(str).to_numpy()
    for s, e, p in zip(starts, ends, preds):
        pred_codes[max(int(s), 0) : min(int(e) + 1, n)] = code_of.get(p, 0)
    pred_codes[pred_codes < 0] = 0
    ax.imshow(pred_codes[None, :], aspect="auto", cmap=cmap, interpolation="nearest",
              extent=[0, x_end, 0, 1], vmin=0, vmax=len(classes) - 1)
    ax.set_yticks([]); ax.set_xlabel(_L("时间（秒，自测试流起点）", "time (s since stream start)"))
    ax.set_ylabel(_L("模型预测", "Prediction"))
    fig.tight_layout()
    return _finish(fig, out_dir / f"07_online_timeline_{tag}.png" if tag else out_dir / "07_online_timeline.png")


# ======================================================================================
# 5. 注意力热力图
# ======================================================================================
def plot_attention(data: PreparedData, attn_path: Path, out_dir: Path | None = None, tag: str = "") -> Path | None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    attn_path = Path(attn_path)
    if not attn_path.exists():
        return None
    w = np.load(attn_path)
    out_dir = Path(out_dir or FIGURE_DIR)
    is_attack = np.asarray(data.y_test_bin) == 1
    pick = []
    if is_attack.any():
        pick += list(np.where(is_attack)[0][:6])
    if (~is_attack).any():
        pick += list(np.where(~is_attack)[0][:3])
    if not pick:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(pick) + 2))
    sns.heatmap(w[pick], cmap="magma", ax=ax, cbar_kws={"label": _L("注意力权重", "attention")},
                yticklabels=[
                    f"#{i} (" + (_L("攻击", "attack") if is_attack[i] else _L("正常", "normal")) + ")"
                    for i in pick
                ])
    ax.set_xlabel(_L("窗口内的时间步（第几条流记录）", "timestep within window"))
    ax.set_title(_L("注意力热力图：模型把权重放在哪些流记录上", "Attention over window timesteps"))
    fig.tight_layout()
    return _finish(fig, out_dir / f"08_attention_{tag}.png" if tag else out_dir / "08_attention.png")


# ======================================================================================
# 6. 消融实验对比
# ======================================================================================
def plot_ablation(reports: dict[str, dict], out_dir: Path | None = None) -> Path | None:
    import matplotlib.pyplot as plt

    if not reports:
        return None
    out_dir = Path(out_dir or FIGURE_DIR)
    variants = list(reports.keys())
    metrics = {
        _L("多分类准确率", "multi acc"): [reports[v]["multiclass"]["accuracy"] for v in variants],
        _L("多分类宏F1", "multi macroF1"): [reports[v]["multiclass"]["macro_f1"] for v in variants],
        _L("二分类F1", "binary F1"): [reports[v]["binary"]["f1"] for v in variants],
        _L("二分类AUC", "binary AUC"): [reports[v]["binary"]["roc_auc"] for v in variants],
    }
    x = np.arange(len(variants))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    for i, (name, vals) in enumerate(metrics.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=name)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=20)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=2)
    ax.set_title(_L("消融实验：各结构对指标的贡献", "Ablation study"))

    ax = axes[1]
    params = [reports[v]["model"]["n_params"] for v in variants]
    accs = [reports[v]["multiclass"]["accuracy"] for v in variants]
    ax.scatter(params, accs, s=90, color="#2563eb")
    for v, p, a in zip(variants, params, accs):
        ax.annotate(v, (p, a), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel(_L("参数量", "parameters")); ax.set_ylabel(_L("多分类准确率", "accuracy"))
    ax.set_title(_L("精度-参数量权衡", "Accuracy vs parameters"))
    fig.tight_layout()
    return _finish(fig, out_dir / "09_ablation.png")


def make_all_figures(
    cfg: Config,
    data: PreparedData,
    report: dict,
    *,
    pred_csv: Path | None = None,
    attn_path: Path | None = None,
    history: dict | None = None,
    tag: str = "",
    out_dir: Path | None = None,
) -> list[Path]:
    """一次性生成全部单模型图表。"""
    init(cfg)
    out_dir = Path(out_dir or FIGURE_DIR)
    paths: list[Path] = []
    tag = tag or cfg.run_name
    for fn, kwargs in [
        (plot_class_distribution, {"data": data}),
        (plot_feature_profiles, {"cfg": cfg}),
        (plot_confusion_matrix, {"report": report, "class_names": data.class_names}),
        (plot_multiclass_roc_pr, {"data": data, "report": report}),
        (plot_binary_detection, {"report": report}),
    ]:
        try:
            p = fn(out_dir=out_dir, tag=tag, **kwargs)
            if p:
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("%s 生成失败：%s", getattr(fn, "__name__", fn), exc)
    if history:
        try:
            paths.append(plot_training_curves(history, out_dir=out_dir, tag=tag))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("训练曲线生成失败：%s", exc)
    if pred_csv is not None:
        try:
            p = plot_online_timeline(data, pred_csv, cfg, out_dir=out_dir, tag=tag)
            if p:
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("时间轴图生成失败：%s", exc)
    if attn_path is not None:
        try:
            p = plot_attention(data, attn_path, out_dir=out_dir, tag=tag)
            if p:
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("注意力图生成失败：%s", exc)
    return paths
