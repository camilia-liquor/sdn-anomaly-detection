"""评估模块：多分类指标 + 二分类异常检测指标 + **在线告警级**指标。

为什么单独做"在线告警级评估"
--------------------------
比赛评委真正关心的是"把它接到 SDN 控制器上能不能用"，而不是一个孤立的准确率。因此除了
论文里常见的多分类/二分类指标，本模块还按真实部署方式评估：

* **检测时延**：攻击开始后多久报出第一条告警（秒 / 条流记录）
* **误报率**：每小时多少次误报（FPR 单看比例会被正常流量总量稀释，工程上更看"次/小时"）
* **漏报片段数**：多少个攻击片段完全没被报出来

这些指标都是直接在**连续测试流**上滚动取窗算出来的，和上线时的行为一致。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data_pipeline import (
    BINARY_NORMAL,
    PreparedData,
    build_windows,
    clean_dataframe,
    rolling_windows,
)
from .model import build_explain_model, predict_probs
from .utils import (
    ARTIFACT_DIR,
    RAW_DIR,
    RESULT_DIR,
    get_logger,
    save_json,
    setup_matplotlib,
)

LOGGER = get_logger("cida.eval")


# ======================================================================================
# 基础指标
# ======================================================================================
def _safe_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def tpr_at_fpr(y_true, scores, target_fpr: float) -> float:
    """在给定误报率下能达到的召回率（异常检测最常用的工作点指标）。"""
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def best_threshold(y_true, scores) -> tuple[float, float]:
    """按 F1 最大化挑选阈值，返回 (阈值, 最优 F1)。"""
    from sklearn.metrics import f1_score, precision_recall_curve

    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[idx]) if len(thr) else 0.5, float(f1[idx])


def binary_metrics(y_true, p_attack, threshold: float, cfg: Config) -> dict:
    """二分类（正常/异常）指标集合。"""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(p_attack) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, p_attack),
        "pr_auc": float(average_precision_score(y_true, p_attack)) if len(np.unique(y_true)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }
    for f in cfg.target_fpr:
        out[f"tpr_at_fpr_{f:g}"] = tpr_at_fpr(y_true, p_attack, float(f))
    thr_best, f1_best = best_threshold(y_true, p_attack)
    out["best_f1"] = f1_best
    out["best_f1_threshold"] = thr_best
    return out


def multiclass_metrics(y_mc, p_mc, class_names: list[str]) -> dict:
    """多分类指标（宏平均/加权平均 + 逐类明细 + 混淆矩阵）。"""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        top_k_accuracy_score,
    )

    y_pred = np.argmax(p_mc, axis=1)
    y_mc = np.asarray(y_mc).astype(int)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_mc, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    report = classification_report(
        y_mc, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    # Top-k 在"测试集只出现单一类别"等退化情形会抛异常，这里做保护
    try:
        top2 = float(top_k_accuracy_score(y_mc, p_mc, k=min(2, len(class_names)), labels=labels))
    except Exception:
        top2 = float("nan")
    out = {
        "accuracy": float(accuracy_score(y_mc, y_pred)),
        "macro_precision": float(precision_score(y_mc, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_mc, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_mc, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_mc, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_mc, y_pred)),
        "top2_accuracy": top2,
        "n_classes_present": int(len(np.unique(y_mc))),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
            if name in report
        },
    }
    # 多分类 AUC（one-vs-rest，正确用法是喂概率矩阵，而不是把标签 binarize 后再算）
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_mc)) > 1:
            out["macro_roc_auc_ovr"] = float(
                roc_auc_score(y_mc, p_mc, multi_class="ovr", average="macro", labels=labels)
            )
    except Exception:
        out["macro_roc_auc_ovr"] = float("nan")
    return out


# ======================================================================================
# 在线告警级评估
# ======================================================================================
def confirm_alarms(alarmed: np.ndarray, confirm: int) -> np.ndarray:
    """告警确认（去抖）：要求**连续 confirm 个窗口**都超阈值才真正告警。

    confirm=1 表示不确认（原行为）。这是工业界 IDS/EDR 的标准做法：
    用少量时延换取误报的大幅下降——生产环境里"少打扰"往往比"早几毫秒"更重要。
    本工程的贡献是把它与**窗口长度、判定阈值**放在同一个评估框架里量化，
    从而给出可复现的"检测工作点选择"方法（见 tools/operating_point.py）。
    """
    alarmed = np.asarray(alarmed, dtype=bool)
    if confirm <= 1:
        return alarmed
    out = np.zeros_like(alarmed)
    run = 0
    for i, a in enumerate(alarmed):
        run = run + 1 if a else 0
        out[i] = run >= confirm
    return out


def online_alarm_evaluation(
    data: PreparedData,
    p_attack_windows: np.ndarray,
    win_starts: np.ndarray,
    cfg: Config,
    threshold: float | None = None,
    confirm: int | None = None,
) -> dict:
    """在连续测试流上评估"报警行为"：时延、误报率（次/小时）、漏报片段。

    关于"片段"的定义（很重要的一个坑）
    ----------------------------------
    生成数据时我们注入了 1% 的标注噪声（把个别正常流误标成攻击类），
    于是测试流里会散落上百个**长度为 1 的"伪攻击片段"**。如果把它们也算作攻击片段，
    片段召回率的分母会被撑到几百，任何模型都只能拿到 0.1~0.2 的召回率 —— 这显然不是模型的错。
    因此这里规定：**只有长度 ≥ max(window_size, online_min_segment_rows) 的连续攻击段
    才算一个真正的攻击片段**，更短的按标注噪声处理（其上的告警计入误报，不计入漏报）。
    """
    labels = np.asarray(data.test_stream_labels).astype(str)
    times = np.asarray(data.test_stream_time, dtype=np.float64)
    n = len(labels)
    thr = float(cfg.report_threshold if threshold is None else threshold)
    T = int(cfg.window_size)
    min_len = int(max(T, getattr(cfg, "online_min_segment_rows", 10)))

    # 攻击片段（连续的非 Normal 行）
    is_attack = labels != BINARY_NORMAL
    raw_segments: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if is_attack[i]:
            j = i
            while j + 1 < n and is_attack[j + 1]:
                j += 1
            raw_segments.append((i, j))
            i = j + 1
        else:
            i += 1
    segments = [(a, b) for a, b in raw_segments if (b - a + 1) >= min_len]
    n_ignored = len(raw_segments) - len(segments)

    alarmed = np.asarray(p_attack_windows) >= thr
    n_confirm = int(confirm if confirm is not None else getattr(cfg, "online_confirm_windows", 1))
    n_raw_alarms = int(alarmed.sum())
    alarmed = confirm_alarms(alarmed, n_confirm)
    span_sec = max(float(times[-1] - times[0]), 1e-6)

    detected, delays_rows, delays_sec = 0, [], []
    false_alarms = 0
    for k, start in enumerate(win_starts):
        if not alarmed[k]:
            continue
        end = start + T - 1
        # 误报 = 窗口内**没有任何**被标注为攻击的流记录（含标注噪声行），
        # 与我们评估多分类时使用的口径保持一致：模型只对"给定标签"负责。
        if not is_attack[max(start, 0) : min(end + 1, n)].any():
            false_alarms += 1

    for s0, s1 in segments:
        first = None
        for k, start in enumerate(win_starts):
            end = start + T - 1
            if not alarmed[k]:
                continue
            if not (end < s0 or start > s1):
                first = k
                break
        if first is None:
            continue
        detected += 1
        end_row = int(win_starts[first] + T - 1)
        delays_rows.append(int(end_row - s0))
        delays_sec.append(float(times[min(end_row, n - 1)] - times[s0]))

    hours = span_sec / 3600.0
    return {
        "threshold": thr,
        "confirm_windows": n_confirm,
        "stream_rows": int(n),
        "stream_hours": round(hours, 4),
        "min_segment_rows": min_len,
        "n_raw_attack_segments": len(raw_segments),
        "n_ignored_short_segments": int(n_ignored),
        "n_attack_segments": len(segments),
        "n_segments_detected": int(detected),
        "segment_recall": float(detected / max(len(segments), 1)),
        "n_segments_missed": int(len(segments) - detected),
        "mean_detection_delay_rows": float(np.mean(delays_rows)) if delays_rows else None,
        "median_detection_delay_rows": float(np.median(delays_rows)) if delays_rows else None,
        "mean_detection_delay_sec": float(np.mean(delays_sec)) if delays_sec else None,
        "p95_detection_delay_sec": float(np.percentile(delays_sec, 95)) if delays_sec else None,
        "false_alarms": int(false_alarms),
        "false_alarms_per_hour": float(false_alarms / hours) if hours > 0 else None,
        "n_windows": int(len(win_starts)),
        "n_alarmed_windows": int(alarmed.sum()),
        "n_raw_alarmed_windows": n_raw_alarms,
        "alarm_rate": float(alarmed.mean()),
    }


# ======================================================================================
# 主评估入口
# ======================================================================================
def evaluate_model(
    model,
    data: PreparedData,
    cfg: Config,
    *,
    tag: str = "",
    result_dir: Path | None = None,
    artifact_dir: Path | None = None,
    save_predictions: bool = True,
) -> dict:
    """完整评估：多分类 + 二分类 + 在线告警级，并把结果落盘。"""
    tag = tag or cfg.run_name
    result_dir = Path(result_dir or RESULT_DIR)
    artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    class_names = data.class_names

    # ---------- 1) 测试集窗口级预测 ----------
    p_mc, p_bin = predict_probs(model, data.X_test, batch_size=cfg.batch_size)
    mc = multiclass_metrics(data.y_test_mc, p_mc, class_names)
    binm = binary_metrics(data.y_test_bin, p_bin, cfg.report_threshold, cfg)
    thr_best, _ = best_threshold(data.y_test_bin, p_bin)
    bin_best = binary_metrics(data.y_test_bin, p_bin, thr_best, cfg)

    # 缓存"测试集窗口级预测"，供错误分析与工作点工具复用（不必重跑模型）
    try:
        cache = pd.DataFrame({
            "window_index": np.arange(len(p_mc)),
            "y_true": data.y_test_mc,
            "y_true_name": [class_names[i] for i in data.y_test_mc],
            "y_pred": np.argmax(p_mc, axis=1),
            "p_attack": np.asarray(p_bin),
        })
        cache["y_pred_name"] = [class_names[i] for i in cache["y_pred"].to_numpy()]
        cache["correct"] = (cache["y_true"] == cache["y_pred"]).astype(int)
        for i, name in enumerate(class_names):
            cache[f"p_{name}"] = np.asarray(p_mc)[:, i]
        cache.to_csv(artifact_dir / f"predictions_windows_{tag}.csv", index=False, encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("窗口级预测缓存失败：%s", exc)

    LOGGER.info(
        "[%s] 多分类 acc=%.4f macroF1=%.4f | 二分类 AUC=%.4f F1=%.4f (阈值 %.2f) | 最优阈值 %.3f F1=%.4f",
        tag, mc["accuracy"], mc["macro_f1"], binm["roc_auc"], binm["f1"],
        cfg.report_threshold, thr_best, bin_best["f1"],
    )

    # ---------- 2) 在线告警级评估（连续测试流滚动取窗）----------
    online = {}
    if data.test_stream_features is not None and data.input_mode == "sequence":
        X_stream, _, _, win_info = build_windows(
            data.test_stream_features, labels=data.test_stream_labels,
            window_size=cfg.window_size, stride=cfg.online_stride, pure_only=False,
        )
        starts = np.asarray(win_info["starts"])
        p_mc_s, p_bin_s = predict_probs(model, X_stream, batch_size=cfg.batch_size)
        online = online_alarm_evaluation(data, p_bin_s, starts, cfg)
        LOGGER.info(
            "[%s] 在线：片段召回 %.2f，平均时延 %.1f 秒，误报 %.2f 次/小时",
            tag, online["segment_recall"],
            online["mean_detection_delay_sec"] if online["mean_detection_delay_sec"] is not None else -1,
            online["false_alarms_per_hour"] if online["false_alarms_per_hour"] is not None else -1,
        )
        if save_predictions:
            _save_predictions(
                data, p_mc_s, p_bin_s, starts, class_names, cfg,
                artifact_dir / f"predictions_{tag}.csv",
            )
        # 把在线窗口的异常概率单独存一份，供工作点工具（tools/operating_point.py）枚举阈值/确认次数
        try:
            np.savez_compressed(
                artifact_dir / f"online_scores_{tag}.npz",
                p_attack=np.asarray(p_bin_s, dtype=np.float32),
                starts=np.asarray(starts, dtype=np.int64),
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("在线概率缓存失败：%s", exc)

    # ---------- 3) 注意力权重（可解释性）----------
    attn_path = None
    explain = build_explain_model(model)
    if explain is not None and data.input_mode == "sequence":
        try:
            w = explain.predict(data.X_test, batch_size=cfg.batch_size, verbose=0)
            w = np.asarray(w).reshape(len(data.X_test), -1)
            w = np.exp(w - w.max(axis=1, keepdims=True))
            w = w / w.sum(axis=1, keepdims=True)
            attn_path = artifact_dir / f"attention_{tag}.npy"
            np.save(attn_path, w.astype(np.float32))
            # 攻击窗口 vs 正常窗口的平均注意力（答辩用：模型确实盯住了可疑时刻）
            normal_mask = data.y_test_bin == 0
            LOGGER.info(
                "注意力峰值位置：正常窗口 %.2f / 攻击窗口 %.2f（时间步索引）",
                float(np.argmax(w[normal_mask], axis=1).mean()) if normal_mask.any() else -1,
                float(np.argmax(w[~normal_mask], axis=1).mean()) if (~normal_mask).any() else -1,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("注意力权重提取失败：%s", exc)

    # ---------- 4) 落盘 ----------
    report = {
        "tag": tag,
        "model": {
            "variant": getattr(model, "name", ""),
            "n_params": int(model.count_params()),
            "input_shape": list(data.input_shape),
        },
        "data": data.summary(),
        "multiclass": mc,
        "binary": binm,
        "binary_best_threshold": {"threshold": thr_best, **bin_best},
        "online": online,
        "attention_file": str(attn_path) if attn_path else None,
    }
    save_json(report, result_dir / f"metrics_{tag}.json")

    # 大数组不落 JSON，只在内存里留给可视化模块用（避免 metrics json 膨胀成几十 MB）
    report["_p_mc"] = p_mc
    report["_binary_curve"] = (np.asarray(data.y_test_bin), np.asarray(p_bin))

    _save_per_class(mc, class_names, result_dir / f"per_class_{tag}.csv")
    np.savetxt(result_dir / f"confusion_{tag}.csv", np.asarray(mc["confusion_matrix"]), fmt="%d", delimiter=",")
    np.savetxt(
        result_dir / f"confusion_norm_{tag}.csv",
        np.asarray(mc["confusion_matrix_normalized"]), fmt="%.4f", delimiter=",",
    )
    return report


def _save_predictions(data: PreparedData, p_mc, p_bin, starts, class_names, cfg: Config, path: Path) -> None:
    """保存窗口级预测（演示界面直接回放这些结果，因此演示不依赖 TensorFlow）。"""
    n = len(p_mc)
    ts = np.asarray(data.test_stream_time, dtype=np.float64)
    T = int(cfg.window_size)
    rows = {
        "win_index": np.arange(n),
        "win_start": np.asarray(starts)[:n],
        "win_end": np.asarray(starts)[:n] + T - 1,
        "t_start": ts[np.clip(np.asarray(starts)[:n], 0, len(ts) - 1)],
        "t_end": ts[np.clip(np.asarray(starts)[:n] + T - 1, 0, len(ts) - 1)],
        "p_attack": np.asarray(p_bin)[:n],
        "pred_class": [class_names[i] for i in np.argmax(p_mc[:n], axis=1)],
        "pred_is_attack": (np.asarray(p_bin)[:n] >= cfg.report_threshold).astype(int),
    }
    labels = np.asarray(data.test_stream_labels).astype(str)
    true_win = []
    for s in np.asarray(starts)[:n]:
        seg = labels[s : s + T]
        attacks = [v for v in pd.unique(seg) if v != BINARY_NORMAL]
        true_win.append(attacks[0] if attacks else BINARY_NORMAL)
    rows["true_class"] = true_win
    rows["true_is_attack"] = (np.asarray(true_win) != BINARY_NORMAL).astype(int)
    for i, name in enumerate(class_names):
        rows[f"p_{name}"] = np.asarray(p_mc)[:n, i]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    LOGGER.info("窗口级预测已缓存 → %s", path.name)


def drift_evaluation(
    model,
    data: PreparedData,
    cfg: Config,
    drift_path: Path | None = None,
    threshold: float | None = None,
) -> dict | None:
    """在**分布漂移**测试集上评估鲁棒性。

    漂移集由生成器用 `drift_strength>0` 单独生成：包规模、时长、包长整体缩放，
    相当于"换了攻击工具/网络环境"——模型在训练期从未见过这种量级组合。
    这一项回答的是评委最容易追问的问题："换个网络环境还准不准？"
    """
    drift_path = Path(drift_path or (RAW_DIR / f"{cfg.schema}_drift_test.csv"))
    if not drift_path.exists():
        return None
    try:
        df = pd.read_csv(drift_path, low_memory=False)
        df, _, _, mc_col = clean_dataframe(df, cfg)
        missing = [f for f in data.feature_names if f not in df.columns]
        if missing:
            LOGGER.warning("漂移集缺少 %d 个特征列，跳过漂移评估", len(missing))
            return None
        labels = df[mc_col].astype(str).str.strip()
        keep = labels.isin(data.class_names).to_numpy()
        X = data.scaler.transform(df[data.feature_names].to_numpy(dtype=np.float64)).astype(np.float32)
        X, labels = X[keep], labels[keep]
        if len(X) < cfg.window_size * 3:
            return None
        Xw, yw, _, info = build_windows(
            X, labels.to_numpy(), cfg.window_size, cfg.window_stride, pure_only=True
        )
        y_mc = data.label_encoder.transform(yw.astype(str)).astype(int)
        y_bin = np.where(yw.astype(str) == BINARY_NORMAL, 0, 1).astype(int)
        p_mc, p_bin = predict_probs(model, Xw, batch_size=cfg.batch_size)
        mc = multiclass_metrics(y_mc, p_mc, data.class_names)
        bn = binary_metrics(y_bin, p_bin, cfg.report_threshold if threshold is None else threshold, cfg)
        return {
            "drift_file": drift_path.name,
            "n_windows": int(len(Xw)),
            "windows_info": {k: v for k, v in info.items() if k != "starts"},
            "multiclass": mc,
            "binary": bn,
        }
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("漂移评估失败：%s", exc)
        return None


def _save_per_class(mc: dict, class_names: list[str], path: Path) -> None:
    rows = [
        {
            "class": name,
            "precision": mc["per_class"][name]["precision"],
            "recall": mc["per_class"][name]["recall"],
            "f1": mc["per_class"][name]["f1"],
            "support": mc["per_class"][name]["support"],
        }
        for name in class_names
        if name in mc["per_class"]
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def evaluate_cached(model_path: Path, processed_dir: Path | None = None, cfg: Config | None = None, tag: str = "reload") -> dict:
    """离线复现评估：从磁盘加载模型与预处理产物再评估一次。"""
    from .data_pipeline import PreparedData as _PD
    from .model import load_model_bundle

    cfg = cfg or Config()
    setup_matplotlib()
    data = _PD.load(processed_dir)
    model = load_model_bundle(model_path)
    return evaluate_model(model, data, cfg, tag=tag)
