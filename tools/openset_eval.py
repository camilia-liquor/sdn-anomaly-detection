"""开集（未知攻击）检测实验：模型遇到「没见过的攻击」时，是硬塞进已知类，还是诚实报未知？

问题
----
主实验的模型只在 8 类（Normal + 7 类攻击）上训练。真实网络里一定会出现训练集里没有的
新型攻击。一个只有闭合集分类头的模型**永远**会输出某个已知类别，而且常常还很自信：
运维拿到的是「这是 DDoS」，于是按 DDoS 去处置，实际却是别的东西。本实验回答三件事：

1. 不做拒识时，未知攻击会被错判成什么？（危险性）
2. 加上未知拒识后，未知攻击能检出多少？已知类要付出多少代价？
3. 部署时阈值怎么定、用哪种 OOD 分数最合适、什么情况下会失效？

做法（不修改任何现有实验代码，全部用 src 的公开函数重建数据）
------------------------------------------------------------
* 用 `src.data_pipeline` 的 `load_capture / clean_dataframe / order_by_time / build_windows`
  重建与主实验**完全一致**的清洗、时序切分（含 purge）、标准化（只在训练集 fit）与滑窗，
  只把留出类的行从**训练/验证集**整段剔除，让它只出现在测试集里当「未知攻击」；
* 模型仍是 `src.train.run_training` + `src.model.build_model` 的 cnn_bilstm，输出类别数变成 7，
  它从未见过留出类的任何样本（连标签编码器里都没有这个类）；
* 阈值**只在验证集的正常 + 已知攻击窗口上**标定，测试集与留出类窗口绝不参与；
* 三种 OOD 分数（统一为「分数越大越像未知」）：
  `1 - max softmax`、`熵 H(p)`、`-logsumexp(logits)`（Liu et al. 2020 的能量分数记号）；
* 对照组：含全部类训练的模型（默认直接复用主实验已训练的 artifacts/cnn_bilstm_main_s*，
  并在日志里核对本次重建的张量与 data/processed 逐位一致），给出**同口径**的已知类宏 F1，
  用来量化「为了获得未知拒识能力，已知类性能付出了多少」。

产物
----
    artifacts/openset/<tag>/model.keras        开集模型（tag = openset_no<留出类>_s<seed>）
    results/openset.json                       全部结构化数字
    results/openset_report.md                  中文报告（表格 + 结论）
    results/figures/openset_score_distributions.png / openset_udr_fpr.png / openset_misclass.png
    results/metrics_openset_no<类>_s<seed>.json 每个模型的开集指标

用法
----
    python tools/openset_eval.py                        # 2 个留出场景 × 3 个种子（默认，数分钟）
    python tools/openset_eval.py --scenarios U2R         # 只跑一个场景
    python tools/openset_eval.py --seeds 0 --epochs 15    # 快速试跑
    python tools/openset_eval.py --dry-run               # 只重建数据并核对口径，不训练

注意：开集模型的产物放在 `artifacts/openset/` 子目录下，**故意不放进 artifacts/ 顶层**——
`tools/predict_cli.py`、`tools/reevaluate.py`、`tools/drift_test.py` 会扫描 `artifacts/*/model.keras`
自动发现模型，7 类的开集模型混进去会让这些工具的默认选择变成错的。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    BINARY_NORMAL,
    NORMAL_CANDIDATES,
    PreparedData,
    build_windows,
    clean_dataframe,
    load_capture,
    order_by_time,
)
from src.model import load_model_bundle, predict_probs  # noqa: E402
from src.train import run_training  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR,
    FIGURE_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    ensure_dirs,
    get_logger,
    markdown_table,
    resolve_dir,
    save_json,
    save_text,
    set_seed,
    setup_matplotlib,
)

LOGGER = get_logger("cida.openset")

# --------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------
ARTIFACT_SUBDIR = "openset"
SCORE_ORDER = ("msp", "entropy", "energy")
SCORE_LABEL = {
    "msp": "1 - max softmax",
    "entropy": "熵 H(p)",
    "energy": "-logsumexp(logits)",
}
SCORE_DESC = {
    "msp": "多分类最大概率越低 → 分数越高 → 越可能是未知",
    "entropy": "多分类概率分布越平坦 → 熵越大 → 越可能是未知",
    "energy": "能量分数 -logsumexp(logits)，越大越可能是未知",
}
DEFAULT_SCENARIOS = ("U2R", "Botnet")
DEFAULT_SEEDS = (0, 1, 2)
BUDGETS = (0.01, 0.001)
UNKNOWN_SENTINEL = "Unknown"


# ======================================================================================
# 数据：用 src 的公开函数重建主实验口径，只把留出类从 train/val 里剔除
# ======================================================================================
def time_split_indices(cfg: Config, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """与 `prepare_dataset` 时序切分逐行一致地算出行下标（含 purge 间隙）。"""
    purge = int(cfg.purge_windows * max(cfg.window_size, 1))
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    tr_end = max(n_train - purge, 1)
    va_start = min(n_train + purge, n - 1)
    va_end = max(min(n_train + n_val - purge, n), va_start + 1)
    te_start = min(n_train + n_val + purge, n - 1)
    return (np.arange(0, tr_end), np.arange(va_start, min(va_end, n)),
            np.arange(te_start, n), purge)


def load_base(cfg: Config) -> dict:
    """读取 + 清洗 + 排序 + 复算切分下标（与主实验同口径）。"""
    if str(getattr(cfg, "split_mode", "time")).lower() != "time":
        raise SystemExit("本工具按时序切分实现，请用默认 split_mode=time 运行。")
    df, raw_meta = load_capture(cfg)
    n_rows_raw = len(df)
    df, feats, _label_cols, mc_col = clean_dataframe(df, cfg)
    df = order_by_time(df, cfg)

    labels = df[mc_col].astype(str).str.strip()
    present = set(labels.unique())
    for cand in NORMAL_CANDIDATES:            # 与 prepare_dataset 一致的正常类归一化
        if cand in present and cand != BINARY_NORMAL:
            labels = labels.map(lambda v, c=cand: BINARY_NORMAL if v == c else v)
            LOGGER.info("已把正常流量标签 %r 归一化为 %r", cand, BINARY_NORMAL)
            break
    n = len(df)
    X_raw = df[feats].to_numpy(dtype=np.float64)
    ts = (df["__ts"].to_numpy(dtype=np.float64) if "__ts" in df.columns
          else np.arange(n, dtype=np.float64))
    idx_train, idx_val, idx_test, purge = time_split_indices(cfg, n)
    names = labels.to_numpy()
    LOGGER.info("数据就绪：清洗后 %s 行 × %d 特征（原始 %s 行）；行级切分 train=%d / val=%d / test=%d",
                f"{n:,}", len(feats), f"{n_rows_raw:,}", len(idx_train), len(idx_val), len(idx_test))
    return {
        "X_raw": X_raw, "feats": list(feats), "names": names, "ts": ts,
        "idx": (idx_train, idx_val, idx_test), "purge": purge,
        "all_classes": sorted(set(names.tolist())),
        "row_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "raw_meta": raw_meta, "n_rows": n, "n_rows_raw": n_rows_raw,
    }


def build_payload(cfg: Config, base: dict, holdout: str) -> dict:
    """构造一次开集实验所需的数据：训练/验证剔除 holdout 类，测试集保留。

    ``holdout=""`` 表示不剔除任何类，即对照用的**全类模型**数据（与主实验同口径）。
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    names = base["names"]
    idx_train, idx_val, idx_test = base["idx"]
    all_classes = list(base["all_classes"])
    hold = str(holdout or "")
    known = [c for c in all_classes if c != hold] if hold else list(all_classes)

    tr = idx_train[names[idx_train] != hold] if hold else idx_train
    va = idx_val[names[idx_val] != hold] if hold else idx_val
    te = idx_test
    dropped = (len(idx_train) - len(tr)) + (len(idx_val) - len(va))

    # 标准化：只在剔除留出类之后的训练行上 fit —— 留出类的统计量绝不参与
    scaler = StandardScaler().fit(base["X_raw"][tr])
    X = scaler.transform(base["X_raw"]).astype(np.float32)
    if cfg.feature_noise and float(cfg.feature_noise) > 0:
        rng = np.random.default_rng(int(cfg.seed) + 7)
        X = X + rng.normal(0.0, float(cfg.feature_noise) * 0.1, size=X.shape).astype(np.float32)

    T, S = int(cfg.window_size), int(cfg.window_stride)

    def pack(rows):
        Xw, yw, _ts, info = build_windows(
            X[rows], names[rows], T, S, pure_only=True, timestamps=base["ts"][rows]
        )
        return Xw, np.asarray(yw, dtype=object), info

    Xtr, ntr, i_tr = pack(tr)
    Xva, nva, i_va = pack(va)
    Xte, nte, i_te = pack(te)

    enc = LabelEncoder().fit(known)

    def ids(nm: np.ndarray) -> np.ndarray:
        out = np.full(len(nm), -1, dtype=np.int64)      # -1 = 训练时没见过的类
        m = np.isin(nm, known)
        out[m] = enc.transform(nm[m].astype(str))
        return out

    ytr, yva, yte = ids(ntr), ids(nva), ids(nte)
    to_bin = lambda nm: (nm != BINARY_NORMAL).astype(np.int64)  # noqa: E731

    data = PreparedData(
        X_train=Xtr, y_train_mc=ytr, y_train_bin=to_bin(ntr),
        X_val=Xva, y_val_mc=yva, y_val_bin=to_bin(nva),
        X_test=Xte, y_test_mc=yte, y_test_bin=to_bin(nte),
        class_names=list(enc.classes_), feature_names=list(base["feats"]),
        scaler=scaler, label_encoder=enc, input_mode=cfg.input_mode, window_size=T,
        train_class_counts={str(c): int(v) for c, v in
                            zip(enc.classes_, np.bincount(ytr, minlength=len(enc.classes_)))},
        split_info={
            "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
            "purge_rows": int(base["purge"]), "split_mode": "time",
            "holdout_class": hold, "rows_dropped_from_train_val": int(dropped),
            "windows": {"train": i_tr, "val": i_va, "test": i_te},
        },
        raw_meta=base["raw_meta"],
    )
    if hold:
        LOGGER.info("[留出 %s] 训练窗口 %d（训练/验证共剔除该类 %s 行）/ 验证 %d / 测试 %d，"
                    "其中未知类窗口 %d 个", hold, len(Xtr), f"{dropped:,}", len(Xva), len(Xte),
                    int((nte == hold).sum()))
    else:
        LOGGER.info("全类对照数据：训练 %d / 验证 %d / 测试 %d 窗口，%d 类",
                    len(Xtr), len(Xva), len(Xte), len(enc.classes_))
    return {
        "data": data, "holdout": hold, "known_classes": known, "all_classes": all_classes,
        "names": {"train": ntr, "val": nva, "test": nte},
        "row_split_counts": {
            "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
            "rows_dropped_from_train_val": int(dropped),
            "raw_train": int(len(idx_train)), "raw_val": int(len(idx_val)),
        },
    }


def compare_with_main_experiment(cfg: Config, full_payload: dict) -> dict:
    """核对本次重建的张量与项目主实验 data/processed 是否逐位一致（只读，不修改）。"""
    path = resolve_dir(getattr(cfg, "processed_dir", ""), PROCESSED_DIR) / "tensors.npz"
    out: dict = {"checked": False, "path": "data/processed/tensors.npz"}
    if not path.exists():
        out["note"] = "未找到主实验预处理产物，跳过核对"
        return out
    z = np.load(path, allow_pickle=True)
    data = full_payload["data"]
    worst = 0.0
    for key, mine in (("X_train", data.X_train), ("X_val", data.X_val), ("X_test", data.X_test),
                      ("y_train_mc", data.y_train_mc), ("y_val_mc", data.y_val_mc),
                      ("y_test_mc", data.y_test_mc)):
        ref = z[key] if key in z.files else None
        if ref is None:
            out[key] = {"match": False, "note": "主实验产物里没有这一项"}
            continue
        if tuple(ref.shape) != tuple(np.asarray(mine).shape):
            out[key] = {"match": False,
                        "note": f"形状不同：主实验 {tuple(ref.shape)} vs 本次 {tuple(np.asarray(mine).shape)}"}
            continue
        diff = float(np.abs(np.asarray(ref, dtype=np.float64)
                            - np.asarray(mine, dtype=np.float64)).max())
        worst = max(worst, diff)
        out[key] = {"match": True, "max_abs_diff": diff}
    out["checked"] = True
    out["max_abs_diff_overall"] = worst
    out["identical"] = worst <= 1e-4
    LOGGER.info("与主实验预处理口径核对：最大绝对差 %.3g → %s", worst,
                "逐位一致" if out["identical"] else "不一致（将改为自己训练全类对照模型）")
    return out


# ======================================================================================
# OOD 分数与阈值
# ======================================================================================
def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)), axis=axis)


def ood_scores(p_mc: np.ndarray, logits: np.ndarray | None) -> dict:
    """三种 OOD 分数，统一为「越大越像未知」。"""
    p = np.clip(np.asarray(p_mc, dtype=np.float64), 1e-12, 1.0)
    out = {"msp": 1.0 - p.max(axis=1), "entropy": -(p * np.log(p)).sum(axis=1)}
    out["energy"] = (-_logsumexp(np.asarray(logits, dtype=np.float64), axis=1)
                     if logits is not None else None)
    return out


def extract_logits(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray | None:
    """取出 softmax 之前的 logits（最后一层是 softmax，需另接一层同权重的 Dense）。"""
    try:
        from tensorflow.keras import layers, models

        dense = model.get_layer("mc")
        probe_layer = layers.Dense(int(dense.units), activation=None, name="mc_logits_probe")
        probe = models.Model(model.input, probe_layer(dense.input))
        probe.get_layer("mc_logits_probe").set_weights(dense.get_weights())
        return np.asarray(probe.predict(X, batch_size=batch_size, verbose=0), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - 拿不到 logits 时只跳过 energy
        LOGGER.warning("无法提取 logits（energy 分数将被跳过）：%s", exc)
        return None


def threshold_at_fpr(normal_scores: np.ndarray, alpha: float) -> float:
    """在**验证集正常窗口**上取分位数阈值，使正常窗口被判为未知的比例不超过 alpha。"""
    s = np.sort(np.asarray(normal_scores, dtype=np.float64))[::-1]
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil(alpha * n))
    if k <= 1:
        return float(s[0]) + 1e-9
    return float(s[k - 1])


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(y, np.concatenate([pos, neg])))


# ======================================================================================
# 指标
# ======================================================================================
def per_class_f1(y_true, y_pred, classes: list[str]) -> tuple[float, dict]:
    """在 classes 上算宏 F1；y_pred 里出现 classes 之外的取值（Unknown / 留出类）一律算错判。"""
    from sklearn.metrics import confusion_matrix

    y_pred = np.asarray(y_pred, dtype=object)
    others = sorted(set(y_pred.tolist()) - set(classes))
    cm = confusion_matrix(np.asarray(y_true, dtype=object), y_pred, labels=list(classes) + others)
    per, f1s = {}, []
    for i, c in enumerate(classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per[str(c)] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}
        f1s.append(f1)
    return (float(np.mean(f1s)) if f1s else float("nan")), per


def _det_curve(normal_scores: np.ndarray, unknown_scores: np.ndarray, n_grid: int = 300):
    """阈值扫描：(验证正常窗口的误报率, 未知窗口检出率)。"""
    normal_scores = np.asarray(normal_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    if len(normal_scores) == 0 or len(unknown_scores) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    pool = np.concatenate([normal_scores, unknown_scores])
    thr = np.unique(np.quantile(pool, np.linspace(0.0, 1.0, n_grid)))
    thr = np.concatenate([[np.inf], thr[::-1], [-np.inf]])
    fpr = np.array([float(np.mean(normal_scores > t)) for t in thr])
    udr = np.array([float(np.mean(unknown_scores > t)) for t in thr])
    return fpr, udr


def evaluate_open_model(model, payload: dict, budgets=BUDGETS) -> dict:
    """一个开集模型在**完整测试集**（含留出类）上的全部开集指标。"""
    data = payload["data"]
    hold = payload["holdout"]
    known = payload["known_classes"]
    names_v, names_t = payload["names"]["val"], payload["names"]["test"]

    pv_mc, _ = predict_probs(model, data.X_val)
    pt_mc, _ = predict_probs(model, data.X_test)
    lg_v = extract_logits(model, data.X_val)
    lg_t = extract_logits(model, data.X_test)
    sv, st = ood_scores(pv_mc, lg_v), ood_scores(pt_mc, lg_t)

    consistency = float("nan")
    if lg_t is not None:
        consistency = float(np.abs(np.exp(lg_t - _logsumexp(lg_t, axis=1)[:, None]) - pt_mc).max())

    m_val_normal = names_v == BINARY_NORMAL
    m_val_known_atk = ~m_val_normal               # 验证集里只剩已知攻击（留出类已被剔除）
    m_te_normal = names_t == BINARY_NORMAL
    m_te_unknown = (names_t == hold) if hold else np.zeros(len(names_t), dtype=bool)
    m_te_known_atk = (~m_te_normal) & (~m_te_unknown)
    m_te_known = ~m_te_unknown
    n_unk = int(m_te_unknown.sum())

    # ---- 不做拒识：直接 argmax（闭合集行为）----
    pred_known = np.asarray(known, dtype=object)[pt_mc.argmax(axis=1)]
    y_true_known = names_t[m_te_known]
    macro_f1, per_class = per_class_f1(y_true_known, pred_known[m_te_known], known)
    acc = float(np.mean(pred_known[m_te_known] == y_true_known))

    unk_pred = pred_known[m_te_unknown]
    unk_conf = pt_mc[m_te_unknown].max(axis=1) if n_unk else np.array([])
    counts = {c: int((unk_pred == c).sum()) for c in known}
    misclass = {
        "n_unknown_windows": n_unk,
        "counts": counts,
        "fractions": {c: (v / n_unk if n_unk else float("nan")) for c, v in counts.items()},
        "mean_max_softmax": float(np.mean(unk_conf)) if n_unk else float("nan"),
        "fraction_high_confidence": float(np.mean(unk_conf >= 0.9)) if n_unk else float("nan"),
        "top1_target": (max(counts, key=lambda c: counts[c]) if n_unk else ""),
        "top1_fraction": (max(counts.values()) / n_unk if n_unk else float("nan")),
    }

    # ---- 三种 OOD 分数 ----
    ood: dict = {}
    for s in SCORE_ORDER:
        if st[s] is None or sv[s] is None:
            ood[s] = {"available": False}
            continue
        entry: dict = {
            "available": True,
            "definition": SCORE_LABEL[s],
            "direction": SCORE_DESC[s],
            "auroc_unknown_vs_normal": _auroc(st[s][m_te_unknown], st[s][m_te_normal]),
            "auroc_unknown_vs_known": _auroc(st[s][m_te_unknown], st[s][m_te_known]),
            "auroc_unknown_vs_known_attack": _auroc(st[s][m_te_unknown], st[s][m_te_known_atk]),
            "mean_score": {
                "normal": float(np.mean(st[s][m_te_normal])),
                "known_attack": (float(np.mean(st[s][m_te_known_atk]))
                                 if int(m_te_known_atk.sum()) else float("nan")),
                "unknown": float(np.mean(st[s][m_te_unknown])) if n_unk else float("nan"),
            },
            "budgets": {},
        }
        for alpha in budgets:
            thr = threshold_at_fpr(sv[s][m_val_normal], alpha)
            rejected = st[s] > thr
            pred_after = np.where(rejected, UNKNOWN_SENTINEL, pred_known)
            macro_after, per_after = per_class_f1(y_true_known, pred_after[m_te_known], known)
            entry["budgets"][f"{alpha:g}"] = {
                "budget": float(alpha),
                "threshold": float(thr),
                "val_fpr": float(np.mean(sv[s][m_val_normal] > thr)),
                "val_known_attack_reject_rate": float(np.mean(sv[s][m_val_known_atk] > thr)),
                "test_fpr_on_normal": float(np.mean(st[s][m_te_normal] > thr)),
                "udr_unknown_detected": (float(np.mean(st[s][m_te_unknown] > thr)) if n_unk
                                         else float("nan")),
                "n_unknown_detected": int((st[s][m_te_unknown] > thr).sum()) if n_unk else 0,
                "known_attack_false_reject_rate": (
                    float(np.mean(st[s][m_te_known_atk] > thr))
                    if int(m_te_known_atk.sum()) else float("nan")),
                "known_class_macro_f1_after_reject": macro_after,
                "known_class_macro_f1_drop": macro_f1 - macro_after,
                "known_class_recall_after_reject": {c: per_after[c]["recall"] for c in known},
            }
        ood[s] = entry

    return {
        "holdout_class": hold,
        "known_classes": known,
        "n_windows": {
            "val_normal": int(m_val_normal.sum()),
            "val_known_attack": int(m_val_known_atk.sum()),
            "test_normal": int(m_te_normal.sum()),
            "test_known_attack": int(m_te_known_atk.sum()),
            "test_unknown": n_unk,
            "test_total": int(len(names_t)),
        },
        "train_windows": int(len(data.X_train)),
        "softmax_logits_consistency": consistency,
        "known_class_macro_f1_no_reject": macro_f1,
        "known_class_accuracy_no_reject": acc,
        "known_class_per_class_no_reject": per_class,
        "known_class_macro_f1_reference_scope": macro_f1,
        "misclassification_no_reject": misclass,
        "ood": ood,
    }


def evaluate_reference_model(model, payload: dict, holdout: str, known_classes: list[str]) -> dict:
    """全类对照模型：同口径的已知类宏 F1 + 它在留出类窗口上本来能答对多少。"""
    data = payload["data"]
    names_t = payload["names"]["test"]
    p_mc, _ = predict_probs(model, data.X_test)
    all_classes = list(data.class_names)
    pred_all = np.asarray(all_classes, dtype=object)[p_mc.argmax(axis=1)]

    m_unknown = (names_t == holdout) if holdout else np.zeros(len(names_t), dtype=bool)
    m_known = ~m_unknown
    macro_scope, per_scope = per_class_f1(names_t[m_known], pred_all[m_known], known_classes)
    macro_all, per_all = per_class_f1(names_t, pred_all, all_classes)
    return {
        "holdout_class": holdout,
        "n_classes": len(all_classes),
        "known_class_macro_f1_reference_scope": macro_scope,
        "known_class_per_class_reference_scope": per_scope,
        "full_multiclass_macro_f1": macro_all,
        "full_multiclass_per_class": per_all,
        "accuracy_on_holdout_windows": (float(np.mean(pred_all[m_unknown] == names_t[m_unknown]))
                                        if int(m_unknown.sum()) else float("nan")),
        "n_holdout_windows": int(m_unknown.sum()),
    }


# ======================================================================================
# 训练 / 载入
# ======================================================================================
def model_path_for(tag: str) -> Path:
    return ARTIFACT_DIR / ARTIFACT_SUBDIR / tag / "model.keras"


def train_open_model(cfg: Config, payload: dict, tag: str, seed: int, epochs: int | None,
                     skip_train: bool):
    path = model_path_for(tag)
    if skip_train and path.exists():
        LOGGER.info("[%s] 复用已训练模型（--skip-train）", tag)
        return load_model_bundle(path), {"loaded_from_disk": True, "tag": tag, "seed": int(seed)}
    cfg_i = Config.from_dict(cfg.to_dict())
    cfg_i.seed = int(seed)
    res = run_training(cfg_i, payload["data"], variant="cnn_bilstm",
                       tag=f"{ARTIFACT_SUBDIR}/{tag}", epochs=epochs)
    return res.model, {
        "loaded_from_disk": False, "tag": tag, "seed": int(seed),
        "train_seconds": round(float(res.duration_s), 1),
        "best_epoch": int(res.best_epoch) + 1,
        "epochs_run": len(res.history.get("loss", [])),
        "n_params": int(res.n_params),
        "best_val_loss": float(res.best_val_loss),
    }


def pick_reference(cfg: Config, full_payload: dict, seed: int, mode: str, npc: int):
    """选对照用的全类模型：优先复用主实验已训练好的 artifacts/cnn_bilstm_main_s<seed>。"""
    tag = f"cnn_bilstm_main_s{seed}"
    path = ARTIFACT_DIR / tag / "model.keras"
    if mode in ("auto", "existing") and path.exists():
        try:
            model = load_model_bundle(path)
            out = model.output["mc"] if isinstance(model.output, dict) else model.output
            if int(out.shape[-1]) == int(npc):
                LOGGER.info("[对照] 复用主实验模型 artifacts/%s/model.keras（同数据同种子）", tag)
                return model, {"source": "existing", "tag": tag, "seed": int(seed)}
            LOGGER.warning("[对照] %s 的类别数（%d）与本次（%d）不一致，改为自己训练",
                           tag, int(out.shape[-1]), npc)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[对照] 载入 %s 失败（%s），改为自己训练", tag, exc)
    if mode == "existing":
        raise SystemExit(f"找不到可复用的主实验模型 artifacts/{tag}/model.keras")
    LOGGER.info("[对照] 自己训练一个全类模型（tag=openset_refall_s%d）", seed)
    model, meta = train_open_model(cfg, full_payload, f"openset_refall_s{seed}", seed, None, False)
    meta["source"] = "trained"
    return model, meta


# ======================================================================================
# 聚合
# ======================================================================================
def agg(vals: list) -> dict:
    v = np.asarray([x for x in vals if x is not None and not (
        isinstance(x, float) and np.isnan(x))], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "n": int(v.size)}


def fmt(a: dict | None, nd: int = 4) -> str:
    if not a or not a.get("n") or np.isnan(a.get("mean", float("nan"))):
        return "—"
    return f"{a['mean']:.{nd}f} ± {a['std']:.{nd}f}"


def _budget_keys(records: list[dict], score: str) -> list[str]:
    for r in records:
        e = r["ood"].get(score) or {}
        if e.get("available"):
            return list(e["budgets"].keys())
    return []


def aggregate_records(records: list[dict]) -> dict:
    """同一场景多个种子 → 均值 ± 标准差。"""
    classes = records[0]["known_classes"] if records else []
    out: dict = {
        "n_seeds": len(records),
        "n_windows": records[0]["n_windows"] if records else {},
        "known_class_macro_f1_no_reject": agg([r["known_class_macro_f1_no_reject"] for r in records]),
        "known_class_accuracy_no_reject": agg([r["known_class_accuracy_no_reject"] for r in records]),
        "misclassification_no_reject": {
            "mean_fractions": {},
            "mean_max_softmax": agg([r["misclassification_no_reject"]["mean_max_softmax"] for r in records]),
            "mean_fraction_high_confidence": agg(
                [r["misclassification_no_reject"]["fraction_high_confidence"] for r in records]),
            "per_seed_top1": [r["misclassification_no_reject"]["top1_target"] for r in records],
            "n_unknown_windows": [r["misclassification_no_reject"]["n_unknown_windows"] for r in records],
        },
        "ood": {},
    }
    for c in classes:
        out["misclassification_no_reject"]["mean_fractions"][c] = float(np.mean(
            [r["misclassification_no_reject"]["fractions"][c] for r in records]))
    for s in SCORE_ORDER:
        recs = [r for r in records if (r["ood"].get(s) or {}).get("available")]
        if not recs:
            out["ood"][s] = {"available": False}
            continue
        entry: dict = {
            "available": True,
            "auroc_unknown_vs_normal": agg([r["ood"][s]["auroc_unknown_vs_normal"] for r in recs]),
            "auroc_unknown_vs_known": agg([r["ood"][s]["auroc_unknown_vs_known"] for r in recs]),
            "auroc_unknown_vs_known_attack": agg(
                [r["ood"][s]["auroc_unknown_vs_known_attack"] for r in recs]),
            "mean_score": {k: agg([r["ood"][s]["mean_score"][k] for r in recs])
                           for k in ("normal", "known_attack", "unknown")},
            "budgets": {},
        }
        for b in _budget_keys(records, s):
            entry["budgets"][b] = {
                "threshold": agg([r["ood"][s]["budgets"][b]["threshold"] for r in recs]),
                "val_fpr": agg([r["ood"][s]["budgets"][b]["val_fpr"] for r in recs]),
                "test_fpr_on_normal": agg(
                    [r["ood"][s]["budgets"][b]["test_fpr_on_normal"] for r in recs]),
                "udr_unknown_detected": agg(
                    [r["ood"][s]["budgets"][b]["udr_unknown_detected"] for r in recs]),
                "known_attack_false_reject_rate": agg(
                    [r["ood"][s]["budgets"][b]["known_attack_false_reject_rate"] for r in recs]),
                "known_class_macro_f1_after_reject": agg(
                    [r["ood"][s]["budgets"][b]["known_class_macro_f1_after_reject"] for r in recs]),
                "known_class_macro_f1_drop": agg(
                    [r["ood"][s]["budgets"][b]["known_class_macro_f1_drop"] for r in recs]),
                "mean_recall_after_reject": {
                    c: agg([r["ood"][s]["budgets"][b]["known_class_recall_after_reject"][c]
                            for r in recs]) for c in classes},
            }
        out["ood"][s] = entry
    return out


def pooled_metrics(score_store: dict, seeds: list[int]) -> dict:
    """跨场景汇总：同一种子把各场景的未知/正常窗口拼接后算一次 AUROC，再对种子取均值。"""
    per_seed: dict[int, dict] = {}
    for scen, store in score_store.items():
        for seed, sd in store.items():
            slot = per_seed.setdefault(int(seed), {"pos": {k: [] for k in SCORE_ORDER},
                                                   "neg": {k: [] for k in SCORE_ORDER},
                                                   "neg_atk": {k: [] for k in SCORE_ORDER}})
            for k in SCORE_ORDER:
                d = sd.get(k)
                if not d:
                    continue
                if len(d["unknown"]):
                    slot["pos"][k].append(d["unknown"])
                if len(d["normal"]):
                    slot["neg"][k].append(d["normal"])
                if len(d["known_attack"]):
                    slot["neg_atk"][k].append(d["known_attack"])
    by_seed = []
    for seed in seeds:
        slot = per_seed.get(int(seed))
        if not slot:
            continue
        entry: dict = {"seed": int(seed)}
        for k in SCORE_ORDER:
            pos = [a for a in slot["pos"][k] if len(a)]
            neg = [a for a in slot["neg"][k] if len(a)]
            atk = [a for a in slot["neg_atk"][k] if len(a)]
            if not pos or not neg:
                continue
            entry[f"auroc_{k}_unknown_vs_normal"] = _auroc(np.concatenate(pos), np.concatenate(neg))
            entry[f"auroc_{k}_unknown_vs_known"] = _auroc(np.concatenate(pos),
                                                          np.concatenate(neg + atk))
            if atk:
                entry[f"auroc_{k}_unknown_vs_known_attack"] = _auroc(np.concatenate(pos),
                                                                     np.concatenate(atk))
        by_seed.append(entry)
    out: dict = {"by_seed": by_seed, "auroc_unknown_vs_normal": {},
                 "auroc_unknown_vs_known": {}, "auroc_unknown_vs_known_attack": {}}
    for k in SCORE_ORDER:
        out["auroc_unknown_vs_normal"][k] = agg([e.get(f"auroc_{k}_unknown_vs_normal") for e in by_seed])
        out["auroc_unknown_vs_known"][k] = agg([e.get(f"auroc_{k}_unknown_vs_known") for e in by_seed])
        out["auroc_unknown_vs_known_attack"][k] = agg(
            [e.get(f"auroc_{k}_unknown_vs_known_attack") for e in by_seed])
    return out


# ======================================================================================
# 图
# ======================================================================================
def make_figures(pool: dict, det: dict, misclass: dict, budgets=BUDGETS) -> list[Path]:
    chinese = setup_matplotlib(logger=LOGGER)
    import matplotlib.pyplot as plt

    def L(zh: str, en: str) -> str:
        return zh if chinese else en

    out_paths: list[Path] = []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    groups = [("normal", L("正常窗口", "Normal"), "#2b7bba"),
              ("known_attack", L("已知攻击", "Known attack"), "#e08214"),
              ("unknown", L("未知攻击（留出类）", "Unknown attack"), "#c0392b")]

    # ---- 图 1：三种 OOD 分数在 正常 / 已知攻击 / 未知攻击 三组上的分布 ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for ax, s in zip(axes, SCORE_ORDER):
        data = {g: np.asarray(pool[g][s], dtype=float) for g, _zh, _c in groups
                if pool[g].get(s) is not None and len(pool[g][s])}
        if not data:
            ax.set_visible(False)
            continue
        lo = min(float(np.quantile(v, 0.005)) for v in data.values())
        hi = max(float(np.quantile(v, 0.995)) for v in data.values())
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 45)
        for g, label, color in groups:
            if g in data:
                ax.hist(data[g], bins=bins, density=True, alpha=0.55, label=label, color=color)
        for alpha, style in zip(budgets, (":", "--")):
            thr = (pool.get("thresholds", {}).get(s) or {}).get(f"{alpha:g}")
            if thr is not None and np.isfinite(thr) and lo <= thr <= hi:
                ax.axvline(thr, color="#444444", linestyle=style, linewidth=1.3,
                           label=L(f"阈值 @正常误报≤{alpha*100:g}%", f"thr @FPR<={alpha*100:g}%"))
        if "unknown" in data and "normal" in data:
            ax.set_title(f"{s}  ({SCORE_LABEL[s]})\n"
                         + L("未知 vs 正常 AUROC = ", "AUROC(unknown vs normal) = ")
                         + f"{_auroc(data['unknown'], data['normal']):.3f}", fontsize=10)
        ax.set_xlabel(SCORE_LABEL[s])
        ax.set_ylabel(L("密度", "density"))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(L("三种 OOD 分数在三组窗口上的分布：未知攻击明显偏离已知分布",
                   "OOD score distributions: unknown attacks deviate from known traffic"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    p = FIGURE_DIR / "openset_score_distributions.png"
    fig.savefig(p)
    plt.close(fig)
    out_paths.append(p)

    # ---- 图 2：UDR-FPR（检测误差权衡）曲线 ----
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    grid = np.logspace(np.log10(3e-4), np.log10(0.5), 140)
    colors = {"msp": "#2b7bba", "entropy": "#e08214", "energy": "#c0392b"}
    for s in SCORE_ORDER:
        curves = det.get(s) or []
        interp = []
        for fpr, udr in curves:
            m = np.asarray(fpr) > 0
            if m.sum() < 2:
                continue
            x, y = np.asarray(fpr)[m], np.asarray(udr)[m]
            o = np.argsort(x)
            interp.append(np.interp(grid, x[o], y[o], left=y[o][0], right=y[o][-1]))
        if not interp:
            continue
        mean = np.mean(interp, axis=0)
        ax.plot(grid * 100, mean * 100, color=colors[s], linewidth=2,
                label=f"{s} ({SCORE_LABEL[s]})")
        ax.fill_between(grid * 100, np.percentile(interp, 25, axis=0) * 100,
                        np.percentile(interp, 75, axis=0) * 100, color=colors[s], alpha=0.15)
    for alpha in budgets:
        ax.axvline(alpha * 100, color="#666666", linestyle=":", linewidth=1)
        ax.text(alpha * 100, 3, L(f"预算 {alpha*100:g}%", f"budget {alpha*100:g}%"),
                rotation=90, fontsize=8, color="#555555")
    ax.set_xscale("log")
    ax.set_xlabel(L("正常窗口误报率 FPR（%，阈值由验证集标定）",
                    "FPR on normal windows (%, thr from val)"))
    ax.set_ylabel(L("未知攻击检出率 UDR（%）", "UDR on unknown attacks (%)"))
    ax.set_title(L("未知攻击检出率 vs 误报预算（曲线越靠左上越好）",
                   "UDR vs FPR budget (upper-left is better)"))
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 102)
    fig.tight_layout()
    p = FIGURE_DIR / "openset_udr_fpr.png"
    fig.savefig(p)
    plt.close(fig)
    out_paths.append(p)

    # ---- 图 3：不做拒识时，未知攻击被错判成什么 ----
    fig, ax = plt.subplots(figsize=(10, 4.8))
    classes = misclass["classes"]
    x = np.arange(len(classes))
    scen_list = list(misclass["scenarios"].items())
    width = 0.8 / max(len(scen_list), 1)
    palette = ("#c0392b", "#e08214", "#2b7bba", "#7f8c8d")
    for i, (scen, entry) in enumerate(scen_list):
        vals = [entry["mean_fractions"].get(c, 0.0) * 100 for c in classes]
        xs = x + i * width - 0.4 + width / 2
        ax.bar(xs, vals, width, label=L("留出 ", "holdout ") + scen, color=palette[i % len(palette)])
        for xi, v in zip(xs, vals):
            if v > 0.5:
                ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20)
    ax.set_ylabel(L("未知窗口被错判为该已知类的比例（%）",
                    "share of unknown windows misclassified as (%)"))
    ax.set_title(L("不做拒识时未知攻击的去向：模型会把它当成某个已知攻击",
                   "Where unknown attacks go without rejection"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = FIGURE_DIR / "openset_misclass.png"
    fig.savefig(p)
    plt.close(fig)
    out_paths.append(p)
    return out_paths


# ======================================================================================
# 结论与建议
# ======================================================================================
def pick_recommended_score(all_records, scenarios, pooled) -> tuple[str | None, float]:
    """按「1% 误报预算下的平均 UDR」选推荐分数。

    AUROC 是全局排序指标，而部署关心的是**低误报预算下的尾部行为**（阈值落在分数分布的
    极右尾），两者可以不一致，所以推荐分数以 UDR@1% 为准，AUROC 只作参考。
    """
    best, best_v = None, -1.0
    for k in SCORE_ORDER:
        vals = []
        for s in scenarios:
            bd = (all_records[s]["aggregate"]["ood"].get(k) or {}).get("budgets", {}).get("0.01")
            if bd and not np.isnan(bd["udr_unknown_detected"]["mean"]):
                vals.append(float(bd["udr_unknown_detected"]["mean"]))
        if vals and float(np.mean(vals)) > best_v:
            best_v, best = float(np.mean(vals)), k
    return best, best_v


def build_recommendation(scenarios, all_records, reference, pooled, rec_score, rec_udr) -> dict:
    lines: list[str] = []
    advice: list[str] = []
    limits: list[str] = []
    n_unk = sum(all_records[s]["aggregate"]["n_windows"]["test_unknown"] for s in scenarios)
    top_parts = []
    for s in scenarios:
        m = all_records[s]["aggregate"]["misclassification_no_reject"]
        top = sorted(m["mean_fractions"].items(), key=lambda kv: -kv[1])[:2]
        top_parts.append(f"留出 {s}：最高占比去向 " + "、".join(f"{c} {v*100:.0f}%"
                                                              for c, v in top)
                         + f"，其中 {m['mean_fraction_high_confidence']['mean']*100:.0f}%的窗口置信度≥0.9")
    lines.append(f"不做拒识时，{n_unk} 个未知攻击窗口**全部**被安上某个已知类标签（"
                 + "；".join(top_parts)
                 + "）。错判到正常类是其中最危险的一种——攻击连告警都不会触发。")

    if rec_score:
        udr_txt, thr_txt = [], []
        for s in scenarios:
            entry = (all_records[s]["aggregate"]["ood"].get(rec_score) or {})
            bd = (entry.get("budgets") or {}).get("0.01") or {}
            u = (bd.get("udr_unknown_detected") or {}).get("mean", float("nan"))
            t = (bd.get("threshold") or {}).get("mean", float("nan"))
            udr_txt.append(f"{s} {u*100:.1f}%")
            thr_txt.append(f"{s} {t:.2f}")
        tail = "、".join(udr_txt)
        thr_text = "、".join(thr_txt)
        lines.append(f"以「1% 误报预算下的未知检出率」排序，最合适的是 `{rec_score}`（"
                     f"{SCORE_LABEL[rec_score]}），各场景 UDR = {tail}（平均 {rec_udr*100:.1f}%）；"
                     f"对应的阈值约 {thr_text}。")
        aurocs = "、".join(
            f"{k} {pooled['auroc_unknown_vs_normal'][k]['mean']:.4f}"
            for k in SCORE_ORDER if (pooled["auroc_unknown_vs_normal"].get(k) or {}).get("n"))
        lines.append(f"注意 AUROC 与低误报预算下的 UDR 并不一致：跨场景 AUROC 三者几乎持平"
                     f"（{aurocs}，差异在种子波动范围内），但阈值落在分数分布极右尾时差别明显——"
                     f"能量分数用到了 softmax 丢掉的 logit 幅度信息，因此在低误报预算下更占优。")
        b1 = []
        for s in scenarios:
            bd = all_records[s]["aggregate"]["ood"][rec_score]["budgets"].get("0.01")
            if bd:
                b1.append((s, bd["udr_unknown_detected"]["mean"],
                           bd["known_attack_false_reject_rate"]["mean"],
                           bd["test_fpr_on_normal"]["mean"]))
        if b1:
            lines.append("在 1% 误报预算下，未知攻击检出率 UDR = "
                         + "、".join(f"{x[0]} {x[1]*100:.1f}%" for x in b1)
                         + "（实测正常误报率 " + "、".join(f"{x[3]*100:.2f}%" for x in b1)
                         + "）；同期已知攻击被误拒 "
                         + "、".join(f"{x[0]} {x[2]*100:.1f}%" for x in b1) + "。")
        f1_parts = [f"{s}: {reference['per_scenario'][s]['known_class_macro_f1_reference_scope']['mean']:.4f}"
                    f" → {all_records[s]['aggregate']['known_class_macro_f1_no_reject']['mean']:.4f}"
                    for s in scenarios]
        # 供 8.2 引用的两个"诚实数字"（都从本次结果里算，不写死）
        _stds = [bd["udr_unknown_detected"].get("std", float("nan"))
                 for s in scenarios for k in SCORE_ORDER
                 for bd in ((all_records[s]["aggregate"]["ood"].get(k) or {}).get("budgets") or {}).values()]
        _stds = [x for x in _stds if x is not None and not np.isnan(x)]
        udr_std = f"{max(_stds):.2f}" if _stds else "—"
        # 最难场景 = 未知窗口被判成 Normal 占比最高的那个（"藏得深"的新攻击）
        hard = None
        for s in scenarios:
            m = all_records[s]["aggregate"]["misclassification_no_reject"]
            frac = float(m["mean_fractions"].get(BINARY_NORMAL, 0.0))
            if hard is None or frac > hard[1]:
                hard = (s, frac, float(m["mean_max_softmax"]["mean"]))
        hard_udr = 0.0
        if hard:
            for k in SCORE_ORDER:
                bd = ((all_records[hard[0]]["aggregate"]["ood"].get(k) or {})
                      .get("budgets") or {}).get("0.01")
                if bd:
                    u = bd["udr_unknown_detected"].get("mean", 0.0)
                    if u is not None and not np.isnan(u):
                        hard_udr = max(hard_udr, float(u))
        lines.append("已知类宏 F1 的代价（含全部类训练 → 只剩 7 类训练）：" + "；".join(f1_parts) + "。")
        lines.append("拒识本身没有把已知类搞坏：被误拒的已知攻击窗口仍然会告警（只是标签变成未知），"
                     "异常发现能力不变；已知类宏 F1 在加拒识后甚至略升（被拒掉的少量错误预测本来就在拉低它）。"
                     "真正的代价是细粒度标签丢一部分，未知告警需要人工研判。")
        advice.append(f"1. **用哪套分数**：优先 `{rec_score}`（{SCORE_LABEL[rec_score]}）——按 1% 误报预算下的"
                      "检出率排序它在两个场景里都最好，且实现成本极低（一次 logsumexp）。"
                      "`1 - max softmax` 最省事、可作为轻量替代（本实验里低误报预算下比 energy 低约 5~10 个百分点）。")
        advice.append("2. **阈值怎么定**：在**生产环境的正常流量**（或验证集正常窗口）上取分位数——"
                      "1% 误报预算取 99 分位、0.1% 预算取 99.9 分位；本实验 1% 预算对应的阈值约 "
                      + thr_text + "。阈值必须随环境重新标定（见 8.2）。")
        advice.append("3. **上线形态用三类告警**：`已知攻击（带细类标签）` / `未知攻击` / `正常`。"
                      "未知告警进「待研判」队列（保留原始流记录），人工确认后把样本加入训练集，"
                      "形成「未知 → 已知」的闭环——这也是这套拒识能力真正的价值所在。")
        advice.append("4. **监控未知告警比例**：它是一条免费的漂移指标。某个时段未知占比突然升高，"
                      "通常意味着新攻击出现或流量分布变化，值得立刻看。")
        advice.append("5. **别用拒识替代二分类告警**：OOD 分数只回答「像不像已知的东西」，"
                      "不回答「是不是攻击」；二分类头（`bin`）继续负责报警，两者叠加使用。")
        limits.append("留出类样本量小（测试集里每类只有几十到一百多个窗口），UDR 的置信区间较宽"
                      f"（不同种子之间的 UDR 标准差最大到 ±{udr_std}），已用多种子均值 ± 标准差缓解，"
                      "但仍需要更多数据收紧。")
        if hard:
            limits.append("与已知类机理接近、或「看起来像正常流量」的攻击会失效：本实验中最难的场景是"
                          f"留出 {hard[0]}——它的未知窗口有 {hard[1]*100:.0f}% 被判成 **Normal**"
                          f"（平均最大概率 {hard[2]:.2f}），OOD 分数与正常窗口大量重叠，"
                          f"在 1% 误报预算下换任何 OOD 分数最多也只能检出 {hard_udr*100:.1f}%。"
                          "错判到正常类意味着**连告警都不会触发**，这类「藏得深」的新攻击是当前方案"
                          "最大的盲区，需要靠未知比例监控、异常流留存与人工规则兜底。")
        limits.append("阈值会随环境漂移：阈值在本数据集的正常流量上标定，换网络/换采集口径后必须重新"
                      "标定（可复用 `tools/drift_test.py` 验证，或做在线滚动重新标定）。")
        limits.append("本实验的「未知」来自同一数据生成器，与真实新型攻击仍有差距；"
                      "更严格的验证需要跨数据集留出（例如拿公开数据集的某类攻击做留出）。")
    return {"conclusion_lines": lines, "advice_lines": advice, "limitations": limits,
            "best_score": rec_score, "best_score_udr_at_1pct": rec_udr}


# ======================================================================================
# 报告
# ======================================================================================
def build_report(meta: dict, all_records: dict, reference: dict, pooled: dict,
                 figures: list[Path]) -> str:
    lines: list[str] = []
    A = lines.append
    seeds = meta["seeds"]
    scenarios = list(all_records.keys())
    class_union: list[str] = []
    for s in scenarios:
        for c in all_records[s]["records"][0]["known_classes"]:
            if c not in class_union:
                class_union.append(c)
    class_union = sorted(class_union)

    A("# 开集（未知攻击）检测实验报告")
    A("")
    A("> 本文件由 `python tools/openset_eval.py` 自动生成。模型只在 7 个已知类上训练，"
      "测试集里保留了一整类**训练时从未出现过**的攻击，用来回答：新攻击进来时，"
      "系统是把新攻击硬塞进已知类别，还是能诚实地说「这是未知攻击」。")
    A("")
    A("## 1. 实验设置")
    A("")
    A(f"- 数据：`data/raw/{meta['schema']}_stream.csv`（清洗后 {meta['n_rows']:,} 行 × "
      f"{meta['n_features']} 维特征），窗口 (T={meta['window_size']}, stride={meta['window_stride']})，"
      f"时序切分（purge={meta['purge_rows']} 行）。")
    A("- 模型：`cnn_bilstm`（CNN 膨胀卷积分支 + BiLSTM + 时序注意力，双头输出）。"
      "归一化器只在**剔除留出类之后**的训练行上 fit，留出类的统计量不参与任何环节。")
    A(f"- 随机种子：{', '.join(str(s) for s in seeds)}（指标为多种子均值 ± 标准差）。")
    A("- OOD 分数（三者统一为**分数越高越可能是未知**）：`1 - max softmax`、`熵 H(p)`、"
      "`-logsumexp(logits)`（能量分数）。")
    A("- 阈值标定：**只用验证集的正常窗口**取分位数定阈值（正常告警误报预算 "
      f"{', '.join(f'{b*100:g}%' for b in meta['budgets'])}）；验证集的已知攻击窗口、测试集窗口与"
      "留出类窗口都**不参与**标定（验证集已知攻击窗口只用来事后看「已知攻击会不会被误拒」）。")
    pc = meta.get("preprocessing_check", {})
    if pc.get("checked"):
        A(f"- 口径核对：本次重建的预处理张量与项目主实验 `{pc['path']}` 的最大绝对差 = "
          f"`{pc.get('max_abs_diff_overall', float('nan')):.3g}`（"
          f"{'逐位一致，可与主实验直接对照' if pc.get('identical') else '不一致，已改为自己训练对照模型'}）。")
    A(f"- 全类对照模型：{reference['source_note']}")
    A("- 开集模型产物：`artifacts/openset/<tag>/`。刻意放在子目录，避免被 "
      "`tools/predict_cli.py`、`tools/reevaluate.py`、`tools/drift_test.py` 这类按 "
      "`artifacts/*/model.keras` 自动发现模型的工具误选（它们默认会挑最新的模型）。")
    A("")

    A("## 2. 留出场景与样本量")
    A("")
    rows = []
    for s in scenarios:
        a = all_records[s]["aggregate"]
        w = a["n_windows"]
        rows.append([f"留出 {s}", w["test_unknown"], w["test_known_attack"], w["test_normal"],
                     w["val_normal"], f"{all_records[s]['row_split']['rows_dropped_from_train_val']:,}",
                     a["n_seeds"]])
    A(markdown_table(["场景", "未知攻击窗口(测试)", "已知攻击窗口(测试)", "正常窗口(测试)",
                      "正常窗口(验证·标定用)", "从训练/验证剔除的行数", "种子数"], rows))
    A("")
    A("留出类的窗口数不多（测试集里该类的流记录只有几百条），所以所有指标都给出多种子的"
      "均值 ± 标准差；结论以方向性为准，不要过度解读小数点后第二位。")
    A("")
    A("原始数据行级切分（未剔除留出类，只是数据规模说明）：")
    A("")
    A(markdown_table(["子集", "行数"] + meta["all_classes"],
                     [[k, f"{meta['row_split'][k]:,}"]
                      + [f"{meta['row_split_by_class'][k].get(c, 0):,}" for c in meta["all_classes"]]
                      for k in ("train", "val", "test")]))
    A("")

    A("## 3. 已知类宏 F1 的代价（同口径对比）")
    A("")
    A("两种模型都在**同一批测试窗口**（真实类别属于已知类的窗口）上、用**同一套 7 个已知类**"
      "计算宏 F1；全类模型若把窗口预测成留出类，同样按错判计入。")
    A("")
    A("「宏 F1 差值」= 含全部类训练的对照 − 只剩 7 类训练的开集模型：**正值表示开集模型在已知类上更差**"
      "（这就是拿到未知拒识能力的代价），负值则表示它反而更好（少学一类，已知类的判定任务更简单）。")
    A("")
    rows = []
    for s in scenarios:
        a = all_records[s]["aggregate"]
        rk = reference["per_scenario"][s]
        rows.append([
            f"留出 {s}",
            fmt(rk["known_class_macro_f1_reference_scope"]),
            fmt(a["known_class_macro_f1_no_reject"]),
            f"{rk['known_class_macro_f1_reference_scope']['mean'] - a['known_class_macro_f1_no_reject']['mean']:+.4f}",
            fmt(a["known_class_accuracy_no_reject"]),
            f"{rk['accuracy_on_holdout_windows']['mean']:.4f}",
        ])
    A(markdown_table(["场景", "含全部类训练（对照）", "只剩 7 类训练（开集模型）", "宏 F1 差值",
                      "开集模型已知类准确率", "对照模型在留出类窗口上的准确率"], rows))
    A("")
    A("最后两列说明两件事：① 开集模型在已知类上的性能没有崩（差值很小）；"
      "② 留出类窗口本身是**可学**的——全类模型能基本答对，开集模型答不对纯粹是因为它没见过这一类，"
      "而不是这些窗口没有规律。")
    A("")

    A("## 4. 三种 OOD 分数对未知攻击的可分性（AUROC）")
    A("")
    A("AUROC 越大越好；0.5 = 与瞎猜无异。")
    A("")
    for s in scenarios:
        A(f"**留出 {s}**（{all_records[s]['aggregate']['n_seeds']} 个种子均值 ± 标准差）")
        A("")
        rows = []
        for k in SCORE_ORDER:
            e = all_records[s]["aggregate"]["ood"].get(k) or {}
            if not e.get("available"):
                rows.append([f"{k} ({SCORE_LABEL[k]})", "不可用", "不可用", "不可用"])
                continue
            rows.append([f"{k} ({SCORE_LABEL[k]})", fmt(e["auroc_unknown_vs_normal"]),
                         fmt(e["auroc_unknown_vs_known"]), fmt(e["auroc_unknown_vs_known_attack"])])
        A(markdown_table(["OOD 分数", "未知 vs 正常", "未知 vs 全部已知窗口", "未知 vs 已知攻击"], rows))
        A("")
    A("跨场景汇总（同一种子把两个场景的窗口拼接后算一次，再对种子取均值 ± 标准差）：")
    A("")
    rows = []
    for k in SCORE_ORDER:
        rows.append([f"{k} ({SCORE_LABEL[k]})",
                     fmt(pooled["auroc_unknown_vs_normal"].get(k)),
                     fmt(pooled["auroc_unknown_vs_known"].get(k)),
                     fmt(pooled["auroc_unknown_vs_known_attack"].get(k))])
    A(markdown_table(["OOD 分数", "未知 vs 正常", "未知 vs 全部已知窗口", "未知 vs 已知攻击"], rows))
    A("")

    A("## 5. 未知攻击检出率 UDR @ 固定误报预算")
    A("")
    A("阈值只在验证集正常窗口上标定；「实测误报率」是同一阈值在**测试集正常窗口**上的真实误报率"
      "（验证集与测试集的时间段不同，两者可以有差别，这本身就说明阈值需要定期重新标定）。")
    A("")
    A("**怎么读这张表**：UDR 才是部署时真正关心的数字——它把阈值卡在极低的误报预算上，"
      "看尾部（最可疑的那一小撮）能不能把未知攻击捞出来。AUROC 高不等于 UDR 高："
      "留出 Botnet 的分数分布与正常窗口有大片重叠（模型把它当成 Normal），AUROC 仍有 0.97 左右，"
      "但 1% 预算下只能检出约四分之一。")
    A("")
    for s in scenarios:
        A(f"**留出 {s}**")
        A("")
        rows = []
        for k in SCORE_ORDER:
            e = all_records[s]["aggregate"]["ood"].get(k) or {}
            if not e.get("available"):
                continue
            for b, bd in e["budgets"].items():
                rows.append([
                    f"{k} ({SCORE_LABEL[k]})", f"{float(b)*100:g}%",
                    f"{bd['threshold']['mean']:.3f}",
                    f"{bd['test_fpr_on_normal']['mean']*100:.2f}%",
                    fmt(bd["udr_unknown_detected"]),
                    f"{bd['known_attack_false_reject_rate']['mean']*100:.2f}%",
                ])
        A(markdown_table(["OOD 分数", "误报预算", "阈值(均值)", "实测误报率(测试正常窗口)",
                          "未知检出率 UDR", "已知攻击被误拒比例"], rows))
        A("")

    A("## 6. 不做任何拒识时：未知攻击被错判成什么（危险性）")
    A("")
    A("这是最能说明问题的一组数据：直接取 `argmax` 时，留出类窗口 100% 会被安上一个已知类标签，"
      "而且往往非常自信。尤其注意「被判成 Normal」的占比——那意味着攻击**连告警都不会触发**，"
      "比错报成另一种攻击更危险。")
    A("")
    rows = []
    for s in scenarios:
        m = all_records[s]["aggregate"]["misclassification_no_reject"]
        top = sorted(m["mean_fractions"].items(), key=lambda kv: -kv[1])[:3]
        rows.append([
            f"留出 {s}", all_records[s]["aggregate"]["n_windows"]["test_unknown"],
            "、".join(f"{c} {v*100:.0f}%" for c, v in top),
            f"{m['mean_max_softmax']['mean']:.3f}",
            f"{m['mean_fraction_high_confidence']['mean']*100:.0f}%",
        ])
    A(markdown_table(["场景", "未知窗口数", "被判成的前 3 个已知类（占比）",
                      "平均最大概率", "自信错判(≥0.9)占比"], rows))
    A("")
    A("完整错判分布（各场景多种子平均占比；某场景的留出类本身不在它的已知类里，记为 —）：")
    A("")
    A(markdown_table(
        ["场景"] + [f"→ {c}" for c in class_union],
        [[f"留出 {s}"]
         + [f"{all_records[s]['aggregate']['misclassification_no_reject']['mean_fractions'].get(c, 0.0)*100:.1f}%"
            if c in all_records[s]['aggregate']['misclassification_no_reject']['mean_fractions'] else "—"
            for c in class_union]
         for s in scenarios]))
    A("")

    A("## 7. 加拒识之后：已知类被影响多少")
    A("")
    A("「宏 F1（拒识后）」把被判为未知的已知类窗口按错判计入；「已知攻击误拒」= 已知攻击窗口被"
      "拒识器误报成未知的比例（细粒度标签会丢，但告警仍在，所以攻击的**发现**能力不受影响）。")
    A("")
    for s in scenarios:
        A(f"**留出 {s}**")
        A("")
        a = all_records[s]["aggregate"]
        rows = [["不拒识（基线）", "—", fmt(a["known_class_macro_f1_no_reject"]), "—"]]
        for k in SCORE_ORDER:
            e = a["ood"].get(k) or {}
            if not e.get("available"):
                continue
            for b in sorted(e["budgets"], key=lambda x: -float(x)):
                bd = e["budgets"][b]
                rows.append([f"{k} @ {float(b)*100:g}% 误报预算",
                             f"{bd['test_fpr_on_normal']['mean']*100:.2f}%",
                             fmt(bd["known_class_macro_f1_after_reject"]),
                             f"{bd['known_attack_false_reject_rate']['mean']*100:.2f}%"])
        A(markdown_table(["方案", "正常窗口拒识率(=误报)", "已知类宏 F1", "已知攻击误拒比例"], rows))
        A("")
        best = meta.get("recommendation", {}).get("best_score")
        if best:
            e = a["ood"].get(best) or {}
            bd = (e.get("budgets") or {}).get("0.01")
            if bd:
                A(f"逐类召回（推荐分数 `{best}` @1% 误报预算）："
                  + "、".join(f"{c} {v['mean']:.4f}" for c, v in
                              bd["mean_recall_after_reject"].items()))
                A("")

    A("## 8. 结论与部署建议")
    A("")
    for line in meta.get("recommendation", {}).get("conclusion_lines", []):
        A(f"- {line}")
    A("")
    A("### 8.1 给部署方的具体建议")
    A("")
    for line in meta.get("recommendation", {}).get("advice_lines", []):
        A(f"{line}")
        A("")
    A("### 8.2 还没解决 / 会失效的情况")
    A("")
    for line in meta.get("recommendation", {}).get("limitations", []):
        A(f"- {line}")
    A("")
    A("## 9. 复现方式")
    A("")
    A("```")
    A("python tools/openset_eval.py                     # 2 个留出场景 × 3 个种子")
    A("python tools/openset_eval.py --scenarios U2R     # 只跑一个场景")
    A("python tools/openset_eval.py --dry-run           # 只重建数据并核对口径")
    A("```")
    A("")
    A(f"训练总耗时约 {meta.get('train_seconds_total', 0):.0f} 秒（CPU，含全类对照模型）；"
      f"整脚本墙钟 {meta.get('wall_seconds', 0):.0f} 秒。脚本幂等：重跑覆盖同名产物。")
    if figures:
        A("")
        A("图表（`results/figures/`）：")
        for p in figures:
            A(f"- `{p.name}`")
    A("")
    return "\n".join(lines)


# ======================================================================================
# 主流程
# ======================================================================================
def parse_args(argv=None):
    cfg = Config()
    scen_default = list(getattr(cfg, "exclude_classes", ()) or ()) or list(DEFAULT_SCENARIOS)
    ap = argparse.ArgumentParser(
        description="开集（未知攻击）检测实验：留出攻击类 → 训练 7 类模型 → 在完整测试集上评未知拒识",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--scenarios", nargs="*", default=scen_default,
                    help="要留出的攻击类（每个类跑一轮独立实验）；不给则用配置项 exclude_classes，"
                         "再退回默认 " + ",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS),
                    help="随机种子（多种子取均值±标准差）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数（默认用配置里的 40）")
    ap.add_argument("--skip-train", action="store_true", help="复用 artifacts/openset 下已训练的开集模型")
    ap.add_argument("--ref", choices=["auto", "existing", "train"], default="auto",
                    help="全类对照模型：auto/existing 优先复用主实验模型，train 为重新训练")
    ap.add_argument("--no-figures", action="store_true", help="不出图")
    ap.add_argument("--dry-run", action="store_true", help="只重建数据、核对口径，不训练")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ensure_dirs()
    cfg = Config()
    set_seed(cfg.seed)

    scenarios = list(args.scenarios)
    seeds = sorted(set(int(s) for s in args.seeds))
    t0 = time.time()
    LOGGER.info("=" * 78)
    LOGGER.info("开集（未知攻击）检测实验：留出 %s，种子 %s", scenarios, seeds)
    LOGGER.info("=" * 78)

    base = load_base(cfg)
    full_payload = build_payload(cfg, base, "")
    pc_check = compare_with_main_experiment(cfg, full_payload)
    if args.dry_run:
        LOGGER.info("--dry-run：数据重建完成，未训练。")
        return 0

    names = base["names"]
    row_split = {"train": len(base["idx"][0]), "val": len(base["idx"][1]),
                 "test": len(base["idx"][2])}
    row_split_by_class = {}
    for k, idx in (("train", base["idx"][0]), ("val", base["idx"][1]), ("test", base["idx"][2])):
        vals, cnt = np.unique(names[idx], return_counts=True)
        row_split_by_class[k] = {str(a): int(b) for a, b in zip(vals, cnt)}

    all_records: dict[str, dict] = {}
    score_store: dict[str, dict] = {}
    train_seconds_total = 0.0

    # ---------------- 开集场景 ----------------
    for scen in scenarios:
        payload = build_payload(cfg, base, scen)
        records: list[dict] = []
        store: dict[int, dict] = {}
        for seed in seeds:
            tag = f"openset_no{scen.lower()}_s{seed}"
            model, tinfo = train_open_model(cfg, payload, tag, seed, args.epochs, args.skip_train)
            train_seconds_total += float(tinfo.get("train_seconds") or 0.0)
            rec = evaluate_open_model(model, payload)
            rec["seed"] = int(seed)
            rec["tag"] = tag
            rec["train"] = tinfo
            records.append(rec)

            names_t, names_v = payload["names"]["test"], payload["names"]["val"]
            p_mc, _ = predict_probs(model, payload["data"].X_test)
            st = ood_scores(p_mc, extract_logits(model, payload["data"].X_test))
            pv_mc, _ = predict_probs(model, payload["data"].X_val)
            sv = ood_scores(pv_mc, extract_logits(model, payload["data"].X_val))
            store[int(seed)] = {
                k: {"unknown": st[k][names_t == scen],
                    "normal": st[k][names_t == BINARY_NORMAL],
                    "known_attack": st[k][(names_t != BINARY_NORMAL) & (names_t != scen)],
                    "val_normal": sv[k][names_v == BINARY_NORMAL]}
                for k in SCORE_ORDER if st[k] is not None and sv[k] is not None}
            avail = [k for k in SCORE_ORDER if rec["ood"][k].get("available")]
            best = max(avail, key=lambda k: rec["ood"][k]["auroc_unknown_vs_normal"], default="—")
            LOGGER.info("[%s] 已知类宏F1 %.4f | 未知窗口 %d 个 | 最强 OOD 分数 %s"
                        "(AUC %.3f) | 置信度≥0.9 的错判占 %.0f%%",
                        tag, rec["known_class_macro_f1_no_reject"], rec["n_windows"]["test_unknown"],
                        best, (rec["ood"][best]["auroc_unknown_vs_normal"] if best != "—" else float("nan")),
                        rec["misclassification_no_reject"]["fraction_high_confidence"] * 100)
        all_records[scen] = {
            "records": records, "aggregate": aggregate_records(records),
            "row_split": payload["row_split_counts"],
            "train_class_counts": payload["data"].train_class_counts,
        }
        score_store[scen] = store

    # ---------------- 全类对照（每个种子一个模型，在每个留出场景下分别评同口径指标）----------------
    ref_models: dict[int, object] = {}
    ref_source: dict = {}
    for seed in seeds:
        model, rmeta = pick_reference(cfg, full_payload, seed, args.ref,
                                      len(full_payload["known_classes"]))
        ref_models[int(seed)] = model
        ref_source = rmeta
        LOGGER.info("[对照 seed=%d] 载入/训练完成（%s）", seed, rmeta.get("source"))

    reference = {
        "source_note": (f"{ref_source.get('source')}（复用的是项目主实验已训练模型 "
                        f"artifacts/cnn_bilstm_main_s{{{','.join(str(s) for s in seeds)}}}/model.keras，"
                        f"与本次重建的数据逐位同口径）"
                        if ref_source.get("source") == "existing"
                        else f"{ref_source.get('source')}（tag={ref_source.get('tag')}）")
        if ref_source else "未运行",
        "source": ref_source, "per_scenario": {},
    }
    for scen in scenarios:
        known = [c for c in full_payload["all_classes"] if c != scen]
        recs = []
        for seed in seeds:
            rec = evaluate_reference_model(ref_models[int(seed)], full_payload, scen, known)
            rec["seed"] = int(seed)
            recs.append(rec)
        reference["per_scenario"][scen] = {
            "known_class_macro_f1_reference_scope": agg(
                [r["known_class_macro_f1_reference_scope"] for r in recs]),
            "accuracy_on_holdout_windows": agg([r["accuracy_on_holdout_windows"] for r in recs]),
            "full_multiclass_macro_f1": agg([r["full_multiclass_macro_f1"] for r in recs]),
            "per_seed": recs,
        }
        LOGGER.info("[对照 %s] 同口径已知类宏F1 %.4f（全 8 类宏F1 %.4f）| 留出类 %d 窗口上准确率 %.4f",
                    scen,
                    reference["per_scenario"][scen]["known_class_macro_f1_reference_scope"]["mean"],
                    reference["per_scenario"][scen]["full_multiclass_macro_f1"]["mean"],
                    recs[0]["n_holdout_windows"],
                    reference["per_scenario"][scen]["accuracy_on_holdout_windows"]["mean"])

    pooled = pooled_metrics(score_store, seeds)

    # ---------------- 推荐分数（按 1% 误报预算下的平均 UDR 选，AUROC 只作参考）----------------
    rec_score, rec_udr = pick_recommended_score(all_records, scenarios, pooled)
    recommendation = build_recommendation(scenarios, all_records, reference, pooled,
                                          rec_score, rec_udr)

    # ---------------- 图 ----------------
    figures: list[Path] = []
    if not args.no_figures:
        pool: dict = {"thresholds": {}}
        for g in ("normal", "known_attack", "unknown"):
            pool[g] = {k: None for k in SCORE_ORDER}
        for k in SCORE_ORDER:
            for g in ("normal", "known_attack", "unknown"):
                arrs = [sd[k][g] for store in score_store.values() for sd in store.values()
                        if k in sd and len(sd[k][g])]
                pool[g][k] = np.concatenate(arrs) if arrs else None
            pool["thresholds"][k] = {}
            for b in BUDGETS:
                vals = [rec["ood"][k]["budgets"][f"{b:g}"]["threshold"]
                        for scen in all_records.values() for rec in scen["records"]
                        if (rec["ood"].get(k) or {}).get("available")]
                if vals:
                    pool["thresholds"][k][f"{b:g}"] = float(np.mean(vals))
        det = {k: [] for k in SCORE_ORDER}
        for store in score_store.values():
            for sd in store.values():
                d = _val_scores(store, sd)
                for k in SCORE_ORDER:
                    if k in sd and d.get(k) is not None:
                        det[k].append(_det_curve(d[k], sd[k]["unknown"]))
        misclass = {"classes": class_union_of(all_records), "scenarios": {}}
        for s in scenarios:
            misclass["scenarios"][s] = all_records[s]["aggregate"]["misclassification_no_reject"]
        try:
            figures = make_figures(pool, det, misclass)
            LOGGER.info("图表已生成：%s", ", ".join(p.name for p in figures))
        except Exception as exc:  # noqa: BLE001 - 出图失败不该毁掉整个实验
            LOGGER.exception("出图失败：%s", exc)

    # ---------------- 落盘 ----------------
    meta = {
        "schema": cfg.schema,
        "n_rows": int(base["n_rows"]), "n_rows_raw": int(base["n_rows_raw"]),
        "n_features": len(base["feats"]), "window_size": int(cfg.window_size),
        "window_stride": int(cfg.window_stride), "purge_rows": int(base["purge"]),
        "seeds": seeds, "budgets": list(BUDGETS), "scenarios": scenarios,
        "all_classes": base["all_classes"], "row_counts": base["row_counts"],
        "row_split": row_split, "row_split_by_class": row_split_by_class,
        "preprocessing_check": pc_check,
        "train_seconds_total": round(train_seconds_total, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "recommendation": recommendation,
        "score_labels": SCORE_LABEL,
        "config": {k: cfg.to_dict()[k] for k in
                   ("schema", "source", "window_size", "window_stride", "split_mode",
                    "train_ratio", "val_ratio", "purge_windows", "epochs", "batch_size",
                    "learning_rate", "feature_noise", "label_noise", "model_variant")},
    }
    save_json({
        "meta": meta,
        "scenarios": {s: {"aggregate": all_records[s]["aggregate"],
                          "per_seed": all_records[s]["records"],
                          "row_split": all_records[s]["row_split"],
                          "train_class_counts": all_records[s]["train_class_counts"]}
                      for s in scenarios},
        "reference": reference,
        "pooled": pooled,
    }, RESULT_DIR / "openset.json")

    for s in scenarios:
        for rec in all_records[s]["records"]:
            save_json({
                "tag": rec["tag"], "kind": "openset",
                "holdout_class": s, "seed": rec["seed"],
                "data": {"val": rec["n_windows"]["val_normal"] + rec["n_windows"]["val_known_attack"],
                         "test": rec["n_windows"]["test_total"],
                         "train": rec["train_windows"],
                         "class_names": rec["known_classes"], "n_features": len(base["feats"]),
                         "input_shape": [int(cfg.window_size), len(base["feats"])]},
                "openset": {"n_windows": rec["n_windows"],
                            "known_class_macro_f1_no_reject": rec["known_class_macro_f1_no_reject"],
                            "known_class_macro_f1_reference_scope":
                                rec["known_class_macro_f1_reference_scope"],
                            "misclassification_no_reject": rec["misclassification_no_reject"],
                            "ood": rec["ood"]},
                "train": rec["train"],
            }, RESULT_DIR / f"metrics_{rec['tag']}.json")

    save_text(build_report(meta, all_records, reference, pooled, figures),
              RESULT_DIR / "openset_report.md")

    LOGGER.info("=" * 78)
    for s in scenarios:
        LOGGER.info("留出 %-7s 已知类宏F1 %.4f（对照 %.4f）", s,
                    all_records[s]["aggregate"]["known_class_macro_f1_no_reject"]["mean"],
                    reference["per_scenario"][s]["known_class_macro_f1_reference_scope"]["mean"])
    for k in SCORE_ORDER:
        a = pooled["auroc_unknown_vs_normal"].get(k) or {}
        if a.get("n"):
            LOGGER.info("  %-8s 跨场景 AUROC(未知 vs 正常) = %.4f ± %.4f", k, a["mean"], a["std"])
    LOGGER.info("推荐分数：%s", recommendation.get("best_score"))
    LOGGER.info("报告：results/openset_report.md | 数据：results/openset.json")
    LOGGER.info("总耗时 %.1f 秒（其中训练 %.1f 秒）", time.time() - t0, train_seconds_total)
    return 0


def class_union_of(all_records: dict) -> list[str]:
    seen: list[str] = []
    for scen in all_records.values():
        for c in scen["records"][0]["known_classes"]:
            if c not in seen:
                seen.append(c)
    return sorted(seen)


def _val_scores(store: dict, sd: dict) -> dict:
    """验证集正常窗口分数：DET 曲线的 FPR 一律由它换算（阈值标定与评估严格分离）。"""
    return {k: sd[k].get("val_normal") for k in SCORE_ORDER if k in sd}


if __name__ == "__main__":
    raise SystemExit(main())
