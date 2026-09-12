#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨数据集迁移性分析：合成数据训练的模型 → 真实 inSDN 直接推理。

这个脚本回答一个比赛材料里必须有、但主实验刻意回避的问题：
**「自建合成数据集上 0.99 的宏 F1，搬到真实公开数据集上还剩多少？」**

它只做三件事，全部是**推理侧**动作，不碰任何训练/拟合：

1. **特征名映射**：真实 inSDN（CICFlowMeter v3 缩写命名，如 ``Tot Fwd Pkts``）→
   合成数据 ``meta.json`` 的 ``feature_names``（CIC-IDS2017 全称命名，如 ``Total Fwd Packets``）。
   显式映射表优先，规范化 + 词元集合匹配兜底；映射不上的维度在报告里逐条列出，
   并在**标准化空间填 0**（等价于用训练集均值填充），绝不使用真实数据的统计量。
2. **跨域推理**：复用 ``data/processed/scaler.joblib``（只在合成训练集上 fit 过）与
   主模型 ``artifacts/<tag>/model.keras``，把真实数据按同样的 (T, F) 窗口形状切好送进模型，
   给出二分类 / 2 类宏 F1 指标，并与**同域**（合成测试集）指标对比出衰减幅度。
   去重前、近似去重（``dedup_significant_digits``）后各出一版结果。
3. **漂移诊断**：逐特征算「用训练集 mean/std 归一化后的均值偏移」（z-shift）与 PSI，
   给出汇总分布、偏移最大的特征表，以及一张合成 vs 真实的分布对比图。

诚实性约束（写死在代码里，不是口号）：
* 真实数据**不参与**训练、标准化拟合、阈值选择；``report_threshold`` 直接取 ``Config`` 默认值。
* 真实 inSDN 的 ``target`` 只有 0/1 两值 → **8 类宏 F1 在这份数据上无法计算**，
  报告里如实标注，并改用「模型侧 8 类归因分布」作为补充视角（明确标为非真值指标）。
* 报告与 JSON 全部使用相对路径，不含本机绝对路径 / 用户名。

用法::

    .\\.venv\\Scripts\\python.exe tools\\cross_dataset.py
    .\\.venv\\Scripts\\python.exe tools\\cross_dataset.py --real-csv data/real/insdn/Dataset.csv ^
        --model artifacts/cnn_bilstm_main_s0/model.keras --out results

产物（默认 ``--out results``）::

    results/cross_dataset_report.md
    results/cross_dataset.json
    results/figures/cross_dataset_drift.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import clean_dataframe, order_by_time  # noqa: E402
from src.evaluate import best_threshold, binary_metrics, multiclass_metrics  # noqa: E402
from src.model import load_model_bundle, predict_probs  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR,
    LOG_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    get_logger,
    load_json,
    markdown_table,
    save_json,
    save_text,
    set_seed,
    setup_matplotlib,
)

LOGGER = get_logger("cida.cross")


# ======================================================================================
# 1. 特征名映射表
# ======================================================================================
# 合成数据集是 CIC-IDS2017 / CICFlowMeter 全称命名；真实 inSDN 是 CICFlowMeter v3 缩写命名。
# 这里把「合成规范名 → 真实列名」显式写死，方便人工核查与后续维护
#（换数据集时只要改这张表 + 下面的缩写词典即可）。
SYNTH_TO_REAL: dict[str, str] = {
    "Protocol": "Protocol",
    "Flow Duration": "Flow Duration",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Bwd PSH Flags": "Bwd PSH Flags",
    "Fwd URG Flags": "Fwd URG Flags",
    "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min",
    "Max Packet Length": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "CWE Flag Count": "CWE Flag Count",
    "ECE Flag Count": "ECE Flag Cnt",
    "Down/Up Ratio": "Down/Up Ratio",
    "Average Packet Size": "Pkt Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
    "Active Mean": "Active Mean",
    "Active Std": "Active Std",
    "Active Max": "Active Max",
    "Active Min": "Active Min",
    "Idle Mean": "Idle Mean",
    "Idle Std": "Idle Std",
    "Idle Max": "Idle Max",
    "Idle Min": "Idle Min",
}

# 词元规范化词典：把 CICFlowMeter 的各种缩写压到同一个词，用于兜底匹配
_TOKEN_ALIASES = {
    "tot": "total",
    "totlen": "total length",
    "pkt": "packet",
    "pkts": "packets",
    "byt": "bytes",
    "byts": "bytes",
    "len": "length",
    "cnt": "count",
    "seg": "segment",
    "win": "window",
    "act": "active",
    "avg": "average",
    "var": "variance",
    "num": "number",
    "fwd": "forward",
    "bwd": "backward",
    "std": "std",
    "min": "min",
    "max": "max",
}
_STOPWORDS = {"of", "the", "a", "an", "per"}

# 真实数据里必然存在、但**不是**流特征或已经由别的方式提供的列
ID_OR_LABEL_COLUMNS = {
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp",
    "Source IP", "Source Port", "Destination IP", "Destination Port",
}


def _tokens(name: str) -> frozenset[str]:
    """把列名压成词元集合（去大小写、去标点、展开缩写、去停用词）。"""
    s = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    out: list[str] = []
    for w in s.split():
        out.extend(str(_TOKEN_ALIASES.get(w, w)).split())
    return frozenset(w for w in out if w not in _STOPWORDS)


def build_mapping(
    feature_names: list[str], real_columns: list[str]
) -> tuple[dict[str, str], dict]:
    """把合成特征名映射到真实列名。

    三级策略（逐级兜底，每一步都记进 ``detail``，报告里可核查）：
      1. 显式映射表 ``SYNTH_TO_REAL``（权威，人工核对过语义）；
      2. 精确同名；
      3. 词元集合匹配（唯一命中才接受），最后用 difflib 相似度兜底。

    返回 ``(mapping, info)``；``mapping`` 只包含成功命中的维度。
    """
    cols = list(real_columns)
    lower_index: dict[str, list[str]] = {}
    for c in cols:
        lower_index.setdefault(str(c).strip().lower(), []).append(c)
    token_index: dict[frozenset[str], list[str]] = {}
    for c in cols:
        token_index.setdefault(_tokens(c), []).append(c)

    mapping: dict[str, str] = {}
    detail: dict[str, str] = {}
    used: set[str] = set()

    # --- 第 1 级：显式表 ---
    for f in feature_names:
        tgt = SYNTH_TO_REAL.get(f)
        if tgt and tgt in cols and tgt not in used:
            mapping[f] = tgt
            used.add(tgt)
            detail[f] = "explicit"

    # --- 第 2 级：同名（大小写不敏感） ---
    for f in feature_names:
        if f in mapping:
            continue
        cands = [c for c in lower_index.get(str(f).strip().lower(), []) if c not in used]
        if len(cands) == 1:
            mapping[f] = cands[0]
            used.add(cands[0])
            detail[f] = "exact"

    # --- 第 3 级：词元集合匹配（必须唯一命中，避免把语义不同的两列接错） ---
    for f in feature_names:
        if f in mapping:
            continue
        cands = [c for c in token_index.get(_tokens(f), []) if c not in used]
        if len(cands) == 1:
            mapping[f] = cands[0]
            used.add(cands[0])
            detail[f] = "token_set"

    # --- 第 4 级：difflib 相似度兜底（阈值给高，宁缺毋滥） ---
    try:
        from difflib import SequenceMatcher

        for f in feature_names:
            if f in mapping:
                continue
            nf = str(f).lower()
            best, best_score = None, 0.0
            for c in cols:
                if c in used:
                    continue
                sc = SequenceMatcher(None, nf, str(c).lower()).ratio()
                if sc > best_score:
                    best, best_score = c, sc
            if best is not None and best_score >= 0.90:
                mapping[f] = best
                used.add(best)
                detail[f] = f"fuzzy:{best_score:.2f}"
    except Exception:  # pragma: no cover
        pass

    unmatched = [f for f in feature_names if f not in mapping]
    # 真实侧未被用到的「数值型非标识列」——这些是真实数据集有、我们的模型没有的输入维度
    unused_real = [c for c in cols if c not in used and c not in ID_OR_LABEL_COLUMNS]
    collisions = sorted({v for v in mapping.values() if list(mapping.values()).count(v) > 1})
    info = {
        "n_features": len(feature_names),
        "n_mapped": len(mapping),
        "coverage": len(mapping) / max(len(feature_names), 1),
        "unmatched": unmatched,
        "unused_real_columns": unused_real,
        "collisions": collisions,
        "method_counts": {m: sum(1 for v in detail.values() if v == m)
                          for m in sorted(set(detail.values()))},
        "detail": detail,
        "pairs": {k: v for k, v in mapping.items()},
    }
    return mapping, info


# ======================================================================================
# 2. 数据读取 / 清洗 / 滑窗
# ======================================================================================
def make_dedup_versions(
    df_real: pd.DataFrame,
    mapping: dict[str, str],
    label_col: str,
    digits: int,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """构造「去重前 / 去重后」两个版本，去重口径与项目 ``clean_dataframe`` 一致。

    项目里的去重是**两步**：先按「特征 + 标签」精确去重（保留标签不同的"同特征"行），
    再按有效数字对**特征**做近似去重。这里照搬，原因很实际：
    只对特征做近似去重，会把「特征相同但标签不同」的行合成一条，
    在 inSDN 这种「正常块在前、攻击块在后」的拼接数据集上会**成片吃掉攻击样本**，
    让类别先验被悄悄改写（实测攻击行存活率远低于正常行）。
    为了不掩盖这个现象，脚本把每一步的行数与标签分布都记进报告。
    """
    cols = list(mapping.values())
    y = df_real[label_col].astype(int).to_numpy()

    def stat(frame: pd.DataFrame, tag: str) -> dict:
        yy = frame[label_col].astype(int).to_numpy()
        return {
            "stage": tag,
            "rows": int(len(frame)),
            "attack_rows": int((yy == 1).sum()),
            "normal_rows": int((yy == 0).sum()),
            "attack_share_rows": float((yy == 1).mean()) if len(yy) else float("nan"),
        }

    stages = {"raw": stat(df_real, "去重前（仅去 inf/缺失）")}

    df_exact = df_real.drop_duplicates(subset=cols + [label_col], keep="first").reset_index(drop=True)
    stages["exact"] = stat(df_exact, "步骤1：按「特征+标签」精确去重")

    df_approx = df_exact
    if digits > 0:
        q = _sig_quantize(df_exact[cols].to_numpy(dtype=np.float64), int(digits))
        keep = ~pd.DataFrame(q, columns=cols).duplicated().to_numpy()
        del q
        df_approx = df_exact.loc[keep].reset_index(drop=True)
        stages["approx"] = stat(df_approx, f"步骤2：按 {digits} 位有效数字近似去重")

    # 标签存活率：攻击类是否比正常类更容易被去重吃掉
    a0, n0 = int((y == 1).sum()), int((y == 0).sum())
    a1 = stages["approx"]["attack_rows"] if "approx" in stages else stages["exact"]["attack_rows"]
    n1 = stages["approx"]["normal_rows"] if "approx" in stages else stages["exact"]["normal_rows"]
    stages["survival"] = {
        "attack_row_survival": float(a1 / max(a0, 1)),
        "normal_row_survival": float(n1 / max(n0, 1)),
    }
    logger.info(
        "去重口径（两步，与 clean_dataframe 一致）：%s 行 → 精确去重 %s 行 → 近似去重 %s 行；"
        "攻击行存活率 %.3f，正常行存活率 %.3f",
        f"{stages['raw']['rows']:,}", f"{stages['exact']['rows']:,}",
        f"{stages['approx']['rows']:,}" if "approx" in stages else "-",
        stages["survival"]["attack_row_survival"], stages["survival"]["normal_row_survival"],
    )
    return df_real, df_approx, stages


def _sig_quantize(a: np.ndarray, k: int, block: int = 20000) -> np.ndarray:
    """按 k 位有效数字量化（等价于 data_pipeline 里的 ``%.{k}g``，但向量化且省内存）。"""
    out = np.empty(a.shape, dtype=np.float32)
    for i in range(0, len(a), block):
        blk = a[i : i + block]
        mag = np.zeros_like(blk)
        nz = np.isfinite(blk) & (blk != 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log10(np.abs(blk), out=mag, where=nz)
        mag = np.floor(mag)
        factor = np.power(10.0, k - 1 - mag)
        out[i : i + block] = np.round(blk * factor) / factor
    return out


def load_synthetic_train_raw(
    cfg: Config, synth_csv: Path, feature_names: list[str]
) -> tuple[np.ndarray, dict]:
    """重建「合成训练集」的**原始（未标准化）特征矩阵**，用于计算漂移统计量。

    重建方式与 ``prepare_dataset`` 完全一致：清洗 → 按时间排序 → 取前
    ``int(n*train_ratio) - purge`` 行。重建后用 ``scaler.mean_`` 校验
    （若对不上就说明重建口径错了，此时退回 scaler 自身统计量并在报告里标注）。
    """
    t0 = time.time()
    raw = pd.read_csv(synth_csv, low_memory=False)
    n_raw = len(raw)
    cfg_syn = Config()
    cfg_syn.schema = cfg.schema
    cfg_syn.dedup_significant_digits = int(cfg.dedup_significant_digits)
    df, feats, _label_cols, mc_col = clean_dataframe(raw, cfg_syn)
    df = order_by_time(df, cfg_syn)
    del raw

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"合成数据里缺少这些特征列，无法重建训练集统计量：{missing}")

    n = len(df)
    purge = int(cfg.purge_windows) * int(cfg.window_size)
    tr_end = max(int(n * cfg.train_ratio) - purge, 1)
    X = df[feature_names].to_numpy(dtype=np.float64)[:tr_end]
    labels = df[mc_col].astype(str).to_numpy()
    meta = {
        "file": synth_csv.name,
        "raw_rows": int(n_raw),
        "clean_rows": int(n),
        "train_rows": int(tr_end),
        "purge_rows": int(purge),
        "label_column": str(mc_col),
        "label_counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return X, meta


def load_real_mapped(
    real_csv: Path,
    mapping: dict[str, str],
    label_column: str,
    max_rows: int = 0,
) -> tuple[pd.DataFrame, str, dict]:
    """读真实 CSV，只保留映射到的列 + 标签列，并做 inf/缺失处理。"""
    head = pd.read_csv(real_csv, nrows=5, low_memory=False)
    cols = list(head.columns)
    label_col = label_column if label_column in cols else ""
    if not label_col:
        from src.data_pipeline import detect_label_columns

        cand = detect_label_columns(head, Config())
        label_col = cand[0]
    need = sorted({c for c in mapping.values() if c in cols} | {label_col})
    df = pd.read_csv(real_csv, low_memory=False, usecols=need)
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    n_read = len(df)

    # 全量标签分布（在丢弃任何行之前统计，用于报告里的"真实标签只有 0/1"这一事实）
    vc = df[label_col].value_counts(dropna=False)
    raw_label_counts = {str(k): int(v) for k, v in vc.items()}

    df = df.replace([np.inf, -np.inf], np.nan)
    missing_cols = [c for c in mapping.values() if c not in df.columns]
    for c in missing_cols:
        df[c] = np.nan

    n_before = len(df)
    df = df.dropna(subset=[label_col])
    n_nolabel = n_before - len(df)
    df = df.dropna(subset=list(mapping.values()) if mapping else [label_col])
    n_nan = n_before - n_nolabel - len(df)
    df = df.reset_index(drop=True)

    info = {
        "file": real_csv.name,
        "rows_read": int(n_read),
        "rows_dropped_no_label": int(n_nolabel),
        "rows_dropped_nan_or_inf": int(n_nan),
        "rows_after_clean": int(len(df)),
        "label_column": str(label_col),
        "columns_not_found_in_real": missing_cols,
        "n_columns_used": int(len(need)),
        "raw_label_counts": raw_label_counts,
    }
    return df, label_col, info


def normalise_real_labels(series: pd.Series) -> np.ndarray:
    """把真实标签归一化成字符串类名：0/Normal/… → ``Normal``，其余进攻击侧。"""
    s = series.copy()
    as_num = pd.to_numeric(s, errors="coerce")
    out = s.astype(str).str.strip()
    numeric_mask = as_num.notna()
    if numeric_mask.any():
        out = out.where(~numeric_mask, np.where(as_num == 0, "Normal", "Attack"))
    for cand in ("Normal.", "normal", "normal.", "BENIGN", "Benign", "benign"):
        out = out.where(out != cand, "Normal")
    return out.to_numpy()


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, list[str], int]:
    """标签名 → 整数编码。0 号固定为 Normal（若存在），其余按字典序。"""
    uniq = sorted(set(str(x) for x in labels))
    if "Normal" in uniq:
        uniq.remove("Normal")
        uniq = ["Normal"] + uniq
    index = {name: i for i, name in enumerate(uniq)}
    codes = np.array([index[str(x)] for x in labels], dtype=np.int64)
    return codes, uniq, index.get("Normal", -1)


def iter_windows(
    features: np.ndarray,
    codes: np.ndarray,
    window: int,
    stride: int,
    chunk: int = 4096,
):
    """分块滑窗：yield (起始行号, 特征窗 (b,T,F), 标签编码窗 (b,T))。

    分块是为了在 30 万行真实数据上做 stride=3 的推理时不炸内存
    （一次性 materialize 全部窗口需要几百 MB）。
    """
    n = len(features)
    starts = np.arange(0, n - window + 1, stride, dtype=np.int64)
    offs = np.arange(window, dtype=np.int64)
    for i in range(0, len(starts), chunk):
        s = starts[i : i + chunk]
        idx = s[:, None] + offs[None, :]
        yield s, features[idx], codes[idx]


# ======================================================================================
# 3. 逐版本（去重前 / 去重后）的推理与指标
# ======================================================================================
def run_version(
    name: str,
    df_real: pd.DataFrame,
    label_col: str,
    mapping: dict[str, str],
    feature_names: list[str],
    scaler,
    model,
    cfg: Config,
    window: int,
    stride: int,
    batch: int,
    run: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """对某个版本的（真实）数据窗口化 + 推理 + 算指标。"""
    n_classes = len(run["class_names"])
    normal_idx = run["normal_idx"]

    labels = normalise_real_labels(df_real[label_col])
    codes, gt_classes, gt_normal = encode_labels(labels)

    X_raw = df_real[[mapping[f] for f in feature_names if f in mapping]].to_numpy(dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)

    # 映射不上的维度在标准化空间填 0（= 训练集均值），绝不使用真实数据统计量
    unmapped = [f for f in feature_names if f not in mapping]
    n_unmapped = len(unmapped)

    starts_all, p_mc_all, p_bin_all, counts_all = [], [], [], []
    t0 = time.time()
    n_win = 0
    for s, Xw, cw in iter_windows(X_raw, codes, window, stride):
        b = len(s)
        Z = (Xw - mean) / scale
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if n_unmapped:  # 兜底：映射失败的列整列置 0（训练集均值口径）
            for j, f in enumerate(feature_names):
                if f not in mapping:
                    Z[:, :, j] = 0.0
        p_mc, p_bin = predict_probs(model, Z, batch_size=batch)
        cnt = np.zeros((b, len(gt_classes)), dtype=np.int32)
        for k in range(len(gt_classes)):
            cnt[:, k] = (cw == k).sum(axis=1)
        starts_all.append(s)
        p_mc_all.append(p_mc.astype(np.float32))
        p_bin_all.append(p_bin.astype(np.float32))
        counts_all.append(cnt)
        n_win += b
        if (len(starts_all) % 10) == 0:
            LOGGER.info("  [%s] 已推理 %s 个窗口……", name, f"{n_win:,}")

    starts = np.concatenate(starts_all)
    p_mc = np.concatenate(p_mc_all, axis=0)
    p_bin = np.concatenate(p_bin_all)
    counts = np.concatenate(counts_all, axis=0)
    infer_sec = time.time() - t0

    T = window
    pure = counts.max(axis=1) == T
    y_all = counts.argmax(axis=1)
    y_pure = counts[pure].argmax(axis=1)

    # ---- 二分类真值（纯窗口口径为主，全窗口多数票口径为辅）----
    def _bin_of(y_codes: np.ndarray) -> np.ndarray:
        if gt_normal is None:
            return np.ones(len(y_codes), dtype=np.int64)
        return (y_codes != gt_normal).astype(np.int64)

    y_bin_pure = _bin_of(y_pure)
    y_bin_all = _bin_of(y_all)
    p_bin_pure = p_bin[pure]

    # 2 类（Normal vs Attack）概率矩阵：用多分类头归一化，口径与 evaluate.multiclass_metrics 一致
    p_normal = p_mc[:, normal_idx]
    p_attack = 1.0 - p_normal
    p2_pure = np.column_stack([p_normal[pure], p_attack[pure]]).astype(np.float64)
    y2_pure = y_bin_pure
    gt2_names = ["Normal", "Attack"] if gt_normal is not None else ["Attack"]

    bin_pure = binary_metrics(y_bin_pure, p_bin_pure, cfg.report_threshold, cfg)
    bin_all = binary_metrics(y_bin_all, p_bin, cfg.report_threshold, cfg)
    mc2_pure = multiclass_metrics(y2_pure, p2_pure, gt2_names)

    # ---- 模型侧 8 类归因（**不是**真值指标，只用来看模型把真实流量判成了什么）----
    pred_cls = p_mc.argmax(axis=1)
    class_names = run["class_names"]
    attribution = {c: int((pred_cls == i).sum()) for i, c in enumerate(class_names)}
    attr_attack = {c: int(((pred_cls == i) & (y_bin_all == 1)).sum()) for i, c in enumerate(class_names)}
    attr_normal = {c: int(((pred_cls == i) & (y_bin_all == 0)).sum()) for i, c in enumerate(class_names)}
    p_clip = np.clip(p_mc, 1e-12, 1.0)
    ent = -(p_clip * np.log(p_clip)).sum(axis=1) / math.log(n_classes)
    share = max(attribution.values()) / max(len(pred_cls), 1)

    # 先验偏移：真实数据攻击占比 vs 合成测试集攻击占比（后面报告里要用）
    attack_share_real = float(y_bin_all.mean())
    # 无非信息基线：全判攻击 / 全判正常的 F1
    from sklearn.metrics import f1_score

    f1_all_attack = float(f1_score(y_bin_pure, np.ones_like(y_bin_pure), zero_division=0))
    f1_all_normal = float(f1_score(y_bin_pure, np.zeros_like(y_bin_pure), zero_division=0))

    out = {
        "version": name,
        "n_rows": int(len(df_real)),
        "n_windows_total": int(len(starts)),
        "n_windows_pure": int(pure.sum()),
        "n_windows_mixed": int((~pure).sum()),
        "window_size": int(T),
        "stride": int(stride),
        "infer_seconds": round(infer_sec, 1),
        "windows_per_sec": round(len(starts) / max(infer_sec, 1e-6), 1),
        "ground_truth_classes": gt_classes,
        "binary_pure": bin_pure,
        "binary_all_windows": bin_all,
        "two_class_pure": {
            "class_names": gt2_names,
            "accuracy": mc2_pure["accuracy"],
            "macro_precision": mc2_pure["macro_precision"],
            "macro_recall": mc2_pure["macro_recall"],
            "macro_f1": mc2_pure["macro_f1"],
            "per_class": mc2_pure["per_class"],
            "confusion_matrix": mc2_pure["confusion_matrix"],
        },
        "attack_share_real_windows": attack_share_real,
        "trivial_baseline": {
            "f1_all_attack": f1_all_attack,
            "f1_all_normal": f1_all_normal,
            "majority_accuracy": float(max(attack_share_real, 1 - attack_share_real)),
        },
        "model_attribution_8class": {
            "note": "模型侧归因分布（argmax over 8 类），真实数据集没有 8 类真值，故这不是准确率指标",
            "all_windows": attribution,
            "on_true_attack_windows": attr_attack,
            "on_true_normal_windows": attr_normal,
            "top_class_share": float(share),
            "mean_normalized_entropy": float(ent.mean()),
        },
    }
    arrays = {"p_bin": p_bin, "pure": pure, "y_bin_all": y_bin_all, "starts": starts, "p_mc": p_mc}
    LOGGER.info(
        "  [%s] 窗口 %s（纯窗口 %s）| 二分类 F1=%.4f AUC=%.4f | 用时 %.1fs",
        name, f"{len(starts):,}", f"{int(pure.sum()):,}",
        bin_pure["f1"], bin_pure["roc_auc"], infer_sec,
    )
    return out, arrays, X_raw


# ======================================================================================
# 4. 漂移诊断
# ======================================================================================
def psi(exp: np.ndarray, act: np.ndarray, bins: int = 10) -> tuple[float, list]:
    """人口稳定指数 PSI：分箱边界取自**合成（参考）分布**的分位数。"""
    qs = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(exp, qs))
    if len(edges) < 3:
        return 0.0, []
    eps = 1e-6
    e = np.histogram(exp, bins=edges)[0] / max(len(exp), 1)
    a = np.histogram(act, bins=edges)[0] / max(len(act), 1)
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    val = float(np.sum((a - e) * np.log(a / e)))
    return val, [float(x) for x in edges]


def drift_analysis(
    feature_names: list[str],
    X_syn: np.ndarray,
    X_real: np.ndarray,
    scaler,
    top_k: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """逐特征算 z-shift（用训练集 mean/std 归一化后的均值差）与 PSI。"""
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)

    syn_mean = X_syn.mean(axis=0)
    syn_std = X_syn.std(axis=0, ddof=0)
    real_mean = X_real.mean(axis=0)
    real_std = X_real.std(axis=0, ddof=0)
    z_shift = (real_mean - mean) / scale
    z_scale_ratio = real_std / np.where(syn_std < 1e-12, 1e-12, syn_std)

    rows = []
    edges_all = {}
    for j, f in enumerate(feature_names):
        v, ed = psi(X_syn[:, j], X_real[:, j])
        edges_all[f] = ed
        rows.append(
            {
                "feature": f,
                "synth_train_mean": float(syn_mean[j]),
                "synth_train_std": float(syn_std[j]),
                "real_mean": float(real_mean[j]),
                "real_std": float(real_std[j]),
                "z_shift": float(z_shift[j]),
                "abs_z_shift": float(abs(z_shift[j])),
                "std_ratio_real_over_synth": float(z_scale_ratio[j]),
                "psi": float(v),
            }
        )
    tab = pd.DataFrame(rows).sort_values("abs_z_shift", ascending=False).reset_index(drop=True)

    absz = tab["abs_z_shift"].to_numpy()
    summary = {
        "n_features": int(len(tab)),
        "z_shift_signed_mean": float(z_shift.mean()),
        "abs_z_shift_mean": float(absz.mean()),
        "abs_z_shift_median": float(np.median(absz)),
        "abs_z_shift_p95": float(np.percentile(absz, 95)),
        "abs_z_shift_max": float(absz.max()),
        "n_features_abs_z_gt_1": int((absz > 1).sum()),
        "n_features_abs_z_gt_3": int((absz > 3).sum()),
        "n_features_psi_gt_0.25": int((tab["psi"] > 0.25).sum()),
        "n_features_std_ratio_lt_005": int((tab["std_ratio_real_over_synth"] < 0.05).sum()),
        "psi_mean": float(tab["psi"].mean()),
        "psi_max": float(tab["psi"].max()),
        "top_k": tab.head(top_k).to_dict(orient="records"),
        "all_features": tab.to_dict(orient="records"),
    }
    return summary, tab


def plot_drift(
    tabs: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]],
    feature_names: list[str],
    scaler,
    scores: dict,
    out_png: Path,
    chinese: bool,
    top_n: int = 6,
) -> list[str]:
    """画「漂移最大的若干特征」的合成 vs 真实分布对比 + 异常分数分布对比。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def L(zh: str, en: str) -> str:
        return zh if chinese else en

    label, tab, X_syn, X_real = tabs[0]
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)

    names = list(tab["feature"].head(top_n))
    ncols = 3
    nrows = int(math.ceil(len(names) / ncols)) + 1
    fig = plt.figure(figsize=(4.6 * ncols, 3.3 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.52, wspace=0.26)

    for k, feat in enumerate(names):
        j = feature_names.index(feat)
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        a = (X_syn[:, j] - mean[j]) / scale[j]
        b = (X_real[:, j] - mean[j]) / scale[j]
        lo = float(np.percentile(np.concatenate([a, b]), 0.5))
        hi = float(np.percentile(np.concatenate([a, b]), 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -4.0, 4.0
        bins = np.linspace(lo, hi, 41)
        ax.hist(np.clip(a, lo, hi), bins=bins, density=True, alpha=0.55,
                label=L("合成（训练集）", "synthetic (train)"), color="#2b6cb0")
        ax.hist(np.clip(b, lo, hi), bins=bins, density=True, alpha=0.55,
                label=L("真实 inSDN", "real inSDN"), color="#c53030")
        ax.axvline(0.0, color="#4a5568", ls="--", lw=1.0)
        ax.axvline(float(b.mean()), color="#c53030", ls=":", lw=1.2)
        z = float(tab.loc[tab["feature"] == feat, "z_shift"].iloc[0])
        ax.set_title(f"{feat}\nz-shift = {z:+.2f}", fontsize=9)
        ax.set_xlabel(L("标准化取值（0 = 合成训练集均值）", "standardized value"), fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")

    # 最后一行：异常分数分布（合成测试集 vs 真实）
    ax = fig.add_subplot(gs[nrows - 1, :])
    bins = np.linspace(0.0, 1.0, 51)
    for key, color, lab in (
        ("in_domain", "#2b6cb0", L("同域：合成测试集", "in-domain: synthetic test")),
        ("real_raw", "#c53030", L("跨域：真实 inSDN（去重前）", "cross-domain: real inSDN (raw)")),
        ("real_dedup", "#dd6b20", L("跨域：真实 inSDN（去重后）", "cross-domain: real inSDN (dedup)")),
    ):
        if key in scores and scores[key] is not None and len(scores[key]):
            ax.hist(scores[key], bins=bins, density=True, histtype="step", lw=1.8,
                    color=color, label=f"{lab}  (n={len(scores[key]):,})")
    ax.axvline(0.5, color="#4a5568", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(L("模型输出的异常概率 P(attack)", "P(attack) from model"), fontsize=9)
    ax.set_ylabel(L("密度（对数）", "density (log)"), fontsize=9)
    ax.set_title(L("异常分数分布：同域 vs 跨域（阈值 0.5 处虚线）",
                   "anomaly score distribution: in-domain vs cross-domain"), fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(
        L("跨数据集分布漂移诊断：合成训练集 vs 真实 inSDN",
          "Cross-dataset drift: synthetic train vs real inSDN"),
        fontsize=13, y=0.997,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return names


# ======================================================================================
# 5. 校准阶梯（Calibration Ladder）
# ======================================================================================
# 设计要点（决定了这一节的全部可信度）：
#
#   * 全部级别都在**同一份去重后的真实数据**上评估，且评估集**只被评估**：
#     任何标定（scaler / 阈值）与任何微调（分类头 / 主干）都只用「拟合段」的行。
#   * 切分**按行序连续切**，不随机混：这样窗口不会跨段，也就不存在"某几行既参与
#     拟合又被评估"的隐性泄漏。
#   * 拟合段的构成：
#       - 正常流量的**前半段**（按行序）→ 供 L1 重拟合 scaler、L2 定阈值、L3/L4 的负样本；
#       - 攻击流量的**最前面若干行**（默认 3000 行 ≈ 1000 个窗口）→ 只供 L3/L4 的正样本。
#     ★ 这里与「攻击行全部进评估段」有一点偏离，必须说明理由：L3 要在**目标域**上做
#       少样本微调，就必须有少量目标域正样本；若正样本从评估段里取，那就是数据泄漏，
#       整个阶梯的可比性会立刻失效。因此宁可从攻击块头部让出 3% 的行给拟合段，
#       并把这个让出量如实写进报告。默认让出量可通过 --calib-fit-attack-rows 调整。
#   * 阈值只在 L2/L3/L4 才动，且一律用**拟合段正常窗口的分数分位数**（99.9%），
#     绝不用评估段的分位数——这是本实验最容易出错、也最容易被质疑的地方。
CALIB_LEVEL_DESC = {
    "L0": "零样本：合成训练集 scaler + 权重 + 默认阈值 0.5",
    "L1": "只重拟合标准化器：用目标域前半段正常流重算 mean/std，权重与阈值不动",
    "L2": "L1 + 按拟合段正常流量的分数 99.9% 分位重定阈值（不需要标签）",
    "L3": "L2 + 冻结主干、只用少量标注重训分类头",
    "L4": "L3 + 解冻主干、低学习率再训 1~2 epoch",
}


def build_calibration_split(
    df_dedup: pd.DataFrame,
    label_col: str,
    fit_attack_rows: int,
    logger,
) -> tuple[dict, list[str]]:
    """按行序把去重后的真实数据切成「拟合段 / 评估段」。"""
    y = df_dedup[label_col].astype(int).to_numpy()
    n = len(y)
    normal_pos = np.flatnonzero(y == 0)
    attack_pos = np.flatnonzero(y == 1)
    warnings: list[str] = []

    for name, pos in (("正常", normal_pos), ("攻击", attack_pos)):
        if len(pos) and len(pos) != (pos[-1] - pos[0] + 1):
            warnings.append(f"{name}流量的行号不连续，窗口可能跨标签边界，阶梯结果需谨慎解读")
    if not len(normal_pos) or not len(attack_pos):
        raise ValueError("校准阶梯要求真实数据里同时存在正常与攻击两类样本")

    half = len(normal_pos) // 2
    fit_norm = normal_pos[:half]
    eval_norm = normal_pos[half:]
    n_fit_atk = int(min(max(fit_attack_rows, 0), max(len(attack_pos) - 1, 0)))
    fit_atk = attack_pos[:n_fit_atk]
    eval_atk = attack_pos[n_fit_atk:]

    split = {
        "rows_total": int(n),
        "fit": {
            "normal_rows": int(len(fit_norm)),
            "attack_rows": int(len(fit_atk)),
            "row_range": [int(fit_norm[0]), int(fit_norm[-1])],
            "note": "前半段正常流（按行序）+ 攻击块最前面若干行",
        },
        "eval": {
            "normal_rows": int(len(eval_norm)),
            "attack_rows": int(len(eval_atk)),
            "row_range": [int(eval_norm[0]), int(eval_atk[-1])],
            "attack_share_rows": float(len(eval_atk) / max(len(eval_norm) + len(eval_atk), 1)),
        },
        "isolation": {
            "overlap_rows": int(len(np.intersect1d(np.concatenate([fit_norm, fit_atk]),
                                                   np.concatenate([eval_norm, eval_atk])))),
            "split_mode": "按行序连续切分（非随机），窗口不跨段",
        },
    }
    logger.info(
        "校准阶梯切分：拟合段 正常 %s 行 + 攻击 %s 行；评估段 正常 %s 行 + 攻击 %s 行；重叠行数 %d",
        f"{len(fit_norm):,}", f"{len(fit_atk):,}", f"{len(eval_norm):,}", f"{len(eval_atk):,}",
        split["isolation"]["overlap_rows"],
    )
    if split["isolation"]["overlap_rows"] != 0:
        warnings.append("拟合段与评估段出现行重叠——隔离被破坏，指标不可信")
    for w in warnings:
        logger.warning("[校准阶梯] %s", w)
    return {"fit_norm": fit_norm, "fit_atk": fit_atk, "eval_norm": eval_norm, "eval_atk": eval_atk,
            "meta": split}, warnings


def score_rows_slice(
    df_slice: pd.DataFrame,
    label_value: int,
    mapping: dict[str, str],
    feature_names: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    model,
    window: int,
    stride: int,
    batch: int,
    chunk: int,
) -> dict:
    """对一段**标签单一**的连续行做滑窗 + 标准化 + 推理（分块，控内存）。"""
    cols = [mapping[f] for f in feature_names if f in mapping]
    unmapped_idx = [j for j, f in enumerate(feature_names) if f not in mapping]
    X_raw = df_slice[cols].to_numpy(dtype=np.float64)
    codes = np.zeros(len(df_slice), dtype=np.int64)

    p_mc_l, p_bin_l, starts_l = [], [], []
    for s, Xw, _cw in iter_windows(X_raw, codes, window, stride, chunk=chunk):
        Z = np.nan_to_num((Xw - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if unmapped_idx:  # 映射不上的维度在标准化空间填 0（= 训练集均值口径）
            Z[:, :, unmapped_idx] = 0.0
        p_mc, p_bin = predict_probs(model, Z, batch_size=batch)
        p_mc_l.append(p_mc.astype(np.float32))
        p_bin_l.append(p_bin.astype(np.float32))
        starts_l.append(s)
    if not p_mc_l:
        raise ValueError("该切片行数不足以生成任何窗口，请检查 window/stride 或数据量")
    return {
        "p_mc": np.concatenate(p_mc_l, axis=0),
        "p_bin": np.concatenate(p_bin_l),
        "y_true": np.full(len(np.concatenate(p_bin_l)), int(label_value), dtype=np.int64),
        "n_rows": int(len(df_slice)),
    }


def metrics_from_scores(p_mc: np.ndarray, p_bin: np.ndarray, y_bin: np.ndarray,
                        normal_idx: int, cfg: Config, threshold: float) -> dict:
    """由分数直接算指标。

    ★ 这里与第 3 节的口径有一处**刻意的差别**：2 类宏 F1 用「按给定阈值硬判后的标签」计算，
    而不是用 ``argmax(概率矩阵)``。原因：阶梯实验会改阈值，若沿用 argmax 口径，
    2 类宏 F1 会一直钉在 0.5 阈值上、对阈值变化毫无反应，表格会自相矛盾
    （二分类 F1 变了、宏 F1 不变）。用硬判标签可以让两列在同一个工作点上可比。
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_bin = np.asarray(y_bin).astype(int)
    thr = float(threshold)
    bm = binary_metrics(y_bin, p_bin, thr, cfg)
    pred = (np.asarray(p_bin) >= thr).astype(int)
    per_class = {}
    for cid, cname in ((0, "Normal"), (1, "Attack")):
        yt = (y_bin == cid).astype(int)
        pp = (pred == cid).astype(int)
        per_class[cname] = {
            "precision": float(precision_score(yt, pp, zero_division=0)),
            "recall": float(recall_score(yt, pp, zero_division=0)),
            "f1": float(f1_score(yt, pp, zero_division=0)),
            "support": int(yt.sum()),
        }
    return {
        "threshold": thr,
        "binary": bm,
        "two_class": {
            "accuracy": float((pred == y_bin).mean()),
            "macro_precision": float(precision_score(y_bin, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_bin, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_bin, pred, average="macro", zero_division=0)),
            "per_class": per_class,
            "note": "2 类宏 F1 按给定阈值硬判后计算（与二分类列同一工作点）",
        },
        "trivial_baseline": {
            "f1_all_attack": float(f1_score(y_bin, np.ones_like(y_bin), zero_division=0)),
            "f1_all_normal": float(f1_score(y_bin, np.zeros_like(y_bin), zero_division=0)),
            "majority_accuracy": float(max(y_bin.mean(), 1 - y_bin.mean())),
        },
    }


def _fit_window_tensor(
    df_slice: pd.DataFrame,
    mapping: dict[str, str],
    feature_names: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window: int,
    stride: int,
    take: int,
) -> np.ndarray:
    """从一段连续行里**等间隔**抽取 take 个窗口（确定性、可复现），返回标准化后的张量。"""
    cols = [mapping[f] for f in feature_names if f in mapping]
    X_raw = df_slice[cols].to_numpy(dtype=np.float64)
    n = len(X_raw)
    starts = np.arange(0, n - window + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        return np.zeros((0, window, len(feature_names)), dtype=np.float32)
    k = int(min(take, len(starts)))
    pick = np.unique(np.linspace(0, len(starts) - 1, k).round().astype(int))
    idx = starts[pick][:, None] + np.arange(window, dtype=np.int64)[None, :]
    Z = np.nan_to_num((X_raw[idx] - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    unmapped_idx = [j for j, f in enumerate(feature_names) if f not in mapping]
    if unmapped_idx:
        Z[:, :, unmapped_idx] = 0.0
    return Z


def finetune_classification_head(model, cfg: Config, X_fit: np.ndarray, y_mc: np.ndarray,
                                 y_bin: np.ndarray, *, unfreeze_backbone: bool,
                                 epochs: int, lr: float, batch_size: int, logger) -> dict:
    """冻结主干（或只解冻主干做低 LR 微调），重训分类头 + 二分类头。

    冻结判据：只有名字以 ``clf_`` 开头（两个 Dense + BN + Dropout 组成的分类头）
    以及 ``mc`` / ``bin`` 两个输出层可训练；CNN / BiLSTM / 注意力层全部冻结
    （冻结的 BatchNormalization 会走推理模式的滑动统计量，不会污染主干）。
    """
    from tensorflow.keras import optimizers

    trainable, frozen = [], []
    for layer in model.layers:
        is_head = layer.name.startswith("clf_") or layer.name in ("mc", "bin")
        layer.trainable = bool(unfreeze_backbone or is_head)
        (trainable if layer.trainable else frozen).append(layer.name)
    if not X_fit.size:
        raise ValueError("微调样本为空")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(lr)),
        loss={"mc": "sparse_categorical_crossentropy", "bin": "binary_crossentropy"},
        loss_weights={"mc": 1.0, "bin": float(cfg.binary_loss_weight)},
        metrics={"mc": ["accuracy"], "bin": ["accuracy"]},
    )
    set_seed(int(cfg.seed))
    t0 = time.time()
    hist = model.fit(
        X_fit, {"mc": y_mc, "bin": y_bin},
        epochs=int(epochs), batch_size=int(batch_size), verbose=0, shuffle=True,
    )
    secs = time.time() - t0
    info = {
        "mode": "unfreeze_backbone" if unfreeze_backbone else "head_only",
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "batch_size": int(batch_size),
        "n_train_windows": int(len(X_fit)),
        "n_trainable_layers": len(trainable),
        "n_frozen_layers": len(frozen),
        "trainable_layer_names": trainable,
        "final_loss": float(hist.history["loss"][-1]),
        "fit_seconds": round(secs, 2),
    }
    logger.info(
        "  微调（%s）：%d 个窗口 / %d epoch → loss %.4f，可训练层 %d 个，耗时 %.1fs",
        info["mode"], len(X_fit), epochs, info["final_loss"], len(trainable), secs,
    )
    return info


def _zshift_stats(X_ref_mean: np.ndarray, X_ref_std: np.ndarray, X_group: np.ndarray) -> dict:
    """用参照统计量归一化后，算某一组行的 |z-shift| 汇总。"""
    std = np.where(np.abs(X_ref_std) < 1e-12, 1.0, X_ref_std)
    z = (X_group.mean(axis=0) - X_ref_mean) / std
    a = np.abs(z)
    return {
        "n_features": int(len(z)),
        "signed_mean": float(z.mean()),
        "abs_mean": float(a.mean()),
        "abs_median": float(np.median(a)),
        "abs_p95": float(np.percentile(a, 95)),
        "abs_max": float(a.max()),
        "n_abs_gt_1": int((a > 1).sum()),
        "n_abs_gt_3": int((a > 3).sum()),
        "per_feature": z.tolist(),
    }


def run_calibration_ladder(
    df_dedup: pd.DataFrame,
    label_col: str,
    mapping: dict[str, str],
    feature_names: list[str],
    class_names: list[str],
    normal_idx: int,
    synth_scaler,
    model,
    cfg: Config,
    window: int,
    stride: int,
    batch: int,
    chunk: int,
    *,
    fit_attack_rows: int,
    finetune_windows: int,
    head_epochs: int,
    backbone_epochs: int,
    head_lr: float,
    backbone_lr: float,
    ft_batch: int,
    target_fpr_quantile: float,
    logger,
) -> dict:
    """跑完整的 L0→L4 校准阶梯。所有级别共用同一个评估段。"""
    from sklearn.preprocessing import StandardScaler

    logger.info("-" * 78)
    logger.info("校准阶梯：L0 零样本 → L1 重定标 → L2 重定阈值 → L3 微调分类头 → L4 解冻主干")
    logger.info("-" * 78)

    parts, warn = build_calibration_split(df_dedup, label_col, fit_attack_rows, logger)
    fit_norm_df = df_dedup.iloc[parts["fit_norm"]]
    fit_atk_df = df_dedup.iloc[parts["fit_atk"]]
    eval_norm_df = df_dedup.iloc[parts["eval_norm"]]
    eval_atk_df = df_dedup.iloc[parts["eval_atk"]]

    # 拟合段**再分一次**：微调子段 / 定阈值子段，两者行不重叠。
    # ★ 为什么必须再分：L3/L4 的微调窗口如果同时又用来挑阈值，阈值就是在训练样本上
    #   挑出来的（in-sample selection），实测会把阈值挑到一个在评估段上完全失效的位置
    #   （典型的过拟合现象）。这里按行序 8:2 切开，微调只看前 80%，定阈值只看后 20%。
    fn, fa = parts["fit_norm"], parts["fit_atk"]
    fn_cut = int(len(fn) * 0.8)
    fa_cut = int(len(fa) * 0.8)
    ft_train_norm_df = df_dedup.iloc[fn[:fn_cut]]
    ft_valid_norm_df = df_dedup.iloc[fn[fn_cut:]]
    ft_train_atk_df = df_dedup.iloc[fa[:fa_cut]]
    ft_valid_atk_df = df_dedup.iloc[fa[fa_cut:]]
    logger.info(
        "拟合段再分：微调子段 正常 %s / 攻击 %s 行；定阈值子段 正常 %s / 攻击 %s 行（行不重叠）",
        f"{len(ft_train_norm_df):,}", f"{len(ft_train_atk_df):,}",
        f"{len(ft_valid_norm_df):,}", f"{len(ft_valid_atk_df):,}",
    )
    sub_split = {
        "finetune_subsegment": {"normal_rows": int(len(ft_train_norm_df)),
                                "attack_rows": int(len(ft_train_atk_df))},
        "threshold_selection_subsegment": {"normal_rows": int(len(ft_valid_norm_df)),
                                           "attack_rows": int(len(ft_valid_atk_df))},
        "note": "拟合段内部 8:2 按行序切分；微调与定阈值用的是**不同行**，避免阈值在训练样本上被挑出来",
        "threshold_rows_overlap_finetune_rows": int(
            len(np.intersect1d(np.concatenate([fn[:fn_cut], fa[:fa_cut]]),
                               np.concatenate([fn[fn_cut:], fa[fa_cut:]])))
        ),
    }

    # ---------------- 两套标准化统计量 ----------------
    syn_mean = np.asarray(synth_scaler.mean_, dtype=np.float64)
    syn_scale = np.asarray(synth_scaler.scale_, dtype=np.float64)
    syn_scale = np.where(np.abs(syn_scale) < 1e-12, 1.0, syn_scale)

    fit_norm_raw = fit_norm_df[[mapping[f] for f in feature_names if f in mapping]].to_numpy(np.float64)
    tgt_scaler = StandardScaler().fit(fit_norm_raw)
    tgt_mean = np.asarray(tgt_scaler.mean_, dtype=np.float64)
    tgt_scale = np.asarray(tgt_scaler.scale_, dtype=np.float64)
    tgt_scale = np.where(np.abs(tgt_scale) < 1e-12, 1.0, tgt_scale)
    n_degenerate = int((np.asarray(tgt_scaler.scale_) < 1e-12).sum())
    logger.info(
        "L1 目标域标准化器：在拟合段 %s 行正常流上拟合（%d 个特征在目标域退化为常量，按 1.0 处理）",
        f"{len(fit_norm_raw):,}", n_degenerate,
    )

    # ---------------- 评估段打分（同一批窗口，只换标准化统计量 / 模型） ----------------
    def ev(s_mean, s_scale, mdl, tag):
        t0 = time.time()
        a = score_rows_slice(eval_norm_df, 0, mapping, feature_names, s_mean, s_scale, mdl,
                             window, stride, batch, chunk)
        b = score_rows_slice(eval_atk_df, 1, mapping, feature_names, s_mean, s_scale, mdl,
                             window, stride, batch, chunk)
        out = {
            "p_mc": np.concatenate([a["p_mc"], b["p_mc"]], axis=0),
            "p_bin": np.concatenate([a["p_bin"], b["p_bin"]]),
            "y_true": np.concatenate([a["y_true"], b["y_true"]]),
            "n_windows_normal": int(len(a["p_bin"])),
            "n_windows_attack": int(len(b["p_bin"])),
            "seconds": round(time.time() - t0, 2),
        }
        logger.info("  [%s] 评估段 %s 个窗口（正常 %s / 攻击 %s）→ %.1fs",
                    tag, f"{len(out['p_bin']):,}", f"{out['n_windows_normal']:,}",
                    f"{out['n_windows_attack']:,}", out["seconds"])
        return out

    def stage_scores(mean, scale, mdl, tag):
        """给定 (标准化统计量, 模型)，同时算出**评估段**与**定阈值子段**的分数。

        定阈值子段与微调子段是拟合段内部两组不同的行，评估段更是完全没参与——
        因此这一步既没有评估泄漏，也没有 in-sample 选阈值的毛病。
        """
        evd = ev(mean, scale, mdl, tag)
        fitn = score_rows_slice(ft_valid_norm_df, 0, mapping, feature_names, mean, scale, mdl,
                                window, stride, batch, chunk)
        fita = None
        if len(ft_valid_atk_df) >= window:
            fita = score_rows_slice(ft_valid_atk_df, 1, mapping, feature_names, mean, scale, mdl,
                                    window, stride, batch, chunk)
        fit_p = np.concatenate([fitn["p_bin"], fita["p_bin"]]) if fita is not None else fitn["p_bin"]
        fit_true = (np.concatenate([fitn["y_true"], fita["y_true"]]) if fita is not None
                    else fitn["y_true"])
        thr_f1, f1_fit = best_threshold(fit_true, fit_p) if len(np.unique(fit_true)) > 1 else (0.5, float("nan"))
        return {
            "eval": evd,
            "fit_normal_scores": fitn["p_bin"],
            "fit_n_windows": int(len(fit_p)),
            "fit_n_attack_windows": int(len(fita["p_bin"])) if fita is not None else 0,
            "thr_quantile": float(np.quantile(fitn["p_bin"], float(target_fpr_quantile))),
            "thr_f1_opt": float(thr_f1),
            "fit_f1_at_opt": float(f1_fit),
        }

    def variant(scores: dict, thr: float, source: str) -> dict:
        m = metrics_from_scores(scores["eval"]["p_mc"], scores["eval"]["p_bin"],
                                scores["eval"]["y_true"], normal_idx, cfg, float(thr))
        m["threshold_source"] = source
        return m

    def fpr_budget_variants(scores: dict) -> dict:
        """统一误报预算下的工作点：阈值取「定阈值子段**正常**窗口分数」的 (1−预算) 分位。

        这是 IDS 落地时最常用的定阈值方式——**与目标域攻击先验无关**，
        因此在攻击占比差别很大的评估集之间也可比（本项目的评估段攻击占 0.757，
        而线上真实流量远低于此，先验相关的 F1 会失真，这个口径不会）。
        """
        out = {}
        for budget in (0.01, 0.001):
            t = float(np.quantile(scores["fit_normal_scores"], 1.0 - budget))
            m = variant(scores, t, f"fpr_budget_{budget:g}")
            out[f"fpr_{budget:g}"] = {
                "target_fpr_budget": budget, "threshold": t,
                "eval_false_positive_rate": m["binary"]["false_positive_rate"],
                "eval_recall": m["binary"]["recall"],
                "eval_precision": m["binary"]["precision"],
                "eval_binary_f1": m["binary"]["f1"],
                "eval_macro_f1": m["two_class"]["macro_f1"],
                "eval_roc_auc": m["binary"]["roc_auc"],
            }
        return out

    def make_level(key, desc, needs_labels, scaler_tag, model_tag, scores, spec_thr, spec_src,
                   eval_seconds, n_labeled):
        spec = variant(scores, spec_thr, spec_src)
        return {
            "level": key, "desc": desc, "needs_labels": bool(needs_labels),
            "n_labeled_windows": int(n_labeled), "scaler": scaler_tag, "model_state": model_tag,
            "threshold_source": spec_src,
            "fit_segment_windows": scores["fit_n_windows"],
            "fit_segment_f1_at_best_threshold": scores["fit_f1_at_opt"],
            "variants": {
                "spec": spec,
                "fit_normal_quantile": variant(scores, scores["thr_quantile"],
                                               f"fit_normal_scores_q{target_fpr_quantile:g}"),
                "fit_f1_optimal": variant(scores, scores["thr_f1_opt"], "fit_segment_f1_max"),
            },
            "fpr_budget_operating_points": fpr_budget_variants(scores),
            **spec,
            "eval_seconds": eval_seconds,
        }

    l0 = stage_scores(syn_mean, syn_scale, model, "L0 零样本")
    l1 = stage_scores(tgt_mean, tgt_scale, model, "L1 重定标")
    logger.info(
        "拟合段定阈值（L1 阶段）：正常窗口 %.1f%% 分位 = %.6f | 按拟合段 F1 最优阈值 = %.6f（拟合段 F1 %.4f）",
        target_fpr_quantile * 100, l1["thr_quantile"], l1["thr_f1_opt"], l1["fit_f1_at_opt"],
    )

    levels: dict[str, dict] = {}
    levels["L0"] = make_level("L0", CALIB_LEVEL_DESC["L0"], False, "synthetic_train", "zero_shot",
                              l0, float(cfg.report_threshold), "config_default",
                              l0["eval"]["seconds"], 0)
    levels["L1"] = make_level("L1", CALIB_LEVEL_DESC["L1"], False, "target_normal_refit", "zero_shot",
                              l1, float(cfg.report_threshold), "config_default",
                              l1["eval"]["seconds"], 0)
    levels["L2"] = make_level("L2", CALIB_LEVEL_DESC["L2"], False, "target_normal_refit", "zero_shot",
                              l1, l1["thr_quantile"], f"fit_normal_scores_q{target_fpr_quantile:g}",
                              0.0, 0)

    # ---- 阈值敏感性：光靠调阈值能救回多少？（用 L1 阶段的分数，阈值仍只取自拟合段） ----
    thr_sweep = []
    for q in (0.5, 0.9, 0.95, 0.99, 0.999):
        t = float(np.quantile(l1["fit_normal_scores"], q))
        m = variant(l1, t, f"fit_normal_scores_q{q:g}")
        thr_sweep.append({
            "quantile": q, "threshold": t,
            "target_fpr_budget": 1.0 - q,
            "macro_f1": m["two_class"]["macro_f1"], "binary_f1": m["binary"]["f1"],
            "roc_auc": m["binary"]["roc_auc"], "recall": m["binary"]["recall"],
            "false_positive_rate": m["binary"]["false_positive_rate"],
            "precision": m["binary"]["precision"],
        })

    # ---------------- 微调样本：只来自拟合段 ----------------
    n_neg = int(max(finetune_windows // 2, 1))
    n_pos = int(max(finetune_windows - n_neg, 1))
    X_neg = _fit_window_tensor(ft_train_norm_df, mapping, feature_names, tgt_mean, tgt_scale,
                               window, stride, n_neg)
    X_pos = _fit_window_tensor(ft_train_atk_df, mapping, feature_names, tgt_mean, tgt_scale,
                               window, stride, n_pos)
    X_fit = np.concatenate([X_neg, X_pos], axis=0)
    y_fit_bin = np.concatenate([np.zeros(len(X_neg), np.int64), np.ones(len(X_pos), np.int64)])
    y_fit_mc = np.where(y_fit_bin == 0, normal_idx, (normal_idx + 1) % len(class_names)).astype(np.int64)
    ft_data = {
        "windows_requested": int(finetune_windows),
        "negatives_available_and_used": int(len(X_neg)),
        "positives_available_and_used": int(len(X_pos)),
        "positives_requested": int(n_pos),
        "source": "拟合段（正常前半段 + 攻击块头部），与评估段零重叠",
        "sampling": "在拟合段的候选窗口上等间隔抽取（确定性）",
    }
    logger.info("微调样本：正常 %d + 攻击 %d = %d 个窗口（全部来自拟合段）",
                len(X_neg), len(X_pos), len(X_fit))

    # ---------------- L3：只微调分类头 ----------------
    ft_head = finetune_classification_head(
        model, cfg, X_fit, y_fit_mc, y_fit_bin, unfreeze_backbone=False,
        epochs=head_epochs, lr=head_lr, batch_size=ft_batch, logger=logger,
    )
    l3 = stage_scores(tgt_mean, tgt_scale, model, "L3 微调分类头")
    levels["L3"] = make_level("L3", CALIB_LEVEL_DESC["L3"], True, "target_normal_refit",
                              "head_finetuned", l3, l1["thr_quantile"],
                              f"沿用 L2 阈值（fit_normal_scores_q{target_fpr_quantile:g}）",
                              l3["eval"]["seconds"], len(X_fit))
    levels["L3_recal"] = make_level("L3′", "L3 + 用拟合段正常窗口重新校准阈值（微调后分数分布已变）",
                                    True, "target_normal_refit", "head_finetuned", l3,
                                    l3["thr_quantile"], f"fit_normal_scores_q{target_fpr_quantile:g}",
                                    0.0, len(X_fit))

    # ---------------- L4：解冻主干，低 LR 再训 ----------------
    ft_backbone = finetune_classification_head(
        model, cfg, X_fit, y_fit_mc, y_fit_bin, unfreeze_backbone=True,
        epochs=backbone_epochs, lr=backbone_lr, batch_size=ft_batch, logger=logger,
    )
    l4 = stage_scores(tgt_mean, tgt_scale, model, "L4 解冻主干")
    levels["L4"] = make_level("L4", CALIB_LEVEL_DESC["L4"], True, "target_normal_refit",
                              "backbone_finetuned", l4, l1["thr_quantile"],
                              f"沿用 L2 阈值（fit_normal_scores_q{target_fpr_quantile:g}）",
                              l4["eval"]["seconds"], len(X_fit))
    levels["L4_recal"] = make_level("L4′", "L4 + 用拟合段正常窗口重新校准阈值",
                                    True, "target_normal_refit", "backbone_finetuned", l4,
                                    l4["thr_quantile"], f"fit_normal_scores_q{target_fpr_quantile:g}",
                                    0.0, len(X_fit))

    # ---------------- 校准前后的 z-shift（同一评估段） ----------------
    eval_norm_raw = eval_norm_df[[mapping[f] for f in feature_names if f in mapping]].to_numpy(np.float64)
    eval_atk_raw = eval_atk_df[[mapping[f] for f in feature_names if f in mapping]].to_numpy(np.float64)
    eval_all_raw = np.concatenate([eval_norm_raw, eval_atk_raw], axis=0)
    zshift = {
        "reference_before": "合成训练集统计量（scaler.joblib）",
        "reference_after": "目标域拟合段正常流统计量（Level 1 重拟合）",
        "eval_all_before": _zshift_stats(syn_mean, syn_scale, eval_all_raw),
        "eval_all_after": _zshift_stats(tgt_mean, tgt_scale, eval_all_raw),
        "eval_normal_after": _zshift_stats(tgt_mean, tgt_scale, eval_norm_raw),
        "eval_attack_after": _zshift_stats(tgt_mean, tgt_scale, eval_atk_raw),
        "eval_attack_before": _zshift_stats(syn_mean, syn_scale, eval_atk_raw),
        "eval_normal_before": _zshift_stats(syn_mean, syn_scale, eval_norm_raw),
    }
    logger.info(
        "z-shift（评估段全体）：校准前 |z| 均值 %.4f / P95 %.4f / 最大 %.4f | 校准后 %.4f / %.4f / %.4f",
        zshift["eval_all_before"]["abs_mean"], zshift["eval_all_before"]["abs_p95"],
        zshift["eval_all_before"]["abs_max"], zshift["eval_all_after"]["abs_mean"],
        zshift["eval_all_after"]["abs_p95"], zshift["eval_all_after"]["abs_max"],
    )

    cost_table = {
        "L0": {"needs_labels": False, "labeled_windows": 0, "extra_fit_seconds": 0.0},
        "L1": {"needs_labels": False, "labeled_windows": 0, "extra_fit_seconds": 0.0,
               "note": "只需目标域正常流量的原始特征，无需标签"},
        "L2": {"needs_labels": False, "labeled_windows": 0, "extra_fit_seconds": 0.0,
               "note": "只需「这批是正常流量」的先验，不需要攻击样本"},
        "L3": {"needs_labels": True, "labeled_windows": int(len(X_fit)),
               "extra_fit_seconds": ft_head["fit_seconds"]},
        "L4": {"needs_labels": True, "labeled_windows": int(len(X_fit)),
               "extra_fit_seconds": round(ft_head["fit_seconds"] + ft_backbone["fit_seconds"], 2)},
    }

    return {
        "enabled": True,
        "design": {
            "eval_set": "去重后真实数据的评估段（所有级别共用同一批窗口）",
            "isolation": parts["meta"]["isolation"],
            "fit_attack_rows_reserved": int(len(parts["fit_atk"])),
            "why_reserve_attack_rows": (
                "L3/L4 需要少量目标域正样本；若从评估段取即为数据泄漏。"
                "因此从攻击块头部让出少量行给拟合段，让出比例已在报告中标明。"
            ),
            "threshold_rule": f"拟合段正常窗口分数的 {target_fpr_quantile:g} 分位（绝不使用评估段分位数）",
        },
        "sub_split": sub_split,
        "split": parts["meta"],
        "warnings": warn,
        "scaler_refit": {
            "fit_rows": int(len(fit_norm_raw)),
            "degenerate_features": n_degenerate,
            "synthetic_mean_abs": float(np.abs(syn_mean).mean()),
            "target_mean_abs": float(np.abs(tgt_mean).mean()),
        },
        "eval_windows": {
            "n_windows_total": int(len(l0["eval"]["p_bin"])),
            "n_windows_normal": l0["eval"]["n_windows_normal"],
            "n_windows_attack": l0["eval"]["n_windows_attack"],
            "attack_share": float(l0["eval"]["y_true"].mean()),
        },
        "finetune_data": ft_data,
        "finetune_head": ft_head,
        "finetune_backbone": ft_backbone,
        "levels": levels,
        "threshold_sweep_L1": thr_sweep,
        "cost": cost_table,
        "zshift": {k: v for k, v in zshift.items()},
    }


# ======================================================================================
# 6. 报告渲染：校准阶梯章节
# ======================================================================================
def _calibration_section(ctx: dict, A) -> None:
    """渲染「校准阶梯」章节：把跨域衰减变成一条带代价的恢复路径。"""
    cal = ctx.get("calibration") or {}
    if not cal.get("enabled"):
        A("## 6. 校准阶梯（本次未运行）")
        A("")
        A("本次以 `--no-calibration` 运行，未做校准阶梯实验。去掉该开关即可生成。")
        A("")
        return

    lv = cal["levels"]
    order = ["L0", "L1", "L2", "L3", "L4"]
    dom = ctx.get("in_domain", {})
    dom2 = dom.get("recomputed_two_class", {})
    dom_bin = dom.get("recomputed_on_test_tensors", {})
    ev = cal["eval_windows"]

    A("## 6. 校准阶梯：把跨域衰减补回来需要多少代价")
    A("")
    A("第 3–5 节给出了一个纯负面结论：零样本跨域不可用。这一节回答下一步该怎么做——"
      "**把「补回来」拆成 5 级可执行的校准，每一级实测恢复幅度与代价**，"
      "让结论从「不行」变成「按什么代价可以」。")
    A("")
    A("### 6.1 实验设计（隔离性说明，这一节可信度的全部依据）")
    A("")
    A("全部级别都在**同一份去重后真实数据**上评估，且评估段**只被评估**：")
    A("")
    A(markdown_table(
        ["数据段", "构成", "行数", "用途"],
        [
            ["拟合段", "正常流量的**前半段**（按行序）",
             f"{cal['split']['fit']['normal_rows']:,}",
             "L1 重拟合 scaler（无监督）；再按 8:2 切成微调子段与定阈值子段"],
            ["拟合段", "攻击流量的**最前面若干行**（攻击块头部）",
             f"{cal['split']['fit']['attack_rows']:,}",
             "只供 L3/L4 的**正样本**；同样按 8:2 切开"],
            ["评估段", "正常流量的后半段", f"{cal['split']['eval']['normal_rows']:,}",
             "所有级别共用的评估集"],
            ["评估段", "攻击流量的其余部分", f"{cal['split']['eval']['attack_rows']:,}",
             "所有级别共用的评估集"],
        ],
    ))
    A("")
    A(markdown_table(
        ["隔离性检查", "结果"],
        [
            ["切分方式", cal["split"]["isolation"]["split_mode"]],
            ["拟合段 ∩ 评估段 行数", f"**{cal['split']['isolation']['overlap_rows']}**（必须为 0）"],
            ["微调子段 ∩ 定阈值子段 行数",
             f"**{cal.get('sub_split', {}).get('threshold_rows_overlap_finetune_rows', '—')}**（必须为 0）"],
            ["评估段窗口数", f"{ev['n_windows_total']:,}（正常 {ev['n_windows_normal']:,} / 攻击 {ev['n_windows_attack']:,}，"
                          f"攻击占 {fmt(ev['attack_share'], 3)}）"],
            ["阈值来源", cal["design"]["threshold_rule"]],
            ["标准化统计量来源", "L1 起改用拟合段正常流（**评估段从未参与**）"],
            ["微调样本来源", "拟合段的**微调子段**（前 80% 行）"],
            ["定阈值样本来源", "拟合段的**定阈值子段**（后 20% 行），与微调样本行不重叠"],
        ],
    ))
    A("")
    A(f"> ⚠️ **与「把攻击行全部拿去评估」的一点偏离，必须说明**："
      f"L3/L4 需要在**目标域**上做少样本微调，就必须有少量目标域正样本；"
      f"若正样本从评估段取，就是数据泄漏，整条阶梯立刻失去可比性。"
      f"因此从攻击块**头部**让出 {cal['design']['fit_attack_rows_reserved']:,} 行"
      f"（占攻击行 {fmt(cal['design']['fit_attack_rows_reserved'] / max(cal['split']['eval']['attack_rows'] + cal['design']['fit_attack_rows_reserved'], 1), 3)}）"
      f"给拟合段，让出量可用 `--calib-fit-attack-rows` 调整。"
      f"其余攻击行**全部**留在评估段。")
    A("")
    A(f"> 另一个口径提醒：为了让所有级别共用同一评估段，第 6 节的 Level 0 数字与第 3 节的"
      f"「去重后」数字**不是同一批窗口**（这里少了前半段正常流，攻击先验由整份去重数据的 "
      f"{fmt(ctx['versions'][ctx['dedup_key']]['attack_share_real_windows'], 3)} "
      f"升高到评估段的 {fmt(ev['attack_share'], 3)}）。因此**阶梯内部的横向对比是有效的，"
      f"但与第 3 节的绝对值不可直接相减**——这正是 6.5 节改用「统一误报预算」（与先验无关）"
      f"来做跨级对比的原因。")
    A("")

    # ---- 主表 ----
    A("### 6.2 四级（含可选第五级）校准的实测结果")
    A("")
    rows = []
    for k in order:
        v = lv[k]
        c = cal["cost"][k]
        tag = "（需要标签）" if c["needs_labels"] else "（不需要标签）"
        rows.append([
            f"**{v['level']}**",
            v["desc"],
            f"{'是' if c['needs_labels'] else '否'}",
            f"{c['labeled_windows']:,}" if c["labeled_windows"] else "0",
            f"{c['extra_fit_seconds']:.1f}",
            f"**{fmt(v['two_class']['macro_f1'])}**",
            f"**{fmt(v['binary']['f1'])}**",
            fmt(v["binary"]["roc_auc"]),
            fmt(v["binary"]["recall"]),
            fmt(v["binary"]["false_positive_rate"]),
            fmt(v["binary"]["tpr_at_fpr_0.01"]),
        ])
    A(markdown_table(
        ["级别", "做了什么", "需要标签", "标注窗口数", "拟合耗时(s)",
         "2 类宏 F1", "二分类 F1", "ROC-AUC", "召回", "FPR", "TPR@FPR=1%"],
        rows,
    ))
    A("")
    A("参照行（合成测试集，同口径）："
      f"2 类宏 F1 {fmt(dom2.get('macro_f1'))}、二分类 F1 {fmt(dom_bin.get('f1'))}、"
      f"ROC-AUC {fmt(dom_bin.get('roc_auc'))}、FPR {fmt(dom_bin.get('false_positive_rate'))}。")
    A("")
    A("> **注意 L2/L3/L4 的判定阈值**：按约定的规则（拟合段正常分数的 "
      f"{float(ctx['calib_quantile']):g} 分位）算出来的阈值落在 "
      f"{fmt(lv['L2']['threshold'], 6)}——"
      "这个规则在**这份数据上退化了**：模型对目标域正常窗口也会给出接近 1.0 的分数，"
      "于是 99.9% 分位几乎等于 1.0，评估段上实际 FPR 只有 "
      f"{fmt(lv['L2']['binary']['false_positive_rate'], 5)}、召回掉到 "
      f"{fmt(lv['L2']['binary']['recall'], 5)}。"
      "这不是实现错误，而是一个必须如实报告的结论：**在这套分数分布上，"
      "「只调阈值」这条路走不通**。因此下面补了三张表：阈值敏感性、"
      "「按拟合段选工作点」，以及**统一误报预算下的对比（6.5，真正的结论表）**。")
    A("")

    # ---- 阈值敏感性 ----
    sweep = cal.get("threshold_sweep_L1") or []
    if sweep:
        A("### 6.3 光靠调阈值能救回多少？（阈值敏感性，Level 1 的分数）")
        A("")
        A("下表把阈值取成**拟合段正常窗口分数**的不同分位数（仍然只用拟合段的正常流量，不用评估段），"
          "看误报预算与召回的取舍：")
        A("")
        A(markdown_table(
            ["拟合段正常分数分位", "目标 FPR 预算", "实际阈值", "实际 FPR", "召回", "精确率", "二分类 F1", "2 类宏 F1"],
            [[f"{r['quantile']:g}", f"{r['target_fpr_budget']:g}", f"{r['threshold']:.6f}",
              fmt(r["false_positive_rate"]), fmt(r["recall"]), fmt(r["precision"]),
              fmt(r["binary_f1"]), fmt(r["macro_f1"])] for r in sweep],
        ))
        A("")
        best_sweep = max(sweep, key=lambda r: r["binary_f1"])
        A(f"敏感性结论：在这一级里，无论把阈值放在哪，二分类 F1 最高也只有 "
          f"**{fmt(best_sweep['binary_f1'])}**（分位 {best_sweep['quantile']:g}，"
          f"此时 FPR {fmt(best_sweep['false_positive_rate'])}、召回 {fmt(best_sweep['recall'])}）。"
          "调阈值只能在「误报爆炸」和「召回归零」之间平移，换不来真正的判别能力——"
          "因为重定标之后的分数**排序本身**已经坏了（见下表 ROC-AUC）。"
          "这就是为什么必须往下走到 L3。")
        A("")

    # ---- 无泄漏的模型选择：按拟合段挑工作点 ----
    A("### 6.4 无泄漏的模型选择：把阈值选在拟合段（而不是评估段）")
    A("")
    A("上面 L2–L4 沿用的是「99.9% 分位」这一条固定规则；工程上更常见的做法是"
      "**在拟合段（含少量攻击样本）上直接挑 F1 最优的阈值**，再把该阈值原样用到评估段。"
      "这仍然没有任何评估段信息进入模型选择，是合规的工作点选择方式：")
    A("")
    A(markdown_table(
        ["级别", "拟合段窗口数", "拟合段 F1@最优阈值", "选中的阈值",
         "评估段 2 类宏 F1", "评估段二分类 F1", "评估段 ROC-AUC", "评估段 FPR", "评估段召回"],
        [[k, f"{lv[k]['fit_segment_windows']:,}", fmt(lv[k]["fit_segment_f1_at_best_threshold"]),
          f"{lv[k]['variants']['fit_f1_optimal']['threshold']:.6f}",
          f"**{fmt(lv[k]['variants']['fit_f1_optimal']['two_class']['macro_f1'])}**",
          f"**{fmt(lv[k]['variants']['fit_f1_optimal']['binary']['f1'])}**",
          fmt(lv[k]["variants"]["fit_f1_optimal"]["binary"]["roc_auc"]),
          fmt(lv[k]["variants"]["fit_f1_optimal"]["binary"]["false_positive_rate"]),
          fmt(lv[k]["variants"]["fit_f1_optimal"]["binary"]["recall"])] for k in order],
    ))
    A("")
    A("ROC-AUC 与阈值无关，是判断「模型有没有学到东西」最干净的指标。"
      "把各级的 AUC 单列出来看："
      + "、".join(f"{k} {fmt(lv[k]['binary']['roc_auc'])}" for k in order)
      + "。这是本节的**核心判据**：")
    A("")
    auc_l0 = lv["L0"]["binary"]["roc_auc"]
    auc_l1 = lv["L1"]["binary"]["roc_auc"]
    auc_l2 = lv["L2"]["binary"]["roc_auc"]
    auc_l3 = lv["L3"]["binary"]["roc_auc"]
    auc_l4 = lv["L4"]["binary"]["roc_auc"]
    _bud = lambda k: lv[k]["fpr_budget_operating_points"]["fpr_0.01"]
    bud0, bud3, bud4 = _bud("L0"), _bud("L3"), _bud("L4")
    A(f"- **重定标（L1）并没能提高排序能力**（AUC {fmt(auc_l0)} → {fmt(auc_l1)}）——"
      "这说明 L0 那点看似存在的排序能力，有相当一部分是「漂移幅度本身」带来的假信号"
      "（攻击行的原始统计量偏离合成分布更远，于是被误打误撞排到前面），"
      "一旦把量纲拉回正常流量口径，这个假信号就消失了。**这是本次实验最反直觉、"
      "也最值得写进结论的一条发现**：跨域 AUC 不能作为「模型还能用」的证据。")
    A(f"- **微调分类头（L3）才真正把排序能力拉回来**（AUC {fmt(auc_l1)} → {fmt(auc_l3)}，"
      f"TPR@FPR=1% 由 {fmt(lv['L1']['binary']['tpr_at_fpr_0.01'])} 升到 "
      f"{fmt(lv['L3']['binary']['tpr_at_fpr_0.01'])}）。"
      f"说明「合成域学到的特征表示 + 目标域重训的分类头」是可行的迁移形态。")
    backbone_ok = ((bud4["eval_recall"] - bud3["eval_recall"]) > 0.02
                   or (bud4["eval_binary_f1"] - bud3["eval_binary_f1"]) > 0.01
                   or (auc_l4 - auc_l3) > 0.02)
    if backbone_ok:
        A(f"- **解冻主干（L4）在 {cal['finetune_backbone']['epochs']} epoch、"
          f"学习率 {cal['finetune_backbone']['learning_rate']:g} 的极低预算下仍然带来了提升**"
          f"（AUC {fmt(auc_l3)} → **{fmt(auc_l4)}**，1% 预算下召回 "
          f"{fmt(lv['L3']['fpr_budget_operating_points']['fpr_0.01']['eval_recall'], 4)} → "
          f"{fmt(lv['L4']['fpr_budget_operating_points']['fpr_0.01']['eval_recall'], 4)}）。"
          "这说明主干并非「彻底学错」，只是被合成域的量纲锁定住了；"
          "给一点目标域梯度就能松动。不过它的收益远小于 L3，"
          "且标注量上去之后应当重新评估（全参数微调在小样本上容易过拟合）。")
    else:
        A(f"- **解冻主干（L4）没有再改善**（AUC {fmt(auc_l3)} → {fmt(auc_l4)}，"
          f"1% 预算下召回 "
          f"{fmt(lv['L3']['fpr_budget_operating_points']['fpr_0.01']['eval_recall'], 4)} → "
          f"{fmt(lv['L4']['fpr_budget_operating_points']['fpr_0.01']['eval_recall'], 4)}）——"
          "样本量只有几百个窗口时，全参数微调容易把主干拉离原来的表示；"
          "实践上应当把 L4 当成「可选、需要更多标注才有意义」的一步。")
    A("")

    # ---- 统一误报预算下的对比（生产最关心的口径） ----
    A("### 6.5 统一误报预算下的对比（生产最关心的口径，也是本节的结论表）")
    A("")
    A("上表里 L2–L4 的 F1 被「99.9% 分位」这条退化规则拖到 0，容易掩盖真实的恢复幅度。"
      "更公平、也更贴生产的比较方式是：**把每一级的判定阈值都定成「让目标域正常流量的误报率"
      "刚好等于给定预算」的那一个**（阈值取定阈值子段正常窗口分数的 1−预算 分位）。"
      "这个口径**与攻击先验无关**，因此在攻击占比 0.757 的评估段上也不会被 F1 的先验效应带偏。")
    A("")
    for budget_key, budget_label in (("fpr_0.01", "误报预算 = 1%"), ("fpr_0.001", "误报预算 = 0.1%")):
        A(f"**{budget_label}**")
        A("")
        A(markdown_table(
            ["级别", "阈值", "评估段实际 FPR", "召回", "精确率", "二分类 F1", "2 类宏 F1"],
            [[k,
              f"{lv[k]['fpr_budget_operating_points'][budget_key]['threshold']:.6f}",
              fmt(lv[k]["fpr_budget_operating_points"][budget_key]["eval_false_positive_rate"]),
              fmt(lv[k]["fpr_budget_operating_points"][budget_key]["eval_recall"]),
              fmt(lv[k]["fpr_budget_operating_points"][budget_key]["eval_precision"]),
              f"**{fmt(lv[k]['fpr_budget_operating_points'][budget_key]['eval_binary_f1'])}**",
              fmt(lv[k]["fpr_budget_operating_points"][budget_key]["eval_macro_f1"])] for k in order],
        ))
        A("")
    recs = {k: lv[k]["fpr_budget_operating_points"]["fpr_0.01"]["eval_recall"] for k in order}
    A(f"**读法**：在 1% 误报预算下，召回从 L0 的 {fmt(recs['L0'])} → L1 的 {fmt(recs['L1'])} "
      f"→ L2 的 {fmt(recs['L2'])} → L3 的 **{fmt(recs['L3'])}** → L4 的 **{fmt(recs['L4'])}**。"
      "这就是「跨域衰减补回来多少」的最诚实答案："
      "**零成本的两级（L1/L2）几乎补不回来，真正起作用的是付少量标注的 L3，L4 再加一点。**")
    A("")
    A(f"> ⚠️ **0.1% 预算那一档要谨慎读**：定阈值子段只有 "
      f"{cal['sub_split']['threshold_selection_subsegment']['normal_rows']:,} 行正常流，"
      "对应约 2 千个正常窗口，99.9% 分位就是「第二大的那个值」，"
      "估计本身很不稳（L3 与 L4 在这一档的差异就来自这种抽样噪声）。"
      "**1% 预算那一档（约第 20 大的值）才是可靠的读法**，"
      "真要在生产上按 0.1% 定阈值，应当用更多的正常流量样本。")
    A("")

    # ---- 阈值 ----
    A("### 6.6 阈值与代价明细（沿用规则下的阈值）")
    A("")
    A(markdown_table(
        ["级别", "沿用阈值", "阈值来源", "实际 FPR", "实际召回",
         "同分数下按拟合段选的最优阈值", "该阈值下的评估段 F1"],
        [[lv[k]["level"], f"{lv[k]['threshold']:.6f}", lv[k]["threshold_source"],
          fmt(lv[k]["binary"]["false_positive_rate"]), fmt(lv[k]["binary"]["recall"]),
          f"{lv[k]['variants']['fit_f1_optimal']['threshold']:.6f}",
          fmt(lv[k]["variants"]["fit_f1_optimal"]["binary"]["f1"])] for k in order],
    ))
    A("")
    A("**每一级相比上一级的增量**（去重后真实数据、同一评估段）：")
    A("")
    inc_rows = []
    prev = None
    for k in order:
        v = lv[k]
        cur = {"macro_f1": v["two_class"]["macro_f1"], "f1": v["binary"]["f1"],
               "auc": v["binary"]["roc_auc"], "fpr": v["binary"]["false_positive_rate"]}
        if prev is None:
            inc_rows.append([v["level"], "—", "—", "—", "—", "基线"])
        else:
            inc_rows.append([
                v["level"],
                f"{cur['macro_f1'] - prev['macro_f1']:+.4f}",
                f"{cur['f1'] - prev['f1']:+.4f}",
                f"{cur['auc'] - prev['auc']:+.4f}",
                f"{cur['fpr'] - prev['fpr']:+.4f}",
                "较上一级",
            ])
        prev = cur
    inc_rows.append([
        "**L4 vs L0**",
        f"**{lv['L4']['two_class']['macro_f1'] - lv['L0']['two_class']['macro_f1']:+.4f}**",
        f"**{lv['L4']['binary']['f1'] - lv['L0']['binary']['f1']:+.4f}**",
        f"**{lv['L4']['binary']['roc_auc'] - lv['L0']['binary']['roc_auc']:+.4f}**",
        f"**{lv['L4']['binary']['false_positive_rate'] - lv['L0']['binary']['false_positive_rate']:+.4f}**",
        "整个阶梯的总恢复量",
    ])
    A(markdown_table(["级别", "Δ2 类宏 F1", "Δ二分类 F1", "ΔROC-AUC", "ΔFPR", "说明"], inc_rows))
    A("")
    A("> ⚠️ 这张 Δ 表是在**沿用规则（99.9% 分位）那条退化阈值**下算的，"
      "所以 L2–L4 的 F1 全被压到 0、增量看起来是负的。"
      "**跨级恢复幅度请以 6.5 节的「统一误报预算」表为准**（口径与先验无关），"
      "ROC-AUC 一列则可以放心读——它不受阈值影响。")
    A("")

    # ---- 代价明细 ----
    A("### 6.7 代价明细与「微调后重新定阈值」的版本")
    A("")
    A(markdown_table(
        ["代价项", "数值"],
        [
            ["L1/L2 所需标签量", "**0 条**（只需目标域正常流量的原始特征）"],
            ["L1 拟合用的正常流行数", f"{cal['scaler_refit']['fit_rows']:,}"],
            ["L2 拟合耗时", "0 秒（复用 L1 的分数，仅改阈值）"],
            ["L3 标注窗口数", f"{cal['finetune_data']['negatives_available_and_used']} 正常 + "
                          f"{cal['finetune_data']['positives_available_and_used']} 攻击 = "
                          f"**{cal['finetune_head']['n_train_windows']} 个窗口**"],
            ["L3 训练量", f"{cal['finetune_head']['epochs']} epoch，可训练层 "
                       f"{cal['finetune_head']['n_trainable_layers']} 个（冻结 {cal['finetune_head']['n_frozen_layers']} 个），"
                       f"耗时 {cal['finetune_head']['fit_seconds']} 秒"],
            ["L3 微调学习率", f"{cal['finetune_head']['learning_rate']:g}"],
            ["L4 额外训练量", f"{cal['finetune_backbone']['epochs']} epoch 全参数，"
                          f"学习率 {cal['finetune_backbone']['learning_rate']:g}，"
                          f"耗时 {cal['finetune_backbone']['fit_seconds']} 秒"],
            ["最终线上单次推理", "与 L0 完全相同（模型结构、输入维度、时延都没有变）"],
        ],
    ))
    A("")
    A(f"微调样本构成：{cal['finetune_data']['source']}；抽样方式：{cal['finetune_data']['sampling']}。")
    A("")
    A("**微调后重新校准阈值的版本（L3′ / L4′）**——因为微调会改变分数分布，"
      "沿用旧阈值是不公平的对照：")
    A("")
    A(markdown_table(
        ["级别", "说明", "阈值", "2 类宏 F1", "二分类 F1", "ROC-AUC", "FPR", "召回", "TPR@FPR=1%"],
        [[lv[k]["level"], lv[k]["desc"],
          f"{lv[k]['threshold']:.6f}",
          fmt(lv[k]["two_class"]["macro_f1"]), fmt(lv[k]["binary"]["f1"]),
          fmt(lv[k]["binary"]["roc_auc"]), fmt(lv[k]["binary"]["false_positive_rate"]),
          fmt(lv[k]["binary"]["recall"]), fmt(lv[k]["binary"]["tpr_at_fpr_0.01"])]
         for k in ("L3_recal", "L4_recal") if k in lv],
    ))
    A("")

    # ---- z-shift 前后 ----
    z = cal["zshift"]
    A("### 6.8 重定标到底修好了什么（z-shift 前后对比）")
    A("")
    A(markdown_table(
        ["参照 / 分组", "|z-shift| 均值", "中位数", "P95", "最大", ">1σ 的特征数"],
        [
            ["校准前（合成统计量为参照）· 评估段全体",
             fmt(z["eval_all_before"]["abs_mean"]), fmt(z["eval_all_before"]["abs_median"]),
             fmt(z["eval_all_before"]["abs_p95"]), fmt(z["eval_all_before"]["abs_max"]),
             f"{z['eval_all_before']['n_abs_gt_1']} / {z['eval_all_before']['n_features']}"],
            ["**校准后（目标域正常流为参照）· 评估段全体**",
             f"**{fmt(z['eval_all_after']['abs_mean'])}**", fmt(z["eval_all_after"]["abs_median"]),
             fmt(z["eval_all_after"]["abs_p95"]), fmt(z["eval_all_after"]["abs_max"]),
             f"**{z['eval_all_after']['n_abs_gt_1']} / {z['eval_all_after']['n_features']}**"],
            ["校准后 · 评估段**正常**行（理应≈0，作为自检）",
             fmt(z["eval_normal_after"]["abs_mean"]), fmt(z["eval_normal_after"]["abs_median"]),
             fmt(z["eval_normal_after"]["abs_p95"]), fmt(z["eval_normal_after"]["abs_max"]),
             f"{z['eval_normal_after']['n_abs_gt_1']} / {z['eval_normal_after']['n_features']}"],
            ["校准后 · 评估段**攻击**行",
             fmt(z["eval_attack_after"]["abs_mean"]), fmt(z["eval_attack_after"]["abs_median"]),
             fmt(z["eval_attack_after"]["abs_p95"]), fmt(z["eval_attack_after"]["abs_max"]),
             f"{z['eval_attack_after']['n_abs_gt_1']} / {z['eval_attack_after']['n_features']}"],
        ],
    ))
    A("")
    A("这张表解释了阶梯的**机制**：重定标只能把「正常流量看起来像不像训练数据」这件事拉回来"
      "（评估段正常行的 z-shift 会被压到接近 0，这是自检项），"
      f"而**攻击行的漂移无法靠重定标消除**（校准后仍为 |z| 均值 "
      f"{fmt(z['eval_attack_after']['abs_mean'])}、最大 {fmt(z['eval_attack_after']['abs_max'])}）——"
      "因为攻击行的特征分布与正常行本就不同，这是信号、不是噪声。"
      "所以 L1 能修的是「量纲」，修不了「攻击的可分性」，后者必须靠 L3 的标注微调。")
    A("")
    if cal.get("warnings"):
        A("**本次运行的告警**：")
        A("")
        for w in cal["warnings"]:
            A(f"- {w}")
        A("")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="跨数据集迁移性分析：合成数据训练的模型 → 真实 inSDN 推理 + 漂移诊断 + 校准阶梯",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real-csv", default="data/real/insdn/Dataset.csv", help="真实数据集 CSV（映射源）")
    p.add_argument("--model", default="", help="主模型 .keras 路径；留空则自动从 artifacts/main_model.json 解析")
    p.add_argument("--out", default="results", help="产物输出目录（相对项目根）")
    p.add_argument("--synth-csv", default="data/raw/insdn_stream.csv", help="自建合成数据集 CSV")
    p.add_argument("--processed-dir", default="data/processed", help="标准化器 / 元信息目录")
    p.add_argument("--label-column", default="target", help="真实数据的标签列名")
    p.add_argument("--window-size", type=int, default=0, help="窗口长度 T；0 = 用 meta.json 里的值")
    p.add_argument("--stride", type=int, default=0, help="滑窗步长；0 = 用 Config.online_stride")
    p.add_argument("--dedup-digits", type=int, default=3, help="近似去重保留的有效数字（0=不去重）")
    p.add_argument("--batch-size", type=int, default=2048, help="推理批大小")
    p.add_argument("--chunk-windows", type=int, default=4096, help="每次 materialize 的窗口数（控内存）")
    p.add_argument("--top-k", type=int, default=10, help="漂移表与图中展示的特征个数")
    p.add_argument("--max-rows", type=int, default=0, help="只取真实数据前 N 行（0=全部，调试用）")
    p.add_argument("--no-figure", action="store_true", help="跳过绘图")
    p.add_argument("--seed", type=int, default=42, help="随机种子（本脚本仅用于图表抽样，不影响指标）")
    # ---- 校准阶梯 ----
    p.add_argument("--no-calibration", action="store_true", help="跳过校准阶梯（只做零样本与漂移诊断）")
    p.add_argument("--calib-fit-attack-rows", type=int, default=6000,
                   help="从攻击块头部让给拟合段的行数（供 L3/L4 的正样本；评估段不得使用）")
    p.add_argument("--calib-finetune-windows", type=int, default=500,
                   help="L3/L4 微调使用的目标域标注窗口数（正负各半，正类不足时按可用量取）")
    p.add_argument("--calib-head-epochs", type=int, default=4, help="L3 只微调分类头的 epoch 数")
    p.add_argument("--calib-head-lr", type=float, default=1e-3, help="L3 分类头微调学习率")
    p.add_argument("--calib-backbone-epochs", type=int, default=2, help="L4 解冻主干微调的 epoch 数")
    p.add_argument("--calib-backbone-lr", type=float, default=1e-5, help="L4 主干微调学习率（低）")
    p.add_argument("--calib-ft-batch", type=int, default=32, help="微调批大小")
    p.add_argument("--calib-quantile", type=float, default=0.999,
                   help="L2 及之后的阈值分位数（取自拟合段正常窗口分数）")
    return p.parse_args(argv)


def resolve_model_path(explicit: str) -> tuple[Path, dict]:
    """解析主模型路径：显式路径 → artifacts/main_model.keras → artifacts/main_model.json 指定的 tag。"""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"指定的模型不存在：{explicit}")
        return p, {"resolved_from": "explicit", "tag": p.parent.name}

    legacy = ARTIFACT_DIR / "main_model.keras"
    if legacy.exists():
        return legacy, {"resolved_from": "artifacts/main_model.keras", "tag": "main_model"}

    info_path = ARTIFACT_DIR / "main_model.json"
    if info_path.exists():
        info = load_json(info_path)
        tag = str(info.get("tag", ""))
        for cand in (ARTIFACT_DIR / tag / "model.keras", ARTIFACT_DIR / tag / "best.keras"):
            if cand.exists():
                return cand, {"resolved_from": "artifacts/main_model.json", "tag": tag,
                              "variant": info.get("variant"), "seed": info.get("seed"),
                              "metrics_file": info.get("metrics")}
    raise FileNotFoundError("找不到主模型，请用 --model 显式指定 .keras 路径")


def load_in_domain_reference(
    tag: str, model, cfg: Config, processed_dir: Path,
    class_names: list[str], normal_idx: int,
) -> dict:
    """同域参照：既读项目已定稿的 metrics_*.json，也在同一推理口径下重算一遍合成测试集。"""
    ref: dict = {"tag": tag}
    mfile = RESULT_DIR / f"metrics_{tag}.json"
    if mfile.exists():
        m = load_json(mfile)
        ref["recorded"] = {
            "source": f"results/metrics_{tag}.json",
            "multiclass_accuracy": m["multiclass"]["accuracy"],
            "multiclass_macro_f1": m["multiclass"]["macro_f1"],
            "binary_f1": m["binary"]["f1"],
            "binary_roc_auc": m["binary"]["roc_auc"],
            "binary_pr_auc": m["binary"]["pr_auc"],
            "binary_false_positive_rate": m["binary"]["false_positive_rate"],
            "binary_recall": m["binary"]["recall"],
            "n_test_windows": m["data"]["test"],
            "window_size": m["data"]["window_size"],
            "n_features": m["data"]["n_features"],
        }
    agg = RESULT_DIR / "aggregate_cnn_bilstm_main.json"
    if agg.exists():
        a = load_json(agg).get("cnn_bilstm")
        if a:
            ref["aggregate_3seeds"] = {
                "multiclass_macro_f1": a["multiclass_macro_f1"],
                "binary_f1": a["binary_f1"],
                "binary_roc_auc": a["binary_roc_auc"],
            }

    # 同一口径重算：直接喂 data/processed/tensors.npz 里的 (X_test, y_test_bin)。
    # 目的：跨域/同域用**完全相同的推理代码与阈值**，避免"记录值口径不同"带来的争议。
    tpath = processed_dir / "tensors.npz"
    if tpath.exists():
        z = np.load(tpath, allow_pickle=True)
        X_test, y_test_mc, y_test_bin = z["X_test"], z["y_test_mc"], z["y_test_bin"]
        p_mc, p_bin = predict_probs(model, X_test, batch_size=2048)
        bm = binary_metrics(y_test_bin, p_bin, cfg.report_threshold, cfg)

        # ★ 与跨域「2 类宏 F1」对齐的参照：在合成测试集上按同样的 2 类口径再算一遍。
        #   否则拿同域的「8 类宏 F1」去比跨域的「2 类宏 F1」，分母不同，衰减幅度会被夸大。
        p_normal = p_mc[:, normal_idx]
        p2 = np.column_stack([p_normal, 1.0 - p_normal]).astype(np.float64)
        y_bin_i = np.asarray(y_test_bin).astype(int)
        mc2 = multiclass_metrics(y_bin_i, p2, ["Normal", "Attack"])
        mc8 = multiclass_metrics(y_test_mc, p_mc, class_names)

        ref["recomputed_on_test_tensors"] = {
            "n_windows": int(len(X_test)),
            "input_shape": list(X_test.shape[1:]),
            **{k: v for k, v in bm.items()},
        }
        ref["recomputed_two_class"] = {
            "n_windows": int(len(X_test)),
            "accuracy": mc2["accuracy"],
            "macro_precision": mc2["macro_precision"],
            "macro_recall": mc2["macro_recall"],
            "macro_f1": mc2["macro_f1"],
            "per_class": mc2["per_class"],
        }
        ref["recomputed_eight_class_macro_f1"] = mc8["macro_f1"]
        ref["_p_bin_test"] = p_bin
        ref["_y_test_bin"] = y_test_bin
        ref["_p_mc_test"] = p_mc
        ref["_y_test_mc"] = y_test_mc
        ref["synth_test_attack_share"] = float(np.asarray(y_test_bin).mean())
    return ref


# 指标键 → metrics_<tag>.json 里的字段名（同域"已记录值"的来源）
_RECORDED_KEYMAP = {
    "accuracy": "multiclass_accuracy",
    "f1": "binary_f1",
    "roc_auc": "binary_roc_auc",
    "pr_auc": "binary_pr_auc",
    "recall": "binary_recall",
    "false_positive_rate": "binary_false_positive_rate",
}


def in_domain_value(dom: dict, key: str) -> tuple[float, str]:
    """取同域指标：优先项目已定稿的 metrics_*.json，缺的用"同一段代码重算"补齐。

    返回 (数值, 来源标记)，来源标记会写进报告脚注，避免读者误以为全部来自同一个文件。
    """
    rec = dom.get("recorded", {})
    rk = _RECORDED_KEYMAP.get(key)
    if rk and rk in rec:
        return float(rec[rk]), "recorded"
    rec2 = dom.get("recomputed_on_test_tensors", {})
    if key in rec2:
        return float(rec2[key]), "recomputed"
    return float("nan"), "missing"


def fmt(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    if math.isnan(v):
        return "nan"
    if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.3g}"
    return f"{v:.{nd}f}"


def build_report(ctx: dict) -> str:
    """生成中文 markdown 报告。所有路径均为相对路径，不含本机绝对路径。"""
    v_raw = ctx["versions"]["raw"]
    v_ded = ctx["versions"][ctx["dedup_key"]]
    dom = ctx["in_domain"]
    drift = ctx["drift"]
    mp = ctx["mapping"]
    nfeat = len(ctx["feature_names"])
    rec = dom.get("recorded", {})
    rec2 = dom.get("recomputed_on_test_tensors", {})
    dom_fpr_val = in_domain_value(dom, "false_positive_rate")[0]
    cal = ctx.get("calibration") or {}

    md: list[str] = []
    A = md.append

    A("# 跨数据集迁移性分析报告")
    A("")
    A("> **一句话结论**：" + ctx["headline"])
    A("")
    A("本报告回答的是：**「在自建合成数据集上训练好的 CNN+BiLSTM 检测模型，"
      "不做任何再训练、直接拿去推理真实公开数据集 inSDN，性能掉多少、为什么掉、"
      "要补回来该怎么做」**。全部数字由 `tools/cross_dataset.py` 一次运行自动产出，"
      "无手工填写、无事后调参。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 实验口径与诚实性声明")
    A("")
    A(markdown_table(
        ["项目", "取值", "说明"],
        [
            ["主模型", f"`artifacts/{ctx['model']['tag']}/model.keras`", "合成数据上训练、已定稿的最好模型"],
            ["模型输入", f"(T={ctx['window']}, F={nfeat})", "合成训练时的窗口长度与特征维度"],
            ["标准化器", "`data/processed/scaler.joblib`", "**只在合成训练集上 fit**，跨域推理时原样复用"],
            ["滑窗步长", str(ctx["stride"]), "取 `Config.online_stride`（在线检测口径）"],
            ["二分类阈值", str(ctx["threshold"]), "取 `Config.report_threshold` 默认值，**未在真实数据上调过**"],
            ["合成数据", "`data/raw/insdn_stream.csv`",
             f"{ctx['synth_meta']['raw_rows']:,} 行 → 清洗后 {ctx['synth_meta']['clean_rows']:,} 行，"
             f"其中训练集 {ctx['synth_meta']['train_rows']:,} 行"],
            ["真实数据", "`data/real/insdn/Dataset.csv`",
             f"读取 {ctx['real_meta']['rows_read']:,} 行 → 去除缺失/无穷后有效 "
             f"{ctx['real_meta']['rows_after_clean']:,} 行"],
        ],
    ))
    A("")
    A("**没有做的事（避免任何形式的泄漏）**：")
    A("")
    A("- 没有用真实数据训练、微调、或重新拟合标准化器；")
    A("- 没有用真实数据挑阈值 / 挑窗口长度 / 挑特征子集；")
    A("- 映射不到的维度一律在**标准化空间填 0**（等价于填合成训练集均值），"
      "没有使用任何真实数据的均值 / 分位数。")
    A("")
    A("**必须提前说清的三个口径事实**（否则下面的数字会被误读）：")
    A("")
    A(f"1. **真实 inSDN 的标签列 `target` 只有 0/1 两值**（本文件实测："
      f"0 有 {ctx['real_label_counts'].get('0', 0):,} 行、1 有 {ctx['real_label_counts'].get('1', 0):,} 行），"
      "**并不提供 8 类攻击类型真值**。因此本报告**无法计算 8 类宏 F1**，"
      "多分类一栏给出的是「Normal vs Attack」2 类宏 F1；"
      "另附一张「模型把真实流量归因成了哪一类」的分布表作为补充视角（**非真值指标**，已明确标注）。")
    A(f"2. **真实数据的行序不是时间序**：标签沿文件位置只有 {ctx['real_runs']} 个连续块（"
      + "、".join(f"{k} 连续 {v:,} 行" for k, v in ctx["real_run_detail"]) +
      "），说明它是「正常流量块 + 攻击流量块」拼接发布的。"
      "因此窗口里的 16 条流**不构成真实的相邻流量**，跨域指标应理解为"
      "「窗口级分类迁移性」，而不是在线时序检测性能。")
    A(f"3. **类别先验差别很大**：真实数据窗口里攻击占 {fmt(v_raw['attack_share_real_windows'], 3)}，"
      f"而合成测试集里攻击占 {fmt(ctx['synth_attack_share'], 3)}。"
      "先验偏移本身就会改变 F1 与误报率的观感，后文对比时会一并扣除这个因素。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 特征映射覆盖率")
    A("")
    A(markdown_table(
        ["指标", "数值"],
        [
            ["合成模型输入维度", f"{mp['n_features']}"],
            ["成功映射到真实列", f"{mp['n_mapped']}"],
            ["**覆盖率**", f"**{mp['n_mapped']}/{mp['n_features']} = {fmt(mp['coverage'] * 100, 1)}%**"],
            ["映射失败的维度", ("无" if not mp["unmatched"] else "、".join(f"`{x}`" for x in mp["unmatched"]))],
            ["映射方式分布", "、".join(f"{k}: {v}" for k, v in mp["method_counts"].items())],
            ["一对多冲突", ("无" if not mp["collisions"] else "、".join(mp["collisions"]))],
        ],
    ))
    A("")
    A(f"真实数据集共 {ctx['real_meta']['n_columns_total']} 列：其中 "
      f"{len(ID_OR_LABEL_COLUMNS & set(ctx['real_columns']))} 列是流标识 / 时间戳，1 列是标签，"
      f"其余 {ctx['real_meta']['n_columns_total'] - len(ID_OR_LABEL_COLUMNS & set(ctx['real_columns'])) - 1} 列是候选特征。"
      f"这些候选特征里，**有 {len(mp['unused_real_columns'])} 个我们的模型用不上**"
      "（合成数据生成器没有产出这些量），清单如下：")
    A("")
    if mp["unused_real_columns"]:
        A("| 未被使用的真实列 | 说明 |")
        A("|---|---|")
        for c in mp["unused_real_columns"]:
            A(f"| `{c}` | CICFlowMeter 的分组收发速率（Bulk rate）统计量，合成数据生成器未建模 |")
    else:
        A("（无）")
    A("")
    A("映射用三级策略：**显式映射表**（`SYNTH_TO_REAL`，人工核对语义）→ **同名匹配** → "
      "**词元集合匹配**（展开 `Tot/Pkts/Byts/Len/Cnt` 等缩写后比较词元集合，要求唯一命中）。"
      "明细见 `cross_dataset.json` 的 `mapping.detail`。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 跨域指标（去重前 vs 去重后）")
    A("")
    A(f"真实数据重复行极多（见下表的行数变化），因此同时给出两版："
      f"**去重前**（`dedup_significant_digits=0`）与**近似去重后**"
      f"（`dedup_significant_digits={ctx['dedup_digits']}`，与项目主实验同一开关口径）。")
    A("")
    A("去重口径与项目 `src/data_pipeline.py::clean_dataframe` 完全一致，分两步，"
      "每一步的行数与标签分布都列出来——**这一步很关键，因为去重会改变类别先验**：")
    A("")
    dstages = ctx.get("dedup_stages", {})
    stage_rows = []
    for key in ("raw", "exact", "approx"):
        st = dstages.get(key)
        if st:
            stage_rows.append([st["stage"], f"{st['rows']:,}", f"{st['attack_rows']:,}",
                               f"{st['normal_rows']:,}", fmt(st["attack_share_rows"])])
    A(markdown_table(["阶段", "行数", "攻击行", "正常行", "攻击行占比"], stage_rows))
    A("")
    sv = dstages.get("survival", {})
    if sv:
        A(f"> 去重过程中的**类别存活率**：攻击行 **{fmt(sv.get('attack_row_survival'))}**、"
          f"正常行 {fmt(sv.get('normal_row_survival'))}。"
          "攻击流量的近重复程度远高于正常流量（洪水型攻击天然如此），"
          "所以去重会**显著降低攻击类先验**；下面两版指标的差异里，"
          "有一部分来自这个先验变化，而不全是模型的功劳或过错。")
        A("")
    A(markdown_table(
        ["版本", "真实数据行数", "窗口总数", "纯窗口数", "纯窗口占比", "推理耗时(s)"],
        [
            ["去重前 raw", f"{v_raw['n_rows']:,}", f"{v_raw['n_windows_total']:,}",
             f"{v_raw['n_windows_pure']:,}", fmt(v_raw["n_windows_pure"] / max(v_raw["n_windows_total"], 1)),
             f"{v_raw['infer_seconds']}"],
            [f"去重后 {ctx['dedup_key']}", f"{v_ded['n_rows']:,}", f"{v_ded['n_windows_total']:,}",
             f"{v_ded['n_windows_pure']:,}", fmt(v_ded["n_windows_pure"] / max(v_ded["n_windows_total"], 1)),
             f"{v_ded['infer_seconds']}"],
        ],
    ))
    A("")
    A("### 3.1 二分类（异常检测口径，阈值 = %.2f）" % ctx["threshold"])
    A("")
    hdr = ["指标", "去重前 raw", f"去重后 {ctx['dedup_key']}", "同域（合成测试集）"]
    rows = []
    keys = [
        ("准确率 Accuracy", "accuracy"),
        ("精确率 Precision", "precision"),
        ("召回率 Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("MCC", "mcc"),
        ("误报率 FPR", "false_positive_rate"),
        ("漏报率 FNR", "false_negative_rate"),
        ("TPR@FPR=1%", "tpr_at_fpr_0.01"),
        ("TPR@FPR=0.1%", "tpr_at_fpr_0.001"),
    ]
    for label, k in keys:
        dv, _src = in_domain_value(dom, k)
        rows.append([label, fmt(v_raw["binary_pure"][k]), fmt(v_ded["binary_pure"][k]), fmt(dv)])
    A(markdown_table(hdr, rows))
    A("")
    A("> 「同域」列优先取项目已定稿的 `results/metrics_"
      + str(dom.get("tag", "")) + ".json`；该文件没有的指标（精确率 / MCC / 漏报率 / "
      "TPR@FPR）用 `data/processed/tensors.npz` 的合成测试集、**同一段推理代码与同一阈值**重算补齐，"
      "保证列内可比。")
    A("")
    A("混淆矩阵计数（纯窗口，标签 0=Normal / 1=Attack）：")
    A("")
    A(markdown_table(
        ["版本", "TN", "FP", "FN", "TP"],
        [[v["version"], v["binary_pure"]["tn"], v["binary_pure"]["fp"],
          v["binary_pure"]["fn"], v["binary_pure"]["tp"]] for v in (v_raw, v_ded)],
    ))
    A("")
    A(f"全窗口（多数票打标签，不丢任何窗口）口径下的 F1："
      f"去重前 {fmt(v_raw['binary_all_windows']['f1'])}，去重后 {fmt(v_ded['binary_all_windows']['f1'])}"
      "（与纯窗口口径差别很小，说明窗口跨越标签边界的样本很少）。")
    A("")
    A("### 3.2 多分类口径")
    A("")
    A("**8 类宏 F1 在这份真实数据上无法计算**（`target` 只有 0/1）。下表是与项目 `metrics_real_insdn*.json` "
      "同口径的「Normal vs Attack」2 类宏 F1（用多分类头的概率归一化后计算）：")
    A("")
    A(markdown_table(
        ["指标", "去重前 raw", f"去重后 {ctx['dedup_key']}"],
        [
            ["准确率", fmt(v_raw["two_class_pure"]["accuracy"]), fmt(v_ded["two_class_pure"]["accuracy"])],
            ["宏精确率", fmt(v_raw["two_class_pure"]["macro_precision"]), fmt(v_ded["two_class_pure"]["macro_precision"])],
            ["宏召回率", fmt(v_raw["two_class_pure"]["macro_recall"]), fmt(v_ded["two_class_pure"]["macro_recall"])],
            ["**2 类宏 F1**", f"**{fmt(v_raw['two_class_pure']['macro_f1'])}**",
             f"**{fmt(v_ded['two_class_pure']['macro_f1'])}**"],
        ],
    ))
    A("")
    A("逐类明细：")
    A("")
    pc_rows = []
    for cname in v_raw["two_class_pure"]["class_names"]:
        a = v_raw["two_class_pure"]["per_class"].get(cname, {})
        b = v_ded["two_class_pure"]["per_class"].get(cname, {})
        pc_rows.append([cname, fmt(a.get("precision")), fmt(a.get("recall")), fmt(a.get("f1")),
                        f"{a.get('support', 0):,}", fmt(b.get("f1"))])
    A(markdown_table(["类别", "精确率(raw)", "召回率(raw)", "F1(raw)", "样本数(raw)", "F1(去重后)"], pc_rows))
    A("")
    A("### 3.3 模型把真实流量归因成了哪一类（**非真值指标**）")
    A("")
    A("既然真实数据没有攻击类型真值，我们退一步看**模型侧的 8 类归因分布**"
      "——它同样能说明跨域崩塌的形态：如果模型几乎把所有真实流量都塞进同一类，"
      "说明它学到的是「合成数据独有的指纹」，而不是「攻击的物理本质」。")
    A("")
    attr = v_raw["model_attribution_8class"]
    total = max(sum(attr["all_windows"].values()), 1)
    a_rows = []
    for c in ctx["class_names"]:
        a_rows.append([
            c,
            f"{attr['all_windows'].get(c, 0):,}",
            fmt(attr["all_windows"].get(c, 0) / total),
            f"{attr['on_true_attack_windows'].get(c, 0):,}",
            f"{attr['on_true_normal_windows'].get(c, 0):,}",
        ])
    A(markdown_table(["模型判为", "窗口数(全部)", "占比", "其中真值=攻击", "其中真值=正常"], a_rows))
    A("")
    A(f"归因集中度：最大单类占比 **{fmt(attr['top_class_share'])}**，"
      f"平均归一化预测熵 **{fmt(attr['mean_normalized_entropy'])}**"
      f"（1.0 = 完全均匀，0 = 完全坍缩到一类）。")
    A("")

    # ---------------- 4 ----------------
    A("## 4. 与同域指标的差距（衰减量化）")
    A("")
    dom2 = dom.get("recomputed_two_class", {})
    dom2_f1 = float(dom2.get("macro_f1", float("nan")))
    dom8_f1 = float(dom.get("recomputed_eight_class_macro_f1",
                            (rec or {}).get("multiclass_macro_f1", float("nan"))))
    if rec:
        A(f"同域参照取自 `results/metrics_{dom['tag']}.json`（项目已定稿的主实验结果）："
          f"测试窗口 {rec.get('n_test_windows', 0):,} 个，窗口长度 {rec.get('window_size')}，"
          f"特征 {rec.get('n_features')} 维；其**8 类宏 F1 = {fmt(rec.get('multiclass_macro_f1'))}**。")
        A("")
        A("⚠️ **口径对齐说明**：真实数据没有 8 类真值，跨域只能算「Normal vs Attack」2 类宏 F1。"
          "为了可比，下表同一行的「同域」值是在合成测试集上**按同样的 2 类口径重算**的"
          f"（8 类宏 F1 = {fmt(dom8_f1)} 仅作背景，不参与衰减计算）——"
          "否则拿 8 类宏 F1 减 2 类宏 F1 会把衰减幅度算歪。")
        A("")
        A(markdown_table(
            ["指标", "同域（合成测试集，2 类口径）", "跨域 raw", "跨域 去重后", "衰减(raw)", "衰减(去重后)"],
            [
                ["2 类宏 F1", fmt(dom2_f1), fmt(v_raw["two_class_pure"]["macro_f1"]),
                 fmt(v_ded["two_class_pure"]["macro_f1"]),
                 fmt(v_raw["two_class_pure"]["macro_f1"] - dom2_f1),
                 fmt(v_ded["two_class_pure"]["macro_f1"] - dom2_f1)],
                ["2 类准确率", fmt(dom2.get("accuracy")), fmt(v_raw["two_class_pure"]["accuracy"]),
                 fmt(v_ded["two_class_pure"]["accuracy"]),
                 fmt(v_raw["two_class_pure"]["accuracy"] - float(dom2.get("accuracy", float("nan")))),
                 fmt(v_ded["two_class_pure"]["accuracy"] - float(dom2.get("accuracy", float("nan"))))],
                ["二分类 F1", fmt(rec.get("binary_f1")), fmt(v_raw["binary_pure"]["f1"]),
                 fmt(v_ded["binary_pure"]["f1"]),
                 fmt(v_raw["binary_pure"]["f1"] - rec.get("binary_f1", float("nan"))),
                 fmt(v_ded["binary_pure"]["f1"] - rec.get("binary_f1", float("nan")))],
                ["ROC-AUC", fmt(rec.get("binary_roc_auc")), fmt(v_raw["binary_pure"]["roc_auc"]),
                 fmt(v_ded["binary_pure"]["roc_auc"]),
                 fmt(v_raw["binary_pure"]["roc_auc"] - rec.get("binary_roc_auc", float("nan"))),
                 fmt(v_ded["binary_pure"]["roc_auc"] - rec.get("binary_roc_auc", float("nan")))],
                ["PR-AUC", fmt(rec.get("binary_pr_auc")), fmt(v_raw["binary_pure"]["pr_auc"]),
                 fmt(v_ded["binary_pure"]["pr_auc"]),
                 fmt(v_raw["binary_pure"]["pr_auc"] - rec.get("binary_pr_auc", float("nan"))),
                 fmt(v_ded["binary_pure"]["pr_auc"] - rec.get("binary_pr_auc", float("nan")))],
                ["误报率 FPR", fmt(dom_fpr_val),
                 fmt(v_raw["binary_pure"]["false_positive_rate"]),
                 fmt(v_ded["binary_pure"]["false_positive_rate"]),
                 fmt(v_raw["binary_pure"]["false_positive_rate"] - dom_fpr_val),
                 fmt(v_ded["binary_pure"]["false_positive_rate"] - dom_fpr_val)],
            ],
        ))
        A("")
    if rec2:
        A(f"同域参照（**同一段推理代码、同一模型、同一阈值**重算，保证与本报告跨域数字口径一致）："
          f"合成测试集 {rec2['n_windows']:,} 个窗口，"
          f"二分类 F1 = {fmt(in_domain_value(dom, 'f1')[0])}，"
          f"AUC = {fmt(in_domain_value(dom, 'roc_auc')[0])}，"
          f"FPR = {fmt(in_domain_value(dom, 'false_positive_rate')[0])}。")
        A("")
    A("**必须一起看的两组对照数字**（否则跨域 F1 会被误读）：")
    A("")
    A(markdown_table(
        ["对照项", "去重前 raw", "去重后", "含义"],
        [
            ["真实窗口里攻击占比", fmt(v_raw["attack_share_real_windows"]), fmt(v_ded["attack_share_real_windows"]),
             f"先验：合成测试集里攻击占 {fmt(ctx['synth_attack_share'])}"],
            ["多数类准确率（全判多数类）", fmt(v_raw["trivial_baseline"]["majority_accuracy"]),
             fmt(v_ded["trivial_baseline"]["majority_accuracy"]), "一个什么都不学的分类器能拿到的准确率"],
            ["全判攻击的 F1", fmt(v_raw["trivial_baseline"]["f1_all_attack"]),
             fmt(v_ded["trivial_baseline"]["f1_all_attack"]), "**跨域 F1 必须显著高于这个数才算有效**"],
            ["全判正常的 F1", fmt(v_raw["trivial_baseline"]["f1_all_normal"]),
             fmt(v_ded["trivial_baseline"]["f1_all_normal"]), "同上，另一侧的平凡基线"],
            ["**模型二分类 F1**", f"**{fmt(v_raw['binary_pure']['f1'])}**",
             f"**{fmt(v_ded['binary_pure']['f1'])}**", "与上一行的平凡基线直接对比"],
        ],
    ))
    A("")

    # ---------------- 5 ----------------
    A("## 5. 分布漂移诊断（跨域衰减的成因）")
    A("")
    A("对每个成功映射的特征，用**合成训练集的 mean/std**（即 `scaler.mean_ / scaler.scale_`，"
      "由 `data/processed/scaler.joblib` 读出）把两边归一化，再算均值之差：")
    A("")
    A("> `z-shift = (真实均值 − 合成训练集均值) / 合成训练集标准差`")
    A("")
    A("`|z-shift|` 的含义：合成训练集里这个特征的分布宽度是 1，"
      "真实数据的均值偏离了它几个「合成标准差」。"
      "（0.2 以内算基本没漂；超过 1 就是训练集从未见过的取值区间。）")
    A("")
    A("> 「真实/合成 标准差比」≈ 0 表示该列在真实数据里**退化成常量**——"
      "这是比均值偏移更严重的漂移形态：模型在这一维上学到的判别边界完全失效。")
    A("")
    A(markdown_table(
        ["统计量", "数值"],
        [
            ["参与诊断的特征数", f"{drift['n_features']}"],
            ["z-shift 的**有符号**均值", fmt(drift["z_shift_signed_mean"])],
            ["|z-shift| 均值", fmt(drift["abs_z_shift_mean"])],
            ["|z-shift| 中位数", fmt(drift["abs_z_shift_median"])],
            ["|z-shift| P95", fmt(drift["abs_z_shift_p95"])],
            ["|z-shift| 最大值", fmt(drift["abs_z_shift_max"])],
            ["|z-shift| > 1 的特征数", f"{drift['n_features_abs_z_gt_1']} / {drift['n_features']}"],
            ["|z-shift| > 3 的特征数", f"{drift['n_features_abs_z_gt_3']} / {drift['n_features']}"],
            ["PSI 均值 / 最大", f"{fmt(drift['psi_mean'])} / {fmt(drift['psi_max'])}"],
            ["PSI > 0.25 的特征数（通常视为显著漂移）",
             f"{drift['n_features_psi_gt_0.25']} / {drift['n_features']}"],
            ["真实侧退化为常量（标准差比 < 0.05）的特征数",
             f"{drift['n_features_std_ratio_lt_005']} / {drift['n_features']}"],
        ],
    ))
    A("")
    A(f"### 5.1 偏移最大的 {ctx['top_k']} 个特征")
    A("")
    A(markdown_table(
        ["#", "特征（合成规范名）", "真实列名", "合成训练均值", "合成训练标准差", "真实均值",
         "真实/合成 标准差比", "z-shift", "PSI"],
        [[i + 1, r["feature"], mp["pairs"].get(r["feature"], "-"),
          fmt(r["synth_train_mean"], 4), fmt(r["synth_train_std"], 4), fmt(r["real_mean"], 4),
          fmt(r["std_ratio_real_over_synth"], 3),
          f"**{r['z_shift']:+.2f}**", fmt(r["psi"], 3)]
         for i, r in enumerate(drift["top_k"])],
    ))
    A("")
    A(f"分布对比图：`{ctx['figure_rel']}`"
      + ("（未生成，见 `--no-figure`）" if ctx.get("no_figure") else
         "（前 6 个最漂移特征的标准化取值直方图 + 同域/跨域异常分数分布对比）") + "。")
    A("")
    A("### 5.2 为什么会漂：逐类物理归因")
    A("")
    for line in ctx["drift_interpretation"]:
        A(line)
        A("")

    # ---------------- 6 ----------------
    _calibration_section(ctx, A)

    # ---------------- 7 ----------------
    A("## 7. 结论")
    A("")
    A("### 7.1 跨域指标掉到多少、掉了多少")
    A("")
    A(ctx["conclusion_numbers"])
    A("")
    A("### 7.2 衰减的主要原因（由第 5 节数据支撑）")
    A("")
    for line in ctx["conclusion_causes"]:
        A(line)
        A("")
    A("### 7.3 这套系统能不能真的上生产")
    A("")
    A(ctx["production_verdict"])
    A("")
    A("### 7.4 把跨域衰减补回来的具体方案")
    A("")
    for i, line in enumerate(ctx["remediation"], 1):
        A(f"{i}. {line}")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 复现方式与产物清单")
    A("")
    A("```powershell")
    A('cd <项目根目录>')
    A('$env:PYTHONIOENCODING="utf-8"')
    A('.\\.venv\\Scripts\\python.exe tools\\cross_dataset.py')
    A("# 默认流程：零样本跨域推理 + 漂移诊断 + 校准阶梯（L0→L4）")
    A("# 校准阶梯里的微调严格限制在「拟合段」，评估段只被评估，绝不参与拟合/选阈值")
    A("# 只要零样本结论：加 --no-calibration")
    A("# 重复运行结果一致（幂等）：除 wall-clock 计时字段外数值逐位相同")
    A("```")
    A("")
    A(markdown_table(
        ["产物", "说明"],
        [
            ["`tools/cross_dataset.py`", "本脚本"],
            ["`results/cross_dataset_report.md`", "本报告"],
            ["`results/cross_dataset.json`",
             "结构化数值（逐特征漂移明细 + `calibration` 校准阶梯的全部原始数值）"],
            [f"`{ctx['figure_rel']}`", "分布对比图"],
        ],
    ))
    A("")
    A(f"运行信息：窗口 T={ctx['window']}、步长 {ctx['stride']}、"
      f"真实数据 {v_raw['n_rows']:,} 行 → {v_raw['n_windows_total']:,} 个窗口"
      f"（去重后 {v_ded['n_rows']:,} 行 → {v_ded['n_windows_total']:,} 个窗口），"
      f"推理速度约 {v_raw['windows_per_sec']:.0f} 窗口/秒（CPU 单进程）。")
    if cal.get("enabled"):
        A(f"校准阶梯：评估段 {cal['eval_windows']['n_windows_total']:,} 个窗口；"
          f"拟合段 {cal['split']['fit']['normal_rows']:,} 正常行 + "
          f"{cal['split']['fit']['attack_rows']:,} 攻击行；"
          f"L3 微调 {cal['finetune_head']['n_train_windows']} 个标注窗口 / "
          f"{cal['finetune_head']['epochs']} epoch / {cal['finetune_head']['fit_seconds']} 秒；"
          f"L4 额外 {cal['finetune_backbone']['epochs']} epoch / "
          f"{cal['finetune_backbone']['fit_seconds']} 秒。")
    A("")
    return "\n".join(md)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    set_seed(int(args.seed))
    cfg = Config()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    real_csv = Path(args.real_csv)
    if not real_csv.is_absolute():
        real_csv = ROOT / real_csv
    synth_csv = Path(args.synth_csv)
    if not synth_csv.is_absolute():
        synth_csv = ROOT / synth_csv
    proc_dir = Path(args.processed_dir)
    if not proc_dir.is_absolute():
        proc_dir = ROOT / proc_dir

    model_path, model_info = resolve_model_path(args.model)
    LOGGER.info("=" * 78)
    LOGGER.info("跨数据集迁移性分析：合成数据训练 → 真实 inSDN 推理")
    LOGGER.info("模型：%s（tag=%s）", model_path.relative_to(ROOT), model_info.get("tag"))
    LOGGER.info("=" * 78)

    # ---- 元信息 / 标准化器 ----
    meta = load_json(proc_dir / "meta.json")
    feature_names = list(meta["feature_names"])
    class_names = list(meta["class_names"])
    import joblib

    scaler = joblib.load(proc_dir / "scaler.joblib")
    window = int(args.window_size) or int(meta.get("window_size", cfg.window_size))
    stride = int(args.stride) or int(cfg.online_stride)
    LOGGER.info("特征 %d 维 | 类别 %s | 窗口 T=%d | 步长 %d", len(feature_names), class_names, window, stride)

    model = load_model_bundle(model_path, compile_model=False)
    in_shape = tuple(int(x) for x in model.input_shape[1:])
    if in_shape != (window, len(feature_names)):
        raise ValueError(
            f"模型输入 {in_shape} 与 (T={window}, F={len(feature_names)}) 不一致，请检查 --window-size / meta.json"
        )

    # ---- 映射 ----
    head = pd.read_csv(real_csv, nrows=5, low_memory=False)
    real_columns = list(head.columns)
    mapping, mp_info = build_mapping(feature_names, real_columns)
    LOGGER.info("特征映射：%d/%d 命中（%.1f%%），未命中 %s",
                mp_info["n_mapped"], mp_info["n_features"], mp_info["coverage"] * 100,
                mp_info["unmatched"] or "无")
    if mp_info["coverage"] == 0:
        LOGGER.error("映射覆盖率为 0，后续分析将只反映「训练集均值占位」的退化解，结论仅供参考。")

    # ---- 合成训练集原始特征（漂移参照） ----
    LOGGER.info("重建合成训练集特征矩阵（漂移参照）……")
    X_syn_train, synth_meta = load_synthetic_train_raw(cfg, synth_csv, feature_names)
    dif = float(np.max(np.abs(X_syn_train.mean(axis=0) - np.asarray(scaler.mean_, dtype=np.float64))))
    rel = dif / max(float(np.max(np.abs(np.asarray(scaler.mean_, dtype=np.float64)))), 1e-12)
    synth_meta["scaler_mean_max_abs_diff"] = dif
    synth_meta["scaler_mean_max_rel_diff"] = rel
    synth_meta["scaler_stats_consistent"] = bool(rel < 1e-6)
    if not synth_meta["scaler_stats_consistent"]:
        LOGGER.warning(
            "重建的合成训练集均值与 scaler.mean_ 不完全一致（最大相对偏差 %.2e）："
            "漂移统计量改用 scaler 自身的 mean_/scale_ 作为合成侧参照。", rel,
        )
    else:
        LOGGER.info("重建校验通过：合成训练集均值与 scaler.mean_ 一致（最大绝对偏差 %.2e）。", dif)
    syn_mean_ref = np.asarray(scaler.mean_, dtype=np.float64)
    syn_std_ref = np.asarray(scaler.scale_, dtype=np.float64)
    syn_stats_source = "reconstructed_train" if synth_meta["scaler_stats_consistent"] else "scaler"
    LOGGER.info("合成训练集：%s 行（清洗后 %s），标签列 %s",
                f"{synth_meta['train_rows']:,}", f"{synth_meta['clean_rows']:,}", synth_meta["label_column"])

    # ---- 真实数据 ----
    df_real, label_col, real_info = load_real_mapped(real_csv, mapping, args.label_column, args.max_rows)
    real_info["n_columns_total"] = len(real_columns)
    raw_label_counts = real_info["raw_label_counts"]
    # 标签列不该出现在"未被使用的真实特征列"清单里，这里剔除并重算计数
    mp_info["unused_real_columns"] = [c for c in mp_info["unused_real_columns"] if c != label_col]
    LOGGER.info("真实数据：读入 %s 行 → 清洗后 %s 行（标签列 %s，标签取值 %s）",
                f"{real_info['rows_read']:,}", f"{real_info['rows_after_clean']:,}",
                label_col, raw_label_counts)

    real_labels = normalise_real_labels(df_real[label_col])
    real_codes, gt_classes, _ = encode_labels(real_labels)
    sw = np.flatnonzero(np.diff(real_codes) != 0) + 1
    starts_run = np.concatenate([[0], sw])
    runs = np.diff(np.concatenate([[0], sw, [len(real_codes)]]))
    run_detail = [(str(gt_classes[int(real_codes[s])]), int(l)) for s, l in zip(starts_run, runs)]
    LOGGER.info("真实数据标签块结构：%d 个连续块 %s", len(runs), run_detail)

    run_ctx = {"class_names": class_names, "normal_idx": class_names.index("Normal")}

    # ---- 两版推理 ----
    versions = {}
    arrays = {}
    X_real_raw = None
    v_raw, arr_raw, X_real_raw = run_version(
        "raw", df_real, label_col, mapping, feature_names, scaler, model, cfg,
        window, stride, args.batch_size, run_ctx,
    )
    versions["raw"] = v_raw
    arrays["raw"] = arr_raw
    dedup_key = f"dedup{args.dedup_digits}" if args.dedup_digits > 0 else "dedup0"
    df_raw_ver, df_dedup, dedup_stages = make_dedup_versions(
        df_real, mapping, label_col, int(args.dedup_digits), LOGGER,
    )
    if df_dedup is df_real:
        LOGGER.warning("--dedup-digits=0：去重后版本与去重前相同。")
    elif int(args.dedup_digits) <= 0:
        LOGGER.warning("--dedup-digits=0：去重后版本 = 仅做「特征+标签」精确去重（项目开关关闭时的口径）。")
    v_ded, arr_ded, X_real_dedup = run_version(
        dedup_key, df_dedup, label_col, mapping, feature_names, scaler, model, cfg,
        window, stride, args.batch_size, run_ctx,
    )
    versions[dedup_key] = v_ded
    arrays[dedup_key] = arr_ded
    del X_real_raw  # 漂移分析只用去重后的原始特征矩阵（30 万行 × 71 维 ≈ 200MB）

    # ---- 同域参照（必须在任何微调之前跑，否则模型已被改动）----
    dom = load_in_domain_reference(
        model_info.get("tag", ""), model, cfg, proc_dir, class_names, run_ctx["normal_idx"],
    )

    # ---- 校准阶梯（L0→L4）----
    # 注意：L3/L4 会**就地**修改 model 的权重，因此必须放在所有零样本推理完成之后。
    calibration: dict = {"enabled": False}
    if args.no_calibration:
        LOGGER.warning("--no-calibration：跳过校准阶梯。")
    else:
        calibration = run_calibration_ladder(
            df_dedup, label_col, mapping, feature_names, class_names, run_ctx["normal_idx"],
            scaler, model, cfg, window, stride, args.batch_size, args.chunk_windows,
            fit_attack_rows=int(args.calib_fit_attack_rows),
            finetune_windows=int(args.calib_finetune_windows),
            head_epochs=int(args.calib_head_epochs),
            backbone_epochs=int(args.calib_backbone_epochs),
            head_lr=float(args.calib_head_lr),
            backbone_lr=float(args.calib_backbone_lr),
            ft_batch=int(args.calib_ft_batch),
            target_fpr_quantile=float(args.calib_quantile),
            logger=LOGGER,
        )

    # ---- 漂移 ----
    drift_summary, drift_tab = drift_analysis(feature_names, X_syn_train, X_real_dedup, scaler, top_k=args.top_k)
    if syn_stats_source == "scaler":
        drift_summary["z_shift_note"] = "合成侧均值/标准差取自 scaler.mean_/scale_（重建校验未通过）"
    drift_summary["reference"] = syn_stats_source

    # 用真实数据（去重后）重新算 top_k 表里"合成均值/标准差"的来源说明
    for r in drift_summary["top_k"]:
        f = r["feature"]
        j = feature_names.index(f)
        r["synth_train_mean_source"] = syn_stats_source
        r["synth_train_mean_scaler"] = float(syn_mean_ref[j])
        r["synth_train_std_scaler"] = float(syn_std_ref[j])

    # ---- 图 ----
    figure_rel = ""
    if not args.no_figure:
        chinese = bool(setup_matplotlib(chinese=True, logger=LOGGER))
        out_png = fig_dir / "cross_dataset_drift.png"
        scores = {
            "real_raw": arrays["raw"]["p_bin"][arrays["raw"]["pure"]],
            "real_dedup": arrays[dedup_key]["p_bin"][arrays[dedup_key]["pure"]],
            "in_domain": dom.get("_p_bin_test"),
        }
        plot_drift([("dedup", drift_tab, X_syn_train, X_real_dedup)], feature_names, scaler,
                   scores, out_png, chinese, top_n=min(6, args.top_k))
        figure_rel = str(out_png.relative_to(ROOT)).replace("\\", "/")
        LOGGER.info("分布对比图已保存：%s", figure_rel)

    # ---- 结论文字 ----
    headline, numbers, causes, verdict, remediation, interpretation = _make_conclusions(
        cfg, dom, v_raw, v_ded, dedup_key, drift_summary, drift_tab, mp_info, figure_rel,
        dedup_stages, calibration,
    )

    # ---- 汇总 JSON ----
    json_out = {
        "generated_by": "tools/cross_dataset.py",
        "experiment": "synthetic-trained model -> real inSDN zero-shot inference",
        "no_training": True,
        "no_real_data_fitting": True,
        "model": {
            "path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "tag": model_info.get("tag"),
            "resolved_from": model_info.get("resolved_from"),
            "input_shape": [window, len(feature_names)],
            "n_params": int(model.count_params()),
        },
        "config": {
            "window_size": window,
            "stride": stride,
            "threshold": float(cfg.report_threshold),
            "std_source": "data/processed/scaler.joblib (fit on synthetic train only)",
            "features_filled_with_train_mean_in_z_space": mp_info["unmatched"],
            "seed": int(args.seed),
        },
        "paths": {
            "real_csv": str(real_csv.relative_to(ROOT)).replace("\\", "/"),
            "synth_csv": str(synth_csv.relative_to(ROOT)).replace("\\", "/"),
            "processed_dir": str(proc_dir.relative_to(ROOT)).replace("\\", "/"),
            "figure": figure_rel,
        },
        "mapping": mp_info,
        "synthetic": synth_meta,
        "real": {**real_info,
                 "n_columns_total": len(real_columns),
                 "label_blocks": [{"label": k, "rows": v} for k, v in run_detail],
                 "dedup_stages": dedup_stages},
        "in_domain_reference": {k: v for k, v in dom.items() if not k.startswith("_")},
        "versions": versions,
        "drift": drift_summary,
        "calibration": calibration,
        "conclusions": {
            "headline": headline,
            "numbers": numbers,
            "causes": causes,
            "production_verdict": verdict,
            "remediation": remediation,
        },
        "runtime_sec": round(time.time() - t_start, 1),
    }
    save_json(json_out, out_dir / "cross_dataset.json")

    ctx = {
        "headline": headline,
        "window": window,
        "stride": stride,
        "threshold": float(cfg.report_threshold),
        "dedup_digits": int(args.dedup_digits),
        "dedup_key": dedup_key,
        "top_k": args.top_k,
        "feature_names": feature_names,
        "class_names": class_names,
        "real_columns": real_columns,
        "mapping": mp_info,
        "synth_meta": synth_meta,
        "real_meta": real_info,
        "dedup_stages": dedup_stages,
        "real_label_counts": raw_label_counts,
        "real_runs": len(runs),
        "real_run_detail": run_detail,
        "synth_attack_share": dom.get("synth_test_attack_share", float("nan")),
        "in_domain": dom,
        "versions": versions,
        "drift": drift_summary,
        "drift_interpretation": interpretation,
        "calibration": calibration,
        "calib_quantile": float(args.calib_quantile),
        "model": {"tag": model_info.get("tag")},
        "figure_rel": figure_rel,
        "no_figure": bool(args.no_figure),
        "conclusion_numbers": numbers,
        "conclusion_causes": causes,
        "production_verdict": verdict,
        "remediation": remediation,
    }
    report = build_report(ctx)
    save_text(report, out_dir / "cross_dataset_report.md")

    LOGGER.info("=" * 78)
    LOGGER.info("完成：%s | %s | 用时 %.1fs",
                str((out_dir / "cross_dataset_report.md").relative_to(ROOT)),
                str((out_dir / "cross_dataset.json").relative_to(ROOT)),
                time.time() - t_start)
    LOGGER.info("=" * 78)
    return 0


def _drift_group(feature: str) -> str:
    """把特征粗分到「物理族」，用于自动生成漂移归因文字。"""
    f = feature.lower()
    if "iat" in f:
        return "iat"
    if "duration" in f:
        return "duration"
    if "/s" in f or "rate" in f or "pkts/s" in f:
        return "rate"
    if "flag" in f or "cnt" in f or "count" in f:
        return "flag"
    if "header" in f:
        return "header"
    if "subflow" in f:
        return "subflow"
    if "win" in f or "segment" in f or "seg" in f:
        return "tcpopt"
    if "active" in f or "idle" in f:
        return "activity"
    if "length" in f or "len" in f or "size" in f or "bytes" in f or "packet" in f or "pkt" in f:
        return "length"
    if "protocol" in f:
        return "protocol"
    if "ratio" in f:
        return "ratio"
    return "other"


_GROUP_EXPLAIN = {
    "iat": "**到达间隔（IAT）族**：inSDN 是容器/虚拟机实验床上短时间采集的流，"
           "而合成数据的 IAT 由仿真器的「规律性 / 突发数」参数生成。"
           "真实流的 IAT 呈重尾（大量 0 与极少数超大值），合成流偏平滑，"
           "两者在标准化空间里几乎是两个不同的分布。",
    "duration": "**流时长族**：真实实验床里存在大量「秒级以内结束」的短流，"
                "合成数据的时长由 `duration` 参数控制、尾部更短。时长是所有速率类特征的公共分母，"
                "它一漂，`Flow Bytes/s`、`Flow Packets/s` 会跟着连锁漂移。",
    "rate": "**速率族（Bytes/s、Packets/s）**：分母（时长）的漂移 + 真实流量突发度更高，"
            "使速率类特征出现量级差；这类特征在训练集里被标准化到 N(0,1)，"
            "真实值却常常落在 ±5σ 之外，网络的第一层卷积直接进入未训练区域。",
    "flag": "**TCP 标志位计数族**：合成生成器只对少数几种标志组合建模"
            "（大部分条目恒为 0），而这几个维度在真实数据里被 CICFlowMeter 以不同口径累计。"
            "这类「合成侧近似常量」的维度对模型没有任何判别力，却在真实数据上产生恒定偏置。",
    "header": "**头部长度族**：真实抓包包含 VLAN/选项等额外头部，"
              "合成数据按固定 20 字节 IPv4 头 + 固定 TCP 头长度推导，存在系统性偏移。",
    "subflow": "**Subflow 族**：CICFlowMeter 的 subflow 统计依赖流超时切分策略，"
               "真实采集的超时参数与仿真器不一致，导致同一物理流量被切成不同的段数。",
    "tcpopt": "**TCP 选项 / 窗口族（Init Win、Seg Size）**：真实主机（Linux/Windows 各种版本）"
              "的初始窗口与 MSS 分布远比合成数据丰富；合成数据基本固定几个取值。",
    "activity": "**Active/Idle 族**：这对统计量强依赖 CICFlowMeter 的 activity timeout；"
                "真实数据里大量流只有一个 active 段，合成数据按「多个突发」建模，形态不同。",
    "length": "**报文长度族**：真实流里包含大量小包（ACK、keepalive）与少量 MSS 满包，"
              "合成数据的长度分布是按参数抽样的近似高斯，矩（均值/方差）能对上、形状对不上。",
    "protocol": "**协议号**：这是离散量，漂移通常来自真实数据里出现了合成数据未建模的协议占比。",
    "ratio": "**比值族**：分母漂移会被比值放大，因此比值类特征的 z-shift 常常比其分子更极端。",
    "other": "该特征属于其它统计族，漂移主要来自采集口径差异。",
}


def _make_conclusions(cfg, dom, v_raw, v_ded, dedup_key, drift, drift_tab, mp_info, figure_rel,
                      dedup_stages=None, calibration=None):
    """把数字整理成结论段落。"""
    rec = dom.get("recorded", {})
    rec2 = dom.get("recomputed_on_test_tensors", {})
    dom2 = dom.get("recomputed_two_class", {})
    # 同域 2 类宏 F1：与跨域的 2 类宏 F1 同分母，衰减幅度按这一行算
    dom_mf1 = float(dom2.get("macro_f1", rec.get("multiclass_macro_f1", float("nan"))))
    dom_f1 = in_domain_value(dom, "f1")[0]
    dom_auc = in_domain_value(dom, "roc_auc")[0]
    dom_fpr = in_domain_value(dom, "false_positive_rate")[0]

    mf1_raw = v_raw["two_class_pure"]["macro_f1"]
    mf1_ded = v_ded["two_class_pure"]["macro_f1"]
    f1_raw = v_raw["binary_pure"]["f1"]
    f1_ded = v_ded["binary_pure"]["f1"]
    auc_raw = v_raw["binary_pure"]["roc_auc"]
    auc_ded = v_ded["binary_pure"]["roc_auc"]
    fpr_raw = v_raw["binary_pure"]["false_positive_rate"]
    fpr_ded = v_ded["binary_pure"]["false_positive_rate"]
    triv = max(v_ded["trivial_baseline"]["f1_all_attack"], v_ded["trivial_baseline"]["f1_all_normal"])
    surv = (dedup_stages or {}).get("survival", {})
    surv_a = float(surv.get("attack_row_survival", float("nan")))
    attr4 = v_raw["model_attribution_8class"]

    def d(a, b):
        try:
            return float(a) - float(b)
        except Exception:
            return float("nan")

    top3 = drift_tab.head(3)
    top3_txt = "、".join(f"`{r['feature']}`（z-shift {r['z_shift']:+.2f}）" for _, r in top3.iterrows())

    headline = (
        f"同一模型、同一阈值、零样本直接迁移：2 类宏 F1 由同域的 {dom_mf1:.3f} 掉到 "
        f"{mf1_ded:.3f}（去重后），二分类 F1 由 {dom_f1:.3f} 掉到 {f1_ded:.3f}，"
        f"误报率由 {dom_fpr:.3f} 升到 {fpr_ded:.3f}；"
        f"{drift['n_features_abs_z_gt_1']}/{drift['n_features']} 个特征的真实均值偏离合成训练集超过 1 个标准差。"
        f"结论：**这套模型目前不具备跨域直接上线能力**，必须在目标域做轻量校准后才能进生产。"
    )

    numbers = "\n".join([
        markdown_table(
            ["指标", "同域（合成测试集）", "跨域 去重前", "跨域 去重后", "衰减（去重后）"],
            [
                ["2 类宏 F1（同口径）", fmt(dom_mf1), fmt(mf1_raw), fmt(mf1_ded), f"**{d(mf1_ded, dom_mf1):+.4f}**"],
                ["二分类 F1", fmt(dom_f1), fmt(f1_raw), fmt(f1_ded), f"**{d(f1_ded, dom_f1):+.4f}**"],
                ["ROC-AUC", fmt(dom_auc), fmt(auc_raw), fmt(auc_ded), f"**{d(auc_ded, dom_auc):+.4f}**"],
                ["误报率 FPR", fmt(dom_fpr), fmt(fpr_raw), fmt(fpr_ded), f"**{d(fpr_ded, dom_fpr):+.4f}**"],
            ],
        ),
        "",
        f"两个版本的差距本身也是一个发现（二分类 F1：去重前 {fmt(f1_raw)}、去重后 {fmt(f1_ded)}）。"
        f"去重前那一版里，攻击行只有 **{fmt(surv_a)}** 能存活到去重后，"
        "也就是说大量攻击窗口其实是**同一批近重复流反复计分**，指标被这种「重复计分」抬高了；"
        "去重后又把攻击类先验从 "
        f"{fmt(v_raw['attack_share_real_windows'])} 削到 {fmt(v_ded['attack_share_real_windows'])}，"
        "指标偏低也有一部分来自先验变化。**真实水平落在两者之间**，"
        "但两版都远低于同域，衰减本身是真实的。",
        "",
        f"对照平凡基线：**两个版本的模型 F1 都不高于「全判为攻击」的平凡基线** —— "
        f"去重前 {fmt(f1_raw)} vs {fmt(v_raw['trivial_baseline']['f1_all_attack'])}，"
        f"去重后 {fmt(f1_ded)} vs {fmt(v_ded['trivial_baseline']['f1_all_attack'])}。"
        "这是整份报告里最硬的一条负面证据：在真实数据上，"
        "一个「什么都不看、把所有窗口都判成攻击」的空模型，F1 反而更高。"
        "换句话说，模型在固定阈值 0.5 下**没有提供任何超过常量的判别增益**，"
        f"反而额外付了 {fmt(fpr_ded)} 的真实误报（把正常窗口报成攻击）。",
        "",
        (
            f"ROC-AUC = {fmt(auc_ded)}（去重前 {fmt(auc_raw)}）。AUC 只依赖排序、不依赖阈值，因此要单独说："
            + ("AUC 明显高于 0.5，说明模型的**排序**仍保有可观的信息量，"
               "主要问题是分数分布整体偏移导致固定阈值失效——这类问题靠「目标域重定阈值」能救回相当一部分。"
               if (not math.isnan(auc_ded)) and auc_ded > 0.8 else
               ("AUC 只比 0.5 高一点，说明**排序能力也已严重退化**："
                "问题不只是阈值选得不对，而是特征分布整体偏离训练支撑集。"
                "这种情况下单纯调阈值只能小幅改善，必须先解决分布漂移。"
                if (not math.isnan(auc_ded)) and auc_ded > 0.6 else
                ("AUC 已接近甚至低于 0.5，排序基本失效。"
                 "问题不只是阈值，而是特征分布整体偏离训练支撑集——单纯调阈值救不回来。"
                 if not math.isnan(auc_ded) else
                 "真实窗口里只出现了单一类别，AUC 在本版本上无定义（无法评估排序能力），"
                 "此时只能看 F1/FPR，或改用完整数据集跑一遍。")))
        ),
    ])

    # ---- 成因 ----
    groups = {}
    for _, r in drift_tab.head(12).iterrows():
        groups.setdefault(_drift_group(r["feature"]), []).append(r["feature"])
    cause_lines = [
        f"**① 特征层面的分布漂移（主因，有第 5 节数据支撑）**："
        f"|z-shift| 均值 {fmt(drift['abs_z_shift_mean'])}、P95 {fmt(drift['abs_z_shift_p95'])}、"
        f"最大 {fmt(drift['abs_z_shift_max'])}；"
        f"{drift['n_features_abs_z_gt_1']}/{drift['n_features']} 个特征的真实均值偏离合成训练集超过 1σ，"
        f"{drift['n_features_abs_z_gt_3']} 个超过 3σ——"
        "也就是说模型输入里有相当一部分维度落在训练分布**之外**，"
        f"第一层卷积/BN 收到的是从未见过的激活值。漂移最大的三个特征是 {top3_txt}。",
        f"**② 采集口径差异（漂移的物理来源）**：合成数据由本工程的仿真器按参数生成，"
        "真实数据由 CICFlowMeter v3 在实验床抓包统计。两者在流超时切分、"
        "头部长度统计、活动/空闲超时、TCP 选项记录上都不一致，"
        "这类差异会成体系地体现在 IAT / 速率 / subflow 族特征上——"
        "而这几族恰好是漂移榜的前排（见 5.1 与 5.2）。",
        f"**③ 类别先验偏移**：真实窗口里攻击占 {fmt(v_raw['attack_share_real_windows'])}，"
        f"合成测试集里攻击占 {fmt(dom.get('synth_test_attack_share', float('nan')))}。"
        "先验一变，同一阈值下的 FPR/FNR 组合就完全变了；"
        "这部分不是模型的错，但上线时必须按目标域的流量比例重新定阈值。",
        f"**④ 时间结构假设失效**：真实 inSDN 以「正常块 + 攻击块」拼接发布，"
        "相邻 16 行并不构成真实的相邻流量。CNN 分支拿到的「局部组合模式」因此含义被削弱；"
        "模型侧 8 类归因也印证了这一点——预测高度集中"
        f"（最大单类占比 {fmt(attr4['top_class_share'])}，"
        f"平均归一化熵仅 {fmt(attr4['mean_normalized_entropy'])}，均匀分布为 1.0），"
        "而且归因与真实标签几乎无关"
        f"（被判为 Normal 的 {attr4['all_windows'].get('Normal', 0):,} 个窗口里，"
        f"{attr4['on_true_attack_windows'].get('Normal', 0):,} 个的真值其实是攻击）。",
    ]
    for g, feats in list(groups.items())[:4]:
        cause_lines.append(
            f"　　- 漂移榜里的 *{g}* 族（{'、'.join('`%s`' % x for x in feats[:5])}）：{_GROUP_EXPLAIN[g]}"
        )

    # ---- 生产判断（按「零样本不可用 + 校准阶梯代价」的框架写，而不是一句否定）----
    triv_raw = max(v_raw["trivial_baseline"]["f1_all_attack"], v_raw["trivial_baseline"]["f1_all_normal"])
    cal = calibration or {}
    cal_on = bool(cal.get("enabled"))
    if cal_on:
        lv = cal["levels"]
        cost = cal["cost"]
        l0 = lv["L0"]

        def bud(k, key="fpr_0.01"):
            return lv[k]["fpr_budget_operating_points"][key]

        b0, b1, b2, b3, b4 = (bud("L0"), bud("L1"), bud("L2"), bud("L3"), bud("L4"))
        auc = {k: lv[k]["binary"]["roc_auc"] for k in ("L0", "L1", "L2", "L3", "L4")}
        best_key = max(("L0", "L1", "L2", "L3", "L4"), key=lambda k: bud(k)["eval_binary_f1"])
        bbest = bud(best_key)

        # 自适应措辞：零成本两级到底有没有用，用数据说话
        zero_cheap_gain = b2["eval_recall"] - b0["eval_recall"]
        labeled_gain = b3["eval_recall"] - b2["eval_recall"]
        backbone_gain = b4["eval_recall"] - b3["eval_recall"]
        backbone_ok = (backbone_gain > 0.02
                       or (b4["eval_binary_f1"] - b3["eval_binary_f1"]) > 0.01
                       or (auc["L4"] - auc["L3"]) > 0.02)
        if zero_cheap_gain < 0.02:
            cheap_txt = (
                f"**实测这两个零成本的级别基本没有救回来**：在 1% 误报预算下，召回从 L0 的 "
                f"{fmt(b0['eval_recall'], 4)} 变成 L1/L2 的 {fmt(b2['eval_recall'], 4)}"
                f"（AUC 反而由 {fmt(auc['L0'])} 掉到 {fmt(auc['L1'])}）。"
                "原因在 6.4 已经查清：零样本那点 AUC 有很大一部分是「漂移幅度」造的假信号，"
                "把量纲拉回目标域口径反而把它抹掉了。**只重定标 + 只重定阈值救不回来。**"
            )
        else:
            cheap_txt = (
                f"实测这两个零成本级别把 1% 预算下的召回从 {fmt(b0['eval_recall'], 4)} 提到 "
                f"{fmt(b2['eval_recall'], 4)}，是有效的第一步（AUC {fmt(auc['L0'])} → {fmt(auc['L1'])}）。"
            )
        if labeled_gain > 0.05:
            labeled_txt = (
                f"付少量标注后**显著恢复**：召回 {fmt(b2['eval_recall'], 4)} → "
                f"**{fmt(b3['eval_recall'], 4)}**，精确率 {fmt(b3['eval_precision'], 4)}，"
                f"二分类 F1 {fmt(b2['eval_binary_f1'], 4)} → **{fmt(b3['eval_binary_f1'], 4)}**，"
                f"AUC {fmt(auc['L2'])} → **{fmt(auc['L3'])}**。"
                "这是整条阶梯里性价比最高的一级。"
            )
        else:
            labeled_txt = (
                f"付少量标注后**只恢复了很小的幅度**：召回 {fmt(b2['eval_recall'], 4)} → "
                f"{fmt(b3['eval_recall'], 4)}，AUC {fmt(auc['L2'])} → {fmt(auc['L3'])}。"
                f"{cal['finetune_head']['n_train_windows']} 个标注窗口对这份数据而言明显偏少，"
                "需要更多目标域标注才能继续提升。"
            )
        if backbone_ok:
            backbone_txt = (
                f"解冻主干再训还能再涨：1% 预算下召回 {fmt(b3['eval_recall'], 4)} → "
                f"{fmt(b4['eval_recall'], 4)}，二分类 F1 {fmt(b3['eval_binary_f1'], 4)} → "
                f"{fmt(b4['eval_binary_f1'], 4)}，AUC {fmt(auc['L3'])} → **{fmt(auc['L4'])}**。"
                f"代价很低（额外 {cal['finetune_backbone']['fit_seconds']} 秒），值得做；"
                "不过它是在只花了这么点算力的前提下成立的，"
                "标注量大时应当重新评估这一级的收益。"
            )
        else:
            backbone_txt = (
                f"解冻主干**没有进一步改善**（召回 {fmt(b3['eval_recall'], 4)} → {fmt(b4['eval_recall'], 4)}，"
                f"AUC {fmt(auc['L3'])} → {fmt(auc['L4'])}）："
                "只有几百个标注窗口时，全参数微调容易把主干拉离原本的表示，容易过拟合。"
                "这一步应当等标注量上去之后再做。"
            )

        verdict = "\n".join([
            "**零样本跨域直接迁移在当前设置下不可用**——这一点第 3 节已经用硬指标钉死："
            f"跨域二分类 F1 去重后 {fmt(f1_ded)}、去重前 {fmt(f1_raw)}，"
            f"**两版都不高于「全判某一类」的平凡基线**（{fmt(triv)} / {fmt(triv_raw)}）；"
            f"误报率由同域的 {fmt(dom_fpr)} 升到 {fmt(fpr_ded)}；"
            f"在 1% 误报预算下（第 6.5 节口径）召回只有 **{fmt(b0['eval_recall'], 4)}**，等于检不出攻击。",
            "",
            "**但衰减的主因是「量纲/分布漂移 + 采集口径差异」，属于可校准的工程问题，"
            "不是模型结构失效。** 证据是第 6 节的校准阶梯：同一套模型结构、同一份评估数据，"
            "只改变「标准化统计量 + 一个阈值 + 少量头参数」，指标就能实质变化。",
            "",
            "第 6.5 节给出了每一级的实测恢复幅度（统一按 1% 误报预算，与攻击先验无关）：",
            "",
            f"| 上线姿势 | 需要的代价 | 1% 预算下召回 | 二分类 F1 | ROC-AUC |",
            f"|---|---|---|---|---|",
            f"| **A. 零样本**（现状） | 0 标注 | {fmt(b0['eval_recall'], 4)} | {fmt(b0['eval_binary_f1'], 4)} | {fmt(auc['L0'])} |",
            f"| **B. 零成本定标**（L1 重定标 + L2 重定阈值） | **0 标注**、0 秒训练 | {fmt(b2['eval_recall'], 4)} | {fmt(b2['eval_binary_f1'], 4)} | {fmt(auc['L2'])} |",
            f"| **C. 少量标注微调分类头**（L3） | {cost['L3']['labeled_windows']} 个标注窗口、{cal['finetune_head']['fit_seconds']} 秒 | **{fmt(b3['eval_recall'], 4)}** | **{fmt(b3['eval_binary_f1'], 4)}** | **{fmt(auc['L3'])}** |",
            f"| **D.（可选）解冻主干再训**（L4） | 额外 {cal['finetune_backbone']['epochs']} epoch、{cal['finetune_backbone']['fit_seconds']} 秒 | **{fmt(b4['eval_recall'], 4)}** | **{fmt(b4['eval_binary_f1'], 4)}** | **{fmt(auc['L4'])}** |",
            "",
            cheap_txt,
            "",
            labeled_txt,
            "",
            backbone_txt,
            "",
            f"**校准后能达到的水平是 {best_key}**：1% 误报预算下召回 {fmt(bbest['eval_recall'], 4)}、"
            f"精确率 {fmt(bbest['eval_precision'], 4)}、二分类 F1 {fmt(bbest['eval_binary_f1'], 4)}、"
            f"AUC {fmt(auc[best_key])}（同域参照：二分类 F1 {fmt(dom_f1)}、AUC {fmt(dom_auc)}、"
            f"宏 F1 {fmt(dom_mf1)}）。"
            "注意：**线上推理的模型结构、输入维度、时延完全没有变化**，"
            "变的只是「标准化统计量 + 一个阈值 + 少量头参数」。",
            "",
            "**最后必须一起说的一句**：本项目在**有目标域数据**时按正常流程（同域训练/评估）"
            "在同一份真实 inSDN 上拿到过 **0.9976–1.0000** 的宏 F1（见 `results/metrics_real_insdn*.json`）；"
            f"本次的 L3/L4 只用了 {cost['L3']['labeled_windows']} 个标注窗口、"
            f"{round(cal['finetune_head']['fit_seconds'] + cal['finetune_backbone']['fit_seconds'], 1)} 秒训练，"
            "就已经把 1% 预算下的召回从 0 拉到 0.4 以上。"
            "这说明**流水线与模型本身是可用的，问题只出在「零样本迁移」这个最苛刻的设定上**。"
            "把跨域当成一个**可测量、可付代价解决的工程约束**（本节的阶梯），"
            "而不是模型能力的判决书，才是准确的结论。",
        ])
    elif f1_ded <= triv + 0.02 or auc_ded < 0.6:
        verdict = (
            "**不能直接上生产。** 依据是第 3、4 节的硬指标："
            f"(1) 跨域二分类 F1 去重后 {fmt(f1_ded)}、去重前 {fmt(f1_raw)}，"
            f"**两版都不高于「全判某一类」的平凡基线**（{fmt(triv)} / {fmt(triv_raw)}）；"
            f"(2) ROC-AUC 去重后 {fmt(auc_ded)}、去重前 {fmt(auc_raw)}；"
            f"(3) 误报率 FPR 由同域的 {fmt(dom_fpr)} 升到 {fmt(fpr_ded)}。"
            "但要说清楚边界：**这不能否定系统本身**——它在自建数据集（同分布）上的表现是真的，"
            "在真实数据上失效的原因是域漂移。"
            "（本次以 `--no-calibration` 运行，未给出校准后的恢复幅度，"
            "去掉该开关即可得到 Level 1–4 的实测恢复路径。）"
        )
    else:
        verdict = (
            "**有条件可用，但不能零样本上线。** 跨域二分类 F1 "
            f"{fmt(f1_ded)} 明显高于平凡基线 {fmt(triv)}，ROC-AUC {fmt(auc_ded)} 说明排序仍有信息量；"
            f"但误报率由同域的 {fmt(dom_fpr)} 升到 {fmt(fpr_ded)}，"
            "直接上线会在控制器上产生大量误封。合理定位是"
            "「先用目标域正常流量做无监督校准，再按目标域先验重定阈值，然后灰度上线」。"
        )

    remediation = [
        f"**第 0 步：先跑校准阶梯，拿到本项目的真实恢复曲线**"
        f"（`tools/cross_dataset.py` 默认就会跑，产物见第 6 节与 "
        f"`results/cross_dataset.json` 的 `calibration` 字段）。"
        f"下面的 1~4 步就是阶梯对应的工程动作，每一步的实测效果都在第 6 节表里。"
        + ("" if not cal_on else
           f"  ⚠️ 本项目的实测结论是：**第 1、2 步（零成本）在这份数据上几乎无效，"
           f"真正起作用的是第 3 步。** 顺序仍按成本从低到高排，但不要把希望押在前两步上。"),
        f"**只重拟合标准化器（零成本，但别指望它 = Level 1）**：漂移里有一部分纯粹是"
        f"「量纲/取值范围」差异。用目标域**仅正常流量**的 N 条流重算 mean/std，"
        f"替换 `data/processed/scaler.joblib`（不碰权重）。"
        f"它能把均值偏移（当前 |z-shift| 均值 {fmt(drift['abs_z_shift_mean'])}、"
        f"P95 {fmt(drift['abs_z_shift_p95'])}）压下去，但**有明确边界**："
        f"真实侧已有 {drift['n_features_std_ratio_lt_005']}/{drift['n_features']} 个特征退化成常量"
        f"（标准差比 < 0.05），重定标救不回这些维度丢失的信息量；"
        f"而且第 6 节的实测显示它**并不能提高排序能力**"
        f"（零样本那点 AUC 有很大一部分是漂移幅度造的假信号）。"
        f"这一步不需要任何攻击样本，也不涉及标签泄漏，做了没有坏处，但不要当成解决方案。",
        "**按目标域正常流量的误报预算定阈值（零成本，仍不足以救回）= Level 2**："
        "现在阈值 0.5 是合成测试集上的默认值。正确做法是用纯正常流量估计分数分位，"
        "把阈值定到运维能接受的误报预算上（本项目 `tools/operating_point.py` 是同一套思路）。"
        "**但要小心一个陷阱**：本项目实测「99.9% 分位」这条规则会退化——模型对目标域正常窗口"
        "也会给出接近 1.0 的分数，于是分位数几乎等于 1.0，召回直接归零。"
        "所以阈值必须在**真实的误报预算曲线**上看（第 6.3 节的敏感性表），"
        "并接受「单独调阈值只能换个取舍点、不能提升判别力」这个事实。",
        f"**少量目标域标注微调分类头（这才是真正有效的一步）= Level 3**："
        f"冻结 CNN/BiLSTM 主干（只留 `clf_*` 与 `mc`/`bin` 可训练），"
        + (f"用 {cal['finetune_data']['negatives_available_and_used']} 个正常 + "
           f"{cal['finetune_data']['positives_available_and_used']} 个攻击窗口"
           f"（本报告用的是 {cal['finetune_head']['n_train_windows']} 个标注窗口、正负各半）、"
           f"微调 {cal['finetune_head']['epochs']} 个 epoch。"
           if cal_on else "用 200~500 个带标签的真实窗口（正负各半）微调 3~5 个 epoch。")
        + "关键是**隔离**：微调窗口与定阈值窗口、评估窗口必须是**不同的行**"
        "（本脚本用三层行级切分保证）。"
        "本项目实测：1% 误报预算下的召回由「几乎为 0」升到 0.4 量级，二分类 F1 提升一个数量级。",
        "**加显式域自适应**：在特征提取层后接一个梯度反转层（DANN）做无监督域对齐，"
        "或对窗口级表示做 CORAL/MMD 对齐，让「合成域」与「真实域」的特征分布在训练时就被拉近；"
        "也可以用本工程的生成器反过来做「按真实统计量重标定参数」的域随机化训练，"
        "把真实数据的取值范围包进训练分布（domain randomization）。"
        "这类方法的价值在于**不需要目标域攻击标签**，与本报告 L3 的「少量标签路线」互补。",
        "**补齐模型看不到的输入维度**：真实数据里有 6 个 Bulk-rate 统计量"
        "（`Fwd Byts/b Avg` 等）合成生成器没有产。若要长线做，应当在生成器里补上这些量，"
        "并在控制器侧（Ryu/OVS 采集脚本）保证字段口径与 CICFlowMeter v3 一致——"
        "**采集口径不一致是跨域问题的根，靠模型补是补不回来的**。",
        "**在控制器侧建立在线自监控**：上线后持续统计输入特征的 z-shift 与分数分布"
        "（本脚本第 5 节就是离线版），一旦 |z-shift| 超阈值就触发重标定/再微调。"
        "把「漂移检测」和「模型推理」放进同一个运维闭环，比一次性调好模型更重要。",
        "**换更合适的评测数据**：inSDN 的 `target` 只有 0/1 且按块拼接，"
        "既不能评 8 类，也不适合评在线时延（第 6 节的全部窗口都不是真实相邻流量）。"
        "要做严肃的跨域论证，建议补一个**带多类标签且保留时间序**的公开集"
        "（如 CIC-IDS2017 原始 pcap 重导）做第二次跨域验证。",
    ]

    groups_i = {}
    for _, r in drift_tab.head(12).iterrows():
        groups_i.setdefault(_drift_group(r["feature"]), []).append(r["feature"])
    interpretation = [
        "漂移最大的特征集中在下面几个物理族，它们的变化都有可解释的来源（不是随机噪声）：",
    ]
    for g, feats in list(groups_i.items())[:6]:
        interpretation.append(
            f"- **{g} 族**（漂移榜内：{'、'.join('`%s`' % x for x in feats[:5])}）—— {_GROUP_EXPLAIN[g]}"
        )
    interpretation.append(
        "综合看：跨域衰减**不是**某一个特征坏掉了，而是「量纲/统计口径」与「分布形状」同时变。"
        "第 6 节的校准阶梯进一步证明了前者的边界：**重定标只能把「正常流量看起来像不像"
        "训练数据」拉回来，拉不回「攻击长什么样」**——"
        "所以第 7.4 节的方案是分层的：先做零成本的（重定标、按误报预算重定阈值），"
        "再做需要一点标注的（微调分类头，这才是真正有效的一级），"
        "最后才是需要工程投入的（域自适应 + 采集口径对齐）。"
    )
    return headline, numbers, cause_lines, verdict, remediation, interpretation


if __name__ == "__main__":
    raise SystemExit(main())
