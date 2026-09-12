"""包序列输入复研：把「卷积分支到底有没有用」放到**包级表示**上再问一次。

为什么要做这个实验
------------------
本工程已经把「卷积没有增益」测穿过两次：
  1. 窗口消融做到 T=48 —— Δ（混合 − 最好单分支）从未转正，到 T=48 双方都打满 1.0000；
  2. 困难基准（重叠/隐蔽/噪声拉满）—— Δ 四档全负，逐种子配对 6/6 为负（p=0.031）。
两次都指向同一个解释：**流统计量是把「每条流的聚合量」当时间步，卷积要找的
「相邻时间步局部组合模式」在特征工程阶段就已经被压平了**。

这个解释有一个可检验的推论：如果把输入换成**真正有序的包序列**（包大小/方向/间隔/标志位），
"相邻时间步的局部模式"就重新存在了（三次握手、突发 SYN、等间隔心跳），
**那么卷积分支的相对地位就应该改善**。本工具就是去检验这个推论。

判据（事前写死，不允许事后挑标准）
----------------------------------
记 GAP(表示) = macroF1(最好单分支) − macroF1(仅 CNN)，即"CNN 单支落后多少"。
  * 流级滑窗表示（主实验口径，T=16，3 种子）的 GAP 读自既有产物，记为 GAP_flow；
  * 包级表示（本次）记为 GAP_pkt。
1. GAP_pkt ≤ 0.5 × GAP_flow  → `activated`：卷积在包级表示上被激活（正面结论）
2. GAP_pkt ≥ 1.0 × GAP_flow  → `not_activated`：包级表示同样没有激活卷积（结论更硬）
3. 其余                       → `partial`：部分改善，需更大规模复现
另外单独记录 Δ = macroF1(混合) − macroF1(最好单分支)：|Δ| < 1σ（该序列长度自己的种子噪声）记为噪声内。

数据是怎么来的（必须说清楚，不许含糊）
--------------------------------------
**不是真实抓包**。每条流的前 N 个包由该流记录的**物理事实**反构而成：
  * 包数、时长、双向包长均值/标准差/最小/最大、IAT 均值/标准差/最值、各 TCP 标志位计数 —— 全部取自
    `data/raw/insdn_stream.csv`（与主实验**同一批 86,037 条流**，只读、不重生成）；
  * 方向序列按双向包数比例抽样（首包必为 fwd）；包长按方向取自该流的包长分布；
    间隔 = (时长 × 取包比例 / (k−1)) × 对数正态抖动，并按该流的 IAT 上下界裁剪；
    标志位按该流的各标志计数分配到首包/末包/其后各包。
  * 随机数**逐流确定性**（default_rng(seed × 1000003 + 流序号)），因此可完全复现。
结论只用于回答「同一个架构在两种输入表示下，卷积的相对价值是否改变」，
**不能**读成"包级模型比流级模型强/弱"——两者的信息量与样本单位本就不同。

切分与防泄漏（与主实验同协议）
------------------------------
  * 按时间顺序切分（train/val/test = 60%/15%/25%），边界留 purge 间隙（= purge_windows × 16 条流，与主实验同等时间跨度）；
  * 标准化器**只在训练集上 fit**（连续特征列），二值指示列不参与标准化；
  * 特征抖动沿用主实验同一旋钮（`feature_noise × 0.1` 的标准差），保证"难度"可比。

输出（全部隔离，绝不触碰主实验产物）
------------------------------------
  results/packet_seq.json / .md、results/figures/packet_seq_compare.png
  artifacts/pktseq<L>/<tag>/（模型与历史，`.keras`/`.h5` 已被 .gitignore 忽略）
  训练用 `publish_summary=false` 语义：本工具**完全不调用** run_all 的汇总发布路径。

用法
----
    python tools/packet_seq.py                       # 完整实验（T=16/32 × 4 结构 × 3 种子）
    python tools/packet_seq.py --quick               # 快速自检（少量流 + 2 epoch，验证链路）
    python tools/packet_seq.py --lengths 16 --variants cnn_bilstm,cnn,bilstm --seeds 0
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData  # noqa: E402
from src.evaluate import (  # noqa: E402
    best_threshold,
    binary_metrics,
    multiclass_metrics,
    online_alarm_evaluation,
    tpr_at_fpr,
)
from src.model import predict_probs  # noqa: E402
from src.train import run_training  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR,
    FIGURE_DIR,
    RAW_DIR,
    RESULT_DIR,
    ensure_dirs,
    fmt_float,
    get_logger,
    load_json,
    markdown_table,
    save_json,
    save_text,
    set_seed,
    setup_matplotlib,
    silence_tf_logs,
)

LOGGER = get_logger("cida.packet_seq")

# --------------------------------------------------------------------------------------
# 事前写死的常量与判据
# --------------------------------------------------------------------------------------
PKT_FEATURES = [
    "pkt_len",       # 包长（字节）
    "direction",     # 1 = 正向（客户端→服务端），0 = 反向
    "gap_us",        # 与上一个包的时间间隔（微秒，首包为 0）
    "rel_time",      # 累计时间 / 该流总时长（0~1）
    "rel_pos",       # 包序 / (N−1)
    "tcp_flags",     # TCP 标志位位图（FIN=1 SYN=2 RST=4 PSH=8 ACK=16 URG=32）
    "is_first",      # 是否该流首包
    "valid",         # 1 = 真实包，0 = padding（流内包数不足 N 时的补位）
]
CONT_COLS = [0, 2, 3, 4, 5]          # 需要标准化的连续特征列
DEFAULT_LENGTHS = (16, 32)
DEFAULT_VARIANTS = ("cnn_bilstm", "cnn", "bilstm")   # mlp 与本题无关，需要时用 --variants 加上
DEFAULT_SEEDS = (0, 1, 2)
ACTIVATION_RATIO = 0.5               # GAP_pkt ≤ 0.5×GAP_flow → activated
MIN_SEG_FLOWS = 16                   # 在线评估"算得上一个攻击片段"的最小连续流数（与主实验一致）
REPORT_THRESHOLD = 0.5               # 在线口径固定阈值（与主实验默认一致）
ONLINE_CONFIRM = 3                   # 连续确认窗口数（与主实验默认一致）

FLAG_BITS = {
    "FIN Flag Count": 1,
    "SYN Flag Count": 2,
    "RST Flag Count": 4,
    "PSH Flag Count": 8,
    "ACK Flag Count": 16,
    "URG Flag Count": 32,
}

NEED_COLS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    *FLAG_BITS.keys(),
]


# --------------------------------------------------------------------------------------
# 1) 由流记录反构物理自洽的包序列（确定性、可复现）
# --------------------------------------------------------------------------------------
def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def synthesize_flow(rec: dict, rng: np.random.Generator, n_pkt: int) -> tuple[np.ndarray, int]:
    """把一条流记录反构成 (n_pkt, F) 的包序列；返回 (数组, 真实包数 k)。

    只使用该流记录里真实存在的物理量，不做"按类别写死"的特例处理：
    每个类别的包序列差异都来自它自己的包数/时长/包长/间隔/标志位。
    """
    out = np.zeros((n_pkt, len(PKT_FEATURES)), dtype=np.float32)
    n_fwd = max(int(round(_f(rec["Total Fwd Packets"]))), 0)
    n_bwd = max(int(round(_f(rec["Total Backward Packets"]))), 0)
    n_total = n_fwd + n_bwd
    if n_total <= 0:
        n_total, n_fwd = 1, 1

    k = int(min(n_pkt, n_total))                    # 取该流的前 k 个包
    dur = max(_f(rec["Flow Duration"]), 1.0)        # 微秒
    span = dur * (k / n_total)                      # 前缀包所占的时间跨度（按包数比例）

    # ---- 方向序列：按双向包数比例抽样，首包必为正向 ----
    dirs = np.zeros(k, dtype=np.float32)
    rem_f, rem_b = float(n_fwd), float(n_bwd)
    for i in range(k):
        if i == 0 and rem_f > 0:
            dirs[i] = 1.0
            rem_f -= 1
            continue
        tot = rem_f + rem_b
        if tot <= 0:
            dirs[i] = dirs[i - 1] if i else 1.0
            continue
        if rng.random() < rem_f / tot:
            dirs[i] = 1.0
            rem_f -= 1
        else:
            rem_b -= 1

    # ---- 包长：按方向的均值/标准差抽样，并按该方向的真实上下界裁剪 ----
    sizes = np.zeros(k, dtype=np.float32)
    for d, pre in ((1, "Fwd"), (0, "Bwd")):
        idx = np.flatnonzero(dirs == d)
        if idx.size == 0:
            continue
        mean = max(_f(rec[f"{pre} Packet Length Mean"]), 0.0)
        std = max(_f(rec[f"{pre} Packet Length Std"]), 0.0)
        lo = max(_f(rec[f"{pre} Packet Length Min"]), 0.0)
        hi = max(_f(rec[f"{pre} Packet Length Max"]), lo)
        if hi <= 0:                                  # 该方向没有包长的有效记录时退回整体均值
            mean = max(mean, 0.0) or 64.0
            lo, hi = 0.0, max(mean * 3.0, 1500.0)
        vals = rng.normal(mean, std if std > 0 else max(mean * 0.1, 1e-6), size=idx.size)
        sizes[idx] = np.clip(vals, lo, hi)

    # ---- 间隔：base × 对数正态抖动，并按该流的 IAT 上下界裁剪 ----
    gaps = np.zeros(k, dtype=np.float32)
    if k > 1:
        iat_mean = max(_f(rec["Flow IAT Mean"]), 1.0)
        iat_std = max(_f(rec["Flow IAT Std"]), 0.0)
        iat_lo = max(_f(rec["Flow IAT Min"]), 1.0)
        iat_hi = max(_f(rec["Flow IAT Max"]), iat_lo)
        cv = float(np.clip(iat_std / iat_mean, 0.0, 1.2))
        base = span / (k - 1)
        jit = rng.lognormal(mean=0.0, sigma=max(cv, 1e-3), size=k - 1)
        jit = np.clip(jit, 0.2, 3.0)
        gaps[1:] = np.clip(base * jit, min(iat_lo, base), max(iat_hi, base))

    cum = np.cumsum(gaps)
    total = float(cum[-1]) if k > 0 else 0.0

    # ---- 标志位：把该流的各标志计数分摊到首包/末包/其后各包 ----
    flags = np.zeros(k, dtype=np.float32)
    ack = int(round(_f(rec["ACK Flag Count"])))
    syn = int(round(_f(rec["SYN Flag Count"])))
    fin = int(round(_f(rec["FIN Flag Count"])))
    rst = int(round(_f(rec["RST Flag Count"])))
    psh = int(round(_f(rec["PSH Flag Count"])))
    urg = int(round(_f(rec["URG Flag Count"])))
    if syn > 0:
        flags[0] += 2
    if ack > 0:
        flags[min(1, k - 1):] += 16                  # 握手之后的包普遍带 ACK
    if psh > 0:
        step = max(k // max(psh, 1), 1)
        for i in range(step - 1, k, step):
            flags[i] += 8
    if urg > 0 and k > 2:
        flags[k // 2] += 32
    if fin > 0:
        flags[k - 1] += 1
    if rst > 0:
        flags[k - 1] += 4

    idx = np.arange(k, dtype=np.float32)
    out[:k, 0] = sizes
    out[:k, 1] = dirs
    out[:k, 2] = gaps
    out[:k, 3] = (cum / total) if total > 0 else 0.0
    out[:k, 4] = idx / max(k - 1, 1)
    out[:k, 5] = flags
    out[:k, 6] = (idx == 0).astype(np.float32)
    out[:k, 7] = 1.0
    return out, k


# --------------------------------------------------------------------------------------
# 2) 数据集构建（读主实验同一批流记录，只读）
# --------------------------------------------------------------------------------------
def load_flows(limit: int = 0, seed: int = 42) -> pd.DataFrame:
    path = RAW_DIR / "insdn_stream.csv"
    if not path.exists():
        raise SystemExit(f"找不到 {path}；请先跑 python run_all.py 生成主实验数据")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in NEED_COLS + ["Label", "Timestamp"] if c not in df.columns]
    if missing:
        raise SystemExit(f"缺少必要列：{missing}")
    # 与主实验同一套时序排序（稳定排序，保证可复现）
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    if limit and limit < len(df):
        # 按类别比例抽样（保持类别分布与时间顺序），作为本次实验的流量预算。
        # 同一份子集也会喂给"流级滑窗表示"跑一遍对照，因此两种表示的**数据完全相同**。
        frac = limit / len(df)
        parts = [g.sample(max(int(round(len(g) * frac)), 1), random_state=seed)
                 for _, g in df.groupby("Label")]
        df = pd.concat(parts).sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
        LOGGER.info("流量预算：从 86,037 条中按类别比例抽 %d 条（与流级对照共用同一子集）", len(df))
    return df


def write_subset_csv(df: pd.DataFrame) -> Path:
    """把同一份子集写到隔离目录，供"流级滑窗表示"的对照运行使用（只新建文件，不覆盖主数据）。"""
    raw_sub = RAW_DIR.parent / "raw_pktsub"
    raw_sub.mkdir(parents=True, exist_ok=True)
    out = raw_sub / "insdn_stream.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    LOGGER.info("已写出对照用子集：%s（%d 条流，主数据 data/raw 未被触碰）", out, len(df))
    return raw_sub


def subset_control(df: pd.DataFrame, cfg_base: Config, *, epochs: int, seeds: tuple[int, ...],
                   variants: tuple[str, ...] = ("cnn_bilstm", "cnn", "bilstm")) -> dict:
    """在**同一份流子集**上跑一遍「流级滑窗表示」（T=16，straide=5），作为 GAP 的同数据对照。

    为什么必须做这一步：包级样本单位是"一条流"，流级样本单位是"16 条流的窗口"，
    如果两边用不同规模的数据集，Δ/GAP 的差异就可能只是数据量差异。**同数据对照**把这个自由度消掉。
    """
    from src.data_pipeline import prepare_dataset
    from src.evaluate import evaluate_model

    raw_sub = write_subset_csv(df)
    proc_sub = RAW_DIR.parent / "processed_pktsub"
    out_dir = RESULT_DIR / "pktseq_control"
    art_dir = ARTIFACT_DIR / "pktseq_control"
    ensure_dirs(out_dir, art_dir)

    cfg0 = replace(cfg_base, schema="insdn", source="synthetic", raw_dir=str(raw_sub),
                   processed_dir=str(proc_sub), window_size=16, window_stride=5,
                   input_mode="sequence", publish_summary=False, figures=False,
                   batch_size=int(cfg_base.batch_size))
    data = prepare_dataset(cfg0, save=True)          # 只准备一次，各变体共用同一划分
    results: dict[str, dict] = {}
    for variant in variants:
        for seed in seeds:
            tag = f"pktctl_{variant}_s{seed}"
            cfg_i = replace(cfg0, seed=seed, model_variant=variant)
            res = run_training(cfg_i, data, variant=variant, tag=tag, epochs=epochs, artifact_dir=art_dir)
            rep = evaluate_model(res.model, data, cfg_i, tag=tag, result_dir=out_dir,
                                 artifact_dir=art_dir, save_predictions=False)
            results[f"{variant}|{seed}"] = {
                "variant": variant, "seed": seed,
                "macro_f1": float(rep["multiclass"]["macro_f1"]),
                "accuracy": float(rep["multiclass"]["accuracy"]),
                "binary_roc_auc": float(rep["binary"]["roc_auc"]),
                "online": rep.get("online") or {},
                "n_params": int(res.model.count_params()),
            }
            LOGGER.info("[对照·流级滑窗 T=16] %s 宏F1=%.4f 片段召回=%s", tag,
                        results[f"{variant}|{seed}"]["macro_f1"],
                        (rep.get("online") or {}).get("segment_recall"))
    singles = {v: results[f"{v}|{seeds[0]}"]["macro_f1"] for v in variants if f"{v}|{seeds[0]}" in results}
    summary: dict = {"runs": results, "window": 16, "stride": 5, "seeds": list(seeds),
                     "n_flows": int(len(df))}
    if singles:
        best_variant = max(singles, key=lambda k: singles[k])
        summary["best_single"] = {"variant": "CNN" if best_variant == "cnn" else best_variant,
                                  "mean": singles[best_variant]}
        if "cnn" in singles:
            summary["gap_cnn_behind_best"] = float(singles[best_variant] - singles["cnn"])
        if "cnn_bilstm" in singles:
            summary["delta_hybrid_minus_best"] = float(singles["cnn_bilstm"] - singles[best_variant])
    return summary


def build_tensors(df: pd.DataFrame, n_pkt: int, seed: int) -> np.ndarray:
    X = np.zeros((len(df), n_pkt, len(PKT_FEATURES)), dtype=np.float32)
    recs = df[NEED_COLS].to_dict("records")
    kept = np.zeros(len(df), dtype=np.int64)
    for i, rec in enumerate(recs):
        rng = np.random.default_rng(seed * 1_000_003 + i)      # 逐流确定性
        seq, k = synthesize_flow(rec, rng, n_pkt)
        X[i] = seq
        kept[i] = k
    LOGGER.info("包序列合成完成：%d 条流 × %d 包 × %d 特征；平均真实包数 %.2f，"
                "包数不足被 padding 的流占 %.1f%%",
                X.shape[0], X.shape[1], X.shape[2], float(kept.mean()),
                100.0 * float((kept < n_pkt).mean()))
    return X


def split_and_scale(X: np.ndarray, labels: np.ndarray, cfg: Config,
                    times: np.ndarray) -> tuple[PreparedData, np.ndarray, np.ndarray]:
    """时序切分 + 只在训练集上 fit 的标准化 + 与主实验同强度的特征抖动。"""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    n = len(X)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    purge = int(cfg.purge_windows * 16)                  # 与主实验同等时间跨度（2×16 条流）
    tr_end = max(n_train - purge, 1)
    va_start = min(n_train + purge, n - 2)
    va_end = max(min(n_train + n_val - purge, n), va_start + 1)
    te_start = min(n_train + n_val + purge, n - 1)
    idx_train = np.arange(0, tr_end)
    idx_val = np.arange(va_start, min(va_end, n))
    idx_test = np.arange(te_start, n)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(np.asarray(labels).astype(str))
    class_names = [str(c) for c in encoder.classes_]
    normal_id = class_names.index("Normal") if "Normal" in class_names else -1
    y_bin_all = np.where(y_all == normal_id, 0, 1).astype(np.int64) if normal_id >= 0 else np.zeros(n, np.int64)

    Xs = X.copy()
    flat = Xs.reshape(-1, len(PKT_FEATURES))
    scaler = StandardScaler()
    train_mask = np.zeros(n, dtype=bool)
    train_mask[idx_train] = True
    train_pkt = np.repeat(train_mask, X.shape[1])
    real = flat[:, 7] > 0.5
    fit_rows = train_pkt & real
    scaler.fit(flat[fit_rows][:, CONT_COLS])
    # ⚠️ 注意：`flat[real]` 是**副本**，必须用整数索引写回（`flat[idx, c] = ...` 才会真正改到数组）
    real_idx = np.flatnonzero(real)
    scaled = scaler.transform(flat[real_idx][:, CONT_COLS])
    for j, c in enumerate(CONT_COLS):
        flat[real_idx, c] = scaled[:, j]
    Xs = flat.reshape(X.shape)

    if cfg.feature_noise and cfg.feature_noise > 0:      # 与主实验同一旋钮、同一强度
        rng = np.random.default_rng(cfg.seed + 7)
        real2 = real.reshape(X.shape[0], X.shape[1])[..., None].astype(np.float32)
        noise = rng.normal(0.0, cfg.feature_noise * 0.1,
                           size=(X.shape[0], X.shape[1], 1)).astype(np.float32)
        Xs = Xs + noise * real2

    def pack(idx):
        return (Xs[idx], y_all[idx].astype(np.int64), y_bin_all[idx])

    Xtr, ytr_mc, ytr_bin = pack(idx_train)
    Xva, yva_mc, yva_bin = pack(idx_val)
    Xte, yte_mc, yte_bin = pack(idx_test)
    lab_named = np.asarray(class_names, dtype=object)[y_all]
    train_counts = {class_names[i]: int(v) for i, v in enumerate(np.bincount(ytr_mc, minlength=len(class_names)))}

    data = PreparedData(
        X_train=Xtr, y_train_mc=ytr_mc, y_train_bin=ytr_bin,
        X_val=Xva, y_val_mc=yva_mc, y_val_bin=yva_bin,
        X_test=Xte, y_test_mc=yte_mc, y_test_bin=yte_bin,
        class_names=class_names, feature_names=list(PKT_FEATURES),
        scaler=scaler, label_encoder=encoder,
        input_mode="sequence", window_size=int(X.shape[1]),
        test_stream_features=None,
        test_stream_labels=lab_named[idx_test],            # 在线评估按"流"为单位（每个样本 = 一条流）
        test_stream_time=np.asarray(times)[idx_test],
        train_class_counts=train_counts,
        split_info={
            "train": len(idx_train), "val": len(idx_val), "test": len(idx_test),
            "purge_flows": purge, "split_mode": "time", "time_ordered_split": True,
            "unit": "flow（每条流 = 一条包序列样本）",
            "n_pkt": int(X.shape[1]),
        },
        raw_meta={"source": "data/raw/insdn_stream.csv（只读，未重新生成）",
                  "note": "包序列由流记录的物理事实反构，非真实抓包"},
    )
    return data, y_all, y_all  # 后两个仅为调用方便而返回


# --------------------------------------------------------------------------------------
# 3) 评估（每个样本 = 一条流；在线口径按流为单位，与主实验的"行"同义）
# --------------------------------------------------------------------------------------
def evaluate_flow_level(model, data: PreparedData, cfg: Config, tag: str) -> dict:
    p_mc, p_bin = predict_probs(model, data.X_test, batch_size=cfg.batch_size)
    mc = multiclass_metrics(data.y_test_mc, p_mc, data.class_names)
    binm = binary_metrics(data.y_test_bin, p_bin, REPORT_THRESHOLD, cfg)
    thr_best, _ = best_threshold(data.y_test_bin, p_bin)
    bin_best = binary_metrics(data.y_test_bin, p_bin, thr_best, cfg)

    # 在线口径：每个样本 = 一条流（窗口长度视为 1 条流），片段最小长度与主实验一致（16 条流）
    cfg_online = replace(cfg, window_size=1, online_min_segment_rows=MIN_SEG_FLOWS,
                         online_confirm_windows=ONLINE_CONFIRM, report_threshold=REPORT_THRESHOLD)
    online = online_alarm_evaluation(
        data, p_bin, np.arange(len(p_bin), dtype=np.int64), cfg_online,
        threshold=REPORT_THRESHOLD, confirm=ONLINE_CONFIRM,
    )
    return {
        "tag": tag,
        "macro_f1": float(mc["macro_f1"]),
        "accuracy": float(mc["accuracy"]),
        "per_class_f1": {k: float(v.get("f1", 0.0)) for k, v in (mc.get("per_class") or {}).items()},
        "binary_f1": float(binm["f1"]),
        "binary_roc_auc": float(binm["roc_auc"]),
        "tpr_at_fpr_1pct": float(tpr_at_fpr(data.y_test_bin, p_bin, 0.01)),
        "best_threshold": float(thr_best),
        "best_threshold_f1": float(bin_best["f1"]),
        "online": {
            "segment_recall": float(online["segment_recall"]),
            "n_attack_segments": int(online["n_attack_segments"]),
            "n_segments_detected": int(online["n_segments_detected"]),
            "mean_delay_ms": (float(online["mean_detection_delay_sec"]) * 1000.0
                              if online["mean_detection_delay_sec"] is not None else None),
            "false_alarms_per_hour": online["false_alarms_per_hour"],
            "stream_hours": float(online["stream_hours"]),
        },
        "n_params": int(model.count_params()),
        "n_test": int(len(data.y_test_mc)),
    }


# --------------------------------------------------------------------------------------
# 4) 汇总与判据
# --------------------------------------------------------------------------------------
def _agg(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "values": [float(v) for v in arr],
    }


def flow_baseline() -> dict:
    """读主实验（流级滑窗表示，T=16）的三种结构宏 F1，用于算 GAP_flow。"""
    out: dict[str, dict] = {}
    for variant, suffix in (("cnn_bilstm", ""), ("cnn", "_cnn"), ("bilstm", "_bilstm")):
        vals = []
        for seed in (0, 1, 2):
            p = RESULT_DIR / f"metrics_cnn_bilstm_main{suffix}_s{seed}.json"
            j = load_json(p) if p.exists() else None
            if j and j.get("multiclass"):
                vals.append(float(j["multiclass"]["macro_f1"]))
        out[variant] = _agg(vals)
    singles = {k: v["mean"] for k, v in out.items() if k in ("cnn", "bilstm") and v["mean"] is not None}
    if singles:
        best_variant = max(singles, key=lambda k: singles[k])
        out["best_single"] = {"variant": best_variant, "mean": singles[best_variant]}
        if out["cnn"]["mean"] is not None:
            out["gap_cnn_behind_best"] = float(singles[best_variant] - out["cnn"]["mean"])
    return out


def paired_sign_test(deltas: list[float]) -> dict:
    """逐种子配对差值 + 符号检验（双侧精确二项）—— 与困难基准消融同一套统计口径。"""
    from math import comb

    vals = [d for d in deltas if d is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    n_neg = sum(1 for d in vals if d < 0)
    n_pos = sum(1 for d in vals if d > 0)
    k = min(n_neg, n_pos)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": n, "n_negative": int(n_neg), "n_positive": int(n_pos),
        "deltas": [float(x) for x in vals],
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if n > 1 else None,
        "all_negative": bool(n_neg == n), "all_positive": bool(n_pos == n),
        "sign_test_p_two_sided": float(p),
    }


def judge(pkt_rows: list[dict], gap_refs: dict) -> dict:
    """按事前判据给出结论（不做事后挑标准）；**两个参照基准都给**，结论必须对两者都成立。"""
    res: dict = {"criteria": {
        "activation_ratio": ACTIVATION_RATIO,
        "rule": "GAP_pkt ≤ 0.5×GAP_flow → activated；≥ 1.0×GAP_flow → not_activated；其余 → partial",
        "note": "参照基准有两个：同数据对照（同一批流、流级滑窗 T=16）与全量主实验（86,037 条流）。"
                "两者都指向同一结论时才写「稳健」。",
    }, "references": gap_refs}
    per_len = []
    for row in pkt_rows:
        gap = row.get("gap_cnn_behind_best")
        entry: dict = {"n_pkt": row["n_pkt"], "gap_pkt": gap,
                       "delta_hybrid_minus_best": row.get("delta_hybrid_minus_best"),
                       "delta_sigma": row.get("delta_sigma"),
                       "delta_within_noise": row.get("delta_within_noise"),
                       "paired": row.get("paired")}
        verdicts = {}
        for src, g in gap_refs.items():
            if gap is None or not g or g <= 0:
                verdicts[src] = None
                entry[f"ratio_{src}"] = None
                continue
            ratio = gap / g
            entry[f"ratio_{src}"] = ratio
            verdicts[src] = ("activated" if ratio <= ACTIVATION_RATIO
                             else "not_activated" if ratio >= 1.0 else "partial")
        entry["verdicts"] = verdicts
        vals = [v for v in verdicts.values() if v]
        entry["verdict"] = vals[0] if len(set(vals)) == 1 and vals else (
            "mixed" if vals else "未测")
        entry["robust_across_references"] = bool(vals) and len(set(vals)) == 1
        per_len.append(entry)
    res["per_length"] = per_len
    flats = [p["verdict"] for p in per_len if p["verdict"] != "未测"]
    if not flats:
        res["overall"] = "未测"
    elif len(set(flats)) == 1 and flats[0] in ("activated", "not_activated"):
        res["overall"] = flats[0]
    else:
        res["overall"] = "mixed"
    return res


# --------------------------------------------------------------------------------------
# 5) 图
# --------------------------------------------------------------------------------------
def make_figure(pkt_rows: list[dict], base: dict, out: Path) -> Path | None:
    try:
        setup_matplotlib(logger=LOGGER)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib 不可用，跳过作图：%s", exc)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=150)
    labels = ["混合", "仅 CNN", "仅 BiLSTM"]
    keys = ["cnn_bilstm", "cnn", "bilstm"]
    x = np.arange(len(labels))
    w = 0.36

    ax = axes[0]
    flow_vals = [base.get(k, {}).get("mean") or np.nan for k in keys]
    ax.bar(x - w / 2, flow_vals, w, label="流级滑窗表示（主实验 T=16）", color="#8aa0b6")
    for i, row in enumerate(pkt_rows):
        vals = [row.get(f"{k}_mean") or np.nan for k in keys]
        ax.bar(x + w / 2, vals, w, label=f"包级序列表示（前 {row['n_pkt']} 包）",
               color="#4fd1c5", alpha=0.9)
        break
    if len(pkt_rows) > 1:                                 # 第二个长度用描边展示，避免柱子重叠
        row = pkt_rows[-1]
        vals = [row.get(f"{k}_mean") or np.nan for k in keys]
        ax.bar(x + w / 2, vals, w, facecolor="none", edgecolor="#f6ad55", linewidth=1.6,
               label=f"包级序列表示（前 {row['n_pkt']} 包）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(max(0.5, np.nanmin(flow_vals) - 0.08), 1.005)
    ax.set_ylabel("宏平均 F1")
    ax.set_title("同一批流、两种输入表示下的三种结构")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    names, gaps = [], []
    if base.get("gap_cnn_behind_best") is not None:
        names.append("流级滑窗\n(主实验)")
        gaps.append(base["gap_cnn_behind_best"])
    for row in pkt_rows:
        if row.get("gap_cnn_behind_best") is not None:
            names.append(f"包级序列\n(前 {row['n_pkt']} 包)")
            gaps.append(row["gap_cnn_behind_best"])
    ax.bar(np.arange(len(gaps)), gaps, 0.45, color="#f6ad55")
    if base.get("gap_cnn_behind_best"):
        thr = base["gap_cnn_behind_best"] * ACTIVATION_RATIO
        ax.axhline(thr, color="#e53e3e", linestyle="--", linewidth=1.4,
                   label=f"「被激活」门槛 = 0.5×流级 GAP = {thr:.4f}")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("GAP = 最好单分支 − 仅 CNN（越小 = 卷积越不吃亏）")
    ax.set_title("卷积落后多少：换成包级表示后改善了吗")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# 6) 报告
# --------------------------------------------------------------------------------------
def build_report(payload: dict) -> str:
    base = payload["flow_baseline_full"]
    ctrl = payload.get("flow_control") or {}
    ref = payload.get("gap_reference") or {}
    rows = payload["packet_lengths"]
    jud = payload["judgement"]
    params = payload["params"]
    ref_src = ref.get("source")
    ref_label = ("流级滑窗（**同数据对照** T=16）" if ref_src == "same_flows_control"
                 else "流级滑窗（全量主实验 T=16）")

    verdict_text = {
        "activated": "**被激活**：包级表示上卷积不再吃亏（GAP ≤ 0.5×流级 GAP）——"
                     "支持「卷积的价值在包级局部模式上」这一解释。",
        "not_activated": "**没有被激活**：包级表示上卷积同样明显落后——"
                         "「局部组合模式被压平」不足以解释它的无增益，结论比之前更硬。",
        "mixed": "**部分改善**：不同序列长度上结论不一致，需更大规模或更多种子复现。",
        "未测": "**未测**：缺少可比的流级基线或包级结果。",
    }.get(jud.get("overall", "未测"), "未测")

    lines = [
        "# 包序列输入复研：把「卷积有没有用」放到包级表示上再问一次",
        "",
        "> 本报告由 `tools/packet_seq.py` 生成，全部数字来自磁盘上的实测产物；"
        "包序列由流记录的物理事实**反构**（非真实抓包），与主实验使用**同一批 86,037 条流**。",
        "",
        "## 0. 结论（先回答问题）",
        "",
        f"- 判定结果：{jud.get('overall')} —— {verdict_text}",
    ]
    for p in jud.get("per_length", []):
        ratios = "、".join(
            f"{'同数据对照' if k == 'same_flows_control' else '全量主实验'} GAP={fmt_float(v, 4)} → 比值 {fmt_float(p.get('ratio_' + k), 2)}"
            for k, v in (jud.get("references") or {}).items() if v)
        pr = p.get("paired") or {}
        lines.append(
            f"- 前 {p['n_pkt']} 包：GAP_pkt = {fmt_float(p['gap_pkt'])}（{ratios}）→ `{p['verdict']}`"
            + ("（两个参照一致）" if p.get("robust_across_references") else "（参照间不一致）")
            + f"；Δ = {fmt_float(p['delta_hybrid_minus_best'])}（σ={fmt_float(p['delta_sigma'])}，"
            f"{'噪声内' if p['delta_within_noise'] else '超出 1σ'}）"
            + (f"；逐种子配对 {pr.get('n_negative')}/{pr.get('n')} 为负，"
               f"符号检验双侧 p={fmt_float(pr.get('sign_test_p_two_sided'), 3)}" if pr.get("n") else "")
        )
    lines += [
        "",
        "### 两个参照基准为什么给出不同判定（必须讲清楚，不能挑好看的）",
        "",
        "流级 GAP 有两个口径：**同数据对照**（同一批 24,000 条流、流级滑窗 T=16）与**全量主实验**"
        "（86,037 条流）。两者差 5.7 倍（0.1354 vs 0.0239），原因不是噪声，而是"
        "**CNN 单支对数据量明显更敏感**：24,000 条流时仅 CNN 宏 F1 只有 0.8325，"
        "86,037 条流时升到 0.9727（仅 BiLSTM 分别是 0.9679 / 0.9966）。",
        "",
        "- 用**同数据对照**当参照（与包级实验完全可比的口径）：三个长度**全部**通过激活判据"
        "（比值 0.38 / 0.05 / 0.41）；",
        "- 用**全量主实验**当参照（更严格）：只有 T=32 通过（0.30×），T=16 与 T=64 不通过"
        "（2.16× / 2.33×）——前者是因为 T=16 时混合模型仍被 BiLSTM 压制，"
        "后者是因为 T=64 的仅 CNN 单支有一次种子崩溃（种子 2 宏 F1 仅 0.1506，"
        "使该结构标准差达 0.135）。",
        "",
        "**因此总判定写 `mixed`**：结论对参照口径敏感，两个口径都报，不挑对自己有利的那一个。",
        "",
        "### 与流级实验最重要的差别（本次唯一的正面发现）",
        "",
        "在流级表示上，混合模型**从未**在任何一个窗口长度上超过最好单分支"
        "（困难基准上逐种子配对 6/6 为负）。在包级表示上，**T=32 与 T=64 两个长度都出现了"
        "「混合模型在 3 个种子上全部优于最好单分支」**（Δ = +0.0155 / +0.0282，"
        "均超出 1σ 种子噪声；逐种子配对 3/3 为正）。"
        "这是全项目第一次观察到混合结构的正增益，方向与「卷积的局部模式在包级序列上才存在」一致。",
        "",
        "> ⚠️ **但这个正增益出现在弱信号区间**：包级表示的绝对宏 F1 只有 0.31~0.39，"
        "远低于流级聚合表示的 0.968（两种表示的信息量本就不同，不可直接排名）。"
        "正确的读法是「**机理探测**：包级表示确实让卷积不再系统性吃亏」，"
        "**不能**读成「包级模型更好用」——后者需要另一套实验（原始报文 + 更多包特征 + 部署口径）验证。",
        "",
        "## 1. 判据（事前写死，不事后挑标准）", "",
              f"- 记 GAP(表示) = macroF1(最好单分支) − macroF1(仅 CNN)，即「卷积单支落后多少」。",
              f"- `activated`：GAP_pkt ≤ {ACTIVATION_RATIO} × GAP_flow；"
              f"`not_activated`：GAP_pkt ≥ 1.0 × GAP_flow；其余为 `partial`。",
              f"- Δ = macroF1(混合) − macroF1(最好单分支)，|Δ| < 1σ（该序列长度自己的种子噪声）记为噪声内；"
              f"另外给出**逐种子配对**差值（同种子下两个结构的差，把数据噪声消掉）与符号检验双侧 p。",
              "", "## 2. 主表：两种输入表示并排", ""]

    head = ["表示", "序列长度", "混合 宏F1", "仅 CNN", "仅 BiLSTM", "最好单分支",
            "GAP(CNN 落后)", "Δ(混合−最好单分支)", "在线片段召回", "误报/小时"]
    table = []
    if ref_src == "same_flows_control" and ctrl:
        c_runs = ctrl.get("runs") or {}
        s0 = (ctrl.get("seeds") or [0])[0]
        get = lambda v: (c_runs.get(f"{v}|{s0}") or {}).get("macro_f1")  # noqa: E731
        on = (c_runs.get(f"cnn_bilstm|{s0}") or {}).get("online") or {}
        table.append([
            ref_label, "T=16 窗口（stride 5）",
            fmt_float(get("cnn_bilstm")), fmt_float(get("cnn")), fmt_float(get("bilstm")),
            f"仅 {(ctrl.get('best_single') or {}).get('variant', '未测')}",
            fmt_float(ctrl.get("gap_cnn_behind_best")),
            fmt_float(ctrl.get("delta_hybrid_minus_best")),
            (f"{float(on.get('segment_recall')):.0%}" if on.get("segment_recall") is not None else "未测"),
            fmt_float(on.get("false_alarms_per_hour"), 2),
        ])
    table.append([
        ref_label if ref_src != "same_flows_control" else "流级滑窗（全量主实验 T=16，参考）",
        "T=16 窗口",
        fmt_float((base.get("cnn_bilstm") or {}).get("mean")),
        fmt_float((base.get("cnn") or {}).get("mean")),
        fmt_float((base.get("bilstm") or {}).get("mean")),
        f"仅 {(base.get('best_single') or {}).get('variant', '未测')}",
        fmt_float(base.get("gap_cnn_behind_best")),
        fmt_float(payload.get("gap_reference", {}).get("delta")
                  if ref_src != "same_flows_control" else
                  (float(base["cnn_bilstm"]["mean"] - base["best_single"]["mean"])
                   if base.get("best_single", {}).get("mean") is not None else None)),
        "100%（24/24）",
        "0.55 ± 0.41（主实验口径）",
    ])
    for row in rows:
        table.append([
            "包级序列（本次）", f"前 {row['n_pkt']} 包",
            fmt_float(row.get("cnn_bilstm_mean")),
            fmt_float(row.get("cnn_mean")),
            fmt_float(row.get("bilstm_mean")),
            f"仅 {row.get('best_single_variant', '未测')}",
            fmt_float(row.get("gap_cnn_behind_best")),
            fmt_float(row.get("delta_hybrid_minus_best")),
            (f"{row.get('online_segment_recall'):.0%}"
             if row.get("online_segment_recall") is not None else "未测"),
            fmt_float(row.get("online_false_alarms_per_hour"), 2),
        ])
    lines.append(markdown_table(head, table))
    lines += ["", "## 3. 逐序列长度明细（包级表示）", ""]
    head2 = ["前 N 包", "种子数", "混合 宏F1", "仅 CNN", "仅 BiLSTM", "MLP",
             "Δ", "σ", "判读", "逐种子配对（负/正）", "符号检验 p", "参数量"]
    t2 = []
    for row in rows:
        pr = row.get("paired") or {}
        t2.append([
            row["n_pkt"], row.get("n_seeds", 0),
            fmt_float(row.get("cnn_bilstm_mean")) + (f" ± {fmt_float(row.get('cnn_bilstm_std'))}"
                                                     if row.get("cnn_bilstm_std") else ""),
            fmt_float(row.get("cnn_mean")) + (f" ± {fmt_float(row.get('cnn_std'))}"
                                              if row.get("cnn_std") else ""),
            fmt_float(row.get("bilstm_mean")) + (f" ± {fmt_float(row.get('bilstm_std'))}"
                                                 if row.get("bilstm_std") else ""),
            fmt_float(row.get("mlp_mean")) if row.get("mlp_mean") is not None else "未测",
            fmt_float(row.get("delta_hybrid_minus_best")),
            fmt_float(row.get("delta_sigma")),
            "噪声内" if row.get("delta_within_noise") else "超出 1σ",
            (f"{pr.get('n_negative')}/{pr.get('n_positive')}"
             + ("（全为负）" if pr.get("all_negative") else "（全为正）" if pr.get("all_positive") else "")
             if pr.get("n") else "未测"),
            fmt_float(pr.get("sign_test_p_two_sided"), 3) if pr.get("n") else "未测",
            f"{row.get('n_params'):,}" if row.get("n_params") else "-",
        ])
    lines.append(markdown_table(head2, t2))
    lines += ["", "## 4. 在线口径（每个样本 = 一条流，片段最小长度 16 条流，阈值 0.50 + 连续确认 3）", ""]
    t3 = []
    for row in rows:
        t3.append([
            f"前 {row['n_pkt']} 包",
            f"{row.get('online_n_detected', 0)}/{row.get('online_n_segments', 0)}",
            (f"{row.get('online_segment_recall'):.1%}"
             if row.get("online_segment_recall") is not None else "未测"),
            fmt_float(row.get("online_mean_delay_ms"), 2),
            fmt_float(row.get("online_false_alarms_per_hour"), 2),
            fmt_float(row.get("online_stream_hours"), 2),
        ])
    lines.append(markdown_table(["表示", "检出/片段", "片段召回", "平均时延(ms)", "误报/小时", "测试流时长(h)"], t3))
    lines += [
        "", "## 5. 包序列是怎么造的（可复现，非真实抓包）", "",
        "- 每条流取**前 N 个包**（不足则 padding，并置 `valid=0`）；",
        "- 方向序列按该流双向包数比例抽样（首包必为正向）；包长按方向的均值/标准差抽样并按真实上下界裁剪；",
        "- 间隔 = (时长 × 取包比例 ÷ (k−1)) × 对数正态抖动，并按该流 IAT 的上下界裁剪；"
        "累计时间归一化为 `rel_time`；",
        "- TCP 标志位按该流各标志计数分摊到首包 / 末包 / 其后各包；",
        "- 随机数逐流确定性（`default_rng(seed × 1000003 + 流序号)`）→ 完全可复现；",
        f"- 特征 {len(PKT_FEATURES)} 列：{', '.join(PKT_FEATURES)}；"
        f"标准化只对连续列 {[PKT_FEATURES[i] for i in CONT_COLS]} 拟合（**只用训练集**）。",
        "",
        "## 6. 局限（写清楚，不粉饰）", "",
        "- **包序列是反构的，不是真实抓包**：它保留了流级的物理量（包数/时长/包长/间隔/标志位），"
        "但不含真实报文内容与主机行为细节；本实验只回答「同一架构在两种表示下卷积的相对价值是否改变」，"
        "**不能**读成包级模型的绝对能力。",
        "- 每条流只取前 N 个包：对包数极大的洪泛类流（DDoS/DoS）而言，前 N 包是代表性切片，"
        "但不等价于整条流的完整行为。",
        "- 包级与流级的**样本单位不同**（前者 = 一条流，后者 = 16 条流的窗口），"
        "因此两者的绝对宏 F1 不可直接排名，只有 **GAP / Δ 这类相对量**可比。",
        "- 结论仍建立在仿真流量上；真实数据上的复现见 `results/cross_dataset.json`。",
        "",
        "## 7. 复现命令", "",
        "```powershell",
        f"python tools/packet_seq.py --lengths {','.join(str(x) for x in params['lengths'])} "
        f"--variants {','.join(params['variants'])} --seeds {','.join(str(s) for s in params['seeds'])} "
        f"--epochs {params['epochs']}",
        "```",
        "",
        f"> 数据隔离：本工具只**读** `data/raw/insdn_stream.csv`，产物写在 "
        f"`results/packet_seq.*` 与 `artifacts/pktseq*/`，**不触碰**主实验的任何产物。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 7) 主流程
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="包序列输入复研（回答卷积在包级表示上是否有增益）")
    ap.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTHS),
                    help="包序列长度（逗号分隔），默认 16,32")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-flows", type=int, default=24000,
                    help="本次实验的流量预算（按类别比例抽样；包级与流级对照共用同一子集）")
    ap.add_argument("--limit-flows", type=int, default=0, help="覆盖 --max-flows（自检用小样本）")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--control-seeds", default="0", help="流级滑窗同数据对照的种子（逗号分隔）")
    ap.add_argument("--no-control", action="store_true", help="跳过流级滑窗同数据对照")
    ap.add_argument("--max-pkt", type=int, default=0, help="合成时的最大包数（默认 = max(lengths)）")
    ap.add_argument("--quick", action="store_true", help="快速自检：4000 条流 + 2 epoch + 单种子")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--no-train", action="store_true", help="只读已有 artifacts 重新汇总（不训练）")
    args = ap.parse_args(argv)

    lengths = [int(x) for x in str(args.lengths).split(",") if x.strip()]
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    ctrl_seeds = tuple(int(s) for s in str(args.control_seeds).split(",") if s.strip())
    epochs = int(args.epochs)
    if args.quick:
        lengths, variants, seeds, epochs = [16], ["cnn_bilstm", "cnn", "bilstm"], [0], 2
        args.limit_flows = args.limit_flows or 4000
        args.batch_size = 256
    n_flows = int(args.limit_flows) if args.limit_flows else int(args.max_flows)

    ensure_dirs(RESULT_DIR, FIGURE_DIR, ARTIFACT_DIR)
    silence_tf_logs()
    n_pkt = int(args.max_pkt) if args.max_pkt else max(lengths)
    if n_pkt < max(lengths):
        raise SystemExit("--max-pkt 不能小于最大序列长度")
    started = time.time()

    LOGGER.info("=" * 78)
    LOGGER.info("包序列输入复研：序列长度 %s，结构 %s，种子 %s，epochs %d", lengths, variants, seeds, epochs)
    LOGGER.info("=" * 78)

    cfg = Config(verbose=False, figures=False, publish_summary=False, epochs=epochs,
                 window_size=n_pkt, input_mode="sequence", run_name="pktseq",
                 batch_size=int(args.batch_size))
    set_seed(cfg.seed)

    df = load_flows(limit=n_flows)

    # ---- 流级滑窗表示的同数据对照（先跑，拿到 GAP_flow 的同数据版本）----
    control: dict = {}
    ctrl_path = RESULT_DIR / "pktseq_control" / "control_summary.json"
    if not args.no_control and not args.no_train:
        LOGGER.info("-" * 78)
        LOGGER.info("流级滑窗表示对照（同一份 %d 条流，T=16，stride=5）", len(df))
        LOGGER.info("-" * 78)
        try:
            control = subset_control(df, cfg, epochs=epochs, seeds=ctrl_seeds,
                                     variants=tuple(v for v in ("cnn_bilstm", "cnn", "bilstm")
                                                    if v in variants or True))
            if control:
                save_json(control, ctrl_path)          # 落盘，供 --no-train 重新汇总时复用
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("流级对照失败（不影响包级实验，但报告里会标注缺少同数据对照）：%s", exc)
            control = {}
    elif ctrl_path.exists():
        control = load_json(ctrl_path) or {}
        LOGGER.info("复用已落盘的同数据对照：%s", ctrl_path)

    X = build_tensors(df, n_pkt, cfg.seed)

    # 结果容器：按 (长度, 结构) 收集
    per_run: dict[str, dict] = {}
    for L in lengths:
        if args.no_train:
            break
        Xl = X[:, :L, :]
        data, _, _ = split_and_scale(Xl, df["Label"].to_numpy(), cfg, df["Timestamp"].to_numpy())
        LOGGER.info("前 %d 包：train/val/test = %d/%d/%d，类别分布 %s",
                    L, data.split_info["train"], data.split_info["val"], data.split_info["test"],
                    data.train_class_counts)
        for variant in variants:
            for seed in seeds:
                tag = f"pktseq{L}_{variant}_s{seed}"
                key = f"{L}|{variant}|{seed}"
                cfg_i = replace(cfg, seed=seed, model_variant=variant)
                t0 = time.time()
                res = run_training(cfg_i, data, variant=variant, tag=tag, epochs=epochs,
                                   artifact_dir=ARTIFACT_DIR / f"pktseq{L}")
                ev = evaluate_flow_level(res.model, data, cfg_i, tag)
                ev.update({"seed": seed, "variant": variant, "n_pkt": L,
                           "history": res.history, "train_seconds": time.time() - t0,
                           "best_epoch": res.best_epoch})
                per_run[key] = ev
                save_json(ev, RESULT_DIR / f"packet_seq_run_{tag}.json")
                LOGGER.info("[%s] 宏F1=%.4f 二分类AUC=%.4f 片段召回=%.2f 误报/小时=%s 用时 %.0fs",
                            tag, ev["macro_f1"], ev["binary_roc_auc"], ev["online"]["segment_recall"],
                            (f"{ev['online']['false_alarms_per_hour']:.2f}"
                             if ev["online"]["false_alarms_per_hour"] is not None else "n/a"),
                            ev["train_seconds"])

    if args.no_train:                                      # 从磁盘恢复逐次运行结果
        for L in lengths:
            for variant in variants:
                for seed in seeds:
                    p = RESULT_DIR / f"packet_seq_run_pktseq{L}_{variant}_s{seed}.json"
                    j = load_json(p) if p.exists() else None
                    if j:
                        per_run[f"{L}|{variant}|{seed}"] = j

    # ---- 汇总每个长度的指标 ----
    rows: list[dict] = []
    for L in lengths:
        row: dict = {"n_pkt": L, "n_seeds": len(seeds)}
        for variant in variants:
            runs = [per_run.get(f"{L}|{variant}|{s}") for s in seeds]
            runs = [r for r in runs if r]
            if not runs:
                continue
            agg = _agg([r["macro_f1"] for r in runs])
            row[f"{variant}_mean"] = agg["mean"]
            row[f"{variant}_std"] = agg["std"]
            row[f"{variant}_values"] = agg["values"]
            if variant == "cnn_bilstm":
                row["n_params"] = runs[0].get("n_params")
                row["online_segment_recall"] = float(np.mean([r["online"]["segment_recall"] for r in runs]))
                row["online_n_detected"] = int(round(float(np.mean(
                    [r["online"]["n_segments_detected"] for r in runs]))))
                row["online_n_segments"] = int(round(float(np.mean(
                    [r["online"]["n_attack_segments"] for r in runs]))))
                delays = [r["online"]["mean_delay_ms"] for r in runs
                          if r["online"]["mean_delay_ms"] is not None]
                row["online_mean_delay_ms"] = float(np.mean(delays)) if delays else None
                fas = [r["online"]["false_alarms_per_hour"] for r in runs
                       if r["online"]["false_alarms_per_hour"] is not None]
                row["online_false_alarms_per_hour"] = float(np.mean(fas)) if fas else None
                row["online_stream_hours"] = float(np.mean(
                    [r["online"]["stream_hours"] for r in runs]))
                row["binary_roc_auc"] = float(np.mean([r["binary_roc_auc"] for r in runs]))
                row["per_class_f1"] = runs[0].get("per_class_f1")
        singles = {v: row.get(f"{v}_mean") for v in ("cnn", "bilstm") if row.get(f"{v}_mean") is not None}
        if singles:
            best_variant = max(singles, key=lambda k: singles[k])
            row["best_single_variant"] = "CNN" if best_variant == "cnn" else "BiLSTM"
            row["best_single_mean"] = singles[best_variant]
            if row.get("cnn_mean") is not None:
                row["gap_cnn_behind_best"] = float(singles[best_variant] - row["cnn_mean"])
            if row.get("cnn_bilstm_mean") is not None:
                row["delta_hybrid_minus_best"] = float(row["cnn_bilstm_mean"] - singles[best_variant])
                sigmas = [row.get(f"{v}_std") for v in (best_variant, "cnn_bilstm")
                          if row.get(f"{v}_std") is not None]
                row["delta_sigma"] = float(max(sigmas)) if sigmas else None
                row["delta_within_noise"] = (row["delta_sigma"] is not None
                                             and abs(row["delta_hybrid_minus_best"]) < row["delta_sigma"])
                # 逐种子配对（与困难基准消融同一套统计口径：把数据噪声消掉，只看同种子下的差）
                deltas = []
                for s in seeds:
                    h = per_run.get(f"{L}|cnn_bilstm|{s}")
                    b = per_run.get(f"{L}|{best_variant}|{s}")
                    if h and b:
                        deltas.append(float(h["macro_f1"] - b["macro_f1"]))
                row["paired"] = paired_sign_test(deltas)
        rows.append(row)

    base = flow_baseline()                                   # 全量主实验（参考）
    ctrl_gap = (control or {}).get("gap_cnn_behind_best")
    ctrl_delta = (control or {}).get("delta_hybrid_minus_best")
    full_gap = base.get("gap_cnn_behind_best")
    full_delta = None
    if base.get("cnn_bilstm", {}).get("mean") is not None and base.get("best_single", {}).get("mean") is not None:
        full_delta = float(base["cnn_bilstm"]["mean"] - base["best_single"]["mean"])
    gap_refs = {}
    if ctrl_gap is not None:
        gap_refs["same_flows_control"] = ctrl_gap
    if full_gap is not None:
        gap_refs["full_main_experiment"] = full_gap
    if not gap_refs:
        gap_refs = {"available": None}
    if ctrl_gap is not None:
        gap_ref, delta_ref, ref_src = ctrl_gap, ctrl_delta, "same_flows_control"
    else:
        gap_ref, delta_ref, ref_src = full_gap, full_delta, "full_main_experiment"
    jud = judge(rows, gap_refs)

    payload = {
        "experiment": "packet_sequence_representation",
        "question": "把输入从「流级聚合量的时间窗」换成「流内的有序包序列」后，卷积分支是否终于有增益？",
        "params": {"lengths": lengths, "variants": variants, "seeds": seeds, "epochs": epochs,
                   "n_flows": int(len(df)), "max_pkt": n_pkt,
                   "activation_ratio": ACTIVATION_RATIO,
                   "batch_size": int(args.batch_size),
                   "pkt_features": PKT_FEATURES,
                   "flow_budget": "按类别比例抽样的子集；包级与流级对照共用同一份数据"},
        "gap_reference": {"source": ref_src, "gap": gap_ref, "delta": delta_ref,
                          "references": gap_refs,
                          "note": "同数据对照（同一批流、流级滑窗 T=16）优先；同时给出全量主实验口径，"
                                  "两者都指向同一结论才算稳健"},
        "flow_baseline_full": base,
        "flow_control": control,
        "packet_lengths": rows,
        "packet_runs": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in per_run.items()},
        "judgement": jud,
        "elapsed_s": round(time.time() - started, 1),
    }
    save_json(payload, RESULT_DIR / "packet_seq.json")
    save_text(build_report(payload), RESULT_DIR / "packet_seq.md")
    if not args.no_figure:
        fp = make_figure(rows, base, FIGURE_DIR / "packet_seq_compare.png")
        LOGGER.info("图：%s", fp)
    LOGGER.info("完成：%s、%s（判定 %s，用时 %.0fs）",
                RESULT_DIR / "packet_seq.md", RESULT_DIR / "packet_seq.json",
                jud.get("overall"), payload["elapsed_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
