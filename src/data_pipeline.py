"""数据管线：读取 → 清洗 → 编码 → 标准化 → 滑窗 → 时序切分。

相比原始 demo，本模块修掉了 4 个会直接毁掉比赛结果的隐患：

1. **硬编码列索引**：原代码写死 ``X = data[:, 0:77] / Y = data[:, 78:]``，
   一旦数据集列数变化就静默错位。这里全部改成「按列名自动识别」。
2. **隐藏的脏特征列**：原代码用 ``df.dropna().reset_index()``（而不是
   ``reset_index(drop=True)``），凭空多出一列行号 ``index``，还把它当成了第 1 个特征，
   同时漏掉了真正的 ``Idle Min`` 特征。本模块用 ``reset_index(drop=True)`` 并做特征白名单。
3. **标准化泄漏**：原代码在**全量数据**上 fit ``StandardScaler`` 再切分，
   测试集信息泄漏进训练。这里只在训练集上 fit。
4. **随机切分 + 滑窗重叠泄漏**：网络流是时间序列，随机切分会让同一条会话的相邻记录
   分散到训练/测试集。这里按时间顺序切分，并在切分边界留出 purge 间隙。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .traffic_generator import (
    LABEL_COLUMNS,
    SCHEMA_ID_COLUMNS,
    TrafficSimulator,
    generate_datasets,
)
from .utils import (
    ARTIFACT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    get_logger,
    load_json,
    resolve_dir,
    save_json,
)

LOGGER = get_logger("cida.data")

BINARY_NORMAL = "Normal"

# 正常流量的常见写法：不同数据集对"正常"的命名不一样
#   inSDN → Normal / Normal. ；CIC-IDS2017 → BENIGN / Benign ；NSL-KDD → normal
NORMAL_CANDIDATES = ["Normal", "normal", "Normal.", "normal.", "BENIGN", "Benign", "benign", "0"]


# ======================================================================================
# 读取
# ======================================================================================
def load_capture(cfg: Config, force_regenerate: bool = False) -> tuple[pd.DataFrame, dict]:
    """读取（必要时生成）原始流量数据，返回 (DataFrame, 元信息)。"""
    if cfg.source == "real":
        if not cfg.real_data_dir:
            raise ValueError("source=real 时必须指定 real_data_dir")
        return load_real_csvs(Path(cfg.real_data_dir), cfg)

    path = resolve_dir(getattr(cfg, "raw_dir", ""), RAW_DIR) / f"{cfg.schema}_stream.csv"
    if force_regenerate or not path.exists():
        LOGGER.info("未找到仿真数据，开始生成……")
        generate_datasets(cfg, out_dir=path.parent)
    df = pd.read_csv(path, low_memory=False)
    meta = {"source": "synthetic", "schema": cfg.schema, "path": str(path)}
    LOGGER.info("读取仿真数据集 %s：%s", path.name, df.shape)
    return df, meta


def load_real_csvs(data_dir: Path, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """读取真实 CSV 目录（例如 inSDN 的 Normal.csv / DDoS.csv / ...）。"""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{data_dir} 下没有找到 CSV 文件")
    frames = []
    for f in csv_files:
        try:
            part = pd.read_csv(f, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            part = pd.read_csv(f, low_memory=False, encoding="gbk")
        part["__source_file"] = f.name
        frames.append(part)
        LOGGER.info("读入 %-28s %s", f.name, part.shape)
    df = pd.concat(frames, axis=0, ignore_index=True)
    meta = {
        "source": "real",
        "schema": "real",
        "files": [f.name for f in csv_files],
        "path": str(data_dir),
    }
    LOGGER.info("合并后共 %s 条记录，%d 列", f"{len(df):,}", df.shape[1])
    return df, meta


# ======================================================================================
# 清洗
# ======================================================================================
def detect_label_columns(df: pd.DataFrame, cfg: Config) -> list[str]:
    """自动识别标签列：优先显式指定，其次按候选名匹配。"""
    if cfg.real_label_column and cfg.real_label_column in df.columns:
        return [cfg.real_label_column]
    found = [c for c in LABEL_COLUMNS if c in df.columns]
    if not found:
        # 兜底：最后一列
        LOGGER.warning("未识别到标签列，暂用最后一列 %r", df.columns[-1])
        found = [df.columns[-1]]
    return found


def infer_feature_columns(df: pd.DataFrame, label_cols: list[str], drop_cols: list[str]) -> list[str]:
    """挑出真正参与建模的数值特征列。

    规则：数值型、非标签、非标识列、非常数列、无全空列。
    """
    ignore = set(label_cols) | set(drop_cols) | {"__source_file"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = []
    for c in numeric:
        if c in ignore:
            continue
        col = df[c]
        if col.isna().all():
            continue
        col_valid = col.replace([np.inf, -np.inf], np.nan).dropna()
        if col_valid.empty or float(col_valid.std(ddof=0)) == 0.0:
            continue  # 常数列对模型无信息量
        feats.append(c)
    return feats


def clean_dataframe(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str], list[str], str]:
    """清洗并返回 (干净数据, 特征列, 标签列, 多分类标签列)。"""
    label_cols = detect_label_columns(df, cfg)
    drop_cols = list(SCHEMA_ID_COLUMNS.get(cfg.schema, []))
    if getattr(cfg, "keep_port_features", False):
        # 保留端口作为特征（可做"是否使用端口特征"的消融实验）
        drop_cols = [c for c in drop_cols if "Port" not in c and "port" not in c and c not in ("tp_src", "tp_dst")]
    df = df.copy()

    # 1) 统一处理无穷大 / 缺失
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2) 多分类标签列：优先 attack_type / Label / target 这类带类别名的列
    mc_col = None
    for c in label_cols:
        if c in {"attack_type", "Label", "label", "target", "attack_cat", "class", "category"}:
            mc_col = c
            break
    if mc_col is None:
        mc_col = label_cols[0]

    # 3) 行级清洗
    before = len(df)
    df = df.dropna(subset=[mc_col])
    feats = infer_feature_columns(df, label_cols, drop_cols)
    if not feats:
        raise ValueError("清洗后没有任何可用特征列，请检查数据格式")
    df = df.dropna(subset=feats)
    df = df.reset_index(drop=True)          # 注意：drop=True，绝不把行号带进特征
    df = df.drop_duplicates(subset=feats + [mc_col], keep="first").reset_index(drop=True)
    LOGGER.info("清洗：%s → %s 条（去重/去空后），特征 %d 列", f"{before:,}", f"{len(df):,}", len(feats))

    # 近似重复去重（可选）：部分公开数据集里同一条流的统计量只差几个有效数字，
    # 随机切分后训练集与测试集会出现大量"几乎相同"的样本，指标被严重抬高。
    # 打开这个开关会按指定有效数字把所有特征取整后再去重，用于得到更诚实的泛化指标。
    k = int(getattr(cfg, "dedup_significant_digits", 0) or 0)
    if k > 0:
        before2 = len(df)
        rounded = df[feats].applymap(lambda v: float(f"%.{k}g" % v) if pd.notna(v) else np.nan)
        keep_mask = ~rounded.duplicated()
        df = df[keep_mask].reset_index(drop=True)
        LOGGER.info(
            "近似重复去重（保留 %d 位有效数字）：%s → %s 条（再删除 %s 条近似重复）",
            k, f"{before2:,}", f"{len(df):,}", f"{before2 - len(df):,}",
        )
    return df, feats, label_cols, mc_col


def order_by_time(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """按时间排序（网络流是时序数据，顺序不能乱）。"""
    for col in ("Timestamp", "timestamp", "Time", "time"):
        if col in df.columns:
            try:
                ts = pd.to_numeric(df[col], errors="coerce")
                if ts.notna().mean() > 0.9:
                    return df.assign(__ts=ts).sort_values("__ts").reset_index(drop=True)
            except Exception:
                pass
    LOGGER.info("未找到可用时间戳，按原始行序（=采集顺序）处理")
    return df.reset_index(drop=True)


# ======================================================================================
# 切分 + 标准化 + 滑窗
# ======================================================================================
@dataclass
class PreparedData:
    """训练/验证/测试所需的全部张量与元信息。"""

    X_train: np.ndarray
    y_train_mc: np.ndarray
    y_train_bin: np.ndarray
    X_val: np.ndarray
    y_val_mc: np.ndarray
    y_val_bin: np.ndarray
    X_test: np.ndarray
    y_test_mc: np.ndarray
    y_test_bin: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    scaler: object
    label_encoder: object
    input_mode: str
    window_size: int
    test_stream_features: np.ndarray | None = None      # (n_rows, F) 连续测试流（在线评估/演示用）
    test_stream_labels: np.ndarray | None = None        # (n_rows,) 行级真值（类别名）
    test_stream_time: np.ndarray | None = None          # (n_rows,) 时间戳
    train_class_counts: dict = field(default_factory=dict)
    split_info: dict = field(default_factory=dict)
    raw_meta: dict = field(default_factory=dict)

    # ---------------- 便捷属性 ----------------
    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[-1] if self.input_mode == "sequence" else self.X_train.shape[1])

    @property
    def input_shape(self) -> tuple:
        return tuple(self.X_train.shape[1:])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def attack_class_ids(self) -> list[int]:
        return [i for i, c in enumerate(self.class_names) if c != BINARY_NORMAL]

    @property
    def normal_class_id(self) -> int:
        return int(self.class_names.index(BINARY_NORMAL)) if BINARY_NORMAL in self.class_names else 0

    def class_weight(self) -> dict[int, float]:
        """按训练集频次做逆频加权（攻击类样本远少于正常流量，必须加权）。"""
        counts = np.bincount(self.y_train_mc, minlength=self.n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        total = counts.sum()
        return {i: float(total / (self.n_classes * counts[i])) for i in range(self.n_classes)}

    def summary(self) -> dict:
        # 摘要里不带每个窗口的起始行号（那只是实现细节，写进日志会刷屏）
        split = dict(self.split_info)
        if isinstance(split.get("windows"), dict):
            split["windows"] = {
                k: {kk: vv for kk, vv in (w or {}).items() if kk != "starts"}
                for k, w in split["windows"].items()
            }
        return {
            "input_mode": self.input_mode,
            "input_shape": list(self.input_shape),
            "window_size": int(self.window_size),
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "train": int(len(self.X_train)),
            "val": int(len(self.X_val)),
            "test": int(len(self.X_test)),
            "class_weight": self.class_weight(),
            "train_class_counts": self.train_class_counts,
            "split": split,
        }

    # ---------------- 落盘 / 读取 ----------------
    def save(self, out_dir: Path | None = None) -> Path:
        import joblib

        out_dir = Path(out_dir or PROCESSED_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "tensors.npz",
            X_train=self.X_train, y_train_mc=self.y_train_mc, y_train_bin=self.y_train_bin,
            X_val=self.X_val, y_val_mc=self.y_val_mc, y_val_bin=self.y_val_bin,
            X_test=self.X_test, y_test_mc=self.y_test_mc, y_test_bin=self.y_test_bin,
            test_stream_features=self.test_stream_features if self.test_stream_features is not None else np.empty(0),
            test_stream_labels=self.test_stream_labels if self.test_stream_labels is not None else np.empty(0, dtype=object),
            test_stream_time=self.test_stream_time if self.test_stream_time is not None else np.empty(0),
        )
        joblib.dump(self.scaler, out_dir / "scaler.joblib")
        joblib.dump(self.label_encoder, out_dir / "label_encoder.joblib")
        save_json(
            {
                "class_names": self.class_names,
                "feature_names": self.feature_names,
                "input_mode": self.input_mode,
                "window_size": self.window_size,
                "train_class_counts": self.train_class_counts,
                "split_info": self.split_info,
                "raw_meta": self.raw_meta,
            },
            out_dir / "meta.json",
        )
        LOGGER.info("已保存预处理产物 → %s", out_dir)
        return out_dir

    @classmethod
    def load(cls, out_dir: Path | None = None) -> "PreparedData":
        import joblib

        out_dir = Path(out_dir or PROCESSED_DIR)
        z = np.load(out_dir / "tensors.npz", allow_pickle=True)
        meta = load_json(out_dir / "meta.json")
        stream_labels = z["test_stream_labels"]
        return cls(
            X_train=z["X_train"], y_train_mc=z["y_train_mc"], y_train_bin=z["y_train_bin"],
            X_val=z["X_val"], y_val_mc=z["y_val_mc"], y_val_bin=z["y_val_bin"],
            X_test=z["X_test"], y_test_mc=z["y_test_mc"], y_test_bin=z["y_test_bin"],
            class_names=list(meta["class_names"]),
            feature_names=list(meta["feature_names"]),
            scaler=joblib.load(out_dir / "scaler.joblib"),
            label_encoder=joblib.load(out_dir / "label_encoder.joblib"),
            input_mode=meta["input_mode"],
            window_size=int(meta["window_size"]),
            test_stream_features=(z["test_stream_features"] if z["test_stream_features"].size else None),
            test_stream_labels=(stream_labels if stream_labels.size else None),
            test_stream_time=(z["test_stream_time"] if z["test_stream_time"].size else None),
            train_class_counts=meta["train_class_counts"],
            split_info=meta["split_info"],
            raw_meta=meta["raw_meta"],
        )


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    *,
    pure_only: bool = True,
    timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """把连续流记录切成 (N, T, F) 的时间窗。

    pure_only=True 时只保留"窗口内标签完全一致"的窗口（训练/评估用，
    对应"攻击片段一旦开始就持续一段时间"的真实情形）；
    pure_only=False 时用多数投票打标签，并额外返回每个窗口的真值集合
    （在线检测/演示用，可以看到告警时延）。
    """
    n = len(features)
    if n < window_size:
        raise ValueError(f"样本数 {n} 少于窗口长度 {window_size}，请减小 window_size 或增加数据量")
    starts = np.arange(0, n - window_size + 1, stride, dtype=np.int64)
    X_list, y_list, ts_list, kept_starts = [], [], [], []
    skipped = 0
    for s in starts:
        seg = labels[s : s + window_size]
        uniq = pd.unique(seg)
        if pure_only:
            if len(uniq) != 1:
                skipped += 1
                continue
            lab = uniq[0]
        else:
            vals, cnts = np.unique(seg, return_counts=True)
            attacks = [(int(c), str(v)) for v, c in zip(vals, cnts) if str(v) != BINARY_NORMAL]
            if attacks:  # 告警导向：窗口内只要出现攻击，就标注为出现最多的那类攻击
                lab = max(attacks, key=lambda t: t[0])[1]
            else:
                lab = max(zip(cnts, vals), key=lambda t: t[0])[1]
        X_list.append(features[s : s + window_size])
        y_list.append(lab)
        kept_starts.append(int(s))
        if timestamps is not None:
            ts_list.append(timestamps[s : s + window_size])
    info = {
        "n_candidates": int(len(starts)),
        "n_kept": int(len(X_list)),
        "n_skipped_mixed": int(skipped),
        "window_size": int(window_size),
        "stride": int(stride),
        "starts": np.asarray(kept_starts, dtype=np.int64),
    }
    if not X_list:
        raise ValueError("没有生成任何窗口，请检查 window_size / stride 与数据量")
    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=object)
    ts = np.stack(ts_list) if ts_list else None
    return X, y, ts, info


def prepare_dataset(
    cfg: Config,
    force_regenerate: bool = False,
    save: bool = True,
    out_dir: Path | None = None,
) -> PreparedData:
    """一条龙：读取 → 清洗 → 时序切分 → 标准化 → 滑窗 → 落盘。"""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df, raw_meta = load_capture(cfg, force_regenerate=force_regenerate)
    # 分组列（如 Flow ID / Src IP）在清洗时会被当作标识列丢弃，这里先另存一份供切分使用
    _gcol = str(getattr(cfg, "split_group_column", "") or "")
    if _gcol and _gcol in df.columns:
        df = df.copy()
        df["__group"] = df[_gcol].astype(str)
        LOGGER.info("已保留分组列 %r 用于按组切分（共 %d 个不同取值）", _gcol, df["__group"].nunique())
    df, feats, label_cols, mc_col = clean_dataframe(df, cfg)
    df = order_by_time(df, cfg)

    labels_raw = df[mc_col].astype(str).str.strip()
    is_binary_only = labels_raw.nunique() <= 2 and set(labels_raw.unique()) <= {"0", "1", "0.0", "1.0"}
    if is_binary_only:
        labels_raw = labels_raw.map(lambda v: BINARY_NORMAL if float(v) == 0 else "Attack")
        LOGGER.info("标签为 0/1 二值形式，已映射为 Normal / Attack")
    else:
        # 不同数据集对"正常流量"的命名不同（inSDN: Normal / CIC-IDS2017: BENIGN / NSL-KDD: normal），
        # 这里统一归一化成 Normal，保证下游的二分类与在线评估口径一致。
        present = set(labels_raw.unique())
        for cand in NORMAL_CANDIDATES:
            if cand in present and cand != BINARY_NORMAL:
                labels_raw = labels_raw.map(lambda v, c=cand: BINARY_NORMAL if v == c else v)
                LOGGER.info("已把正常流量标签 %r 归一化为 %r", cand, BINARY_NORMAL)
                break

    # ---- 标签编码（多分类 + 二分类） ----
    encoder = LabelEncoder()
    y_all_mc = encoder.fit_transform(labels_raw.to_numpy())
    class_names = [str(c) for c in encoder.classes_]
    normal_id = class_names.index(BINARY_NORMAL) if BINARY_NORMAL in class_names else -1
    y_all_bin = np.where(y_all_mc == normal_id, 0, 1).astype(np.int64) if normal_id >= 0 else np.zeros(len(y_all_mc), np.int64)

    # ---- 切分（time = 按时间顺序；random = 分层随机，用于"分块发布"的公开数据集）----
    n = len(df)
    split_mode = str(getattr(cfg, "split_mode", "time")).lower()
    purge = int(cfg.purge_windows * max(cfg.window_size, 1)) if split_mode == "time" else 0
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    if split_mode == "random":
        rng = np.random.default_rng(cfg.seed + 12345)
        group_col = None
        for cand in ["__group", str(getattr(cfg, "split_group_column", "") or "")]:
            if cand and cand in df.columns:
                group_col = cand
                break
        if group_col:
            # ★ 按组切分（group split）：同一会话/主机的流记录只能进同一个子集，
            #   这是流量数据集防"近似重复泄漏"的标准做法（随机按行切分会让指标虚高）。
            gkey = pd.factorize(df[group_col].astype(str))[0]
            n_groups = int(gkey.max()) + 1
            gsum = np.zeros((n_groups, len(class_names)), dtype=np.int64)
            np.add.at(gsum, (gkey, y_all_mc), 1)
            gcls = gsum.argmax(axis=1)             # 每个组的多数类别，用它做分层
            gsize = gsum.sum(axis=1)
            chosen: list[np.ndarray] = []
            for cls in range(len(class_names)):
                gids = np.flatnonzero(gcls == cls)
                rng.shuffle(gids)
                # 按"行数"比例切分，保证三个子集的样本量接近目标比例
                cum = np.cumsum(gsize[gids])
                total_cls = int(cum[-1]) if len(cum) else 0
                i1 = int(np.searchsorted(cum, total_cls * cfg.train_ratio))
                i2 = int(np.searchsorted(cum, total_cls * (cfg.train_ratio + cfg.val_ratio)))
                chosen.append(gids[: max(i1, 1)])
                chosen.append(gids[max(i1, 1) : max(i2, max(i1, 1) + 1)])
                chosen.append(gids[max(i2, max(i1, 1) + 1) :])
            gtrain = np.concatenate([chosen[i] for i in range(0, len(chosen), 3)])
            gval = np.concatenate([chosen[i] for i in range(1, len(chosen), 3)])
            gtest = np.concatenate([chosen[i] for i in range(2, len(chosen), 3)])
            idx_train = np.sort(np.flatnonzero(np.isin(gkey, gtrain)))
            idx_val = np.sort(np.flatnonzero(np.isin(gkey, gval)))
            idx_test = np.sort(np.flatnonzero(np.isin(gkey, gtest)))
            LOGGER.info(
                "按组切分（分组列 = %s，共 %d 个组）：train=%d, val=%d, test=%d",
                group_col, n_groups, len(idx_train), len(idx_val), len(idx_test),
            )
        else:
            perm = rng.permutation(n)
            # 分层：保证每个类别在三个子集里都按比例出现（对分块发布的公开数据集必须如此）
            blocks: list[np.ndarray] = []
            for cls in np.unique(y_all_mc):
                idx_cls = perm[y_all_mc[perm] == cls]
                blocks.append(idx_cls)
            order_arr = np.concatenate(blocks)
            # ★ 关键一步：分块拼接后必须打散，否则切出来的 train/val/test 会各自"整块整块"地
            #   只包含单一类别（这里的 bug 真实发生过：val/test 只剩攻击类，评估直接失效）。
            rng.shuffle(order_arr)
            idx_train = np.sort(order_arr[:n_train])
            idx_val = np.sort(order_arr[n_train : n_train + n_val])
            idx_test = np.sort(order_arr[n_train + n_val :])
            LOGGER.info(
                "分层随机切分：train=%d, val=%d, test=%d（split_mode=random）",
                len(idx_train), len(idx_val), len(idx_test),
            )
        dist = {
            "train": {class_names[i]: int((y_all_mc[idx_train] == i).sum()) for i in range(len(class_names))},
            "val": {class_names[i]: int((y_all_mc[idx_val] == i).sum()) for i in range(len(class_names))},
            "test": {class_names[i]: int((y_all_mc[idx_test] == i).sum()) for i in range(len(class_names))},
        }
        LOGGER.info("各子集类别分布：%s", dist)
    else:
        tr_end = max(n_train - purge, 1)
        va_start = min(n_train + purge, n - 1)
        va_end = max(min(n_train + n_val - purge, n), va_start + 1)
        te_start = min(n_train + n_val + purge, n - 1)
        idx_train = np.arange(0, tr_end)
        idx_val = np.arange(va_start, min(va_end, n))
        idx_test = np.arange(te_start, n)
        LOGGER.info(
            "时序切分：train=%d, val=%d, test=%d（边界各留 %d 条 purge 间隙，避免滑窗跨集泄漏）",
            len(idx_train), len(idx_val), len(idx_test), purge,
        )
    if min(len(idx_train), len(idx_val), len(idx_test)) < cfg.window_size + 1:
        raise ValueError("切分后某个子集样本过少，请增大数据量或调整 train_ratio/val_ratio")

    X_raw = df[feats].to_numpy(dtype=np.float64)

    # ---- 标准化：只在训练集上 fit（修掉原始代码的泄漏）----
    scaler = StandardScaler()
    scaler.fit(X_raw[idx_train])
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    if cfg.feature_noise and cfg.feature_noise > 0:
        # 轻微特征抖动 → 模拟真实采集中的量测噪声，防止模型死记硬背
        rng = np.random.default_rng(cfg.seed + 7)
        sigma = cfg.feature_noise * 0.1
        X_scaled = X_scaled + rng.normal(0.0, sigma, size=X_scaled.shape).astype(np.float32)

    ts_all = df["__ts"].to_numpy() if "__ts" in df.columns else np.arange(n, dtype=np.float64)
    labels_named = np.asarray(class_names, dtype=object)[y_all_mc]

    # ---- 滑窗 ----
    if cfg.input_mode == "sequence":
        def pack(idx):
            X, y, _, info = build_windows(
                X_scaled[idx], labels_named[idx], cfg.window_size, cfg.window_stride,
                pure_only=True, timestamps=ts_all[idx],
            )
            y_mc = encoder.transform(y.astype(str))
            y_bin = np.where(y_mc == normal_id, 0, 1).astype(np.int64) if normal_id >= 0 else np.zeros(len(y_mc), np.int64)
            return X, y_mc.astype(np.int64), y_bin, info

        X_train, y_train_mc, y_train_bin, info_tr = pack(idx_train)
        X_val, y_val_mc, y_val_bin, info_va = pack(idx_val)
        X_test, y_test_mc, y_test_bin, info_te = pack(idx_test)
        stream_feat, stream_lab, stream_ts = X_scaled[idx_test], labels_named[idx_test], ts_all[idx_test]
    elif cfg.input_mode == "sample":
        # 单条流分类（与原始 demo 完全同构，用于消融对比）
        def pack2(idx):
            X = X_scaled[idx][:, :, None]
            y_mc = encoder.transform(labels_named[idx].astype(str)).astype(np.int64)
            y_bin = np.where(y_mc == normal_id, 0, 1).astype(np.int64) if normal_id >= 0 else np.zeros(len(y_mc), np.int64)
            return X, y_mc, y_bin, {"n_kept": int(len(idx))}

        X_train, y_train_mc, y_train_bin, info_tr = pack2(idx_train)
        X_val, y_val_mc, y_val_bin, info_va = pack2(idx_val)
        X_test, y_test_mc, y_test_bin, info_te = pack2(idx_test)
        stream_feat, stream_lab, stream_ts = X_scaled[idx_test], labels_named[idx_test], ts_all[idx_test]
    else:
        raise ValueError(f"未知 input_mode：{cfg.input_mode}")

    train_counts = {class_names[i]: int(v) for i, v in enumerate(np.bincount(y_train_mc, minlength=len(class_names)))}
    data = PreparedData(
        X_train=X_train, y_train_mc=y_train_mc, y_train_bin=y_train_bin,
        X_val=X_val, y_val_mc=y_val_mc, y_val_bin=y_val_bin,
        X_test=X_test, y_test_mc=y_test_mc, y_test_bin=y_test_bin,
        class_names=class_names, feature_names=feats, scaler=scaler, label_encoder=encoder,
        input_mode=cfg.input_mode, window_size=int(cfg.window_size),
        test_stream_features=stream_feat, test_stream_labels=stream_lab, test_stream_time=stream_ts,
        train_class_counts=train_counts,
        split_info={
            "train": len(idx_train), "val": len(idx_val), "test": len(idx_test),
            "purge_rows": purge, "windows": {"train": info_tr, "val": info_va, "test": info_te},
            "time_ordered_split": split_mode == "time",
            "split_mode": split_mode,
        },
        raw_meta=raw_meta,
    )
    LOGGER.info("数据准备完成：%s", data.summary())
    if save:
        data.save(out_dir or resolve_dir(getattr(cfg, "processed_dir", ""), PROCESSED_DIR))
    return data


# ======================================================================================
# 在线（流式）窗口：给告警级评估与可视化演示用
# ======================================================================================
def rolling_windows(features: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """在连续测试流上滚动取窗（允许跨片段），返回 (窗口张量, 每个窗口的起始行号)。"""
    n = len(features)
    starts = np.arange(0, max(n - window_size + 1, 1), stride, dtype=np.int64)
    starts = starts[starts + window_size <= n]
    X = np.stack([features[s : s + window_size] for s in starts]).astype(np.float32)
    return X, starts


def load_predictions_cache(path: Path | None = None) -> pd.DataFrame | None:
    """读取评估阶段缓存的窗口级预测（演示界面在没装 TF 时也能跑）。

    path 为空时自动挑选 ``artifacts/predictions_*.csv`` 中最新的一个。
    """
    if path is None:
        candidates = sorted(ARTIFACT_DIR.glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None
        path = candidates[-1]
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


__all__ = [
    "PreparedData",
    "build_windows",
    "clean_dataframe",
    "detect_label_columns",
    "load_capture",
    "load_real_csvs",
    "prepare_dataset",
    "rolling_windows",
    "TrafficSimulator",
    "LABEL_COLUMNS",
]
