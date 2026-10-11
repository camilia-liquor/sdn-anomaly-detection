"""真实数据上的无泄漏评测（主指标：给定误报预算下的攻击召回）。

设计要点（事前写死，不允许事后调整）：
1. **无泄漏**：先按项目口径去重（exact + 3 位有效数字近似），再按「流身份」分组切分
   （组 = 源IP|目的IP|目的端口|协议），并**验证**训练/评估两侧的组交集与近重复交集都为 0；
2. **阈值不看评估段**：判定阈值一律取「拟合段正常窗口分数」的 1−预算 分位（不需要标签）；
3. **主指标** = 1% 误报预算下的攻击召回；同时给出 0.1% 预算与 ROC-AUC 作为次要口径；
4. 三个协议：
   - `block`（默认）：每个连续同标签段内前后切分 + purge 间隔。保住行连续性（窗口才开得出来），
     两侧都含正常与攻击；近重复由全局去重保证并在切分后校验。
   - `group`：按流身份（源IP|目的IP|目的端口|协议）哈希分组切分。**注意**：哈希会把连续行打散，
     同侧窗口会大幅减少，极端情况下拟合段没有正常窗口——本工具遇到这种情况会直接报错而不是出结论。
   - `cross_day`：按天切分（拟合段用前两天、评估段用其余天），是更硬的「跨场景/未见攻击族」口径。
5. 每份数据都记录：去重前后行数、两侧标签分布、纯窗口数、真实误报率；拟合段无正常窗口时报错退出。

用法：
    python tools/real_eval.py --dataset insdn
    python tools/real_eval.py --dataset cic_ids2017 --protocol cross_day
    python tools/real_eval.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

# joblib/sklearn 会去探测物理核数并 spawn 子进程；在受限环境里这一步会抛异常并让
# 正常跑完的进程返回非 0 退出码。显式给出核数即可跳过探测。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config                      # noqa: E402
from src.data_pipeline import clean_dataframe      # noqa: E402
from src.model import load_model_bundle, predict_probs  # noqa: E402

RESULT_DIR = ROOT / "results"
# 在线口径：评估时长超过该值即认为数据集的时间戳不可信（不做"误报/小时"与运维换算）。
# 依据：实测抓包通常几小时~几天；CIC-IDS2017 的时间戳跨度约 1,476 h、inSDN 约 6,733 h，
# 均为数据集构造所致，其"误报/小时"分母虚大、时延无物理意义。
SPAN_LIMIT_HOURS = 168.0
# 攻击片段计入分母的**窗口数**下限。真实数据侧只有窗口级标签，无法像合成域那样按行数判定
# （合成域用 max(window_size, online_min_segment_rows) 行），故此处按窗口计。
# 作用相同：把"单窗抖动"排除在召回分母之外。
MIN_SEG_WINDOWS = 2
META = json.loads((ROOT / "data" / "processed" / "meta.json").read_text(encoding="utf-8"))
FEATURES: list[str] = list(META["feature_names"])
SYN_SCALER = ROOT / "data" / "processed" / "scaler.joblib"
MAIN_MODEL = ROOT / "artifacts" / "cnn_bilstm_main_s0" / "model.keras"

# ------------------------------------------------------------------ 数据集登记
# label: 标签列；time: 时间列；group: 流身份列；days: 按天切分时的排序键（文件名）
DATASETS: dict[str, dict] = {
    "insdn": {
        "title": "inSDN（真实公开集，仅 0/1 标签）",
        "files": ["data/real/insdn/Dataset.csv"],
        "label": "target",
        "time": "Timestamp",
        "group": ["Src IP", "Dst IP", "Dst Port", "Protocol"],
        "day_from": None,
        "family": {"0": "Normal", "1": "Attack"},
        "notes": ["行序为「正常块 + 攻击块」拼接，非时间序；因此只能用 group/block 协议，不能做跨天时序评测",
                  "时间戳仅 1,392 个分钟值却跨 8 个月：可用于排序，但**不能**当作真实经过时长折算误报率"],
    },
    "cic_ids2017": {
        "title": "CIC-IDS2017（GeneratedLabelledFlows：含 Timestamp 与流标识）",
        "files": ["data/real/cic_ids2017/extracted/GeneratedLabelledFlows"],
        "label": "Label",
        "time": "Timestamp",
        "group": ["Source IP", "Destination IP", "Destination Port", "Protocol"],
        "day_from": True,
        "family": {},
        "notes": ["8 天文件；标签含编码变体与缺失值，聚合到本工程 8 大类前必须归一化"],
    },
    "cic_ids2018": {
        "title": "CSE-CIC-IDS2018（3 天逐日 CSV）",
        "files": ["data/real/cic_ids2018"],
        "label": "Label",
        "time": "Timestamp",
        "group": ["Src IP", "Dst IP", "Dst Port", "Protocol"],
        "day_from": True,
        "family": {},
        "clean": "light",
        "notes": ["镜像三个文件行数均为 1,048,575（Excel 上限），疑被截断；行序非时间序，需先排序",
                  "含大量空值/Infinity：按 clean_dataframe 的「任一特征缺失即丢行」实测只剩 0.002% 数据，"
                  "故本数据集走 light 轻清洗（特征缺失保留，由标准化承接）"],
    },
    # ---- 以下三份的特征体系与本工程 71 维不通用（映射 0/71），只能用「原生特征自训」口径 ----
    "unsw_nb15": {
        "title": "UNSW-NB15（官方 train/test，Zeek/Argus 短名 45 列）",
        "files": ["data/real/unsw_nb15"],
        "label": "label",            # 0/1
        "label_multi": "attack_cat",  # 9 类攻击族
        "time": None,
        "group": None,               # 该版本无 IP/端口列，做不了身份级组切分
        "day_from": True,            # train.csv / test.csv 即官方划分 → 用 cross_day 做文件级切分
        "family": {},
        "clean": "light",
        "notes": ["镜像的 train.csv = 82,332 行、test.csv = 175,341 行，与官方（175,341/82,332）相反，疑为文件名对调",
                  "特征映射 0/71：Zeek/Argus 体系与本工程 CICFlowMeter 71 维不通用，不得沿用现有权重"],
    },
    "ton_iot": {
        "title": "TON_IoT（Argus/Zeek 44 列，含 label 与 type 两级标签）",
        "files": ["data/real/ton_iot"],
        "label": "label",
        "label_multi": "type",
        "time": None,
        "group": ["src_ip", "dst_ip", "dst_port", "proto"],
        "day_from": None,
        "family": {},
        "clean": "light",
        "notes": ["特征映射 0/71：Argus 体系不通用", "无时间列，不能做在线告警口径评测"],
    },
    "cic_iot2023": {
        "title": "CIC-IoT2023（CICIoT2023 39 列，Benign + 8 族 DDoS，标签只在文件名）",
        "files": ["data/real/cic_iot2023"],
        "label": None,               # 数据里没有标签列 → 按文件名推断
        "label_multi": None,
        "time": None,
        "group": None,
        "day_from": None,
        "family": {},
        "clean": "light",
        "notes": ["标签由文件名推断（DDoS-SlowLoris / BenignTraffic …），已明确标注来源",
                  "特征映射 0/71：CICIoT2023 的 39 维不通用", "无时间列"],
    },
}

# CIC-IDS2017 / 2018 的原始标签 → 本工程 8 大类（含编码变体归一化）
FAMILY_RULES: list[tuple[str, str]] = [
    (r"^benign$", "Normal"),
    (r"^(dos hulk|dos goldeneye|dos slowloris|dos slowhttptest|dos attacks-hulk|dos attacks-goldeneye"
     r"|dos attacks-slowloris|dos attacks-slowhttptest|ddos attacks-loic-http|ddos attacks-loic-udp"
     r"|ddos attacks-hoic|ddos attacks-slowhttptest)$", "DoS"),
    (r"^(ddos|ddos attack-hoic|ddos attack-loic-udp|ddos attack-loic-http)$", "DDoS"),
    (r"^(portscan|port scan)$", "Probe"),
    (r"^(ftp-patator|ssh-patator|brute force -web|brute force -xss|ftp brute force|ssh brute force"
     r"|web brute force)$", "BruteForce"),
    (r"^web attack.*$", "WebAttack"),
    (r"^(bot|botnet.*)$", "Botnet"),
    (r"^infilteration$|^infiltration$", "Infiltration"),
    (r"^heartbleed$", "Heartbleed"),
    (r"^sql injection$", "WebAttack"),
]


def norm_label(s: str) -> str:
    s = str(s).strip()
    for bad in ("\u2013", "\u2014", "\x96", "\ufffd", "\u2019"):
        s = s.replace(bad, "-")
    return re.sub(r"\s+", " ", s).strip()


DAY_ORDER = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def day_rank(name: str) -> int:
    """按星期几排序（文件名里含 Monday..Friday）；找不到星期的（如 UNSW 的 train.csv/test.csv）
    返回统一秩并**按文件名兜底排序**——否则 fit/eval 的归属会随文件枚举顺序漂移，不可复现。"""
    low = str(name).lower()
    for i, d in enumerate(DAY_ORDER):
        if d in low:
            return i
    return len(DAY_ORDER)


def to_family(raw: str) -> str:
    t = norm_label(raw).lower()
    for pat, fam in FAMILY_RULES:
        if re.match(pat, t):
            return fam
    return "Other:" + norm_label(raw)


def load_build_mapping():
    spec = importlib.util.spec_from_file_location("_cd_map", ROOT / "tools" / "cross_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cd_map"] = mod
    spec.loader.exec_module(mod)
    return mod.build_mapping


def load_dataset(name: str, row_cap: int, log) -> tuple[pd.DataFrame, list[str], dict]:
    spec = DATASETS[name]
    paths: list[Path] = []
    for f in spec["files"]:
        p = ROOT / f
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.csv")))
        elif p.exists():
            paths.append(p)
    if not paths:
        raise SystemExit(f"找不到 {name} 的数据文件：{spec['files']}")
    frames = []
    per_file_rows = max(row_cap // max(len(paths), 1), 1000) if row_cap else 0
    for p in paths:
        df = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(p, low_memory=False, encoding=enc)
                break
            except Exception:  # noqa: BLE001
                continue
        if df is None:
            raise SystemExit(f"读取失败：{p}")
        df.columns = [str(c).strip() for c in df.columns]
        if per_file_rows and len(df) > per_file_rows:
            # 确定性等间隔抽样：保留时间覆盖，不偏向文件头部
            idx = np.linspace(0, len(df) - 1, per_file_rows).astype(int)
            log(f"  {p.name}: {len(df):,} 行 → 等间隔抽 {per_file_rows:,} 行")
            df = df.iloc[idx].reset_index(drop=True)
        else:
            log(f"  {p.name}: {len(df):,} 行")
        df["__day__"] = p.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out, [p.name for p in paths], spec


def sig3(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = np.floor(np.log10(np.abs(a)))
        exp = np.where(np.isfinite(exp), exp, 0.0)
        out = np.where(a == 0, 0.0, np.round(a / 10.0 ** (exp - 2)) * 10.0 ** (exp - 2))
    return out


def row_hashes(mat: np.ndarray) -> set[str]:
    if mat.size == 0:
        return set()
    r = sig3(mat)
    return {hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest() for v in r}


def build_windows(X: np.ndarray, y: np.ndarray, side: np.ndarray,
                  window: int, stride: int, log) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只在**同一侧**的连续行上开窗（避免窗口跨切分边界造成泄漏）。

    返回 (Xw, yw, pure, w_side, starts)：w_side 由本函数直接给出，绝不在外部重算——
    外部重算一旦条件不一致会静默把窗口错配到另一侧。starts = 各窗口起始行号（在线口径需要时间轴）。
    """
    n = len(X)
    starts_all = np.arange(0, n - window + 1, stride)
    keep, pure = [], []
    for s in starts_all:
        sl = side[s:s + window]
        if sl[0] < 0 or sl.min() != sl.max():   # side<0 = 被 purge 掉的行，不参与
            continue
        keep.append(s)
        lab = y[s:s + window]
        pure.append(bool(lab.min() == lab.max()))
    keep = np.asarray(keep, dtype=int)
    w_side = side[keep].astype(np.int64) if len(keep) else np.zeros(0, dtype=np.int64)
    Xw = np.stack([X[s:s + window] for s in keep]) if len(keep) else np.zeros((0, window, X.shape[1]), np.float32)
    yw = np.array([int(y[s:s + window].mean() > 0.5) for s in keep], dtype=np.int64)
    pw = np.asarray(pure, dtype=bool)
    assert len(Xw) == len(yw) == len(pw) == len(w_side) == len(keep), "开窗输出长度不一致"
    log(f"  候选窗口 {len(starts_all):,} → 同侧窗口 {len(keep):,}（其中纯窗口 {int(pw.sum()):,}）")
    return Xw, yw, pw, w_side, keep


def scale_safe(scaler, X: np.ndarray, clip: float = 10.0) -> tuple[np.ndarray, float]:
    """标准化并做数值保护。

    两个必须挡住的陷阱（都会产出「看起来像发现」的假结果）：
    1. `np.nan_to_num` 默认把 ±inf 变成 1.8e308 —— 平方后直接溢出，
       马氏距离被少数维度劫持（实测过 AUC 掉到 0.0085 这种「完全反相关」的假象）；
    2. 流统计量重尾，均值/方差被离群点撑爆。因此统一 clip 到 ±clip，
       并回报被裁剪比例，让读者知道有多少值被动了。
    """
    flat = np.nan_to_num(np.asarray(X, dtype=np.float64).reshape(-1, X.shape[-1]),
                         nan=0.0, posinf=0.0, neginf=0.0)
    z = np.nan_to_num(scaler.transform(flat), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = float((np.abs(z) > clip).mean())
    return np.clip(z, -clip, clip).reshape(X.shape).astype(np.float32), clipped


def quantile_threshold(scores: np.ndarray, budget: float) -> float:
    """阈值 = 拟合段正常分数的 1−预算 分位（不需要任何标签信息之外的假设）。"""
    if scores.size == 0:
        return 1.0
    return float(np.quantile(scores, 1.0 - budget))


def binary_metrics_at(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"threshold": round(float(thr), 6), "recall": round(rec, 4), "precision": round(prec, 4),
            "f1": round(f1, 4), "fpr": round(fpr, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """在**评估集**上取「误报率 ≤ target」时能达到的最大召回（TPR@FPR）。

    这是标准评测量（项目既有指标表也用同一个），只用于**报告**，不参与任何模型/阈值选择，
    因此不构成泄漏；它的意义是：让各评测臂在**同一个实测误报率**下可比——
    否则「预算召回」会被各臂实际达到的 FPR 高低带偏。
    """
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(-s)
    ss, ys = s[order], y[order]
    fp = tp = 0
    best = 0.0
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        for k in range(i, j + 1):
            if ys[k] == 1:
                tp += 1
            else:
                fp += 1
        if fp / n0 <= target:
            best = max(best, tp / n1)
        i = j + 1
    return float(best)


def online_metrics(times: np.ndarray, y_win: np.ndarray, scores: np.ndarray,
                   thr: float, confirm: int) -> dict:
    """在线告警口径（**真实数据版**）：片段召回 / 误报每小时 / 检测时延。

    这是把工程原本只在合成数据上算的招牌指标，搬到真实数据 + 无泄漏切分上。

    * 按时间排序窗口；告警 = **连续 confirm 个窗口**都 ≥ 阈值（对应工程里的"连续确认"机制）；
    * 攻击片段 = 时间序上连续的攻击窗口，长度 ≥ MIN_SEG_WINDOWS（默认 2）个窗口才计入分母。
      ⚠️ 这里与合成域**只是同一种做法，不是同一个阈值**：合成域按**行数**判定
      （`max(window_size, online_min_segment_rows)` 行），本工具按**窗口数**判定，
      因为真实数据侧只有窗口级标签、拿不到行级标签。两者都丢弃过短片段，
      但分母口径不同，跨数据集的片段召回**不能直接对比**（误报口径已对齐，见函数末尾说明）；
    * 片段召回 = 至少产生过一次告警的片段占比；
    * 检测时延 = 片段首窗时间 → 该片段内首个告警窗时间；
    * 误报 = 不落在任何攻击片段内的告警数 ÷ 评估段总时长（小时）。
    """
    order = np.argsort(times, kind="stable")
    t = np.asarray(times, dtype=float)[order]
    y = np.asarray(y_win)[order]
    s = np.asarray(scores, dtype=float)[order]
    above = s >= thr
    alarm = np.zeros(len(s), dtype=bool)
    if confirm <= 1:
        alarm = above.copy()
    else:
        run = 0
        for i in range(len(s)):
            run = run + 1 if above[i] else 0
            if run >= confirm:
                alarm[i] = True

    segs: list[tuple[int, int]] = []
    raw_segs: list[tuple[int, int]] = []
    i = 0
    while i < len(y):
        if y[i] == 1:
            j = i
            while j + 1 < len(y) and y[j + 1] == 1:
                j += 1
            raw_segs.append((i, j))
            if (j - i + 1) >= MIN_SEG_WINDOWS:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1

    hits, lats = 0, []
    alarm_in_seg = np.zeros(len(s), dtype=bool)
    for a, b in segs:
        idx = np.flatnonzero(alarm[a:b + 1])
        if len(idx):
            hits += 1
            alarm_in_seg[a + idx] = True
            lats.append(float(t[a + idx[0]] - t[a]))
    span_h = (t[-1] - t[0]) / 3600.0 if len(t) > 1 else float("nan")
    # 时间跨度可信度自检：公开集的时间戳跨度常被数据集构造拉长（空档极多）。
    # 用「相邻窗口时间差的中位数 × 窗口数」估一个"活跃时长"，若跨度远大于它，说明跨度不可信。
    dt = np.diff(t)
    # 用 75 分位间隔而不是中位数：公开集时间戳常为分钟粒度，大量相邻窗口间隔为 0，
    # 中位数会被压成 0 而把可信数据误判为不可信（实测 CIC-IDS2018 就被误判过）。
    gap = float(np.percentile(dt, 75)) if len(dt) else float("nan")
    implied_h = (gap * len(t) / 3600.0) if gap == gap else float("nan")
    # 可信度判据改为**显式的时长上限**：公开集的时间戳跨度常被数据集构造拉长
    # （CIC-IDS2017 约 1,476 h、inSDN 约 6,733 h，而实测抓包只有几小时到几天）。
    # 用"跨度 ≤ 7 天"作为可信门槛，简单、可解释、可复现；超过就不做运维换算。
    reliable = bool(span_h == span_h and 0 < span_h <= SPAN_LIMIT_HOURS)
    # ⚠️ 误报口径必须与合成域**完全一致**。合成域（src/evaluate.py::online_alarm_evaluation）
    # 的误报定义是「该告警窗内不含任何攻击行」——只看**窗口自身**的标签。
    # 而「告警不落在保留片段内」会把落在**被丢弃的过短片段**上的告警也算成误报：
    # 同一物理情形在两边得到相反判定，导致真实侧误报被系统性抬高、两组数字不可比。
    fa = int((alarm & (y == 0)).sum())
    fa_legacy = int((alarm & ~alarm_in_seg).sum())   # 旧口径，仅保留用于量化差异
    fa_per_h = (fa / span_h) if (reliable and span_h > 0) else None
    fa_per_h_active = (fa / implied_h) if (implied_h and implied_h > 0) else None
    ops = None
    if fa_per_h is not None:      # 仅在时间跨度可信时做运维换算
        ops = {"false_alarms_per_day": round(fa_per_h * 24, 1),
               "analyst_hours_per_day": round(fa_per_h * 24 * 6 / 60, 2),
               "basis": "6 分钟/告警、24 小时/天（与 tools/ops_impact.py 同口径）"}
    return {
        "confirm": int(confirm), "threshold": round(float(thr), 6),
        "windows": int(len(s)), "segments": len(segs), "segments_hit": int(hits),
        "segment_recall": round(hits / len(segs), 4) if segs else None,
        "alarms": int(alarm.sum()), "false_alarms": fa,
        "false_alarms_legacy": fa_legacy,
        "n_raw_attack_segments": len(raw_segs),
        "n_ignored_short_segments": len(raw_segs) - len(segs),
        "min_segment_windows": int(MIN_SEG_WINDOWS),
        "false_alarms_per_hour": round(fa_per_h, 3) if fa_per_h is not None else None,
        "false_alarms_per_hour_active": round(fa_per_h_active, 3) if fa_per_h_active else None,
        "latency_s_mean": round(float(np.mean(lats)), 2) if lats else None,
        "latency_s_median": round(float(np.median(lats)), 2) if lats else None,
        "eval_hours": round(span_h, 2) if span_h == span_h else None,
        "window_gap_q75_s": round(gap, 2) if gap == gap else None,
        "implied_hours": round(implied_h, 2) if implied_h == implied_h else None,
        "time_span_reliable": reliable,
        "ops": ops,
    }


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 并列分数取平均秩
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="真实数据无泄漏评测")
    ap.add_argument("--dataset", default="insdn", choices=sorted(DATASETS))
    ap.add_argument("--protocol", default="block", choices=["block", "group", "cross_day", "family_holdout", "stratified", "chunked"],
                    help="切分协议：block=连续段前后切分（默认）；group=按流身份组；cross_day=按天；"
                         "family_holdout=留出攻击族；stratified=按标签分层随机；"
                         "chunked=按块切分（类别成块+无时间列的数据集用它，常与 --shuffle 配合）")
    ap.add_argument("--holdout-families", nargs="*", default=None,
                    help="family_holdout 协议要留出的攻击族名（缺省取前两个非 benign 家族）")
    ap.add_argument("--shuffle", action="store_true",
                    help="切分前按 seed 打乱行序（用于类别成块且无时间列的数据集，如 TON_IoT；"
                         "行级随机切分与滑窗不兼容，必须先打乱再用 block 协议）")
    ap.add_argument("--budget", type=float, default=0.01, help="主指标误报预算（默认 1%%）")
    ap.add_argument("--fit-frac", type=float, default=0.4, help="group 协议下拟合段占比")
    ap.add_argument("--fit-days", type=int, default=2, help="cross_day 协议下用前几天做拟合段（默认 2）")
    ap.add_argument("--native", action="store_true",
                    help="额外跑「原生特征自训」臂：用该数据自己的数值列 + 逻辑回归（特征映射 0/71 时必须用这个）")
    ap.add_argument("--online", action="store_true",
                    help="额外算**在线告警口径**（片段召回 / 误报每小时 / 检测时延），需该数据集有时间列")
    ap.add_argument("--confirms", type=int, nargs="+", default=[1, 2, 3, 5],
                    help="在线口径的连续确认窗口数 N（默认 1 2 3 5；N 越大误报越少、时延越大）")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--row-cap", type=int, default=250000, help="总行数上限（0=不限；按文件等间隔抽样）")
    ap.add_argument("--clean", choices=["pipeline", "light"], default=None,
                    help="清洗方式；缺省按数据集登记表（cic_ids2018 为 light）")
    ap.add_argument("--dedup-digits", type=int, default=3,
                    help="近似去重的有效数字位数（与项目主实验同一开关口径；0=只做精确去重）")
    ap.add_argument("--max-windows", type=int, default=60000,
                    help="候选窗口**总数**上限（对整体候选窗口等间隔抽样；此前误写为「每侧」）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true",
                    help="允许用不同参数覆盖同 tag 的既有记录（默认拒绝，防止静默丢结果）")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for k, v in DATASETS.items():
            print(f"{k:14} {v['title']}")
            print(f"   文件: {v['files']}  协议可用: {'group, cross_day' if v['day_from'] else 'group'}")
            for n in v.get("notes", []):
                print(f"   ! {n}")
        return 0

    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.time()
    tag = args.tag or f"{args.dataset}_{args.protocol}"
    log(f"=== 真实数据无泄漏评测：{args.dataset} / {args.protocol}")
    log(f"数据：{DATASETS[args.dataset]['title']}")
    for n in DATASETS[args.dataset].get("notes", []):
        log(f"  ! {n}")

    df, files, spec = load_dataset(args.dataset, args.row_cap, log)
    log(f"读入 {len(df):,} 行 × {len(df.columns)} 列（{len(files)} 个文件）")

    # ---- 标签归一化 ----
    # 两种口径：① 0/1 编码列（UNSW-NB15/TON_IoT/inSDN）→ 直接当二分类，细类取 label_multi；
    #            ② 带类别名的列（CIC-IDS2017/2018）或多分类列 → 按家族规则归一化。
    if spec.get("label") and spec["label"] in df.columns:
        lb = df[spec["label"]]
        as_num = pd.to_numeric(lb, errors="coerce")
        is_binary_code = bool(as_num.isin([0, 1]).mean() > 0.99)
    else:
        is_binary_code = False

    if is_binary_code:
        ybin = (as_num.fillna(0) > 0).astype(int)
        multi = spec.get("label_multi")
        if multi and multi in df.columns:
            fams = df[multi].astype(str).map(norm_label)
            fams = np.where(ybin == 1, np.where(fams.isin(["", "nan", "None"]), "Attack", fams), "Normal")
        else:
            fams = np.where(ybin == 1, "Attack", "Normal")
        df["__y__"] = ybin.to_numpy()
        df["__fam__"] = fams
        df["__raw_label__"] = df[spec["label"]].astype(str).map(norm_label)
        n_missing = int(as_num.isna().sum())
        log(f"标签口径：0/1 编码列 `{spec['label']}`（细类来自 `{multi}`）｜缺失标签 {n_missing:,} 行")
    else:
        if spec.get("label") and spec["label"] in df.columns:
            raw_labels = df[spec["label"]].astype(str)
            df["__raw_label__"] = raw_labels.map(norm_label)
            df["__fam__"] = (df["__raw_label__"].map(lambda x: spec["family"].get(x, "Other:" + x))
                             if spec["family"] else df["__raw_label__"].map(to_family))
        else:
            # 无标签列 → 按文件名推断（CIC-IoT2023 的攻击族只在文件名里）
            stem = df["__day__"].astype(str).str.split(".").str[0]
            df["__raw_label__"] = stem
            df["__fam__"] = stem.str.replace(r"\d+$", "", regex=True)
        n_missing = int(df["__raw_label__"].str.lower().isin(["nan", "", "none"]).sum())
        # Normal 判定：家族名含 benign/normal 的一律算正常（CIC-IoT2023 的 benign 文件名是
        # BenignTraffic，不识别会把攻击先验算成 1.0000 —— 那是会直接把结论带偏的错误）
        is_normal = df["__fam__"].astype(str).str.lower().str.contains("benign|normal", regex=True)
        df["__y__"] = (~is_normal).astype(int)
        if n_missing:
            log(f"  ! 缺失标签 {n_missing:,} 行，已剔除（不得当成一个类别）")
            df = df[~df["__raw_label__"].str.lower().isin(["nan", "", "none"])].reset_index(drop=True)
            is_normal = df["__fam__"].astype(str).str.lower().str.contains("benign|normal", regex=True)
            df["__y__"] = (~is_normal).astype(int)

    fam_counts = df["__fam__"].value_counts()
    log("家族分布（归一化后）：" + "、".join(f"{k}={v:,}" for k, v in fam_counts.head(12).items()))
    log(f"二分类先验：攻击 {df['__y__'].mean():.4f}")

    # ---- 去重（与项目口径一致）----
    cfg = Config()
    before = len(df)
    clean_mode = args.clean or spec.get("clean", "pipeline")
    if clean_mode == "light":
        # 轻清洗：只丢「标签缺失」与「精确重复行」，**保留特征缺失**。
        # 理由：CIC-IDS2018 含大量空值/Infinity，而 clean_dataframe 的策略是
        # 「任一特征缺失即丢行」——实测 5,000 行只剩 129 行、全量只剩 7 行，
        # 会把数据集清空。特征缺失改由后面的标准化承接：映射不到的维度填 0
        # （标准化空间 = 拟合段/合成域均值），既不丢行也不引入评估段信息。
        df2 = df.dropna(subset=[spec["label"]]).copy() if spec.get("label") else df.copy()
        df2 = df2.replace([np.inf, -np.inf], np.nan)
        feat_cols_all = [c for c in df2.columns if not c.startswith("__") and c != spec["label"]]
        df2 = df2.drop_duplicates(subset=feat_cols_all, keep="first").reset_index(drop=True)
        note = f"light（手工轻清洗）｜特征缺失保留待标准化承接，缺失率 {float(df2[feat_cols_all].isna().to_numpy().mean()):.2%}"
        log(f"清洗（light）：{before:,} → {len(df2):,} 行  {note}")
    else:
        df2, _, _, note = clean_dataframe(df, cfg)
        log(f"去重（clean_dataframe，dedup={cfg.dedup_significant_digits}）：{before:,} → {len(df2):,} 行"
            f"（{-100*(1-len(df2)/max(before,1)):.1f}%）  {note}")

    # ---- 特征映射 ----
    build_mapping = load_build_mapping()
    real_cols = [c for c in df2.columns if not c.startswith("__")]
    mapping, mp_info = build_mapping(FEATURES, real_cols)
    log(f"特征映射：{len(mapping)}/{len(FEATURES)}（{len(mapping)/len(FEATURES)*100:.1f}%）")
    if not mapping and not args.native:
        raise SystemExit("特征映射为 0：这份数据的特征体系与本工程 71 维不通用，"
                         "不能用现有模型评测；若要评测请加 --native（用其原生特征自训）")
    if not mapping:
        log("  ! 映射为 0：本工程 71 维不可用，仅出「原生特征自训」臂的结果（--native）")

    # ---- 全局近似去重（在**映射后的特征子集**上，N 位有效数字）----
    # 放在切分之前：这样「训练/评估近重复交集 = 0」是构造上的保证，
    # 后面的校验才有意义（用 3 位有效数字与项目主实验同一开关口径）。
    mapped_cols = [mapping[f] for f in FEATURES if f in mapping]
    if not mapped_cols:
        # 映射为空时若继续用空特征列做哈希，所有行会被判成"同一行"→ 数据被压成 1 行。
        # 这种静默毁数据的路径必须堵死：退回用该数据集自己的数值列做去重键与泄漏校验键。
        mapped_cols = [c for c in df2.columns if not c.startswith("__")
                       and c != (spec.get("label") or "") and pd.api.types.is_numeric_dtype(df2[c])]
        log(f"  ! 无映射特征：近似去重与泄漏校验改用该数据集自己的 {len(mapped_cols)} 个数值列")
    if not mapped_cols:
        raise SystemExit("既没有可映射特征、也没有数值列：无法去重与泄漏校验，拒绝出结论")
    raw_feat_all = df2[mapped_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    dig = args.dedup_digits
    if dig > 0:
        seen: set[str] = set()
        keep_mask = np.zeros(len(raw_feat_all), dtype=bool)
        rounded = sig3(raw_feat_all)
        for i, v in enumerate(rounded):
            h = hashlib.md5(np.ascontiguousarray(np.nan_to_num(v)).tobytes()).hexdigest()
            if h not in seen:
                seen.add(h)
                keep_mask[i] = True
        dropped = int((~keep_mask).sum())
        log(f"近似去重（{dig} 位有效数字，特征子集 {len(mapped_cols)} 列）："
            f"{len(df2):,} → {int(keep_mask.sum()):,} 行（去掉 {dropped:,}，{dropped/max(len(df2),1):.1%}）")
        df2 = df2[keep_mask].reset_index(drop=True)
    else:
        log("近似去重：已关闭（digits=0）")

    # ---- 可选：打乱行序 ----
    # 用途：类别成块且**无时间列**的数据集（如 TON_IoT），行序本身没有含义，
    # 而"行级随机/分层切分"与滑窗**不兼容**（窗口需要 16 条连续同侧行——实测 43,618 个候选窗口
    # 只剩 11 个）。先按固定种子打乱行序，再用 block 协议切分，等价于一次可复现的随机切分。
    if args.shuffle:
        df2 = df2.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        log(f"已按 seed={args.seed} 打乱行序（数据集无时间列，行序无含义；近重复已全局去重，"
            "打乱不会引入行级泄漏，切分后仍会校验）")

    # ---- 切分 ----
    rng = np.random.default_rng(args.seed)
    gcol = [c for c in (spec.get("group") or []) if c in df2.columns]
    if args.protocol == "group" and not gcol:
        raise SystemExit(f"{args.dataset} 没有流身份列（{spec.get('group')}），无法做组切分："
                         "请改用 --protocol block（连续段前后切分），不要假装做了身份级隔离")
    gkey = df2[gcol].astype(str).agg("|".join, axis=1) if gcol else pd.Series(["__nogroup__"] * len(df2))
    if args.protocol == "cross_day" and spec["day_from"]:
        # 按真实星期顺序排（文件名里含 Monday..Friday），不能按字母序——
        # 字母序会把 Friday 排到最前，得到「用周五训练、用周一~周四评测」这种没有依据的口径。
        days = sorted(df2["__day__"].unique(), key=lambda n: (day_rank(n), str(n)))
        fit_days = set(days[:max(1, args.fit_days)])
        side = np.where(df2["__day__"].isin(fit_days), 0, 1).astype(np.int64)
        log(f"按天切分（时间顺序）：拟合段 {[d[:38] for d in days[:max(1, args.fit_days)]]}")
        log(f"                   评估段 {[d[:38] for d in days[max(1, args.fit_days):]]}")
    elif args.protocol == "block":
        # 连续同标签段内前后切分 + purge 间隔：保住连续性（窗口可开），
        # 又让两侧都含正常与攻击（不退化）。近重复由全局去重保证，切分后仍会校验。
        y_all = df2["__y__"].to_numpy()
        runs, start = [], 0
        for i in range(1, len(y_all) + 1):
            if i == len(y_all) or y_all[i] != y_all[start]:
                runs.append((start, i))
                start = i
        side = np.full(len(df2), -1, dtype=np.int64)
        used_runs = 0
        for a, b in runs:
            n = b - a
            if n < 2 * args.window:
                continue
            k = a + int(n * args.fit_frac)
            side[a:k] = 0
            side[k:b] = 1
            p = min(args.window, n // 4)
            side[max(a, k - p):min(b, k + p)] = -1     # purge 间隔
            used_runs += 1
        log(f"按连续段前后切分：{len(runs)} 段 → 采用 {used_runs} 段；"
            f"purge 间隔 {min(args.window, 4)} 行；未使用（purge/过短）行 {int((side < 0).sum()):,}")
    elif args.protocol == "family_holdout":
        # 留出攻击族：把指定的若干攻击族整体放到评估段，其余攻击族与一半正常流量留在拟合段。
        # 适用：逐攻击族一个文件的数据集（如 CIC-IoT2023），能回答"没见过这类攻击能不能发现"。
        fams = df2["__fam__"].astype(str)
        is_norm = fams.str.lower().str.contains("benign|normal", regex=True).to_numpy()
        held = list(args.holdout_families or [])
        if not held:
            pool = [f for f in fams.value_counts().index if not str(f).lower().startswith(("benign", "normal"))]
            held = list(pool[:2])
        held = [f for f in held if (fams == f).any()]
        if not held:
            raise SystemExit(f"family_holdout 没命中任何家族；可用家族：{sorted(fams.unique())[:10]}")
        side = np.zeros(len(df2), dtype=np.int64)
        idx_norm = np.flatnonzero(is_norm)
        side[idx_norm[len(idx_norm) // 2:]] = 1          # 一半正常流量进评估段（否则评估段无正常窗口）
        side[np.flatnonzero(fams.isin(held).to_numpy())] = 1
        log(f"留出攻击族切分：评估段 = {held} + 后半正常流量；拟合段 = 其余攻击族 + 前半正常流量")
        for nm, m in (("拟合段", side == 0), ("评估段", side == 1)):
            vs = fams[m].value_counts()
            log(f"  {nm}家族： " + "、".join(f"{k}={v}" for k, v in list(vs.items())[:8]))
    elif args.protocol == "stratified":
        # 按标签分层随机切分：适用于类别成块、无时间列的数据集（如 TON_IoT）。
        rng2 = np.random.default_rng(args.seed)
        y_all = df2["__y__"].to_numpy()
        side = np.zeros(len(df2), dtype=np.int64)
        for v in np.unique(y_all):
            idx = np.flatnonzero(y_all == v)
            perm = rng2.permutation(len(idx))
            side[idx[perm[int(len(idx) * args.fit_frac):]]] = 1
        log(f"按标签分层随机切分（seed={args.seed}，拟合 {args.fit_frac:.0%}）："
            f"各类别内部按比例分配，避免类别成块导致两侧分布失衡")
    elif args.protocol == "chunked":
        # 按"块"切分：把（通常已打乱的）行序切成 ≥4×窗口 的连续块，整块随机分配给拟合/评估。
        # 为什么需要它：窗口需要 ≥window 条**连续同侧**行，而类别成块的数据集若做行级随机切分，
        # 同侧连续段会退化到几乎没有（实测 TON_IoT：43,618 个候选窗口只剩 11 个；
        # 打乱后 block 协议更是只剩 17 行）。块切分同时满足"连续"与"类别均衡"。
        csize = max(int(args.window * 4), int(args.window) + 1)
        n_chunks = len(df2) // csize
        if n_chunks < 10:
            raise SystemExit(f"chunked 协议：数据只够切出 {n_chunks} 个块（块大小 {csize} 行），不足以评测")
        rng3 = np.random.default_rng(args.seed)
        assign = rng3.random(n_chunks) < args.fit_frac
        side = np.full(len(df2), -1, dtype=np.int64)
        for i in range(n_chunks):
            side[i * csize:(i + 1) * csize] = 0 if assign[i] else 1
        log(f"按块切分：块大小 {csize} 行 × {n_chunks} 块（seed={args.seed}，拟合 {int(assign.sum())} 块）"
            f"；尾部 {len(df2) - n_chunks * csize} 行按 purge 丢弃")
    else:
        h = gkey.map(lambda s: int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % 1000)
        side = np.where(h < int(args.fit_frac * 1000), 0, 1).astype(np.int64)
        log(f"按流身份组切分：拟合段 {args.fit_frac:.0%} / 评估段 {1-args.fit_frac:.0%}（组数 {gkey.nunique():,}）")
        log("  ! 哈希组切分会打散连续行：同侧窗口会大幅减少，极端情况下拟合段可能没有正常窗口")
    df2 = df2[side >= 0].reset_index(drop=True)
    side = side[side >= 0]
    log(f"可用于开窗的行：{len(df2):,}（拟合 {int((side==0).sum()):,} / 评估 {int((side==1).sum()):,}）")

    # ---- 泄漏校验（必须全为 0）----
    raw_feat = df2[mapped_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    fit_rows = np.where(side == 0)[0]
    ev_rows = np.where(side == 1)[0]
    grp_fit = set(gkey.iloc[fit_rows]); grp_ev = set(gkey.iloc[ev_rows])
    grp_overlap = len(grp_fit & grp_ev)
    dup_overlap = len(row_hashes(raw_feat[fit_rows]) & row_hashes(raw_feat[ev_rows]))
    log(f"泄漏校验：组交集 = {grp_overlap}｜近重复行交集 = {dup_overlap}（行级必须 0）")
    if dup_overlap:
        raise SystemExit(f"切分存在**行级**泄漏（近重复 {dup_overlap} 行横跨两侧），拒绝出结论")
    if args.protocol == "group" and grp_overlap:
        raise SystemExit(f"group 协议下组交集应为 0，实测 {grp_overlap}：切分实现有 bug，拒绝出结论")
    if args.protocol != "group" and grp_overlap:
        log(f"  ! 身份级重叠 {grp_overlap} 个组（{args.protocol} 协议固有：同一主机/端口在两侧都出现过）。"
            "本协议保证的是**行级**无泄漏；该数字如实记录，不用它冒充身份级隔离。")

    for nm, idx in (("拟合段", fit_rows), ("评估段", ev_rows)):
        yy = df2["__y__"].to_numpy()[idx]
        log(f"  {nm}：{len(idx):,} 行，攻击占比 {yy.mean():.4f}；家族 "
            + "、".join(f"{k}={v}" for k, v in df2["__fam__"].iloc[idx].value_counts().head(8).items()))

    # ---- 组窗口张量 ----
    X_raw = np.full((len(df2), len(FEATURES)), np.nan, dtype=np.float32)
    for j, f in enumerate(FEATURES):
        if f in mapping:
            X_raw[:, j] = pd.to_numeric(df2[mapping[f]], errors="coerce").to_numpy(np.float32)
    Xw, yw, pure, win_side, win_starts = build_windows(X_raw, df2["__y__"].to_numpy(), side,
                                                       args.window, args.stride, log)

    def cap(arr_list):
        n = len(arr_list[0])
        if not args.max_windows or n <= args.max_windows:
            return arr_list
        idx = np.linspace(0, n - 1, args.max_windows).astype(int)
        log(f"  窗口 {n:,} → 等间隔抽 {args.max_windows:,}（控制内存）")
        return [a[idx] for a in arr_list]

    Xw, yw, pure, win_side, win_starts = cap([Xw, yw, pure, win_side, win_starts])
    fit_w = win_side == 0
    ev_w = win_side == 1
    log(f"窗口：拟合段 {int(fit_w.sum()):,}（纯窗口 {int((pure & fit_w).sum()):,}）"
        f"｜评估段 {int(ev_w.sum()):,}（纯窗口 {int((pure & ev_w).sum()):,}）")

    # ---- 在线口径的时间轴（仅当该数据集有时间列时可用）----
    win_times = None
    if args.online:
        tcol = spec.get("time")
        if not tcol or tcol not in df2.columns:
            log("  ! --online 需要时间列，本数据集没有 → 跳过在线口径（窗口级指标不受影响）")
        else:
            ts = pd.to_datetime(df2[tcol], errors="coerce", format="mixed")
            secs = ts.astype("int64").to_numpy() / 1e9
            secs = np.where(np.isfinite(secs), secs, np.nan)
            win_times = secs[win_starts]
            bad = int(np.isnan(win_times).sum())
            if bad > len(win_times) * 0.02:
                log(f"  ! 时间解析失败窗口 {bad:,} 个（>2%）→ 关闭在线口径")
                win_times = None
            else:
                log(f"  在线口径时间轴就绪：窗口时间来自 `{tcol}`（不可解析 {bad} 个已剔除）")

    # ---- 各评测臂 ----
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    model = load_model_bundle(str(MAIN_MODEL))
    syn_scaler = joblib.load(SYN_SCALER)
    log(f"已加载主模型 {MAIN_MODEL.name} 与合成域标准化器")

    # 只用拟合段的**正常窗口**拟合目标域统计量（不需要标签之外的信息）
    fit_norm_rows = fit_w & (yw == 0)
    ev_pure = ev_w & pure
    if int(fit_norm_rows.sum()) == 0:
        raise SystemExit(
            f"拟合段没有正常窗口（拟合窗口 {int(fit_w.sum()):,}，其中正常 {int((fit_w & (yw == 0)).sum()):,}）："
            "此时阈值会退化成 1.0 并产出假的「召回 0」。请改用 --protocol block，"
            "或减小 --row-cap/换协议后重跑——绝不在此基础上出结论。")
    y_ev = yw[ev_pure]
    pure_short = int(ev_pure.sum()) < 500
    if pure_short:
        # 类别在行级交错的数据集（如打乱后的 TON_IoT）几乎产生不了纯窗口：
        # P(全攻击)=0.73^16≈0.6%、P(全正常)=0.27^16≈1e-9。此时退化为**多数票口径**
        # （用全部评估窗口，标签取窗口内多数类），与工程既有的"纯窗口/多数票"双口径一致。
        log(f"  ! 评估段纯窗口仅 {int(ev_pure.sum())} 个（类别行级交错）→ 退化为**多数票口径**："
            f"用全部 {int(ev_w.sum()):,} 个评估窗口，标签取窗口内多数类")
        ev_pure = ev_w
        y_ev = yw[ev_pure]
    log(f"评估用窗口 {int(ev_pure.sum()):,}（攻击占比 {y_ev.mean():.4f}）"
        + ("（多数票口径）" if pure_short else "（纯窗口口径）"))

    results: dict = {}
    store: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add_arm(name: str, fit_scores: np.ndarray, ev_scores: np.ndarray, desc: str) -> None:
        entry = {"desc": desc}
        for b in (args.budget, 0.001):
            thr = quantile_threshold(fit_scores, b)
            m = binary_metrics_at(y_ev, ev_scores, thr)
            entry[f"at_{b:g}"] = m
        entry["roc_auc"] = round(roc_auc(y_ev, ev_scores), 4) if len(np.unique(y_ev)) > 1 else None
        entry["tpr_at_fpr1pct"] = round(tpr_at_fpr(y_ev, ev_scores, 0.01), 4)
        entry["tpr_at_fpr0p1pct"] = round(tpr_at_fpr(y_ev, ev_scores, 0.001), 4)
        # 按组分数分布：这是解释「AUC < 0.5（反转）」的关键诊断——
        # 若 eval 正常窗口的分数高于 eval 攻击窗口，排序本身就反了，
        # 根因通常是目标域正常流量画像漂移（跨日/跨场景），而不是模型"学到了反向规律"。
        m_fit_n = float(np.median(fit_scores)) if fit_scores.size else float("nan")
        m_ev_n = float(np.median(ev_scores[y_ev == 0])) if int((y_ev == 0).sum()) else float("nan")
        m_ev_a = float(np.median(ev_scores[y_ev == 1])) if int((y_ev == 1).sum()) else float("nan")
        entry["score_median"] = {"fit_normal": round(m_fit_n, 4), "eval_normal": round(m_ev_n, 4),
                                 "eval_attack": round(m_ev_a, 4),
                                 "attack_minus_normal": round(m_ev_a - m_ev_n, 4),
                                 "inverted": bool(m_ev_a < m_ev_n)}
        # 在线告警口径（片段召回/误报每小时/检测时延）——真实数据版
        # 阈值仍只来自拟合段正常分数；确认窗口数 N 只改变"连续多少次才算告警"，不引入评估段信息。
        if win_times is not None:
            ev_idx = np.flatnonzero(ev_pure)
            t_ev = win_times[ev_idx]
            keep_ok = ~np.isnan(t_ev)
            if keep_ok.sum() >= 10:
                entry["online"] = {}
                for c in args.confirms:
                    entry["online"][str(c)] = online_metrics(
                        t_ev[keep_ok], y_ev[keep_ok], np.asarray(ev_scores)[keep_ok],
                        quantile_threshold(fit_scores, args.budget), c)
                best = entry["online"][str(args.confirms[0])]
                log(f"      在线口径(N={best['confirm']}): 片段召回 {best['segment_recall']}"
                    f"｜误报 {best['false_alarms_per_hour']}/h｜时延中位 {best['latency_s_median']}s"
                    f"｜片段 {best['segments_hit']}/{best['segments']}｜{best['eval_hours']}h")
        results[name] = entry
        store[name] = (np.asarray(fit_scores, dtype=np.float64), np.asarray(ev_scores, dtype=np.float64))
        main = entry[f"at_{args.budget:g}"]
        log(f"  [{name}] 预算 {args.budget:.1%} → 召回 {main['recall']:.4f}｜精确率 {main['precision']:.4f}"
            f"｜F1 {main['f1']:.4f}｜实测误报率 {main['fpr']:.4f}"
            f"｜TPR@FPR1% {entry['tpr_at_fpr1pct']:.4f}｜AUC {entry['roc_auc']}")

    # A0：合成域标准化器 + 主模型（零样本迁移）
    # 映射为 0 时 X 全是填 0 的常数 → 模型对每个窗口给同一个分数，
    # 阈值等于该常数 → 全部判为攻击（实测误报率 1.0000、AUC 0.5）。
    # 那是**纯假象**，必须整臂跳过，不允许出现在结果里。
    skip_deep = (len(mapping) == 0)
    if skip_deep:
        log("  ! 映射为 0：跳过全部依赖本工程 71 维的评测臂（否则得到常数输入 → 全判攻击的假象）")
    X_syn, clip_syn = scale_safe(syn_scaler, Xw)
    log(f"  合成域标准化：被裁剪(±10)比例 {clip_syn:.4%}")
    _, p_syn = predict_probs(model, X_syn)
    if not skip_deep:
        add_arm("zeroshot_syn_scaler", p_syn[fit_norm_rows], p_syn[ev_pure],
                "合成域 scaler + 主模型权重，零样本迁移到目标域")

    # A1：在目标域重拟合标准化器（只用拟合段正常行）
    fit_norm_flat = np.nan_to_num(Xw[fit_norm_rows].reshape(-1, Xw.shape[-1]),
                                  nan=0.0, posinf=0.0, neginf=0.0)
    tgt_scaler = StandardScaler().fit(fit_norm_flat)
    X_tgt, clip_tgt = scale_safe(tgt_scaler, Xw)
    log(f"  目标域标准化：被裁剪(±10)比例 {clip_tgt:.4%}")
    _, p_tgt = predict_probs(model, X_tgt)
    if not skip_deep:
        add_arm("refit_scaler", p_tgt[fit_norm_rows], p_tgt[ev_pure],
                "在目标域正常流量上重拟合 scaler，权重不变")

    # A2/A3：无标签的目标域原生基线（稳健马氏 / 近邻距离）
    flat_tgt = X_tgt.reshape(len(X_tgt), -1)
    ref = flat_tgt[fit_norm_rows]
    # 用中位数 + MAD（而不是均值/方差）：流统计量重尾，均值/方差会被离群点撑爆，
    # 使距离对攻击不敏感（实测出现过 AUC≈0 的反相关假象）。
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0) * 1.4826 + 1e-3
    d2 = (((flat_tgt - med) / mad) ** 2).mean(axis=1)
    add_arm("normal_maha", d2[fit_norm_rows], d2[ev_pure],
            "只用拟合段正常窗口的稳健马氏距离（中位数+MAD，无监督、不需要任何标签）") if not skip_deep else None

    sub = np.where(fit_norm_rows)[0]
    if len(sub) > 3000:
        sub = sub[np.linspace(0, len(sub) - 1, 3000).astype(int)]
    nn = NearestNeighbors(n_neighbors=min(5, max(len(sub) - 1, 1))).fit(flat_tgt[sub])
    dk, _ = nn.kneighbors(flat_tgt)
    knn_score = dk[:, -1]
    add_arm("normal_knn", knn_score[fit_norm_rows], knn_score[ev_pure],
            "只用拟合段正常窗口的 kNN 距离（无监督）") if not skip_deep else None

    # A4：融合——把两个互补分数各自按拟合段正常分数做秩归一化后平均。
    # 动机：深度模型排序尚可（AUC 高）但固定预算下抓不到攻击；马氏在固定预算下能抓到。
    def rankify(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
        r = np.sort(np.asarray(ref, dtype=np.float64))
        return np.searchsorted(r, x, side="right") / max(len(r), 1)

    if "zeroshot_syn_scaler" in store and "normal_maha" in store:
        f1r, e1r = store["zeroshot_syn_scaler"]
        f2r, e2r = store["normal_maha"]
        add_arm("fusion_deep_maha",
                (rankify(f1r, f1r) + rankify(f2r, f2r)) / 2.0,
                (rankify(f1r, e1r) + rankify(f2r, e2r)) / 2.0,
                "深度迁移分数与目标域马氏分数各自秩归一化后等权平均（无监督、只看拟合段正常流量）")

    # A5：原生特征自训——用该数据集自己的数值列，不依赖本工程 71 维映射。
    # 映射覆盖率 0/71 的数据集（UNSW-NB15 / TON_IoT / CIC-IoT2023）只能走这个口径；
    # 它使用拟合段的标签（这是真实部署里"少量标注"的等价物），阈值仍只取拟合段正常分数的分位。
    if args.native:
        from sklearn.linear_model import LogisticRegression
        native_cols = [c for c in df2.columns
                       if not c.startswith("__") and c != (spec.get("label") or "")
                       and pd.api.types.is_numeric_dtype(df2[c])]
        if len(native_cols) < 3:
            log("  ! 原生臂跳过：可用数值列不足 3 个")
        else:
            Xn_raw = df2[native_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
            Xnw, yw_n, pure_n, side_n, _n_starts = build_windows(Xn_raw, df2["__y__"].to_numpy(), side,
                                                                 args.window, args.stride, log)
            Xnw, yw_n, pure_n, side_n, _n_starts = cap([Xnw, yw_n, pure_n, side_n, _n_starts])
            if not (np.array_equal(yw_n, yw) and np.array_equal(side_n, win_side)):
                raise SystemExit("原生臂的窗口与主臂不一致：拒绝出结论（切分实现有问题）")
            flat_n = np.nan_to_num(Xnw.reshape(-1, len(native_cols)), nan=0.0, posinf=0.0, neginf=0.0)
            sc_n = StandardScaler().fit(flat_n[(np.repeat(fit_norm_rows, args.window))])
            Z = np.clip(np.nan_to_num(sc_n.transform(flat_n), nan=0.0, posinf=0.0, neginf=0.0),
                        -10, 10).reshape(len(Xnw), -1)
            lr = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=1)
            lr.fit(Z[fit_w], yw[fit_w])
            s_nat = lr.predict_proba(Z)[:, 1]
            add_arm("native_lr", s_nat[fit_norm_rows], s_nat[ev_pure],
                    f"用该数据集自己的 {len(native_cols)} 个数值列 + 逻辑回归自训"
                    "（拟合段标签全用；特征映射 0/71 时这是唯一可行口径）")

    # ---- 产物 ----
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "tag": tag, "dataset": args.dataset, "protocol": args.protocol,
        "budget_primary": args.budget, "window": args.window, "stride": args.stride,
        "row_cap": args.row_cap, "seed": args.seed,
        "data": {"files": files, "rows_raw": int(before), "rows_after_dedup": int(len(df2)),
                 "duplicate_removed_frac": round(1 - len(df2) / max(before, 1), 4),
                 "families": {k: int(v) for k, v in fam_counts.items()},
                 "missing_labels_removed": n_missing},
        "leakage_check": {"group_overlap": grp_overlap, "near_duplicate_overlap": dup_overlap,
                          "fit_rows": int(len(fit_rows)), "eval_rows": int(len(ev_rows))},
        # 两侧的文件构成：block/cross_day 下若两侧共用同一文件，说明评测只反映
        # 「同场景内插值」，不能当作跨场景泛化能力（CIC-IoT2023 这类逐族文件尤其明显）。
        "files_per_side": {
            "fit": sorted(set(df2["__day__"].iloc[fit_rows].astype(str))),
            "eval": sorted(set(df2["__day__"].iloc[ev_rows].astype(str))),
            "shared": sorted(set(df2["__day__"].iloc[fit_rows].astype(str))
                             & set(df2["__day__"].iloc[ev_rows].astype(str))),
        },
        "mapping": {"matched": len(mapping), "coverage": round(len(mapping) / len(FEATURES), 4)},
        "windows": {"fit": int(fit_w.sum()), "eval": int(ev_w.sum()),
                    "fit_pure": int((pure & fit_w).sum()), "eval_pure": int(ev_pure.sum()),
                    "eval_attack_prior": round(float(y_ev.mean()), 4)},
        "arms": results,
        "elapsed_sec": round(time.time() - t0, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    jp = RESULT_DIR / "real_eval.json"
    old = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else {}
    # ⚠️ tag 只由「数据集_协议」构成，**不含参数指纹**（行数上限/步长/种子/…）。
    # 于是"用不同参数重跑同一个 tag"会**静默覆盖**已有结果，报告里的数字就和磁盘产物对不上了。
    # 本轮实测踩过一次：用默认 row-cap 覆盖了原次的 300000，段数/召回全变但没有任何提示。
    # 现在关键参数不一致时默认拒绝写入，必须显式 --force。
    prev = old.get(tag)
    mism = [f"{k}: {prev[k]} → {out.get(k)}"
            for k in ("dataset", "protocol", "row_cap", "stride", "window", "seed")
            if prev and k in prev and k in out and prev[k] != out[k]]
    if mism and not getattr(args, "force", False):
        log(f"✗ 拒绝覆盖 tag={tag!r}：已有记录的关键参数与本次不同——")
        for m in mism:
            log(f"    {m}")
        log("  tag 不含参数指纹，直接覆盖会让报告数字与产物对不上。")
        log("  确认要用新参数替换旧记录，请加 --force 重跑。")
        return 2
    old[tag] = out
    jp.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")

    # 追加式 markdown
    mp = RESULT_DIR / "real_eval.md"
    mp.write_text(render_md(old, args.budget), encoding="utf-8")

    log(f"\n产物：{jp}\n      {mp}（全量重生成，{len(old)} 个配置）\n耗时 {out['elapsed_sec']}s")
    return 0


def render_md(report: dict, budget: float) -> str:
    """从 JSON 全量重生成 markdown（幂等：重跑不会产生重复行）。"""
    L: list[str] = []
    L.append("# 真实数据无泄漏评测（主指标：给定误报预算下的攻击召回）\n")
    L.append("> 由 `python tools/real_eval.py` 生成，全部数字取自磁盘上的实测产物。")
    L.append("> **纪律**：先按项目口径去重（exact + 3 位有效数字），再切分；"
             "拟合段与评估段的**行级近重复交集必须为 0**（否则直接报错、不出结论）；"
             "判定阈值一律取「拟合段正常窗口分数」的 1−预算 分位，**从不使用评估段信息**。")
    L.append("> `TPR@FPR=1%` 与 `ROC-AUC` 是阈值无关的评测量（只用于报告、不参与选择），"
             "用来消除各臂**实测误报率不同**造成的不可比。")
    L.append("> 每份数据同时给出该工作点**实际达到**的误报率——预算在段间漂移下未必守得住，"
             "两个数一起看才算诚实。\n")
    for tag in sorted(report):
        e = report[tag]
        L.append(f"\n## {tag}\n")
        L.append(f"- 数据集：{e['dataset']}｜协议：`{e['protocol']}`｜窗口 T={e['window']} 步长 {e['stride']}"
                 f"｜行数上限 {e['row_cap']:,}｜种子 {e['seed']}")
        d = e["data"]
        L.append(f"- 读入 {d['rows_raw']:,} 行 → 去重后 {d['rows_after_dedup']:,} 行"
                 f"（去掉 {d['duplicate_removed_frac']:.1%}）｜缺失标签剔除 {d['missing_labels_removed']:,} 行")
        fam = "、".join(f"{k}={v:,}" for k, v in list(d["families"].items())[:8])
        L.append(f"- 家族分布：{fam}")
        lk = e["leakage_check"]
        L.append(f"- **泄漏校验**：行级近重复交集 **{lk['near_duplicate_overlap']}**（必须 0）"
                 f"｜身份级组交集 {lk['group_overlap']}"
                 f"（`block`/`cross_day` 协议固有，如实记录，不冒充身份级隔离）")
        L.append(f"- 特征映射 {e['mapping']['matched']}/{len(FEATURES)}"
                 f"（{e['mapping']['coverage']:.1%}）"
                 f"｜窗口：拟合 {e['windows']['fit']:,}（纯 {e['windows']['fit_pure']:,}）"
                 f"/ 评估 {e['windows']['eval']:,}（纯 {e['windows']['eval_pure']:,}，"
                 f"攻击占比 {e['windows']['eval_attack_prior']}）")
        L.append("")
        L.append("| 评测臂 | **TPR@FPR=1%** | ROC-AUC | 预算召回 | 该点实测误报率 | 精确率 | F1 | 0.1% 预算召回 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for name, a in e["arms"].items():
            m = a[f"at_{budget:g}"]
            L.append(f"| `{name}` | **{a['tpr_at_fpr1pct']}** | {a['roc_auc']} | {m['recall']} | {m['fpr']} | "
                     f"{m['precision']} | {m['f1']} | {a['at_0.001']['recall']} |")
        L.append("")
        L.append("| 评测臂说明 | |")
        L.append("|---|---|")
        for name, a in e["arms"].items():
            L.append(f"| `{name}` | {a['desc']} |")
        onl = {n: a["online"] for n, a in e["arms"].items() if a.get("online")}
        if onl:
            L.append("")
            L.append("**在线告警口径（真实数据；阈值同上面工作点，N = 连续确认窗口数）**")
            L.append("")
            L.append("| 评测臂 | N | 片段召回 | 检出/片段 | 误报次数 | 误报/小时 | 时延中位(s) | 评估时长(h) |")
            L.append("|---|---|---|---|---|---|---|---|")
            for name, d in onl.items():
                for c in sorted(d, key=int):
                    v = d[c]
                    L.append(f"| `{name}` | {c} | {v['segment_recall']} | {v['segments_hit']}/{v['segments']} | "
                             f"{v['false_alarms']} | {v['false_alarms_per_hour']} | {v['latency_s_median']} | "
                             f"{v['eval_hours']} |")
            L.append("")
            L.append("> 在线口径说明：告警 = 连续 N 个窗口分数越过阈值；攻击片段 = 时间序上连续的攻击窗口"
                     "（长度 < 2 个窗口不计入分母）；时延 = 片段首窗 → 该片段首个告警窗。")
            L.append(">")
            L.append("> ⚠️ **评估时长的坑**：公开集的时间戳跨度常远大于真实抓包时长"
                     "（如 CIC-IDS2017 / inSDN 的时间戳跨数月，系数据集构造所致），"
                     "因此这里的「误报/小时」分母偏大、数值偏乐观。**只可用于同一数据集内的横向比较，"
                     "不可与工程在合成测试流上按真实回放时长折算的误报率直接对比。**")
        L.append(f"\n*生成时间：{e.get('generated_at')}｜耗时 {e.get('elapsed_sec')}s*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
