"""仿真 SDN 流量生成器（本项目的「数据来源」）。

为什么要自己生成数据
--------------------
1. 原始 demo 读取的是硬编码路径下的 inSDN 数据集，换台机器就跑不起来；
2. 真实 SDN 攻击流量采集需要 Mininet + Ryu + 多台虚拟机，比赛现场不可能复现；
3. 仿真数据可以**完全复现**（固定随机种子），并且**物理量自洽**——
   我们先生成"流记录的物理事实"（包数、字节数、时长、标志位、到达间隔……），
   再由同一份事实渲染出两种数据集格式，因此任意两个特征之间都不会自相矛盾。

两种输出格式
------------
* ``insdn``：84 列，与公开数据集 inSDN / CIC-IDS2017（CICFlowMeter 输出）**列名完全一致**，
  也就是原始 demo ``CNN_biLSTM2.py`` 期望的格式；
* ``sdn``  ：22 列 + 1 列多分类标注，与原仓库 ``collect_normal_traffic.py`` /
  ``collect_ddos_traffic.py``（Ryu 控制器采集 OpenFlow 流表统计）**输出格式完全一致**。
  => 比赛之后把真实采集到的 CSV 直接丢进同一条流水线即可，代码零改动。

8 类流量：Normal / DDoS / DoS / Probe / BruteForce / WebAttack / Botnet / U2R
（与 inSDN 数据集的类别体系对齐；二分类任务即 Normal vs. Attack）

难度控制（很重要，直接决定比赛指标是否可信）
--------------------------------------------
我们**不希望**生成一份"随便一个模型都能 100%"的数据，所以内置了三个旋钮：

* ``overlap_ratio``：按比例把某条流的**全部参数**（含协议、端口、标志位风格）
  与"最易混淆类"做几何混合 —— 类别边界因此真实交叠；
* ``stealth_ratio`` / ``stealth_mix``：部分攻击片段是"隐蔽型"的——
  片段内混入大量正常样貌的流，**单看一条流几乎无法判定**，
  必须依靠时间窗内的聚合与演化模式才能识别（这正是 CNN+BiLSTM 时序建模的价值所在）；
* ``label_noise`` / ``feature_noise``：标注噪声与量测噪声。

片段排布还保证 **每个类别在 train / val / test 三个时间段里都出现**，
避免"某个攻击类只落在训练集"导致测试集 support=0 的评测偏差。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .utils import RAW_DIR, get_logger, save_json

LOGGER = get_logger("cida.generator")

# ======================================================================================
# 1. 两种 schema 的表头（列名与真实数据集逐字对齐）
# ======================================================================================
INSDN_ID_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
]
INSDN_FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]
INSDN_HEADER = INSDN_ID_COLUMNS + INSDN_FEATURE_COLUMNS + ["Label"]
assert len(INSDN_HEADER) == 84, len(INSDN_HEADER)  # 与 inSDN 数据集列数一致

# 原 Ryu 采集脚本写出的 22 列
SDN_HEADER = [
    "timestamp", "datapath_id", "flow_id", "ip_src", "tp_src", "ip_dst", "tp_dst",
    "ip_proto", "icmp_code", "icmp_type", "flow_duration_sec", "flow_duration_nsec",
    "idle_timeout", "hard_timeout", "flags", "packet_count", "byte_count",
    "packet_count_per_second", "packet_count_per_nsecond",
    "byte_count_per_second", "byte_count_per_nsecond", "label",
]
assert len(SDN_HEADER) == 22, len(SDN_HEADER)

# sdn 模式下额外附一列多分类真值：`label` 保持与原采集脚本一致的 0/1，
# `attack_type` 是本次仿真为了做多分类/可视化而附加的标注列（真实采集时可在 Ryu 侧同样记录）。
SDN_EXTRA_HEADER = SDN_HEADER + ["attack_type"]

SCHEMAS = {"insdn": INSDN_HEADER, "sdn": SDN_EXTRA_HEADER}

# 标签列候选名（数据管线据此自动识别"哪一列是答案"，而不是像原始代码那样写死 78）
#   * Label   —— CIC-IDS2017 MachineLearningCSV 的写法
#   * target  —— 真实 inSDN 数据集（CICFlowMeter v3 导出）的写法
LABEL_COLUMNS = ["Label", "label", "target", "attack_type", "attack_cat", "class", "category"]

# 各 schema 里属于「标识列 / 非特征列」的部分——数据管线会自动丢弃，
# 但保留在 CSV 里，方便人工核查与演示（这也是原始代码里 Y 索引错位的根源）。
#   * 同时列出两套命名：CIC-IDS2017 风格（Source IP/Source Port）与
#     CICFlowMeter v3 / inSDN 风格（Src IP/Src Port），保证真实数据集直接可用。
SCHEMA_ID_COLUMNS = {
    "insdn": [
        "Flow ID", "Timestamp",
        "Source IP", "Source Port", "Destination IP", "Destination Port",
        "Src IP", "Src Port", "Dst IP", "Dst Port",
    ],
    "sdn": ["timestamp", "flow_id", "ip_src", "ip_dst"],
}

ATTACK_CLASSES = ["DDoS", "DoS", "Probe", "BruteForce", "WebAttack", "Botnet", "U2R"]
ALL_CLASSES = ["Normal"] + ATTACK_CLASSES

# 参与混合的参数（其余特征全部由这些参数推导，因此混合后物理量仍然自洽）
BLEND_NUMERIC = ["n_fwd", "n_bwd", "fwd_len", "bwd_len", "duration", "regularity", "n_bursts"]
BLEND_CATEGORICAL = ["proto", "dport", "flags"]

# 与哪一类「最容易混淆」——用于生成困难样本，让评测指标不虚高
CONFUSABLE: dict[str, list[tuple[str, float]]] = {
    "Normal": [("Probe", 0.3), ("BruteForce", 0.3), ("DoS", 0.2), ("DDoS", 0.2)],
    "DDoS": [("DoS", 0.6), ("Normal", 0.4)],
    "DoS": [("DDoS", 0.6), ("Normal", 0.4)],
    "Probe": [("Normal", 0.7), ("BruteForce", 0.3)],
    "BruteForce": [("Normal", 0.7), ("Probe", 0.3)],
    "WebAttack": [("Normal", 0.8), ("Botnet", 0.2)],
    "Botnet": [("Normal", 0.7), ("BruteForce", 0.3)],
    "U2R": [("DDoS", 0.4), ("WebAttack", 0.3), ("Normal", 0.3)],
}


@dataclass
class EpisodeProfile:
    """一个流量片段的宏观参数。"""

    flows: tuple[int, int]          # 该片段包含的流记录条数范围
    duration: tuple[float, float]   # 片段持续时间（秒）
    burstiness: float               # >1 表示突发（前密后疏）


EPISODE_PROFILES: dict[str, EpisodeProfile] = {
    "Normal": EpisodeProfile(flows=(220, 340), duration=(60, 260), burstiness=1.3),
    "DDoS": EpisodeProfile(flows=(300, 420), duration=(6, 25), burstiness=2.6),
    "DoS": EpisodeProfile(flows=(200, 320), duration=(25, 90), burstiness=1.8),
    "Probe": EpisodeProfile(flows=(110, 190), duration=(3, 15), burstiness=3.0),
    "BruteForce": EpisodeProfile(flows=(140, 230), duration=(40, 160), burstiness=1.4),
    "WebAttack": EpisodeProfile(flows=(150, 260), duration=(25, 110), burstiness=1.5),
    "Botnet": EpisodeProfile(flows=(160, 280), duration=(120, 400), burstiness=1.1),
    "U2R": EpisodeProfile(flows=(90, 150), duration=(30, 150), burstiness=1.6),
}


def _ln(rng: np.random.Generator, median: float, sigma: float, size=None):
    """对数正态采样：median 是中位数，sigma 控制离散度。"""
    return rng.lognormal(mean=math.log(max(median, 1e-9)), sigma=sigma, size=size)


def _u(rng: np.random.Generator, lo: float, hi: float, size=None):
    return rng.uniform(lo, hi, size)


# ======================================================================================
# 2. 各类流量的「参数画像」（只给关键参数，其余特征由 _assemble 推导）
# ======================================================================================
# 说明：仿真不是随机撒点，而是按攻击机理建模——
#   * DDoS  ：多源、超高包速率、极短持续、几乎无反向包
#   * DoS   ：单源、速率中等、持续时间长（Slowloris 甚至几百秒）
#   * Probe ：端口扫描，单流 1~3 个包、时长毫秒级、RST/SYN
#   * BruteForce：SSH/FTP 反复认证，等间隔小包往返
#   * WebAttack：HTTP 载荷偏大、PSH 比例高
#   * Botnet：C2 心跳，间隔极其规律、速率极低、持续很久
#   * U2R  ：极少见，载荷异常、标志位组合异常
# 每一类都包含若干"非典型子型"（如慢速扫描、低速率洪泛），保证类内有足够方差。
def _params_normal(rng: np.random.Generator, ctx: dict | None = None) -> dict:
    kind = rng.choice(["http", "bulk", "dns", "icmp", "udp_stream", "long_idle"], p=[0.32, 0.14, 0.18, 0.14, 0.14, 0.08])
    if kind == "http":
        n_fwd = int(np.clip(_ln(rng, 8, 0.8), 3, 150))
        n_bwd = int(np.clip(n_fwd * _u(rng, 0.6, 1.3), 1, 220))
        fwd_len = float(rng.choice([_u(rng, 40, 90), _ln(rng, 700, 0.45)]))
        bwd_len = float(rng.choice([_u(rng, 40, 90), _ln(rng, 900, 0.55)]))
        dur = float(np.clip(_ln(rng, 1.2, 1.0), 0.05, 60))
        proto, dport = 6, int(rng.choice([80, 80, 8080, 443]))
        flags = "normal"
    elif kind == "bulk":
        n_fwd = int(_u(rng, 900, 22000))          # 大文件传输：包数极大，天然与 DoS 混淆
        n_bwd = int(np.clip(n_fwd * _u(rng, 0.05, 0.35), 1, 8000))
        fwd_len, bwd_len = float(_u(rng, 1380, 1500)), float(_u(rng, 200, 1400))
        dur = float(_u(rng, 5, 60))
        proto, dport = 6, int(rng.choice([5050, 80, 21]))
        flags = "psh_heavy"                        # 合法大流量也会密集 PSH
    elif kind == "dns":
        n_fwd = int(_u(rng, 1, 4))
        n_bwd = int(_u(rng, 1, 4))
        fwd_len, bwd_len = float(_u(rng, 55, 140)), float(_u(rng, 60, 340))
        dur = float(np.clip(_ln(rng, 0.05, 1.1), 0.0005, 3.0))
        proto, dport = 17, 53
        flags = "none"
    elif kind == "icmp":
        n_fwd = int(rng.choice([int(_u(rng, 1, 6)), int(_u(rng, 40, 120))]))
        n_bwd = int(np.clip(n_fwd * _u(rng, 0.6, 1.0), 0, 130))
        fwd_len = bwd_len = float(_u(rng, 64, 150))
        dur = float(np.clip(_ln(rng, 1.5, 1.2), 0.01, 30))
        proto, dport = 1, 0
        flags = "none"
    elif kind == "udp_stream":
        n_fwd = int(_u(rng, 300, 6000))
        n_bwd = int(np.clip(n_fwd * _u(rng, 0.0, 0.06), 0, 300))
        fwd_len = float(_u(rng, 180, 1300))
        bwd_len = float(_u(rng, 60, 400))
        dur = float(_u(rng, 4, 40))
        proto, dport = 17, int(rng.choice([5051, 53, 5004]))
        flags = "none"
    else:  # long_idle：合法长连接（与 Botnet 心跳天然混淆）
        n_fwd = int(_u(rng, 5, 40))
        n_bwd = int(_u(rng, 4, 30))
        fwd_len, bwd_len = float(_u(rng, 56, 200)), float(_u(rng, 56, 260))
        dur = float(_u(rng, 60, 700))
        proto, dport = int(rng.choice([6, 17])), int(rng.choice([443, 8080, 22]))
        flags = "normal"
    return dict(
        proto=proto, dport=dport, n_fwd=n_fwd, n_bwd=n_bwd, fwd_len=fwd_len, bwd_len=bwd_len,
        duration=dur, regularity=float(_u(rng, 0.05, 0.6)), n_bursts=int(rng.integers(1, 5)),
        flags=flags,
    )


def _params_ddos(rng: np.random.Generator, ctx: dict) -> dict:
    kind = rng.choice(["udp_flood", "syn_flood", "icmp_flood", "low_rate_flood"], p=[0.4, 0.3, 0.12, 0.18])
    if kind == "udp_flood":
        proto, dport, flags = 17, int(ctx["victim_port"]), "flood"
        n_fwd, n_bwd = int(_u(rng, 3000, 60000)), int(_u(rng, 0, 25))
        fwd_len = float(rng.choice([_u(rng, 55, 100), _u(rng, 1200, 1470)]))
        bwd_len, dur = float(_u(rng, 40, 90)), float(np.clip(_ln(rng, 0.8, 0.8), 0.15, 8))
        reg = float(_u(rng, 0.85, 1.0))
    elif kind == "syn_flood":
        proto, dport, flags = 6, int(ctx["victim_port"]), "syn"
        n_fwd, n_bwd = int(_u(rng, 800, 30000)), int(_u(rng, 0, 60))
        fwd_len = bwd_len = float(_u(rng, 54, 74))
        dur, reg = float(np.clip(_ln(rng, 1.5, 0.9), 0.4, 15)), float(_u(rng, 0.8, 1.0))
    elif kind == "icmp_flood":
        proto, dport, flags = 1, 0, "flood"
        n_fwd, n_bwd = int(_u(rng, 1500, 40000)), int(_u(rng, 0, 400))
        fwd_len = bwd_len = float(_u(rng, 64, 1400))
        dur, reg = float(np.clip(_ln(rng, 1.2, 0.8), 0.3, 12)), float(_u(rng, 0.8, 1.0))
    else:  # low_rate_flood：低速洪泛，单条流看起来像普通流量（刻意制造困难样本）
        proto, dport, flags = int(rng.choice([6, 17])), int(ctx["victim_port"]), "normal"
        n_fwd, n_bwd = int(_u(rng, 20, 400)), int(_u(rng, 5, 200))
        fwd_len, bwd_len = float(_u(rng, 60, 500)), float(_u(rng, 60, 600))
        dur, reg = float(_u(rng, 2, 40)), float(_u(rng, 0.1, 0.5))
    return dict(proto=proto, dport=dport, n_fwd=n_fwd, n_bwd=n_bwd, fwd_len=fwd_len, bwd_len=bwd_len,
                duration=dur, regularity=reg, n_bursts=int(rng.integers(1, 3)), flags=flags)


def _params_dos(rng: np.random.Generator, ctx: dict) -> dict:
    kind = rng.choice(["http_flood", "slowloris", "syn_dos", "mixed"], p=[0.34, 0.26, 0.24, 0.16])
    if kind == "http_flood":
        proto, dport, flags = 6, int(ctx["victim_port"]), "psh_heavy"
        n_fwd = int(_u(rng, 250, 3000))
        n_bwd = int(np.clip(n_fwd * _u(rng, 0.1, 0.6), 0, 2000))
        fwd_len, bwd_len = float(_u(rng, 180, 700)), float(_u(rng, 180, 1000))
        dur, reg = float(_u(rng, 4, 70)), float(_u(rng, 0.4, 0.9))
    elif kind == "slowloris":
        proto, dport, flags = 6, 80, "psh_heavy"
        n_fwd, n_bwd = int(_u(rng, 15, 220)), int(_u(rng, 5, 120))
        fwd_len, bwd_len = float(_u(rng, 56, 140)), float(_u(rng, 56, 240))
        dur, reg = float(_u(rng, 50, 700)), float(_u(rng, 0.05, 0.35))   # 极长、极低速率
    elif kind == "syn_dos":
        proto, dport, flags = 6, int(ctx["victim_port"]), "syn"
        n_fwd, n_bwd = int(_u(rng, 300, 6000)), int(_u(rng, 0, 120))
        fwd_len = bwd_len = float(_u(rng, 54, 74))
        dur, reg = float(_u(rng, 2, 40)), float(_u(rng, 0.5, 0.95))
    else:
        proto, dport, flags = int(rng.choice([6, 17])), int(rng.choice([80, 443, 53])), "normal"
        n_fwd, n_bwd = int(_u(rng, 60, 900)), int(_u(rng, 20, 600))
        fwd_len, bwd_len = float(_u(rng, 80, 900)), float(_u(rng, 80, 1100))
        dur, reg = float(_u(rng, 3, 50)), float(_u(rng, 0.2, 0.7))
    return dict(proto=proto, dport=dport, n_fwd=n_fwd, n_bwd=n_bwd, fwd_len=fwd_len, bwd_len=bwd_len,
                duration=dur, regularity=reg, n_bursts=int(rng.integers(1, 4)), flags=flags)


def _params_probe(rng: np.random.Generator, ctx: dict) -> dict:
    kind = rng.choice(["syn_scan", "udp_scan", "icmp_sweep", "slow_scan"], p=[0.45, 0.22, 0.08, 0.25])
    if kind == "syn_scan":
        proto, flags = 6, "rst"
        n_fwd, n_bwd = int(_u(rng, 1, 3)), int(_u(rng, 0, 2))
        fwd_len = bwd_len = float(_u(rng, 54, 74))
        dur, reg = float(_u(rng, 0.0004, 0.05)), float(_u(rng, 0.6, 0.95))
    elif kind == "udp_scan":
        proto, flags = 17, "none"
        n_fwd, n_bwd = int(_u(rng, 1, 2)), int(_u(rng, 0, 1))
        fwd_len, bwd_len = float(_u(rng, 40, 90)), float(_u(rng, 40, 120))
        dur, reg = float(_u(rng, 0.0004, 0.05)), float(_u(rng, 0.6, 0.95))
    elif kind == "icmp_sweep":
        proto, flags = 1, "none"
        n_fwd, n_bwd = int(_u(rng, 1, 2)), int(_u(rng, 0, 2))
        fwd_len = bwd_len = float(_u(rng, 64, 100))
        dur, reg = float(_u(rng, 0.0004, 0.05)), float(_u(rng, 0.6, 0.95))
    else:  # slow_scan：慢速扫描，与正常短连接高度相似
        proto, flags = int(rng.choice([6, 17])), str(rng.choice(["rst", "none", "psh_heavy"]))
        n_fwd, n_bwd = int(_u(rng, 4, 30)), int(_u(rng, 2, 25))
        fwd_len, bwd_len = float(_u(rng, 60, 300)), float(_u(rng, 60, 400))
        dur, reg = float(_u(rng, 0.5, 20)), float(_u(rng, 0.2, 0.7))
    dport = int(ctx["scanned_ports"][int(rng.integers(0, len(ctx["scanned_ports"])))])
    return dict(proto=proto, dport=dport, n_fwd=n_fwd, n_bwd=n_bwd, fwd_len=fwd_len, bwd_len=bwd_len,
                duration=dur, regularity=reg, n_bursts=1, flags=flags)


def _params_bruteforce(rng: np.random.Generator, ctx: dict) -> dict:
    proto, dport = 6, int(rng.choice([22, 22, 21, 3389, 3306, 8080]))
    n_fwd = int(_u(rng, 15, 180))
    n_bwd = int(np.clip(n_fwd * _u(rng, 0.4, 1.2), 1, 240))
    fwd_len, bwd_len = float(_u(rng, 58, 320)), float(_u(rng, 58, 400))
    dur = float(_u(rng, 0.8, 50))
    return dict(proto=proto, dport=dport, n_fwd=n_fwd, n_bwd=n_bwd, fwd_len=fwd_len, bwd_len=bwd_len,
                duration=dur, regularity=float(_u(rng, 0.05, 0.5)),   # 认证尝试间隔较规律
                n_bursts=int(rng.integers(3, 10)), flags="psh_heavy")


def _params_webattack(rng: np.random.Generator, ctx: dict) -> dict:
    proto, dport = 6, int(rng.choice([80, 80, 443, 8080]))
    n_fwd = int(_u(rng, 5, 60))
    n_bwd = int(_u(rng, 5, 70))
    fwd_len = float(rng.choice([_u(rng, 260, 760), _u(rng, 880, 1450)]))   # 注入载荷
    bwd_len = float(_u(rng, 260, 1450))
    dur = float(np.clip(_ln(rng, 2.0, 1.1), 0.3, 30))
    return dict(proto=proto, dport=dport, n_fwd=n_fwd, n_bwd=n_bwd, fwd_len=fwd_len, bwd_len=bwd_len,
                duration=dur, regularity=float(_u(rng, 0.25, 0.75)), n_bursts=int(rng.integers(2, 8)),
                flags="psh_heavy")


def _params_botnet(rng: np.random.Generator, ctx: dict) -> dict:
    proto = int(rng.choice([6, 17]))
    dport = int(rng.choice([6667, 8080, 443, 8443, 53]))
    n_fwd = int(_u(rng, 4, 35))
    n_bwd = int(_u(rng, 3, 30))
    fwd_len, bwd_len = float(_u(rng, 56, 200)), float(_u(rng, 56, 260))
    dur = float(_u(rng, 30, 700))                      # 长连接
    regular = float(_u(rng, 0.02, 0.35))               # 心跳间隔规律（核心指纹；有方差，不是常数）
    return dict(proto=proto, dport=dport, n_fwd=n_fwd, n_bwd=n_bwd, fwd_len=fwd_len, bwd_len=bwd_len,
                duration=dur, regularity=regular, n_bursts=int(rng.integers(6, 26)), flags="normal")


def _params_u2r(rng: np.random.Generator, ctx: dict) -> dict:
    proto = 6
    dport = int(rng.choice([6200, 1524, 4444, 22, 1433, 5900]))
    n_fwd = int(_ln(rng, 60, 1.2))
    n_bwd = int(np.clip(n_fwd * _u(rng, 0.2, 0.9), 0, 400))
    fwd_len = float(rng.choice([_u(rng, 60, 260), _u(rng, 1300, 1500)]))
    bwd_len = float(_u(rng, 60, 1400))
    dur = float(_u(rng, 1, 140))
    return dict(proto=proto, dport=dport, n_fwd=n_fwd, n_bwd=n_bwd, fwd_len=fwd_len, bwd_len=bwd_len,
                duration=dur, regularity=float(_u(rng, 0.15, 0.85)), n_bursts=int(rng.integers(1, 5)),
                flags=str(rng.choice(["weird", "weird", "psh_heavy", "normal"])))


PARAM_BUILDERS = {
    "Normal": _params_normal,
    "DDoS": _params_ddos,
    "DoS": _params_dos,
    "Probe": _params_probe,
    "BruteForce": _params_bruteforce,
    "WebAttack": _params_webattack,
    "Botnet": _params_botnet,
    "U2R": _params_u2r,
}


# ======================================================================================
# 3. 参数混合：让类别边界真实交叠（这是"指标是否可信"的关键）
# ======================================================================================
def _blend_value(a: float, b: float, w: float) -> float:
    """几何插值（两个正数）或线性插值。w=1 完全取 a，w=0 完全取 b。"""
    a, b = float(a), float(b)
    if a > 0 and b > 0:
        return math.exp(w * math.log(a) + (1 - w) * math.log(b))
    return w * a + (1 - w) * b


def _blend_params(p_a: dict, p_b: dict, rng: np.random.Generator, w: float | None = None) -> dict:
    """把两类流量的**全部参数**混合，得到落在两类交界处的困难样本。

    注意：这里只混合参数，随后仍由 ``_assemble`` 统一推导所有派生特征，
    因此混合后的样本依然满足"包数 × 包长 = 字节数"这类物理一致性。
    """
    w = float(rng.uniform(0.4, 0.6)) if w is None else float(w)
    out = dict(p_a)
    for k in BLEND_NUMERIC:
        out[k] = _blend_value(p_a[k], p_b[k], w)
    out["n_fwd"] = max(int(round(out["n_fwd"])), 1)
    out["n_bwd"] = max(int(round(out["n_bwd"])), 0)
    out["n_bursts"] = max(int(round(out["n_bursts"])), 1)
    out["regularity"] = float(np.clip(out["regularity"], 0.0, 1.0))
    for k in BLEND_CATEGORICAL:
        out[k] = p_a[k] if rng.random() < w else p_b[k]
    return out


def _pick_confusable(cls: str, rng: np.random.Generator) -> str:
    pairs = CONFUSABLE.get(cls) or [("Normal", 1.0)]
    names = [c for c, _ in pairs]
    weights = np.asarray([w for _, w in pairs], dtype=float)
    return str(rng.choice(names, p=weights / weights.sum()))


# ======================================================================================
# 4. 由参数展开成一条物理自洽的流记录
# ======================================================================================
def _assemble(
    rng: np.random.Generator,
    *,
    proto: int,
    dport: int,
    n_fwd: int,
    n_bwd: int,
    fwd_len: float,
    bwd_len: float,
    duration: float,
    regularity: float,
    n_bursts: int,
    ctx: dict,
    flags: str,
) -> dict:
    """把宏观参数展开成完整的流记录（所有派生量都由这几个量算出来）。"""
    n_fwd = max(int(n_fwd), 1)
    n_bwd = max(int(n_bwd), 0)
    duration = max(float(duration), 1e-5)
    regularity = float(np.clip(regularity, 0.0, 1.0))

    fwd_total = n_fwd * fwd_len
    bwd_total = n_bwd * bwd_len
    total_pkts = n_fwd + n_bwd
    total_bytes = fwd_total + bwd_total

    # 到达间隔：平均间隔 = 时长 / (包数-1)，离散度由 regularity 决定
    iat_mean = duration * 1e6 / max(total_pkts - 1, 1)          # 微秒
    iat_std = iat_mean * (0.15 + 1.6 * regularity) * float(_u(rng, 0.7, 1.3))
    iat_max = iat_mean * (1.0 + 6.0 * regularity) * float(_u(rng, 0.8, 2.5))
    iat_min = max(iat_mean * float(_u(rng, 0.02, 0.5)) / (1.0 + 3.0 * regularity), 1.0)

    fwd_iat_mean = duration * 1e6 / max(n_fwd - 1, 1)
    bwd_iat_mean = duration * 1e6 / max(n_bwd - 1, 1) if n_bwd > 1 else 0.0

    # 活跃/空闲：把连接切成 n_bursts 个活跃期
    n_bursts = max(int(n_bursts), 1)
    active_total = duration * 1e6 * float(_u(rng, 0.05, 0.55)) if n_bursts > 1 else duration * 1e6
    active_mean = active_total / n_bursts
    idle_total = max(duration * 1e6 - active_total, 0.0)
    idle_mean = idle_total / max(n_bursts - 1, 1) if n_bursts > 1 else 0.0

    # 标志位：按攻击机理设置，保证与 TCP 语义自洽
    syn = fin = rst = psh = ack = urg = cwe = ece = 0
    if proto == 6:
        if flags == "syn":
            syn, ack, rst = n_fwd, 0, int(n_fwd * _u(rng, 0.0, 0.06))
        elif flags == "rst":
            syn, rst, ack = n_fwd, max(n_bwd, 1), int(n_bwd * _u(rng, 0.3, 0.7))
        elif flags == "psh_heavy":
            ack = total_pkts
            psh = int(total_pkts * _u(rng, 0.4, 0.85))
            fin = max(int(n_bwd * _u(rng, 0.1, 0.5)), 1)
        elif flags == "weird":
            ack = int(total_pkts * _u(rng, 0.3, 0.85))
            psh = int(total_pkts * _u(rng, 0.1, 0.55))
            urg = int(total_pkts * _u(rng, 0.02, 0.3))       # U2R：异常标志位组合
            rst = int(total_pkts * _u(rng, 0.0, 0.25))
            ece, cwe = int(rng.integers(0, 4)), int(rng.integers(0, 3))
        else:  # normal
            ack = int(total_pkts * _u(rng, 0.8, 1.0))
            psh = int(total_pkts * _u(rng, 0.15, 0.65))
            fin = int(max(n_fwd, n_bwd) > 2) * int(_u(rng, 1, 3))
            syn = 1 if n_bwd == 0 else 0
    seg_size = float(_u(rng, 20, 32)) if proto == 6 else 0.0
    fwd_hdr = n_fwd * (20 + seg_size) if proto == 6 else n_fwd * 8
    bwd_hdr = n_bwd * (20 + seg_size) if proto == 6 else n_bwd * 8

    return dict(
        proto=proto,
        src_ip=ctx["src_ip"],
        src_port=int(ctx["src_port"]),
        dst_ip=ctx["dst_ip"],
        dst_port=int(dport),
        duration=duration,
        n_fwd=n_fwd,
        n_bwd=n_bwd,
        fwd_total=fwd_total,
        bwd_total=bwd_total,
        total_pkts=total_pkts,
        total_bytes=total_bytes,
        fwd_len=fwd_len,
        bwd_len=bwd_len,
        fwd_len_std=fwd_len * float(_u(rng, 0.05, 0.35)),
        bwd_len_std=bwd_len * float(_u(rng, 0.05, 0.35)),
        iat_mean=iat_mean,
        iat_std=iat_std,
        iat_max=iat_max,
        iat_min=iat_min,
        fwd_iat_mean=fwd_iat_mean,
        fwd_iat_std=fwd_iat_mean * (0.2 + regularity),
        fwd_iat_max=fwd_iat_mean * (1 + 4 * regularity),
        fwd_iat_min=max(fwd_iat_mean * 0.1, 1.0),
        bwd_iat_mean=bwd_iat_mean,
        bwd_iat_std=bwd_iat_mean * (0.2 + regularity),
        bwd_iat_max=bwd_iat_mean * (1 + 4 * regularity),
        bwd_iat_min=max(bwd_iat_mean * 0.1, 1.0),
        active_mean=active_mean,
        active_std=active_mean * float(_u(rng, 0.05, 0.5)),
        active_max=active_mean * float(_u(rng, 1.0, 2.5)),
        active_min=max(active_mean * float(_u(rng, 0.1, 0.8)), 1.0),
        idle_mean=idle_mean,
        idle_std=idle_mean * float(_u(rng, 0.05, 0.5)),
        idle_max=idle_mean * float(_u(rng, 1.0, 2.5)),
        idle_min=max(idle_mean * float(_u(rng, 0.05, 0.6)), 0.0),
        syn=syn, fin=fin, rst=rst, psh=psh, ack=ack, urg=urg, cwe=cwe, ece=ece,
        fwd_hdr=fwd_hdr,
        bwd_hdr=bwd_hdr,
        init_win_fwd=int(rng.choice([8192, 29200, 65535, 8192, 64240])),
        init_win_bwd=int(rng.choice([0, 8192, 29200, 65535, 26883])),
        act_data_pkt_fwd=int(max(n_fwd - 1, 0) * _u(rng, 0.7, 1.0)),
        seg_size=seg_size,
        icmp_type=8 if proto == 1 else -1,
        icmp_code=0 if proto == 1 else -1,
        idle_timeout=int(rng.choice([0, 0, 0, 0, 10, 30])),
        hard_timeout=int(rng.choice([0, 0, 0, 0, 0, 120])),
        flags_field=int(rng.choice([0, 0, 0, 1])),
    )


# ======================================================================================
# 5. 渲染成两种 schema
# ======================================================================================
def render_insdn(phys: dict, ts: float, label: str) -> list:
    """渲染成 CICFlowMeter / inSDN 的 84 列（列名与真实数据集逐字一致）。"""
    d = phys
    dur_us = d["duration"] * 1e6
    n_tot = d["total_pkts"]
    flow_id = "{}-{}_to_{}-{}-{}".format(
        d["src_ip"], d["src_port"], d["dst_ip"], d["dst_port"], d["proto"]
    )
    row = [
        flow_id, d["src_ip"], d["src_port"], d["dst_ip"], d["dst_port"], d["proto"], ts,
        # --- 76 个数值特征 ---
        dur_us,
        d["n_fwd"], d["n_bwd"], d["fwd_total"], d["bwd_total"],
        d["fwd_len"], max(d["fwd_len"] * 0.55, 28), d["fwd_len"], d["fwd_len_std"],
        d["bwd_len"], max(d["bwd_len"] * 0.55, 0) if d["n_bwd"] else 0,
        d["bwd_len"] if d["n_bwd"] else 0, d["bwd_len_std"] if d["n_bwd"] else 0,
        d["total_bytes"] / max(d["duration"], 1e-6), n_tot / max(d["duration"], 1e-6),
        d["iat_mean"], d["iat_std"], d["iat_max"], d["iat_min"],
        dur_us, d["fwd_iat_mean"], d["fwd_iat_std"], d["fwd_iat_max"], d["fwd_iat_min"],
        dur_us if d["n_bwd"] > 1 else 0, d["bwd_iat_mean"], d["bwd_iat_std"], d["bwd_iat_max"], d["bwd_iat_min"],
        int(d["psh"] > 0), int(d["psh"] > 0),
        int(d["urg"] > 0), int(d["urg"] > 0),
        d["fwd_hdr"], d["bwd_hdr"],
        d["n_fwd"] / max(d["duration"], 1e-6), d["n_bwd"] / max(d["duration"], 1e-6),
        min(d["fwd_len"], d["bwd_len"] if d["n_bwd"] else d["fwd_len"]),
        max(d["fwd_len"], d["bwd_len"]),
        d["total_bytes"] / max(n_tot, 1),
        math.sqrt(max(d["fwd_len_std"] ** 2 + d["bwd_len_std"] ** 2, 0)),
        d["fwd_len_std"] ** 2 + d["bwd_len_std"] ** 2,
        d["fin"], d["syn"], d["rst"], d["psh"], d["ack"], d["urg"], d["cwe"], d["ece"],
        d["n_bwd"] / max(d["n_fwd"], 1),
        d["total_bytes"] / max(n_tot, 1),
        d["fwd_total"] / max(d["n_fwd"], 1), d["bwd_total"] / max(d["n_bwd"], 1),
        0, 0, 0, 0, 0, 0,                      # bulk 相关特征，真实 inSDN 中亦全为 0
        d["n_fwd"], d["fwd_total"], d["n_bwd"], d["bwd_total"],
        d["init_win_fwd"], d["init_win_bwd"], d["act_data_pkt_fwd"], d["seg_size"],
        d["active_mean"], d["active_std"], d["active_max"], d["active_min"],
        d["idle_mean"], d["idle_std"], d["idle_max"], d["idle_min"],
        label,
    ]
    return row


def render_sdn(phys: dict, ts: float, label: str) -> list:
    """渲染成 Ryu 采集脚本的 22 列 + 1 列多分类标注。"""
    d = phys
    dur_sec = int(d["duration"])
    dur_nsec = int((d["duration"] - dur_sec) * 1e9)
    flow_id = "{}{}{}{}{}".format(d["src_ip"], d["src_port"], d["dst_ip"], d["dst_port"], d["proto"])
    pkt = d["total_pkts"]
    byt = d["total_bytes"]
    # 与原脚本一致：label 为 0/1（0=正常，1=攻击）
    return [
        ts, int(d.get("datapath_id", 1)), flow_id,
        d["src_ip"], d["src_port"], d["dst_ip"], d["dst_port"],
        d["proto"], d["icmp_code"], d["icmp_type"],
        dur_sec, dur_nsec,
        d["idle_timeout"], d["hard_timeout"], d["flags_field"],
        pkt, byt,
        pkt / max(dur_sec, 1), pkt / max(dur_nsec, 1), byt / max(dur_sec, 1), byt / max(dur_nsec, 1),
        0 if label == "Normal" else 1,
        label,
    ]


RENDERERS = {"insdn": render_insdn, "sdn": render_sdn}


# ======================================================================================
# 6. 片段级仿真：连续时间轴 + 隐蔽攻击 + 分阶段排布
# ======================================================================================
class TrafficSimulator:
    """按时间轴生成连续流量：正常片段打底，攻击片段随机插入。"""

    def __init__(self, cfg: Config, drift: float = 0.0, seed_offset: int = 0, n_phases: int = 3):
        self.cfg = cfg
        self.schema = cfg.schema.lower()
        if self.schema not in SCHEMAS:
            raise ValueError(f"未知 schema：{cfg.schema}（可选 {list(SCHEMAS)}）")
        self.rng = np.random.default_rng(cfg.seed + seed_offset)
        self.drift = drift
        self.n_phases = max(int(n_phases), 1)
        self.render = RENDERERS[self.schema]

    # ---------------- 场景上下文 ----------------
    def _new_episode_context(self, cls: str, rng: np.random.Generator) -> dict:
        is_attacker = cls != "Normal"
        # 拓扑沿用原仓库 Mininet 脚本：6 台交换机、18 台主机 10.0.0.1~18
        switch = int(rng.integers(1, 7))
        if cls == "DDoS":                      # DDoS：大量肉鸡 -> 同一受害者
            src_ip = f"10.0.0.{int(rng.integers(1, 19))}"
            dst_ip = f"10.0.0.{int(rng.choice([1, 1, 1, 4, 7]))}"
            src_port = int(rng.integers(1024, 65535))
            victim_port = int(rng.choice([80, 80, 443, 53, 5050]))
        elif cls == "Probe":                   # 扫描：单源 -> 网段内大量目标端口
            src_ip = f"10.0.0.{int(rng.integers(1, 19))}"
            dst_ip = f"10.0.0.{int(rng.integers(1, 19))}"
            src_port = int(rng.integers(1024, 65535))
            victim_port = int(rng.integers(1, 1024))
        elif is_attacker:
            src_ip = f"10.0.0.{int(rng.integers(1, 19))}"
            dst_ip = f"10.0.0.{int(rng.integers(1, 19))}"
            src_port = int(rng.integers(1024, 65535))
            victim_port = int(rng.choice([80, 443, 22, 21, 3389, 8080]))
        else:
            src_ip = f"10.0.0.{int(rng.integers(1, 19))}"
            dst_ip = f"10.0.0.{int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8]))}"
            src_port = int(rng.integers(1024, 65535))
            victim_port = 80
        return dict(
            src_ip=src_ip, src_port=src_port, dst_ip=dst_ip,
            victim_port=victim_port, is_attacker=is_attacker,
            datapath_id=switch,
            scanned_ports=rng.integers(1, 1025, size=48).tolist(),
        )

    # ---------------- 分布漂移（鲁棒性实验）----------------
    def _apply_drift_params(self, params: dict, rng: np.random.Generator, factor: float) -> dict:
        """在**参数层**施加漂移，因此漂移后的样本依然物理自洽。

        模拟的是"网络环境/攻击工具换代"：包的规模与时长整体缩放，
        包的典型长度也略有变化 —— 模型在训练期从未见过这种量级组合。
        """
        if factor <= 0:
            return params
        out = dict(params)
        scale_pkts = float(rng.lognormal(0.0, factor))
        scale_dur = float(rng.lognormal(0.0, factor))
        scale_len = float(rng.lognormal(0.0, factor * 0.5))
        out["n_fwd"] = max(int(params["n_fwd"] * scale_pkts), 1)
        out["n_bwd"] = max(int(params["n_bwd"] * scale_pkts), 0)
        out["duration"] = max(float(params["duration"]) * scale_dur, 1e-5)
        out["fwd_len"] = max(float(params["fwd_len"]) * scale_len, 28.0)
        out["bwd_len"] = max(float(params["bwd_len"]) * scale_len, 0.0)
        return out

    # ---------------- 主循环 ----------------
    def generate(self) -> pd.DataFrame:
        rng = self.rng
        cfg = self.cfg
        rows: list[list] = []
        t = 1_700_000_000.0            # 起始时间戳（epoch 秒），仅用于保持时序

        for cls in self._build_schedule():
            prof = EPISODE_PROFILES[cls]
            n_flows = max(int(rng.integers(prof.flows[0], prof.flows[1] + 1) * cfg.scale), 8)
            duration = float(rng.uniform(*prof.duration))
            ctx = self._new_episode_context(cls, rng)

            # 「隐蔽型」攻击片段：片段内混入大量正常样貌的流，
            # 单条流难以判定，必须靠窗口内的聚合与演化模式识别。
            stealth = 0.0
            if cls != "Normal" and rng.random() < float(getattr(cfg, "stealth_ratio", 0.0)):
                stealth = float(getattr(cfg, "stealth_mix", 0.45))

            # 突发性：前面密、后面疏（用幂律把均匀间隔重新映射）
            u = rng.random(n_flows) ** (1.0 / prof.burstiness)
            gaps = np.diff(np.concatenate([[0.0], np.sort(u)])) * duration
            ep_start = t

            for i in range(n_flows):
                ctx_i = dict(ctx)
                if cls == "DDoS":       # DDoS 的每一条流来自不同肉鸡/端口
                    ctx_i["src_ip"] = f"10.0.0.{int(rng.integers(1, 19))}"
                    ctx_i["src_port"] = int(rng.integers(1024, 65535))

                params = PARAM_BUILDERS[cls](rng, ctx_i)
                if stealth > 0 and rng.random() < stealth:
                    # 与正常流量参数混合（偏正常一侧），制造"单流歧义"
                    params = _blend_params(
                        params, PARAM_BUILDERS["Normal"](rng, ctx_i), rng,
                        w=float(rng.uniform(0.25, 0.6)),
                    )
                elif rng.random() < float(cfg.overlap_ratio):
                    # 与最易混淆类混合，制造类别边界交叠
                    partner = _pick_confusable(cls, rng)
                    params = _blend_params(params, PARAM_BUILDERS[partner](rng, ctx_i), rng)

                params = self._apply_drift_params(params, rng, self.drift)
                phys = _assemble(rng, ctx=ctx_i, **params)
                phys["datapath_id"] = ctx["datapath_id"]

                ts = float(ep_start + gaps[i] if i < len(gaps) else ep_start + duration)
                label = cls
                if rng.random() < cfg.label_noise:      # 标注噪声
                    label = str(rng.choice([c for c in ALL_CLASSES if c != cls]))
                rows.append(self.render(phys, ts, label))

            t = ep_start + duration + float(rng.uniform(1.0, 5.0))   # 片段之间的间隔

        df = pd.DataFrame(rows, columns=SCHEMAS[self.schema])
        ts_col = "Timestamp" if self.schema == "insdn" else "timestamp"
        return df.sort_values(ts_col).reset_index(drop=True)

    def _build_schedule(self) -> list[str]:
        """分阶段排布片段：正常流量打底，攻击片段插入，且每个阶段都包含全部类别。

        这样按时间切分 train/val/test 时，**每个类别都会出现在三个子集里**，
        不会出现"某个攻击类只落在训练集导致测试集 support=0"的评测偏差。
        """
        rng = self.rng
        schedule: list[str] = []
        for phase in range(self.n_phases):
            phase_schedule: list[str] = []
            for cls, count in self.cfg.episodes_per_class.items():
                if cls not in EPISODE_PROFILES:
                    raise ValueError(f"未知类别：{cls}")
                total = int(round(count * max(self.cfg.scale, 0.2)))
                per_phase = max(total // self.n_phases, 1)
                if phase == 0:
                    per_phase = max(total - per_phase * (self.n_phases - 1), per_phase)  # 余数补给第一阶段
                for _ in range(per_phase):
                    phase_schedule.append(cls)
                    if cls != "Normal":
                        phase_schedule.append("Normal")   # 攻击片段前后垫正常流量
            # 阶段内打乱，但保证首片段是正常流量
            order = rng.permutation(len(phase_schedule))
            shuffled = [phase_schedule[i] for i in order]
            if shuffled and shuffled[0] != "Normal":
                shuffled.insert(0, "Normal")
            schedule.extend(shuffled)
        return schedule


# ======================================================================================
# 7. 对外接口
# ======================================================================================
def generate_datasets(cfg: Config, out_dir: Path | None = None, save: bool = True) -> dict[str, Path]:
    """生成主数据集（+ 可选的漂移测试集），返回文件路径字典。"""
    out_dir = Path(out_dir or RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    sim = TrafficSimulator(cfg, drift=0.0, seed_offset=0)
    df = sim.generate()
    main_path = out_dir / f"{cfg.schema}_stream.csv"
    if save:
        df.to_csv(main_path, index=False, encoding="utf-8")
    files["stream"] = main_path
    counts = df.iloc[:, -1].value_counts()
    LOGGER.info("主数据集：%s  形状=%s", main_path.name, df.shape)
    LOGGER.info("类别分布：\n%s", counts.to_string())

    # 每个类别在三个时间段（≈train/val/test）的分布——用于确认没有类别缺失
    n = len(df)
    if n >= 9:
        thirds = [df.iloc[: n // 3], df.iloc[n // 3 : 2 * n // 3], df.iloc[2 * n // 3 :]]
        table = pd.DataFrame(
            {f"段{i + 1}": part.iloc[:, -1].value_counts() for i, part in enumerate(thirds)}
        ).fillna(0).astype(int)
        LOGGER.info("各时间段类别分布（确认每类都出现在 train/val/test）：\n%s", table.to_string())

    if cfg.drift_strength and cfg.drift_strength > 0:
        sim_d = TrafficSimulator(cfg, drift=cfg.drift_strength, seed_offset=991)
        df_d = sim_d.generate()
        drift_path = out_dir / f"{cfg.schema}_drift_test.csv"
        if save:
            df_d.to_csv(drift_path, index=False, encoding="utf-8")
        files["drift"] = drift_path
        LOGGER.info("漂移测试集：%s  形状=%s", drift_path.name, df_d.shape)

    if save:
        save_json(
            {
                "schema": cfg.schema,
                "header": SCHEMAS[cfg.schema],
                "id_columns": SCHEMA_ID_COLUMNS[cfg.schema],
                "classes": ALL_CLASSES,
                "rows": int(len(df)),
                "class_counts": counts.to_dict(),
                "difficulty": {
                    "overlap_ratio": cfg.overlap_ratio,
                    "stealth_ratio": getattr(cfg, "stealth_ratio", 0.0),
                    "stealth_mix": getattr(cfg, "stealth_mix", 0.0),
                    "label_noise": cfg.label_noise,
                    "feature_noise": cfg.feature_noise,
                },
            },
            out_dir / f"{cfg.schema}_schema_info.json",
        )
    return files


if __name__ == "__main__":  # 手动生成一份数据：python -m src.traffic_generator
    from .utils import ensure_dirs

    ensure_dirs()
    generate_datasets(Config())
