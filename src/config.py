"""全局配置：一个 dataclass 管住「数据生成 → 训练 → 评估 → 可视化 → 演示」全流程。

用法：
    cfg = Config()                       # 默认（比赛推荐）配置
    cfg = Config.from_json("cfg.json")   # 从 JSON 读
    cfg = Config(**{**Config().to_dict(), **overrides})   # 代码里改
    run_all.py --set train_epochs=5 --set schema=sdn      # 命令行改

所有字段都会被写进 artifacts/config_used.json，保证实验可追溯。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .utils import load_json, save_json


@dataclass
class Config:
    # ---------------- 通用 ----------------
    seed: int = 42
    run_name: str = "cnn_bilstm"
    verbose: bool = True

    # ---------------- 数据来源 ----------------
    schema: str = "insdn"            # insdn = 84 列 CICFlowMeter/inSDN 同构；sdn = 22 列（与 Ryu 采集脚本完全一致）
    source: str = "synthetic"        # synthetic = 本地仿真生成；real = 读取真实 CSV 目录
    real_data_dir: str = ""          # source=real 时：存放 CSV 的目录（自动合并目录下所有 csv）
    real_label_column: str = ""      # 留空则自动识别 Label/label/class/attack_cat
    raw_dir: str = ""                # 原始 CSV 目录（留空 = data/raw；快跑自检会指向 data/raw_smoke）
    processed_dir: str = ""          # 预处理产物目录（留空 = data/processed）
    figure_dir: str = ""             # 图表输出目录（留空 = results/figures；补充实验可单独指定）
    publish_summary: bool = True     # 是否把本次结果写成 results/汇总报告.md 与 summary.json（补充实验设 false，避免覆盖主实验结果）

    # ---------------- 仿真数据规模 ----------------
    scale: float = 1.0               # 全局样本量缩放系数（1.0 ≈ 9 万条流记录）
    # 片段（episode）数量：每个攻击片段后面会自动跟一段正常流量，
    # 因此实际正常流片段 = Normal + 所有攻击片段数之和。
    # 攻击类给足片段数，保证每个类别在 train/val/test 三个时间段都有一两百个窗口可评。
    episodes_per_class: dict = field(
        default_factory=lambda: {
            "Normal": 60,
            "DDoS": 16,
            "DoS": 18,
            "Probe": 24,
            "BruteForce": 22,
            "WebAttack": 22,
            "Botnet": 20,
            "U2R": 22,
        }
    )
    label_noise: float = 0.01        # 标签噪声比例（模拟人工标注错误，避免 100% 虚高指标）
    feature_noise: float = 0.05      # 特征抖动强度（让类别边界更真实、更难点）
    dedup_significant_digits: int = 0  # >0 时按"有效数字"做近似重复去重（公开数据集重复率高时用，0=关）
    overlap_ratio: float = 0.35      # 每类中「与最易混淆类混合」的困难样本比例
    stealth_ratio: float = 0.45      # 攻击片段中「隐蔽型片段」的比例
    stealth_mix: float = 0.45        # 隐蔽型片段内，混入"正常样貌"的流所占比例
    drift_strength: float = 0.0      # >0 时给测试集注入分布漂移（鲁棒性实验用）

    # ---------------- 序列建模 ----------------
    input_mode: str = "sequence"     # sequence = (T, F) 滑窗时序建模（推荐）；sample = 单条流分类（对应原始 demo）
    window_size: int = 10            # 时间窗内的流记录条数 T
    window_stride: int = 5           # 滑窗步长
    online_stride: int = 3           # 在线检测（告警级评估 / 演示）的滑窗步长

    # ---------------- 切分 ----------------
    train_ratio: float = 0.60
    val_ratio: float = 0.15          # 剩下为 test
    split_mode: str = "time"         # time = 按时间顺序切分（连续采集流，推荐）；random = 分层随机切分
    #   ★ 为什么需要 random：部分公开数据集（如本工程使用的真实 inSDN 衍生集）是以
    #     "正常流量块 + 攻击流量块"两段拼接发布的，只有一次标签切换。此时若按时间切分，
    #     val/test 会完全落在攻击块里（类别不相交），评估将失去意义。
    #     遇到这种数据就用 --set split_mode=random，并在报告中如实说明。
    purge_windows: int = 2           # 片段间丢弃的窗口数，切断滑窗重叠带来的泄漏（仅 time 模式生效）
    split_group_column: str = ""     # 按"会话/主机"分组切分（如 Flow ID / Src IP），防止同一会话的近似重复流
    #   ★ 随机按行切分时，同一会话的高度相似流记录会同时落进训练集与测试集，指标会虚高。
    #     流量数据集的标准做法是**按组切分**（group split）：同组的行只能进同一个子集。
    split_group_eval: bool = False   # 是否在随机切分之外，额外做一次"严格按组切分"的对照评估
    keep_port_features: bool = False  # 是否把源/目的端口也当特征（默认丢弃，避免模型记住具体服务端口"作弊"）

    # ---------------- 模型 ----------------
    model_variant: str = "cnn_bilstm"   # cnn_bilstm | cnn | bilstm | mlp  （后三者用于消融实验）
    conv_filters: tuple = (64, 128)
    conv_kernel: int = 3
    lstm_units: int = 64
    dense_units: tuple = (128, 64)
    dropout: float = 0.3
    l2_reg: float = 1e-4
    batch_norm: bool = True
    use_attention: bool = True      # 时间维注意力池化（既提精度，又给可解释热力图）
    keep_sequence_in_hybrid: bool = True  # 混合模型里不用池化、改用膨胀卷积保住时间步（关掉可复现旧的错误做法）

    # ---------------- 训练 ----------------
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    lr_patience: int = 4
    lr_factor: float = 0.5
    early_stop_patience: int = 8
    use_class_weight: bool = True    # 类别不平衡：攻击类样本远少于正常流量
    use_balanced_batch: bool = False # 训练集过采样到类别均衡（可选）
    binary_loss_weight: float = 0.3  # 多分类 + 二分类联合损失中，二分类头权重

    # ---------------- 评估 ----------------
    report_threshold: float = 0.5    # 二分类判定阈值
    target_fpr: tuple = (0.01, 0.001)  # 报告 TPR@FPR=1%/0.1%
    online_min_segment_rows: int = 10  # 在线评估中，"算得上一个攻击片段"的最小连续长度
    figures: bool = True

    # ---------------- 传统机器学习基线 ----------------
    baseline_n_jobs: int = -1        # -1 = 用满所有核心；某些受限环境里并行会失败，会自动回退单进程

    # ---------------- 演示服务 ----------------
    demo_host: str = "127.0.0.1"
    demo_port: int = 5000
    demo_speed: float = 20.0         # 演示回放倍速（20 = 20 倍实时）

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["conv_filters"] = list(self.conv_filters)
        d["dense_units"] = list(self.dense_units)
        d["target_fpr"] = list(self.target_fpr)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        clean: dict[str, Any] = {}
        for k, v in (data or {}).items():
            if k not in known:
                continue
            if k in {"conv_filters", "dense_units", "target_fpr"} and v is not None:
                v = tuple(v)
            clean[k] = v
        return cls(**clean)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        return cls.from_dict(load_json(path))

    def save(self, path: str | Path) -> Path:
        return save_json(self.to_dict(), path)

    def apply_overrides(self, overrides: dict[str, str]) -> "Config":
        """`--set key=value` 的解析：按已有字段的类型做转换。"""
        types = {f.name: f.type for f in fields(self)}
        for key, raw in overrides.items():
            if key not in types:
                raise KeyError(f"未知配置项：{key}")
            cur = getattr(self, key)
            setattr(self, key, _coerce(raw, cur))
        return self

    # 便捷属性
    @property
    def class_names(self) -> list[str]:
        return list(self.episodes_per_class.keys())


def _coerce(raw: str, cur: Any) -> Any:
    """把命令行字符串转成与当前值同类型。"""
    if isinstance(cur, bool):
        return str(raw).lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(float(raw))
    if isinstance(cur, float):
        return float(raw)
    if isinstance(cur, (tuple, list)):
        parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(int(p))
            except ValueError:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(p)
        return tuple(vals) if isinstance(cur, tuple) else vals
    if isinstance(cur, dict):
        import json

        return json.loads(raw)
    return raw
