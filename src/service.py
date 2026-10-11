"""在线检测服务引擎：把"训练好的模型"变成一个**可被控制器调用的在线服务**。

这是本作品从"课题"走向"可上线系统"的关键一环。它负责：

1. **滑动缓冲**：控制器每次推送一批流记录，服务把它们依次放入长度 T 的窗口；
2. **在线推理**：窗口满即推理一次，输出攻击类型与异常概率；
3. **告警判定**：阈值 + **连续 N 个窗口确认**（与离线评估同一口径，保证"演示=报告"）；
4. **处置建议**：对确认的告警给出可执行的处置动作（封禁源 IP / 限速 / 仅观察），
   并保留白名单与自保护逻辑，避免把控制器自身或管理网断掉；
5. **审计台账**：每条告警落一条 JSONL 审计记录（时间、证据、动作、处置状态），
   这是"可用系统"与"课题 demo"最直观的区别——出事能追溯；
6. **运行指标**：累计行数/窗口数/告警数/分位数时延，供 /metrics 暴露给监控系统。

设计约束
--------
* **同一份判定逻辑**：阈值与确认次数来自配置（与 `src/evaluate.py` 完全一致），
  因此"服务里的误报率"就是报告里的误报率，不存在两套口径；
* **无 TF 也能启动**：模型缺失时服务返回明确的 503 与修复指引，而不是崩溃；
* **不阻塞**：单次推理在 CPU 上为毫秒级（见 `tools/benchmark_latency.py`）。
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import Config
from .utils import (ARTIFACT_DIR, PROCESSED_DIR, RESULT_DIR, append_jsonl, get_logger,
                    load_json, resolve_dir)

LOGGER = get_logger("cida.service")

ACTIONS = ("observe", "throttle", "block")

# 被标记为「未知形态」时对外显示的类别名（不参与训练，只用于告警与审计的可读性）
UNKNOWN_CLASS_NAME = "Unknown（未知形态）"


@dataclass
class DetectorConfig:
    """在线服务的运行参数（可由 /api/policy 热更新）。"""

    threshold: float = 0.5
    confirm_windows: int = 3
    action: str = "observe"          # observe=只告警；throttle=建议限速；block=建议封禁
    whitelist: tuple = ()            # 永不处置的源 IP（例如控制器自身、管理网段）
    dry_run: bool = True             # True=只输出建议动作，不真正下发
    # 「未知形态」标记阈值：多分类概率**熵**超过它就认为"模型没见过这种形态"，判定类别改为 Unknown。
    # 0（默认）= 关闭。阈值由 `tools/openset_calibrate.py` 在**正常窗口**上按误报预算标定，
    # 语义边界见该工具与 results/openset_report.md（它标记的是"新颖性"，不是已验证的未知攻击检测器）。
    unknown_threshold: float = 0.0
    # ---------------- 未知形态与漏报的兜底（观察池 / 回看缓冲 / 待研判队列） ----------------
    # 未知形态**永不直接封禁**：改用保守动作（throttle=限速），把最终判断权交给人。
    # 理由见 competition/06_检测与处置体系.md §2.2：未知 ≠ 恶意，误封一条正常业务链路的代价，
    # 可能高于放行一次可疑流量；限速既压住可能的滥用，又不切断业务，给研判留出时间窗。
    unknown_action: str = "throttle"
    # 观察池下界 = observe_margin × threshold。落在 [下界, threshold) 且**未被确认**的窗口进池留痕，
    # 事后发现漏报时能回答"当时它打了几分、离阈值差多远"——过去这些窗口是不留任何痕迹的。
    observe_margin: float = 0.5
    observe_pool_size: int = 500      # 观察池容量（有界，防止长期运行吃内存）
    review_queue_size: int = 200      # 待研判队列容量
    recent_windows: int = 1000        # 回看环形缓冲容量（回答"这段为什么没报警"）
    # ---------------- 长期运行的资源边界与运维面 ----------------
    # 告警在内存里保留多少条（`self.alerts` 过去是**无上界 list**，长跑会一直涨）。
    # 总量另有 alerts_total 计数器，因此"保留条数"不影响统计口径。
    alerts_retained: int = 5000
    audit_max_bytes: int = 20_000_000  # 审计台账单文件上限（约 20MB），超过即轮转
    audit_backups: int = 3             # 审计台账保留的历史文件数
    observe_evidence_rows: int = 16    # 观察池条目随附的窗口内流记录条数（0=不留）
    # 反复告警的源自动**升级**动作：同源在 escalate_window_sec 内告警达到 escalate_after 次，
    # 建议动作上抬一级（observe→throttle→block）。未知形态受硬护栏约束，最高只到 throttle。
    escalate_after: int = 3
    escalate_window_sec: int = 300
    # 是否把进入研判队列的**窗口特征向量**落盘（`results/review_evidence.jsonl`）。
    # 为什么必须存：`review_feedback.jsonl` 只有结论与分数，**没有特征**——那样人工标了结论
    # 也无法用于重训，数据在源头就丢了，"回流"就成了名不副实的空闭环。
    # 证据与结论分开存（按 review_id 关联）：结论可以改判而不必重复存特征。
    review_evidence: bool = True

    @classmethod
    def from_env(cls, cfg: Config | None = None, **overrides) -> "DetectorConfig":
        """从环境变量读初始策略，便于 systemd / 容器 / 服务化部署。

        支持（见 `deploy/cida-detect.env`）：
        `CIDA_POLICY_ACTION` / `CIDA_DRY_RUN` / `CIDA_THRESHOLD` /
        `CIDA_CONFIRM_WINDOWS` / `CIDA_WHITELIST`（空格分隔的源 IP 列表）。

        传入的关键字参数优先级最高（用于代码内显式指定）。
        """
        cfg = cfg or Config()
        action = str(os.environ.get("CIDA_POLICY_ACTION", "observe")).strip().lower()
        if action not in ACTIONS:
            LOGGER.warning("CIDA_POLICY_ACTION=%r 不是合法动作 %s，回退为 observe", action, ACTIONS)
            action = "observe"
        dry_raw = str(os.environ.get("CIDA_DRY_RUN", "1")).strip().lower()
        params: dict[str, Any] = {
            "threshold": float(os.environ.get("CIDA_THRESHOLD", cfg.report_threshold)),
            "confirm_windows": int(os.environ.get("CIDA_CONFIRM_WINDOWS", cfg.online_confirm_windows)),
            "action": action,
            "dry_run": dry_raw not in ("0", "false", "no", "off"),
            "whitelist": tuple(str(os.environ.get("CIDA_WHITELIST", "")).split()),
            "unknown_threshold": float(os.environ.get("CIDA_UNKNOWN_THRESHOLD", 0.0) or 0.0),
            "unknown_action": str(os.environ.get("CIDA_UNKNOWN_ACTION", "throttle")).strip().lower(),
            "observe_margin": float(os.environ.get("CIDA_OBSERVE_MARGIN", 0.5) or 0.0),
            "audit_max_bytes": int(os.environ.get("CIDA_AUDIT_MAX_BYTES", 20_000_000) or 0),
            "audit_backups": int(os.environ.get("CIDA_AUDIT_BACKUPS", 3) or 3),
            "review_evidence": str(os.environ.get("CIDA_REVIEW_EVIDENCE", "1")).strip().lower()
            not in ("0", "false", "no", "off"),
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        conf = cls(**params)
        if conf.unknown_action not in ACTIONS:
            LOGGER.warning("CIDA_UNKNOWN_ACTION=%r 不是合法动作 %s，未知形态回退为 observe",
                           conf.unknown_action, ACTIONS)
            conf.unknown_action = "observe"
        if conf.whitelist:
            LOGGER.info("已加载白名单 %d 个源 IP（这些地址永不被处置）", len(conf.whitelist))
        else:
            LOGGER.warning("白名单为空：上线前请通过 CIDA_WHITELIST 或 /api/policy 至少加入控制器与管理网段")
        return conf


@dataclass
class Alert:
    """一条告警（也是审计记录）。"""

    time: float
    window_index: int
    p_attack: float
    pred_class: str
    true_class: str | None
    action: str
    src_ip: str | None = None
    dst_ip: str | None = None
    dst_port: int | None = None
    rows_covered: tuple = ()
    executed: bool = False
    unknown: bool = False            # 是否为「未知形态」窗口（见 DetectorConfig.unknown_threshold）
    novelty: float = 0.0             # 触发未知标记时的新颖性分数（熵），便于审计与调阈值
    escalated: bool = False          # 本条动作是否由"反复告警"自动升级而来（见 DetectorConfig.escalate_after）

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 3),
            "window_index": self.window_index,
            "p_attack": round(float(self.p_attack), 4),
            "pred_class": self.pred_class,
            "action": self.action,
            "executed": self.executed,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "rows_covered": list(self.rows_covered),
            "ground_truth": self.true_class,
            "unknown": bool(self.unknown),
            "novelty": round(float(self.novelty), 4),
            "escalated": bool(self.escalated),
        }


@dataclass
class Observation:
    """观察池条目：模型**没有确认告警**、但值得留痕的窗口。

    为什么必须有：窗口被判为正常时，系统过去**不留任何痕迹**。一旦事后发现那次真发生了攻击
    （漏报），运维无法回答"当时它打了多少分、离阈值差多远"——漏报因此不可复盘。
    观察池把这段证据留下来，并支持随时提升进待研判队列。
    """

    time: float
    window_index: int
    p_attack: float
    pred_class: str
    novelty: float = 0.0
    unknown: bool = False
    src_ip: str | None = None
    dst_ip: str | None = None
    rows_covered: tuple = ()
    # 窗口内的流记录（只保留定位字段：五元组/协议/标签），让复核能追到具体流而不是只有一个分数。
    # 存定位字段而不是全部 71 维特征，是容量与可用性的折中：500 条观察项 × 16 行仍只有 MB 级。
    evidence: list = field(default_factory=list)
    # 该窗口的模型输入向量（T×F，float32）。不序列化进 to_dict（太大且是二进制语义），
    # 只在需要落盘/重训时使用；由 deque 上界保证内存恒定。
    vector: Any = None
    # near_threshold：差一点到阈值；unknown_observed：形态新颖但未达确认条件
    why: str = "near_threshold"

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 3),
            "window_index": self.window_index,
            "p_attack": round(float(self.p_attack), 4),
            "pred_class": self.pred_class,
            "novelty": round(float(self.novelty), 4),
            "unknown": bool(self.unknown),
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "rows_covered": list(self.rows_covered),
            "evidence": [dict(r) for r in self.evidence],
            "why": self.why,
        }


class DetectionService:
    """在线检测服务：喂流记录 → 得到判定、告警与处置建议。"""

    def __init__(self, cfg: Config | None = None, processed_dir: Path | None = None,
                 artifact_dir: Path | None = None, model_tag: str | None = None):
        self.cfg = cfg or Config()
        self.processed_dir = Path(processed_dir or resolve_dir(self.cfg.processed_dir, PROCESSED_DIR))
        self.artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
        self.model_tag = model_tag
        # 初始策略：默认「只观察 + 演练模式」，可由 CIDA_* 环境变量设定，也可用 /api/policy 热更新
        self.policy = DetectorConfig.from_env(self.cfg)
        self.lock = threading.Lock()
        self.buffer: deque = deque(maxlen=int(self.cfg.window_size))
        # 与 self.buffer 同进同出的「原始行」缓冲：滑窗里每条向量对应的流记录（只留定位字段），
        # 用于给观察池条目附上可追溯的证据。只留 window_size 条，内存恒定。
        self.buffer_rows: deque = deque(maxlen=int(self.cfg.window_size))
        # ⚠️ 这里过去是**无上界 list**：长跑时内存只增不减。改为有界 deque，
        # 总量另用 alerts_total 计数，因此"保留多少条"不影响任何统计口径。
        self.alerts: deque = deque(maxlen=max(1, int(self.policy.alerts_retained)))
        # 同源告警时间戳（用于反复告警自动升级）；有界，防止大量源 IP 撑爆内存
        self._escalation: dict[str, list[float]] = {}
        self.audit_path = RESULT_DIR / "audit_alerts.jsonl"
        # 漏报兜底三件套（都有界，长期运行不会吃内存）：
        #   observations 观察池：差一点到阈值 / 形态新颖但未确认的窗口留痕
        #   review_queue 待研判队列：系统承认"不认识"的告警，等人给结论
        #   recent       回看环形缓冲：**每个**窗口的分数都留一份，回答"这段为什么没报警"
        self.observations: deque = deque(maxlen=max(1, int(self.policy.observe_pool_size)))
        self.review_queue: deque = deque(maxlen=max(1, int(self.policy.review_queue_size)))
        self.recent: deque = deque(maxlen=max(1, int(self.policy.recent_windows)))
        self.review_seq = 0
        # 人工研判结论的回流落点：下一次重训直接读它，而不是靠人回忆（"未知 → 已知"闭环的落地）
        self.feedback_path = RESULT_DIR / "review_feedback.jsonl"
        # 研判证据（窗口特征向量）单独落盘：只有结论没有特征，人工标注就无法用于重训。
        self.review_evidence_path = RESULT_DIR / "review_evidence.jsonl"
        self.stats: dict[str, Any] = {
            "rows": 0, "windows": 0, "alarms": 0, "alarms_confirmed": 0,
            "alarm_run": 0, "latency_ms": deque(maxlen=2000),
            "by_class": {}, "started_at": time.time(),
            "observations": 0, "review_enqueued": 0, "review_labeled": 0,
            "review_confirmed_attack": 0, "alerts_total": 0, "escalations": 0,
            "review_evidence": 0,
        }
        self.model = None
        self.feature_names: list[str] = []
        self.class_names: list[str] = []
        self.scaler = None
        self.ready = False
        self.reason = ""
        self._load()

    # ------------------------------------------------------------------ 加载
    def _load(self) -> None:
        meta_path = self.processed_dir / "meta.json"
        if not meta_path.exists():
            self.reason = "缺少预处理产物（先运行 python run_all.py）"
            return
        try:
            import joblib

            meta = load_json(meta_path)
            self.feature_names = list(meta["feature_names"])
            self.class_names = list(meta["class_names"])
            self.scaler = joblib.load(self.processed_dir / "scaler.joblib")
        except Exception as exc:  # pragma: no cover
            self.reason = f"预处理产物读取失败：{exc}"
            return
        # ---- 选模型：必须与预处理产物的 (T, F) 匹配，否则宁可拒绝服务也不给出错误判定 ----
        want = (int(self.cfg.window_size), len(self.feature_names))
        model_path, tried = self._resolve_model(want)
        if model_path is None:
            self.reason = (f"没有与预处理产物（窗口 {want[0]}、特征 {want[1]}）匹配的模型；"
                           f"已尝试：{tried}。请先运行 python run_all.py，"
                           f"或用 --model-tag 指定正确的模型")
            LOGGER.error("%s", self.reason)
            return
        try:
            from .model import load_model_bundle

            self.model = load_model_bundle(model_path)
        except Exception as exc:  # pragma: no cover
            self.reason = f"模型加载失败（可能未安装 TensorFlow）：{exc}"
            return
        self.model_path = model_path
        self.ready = True
        self._build_infer_fn()
        LOGGER.info("在线检测服务就绪：窗口 T=%d，特征 %d，类别 %d，模型 %s",
                    self.cfg.window_size, len(self.feature_names), len(self.class_names), model_path)

    def _resolve_model(self, want: tuple[int, int]) -> tuple[Path | None, list[str]]:
        """按优先级挑出**输入形状匹配**的模型，返回 (路径, 已尝试过的候选说明)。

        优先级：显式 --model-tag → run_all.py 记录的「主模型」tag → artifacts/main_model.keras
        → 其余 artifacts/*/model.keras（新到旧）。

        为什么要按形状过滤：`artifacts/` 下同时存在多种口径的模型
        （84 列 CICFlowMeter 口径 71 维、控制器原生字段口径 17 维、真实数据口径 65 维）。
        早期实现直接取「修改时间最新的一个」，在训练过其他口径模型后会把 17 维模型
        配上 71 维的预处理产物，运行时报 `expected shape=(None,16,71), found (15,16,17)`。
        这里改为逐个加载并校验 `model.input_shape`，形状不符就跳过——**宁可启动失败也不给错判定**。
        """
        cands: list[Path] = []
        if self.model_tag:
            cands.append(self.artifact_dir / self.model_tag / "model.keras")
        else:
            ptr = self.artifact_dir / "main_model.json"
            cands.append(self.artifact_dir / "main_model.keras")
            if ptr.exists():
                try:
                    tag = load_json(ptr).get("tag")
                    if tag:
                        cands.append(self.artifact_dir / str(tag) / "model.keras")
                except Exception:  # pragma: no cover
                    pass
            cands.extend(sorted(self.artifact_dir.glob("*/model.keras"),
                                key=lambda p: p.stat().st_mtime, reverse=True))
        seen: set[Path] = set()
        tried: list[str] = []
        for cand in cands:
            cand = Path(cand)
            if cand in seen or not cand.exists():
                continue
            seen.add(cand)
            shape = self._model_shape(cand)
            if shape is None:
                tried.append(f"{cand.name}(无法读取形状)")
                continue
            if shape != want:
                tried.append(f"{cand.parent.name}({shape[0]}×{shape[1]})")
                continue
            return cand, tried
        return None, tried or ["（artifacts 下没有任何 model.keras）"]

    @staticmethod
    def _model_shape(path: Path) -> tuple[int, int] | None:
        """读取模型期望的 (T, F)，读不出来返回 None。"""
        try:
            from .model import load_model_bundle

            m = load_model_bundle(path)
            shape = getattr(m, "input_shape", None) or []
            if len(shape) == 3 and shape[1] and shape[2]:
                return int(shape[1]), int(shape[2])
        except Exception:  # pragma: no cover
            return None
        return None

    # ------------------------------------------------------------------ 推理
    def _build_infer_fn(self):
        """构造批量推理函数：一次调用处理一批窗口（而不是每条流调一次）。

        这是服务端最关键的工程优化：
        ① **批内合并**——一批 40 条流记录只推理一次，而不是 40 次；
        ② **直接调用推理图**——绕开 Keras `predict()` 每次约 36 ms 的固定包装开销
           （见 `tools/benchmark_latency.py` 的两条路径对比）；
        ③ **按在线步长判定**——与离线评估同口径（`online_stride`），不重复计算重叠窗口。
        """
        try:
            import tensorflow as tf

            @tf.function(reduce_retracing=True)
            def _f(x):
                return self.model(x, training=False)

            _f(tf.zeros((2, int(self.cfg.window_size), len(self.feature_names)), dtype=tf.float32))  # 追踪

            def _infer(window_raw: np.ndarray):
                Xs = self.scaler.transform(window_raw.reshape(-1, len(self.feature_names)))
                Xs = Xs.reshape(window_raw.shape).astype(np.float32)
                out = _f(Xs)
                if isinstance(out, dict):
                    p_mc = np.asarray(out["mc"])
                    p_bin = np.asarray(out["bin"]).reshape(-1)
                elif isinstance(out, (list, tuple)):
                    p_mc, p_bin = np.asarray(out[0]), np.asarray(out[1]).reshape(-1)
                else:
                    p_mc = np.asarray(out)
                    p_bin = 1.0 - p_mc[:, 0]
                return p_mc, np.clip(p_bin, 1e-7, 1 - 1e-7)

            self._infer = _infer
            self.infer_path = "tf_direct"
            LOGGER.info("推理路径：直接调用推理图（批量合并）")
        except Exception as exc:  # pragma: no cover
            from .model import predict_probs

            def _infer_fallback(window_raw: np.ndarray):
                Xs = self.scaler.transform(window_raw.reshape(-1, len(self.feature_names)))
                Xs = Xs.reshape(window_raw.shape).astype(np.float32)
                return predict_probs(self.model, Xs, batch_size=len(Xs))

            self._infer = _infer_fallback
            self.infer_path = "keras_predict"
            LOGGER.warning("直接调用不可用（%s），回退 Keras predict（时延更高）", exc)

    def _vectorize(self, row: dict) -> np.ndarray:
        """把一条流记录（按特征名给的字典）转成模型输入向量（未标准化）。"""
        missing = [f for f in self.feature_names if f not in row]
        if missing:
            raise KeyError(f"缺少特征列：{missing[:5]}{'…' if len(missing) > 5 else ''}")
        vals = [float(row[f]) if row[f] not in (None, "") else 0.0 for f in self.feature_names]
        return np.asarray(vals, dtype=np.float64).reshape(1, -1)

    def _decide_action(self, src_ip: str | None, unknown: bool = False) -> str:
        """决定这条告警建议什么动作。未知形态走**保守分支**。

        未知形态即便在全局策略为 block 时也最多限速：未知 ≠ 恶意，
        一次误配就把未知流量全量封禁的风险，远大于"先限速再看一眼"的代价。
        """
        if self.policy.action == "observe":
            return "observe"
        if src_ip and src_ip in tuple(self.policy.whitelist):
            return "observe"
        if unknown:
            aft = str(getattr(self.policy, "unknown_action", "throttle") or "throttle")
            if aft not in ACTIONS:
                aft = "observe"
            # 硬护栏：未知分支**不允许**升级为 block，配置成 block 也降级为 throttle。
            return "throttle" if aft == "block" else aft
        return self.policy.action

    def reset_stream(self, reason: str = "") -> dict:
        """清空滑窗缓冲与连续告警计数（**数据中断/换流后的正确做法**）。

        为什么必须有这个方法：滑窗是"连续 T 条流记录"的语义，一旦上游数据出现**中断或换段**
        （控制器重连、采集重启、回放跳转、日志分段），把中断前后的行拼进同一个窗口就会生成
        **现实中不存在的窗口**——它既不是真实相邻流量，也不属于任何一次离线评估。
        真机上的对应动作是"给流量打一个断点标记，从断点之后重新积累窗口"。
        返回清空前的状态，便于审计"这次中断丢掉了多少已积累的行"。
        """
        with self.lock:
            dropped = len(self.buffer)
            self.buffer.clear()
            self.buffer_rows.clear()
            # 流被重置意味着"攻击可能已经中断"，升级计数也应归零，否则换段后会被误升级。
            self._escalation.clear()
            self.stats["alarm_run"] = 0
            self.stats.setdefault("stream_resets", 0)
            self.stats["stream_resets"] += 1
            if reason:
                LOGGER.warning("流被重置（%s）：丢弃已积累的 %d 条流记录，窗口从头累积", reason, dropped)
            return {"dropped_rows": int(dropped), "reason": reason,
                    "stream_resets": int(self.stats["stream_resets"])}

    # ------------------------------------------------------------------ 主入口
    def push(self, rows: Iterable[dict], meta: dict | None = None) -> dict:
        """推送一批流记录（按特征名给值的字典），返回逐条结果与告警。

        实现要点：**先收集本批产生的所有窗口，再一次性批量推理**，
        因此每批的网络往返只有一次，CPU 也只被占用一次（见端到端实测报告）。
        """
        if not self.ready:
            return {"ok": False, "error": self.reason}
        meta = meta or {}
        stride = max(int(self.cfg.online_stride), 1)
        t0 = time.perf_counter()
        with self.lock:
            pending: list[np.ndarray] = []      # 本批待判定的窗口
            pending_rows: list = []             # 与窗口对应的原始行（用于取 IP、真值）
            out_rows: list[dict] = []
            for i, row in enumerate(rows):
                try:
                    vec = self._vectorize(row)
                except KeyError as exc:
                    return {"ok": False, "error": str(exc), "row_index": i}
                self.buffer.append(vec[0])
                self.buffer_rows.append(self._id_fields(row))
                self.stats["rows"] += 1
                item = {"row_index": i, "window": None}
                if len(self.buffer) == self.buffer.maxlen and self.stats["rows"] % stride == 0:
                    pending.append(np.stack(self.buffer))
                    pending_rows.append((len(out_rows), row))
                out_rows.append(item)

            new_alerts: list[dict] = []
            if pending:
                raw = np.stack(pending)
                p_mc, p_bin = self._infer(raw)
                for k, (row_pos, row) in enumerate(pending_rows):
                    self.stats["windows"] += 1
                    p_attack = float(p_bin[k])
                    idx = int(np.argmax(p_mc[k]))
                    pred_class = self.class_names[idx]
                    # 「未知形态」标记：多分类概率的熵超过阈值 → 模型没见过这种形态。
                    # 注意它**不改变是否告警**（告警仍由二分类头 + 阈值 + 连续确认决定），
                    # 只是把细分类别换成 Unknown，提示运维"这条要人工研判"。
                    novelty = 0.0
                    unknown = False
                    thr_u = float(getattr(self.policy, "unknown_threshold", 0.0) or 0.0)
                    if thr_u > 0:
                        pk = np.clip(np.asarray(p_mc[k], dtype=np.float64), 1e-12, 1.0)
                        pk = pk / pk.sum()
                        novelty = float(-(pk * np.log(pk)).sum())
                        unknown = novelty > thr_u
                        if unknown:
                            pred_class = UNKNOWN_CLASS_NAME
                    alarm = p_attack >= self.policy.threshold
                    self.stats["alarm_run"] = self.stats["alarm_run"] + 1 if alarm else 0
                    confirmed = self.stats["alarm_run"] >= int(self.policy.confirm_windows)
                    if alarm:
                        self.stats["alarms"] += 1
                    if unknown:
                        self.stats["unknown_windows"] = int(self.stats.get("unknown_windows", 0)) + 1
                    # 回看环形缓冲：**每一个**窗口都留一份分数（不论报没报），
                    # 这样事后追问"这段为什么没报警"时能直接调出当时的证据，而不是无从查证。
                    self.recent.append({
                        "window_index": int(self.stats["windows"]),
                        "time": round(time.time(), 3),
                        "p_attack": round(p_attack, 4),
                        "pred_class": pred_class,
                        "novelty": round(float(novelty), 4),
                        "unknown": bool(unknown),
                        "alarm": bool(alarm),
                        "confirmed": bool(confirmed),
                    })
                    # 观察池：未确认、但"差一点到阈值"或"形态新颖"的窗口留痕。
                    # 漏报之所以难查，正是因为判为正常的窗口过去不留任何痕迹。
                    if not confirmed:
                        ob_margin = max(0.0, float(getattr(self.policy, "observe_margin", 0.5) or 0.0))
                        near = p_attack >= ob_margin * float(self.policy.threshold)
                        if near or unknown:
                            ob_src = row.get("Src IP") or row.get("Source IP") or meta.get("src_ip")
                            ob_dst = row.get("Dst IP") or row.get("Destination IP") or meta.get("dst_ip")
                            self.observations.append(Observation(
                                time=time.time(), window_index=int(self.stats["windows"]),
                                p_attack=float(p_attack), pred_class=pred_class,
                                novelty=float(novelty), unknown=bool(unknown),
                                src_ip=str(ob_src) if ob_src else None,
                                dst_ip=str(ob_dst) if ob_dst else None,
                                rows_covered=(self.stats["rows"] - self.cfg.window_size + 1,
                                              self.stats["rows"]),
                                # 必须 copy：raw 是整批窗口的大数组，存切片会让整批常驻内存
                                vector=np.array(raw[k], dtype=np.float32),
                                evidence=list(self.buffer_rows)[
                                    -max(0, int(getattr(self.policy, "observe_evidence_rows", 16))):]
                                or [],
                                why="unknown_observed" if unknown else "near_threshold",
                            ))
                            self.stats["observations"] = int(self.stats.get("observations", 0)) + 1
                    if confirmed:
                        src_ip = row.get("Src IP") or row.get("Source IP") or meta.get("src_ip")
                        dst_ip = row.get("Dst IP") or row.get("Destination IP") or meta.get("dst_ip")
                        dst_port = row.get("Dst Port") or row.get("Destination Port") or meta.get("dst_port")
                        try:
                            dst_port = int(dst_port) if dst_port is not None else None
                        except (TypeError, ValueError):
                            dst_port = None
                        base_action = self._decide_action(str(src_ip) if src_ip else None,
                                                          unknown=bool(unknown))
                        action, escalated = self._escalate(str(src_ip) if src_ip else None,
                                                           base_action, bool(unknown))
                        if escalated:
                            self.stats["escalations"] = int(self.stats.get("escalations", 0)) + 1
                        alert = Alert(
                            time=time.time(), window_index=self.stats["windows"], p_attack=p_attack,
                            pred_class=pred_class, true_class=row.get("Label") or row.get("target"),
                            action=action,
                            escalated=bool(escalated),
                            src_ip=str(src_ip) if src_ip else None,
                            dst_ip=str(dst_ip) if dst_ip else None,
                            dst_port=dst_port,
                            rows_covered=(self.stats["rows"] - self.cfg.window_size + 1, self.stats["rows"]),
                            executed=not self.policy.dry_run,
                            unknown=bool(unknown), novelty=float(novelty),
                        )
                        self.alerts.append(alert)
                        self.stats["alerts_total"] = int(self.stats.get("alerts_total", 0)) + 1
                        self.stats["alarms_confirmed"] += 1
                        self.stats["by_class"][pred_class] = self.stats["by_class"].get(pred_class, 0) + 1
                        d = alert.to_dict()
                        if unknown:
                            # 未知形态告警**必须进待研判队列**：系统已经承认自己不认识它，
                            # 那唯一正确的出口就是让人看一眼，并把结论回流成训练样本。
                            d = {**d, "review_id": self._enqueue_review(
                                d, vector=np.array(raw[k], dtype=np.float32))}
                        new_alerts.append(d)
                        self._append_audit(d)
                        LOGGER.warning("告警 #%d：%s p=%.3f 处置=%s%s%s 源=%s",
                                       alert.window_index, pred_class, p_attack, alert.action,
                                       "（升级）" if escalated else "",
                                       "（已执行）" if alert.executed else "（演练）", alert.src_ip)
                    out_rows[row_pos]["window"] = {
                        "p_attack": round(p_attack, 4), "pred_class": pred_class,
                        "alarm": bool(alarm), "confirmed": bool(confirmed),
                        "unknown": bool(unknown), "novelty": round(float(novelty), 4),
                        "probs": {c: float(p_mc[k][j]) for j, c in enumerate(self.class_names)},
                    }
            self.stats["batches"] = int(self.stats.get("batches", 0)) + 1
            self.stats["windows_per_batch"] = len(pending)
        self.stats["latency_ms"].append((time.perf_counter() - t0) * 1000.0)
        return {
            "ok": True,
            "rows": out_rows,
            "alerts": new_alerts,
            "policy": self._policy_dict(),
            "stats": self.snapshot(),
        }

    def _policy_dict(self) -> dict:
        """当前策略快照（供 /api/policy 与 /api/detect 回显，含未知标记阈值）。"""
        return {"threshold": self.policy.threshold,
                "confirm_windows": self.policy.confirm_windows,
                "action": self.policy.action,
                "dry_run": self.policy.dry_run,
                "whitelist": list(self.policy.whitelist),
                "unknown_threshold": float(getattr(self.policy, "unknown_threshold", 0.0) or 0.0),
                "unknown_action": str(getattr(self.policy, "unknown_action", "throttle")),
                "observe_margin": float(getattr(self.policy, "observe_margin", 0.5) or 0.0)}

    def _append_audit(self, record: dict) -> None:
        """写审计台账（JSONL，追加 + 超限轮转）：出事可追溯，这是"可用系统"的基本要求。

        轮转是必需的：台账只追加不删除，长期运行会把磁盘写满，
        也会让"出事回查"变成打不开的巨型文件（见 `src/utils.py::append_jsonl`）。
        """
        try:
            append_jsonl(self.audit_path, record,
                         max_bytes=int(getattr(self.policy, "audit_max_bytes", 0) or 0),
                         backups=int(getattr(self.policy, "audit_backups", 3) or 3))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("审计日志写入失败：%s", exc)

    # ------------------------------------------------------------------ 观测
    def _id_fields(self, row: dict) -> dict:
        """从一条流记录里挑出**定位用**字段（五元组/协议/标签），作为观察池证据。

        为什么不整条留下：完整 71 维特征 × 500 条观察项 × 16 行会让观察池吃到几十 MB。
        复核真正需要的是"这条窗口里是哪些流"，而不是全部标准化后的数值。
        """
        keep = ("Src IP", "Source IP", "Dst IP", "Destination IP", "Src Port", "Source Port",
                "Dst Port", "Destination Port", "Protocol", "Label", "target")
        return {k: row[k] for k in keep if k in row}

    def _escalate(self, src_ip: str | None, base_action: str, unknown: bool) -> tuple[str, bool]:
        """反复告警的源自动升级动作，返回 (动作, 是否发生了升级)。

        为什么需要：单窗口判定看的是"此刻像不像攻击"，而持续攻击的特征是**反复触发**。
        同一源在时间窗内连续告警，说明它不是偶发抖动，应当抬高处置级别。
        未知形态受硬护栏约束——升级也**不得越过 throttle**。
        """
        if not src_ip or base_action not in ACTIONS:
            return base_action, False
        now = time.time()
        win = max(1, int(getattr(self.policy, "escalate_window_sec", 300) or 300))
        need = max(2, int(getattr(self.policy, "escalate_after", 3) or 3))
        hist = self._escalation.setdefault(str(src_ip), [])
        hist[:] = [t for t in hist if now - t <= win]
        hist.append(now)
        # 有界：跟踪的源 IP 数量也要有上限，否则被大量不同源刷爆内存
        if len(self._escalation) > 2048:
            for k in list(self._escalation)[:1024]:
                self._escalation.pop(k, None)
        if len(hist) < need:
            return base_action, False
        idx = min(len(ACTIONS) - 1, ACTIONS.index(base_action) + 1)
        esc = ACTIONS[idx]
        if unknown and ACTIONS.index(esc) > ACTIONS.index("throttle"):
            esc = "throttle"                       # 未知分支硬护栏
        return esc, esc != base_action

    # ------------------------------------------------------ 待研判队列与人工回流
    def _write_evidence(self, review_id: int, kind: str, window_index: int,
                        vector, extra: dict | None = None) -> None:
        """把进入研判队列的窗口特征向量落盘，使人工结论**真的可以用于重训**。

        为什么必须单独存：`review_feedback.jsonl` 只有结论与分数、**没有特征**，
        那样标注完也训不了模型，"未知 → 已知"就只是话术。
        证据与结论分开存并以 `review_id` 关联——结论可以改判，特征不必重复写。
        落盘失败只告警、不抛异常：证据缺失不该让检测主链路中断。
        """
        if not bool(getattr(self.policy, "review_evidence", True)) or vector is None:
            return
        try:
            arr = np.asarray(vector, dtype=np.float32).ravel()
            append_jsonl(self.review_evidence_path, {
                "review_id": int(review_id), "kind": str(kind),
                "window_index": int(window_index),
                "window_size": int(self.cfg.window_size),
                "n_features": int(len(self.feature_names)),
                # 自带特征名，使该文件**自描述**——否则日后重训无法确定列序
                "feature_names": list(self.feature_names),
                "X": [round(float(v), 6) for v in arr],
                "time": round(time.time(), 3),
                **(extra or {}),
            }, max_bytes=int(getattr(self.policy, "audit_max_bytes", 0) or 0),
                backups=int(getattr(self.policy, "audit_backups", 3) or 3))
            self.stats["review_evidence"] = int(self.stats.get("review_evidence", 0)) + 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("研判证据落盘失败（不影响检测主链路）：%s", exc)

    def _enqueue_review(self, alert: dict, vector=None) -> int:
        """把一条告警放进待研判队列，返回 review_id。"""
        self.review_seq += 1
        self.review_queue.append({
            "review_id": self.review_seq,
            "kind": "alert",
            "status": "pending",
            "enqueued_at": round(time.time(), 3),
            "window_index": alert.get("window_index"),
            "p_attack": alert.get("p_attack"),
            "pred_class": alert.get("pred_class"),
            "novelty": alert.get("novelty"),
            "action": alert.get("action"),
            "src_ip": alert.get("src_ip"),
            "dst_ip": alert.get("dst_ip"),
            "why": "unknown_shape",       # 未知形态告警：模型不知道它是什么
            "verdict": None,
            "note": "",
        })
        self.stats["review_enqueued"] = int(self.stats.get("review_enqueued", 0)) + 1
        self._write_evidence(self.review_seq, "alert", int(alert.get("window_index") or 0), vector,
                             {"pred_class": alert.get("pred_class"),
                              "novelty": alert.get("novelty"),
                              "p_attack": alert.get("p_attack")})
        return self.review_seq

    def promote_observation(self, index: int = -1, why: str = "manual") -> dict:
        """把观察池里的一条（默认最新一条）提升进待研判队列。

        用途：值班员做漏报复盘时，或某个源 IP 反复出现在观察池里时，
        把它拉进队列做一次正式研判，而不是让它一直躺在池子里。
        """
        with self.lock:
            pool = list(self.observations)
            if not pool:
                return {"ok": False, "error": "观察池为空"}
            try:
                ob = pool[index]
            except IndexError:
                return {"ok": False, "error": f"观察池下标越界：{index}（当前 {len(pool)} 条）"}
            self.review_seq += 1
            item = {**ob.to_dict(), "review_id": self.review_seq, "kind": "observation",
                    "status": "pending", "enqueued_at": round(time.time(), 3),
                    "why": str(why or "manual"), "verdict": None, "note": ""}
            self.review_queue.append(item)
            self.stats["review_enqueued"] = int(self.stats.get("review_enqueued", 0)) + 1
            self._write_evidence(self.review_seq, "observation", int(ob.window_index), ob.vector,
                                 {"why": str(why or "manual"), "pred_class": ob.pred_class,
                                  "p_attack": round(float(ob.p_attack), 4)})
            return {"ok": True, "item": item}

    def list_review(self, limit: int = 50, status: str | None = None) -> list[dict]:
        """列出待研判队列（默认全部状态，按入队顺序取最后 limit 条）。"""
        items = list(self.review_queue)
        if status:
            items = [x for x in items if x.get("status") == status]
        return items[-max(1, int(limit)):]

    def list_observations(self, limit: int = 50) -> list[dict]:
        """列出观察池（差一点到阈值 / 形态新颖但未确认的窗口）。"""
        return [o.to_dict() for o in list(self.observations)[-max(1, int(limit)):]]

    def label_review(self, review_id: int, verdict: str, note: str = "") -> dict:
        """对一条待研判项给出人工结论，并**回流**落盘。这是"未知 → 已知"闭环的落地点。

        verdict 取值：`attack:<类别名>`（确认是攻击并给出类别）或 `normal`（确认为误报/正常）。
        结论以 JSONL 追加写入 `results/review_feedback.jsonl`，下一次重训直接读它，
        而不是靠人回忆当时判断了什么。
        """
        v = str(verdict or "").strip()
        if v != "normal" and not v.startswith("attack:"):
            return {"ok": False, "error": "verdict 必须是 'normal' 或 'attack:<类别名>'"}
        with self.lock:
            target = None
            for it in self.review_queue:
                if int(it.get("review_id", -1)) == int(review_id):
                    target = it
                    break
            if target is None:
                return {"ok": False, "error": f"待研判队列里没有 review_id={review_id}"}
            target["status"] = "labeled"
            target["verdict"] = v
            target["note"] = str(note or "")
            target["labeled_at"] = round(time.time(), 3)
            rec = {
                "review_id": int(review_id), "verdict": v, "note": str(note or ""),
                "kind": target.get("kind"), "why": target.get("why"),
                "p_attack": target.get("p_attack"), "pred_class": target.get("pred_class"),
                "novelty": target.get("novelty"), "src_ip": target.get("src_ip"),
                "window_index": target.get("window_index"), "labeled_at": target["labeled_at"],
            }
            try:
                append_jsonl(self.feedback_path, rec,
                             max_bytes=int(getattr(self.policy, "audit_max_bytes", 0) or 0),
                             backups=int(getattr(self.policy, "audit_backups", 3) or 3))
            except OSError as exc:
                # 落盘失败**不能吞掉**：否则"回流"会无声失灵，而这正是本机制的全部意义。
                LOGGER.error("标注回流写入失败：%s", exc)
                return {"ok": False, "error": f"结论已记入队列，但回流文件写入失败：{exc}",
                        "item": target}
            self.stats["review_labeled"] = int(self.stats.get("review_labeled", 0)) + 1
            if v.startswith("attack:"):
                self.stats["review_confirmed_attack"] = int(
                    self.stats.get("review_confirmed_attack", 0)) + 1
            return {"ok": True, "item": target, "feedback_path": str(self.feedback_path)}

    def snapshot(self) -> dict:
        lat = np.asarray(list(self.stats["latency_ms"]), dtype=float) if self.stats["latency_ms"] else np.zeros(1)
        return {
            "ready": self.ready,
            "rows": int(self.stats["rows"]),
            "windows": int(self.stats["windows"]),
            "alarms": int(self.stats["alarms"]),
            "alarms_confirmed": int(self.stats["alarms_confirmed"]),
            "alerts_total": int(self.stats.get("alerts_total", 0)),
            "alerts_retained": len(self.alerts),
            "escalations": int(self.stats.get("escalations", 0)),
            "by_class": dict(self.stats["by_class"]),
            "unknown_windows": int(self.stats.get("unknown_windows", 0)),
            "observations": len(self.observations),
            "observations_total": int(self.stats.get("observations", 0)),
            "review_pending": sum(1 for x in self.review_queue if x.get("status") == "pending"),
            "review_total": len(self.review_queue),
            "review_labeled": int(self.stats.get("review_labeled", 0)),
            "review_confirmed_attack": int(self.stats.get("review_confirmed_attack", 0)),
            "review_evidence_total": int(self.stats.get("review_evidence", 0)),
            "stream_resets": int(self.stats.get("stream_resets", 0)),
            "uptime_sec": round(time.time() - self.stats["started_at"], 1),
            "batches": int(self.stats.get("batches", 0)),
            "batches_per_sec": round(self.stats.get("batches", 0) / max(time.time() - self.stats["started_at"], 1e-9), 2),
            "infer_path": getattr(self, "infer_path", "unknown"),
            "online_stride": int(self.cfg.online_stride),
            "batch_latency_ms": {
                "p50": round(float(np.percentile(lat, 50)), 3),
                "p95": round(float(np.percentile(lat, 95)), 3),
                "p99": round(float(np.percentile(lat, 99)), 3),
            },
            "policy": self._policy_dict(),
            "feature_names": self.feature_names,
            "class_names": self.class_names,
        }

    def prometheus(self) -> str:
        """暴露 Prometheus 文本格式指标，便于接入现有监控体系。"""
        s = self.snapshot()
        lines = [
            "# HELP sdn_ids_rows_total 已处理的流记录总数",
            "# TYPE sdn_ids_rows_total counter",
            f"sdn_ids_rows_total {s['rows']}",
            "# HELP sdn_ids_windows_total 已判定的窗口数",
            "# TYPE sdn_ids_windows_total counter",
            f"sdn_ids_windows_total {s['windows']}",
            "# HELP sdn_ids_alarms_confirmed_total 已确认告警数",
            "# TYPE sdn_ids_alarms_confirmed_total counter",
            f"sdn_ids_alarms_confirmed_total {s['alarms_confirmed']}",
            "# HELP sdn_ids_batch_latency_ms 批量推理时延分位数",
            "# TYPE sdn_ids_batch_latency_ms gauge",
            f"sdn_ids_batch_latency_ms{{quantile=\"0.5\"}} {s['batch_latency_ms']['p50']}",
            f"sdn_ids_batch_latency_ms{{quantile=\"0.95\"}} {s['batch_latency_ms']['p95']}",
            "# HELP sdn_ids_up 服务就绪状态",
            "# TYPE sdn_ids_up gauge",
            f"sdn_ids_up {1 if s['ready'] else 0}",
            # 「未知形态」窗口数与当前阈值：**未知占比本身就是一条免费的漂移指标**——
            # 它突然升高通常意味着新攻击出现或流量分布变化，值得立刻人工看一眼。
            "# HELP sdn_ids_unknown_windows_total 被标记为「未知形态」的窗口数",
            "# TYPE sdn_ids_unknown_windows_total counter",
            f"sdn_ids_unknown_windows_total {s['unknown_windows']}",
            "# HELP sdn_ids_unknown_threshold 当前未知标记阈值（0=关闭）",
            "# TYPE sdn_ids_unknown_threshold gauge",
            f"sdn_ids_unknown_threshold {s['policy']['unknown_threshold']}",
            "# HELP sdn_ids_stream_resets_total 滑窗缓冲被重置的次数（上游数据中断）",
            "# TYPE sdn_ids_stream_resets_total counter",
            f"sdn_ids_stream_resets_total {s['stream_resets']}",
            # 观察池与待研判队列：**待研判积压量是一条运维健康指标**——
            # 它持续大于 0 说明有人在等结论，团队没有消化系统发现的新东西。
            "# HELP sdn_ids_observations 观察池当前条数（差一点到阈值 / 形态新颖但未确认的窗口）",
            "# TYPE sdn_ids_observations gauge",
            f"sdn_ids_observations {s['observations']}",
            "# HELP sdn_ids_review_pending 待研判队列里未处理项数（>0 表示有人在等结论）",
            "# TYPE sdn_ids_review_pending gauge",
            f"sdn_ids_review_pending {s['review_pending']}",
            "# HELP sdn_ids_review_confirmed_attack_total 人工研判确认为攻击并已回流的条数",
            "# TYPE sdn_ids_review_confirmed_attack_total counter",
            f"sdn_ids_review_confirmed_attack_total {s['review_confirmed_attack']}",
            # 累计告警数是**计数器**（不受内存保留条数影响）；内存里只保留最近 alerts_retained 条。
            "# HELP sdn_ids_alerts_total 累计产生的告警数（不受内存保留上限影响）",
            "# TYPE sdn_ids_alerts_total counter",
            f"sdn_ids_alerts_total {s['alerts_total']}",
            "# HELP sdn_ids_escalations_total 因同源反复告警而自动升级动作的次数",
            "# TYPE sdn_ids_escalations_total counter",
            f"sdn_ids_escalations_total {s['escalations']}",
        ]
        for cls, n in s["by_class"].items():
            lines.append(f'sdn_ids_alarms_by_class_total{{class="{cls}"}} {n}')
        return "\n".join(lines) + "\n"

    def reload_policy(self, **kw) -> dict:
        for k, v in kw.items():
            if not hasattr(self.policy, k):
                continue
            if k == "whitelist" and isinstance(v, (list, tuple)):
                setattr(self.policy, k, tuple(str(x) for x in v))
            elif k in ("threshold", "confirm_windows", "unknown_threshold", "observe_margin"):
                cur = getattr(self.policy, k)
                newv = type(cur)(v)
                if k == "unknown_threshold":
                    # 负值无意义；0 = 关闭。顺手夹到合法区间，避免误配把每个窗口都标成未知。
                    newv = max(0.0, float(newv))
                if k == "observe_margin":
                    # 观察池下界比例，夹到 [0,1]：>1 会让观察池永远为空（静默失效）。
                    newv = min(1.0, max(0.0, float(newv)))
                setattr(self.policy, k, newv)
            elif k in ("action", "dry_run", "unknown_action"):
                if k in ("action", "unknown_action") and str(v) not in ACTIONS:
                    continue
                setattr(self.policy, k, bool(v) if k == "dry_run" else str(v))
        LOGGER.info("策略已更新：threshold=%.2f confirm=%d action=%s dry_run=%s "
                    "unknown_threshold=%.4f unknown_action=%s",
                    self.policy.threshold, self.policy.confirm_windows, self.policy.action,
                    self.policy.dry_run, float(getattr(self.policy, "unknown_threshold", 0.0) or 0.0),
                    str(getattr(self.policy, "unknown_action", "throttle")))
        return self.snapshot()["policy"]
