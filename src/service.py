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

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import Config
from .utils import ARTIFACT_DIR, PROCESSED_DIR, RESULT_DIR, get_logger, load_json, resolve_dir

LOGGER = get_logger("cida.service")

ACTIONS = ("observe", "throttle", "block")


@dataclass
class DetectorConfig:
    """在线服务的运行参数（可由 /api/policy 热更新）。"""

    threshold: float = 0.5
    confirm_windows: int = 3
    action: str = "observe"          # observe=只告警；throttle=建议限速；block=建议封禁
    whitelist: tuple = ()            # 永不处置的源 IP（例如控制器自身、管理网段）
    dry_run: bool = True             # True=只输出建议动作，不真正下发


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
        }


class DetectionService:
    """在线检测服务：喂流记录 → 得到判定、告警与处置建议。"""

    def __init__(self, cfg: Config | None = None, processed_dir: Path | None = None,
                 artifact_dir: Path | None = None, model_tag: str | None = None):
        self.cfg = cfg or Config()
        self.processed_dir = Path(processed_dir or resolve_dir(self.cfg.processed_dir, PROCESSED_DIR))
        self.artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
        self.model_tag = model_tag
        self.policy = DetectorConfig(
            threshold=float(self.cfg.report_threshold),
            confirm_windows=int(self.cfg.online_confirm_windows),
            action="observe",
            dry_run=True,
        )
        self.lock = threading.Lock()
        self.buffer: deque = deque(maxlen=int(self.cfg.window_size))
        self.alerts: list[Alert] = []
        self.audit_path = RESULT_DIR / "audit_alerts.jsonl"
        self.stats: dict[str, Any] = {
            "rows": 0, "windows": 0, "alarms": 0, "alarms_confirmed": 0,
            "alarm_run": 0, "latency_ms": deque(maxlen=2000),
            "by_class": {}, "started_at": time.time(),
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

    def _decide_action(self, src_ip: str | None) -> str:
        if self.policy.action == "observe":
            return "observe"
        if src_ip and src_ip in tuple(self.policy.whitelist):
            return "observe"
        return self.policy.action

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
                    alarm = p_attack >= self.policy.threshold
                    self.stats["alarm_run"] = self.stats["alarm_run"] + 1 if alarm else 0
                    confirmed = self.stats["alarm_run"] >= int(self.policy.confirm_windows)
                    if alarm:
                        self.stats["alarms"] += 1
                    if confirmed:
                        src_ip = row.get("Src IP") or row.get("Source IP") or meta.get("src_ip")
                        dst_ip = row.get("Dst IP") or row.get("Destination IP") or meta.get("dst_ip")
                        dst_port = row.get("Dst Port") or row.get("Destination Port") or meta.get("dst_port")
                        try:
                            dst_port = int(dst_port) if dst_port is not None else None
                        except (TypeError, ValueError):
                            dst_port = None
                        alert = Alert(
                            time=time.time(), window_index=self.stats["windows"], p_attack=p_attack,
                            pred_class=pred_class, true_class=row.get("Label") or row.get("target"),
                            action=self._decide_action(str(src_ip) if src_ip else None),
                            src_ip=str(src_ip) if src_ip else None,
                            dst_ip=str(dst_ip) if dst_ip else None,
                            dst_port=dst_port,
                            rows_covered=(self.stats["rows"] - self.cfg.window_size + 1, self.stats["rows"]),
                            executed=not self.policy.dry_run,
                        )
                        self.alerts.append(alert)
                        self.stats["alarms_confirmed"] += 1
                        self.stats["by_class"][pred_class] = self.stats["by_class"].get(pred_class, 0) + 1
                        d = alert.to_dict()
                        new_alerts.append(d)
                        self._append_audit(d)
                        LOGGER.warning("告警 #%d：%s p=%.3f 处置=%s%s 源=%s",
                                       alert.window_index, pred_class, p_attack, alert.action,
                                       "（已执行）" if alert.executed else "（演练）", alert.src_ip)
                    out_rows[row_pos]["window"] = {
                        "p_attack": round(p_attack, 4), "pred_class": pred_class,
                        "alarm": bool(alarm), "confirmed": bool(confirmed),
                        "probs": {c: float(p_mc[k][j]) for j, c in enumerate(self.class_names)},
                    }
            self.stats["batches"] = int(self.stats.get("batches", 0)) + 1
            self.stats["windows_per_batch"] = len(pending)
        self.stats["latency_ms"].append((time.perf_counter() - t0) * 1000.0)
        return {
            "ok": True,
            "rows": out_rows,
            "alerts": new_alerts,
            "policy": {"threshold": self.policy.threshold, "confirm_windows": self.policy.confirm_windows,
                       "action": self.policy.action, "dry_run": self.policy.dry_run},
            "stats": self.snapshot(),
        }

    def _append_audit(self, record: dict) -> None:
        """写审计台账（JSONL，追加）：出事可追溯，这是"可用系统"的基本要求。"""
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("审计日志写入失败：%s", exc)

    # ------------------------------------------------------------------ 观测
    def snapshot(self) -> dict:
        lat = np.asarray(list(self.stats["latency_ms"]), dtype=float) if self.stats["latency_ms"] else np.zeros(1)
        return {
            "ready": self.ready,
            "rows": int(self.stats["rows"]),
            "windows": int(self.stats["windows"]),
            "alarms": int(self.stats["alarms"]),
            "alarms_confirmed": int(self.stats["alarms_confirmed"]),
            "by_class": dict(self.stats["by_class"]),
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
            "policy": {"threshold": self.policy.threshold, "confirm_windows": self.policy.confirm_windows,
                       "action": self.policy.action, "dry_run": self.policy.dry_run},
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
            elif k in ("threshold", "confirm_windows"):
                cur = getattr(self.policy, k)
                setattr(self.policy, k, type(cur)(v))
            elif k in ("action", "dry_run"):
                if k == "action" and str(v) not in ACTIONS:
                    continue
                setattr(self.policy, k, bool(v) if k == "dry_run" else str(v))
        LOGGER.info("策略已更新：threshold=%.2f confirm=%d action=%s dry_run=%s",
                    self.policy.threshold, self.policy.confirm_windows, self.policy.action, self.policy.dry_run)
        return self.snapshot()["policy"]
