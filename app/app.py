"""可视化演示服务：把"训练好的模型 + 留出测试流"变成可现场演示的实时检测大屏。

设计要点（为比赛现场考虑）
--------------------------
1. **不依赖 TensorFlow**：检测所需的一切（窗口级预测、注意力权重、指标）在
   ``run_all.py`` 阶段就缓存好了，演示服务只做"回放 + 呈现"，
   因此在评委电脑上没装 TF 也能跑起来。
2. **完全离线**：不引用任何 CDN，前端图表用原生 canvas/SVG 绘制，断网照常演示。
3. **真实留出数据**：回放的是**测试集**里的真实流记录与模型真实预测，
   不是编出来的动画；"注入攻击"只是把播放位置跳到真实攻击片段起点，
   因此看到的告警时延、误报都是真实行为。
4. **可交互**：可暂停/单步/调速，可一键注入 8 类流量，可开关告警阈值。

启动：
    python app/app.py                 # 默认 http://127.0.0.1:5000
    python app/app.py --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request, send_file  # noqa: E402

from src.config import Config  # noqa: E402
from src.utils import ARTIFACT_DIR, PROCESSED_DIR, RESULT_DIR, FIGURE_DIR, get_logger  # noqa: E402

LOGGER = get_logger("cida.demo")
app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))

# 由 --model-tag 指定（如 sdn_ctrl）；空串表示用 run_all.py 记录的"主模型"。
# 注意：必须在 DemoState 实例化之前定义，因为模块级就会创建 STATE。
MODEL_TAG: str = ""

# 展示用的优先特征（按名字匹配，找不到就退回前几个特征）
DISPLAY_FEATURES_PRIORITY = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Packets/s",
    "Flow Bytes/s", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "Idle Mean",
    "Fwd IAT Mean", "Average Packet Size", "Destination Port",
    "packet_count", "byte_count", "flow_duration_sec", "packet_count_per_second",
    "byte_count_per_second", "tp_dst", "ip_proto", "flags",
]


class DemoState:
    """演示所需的状态：测试流、窗口预测、指标、特征名。"""

    def __init__(self, processed_dir: Path | None = None, artifact_dir: Path | None = None):
        self.ok = False
        self.reason = ""
        # 注意：默认值在调用时从模块全局取，这样命令行 `--processed-dir` 能生效
        self.processed_dir = Path(processed_dir or PROCESSED_DIR)
        self.artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
        self.load()

    # ---------------- 数据加载 ----------------
    def load(self) -> None:
        meta_path = self.processed_dir / "meta.json"
        tensors = self.processed_dir / "tensors.npz"
        if not meta_path.exists() or not tensors.exists():
            self.reason = "还没有预处理数据，请先运行：python run_all.py（或 python run_all.py --smoke）"
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        z = np.load(tensors, allow_pickle=True)
        if z["test_stream_features"].size == 0:
            self.reason = "预处理产物里没有测试流数据，请重跑 run_all.py"
            return

        self.feature_names: list[str] = list(meta["feature_names"])
        self.class_names: list[str] = list(meta["class_names"])
        self.window_size = int(meta["window_size"])
        self.input_mode = str(meta.get("input_mode", "sequence"))
        self.X = np.asarray(z["test_stream_features"], dtype=np.float32)
        self.labels = np.asarray(z["test_stream_labels"]).astype(str)
        self.times = np.asarray(z["test_stream_time"], dtype=np.float64)
        self.n_rows = len(self.X)

        # 反标准化 → 展示真实物理量（包数、字节数、时长……）
        try:
            import joblib

            scaler = joblib.load(self.processed_dir / "scaler.joblib")
            idx = [self.feature_names.index(f) for f in DISPLAY_FEATURES_PRIORITY if f in self.feature_names]
            if not idx:
                idx = list(range(min(6, len(self.feature_names))))
            self.display_idx = idx
            self.display_names = [self.feature_names[i] for i in idx]
            self.display_values = scaler.inverse_transform(self.X)[:, idx]
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("反标准化失败（%s），展示标准化后的数值", exc)
            self.display_idx = list(range(min(6, self.X.shape[1])))
            self.display_names = [self.feature_names[i] for i in self.display_idx]
            self.display_values = self.X[:, self.display_idx]

        # 时间轴（相对秒）
        self.t_rel = self.times - self.times[0]

        # ---------------- 窗口级预测 ----------------
        # 优先用 run_all.py 记录的"主模型"（artifacts/main_model.json），
        # 这样演示界面的预测、指标、注意力图三者一定来自同一个模型。
        #
        # ⚠️ 但 artifacts/ 下同时存在多种特征口径的预测缓存（完整流特征 71 维、
        #    控制器原生字段 17 维……）。若直接取"主模型"或"最新的一个"，在切换到
        #    控制器原生口径（--processed-dir data/processed_sdn）时会拿 71 维的预测
        #    去配 17 维的数据流，页面显示的判定与在线接口的判定不是同一个模型。
        #    这里按「预测缓存对应的 metrics 里记录的特征数」与当前预处理产物的特征数
        #    做一致性校验，形状不符的候选一律跳过——宁可页面提示"无预测缓存"，
        #    也不显示错误口径的判定。
        want_f = len(self.feature_names)
        want_t = int(self.window_size)
        main_tag = self._read_main_tag()
        cands: list[Path] = []
        if MODEL_TAG:
            cands.append(self.artifact_dir / f"predictions_{MODEL_TAG}.csv")
        if main_tag:
            cands.append(self.artifact_dir / f"predictions_{main_tag}.csv")
        cands.extend(sorted(self.artifact_dir.glob("predictions_*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True))
        seen: set[Path] = set()
        skipped: list[str] = []
        pred_files: list[Path] = []
        for c in cands:
            c = Path(c)
            if c in seen or not c.exists():
                continue
            seen.add(c)
            tag = c.stem.replace("predictions_", "")
            mt = self._metrics_shape(tag)
            if mt is not None and mt != (want_t, want_f):
                skipped.append(f"{tag}({mt[0]}×{mt[1]})")
                continue
            pred_files = [c]
            break
        if not pred_files and skipped:
            LOGGER.warning("预测缓存与当前数据口径（窗口 %d、特征 %d）都不匹配，已跳过：%s",
                           want_t, want_f, "、".join(skipped))
        if pred_files:
            self.pred_file = pred_files[-1]
            self.tag = self.pred_file.stem.replace("predictions_", "")
            self.pred = pd.read_csv(self.pred_file)
            self.threshold = 0.5
            metrics_path = RESULT_DIR / f"metrics_{self.tag}.json"
            if not metrics_path.exists():
                cands = sorted(RESULT_DIR.glob("metrics_*.json"), key=lambda p: p.stat().st_mtime)
                metrics_path = cands[-1] if cands else None
            if metrics_path is not None and Path(metrics_path).exists():
                self.metrics_file = Path(metrics_path)
                try:
                    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
                    self.threshold = float(m.get("binary", {}).get("threshold", 0.5))
                    self.metrics = m
                except Exception:
                    self.metrics = None
            else:
                self.metrics, self.metrics_file = None, None
        else:
            self.pred = None
            self.tag = ""
            self.threshold = 0.5
            self.metrics, self.metrics_file, self.pred_file = None, None, None

        # 行 → 最近一个"已完成"窗口（模拟在线推理：窗口填满才出结论）
        self.row_window = np.full(self.n_rows, -1, dtype=np.int32)
        if self.pred is not None:
            for i, (s, e) in enumerate(zip(self.pred["win_start"].to_numpy(), self.pred["win_end"].to_numpy())):
                e = min(int(e), self.n_rows - 1)
                self.row_window[max(int(s), 0) : e + 1] = i

        # 攻击片段（用于"注入攻击"）
        self.segments = self._build_segments()
        self.ok = True
        LOGGER.info(
            "演示数据就绪：%d 条流记录，%d 个窗口预测，%d 个攻击片段，阈值 %.2f",
            self.n_rows, 0 if self.pred is None else len(self.pred), len(self.segments), self.threshold,
        )

    def _read_main_tag(self) -> str:
        """读取 run_all.py 写下的主模型标签（没有就返回空串）。"""
        try:
            p = self.artifact_dir / "main_model.json"
            if p.exists():
                return str(json.loads(p.read_text(encoding="utf-8")).get("tag", ""))
        except Exception:
            pass
        return ""

    @staticmethod
    def _metrics_shape(tag: str) -> tuple[int, int] | None:
        """从 metrics_<tag>.json 里读出该预测缓存对应的 (窗口长度, 特征数)。

        用于把「预测缓存」与「当前预处理产物」做口径一致性校验：读不到就返回 None
        （视为未知，允许作为兜底候选）。
        """
        p = RESULT_DIR / f"metrics_{tag}.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8")).get("data", {})
            t = d.get("window_size")
            f = d.get("n_features")
            if t is None or f is None:
                shape = d.get("input_shape") or []
                if len(shape) == 2:
                    t, f = shape
            if t is None or f is None:
                return None
            return int(t), int(f)
        except Exception:
            return None

    def _build_segments(self) -> list[dict]:
        segs: list[dict] = []
        i = 0
        n = self.n_rows
        while i < n:
            if self.labels[i] != "Normal":
                j = i
                while j + 1 < n and self.labels[j + 1] == self.labels[i]:
                    j += 1
                segs.append({
                    "start": int(i), "end": int(j), "class": str(self.labels[i]),
                    "t_start": float(self.t_rel[i]), "t_end": float(self.t_rel[j]),
                    "length": int(j - i + 1),
                })
                i = j + 1
            else:
                i += 1
        return segs

    # ---------------- 提供给前端的数据 ----------------
    def overview(self) -> dict:
        out = {
            "ok": self.ok,
            "reason": self.reason,
            "n_rows": int(self.n_rows),
            "duration_sec": float(self.t_rel[-1] - self.t_rel[0]) if self.n_rows else 0.0,
            "classes": self.class_names,
            "window_size": self.window_size,
            "input_mode": self.input_mode,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            # 告警确认窗口数：与后端在线评估保持同一口径（连续 N 个窗口都超阈值才算告警）
            "confirm_windows": int((self.metrics or {}).get("online", {}).get("confirm_windows", 1) or 1),
            "has_predictions": self.pred is not None,
            "tag": getattr(self, "tag", ""),
            "pred_file": str(self.pred_file.name) if self.pred is not None else None,
            "segments": self.segments,
            "display_features": self.display_names,
        }
        if self.metrics:
            out["metrics"] = {
                "multiclass": {k: v for k, v in self.metrics.get("multiclass", {}).items()
                               if not k.startswith("confusion") and k != "per_class"},
                "per_class": self.metrics.get("multiclass", {}).get("per_class", {}),
                "binary": self.metrics.get("binary", {}),
                "online": self.metrics.get("online", {}),
                "model": self.metrics.get("model", {}),
                "data": {k: v for k, v in self.metrics.get("data", {}).items()
                         if k in ("n_features", "n_classes", "train", "val", "test", "class_names")},
            }
        return out

    def batch(self, pos: int, count: int, threshold: float | None = None) -> dict:
        pos = max(0, min(int(pos), self.n_rows - 1))
        end = min(pos + int(count), self.n_rows)
        thr = float(self.threshold if threshold is None else threshold)
        rows = []
        for i in range(pos, end):
            wi = int(self.row_window[i])
            item = {
                "i": i,
                "t": round(float(self.t_rel[i]), 3),
                "true": str(self.labels[i]),
                "attack": bool(self.labels[i] != "Normal"),
                "feat": [None if not np.isfinite(v) else round(float(v), 3) for v in self.display_values[i]],
                "win": None,
            }
            if wi >= 0 and self.pred is not None:
                r = self.pred.iloc[wi]
                probs = {c: float(r.get(f"p_{c}", 0.0)) for c in self.class_names}
                item["win"] = {
                    "idx": wi,
                    "p": round(float(r["p_attack"]), 4),
                    "pred": str(r["pred_class"]),
                    "alarm": bool(float(r["p_attack"]) >= thr),
                    "probs": probs,
                    # 窗口级真值（"窗口内是否含攻击行"）——与后端在线评估口径保持一致，
                    # 否则界面会把"窗口里确实有攻击、但窗口最后一行是正常"的情况误算成误报。
                    "true": str(r.get("true_class", "")),
                    "true_attack": bool(int(r.get("true_is_attack", 0))),
                }
            rows.append(item)
        return {"pos": pos, "end": end, "rows": rows, "n_rows": self.n_rows}


STATE = DemoState()


def json_safe(obj):
    """把 NaN/Inf 换成 None —— 否则 Flask 会吐出 `NaN` 这种**非法 JSON**，
    浏览器 JSON.parse 直接报错（演示现场最怕这种坑）。"""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


# ======================================================================================
# 路由
# ======================================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    return jsonify(json_safe(STATE.overview()))


@app.route("/api/stream")
def api_stream():
    pos = request.args.get("pos", default=0, type=int)
    count = request.args.get("count", default=40, type=int)
    thr = request.args.get("threshold", default=None, type=float)
    if not STATE.ok:
        return jsonify({"error": STATE.reason}), 400
    return jsonify(json_safe(STATE.batch(pos, count, thr)))


@app.route("/api/segment/<int:idx>")
def api_segment(idx: int):
    """跳转到第 idx 个攻击片段的起点（"注入攻击"按钮）。"""
    if not STATE.ok or not STATE.segments:
        return jsonify({"error": "没有可用的攻击片段"}), 400
    idx = max(0, min(idx, len(STATE.segments) - 1))
    seg = STATE.segments[idx]
    # 回退一点正常流量，便于观察"从正常到异常"的过渡
    pos = max(0, seg["start"] - STATE.window_size)
    return jsonify({"segment": seg, "pos": int(pos)})


@app.route("/api/first_segment_of/<cls>")
def api_first_segment_of(cls: str):
    """按类型注入：返回该攻击类型第一个片段的起点。"""
    if not STATE.ok:
        return jsonify({"error": STATE.reason}), 400
    for idx, seg in enumerate(STATE.segments):
        if seg["class"].lower() == str(cls).lower():
            pos = max(0, seg["start"] - STATE.window_size)
            return jsonify({"segment": seg, "index": idx, "pos": int(pos)})
    return jsonify({"error": f"测试流里没有 {cls} 片段"}), 404


@app.route("/api/metrics")
def api_metrics():
    return jsonify(STATE.metrics or {})


@app.route("/api/figures")
def api_figures():
    if not FIGURE_DIR.exists():
        return jsonify([])
    return jsonify(sorted(p.name for p in FIGURE_DIR.glob("*.png")))


@app.route("/figures/<path:name>")
def figure_file(name: str):
    path = FIGURE_DIR / Path(name).name
    if not path.exists():
        return jsonify({"error": "figure not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/health")
def api_health():
    return jsonify({"ok": STATE.ok, "reason": STATE.reason})


# ======================================================================================
# 在线检测服务（给 SDN 控制器调用）—— 这是"课题"与"可用系统"的分界线
# ======================================================================================
DETECTOR = None
DETECTOR_LOCK = threading.Lock()


def get_detector():
    """惰性加载检测服务（避免没有 TensorFlow 时连演示大屏都起不来）。"""
    global DETECTOR
    with DETECTOR_LOCK:
        if DETECTOR is None:
            from src.service import DetectionService

            DETECTOR = DetectionService(processed_dir=PROCESSED_DIR, artifact_dir=ARTIFACT_DIR,
                                        model_tag=MODEL_TAG)
            if DETECTOR.ready:
                LOGGER.info("在线检测服务已就绪：%s", DETECTOR.model_path)
            else:
                LOGGER.warning("在线检测服务不可用：%s", DETECTOR.reason)
    return DETECTOR


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """控制器推送一批流记录 → 返回逐条判定、告警与处置建议。

    请求体（按特征名给值，字段名与 `feature_names` 一致）::

        {"meta": {"src_ip": "10.0.0.5", "dst_ip": "10.0.0.1", "dst_port": 80},
         "rows": [{"Flow Duration": 1234, "Tot Fwd Pkts": 12, ...}, ...]}

    响应：逐条窗口判定 + 新产生的告警（含处置动作）+ 运行统计。
    """
    det = get_detector()
    if not det.ready:
        return jsonify({"ok": False, "error": det.reason,
                        "hint": "先运行 python run_all.py 生成模型与预处理产物，并确认已安装 TensorFlow"}), 503
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return jsonify({"ok": False, "error": "rows 必须是非空数组"}), 400
    return jsonify(json_safe(det.push(rows, meta=payload.get("meta") or {})))


@app.route("/api/alerts")
def api_alerts():
    """告警审计台账（可用系统的基本要求：每条告警可追溯）。"""
    det = get_detector()
    limit = request.args.get("limit", default=100, type=int)
    return jsonify(json_safe([a.to_dict() for a in det.alerts[-limit:]][::-1]))


@app.route("/api/policy", methods=["GET", "POST"])
def api_policy():
    """查询/热更新检测策略（阈值、确认窗口数、处置动作、白名单、演练模式）。"""
    det = get_detector()
    if request.method == "GET":
        return jsonify(json_safe(det.snapshot()["policy"]))
    body = request.get_json(silent=True) or {}
    return jsonify(json_safe(det.reload_policy(**body)))


@app.route("/api/stats")
def api_stats():
    det = get_detector()
    return jsonify(json_safe(det.snapshot()))


@app.route("/metrics")
def metrics():
    """Prometheus 文本格式指标，可直接被现有监控系统抓取。"""
    from flask import Response

    return Response(get_detector().prometheus(), mimetype="text/plain; version=0.0.4")


def main(argv: list[str] | None = None) -> int:
    global STATE, DETECTOR, PROCESSED_DIR, ARTIFACT_DIR, MODEL_TAG
    cfg = Config()
    ap = argparse.ArgumentParser(description="CNN+BiLSTM 异常检测演示大屏")
    # 默认值支持用环境变量配置（服务化部署：见 deploy/cida-detect.env 与 deploy/cida-detect.service）
    ap.add_argument("--host", default=os.environ.get("CIDA_HOST", cfg.demo_host))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CIDA_PORT", cfg.demo_port)))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--processed-dir", default=os.environ.get("CIDA_PROCESSED_DIR") or None,
                    help="预处理产物目录（默认 data/processed；控制器原生字段场景用 data/processed_sdn）")
    ap.add_argument("--artifact-dir", default=os.environ.get("CIDA_ARTIFACT_DIR") or None,
                    help="模型产物目录（默认 artifacts）")
    ap.add_argument("--model-tag", default=os.environ.get("CIDA_MODEL_TAG") or None,
                    help="指定模型子目录名（如 sdn_ctrl）；默认用 run_all.py 记录的主模型")
    args = ap.parse_args(argv)

    if args.processed_dir or args.artifact_dir or args.model_tag:
        if args.processed_dir:
            PROCESSED_DIR = Path(args.processed_dir)
        if args.artifact_dir:
            ARTIFACT_DIR = Path(args.artifact_dir)
        MODEL_TAG = args.model_tag or ""
        DETECTOR = None  # 触发按需重建，确保检测服务与演示大屏用同一口径
        STATE = DemoState(PROCESSED_DIR, ARTIFACT_DIR)
        LOGGER.info("已切换数据/模型口径：processed=%s artifacts=%s tag=%s",
                    PROCESSED_DIR, ARTIFACT_DIR, MODEL_TAG or "（主模型）")

    if not STATE.ok:
        LOGGER.warning("演示数据未就绪：%s", STATE.reason)
    LOGGER.info("演示服务启动： http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
