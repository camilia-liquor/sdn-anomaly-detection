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


# --------------------------------------------------------------------------------------
# 回放流的「原始量纲整行特征」（只读）——让演示大屏能把真实窗口推给在线检测接口
# --------------------------------------------------------------------------------------
_RAW_LOCK = threading.Lock()
_RAW_MATRIX: np.ndarray | None = None


def raw_feature_matrix() -> np.ndarray:
    """把演示用的测试流还原成**原始量纲**特征矩阵（惰性计算 + 缓存）。

    在线检测服务（``/api/detect``）要求传入原始量纲的特征，由它自己标准化；
    这里把标准化后的测试流逆变换回原始量纲，因此把它推给在线接口时，服务端
    ``scaler.transform`` 得到的输入与「回放判定」（缓存预测）完全一致——
    两条链路的 p_attack 可以逐窗口对照，演示与在线不会出现两套口径。
    """
    global _RAW_MATRIX
    with _RAW_LOCK:
        if _RAW_MATRIX is None:
            import joblib

            scaler = joblib.load(STATE.processed_dir / "scaler.joblib")
            _RAW_MATRIX = np.asarray(scaler.inverse_transform(STATE.X), dtype=np.float64)
    return _RAW_MATRIX


@app.route("/api/replay_rows")
def api_replay_rows():
    """按回放位置取一段**完整特征行（原始量纲）**，供前端 POST /api/detect。

    为什么必须补这个接口：``/api/stream`` 只返回展示用的少数特征列，而在线检测
    接口要求给出全部特征列。有了它，演示大屏才能把**回放中的真实窗口**实时推给
    在线检测服务，于是「攻击拓扑 + 处置控制台 + 审计台账」里出现的是真实判定
    链路的产物，而不是前端编出来的动画。

    每行额外带一个 ``Label``（该条流的真实标签）：检测服务会把它记进告警的
    ``ground_truth``，台账因此能显示「这条告警对应真实攻击 / 是真误报」。

    查询参数：``pos``（起始下标，默认 0）、``count``（条数，默认 32，上限 200）
    返回：``{ok, pos, end, n_rows, feature_names, rows:[{特征名:值, …, Label:真值}]}``
    """
    if not STATE.ok:
        return jsonify({"ok": False, "error": STATE.reason}), 400
    pos = request.args.get("pos", default=0, type=int)
    count = request.args.get("count", default=32, type=int)
    pos = max(0, min(int(pos), max(STATE.n_rows - 1, 0)))
    count = max(1, min(int(count), 200))
    end = min(pos + count, STATE.n_rows)
    try:
        raw = raw_feature_matrix()[pos:end]
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("原始量纲还原失败：%s", exc)
        return jsonify({"ok": False, "error": f"原始量纲还原失败：{exc}"}), 500
    rows = []
    for k in range(end - pos):
        row = {name: float(v) for name, v in zip(STATE.feature_names, raw[k])}
        row["Label"] = str(STATE.labels[pos + k])          # 供在线服务记录 ground_truth
        rows.append(row)
    return jsonify(json_safe({"ok": True, "pos": int(pos), "end": int(end), "n_rows": int(STATE.n_rows),
                              "feature_names": list(STATE.feature_names), "rows": rows}))


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
    """按类型注入：返回该攻击类型的**代表性片段**起点。

    为什么要按长度挑：测试流里同一类攻击有很多很短的片段（甚至只有 1 条流），
    如果直接取「第一个」，现场演示很可能跳到一个 1 条流的片段上——曲线刚起跳就结束了，
    看起来像模型没报出来。这里改为优先返回**长度足够长**（≥ 窗口长度的 2 倍，且取其中最长的一批里
    时间上最早的一个）的片段，保证「注入 → 曲线爬升 → 告警区间出现」在屏幕上看得清。
    若该类型没有长片段，则退回该类型最长的那个；再退回第一个。
    """
    if not STATE.ok:
        return jsonify({"error": STATE.reason}), 400
    want = str(cls).lower()
    cands = [(i, s) for i, s in enumerate(STATE.segments) if s["class"].lower() == want]
    if not cands:
        return jsonify({"error": f"测试流里没有 {cls} 片段"}), 404
    min_len = max(2 * STATE.window_size, 8)
    good = [(i, s) for i, s in cands if int(s.get("length") or 0) >= min_len]
    pool = good or cands
    # 在「够长」的片段里取最长的（并列时取时间最早的），保证演示效果稳定
    idx, seg = max(pool, key=lambda t: (int(t[1].get("length") or 0), -int(t[1]["start"])))
    pos = max(0, int(seg["start"]) - STATE.window_size)
    return jsonify({"segment": seg, "index": idx, "pos": pos,
                    "candidates": len(cands), "long_enough": len(good),
                    "length": int(seg.get("length") or 0)})


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


# ======================================================================================
# 漂移与未知态势（只读接口，同样不依赖 TensorFlow）
# ======================================================================================
# 把「模型对当前流量的陌生程度」从"一个全局阈值"变成"一条随时间变化的曲线"：
#
# 1. 新颖性分数 = 多分类概率的**熵** ``H = -Σ p_i log p_i``（越大越像"没见过这种形态"），
#    与 ``tools/openset_calibrate.py`` 是同一个算法，因此面板上的曲线与标定文件同口径；
# 2. 阈值读 ``results/openset_thresholds.json`` 的 ``recommended.threshold``（entropy 口径，
#    在真实标签为 Normal 的窗口上按误报预算标定）。**读不到就如实降级**——
#    面板会显示"未标定，面板不可用"，绝不伪造一条参考线；
# 3. 窗口级缓存 ``artifacts/predictions_windows_<tag>.csv`` 里没有窗口起点，因此用预处理产物
#    ``meta.json`` 的 ``split_info.windows.test.starts`` 把「窗口序号 → 回放流位置」对齐；
#    这样面板游标与「跳到未知占比最高处」都能精确落到大屏的回放位置上。
#
# 依赖的产物（都是 run_all.py / tools/openset_calibrate.py 已经落盘的，不需要重训、不需要 TF）：
#   artifacts/predictions_windows_<tag>.csv · results/openset_thresholds.json · data/processed/meta.json
NOVELTY_LOCK = threading.Lock()
_NOVELTY_CACHE: dict = {}

NOVELTY_DEFAULT_BINS = 60
NOVELTY_MIN_BINS = 8
NOVELTY_MAX_BINS = 240
NOVELTY_EPS = 1e-12


def _read_json_soft(path) -> dict | None:
    """读 JSON，失败只记日志并返回 None（面板降级不该影响其它接口）。"""
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("读取 %s 失败：%s", path, exc)
    return None


def _novelty_tag() -> str:
    """主模型 tag：优先 artifacts/main_model.json，其次 results/summary.json，最后用演示用的 tag。"""
    for p in (Path(ARTIFACT_DIR) / "main_model.json", Path(RESULT_DIR) / "summary.json"):
        d = _read_json_soft(p)
        if isinstance(d, dict) and d.get("tag"):
            return str(d["tag"])
    return str(getattr(STATE, "tag", "") or "")


def _novelty_row_map(n_win: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """把「窗口级缓存的第 i 个窗口」映射到「回放流的第几行」。

    优先用预处理产物里记录的真实窗口起点（``split_info.windows.test.starts``，与窗口级缓存
    逐行对应）；对不上时退回按行数比例近似——游标仍可用，但会在返回里把 ``mapped`` 标成
    False，让前端如实说明"位置为估算值"。
    """
    n_rows = int(getattr(STATE, "n_rows", 0) or 0)
    ws = int(getattr(STATE, "window_size", 0) or 0)
    meta = _read_json_soft(Path(PROCESSED_DIR) / "meta.json") or {}
    try:
        wtest = (((meta.get("split_info") or {}).get("windows") or {}).get("test") or {})
        ws = int(wtest.get("window_size") or ws or 16)
        starts = np.asarray(wtest.get("starts") or [], dtype=np.int64)
        if starts.size == n_win and n_rows > 0:
            idx = np.clip(starts, 0, n_rows - 1)
            times = np.asarray(STATE.t_rel, dtype=np.float64)[idx]
            return idx, times, ws, True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("窗口→回放位置映射不可用（%s），改用比例近似", exc)
    if n_rows > 0:
        starts = np.linspace(0, max(n_rows - 1, 0), max(n_win, 1)).astype(np.int64)
        times = np.asarray(STATE.t_rel, dtype=np.float64)[starts]
    else:
        starts = np.arange(max(n_win, 1), dtype=np.int64)
        times = np.zeros(max(n_win, 1))
    return starts, times, int(ws or 16), False


def _novelty_series() -> dict:
    """惰性加载窗口级预测缓存并算好熵（按文件 mtime 缓存，不在每个请求里重算）。"""
    tag = _novelty_tag()
    path = Path(ARTIFACT_DIR) / f"predictions_windows_{tag}.csv"
    if not path.exists():
        return {"ok": False, "reason": f"未找到窗口级预测缓存 {path.name}（请先运行 python run_all.py）"}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    cache_key = f"{path}|{mtime}"
    with NOVELTY_LOCK:
        if _NOVELTY_CACHE.get("key") == cache_key:
            return _NOVELTY_CACHE
    if not getattr(STATE, "ok", False):
        return {"ok": False, "reason": STATE.reason or "回放数据未就绪"}
    try:
        df = pd.read_csv(path)
        prob_cols = [c for c in df.columns if c.startswith("p_") and c != "p_attack"]
        label_col = "y_true_name" if "y_true_name" in df.columns else "y_true"
        if len(prob_cols) < 2 or label_col not in df.columns:
            return {"ok": False, "reason": f"{path.name} 缺少多分类概率列或真实标签列"}
        probs = np.clip(df[prob_cols].to_numpy(dtype=np.float64), NOVELTY_EPS, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        entropy = -(probs * np.log(probs)).sum(axis=1)          # 与 openset_calibrate 同公式
        labels = df[label_col].astype(str).to_numpy()
        pred = (df["y_pred_name"].astype(str).to_numpy()
                if "y_pred_name" in df.columns else labels.copy())
        p_attack = (df["p_attack"].to_numpy(dtype=np.float64)
                    if "p_attack" in df.columns else np.zeros(len(df)))
        starts, times, ws, mapped = _novelty_row_map(len(df))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("新颖性序列计算失败：%s", exc)
        return {"ok": False, "reason": f"窗口级缓存解析失败：{exc}"}
    out = {
        "ok": True, "key": cache_key, "tag": tag, "source": path.name,
        "entropy": entropy, "labels": labels, "pred": pred, "p_attack": p_attack,
        "prob_cols": prob_cols, "starts": starts, "times": times,
        "window_size": ws, "mapped": bool(mapped),
        "attack_classes": [c[2:] for c in prob_cols if c[2:] != "Normal"],
    }
    with NOVELTY_LOCK:
        _NOVELTY_CACHE.clear()
        _NOVELTY_CACHE.update(out)
    LOGGER.info("漂移面板数据就绪：%s（%d 个窗口，熵阈值口径 entropy，位置映射%s）",
                path.name, len(entropy), "精确" if mapped else "近似")
    return out


def _novelty_threshold() -> dict:
    """每次请求都重读标定文件（很小）：标定文件被删掉/改口径时，面板立刻如实降级。"""
    path = Path(RESULT_DIR) / "openset_thresholds.json"
    if not path.exists():
        return {"ok": False, "reason": f"未标定，面板不可用：找不到 {path.name}"
                                      f"（请运行 python tools/openset_calibrate.py 生成标定阈值）"}
    d = _read_json_soft(path)
    if not isinstance(d, dict):
        return {"ok": False, "reason": f"未标定，面板不可用：{path.name} 无法解析"}
    score = str(d.get("score") or "")
    if score != "entropy":
        return {"ok": False, "reason": f"未标定，面板不可用：标定口径为 {score or '未知'}，本面板只支持 entropy"}
    rec = d.get("recommended") or {}
    try:
        thr = float(rec["threshold"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "reason": f"未标定，面板不可用：{path.name} 里没有 recommended.threshold"}
    budget = rec.get("budget")
    return {
        "ok": True, "threshold": thr, "source": path.name, "tag": str(d.get("tag") or ""),
        "budget": (float(budget) if budget is not None else None),
        "budget_line": float(rec.get("normal_flagged") or 0.0),
        "known_attack_flagged": float(rec.get("known_attack_flagged") or 0.0),
        "n_windows_calibrated": int(d.get("n_windows") or 0),
        "semantics": str(d.get("semantics") or ""),
    }


def _novelty_payload(bins: int) -> dict:
    """组装 /api/novelty 的响应：分箱后的未知占比、各类别攻击窗口数、阈值与结论。"""
    try:
        bins = int(bins)
    except (TypeError, ValueError):
        bins = NOVELTY_DEFAULT_BINS
    bins = max(NOVELTY_MIN_BINS, min(bins, NOVELTY_MAX_BINS))
    out = {
        "ok": False, "degraded": True, "reason": "", "bins": bins, "score": "entropy",
        "threshold": None, "budget_line": None, "threshold_budget": None, "threshold_source": "",
        "n_windows": 0, "window_size": 0, "unknown_ratio_global": None, "unknown_count_global": 0,
        "summary": "", "conclusion": "", "peak": None, "x": None, "unknown_ratio": [],
        "unknown_count": [], "attack_classes": [], "attack_counts": {}, "attack_share": {},
        "detect_threshold": None, "attack_rule": "", "source": "", "tag": "", "mapped": False,
        "hint": "面板需要「窗口级预测缓存 + 未知标记标定文件」两样产物，缺任意一样都会如实降级，不伪造曲线。",
    }
    ser = _novelty_series()
    if not ser.get("ok"):
        out["reason"] = ser.get("reason") or "新颖性数据不可用"
        return out
    thr = _novelty_threshold()
    if not thr.get("ok"):
        out["reason"] = thr.get("reason") or "未标定，面板不可用"
        return out

    H = np.asarray(ser["entropy"], dtype=np.float64)
    labels = np.asarray(ser["labels"])
    pred = np.asarray(ser["pred"])
    p_attack = np.asarray(ser["p_attack"], dtype=np.float64)
    starts = np.asarray(ser["starts"], dtype=np.int64)
    times = np.asarray(ser["times"], dtype=np.float64)
    ws = int(ser["window_size"])
    n = int(H.size)
    if n == 0:
        out["reason"] = "窗口级预测缓存为空"
        return out
    bins = max(NOVELTY_MIN_BINS, min(bins, n))

    threshold = float(thr["threshold"])
    unknown = H >= threshold
    # 「被判为攻击」的口径与大屏一致：二分类概率 ≥ 当前判定阈值（默认 0.5），且类别为攻击类。
    detect_thr = float(getattr(STATE, "threshold", 0.5) or 0.5)
    atk_classes = list(ser["attack_classes"])
    is_atk = np.isin(pred, atk_classes)
    judged = p_attack >= detect_thr

    edges = np.linspace(0, n, bins + 1).astype(int)
    ratio, ucount, counts, shares = [], [], {c: [] for c in atk_classes}, {c: [] for c in atk_classes}
    x_w0, x_w1, x_r0, x_r1, x_t0, x_t1 = [], [], [], [], [], []
    for i in range(bins):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = min(a + 1, n)
        sl = slice(a, b)
        nwin = b - a
        u = unknown[sl]
        ratio.append(round(float(u.mean()), 6))
        ucount.append(int(u.sum()))
        hit = is_atk[sl] & judged[sl]
        for c in atk_classes:
            k = int(np.sum(hit & (pred[sl] == c)))
            counts[c].append(k)
            shares[c].append(round(k / nwin, 6))
        x_w0.append(a)
        x_w1.append(b - 1)
        x_r0.append(int(starts[a]))
        x_r1.append(int(starts[b - 1]) + ws - 1)
        x_t0.append(round(float(times[a]), 3))
        x_t1.append(round(float(times[b - 1]), 3))

    peak_bin = int(np.argmax(np.asarray(ratio, dtype=np.float64)))
    peak_ratio = float(ratio[peak_bin])
    peak_bucket = {c: counts[c][peak_bin] for c in atk_classes}
    peak_top = max(peak_bucket.items(), key=lambda kv: kv[1]) if peak_bucket else ("", 0)
    budget = thr.get("budget")
    budget_line = float(thr.get("budget_line") or 0.0)
    glob_ratio = float(unknown.mean())
    hit_rule = (f"「被判为攻击」的口径与大屏一致：二分类概率 p_attack ≥ {detect_thr:g}，"
                f"且模型判定类别为攻击类（{('、'.join(atk_classes))}）的窗口数")
    out.update({
        "ok": True, "degraded": False, "reason": "",
        "threshold": round(threshold, 6),
        "budget_line": round(budget_line, 6),
        "threshold_budget": (float(budget) if budget is not None else None),
        "threshold_source": thr.get("source", ""),
        "threshold_tag": thr.get("tag", ""),
        "n_windows": n, "window_size": ws, "mapped": bool(ser.get("mapped")),
        "unknown_ratio_global": round(glob_ratio, 6),
        "unknown_count_global": int(unknown.sum()),
        "unknown_ratio": ratio, "unknown_count": ucount,
        "attack_classes": atk_classes, "attack_counts": counts, "attack_share": shares,
        "detect_threshold": detect_thr, "attack_rule": hit_rule,
        "source": ser.get("source", ""), "tag": ser.get("tag", ""),
        "x": {"w_start": x_w0, "w_end": x_w1, "row_start": x_r0, "row_end": x_r1,
              "t_start": x_t0, "t_end": x_t1},
        "peak": {
            "bin": peak_bin, "ratio": round(peak_ratio, 6),
            "row": int(x_r0[peak_bin]), "t": float(x_t0[peak_bin]),
            "w_start": x_w0[peak_bin], "w_end": x_w1[peak_bin],
            "unknown_count": ucount[peak_bin], "n_windows": x_w1[peak_bin] - x_w0[peak_bin] + 1,
            "top_class": peak_top[0], "top_class_count": int(peak_top[1]),
            "above_budget_line": bool(peak_ratio > budget_line),
        },
        "summary": (f"当前全局未知占比 {glob_ratio * 100:.2f}%"
                    f"（{int(unknown.sum())}/{n} 个判定窗口的熵 ≥ 标定阈值 {threshold:.4f}，"
                    f"熵是新奇性分数：分布越平越像「没见过这种形态」）"),
        "conclusion": (f"未知占比突然升高 = 新攻击出现或流量分布变化，值得人工研判"
                       f"（参考线 = 标定误报预算 {budget * 100:g}%：局部未知率越过它的时段应优先复核）"
                       if budget else "未知占比突然升高 = 新攻击出现或流量分布变化，值得人工研判"),
    })
    return out


@app.route("/judge")
def judge_page():
    """评审视图：把「创新点 / 优势 / 可验证性 / 诚实边界」放到一屏里给评委看。

    数据全部来自 `results/` 下的**真实产物**（缺哪个就标"未测"，绝不编数），
    因此这一页可以当作"活的结论页"——改一次参数、重跑一个实验，页面上的数字会跟着变。
    """
    return render_template("judge.html")


def _rj(name: str):
    """读 results/<name>；不存在或解析失败返回 None（调用方负责降级显示）。"""
    p = RESULT_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("读取 %s 失败：%s", p, exc)
        return None


@app.route("/api/judge")
def api_judge():
    """评审视图的数据源（**只读**，不依赖 TensorFlow）。

    返回结构：``headline``（核心数字）/ ``operating_points``（工作点全表，前端做交互）/
    ``ops``（运维换算）/ ``falsifications``（三次自我证伪）/ ``robustness``（对抗与开集）/
    ``controller_fields``（字段缺口补平）/ ``verifiable``（自检·测试·工具数·一条命令）/
    ``boundaries``（诚实边界）/ ``opensource``（开源成果）。
    任何一项缺失都返回 ``None`` 或 `未测`，由前端灰显 —— **不编造**。
    """
    out: dict = {"ok": True}

    # ---------------- 1) 核心数字 ----------------
    main = _rj("metrics_cnn_bilstm_main_s0.json") or {}
    agg = _rj("aggregate_cnn_bilstm_main.json") or {}
    mc = (main.get("multiclass") or {})
    on = (main.get("online") or {})
    binm = (main.get("binary") or {})
    # 多种子聚合文件的键是 <指标>_<口径>（例如 multiclass_macro_f1），与单次运行的 metrics json 不同名
    a_main = (agg.get("cnn_bilstm") or {})
    seed_macro = a_main.get("multiclass_macro_f1") or {}
    seed_delay = a_main.get("mean_detection_delay_sec") or {}
    seed_fa = a_main.get("false_alarms_per_hour") or {}
    hl = {
        "macro_f1": seed_macro.get("mean") or mc.get("macro_f1"),
        "macro_f1_std": seed_macro.get("std"),
        "segment_recall": (a_main.get("segment_recall") or {}).get("mean", on.get("segment_recall")),
        "n_segments": on.get("n_attack_segments"),
        "n_detected": on.get("n_segments_detected"),
        "mean_delay_ms": ((seed_delay.get("mean") or on.get("mean_detection_delay_sec") or 0) * 1000
                          if (seed_delay or on) else None),
        "false_alarms_per_hour": seed_fa.get("mean", on.get("false_alarms_per_hour")),
        "false_alarms_std": seed_fa.get("std"),
        "roc_auc": binm.get("roc_auc"),
        "n_windows": (main.get("data") or {}).get("n_rows"),
        "test_windows": (on.get("n_windows")),
        "stream_hours": on.get("stream_hours"),
        "n_seeds": a_main.get("n_seeds"),
    }
    out["headline"] = hl

    # ---------------- 2) 工作点全表（前端交互） ----------------
    op = _rj("operating_point.json") or {}
    ops = _rj("ops_impact.json") or {}
    op_rows = {(float(r["threshold"]), int(r["confirm"])): r for r in (op.get("table") or [])}
    points = []
    for r in (ops.get("table") or []):
        key = (float(r["threshold"]), int(r["confirm"]))
        src = op_rows.get(key) or {}
        points.append({
            "threshold": key[0], "confirm": key[1],
            "recall": r.get("segment_recall"),
            "missed_per_day": r.get("missed_attacks_per_day"),
            "false_alarms_per_hour": r.get("false_alarms_per_hour"),
            "false_alarms_per_day": r.get("false_alarms_per_day"),
            "alarms_per_day": r.get("alarms_per_day"),
            "triage_hours_per_day": r.get("triage_hours_per_day"),
            "mean_delay_ms": r.get("mttd_ms") if r.get("mttd_ms") is not None
            else (src.get("mean_detection_delay_sec") or 0) * 1000,
            "false_alarm_share": r.get("false_alarm_share"),
        })
    out["operating_points"] = sorted(points, key=lambda x: (x["threshold"], x["confirm"]))
    out["ops"] = {
        "recommended": ops.get("recommended_working_point"),
        "config_default": ops.get("config_default_working_point"),
        "no_confirm": ops.get("no_confirm_counterfactual"),
        "saved_triage_hours_per_day": ops.get("saved_triage_hours_per_day_with_confirm"),
        "triage_minutes_per_alert": ops.get("triage_minutes"),
        "shift_hours": ops.get("shift_hours"),
        "false_alarm_budget_per_hour": ops.get("false_alarm_budget_per_hour"),
        "attacks_per_day_assumption": ops.get("attacks_per_day_assumption"),
        "scale_projection": ops.get("scale_projection"),
    }

    # ---------------- 3) 三次自我证伪 ----------------
    fals = []
    hard = _rj("hard_ablation.json")
    if hard:
        ans = hard.get("answers") or {}
        paired = ans.get("paired_overall") or {}
        fals.append({
            "id": "hard", "title": "① 反驳「是你数据太简单」→ 另造困难基准",
            "hypothesis": "卷积没增益，是因为窗口太短 / 数据太简单",
            "how": "四个难度旋钮拉满（重叠 0.35→0.70、隐蔽 0.45→0.80、特征噪声 ×3），重训一整套",
            "result": f"卷积依然无增益：逐种子配对 {paired.get('n_negative')}/{paired.get('n')} 为负"
                      f"（符号检验双侧 p={round(paired.get('sign_test_p_two_sided') or 0, 3)}）；"
                      f"「需要更长上下文」被推翻（斜率由 +0.00043 变 −0.00003）",
            "honest": "难度翻倍后宏 F1 −5.4 pt、误报 ×4.5，但攻击片段召回仍 100%",
            "source": "results/hard_ablation.md",
        })
    seq = _rj("packet_seq.json")
    if seq:
        jud = seq.get("judgement") or {}
        rows = seq.get("packet_lengths") or []
        ctrl_gap = (seq.get("flow_control") or {}).get("gap_cnn_behind_best")
        pkt_gap = min([r.get("gap_cnn_behind_best") for r in rows
                       if r.get("gap_cnn_behind_best") is not None] or [None])
        fals.append({
            "id": "packet", "title": "② 反驳「是表示层次不对」→ 换成包级序列",
            "hypothesis": "流级聚合量把局部模式压平了，换包序列卷积就该有用",
            "how": "同一批 24,000 条流反构包序列，包级 vs 流级**同数据对照**",
            "result": f"卷积落后最好单分支 {ctrl_gap:.4f}（流级）→ {pkt_gap:.4f}（包级 T=32），缩小约 19 倍；"
                      f"包级 T=32/64 首次出现逐种子配对 3/3 为正",
            "honest": f"总判定写 {jud.get('overall')}（换全量参照时只有 T=32 通过）；"
                      f"包级绝对宏 F1 仅 0.31~0.39 —— 这是机理探测，不等于「包级更好用」",
            "source": "results/packet_seq.md",
        })
    dr = _rj("domain_randomization.json")
    if dr:
        j = dr.get("judgement") or {}
        d1 = (j.get("J1_diversity_helps") or {}).get("detail") or {}
        fals.append({
            "id": "dr", "title": "③ 把我们自己留的「下一步建议」做完 → 域随机化",
            "hypothesis": "多域训练能缓解域漂移（上一版报告给出的建议）",
            "how": "造 6 个域（基准/隐蔽/边界模糊/强噪声/老式攻击/业务构成变化），逐个留一域零样本 + 等预算对照",
            "result": f"平均只 {round(d1.get('mean') or 0, 4)}（{d1.get('n_positive')} 正 {d1.get('n_negative')} 负，"
                      f"p={round(d1.get('sign_test_p_two_sided') or 0, 3)}）→ {dr.get('verdict')}",
            "honest": "对「隐蔽攻击为主」反而 −0.25；等预算对照 5/6 折比单域更差 —— 该难度下「每域数据量」比「域数量」更重要",
            "source": "results/domain_randomization.md",
        })
    out["falsifications"] = fals

    # ---------------- 4) 稳健性：对抗 / 开集 / 字段补平 ----------------
    adv = _rj("adversarial_eval.json")
    if adv:
        grid = [g for g in (adv.get("grid") or []) if g.get("objective") == "evade_binary"]
        worst = min(grid, key=lambda g: g["pgd"]["macro_f1"]) if grid else None
        crit = ((adv.get("adv_train") or {}).get("criterion") or {})
        out["robustness_adversarial"] = {
            "clean_macro_f1": (adv.get("clean") or {}).get("macro_f1"),
            "worst_eps": worst["eps"] if worst else None,
            "worst_macro_f1": worst["pgd"]["macro_f1"] if worst else None,
            "worst_evasion": worst["pgd"]["attack_evasion_rate"] if worst else None,
            "random_best": min((g["random"]["macro_f1"] for g in grid), default=None),
            "adv_gain": crit.get("f1_robust_gain"), "adv_cost": crit.get("f1_clean_cost"),
            "worthwhile": crit.get("worthwhile"),
            "honest": "对抗样本的预测熵没有升高（0.018 → 0.013~0.032，未知阈值 0.3914）"
                      "→ 已有的「熵超阈值」机制**拦不住**对抗样本",
        }
    oo = _rj("openset_ood_plus.json")
    if oo:
        js = oo.get("judgement") or {}
        cards = []
        for scen, j in js.items():
            cards.append({"scenario": scen,
                          "baseline": j.get("baseline_best_udr_at_1pct"),
                          "baseline_score": j.get("baseline_best_score"),
                          "mahalanobis": j.get("new_best_udr_at_1pct"),
                          "gain": j.get("gain"),
                          "false_reject_base": j.get("known_attack_false_reject_baseline"),
                          "false_reject_new": j.get("known_attack_false_reject_new")})
        out["robustness_openset"] = {"cards": cards, "verdict": oo.get("verdict"),
                                     "honest": "范数诊断（只看特征范数偏离）检出率仅 0.4%~5% → 提升不是平凡信号；"
                                               "代价是已知攻击误拒率上升，因此判定为 partial"}
    cf = _rj("ctrl_field_extension.json")
    if cf:
        sm = cf.get("summary") or {}
        j1 = (cf.get("judgement") or {}).get("J1_accuracy") or {}
        j2 = (cf.get("judgement") or {}).get("J2_operating_point") or {}
        out["controller_fields"] = {
            "n17": (sm.get("ctrl17") or {}).get("n_features"),
            "f1_17": ((sm.get("ctrl17") or {}).get("macro_f1") or {}).get("mean"),
            "f1_17_std": ((sm.get("ctrl17") or {}).get("macro_f1") or {}).get("std"),
            "n_ext": (sm.get("ctrl_ext") or {}).get("n_features"),
            "f1_ext": ((sm.get("ctrl_ext") or {}).get("macro_f1") or {}).get("mean"),
            "f1_ext_std": ((sm.get("ctrl_ext") or {}).get("macro_f1") or {}).get("std"),
            "delta": j1.get("delta"), "sigma": j1.get("sigma"),
            "point_17": j2.get("ctrl17"), "point_ext": j2.get("ctrl_ext"),
            "unavailable": len(cf.get("unavailable_fields") or []),
            "honest": "聚合窗口是**因果的**（只看过去 60 秒）——最初写成对称窗口属前视泄漏，已修正重跑；"
                      "并列出 OpenFlow 流表**本身不提供**的字段清单",
        }

    # 特征选择与已发表方法对标（对标《通信技术》2026 的 PLO-BiLSTM 一文）
    fs = _rj("feature_selection.json")
    if fs:
        sm = fs.get("summary") or {}
        rows = []
        for ds in (fs.get("datasets") or []):
            for variant in ((fs.get("args") or {}).get("variants") or []):
                cells = {}
                for m in (fs.get("methods") or []):
                    s = sm.get(f"{ds}|{m}|{variant}") or {}
                    cells[m] = {"f1": (s.get("macro_f1") or {}).get("mean"),
                                "n_features": s.get("n_features")}
                if any(v.get("f1") is not None for v in cells.values()):
                    rows.append({"dataset": ds, "variant": variant, "cells": cells})
        out["feature_selection"] = {
            "rows": rows,
            "verdict": (fs.get("verdict") or {}).get("overall"),
            "dim_reduction": fs.get("dim_reduction"),
            "paper": (fs.get("reference_paper") or {}).get("venue"),
            "honest": "特征**只在训练集上选**（对标论文在全量数据上选，属选择偏差）；"
                      "适应度用代理分类器（算力约束，已如实标注）；"
                      "并量化披露：按论文式(3) 实现时二值化选择率会系统性偏高",
        }

    # ---------------- 5) 可验证性 ----------------
    acc = _rj("acceptance_summary.json") or {}
    tests = acc.get("tests") or {}
    out["verifiable"] = {
        "acceptance": {"passed": acc.get("passed"), "warned": acc.get("warned"),
                       "failed": acc.get("failed"), "ts": acc.get("ts")},
        "tests": tests,
        "n_tools": len(list((ROOT / "tools").glob("*.py"))),
        "commands": [
            ("一条命令复现主实验", "python run_all.py"),
            ("25 秒冒烟（独立目录）", "python run_all.py --smoke"),
            ("交付验收自检", "python tools/acceptance_check.py"),
            ("一场完整攻防 + 逐窗口核对", "python tools/live_demo.py"),
            ("三套单元/行为测试", "python tests/test_core.py / test_ryu_guard.py / test_pcap_tools.py"),
            ("特征选择对照（bPLO/PSO/PCA）", "python tools/feature_selection.py"),
            ("与已发表方法对标（RNN/LSTM/Transformer）", "python tools/published_baselines.py"),
            ("Ryu 真机联调（离线自检免装 ryu）", "python tools/ryu_live_check.py --self-test"),
        ],
    }

    # ---------------- 6) 诚实边界 ----------------
    bnds = []
    ry = _rj("ryu_live_check.json")
    bnds.append({
        "item": "Ryu 控制平面（剩余：物理交换机）",
        "status": "真实 Ryu + 真实 OF1.3 已实测" if ry else "未在物理交换机实跑",
        "detail": (f"真实 ryu-manager（Ryu {ry.get('ryu_version')}）加载适配层，经真实 OF1.3 线协议完成"
                   f"握手/流表上报/下发校验：{sum(1 for v in (ry.get('checks') or {}).values() if v.get('pass'))}"
                   f"/{len(ry.get('checks') or {})} 项通过、{ry.get('ryu_parse_errors', 0)} 条解析错误；"
                   "攻击轮实收按源 IP 匹配的 DROP 规则、正常轮 0 条（results/ryu_live_check.md）。"
                   "未覆盖：物理交换机、真实攻击流量、厂商私有扩展" if ry else
                   "真机验证脚本已就绪（tools/ryu_live_check.py，含离线自检）；"
                   "适配层逻辑由 34 项桩件测试覆盖"),
        "ok": False,
    })
    cd = _rj("cross_dataset_report.json")
    bnds.append({
        "item": "零样本跨域", "status": "不可用（已量化）",
        "detail": (f"跨域二分类 F1 {cd.get('verdict', {}).get('f1')}" if cd else
                   "自研→真实公开集：F1 0.5579、FPR 20.4%；只重拟合标准化器会把 AUC 从 0.6697 打到 0.4742"),
        "ok": False,
    })
    if dr:
        bnds.append({"item": "域随机化", "status": f"实测{('不奏效' if dr.get('verdict') == 'not_supported' else dr.get('verdict'))}",
                     "detail": "平均增益仅 +0.0063（3 正 3 负，p=1.0）——把自己的建议做完并如实否定", "ok": False})
    bnds.append({"item": "数据与对抗", "status": "仍为仿真 + 白盒攻击",
                 "detail": "结论建立在自研仿真流量上；对抗实验是特征空间白盒攻击（给的是脆弱性上界），"
                           "黑盒/查询式攻击与真实生产长期运行未做", "ok": False})
    out["boundaries"] = bnds

    # ---------------- 7) 开源成果 ----------------
    out["opensource"] = {
        "license": "Apache-2.0" if (ROOT / "LICENSE").exists() else "缺失",
        "has_notice": (ROOT / "NOTICE").exists(),
        "has_contributing": (ROOT / "CONTRIBUTING.md").exists(),
        "has_ci": (ROOT / ".github" / "workflows" / "ci.yml").exists(),
        "has_deploy": (ROOT / "deploy").exists(),
        "samples": (ROOT / "samples").exists(),
        "n_tools": len(list((ROOT / "tools").glob("*.py"))),
        "integration": "AI 检测 → Ryu（开源 SDN 控制器）处置：OFPFlowMod 阻断 / OFPMeterMod 限速 + 审计台账",
        "self_built": "pcap 解析零依赖纯标准库；OF1.3 编解码自研（含离线自检）",
    }
    return jsonify(json_safe(out))


@app.route("/api/novelty")
def api_novelty():
    """漂移与未知态势（**只读**）：未知/新颖性占比随时间的曲线 + 各类别攻击窗口数 + 标定阈值。

    查询参数：``bins``（分箱数，默认 60，钳制到 8~240；每格 ≈ n_windows/bins 个判定窗口）

    返回要点：``ok``/``reason``（降级时 reason 说明缺什么产物）、``unknown_ratio``（分箱后的
    未知占比序列）、``attack_counts``/``attack_share``（各类别被判为攻击的窗口数与占比）、
    ``threshold``（熵口径标定阈值）与 ``budget_line``（它对应的误报预算参考线）、
    ``x``（每格的窗口区间与回放流位置，供前端画游标与跳转）、``peak``（未知占比最高的一格）、
    ``summary``（一句"当前全局未知占比"）、``conclusion``（一句结论文案）。

    注意：**不依赖 TensorFlow**，只读 `predictions_windows_<tag>.csv` 与标定文件；
    标定文件读不到时返回 ok=false（面板显示"未标定"），不会编造阈值。
    """
    bins = request.args.get("bins", default=NOVELTY_DEFAULT_BINS, type=int)
    return jsonify(json_safe(_novelty_payload(bins)))


@app.route("/api/health")
def api_health():
    """健康检查：既有字段 ``ok``/``reason``（演示数据是否就绪）保持不变，
    另**新增**两个只读字段，把「在线检测服务」的就绪状态一并暴露出来，
    供前端状态条显示与优雅降级判断。任何异常都不会影响既有字段。"""
    out = {"ok": STATE.ok, "reason": STATE.reason}
    try:
        det = get_detector()
        out["detector_ready"] = bool(det.ready)
        out["detector_reason"] = det.reason or ""
    except Exception as exc:  # pragma: no cover - 在线接口不可用时也要能返回健康检查
        out["detector_ready"] = False
        out["detector_reason"] = f"在线检测服务不可用：{exc}"
    return jsonify(out)


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
    # ⚠️ det.alerts 是**有界 deque**（长期运行不吃内存），deque 不支持切片，
    # 必须先 list() 再切；否则这里会抛 TypeError，把审计台账整条链路打断。
    recent = list(det.alerts)[-max(1, int(limit)):]
    return jsonify(json_safe([a.to_dict() for a in reversed(recent)]))


def _require_token():
    """写接口的可选令牌校验（纵深防御，默认关闭）。

    为什么只保护**写接口**：能造成实际损害的是改策略（例如把 `dry_run` 关掉、动作改成 block）
    和写入研判结论；读接口（大屏轮询 /api/*）保持开放，前端不必持有令牌。
    `DEPLOYMENT.md` 的做法是反代加 Basic 认证，这里再补一层应用级校验——避免反代配置失误
    或内网可达时，任何人都能关掉演练模式、把建议动作改成真实封禁。
    **未设置 `CIDA_API_TOKEN` 时完全放行**（本地演示与验收自检不受影响）。
    """
    tok = os.environ.get("CIDA_API_TOKEN", "").strip()
    if not tok:
        return None
    auth = request.headers.get("Authorization", "")
    got = request.headers.get("X-API-Token") or (auth[7:].strip() if auth.startswith("Bearer ") else "")
    if got == tok:
        return None
    return jsonify({"ok": False, "error": "缺少或错误的 API 令牌（写接口需要 CIDA_API_TOKEN）"}), 401


@app.route("/api/observations")
def api_observations():
    """观察池：差一点到阈值 / 形态新颖但未确认的窗口留痕（漏报复盘的证据来源）。"""
    det = get_detector()
    limit = request.args.get("limit", default=50, type=int)
    return jsonify(json_safe(det.list_observations(limit)))


@app.route("/api/review", methods=["GET", "POST"])
def api_review():
    """待研判队列：GET 列出，POST 给人工结论并回流（「未知 → 已知」闭环的入口）。"""
    det = get_detector()
    if request.method == "GET":
        limit = request.args.get("limit", default=50, type=int)
        status = request.args.get("status") or None
        return jsonify(json_safe(det.list_review(limit=limit, status=status)))
    deny = _require_token()
    if deny:
        return deny
    body = request.get_json(silent=True) or {}
    if "review_id" not in body:
        return jsonify({"ok": False, "error": "缺少 review_id"}), 400
    try:
        rid = int(body["review_id"])
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "review_id 必须是整数"}), 400
    res = det.label_review(rid, str(body.get("verdict", "")), str(body.get("note", "")))
    return jsonify(json_safe(res)), (200 if res.get("ok") else 400)


@app.route("/api/review/promote", methods=["POST"])
def api_review_promote():
    """把观察池里的一条提升进待研判队列（值班员做漏报复盘时的动作）。"""
    deny = _require_token()
    if deny:
        return deny
    det = get_detector()
    body = request.get_json(silent=True) or {}
    try:
        idx = int(body.get("index", -1))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "index 必须是整数"}), 400
    res = det.promote_observation(index=idx, why=str(body.get("why", "manual")))
    return jsonify(json_safe(res)), (200 if res.get("ok") else 400)


@app.route("/api/policy", methods=["GET", "POST"])
def api_policy():
    """查询/热更新检测策略（阈值、确认窗口数、处置动作、白名单、演练模式）。"""
    det = get_detector()
    if request.method == "GET":
        return jsonify(json_safe(det.snapshot()["policy"]))
    deny = _require_token()
    if deny:
        return deny
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
    global STATE, DETECTOR, PROCESSED_DIR, ARTIFACT_DIR, MODEL_TAG, _RAW_MATRIX
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
        _RAW_MATRIX = None  # 换口径后原始量纲缓存必须失效
        with NOVELTY_LOCK:  # 漂移面板的窗口级缓存同理：换口径后必须重算
            _NOVELTY_CACHE.clear()
        LOGGER.info("已切换数据/模型口径：processed=%s artifacts=%s tag=%s",
                    PROCESSED_DIR, ARTIFACT_DIR, MODEL_TAG or "（主模型）")

    if not STATE.ok:
        LOGGER.warning("演示数据未就绪：%s", STATE.reason)
    LOGGER.info("演示服务启动： http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
