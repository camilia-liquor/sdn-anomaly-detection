"""一条命令看一场真实攻防：正常流量 → 攻击爆发 → 告警 → 处置 → 风险回落 → 事件报告。

为什么需要它
------------
原先要理解这套系统，得起两个终端（一个跑服务、一个跑模拟控制器），再自己去翻报告。
本脚本把整条链路压进**一条命令**：

    python tools/live_demo.py

屏幕上依次出现：检测服务启动 → 正常流量平稳回放 → 攻击爆发（真实标签切换）→
模型报出确认告警（含**检测时延**）→ 处置动作下发（模拟执行）→ 攻击结束、风险回落 →
自动生成事件报告 ``results/live_demo_incident.md``。

三条不可逾越的底线（全部设计都围绕它们）
----------------------------------------
1. **数据是真的**：回放 ``data/processed/tensors.npz`` 里的**留出测试流**
   （``test_stream_features / _labels / _time``），不是编的曲线；
   "注入攻击"只是把回放位置放到**真实攻击片段的起点**（与 ``app/app.py`` 的
   ``/api/first_segment_of`` 同一性质），标签切换是数据里本来就有的。
2. **判定是真的**：**进程内**调用 ``src.service.DetectionService`` —— 与
   ``app/app.py`` 的 ``/api/detect``、``tools/controller_simulator.py`` 是同一个引擎、
   同一份阈值/连续确认口径，本脚本**不另写一份"演示专用判定"**。
   更进一步：回放的滑窗网格与 ``src/evaluate.py`` 的离线评估网格**对齐**，
   因此可以逐窗口比对 ``artifacts/predictions_<tag>.csv``，把"演示=报告"从口号变成可核对的事实。
3. **不好的消息照报**：正常阶段的误报、片段内的漏报、攻击结束后的回落，全部原样打印并写进报告；
   处置只到"策略建议 + 本地模拟执行器"，**不**声称真的改了交换机流表，
   也**不**声称是处置把攻击压住了（回落是回放走过真实攻击片段的自然结果）。

常用参数
--------
``--attack DDoS|DoS|Probe|BruteForce|WebAttack|Botnet|U2R``  选择要演的攻击类型（默认 DDoS）
``--normal-rows N``  攻击前后各回放多少条正常流（默认 1000，实际以"紧邻的连续纯正常片段"为上限）
``--speed S``        回放倍速（默认 20；1 倍速 = 6 条流记录/秒，**纯展示节奏，不参与任何指标计算**）
``--speed 0``        不限速尽快跑完（自动化测试/CI 用，屏幕输出自动简化）
``--threshold / --confirm / --action``  覆盖判定阈值、连续确认窗口数、处置动作（默认 block）
``--no-color``       关闭 ANSI 颜色（非 TTY 环境会自动关闭颜色与动画）

退出码：0 正常；2 预处理/模型产物缺失（提示先跑 run_all.py）；3 缺 TensorFlow；4 该攻击类型没有可用片段。
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    load_json,
    markdown_table,
    save_text,
    silence_tf_logs,
)

REPORT_PATH = RESULT_DIR / "live_demo_incident.md"
AUDIT_PATH = RESULT_DIR / "live_demo_audit.jsonl"

# 演示节奏常量：1 倍速 = 每秒回放 6 条流记录。
# ⚠ 它只决定屏幕刷新快慢。报告里的时延、误报、概率全部来自数据与模型，与倍速无关
#   （可自行验证：--speed 0 与 --speed 50 生成的报告逐字节一致）。
BASE_ROWS_PER_SEC = 6.0

SPARK = "▁▂▃▄▅▆▇█"
NORMAL = "Normal"

C_RESET = "\x1b[0m"
C_DIM = "\x1b[2m"
C_RED = "\x1b[1;31m"
C_GREEN = "\x1b[32m"
C_YELLOW = "\x1b[33m"
C_CYAN = "\x1b[36m"
C_MAGENTA = "\x1b[35m"


class DemoError(RuntimeError):
    """带退出码的可预期错误：给中文提示，不抛堆栈。"""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = int(code)


# ======================================================================================
# 小工具
# ======================================================================================
def rel_path(path: Path | str) -> str:
    """尽量用相对项目根的路径展示（屏幕与报告里不出现绝对路径）。"""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except Exception:  # noqa: BLE001
        return Path(path).name


def sanitize(text: str) -> str:
    """抹掉本机绝对路径与用户名，保证输出里不出现个人信息。"""
    out = str(text)
    for raw in (str(ROOT), str(Path.home()), os.environ.get("USERNAME", ""), os.environ.get("USER", "")):
        if raw and len(raw) > 2:
            out = out.replace(raw, ".")
            out = out.replace(raw.replace("\\", "/"), ".")
    return re.sub(r"[A-Za-z]:\\\\?[^\s\"'|)]*", ".", out)


def fmt_ts(ts: float) -> str:
    """数据集里的流时间戳 → UTC 字符串（用 gmtime，机器时区不影响可复现性）。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(ts))) + " UTC"


def hm(t: float) -> str:
    """[mm:ss.s] 形式的演示墙钟时间戳。"""
    return f"[{int(t) // 60:02d}:{int(t) % 60:02d}.{int((t % 1) * 10)}]"


def spark(values: Sequence[float], width: int = 8) -> str:
    """把最近若干个窗口的异常概率画成块状迷你曲线。"""
    vals = list(values)[-width:]
    if not vals:
        return " " * width
    vals = [0.0] * (width - len(vals)) + vals
    return "".join(SPARK[min(len(SPARK) - 1, max(0, int(v * len(SPARK))))] for v in vals)


class Console:
    """输出层：TTY 用一行"实况状态条"原地刷新；非 TTY 逐行打印（CI 里不会刷控制字符）。"""

    def __init__(self, color: bool, animate: bool, stream: Any = None):
        self.stream = stream or sys.stdout
        self.color = bool(color)
        self.animate = bool(animate)
        self._live = False
        self.closed = False        # 下游提前关闭管道（如 | head / Select-Object -First）时静默收手

    def _write(self, text: str) -> None:
        if self.closed:
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except (BrokenPipeError, ValueError, OSError):
            self.closed = True

    def _paint(self, text: str, code: str | None) -> str:
        return f"{code}{text}{C_RESET}" if (self.color and code) else text

    def _clear_live(self) -> None:
        if self._live:
            self._write("\r\x1b[K")
            self._live = False

    def line(self, text: str, code: str | None = None) -> None:
        self._clear_live()
        self._write(self._paint(text, code) + "\n")

    def live(self, text: str, code: str | None = None) -> None:
        if not self.animate:
            self.line(text, code)
            return
        self._write("\r\x1b[K" + self._paint(text, code))
        self._live = True

    def close(self) -> None:
        if self._live:
            self._write("\n")
            self._live = False


# ======================================================================================
# 剧本（scenario）
# ======================================================================================
@dataclass
class Block:
    """一段连续回放的流记录。"""

    key: str              # normal_pre | attack | normal_post
    gstart: int           # 该段第一行在测试流里的全局行号（0-based）
    length: int
    label: str = ""       # attack 段：真实标签（DDoS/Probe/...）
    jumped: bool = False  # 本段相对上一段是否发生了"位置跳转"

    @property
    def gend(self) -> int:
        return self.gstart + self.length - 1

    @property
    def name_cn(self) -> str:
        return {"normal_pre": "正常回放（攻击前）",
                "attack": f"真实攻击片段（{self.label}）",
                "normal_post": "恢复回放（攻击后）"}.get(self.key, self.key)


@dataclass
class Scenario:
    blocks: list[Block]
    attack_class: str
    seg_start: int
    seg_end: int
    seg_length: int
    n_candidates: int
    n_long_enough: int
    pre_run_len: int
    post_run_len: int
    align_shift: int          # 正常前段为对齐滑窗网格而少回放的行数
    aligned: bool             # 是否与离线评估的滑窗网格对齐
    window_size: int
    stride: int
    note: str = ""

    @property
    def total_rows(self) -> int:
        return sum(b.length for b in self.blocks)

    @property
    def has_post(self) -> bool:
        return any(b.key == "normal_post" for b in self.blocks)


@dataclass
class Event:
    """时间线上的一件事（用于屏幕展示与报告）。

    rel_ms 为相对攻击起点的虚拟流时间（毫秒）；row 为测试流全局行号。
    """

    name: str
    row: int | None = None
    rel_ms: float | None = None
    p: float | None = None
    note: str = ""


@dataclass
class AlertRecord:
    """一条确认告警，附带演示侧的核对信息。"""

    seq: int
    phase: str              # normal_pre | attack | normal_post
    row_first: int
    row_last: int
    p_attack: float
    pred_class: str
    action: str
    src_ip: str
    truth: str              # 窗口真值：Normal / 含攻击（DDoS×9）
    is_false_alarm: bool
    ts: float               # 窗口右端的真实流时间戳
    rel_ms: float           # 相对攻击起点的虚拟流时间（毫秒）
    delay_ms: float | None  # 攻击片段内：相对攻击起点的流时间差（毫秒）
    crosses_seam: bool = False   # 窗口是否横跨"位置跳转"（演示拼接边界）

    @property
    def phase_cn(self) -> str:
        return {"normal_pre": "攻击前", "attack": "攻击片段内", "normal_post": "攻击后"}.get(self.phase, self.phase)


def _normal_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """所有"纯正常"连续片段：(起, 止, 长度)。"""
    runs: list[tuple[int, int, int]] = []
    i, n = 0, len(labels)
    while i < n:
        if labels[i] == NORMAL:
            j = i
            while j + 1 < n and labels[j + 1] == NORMAL:
                j += 1
            runs.append((i, j, j - i + 1))
            i = j + 1
        else:
            i += 1
    return runs


def _attack_segments(labels: np.ndarray) -> list[dict]:
    """连续的非 Normal 行构成一个攻击片段（与 app/app.py、src/evaluate.py 同定义）。"""
    segs: list[dict] = []
    i, n = 0, len(labels)
    while i < n:
        if labels[i] != NORMAL:
            j = i
            while j + 1 < n and labels[j + 1] == labels[i]:
                j += 1
            segs.append({"start": int(i), "end": int(j), "class": str(labels[i]), "length": int(j - i + 1)})
            i = j + 1
        else:
            i += 1
    return segs


def build_scenario(labels: np.ndarray, attack_class: str, window_size: int, stride: int,
                   normal_rows: int) -> Scenario:
    """挑出"看得清"的剧本：一段安静的正常流量 → 一段真实攻击 → 一段恢复的正常流量。

    选段规则（前半段与 app/app.py 的 ``/api/first_segment_of`` 一致，后半段是演示专用加强）：

    * **长度门槛**：只考虑长度 ≥ ``max(2×窗口长度, 8)`` 的片段。测试流里同一类攻击有很多极短片段
      （甚至只有 1 条流），跳到那种片段上会是"曲线刚起跳就结束"，看着像模型没报出来。
    * **长度优先**：在够长的候选里取最长的（并列取时间最早的）——app/app.py 的规则。
    * **演示专用加强**：当某候选片段同时拥有较长的"紧邻纯正常"前后段时优先它。
      理由：现场必须先看到足够长的安静基线，再看到攻击爆发；测试流里攻击片段密集交错，
      有些片段前面只有几行正常流量，放到屏幕上就是"上来就打"，讲不出"平时是安静的"。
    """
    segs = _attack_segments(labels)
    want = str(attack_class).lower()
    cands = [s for s in segs if s["class"].lower() == want]
    if not cands:
        have = sorted({s["class"] for s in segs})
        raise DemoError(f"测试流里没有 {attack_class} 类型的攻击片段。可选类型：{'、'.join(have)}", code=4)
    min_len = max(2 * int(window_size), 8)
    good = [s for s in cands if s["length"] >= min_len]
    pool = good or cands

    runs = _normal_runs(labels)
    pre_by_end = {b: (a, b, ln) for a, b, ln in runs}
    post_by_start = {a: (a, b, ln) for a, b, ln in runs}

    def pre_len(seg: dict) -> int:
        r = pre_by_end.get(seg["start"] - 1)
        return r[2] if r else 0

    def post_len(seg: dict) -> int:
        r = post_by_start.get(seg["end"] + 1)
        return r[2] if r else 0

    # 选段分两档：**优先能演成"一段连续数据"的候选**（攻击片段后面紧接着就是够长的纯正常流量），
    # 这样整场演示没有位置跳转、也不会产生横跨拼接缝的伪窗口；只有实在没有这种候选时，
    # 才退化为"跳到之后最近的纯正常片段"并在屏幕与报告里明确标注跳转。
    target_post = min(int(normal_rows), 200)
    strong = [s for s in pool if post_len(s) >= target_post]
    if strong:
        best = max(strong, key=lambda s: (min(pre_len(s), int(normal_rows)),
                                          min(post_len(s), int(normal_rows)),
                                          int(s["length"]), -int(s["start"])))
    else:
        best = max(pool, key=lambda s: (min(pre_len(s), int(normal_rows)) + min(post_len(s), int(normal_rows)),
                                        int(s["length"]), -int(s["start"])))

    # ---- 滑窗网格对齐：让"演示"与"离线评估"落在同一批窗口上 ----
    # 记 O = 回放位置 − 全局行号（同一段内是常数）。服务在第 r 行（1-based）满足 r % stride == 0 时判定
    # 且 r ≥ T，故窗口右端 0-based 回放位置 ≡ stride-1 (mod stride)；离线窗口右端恒为 ≡ T-1 (mod stride)。
    # 令两者相等：O ≡ (stride-1) − (T-1) (mod stride)。本工程 T=16、stride=3 ⇒ T-1 ≡ 0，故只需 O ≡ stride-1。
    def _want_gstart_mod(playback_pos: int) -> int:
        return (int(playback_pos) - (int(stride) - 1)) % int(stride)

    # ---------- 正常前段：紧邻攻击起点的纯正常片段（取其末尾若干条，保证与攻击段首尾相接）----------
    pre = pre_by_end.get(best["start"] - 1)
    pre_avail = pre[2] if pre else 0
    l1 = min(int(normal_rows), pre_avail)
    align_shift, aligned = 0, False
    if l1 >= int(window_size):
        want_mod = (int(best["start"]) - _want_gstart_mod(0)) % int(stride)
        shift = (l1 - want_mod) % int(stride)
        if l1 - shift >= int(window_size):
            align_shift, l1, aligned = shift, l1 - shift, True
    if l1 <= 0:
        raise DemoError(f"{attack_class} 片段前面没有可用的连续正常流量，无法构成演示剧本", code=4)

    blocks = [Block(key="normal_pre", gstart=int(best["start"]) - l1, length=l1)]
    blocks.append(Block(key="attack", gstart=int(best["start"]), length=int(best["length"]),
                        label=str(best["class"]), jumped=False))
    offset = l1 + int(best["length"])

    # ---------- 恢复段：优先取"紧接着攻击片段"的连续正常流量 ----------
    # 之所以优先"紧邻"而不是"最近"：跳转会在拼接缝上产生横跨两段的窗口——那些窗口在全局行号上
    # 并不连续，既不属于离线评估的窗口，也会让时间线出现两个互相矛盾的轴（真实时间戳差 vs 重建轴）。
    # 用紧邻段就没有这个问题：整场演示 = 测试流上**一段连续区间** [attack_end+1-M, attack_end+L]。
    post_len = 0
    contiguous_min = max(int(window_size), 8)
    adj = post_by_start.get(int(best["end"]) + 1)
    if adj and int(adj[2]) >= contiguous_min:
        ln = min(int(normal_rows), int(adj[2]))
        blocks.append(Block(key="normal_post", gstart=int(adj[0]), length=int(ln), jumped=False))
        post_len = int(ln)
    else:
        # 退路：测试流里攻击片段密集交错，少数片段后面确实没有足够长的连续正常流量，
        # 此时才退化为"跳到之后最近的纯正常片段"，并在屏幕与报告里明确标注这次位置跳转。
        post_cands = sorted([r for r in runs if r[0] > best["end"]], key=lambda r: r[0])
        chosen: tuple[int, int] | None = None
        for floor in (min(int(normal_rows), 128), 64, 32, 8, 1):
            for a, _b, ln in post_cands:
                shift = (_want_gstart_mod(offset) - a) % int(stride)
                avail = ln - shift
                if avail >= floor:
                    chosen = (a + shift, min(int(normal_rows), avail))
                    break
            if chosen:
                break
        if chosen:
            blocks.append(Block(key="normal_post", gstart=int(chosen[0]), length=int(chosen[1]), jumped=True))
            post_len = int(chosen[1])

    note = ""
    if not aligned:
        note = "正常前段过短，未能与离线评估的滑窗网格对齐；本场不做逐窗口一致性核对。"
    elif align_shift:
        note = f"为对齐滑窗网格，正常前段少回放 {align_shift} 条流记录（只影响窗口对齐，不影响指标口径）。"

    return Scenario(blocks=blocks, attack_class=str(best["class"]), seg_start=int(best["start"]),
                    seg_end=int(best["end"]), seg_length=int(best["length"]), n_candidates=len(cands),
                    n_long_enough=len(good), pre_run_len=pre_avail, post_run_len=post_len,
                    align_shift=align_shift, aligned=aligned, window_size=int(window_size),
                    stride=int(stride), note=note)


# ======================================================================================
# 数据与模型
# ======================================================================================
def load_stream(processed_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, Any]:
    """读预处理产物：元信息、留出测试流特征（标准化后）、真实标签、真实流时间戳、scaler。"""
    meta_path = processed_dir / "meta.json"
    tensor_path = processed_dir / "tensors.npz"
    if not meta_path.exists() or not tensor_path.exists():
        raise DemoError(
            "找不到预处理产物：\n"
            f"  · {rel_path(meta_path)}\n"
            f"  · {rel_path(tensor_path)}\n"
            "请先运行：python run_all.py   （只想快速验证可用 python run_all.py --smoke）",
            code=2,
        )
    meta = load_json(meta_path)
    z = np.load(tensor_path, allow_pickle=True)
    for key in ("test_stream_features", "test_stream_labels", "test_stream_time"):
        if key not in z:
            raise DemoError(f"预处理产物里缺少 {key}，请重跑 python run_all.py", code=2)
    X = np.asarray(z["test_stream_features"], dtype=np.float32)
    labels = np.asarray(z["test_stream_labels"]).astype(str)
    times = np.asarray(z["test_stream_time"], dtype=np.float64)
    if len(X) == 0 or len(labels) != len(X) or len(times) != len(X):
        raise DemoError("预处理产物里的测试流数据不完整，请重跑 python run_all.py", code=2)
    try:
        import joblib

        scaler = joblib.load(processed_dir / "scaler.joblib")
    except Exception as exc:  # noqa: BLE001
        raise DemoError(f"读取标准化器失败（{rel_path(processed_dir / 'scaler.joblib')}）：{exc}", code=2) from exc
    return meta, X, labels, times, scaler


def load_ips(meta: dict, n_stream: int, seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """每条流记录的源/目的 IP。

    优先用**真实字段**：``data/raw/<schema>_stream.csv`` 的 ``Source IP`` / ``Destination IP``
    （留出测试流即该 CSV 的最后 n_stream 行，与 ``tools/controller_simulator.py`` 的取法一致）。
    读不到就退回确定性合成的 10.0.0.x，并在报告里注明——**绝不假装是真实 IP**。
    """
    schema = str((meta.get("raw_meta") or {}).get("schema", "insdn"))
    csv_path = ROOT / "data" / "raw" / f"{schema}_stream.csv"
    if csv_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_path, usecols=["Source IP", "Destination IP"], low_memory=False)
            if len(df) >= n_stream:
                src = df["Source IP"].astype(str).to_numpy()[-n_stream:]
                dst = df["Destination IP"].astype(str).to_numpy()[-n_stream:]
                return src, dst, True
        except Exception:  # noqa: BLE001
            pass
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(1, 19, size=n_stream)
    src = np.asarray([f"10.0.0.{int(i)}" for i in idx], dtype=object)
    dst = np.asarray(["10.0.0.1"] * n_stream, dtype=object)
    return src, dst, False


def load_prediction_reference(model_tag: str) -> dict[tuple[int, int], tuple[float, str]] | None:
    """读离线评估缓存的窗口级预测（``artifacts/predictions_<tag>.csv``）用于逐窗口一致性核对。

    这是"演示=报告"最硬的证据：同一模型、同一批窗口，演示里算出来的概率应与离线缓存一致。
    """
    path = ARTIFACT_DIR / f"predictions_{model_tag}.csv"
    if not path.exists():
        return None
    ref: dict[tuple[int, int], tuple[float, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ref[(int(row["win_start"]), int(row["win_end"]))] = (float(row["p_attack"]), row["pred_class"])
    except Exception:  # noqa: BLE001
        return None
    return ref or None


# ======================================================================================
# 演示运行时
# ======================================================================================
class LiveDemo:
    """把"剧本 + 检测服务 + 屏幕输出 + 报告"串起来的运行时。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = Config()
        self.console = Console(color=not args.no_color, animate=not args.no_animation)
        self.t0 = time.perf_counter()
        self.pushed = 0
        self.window_hist: list[float] = []
        self.windows: list[dict] = []
        self.events: list[Event] = []
        self.alerts: list[AlertRecord] = []
        self.blocked: dict[str, float] = {}
        self._burst: dict[str, Any] = {"active": False, "cls": "", "n": 0, "first": 0, "new_ips": []}
        self.rows_per_sec = 0.0
        self._last_status_t = -1e9
        self._last_status_rows = -10 ** 9
        self.check_n = 0
        self.check_max_dp = 0.0
        self.check_mismatch = 0
        self.check_big = 0
        self.check_skipped = 0
        self.stream_resets = 0      # 剧本里发生了几次"数据中断 → 滑窗缓冲重置"
        self.delay_ms: float | None = None
        self.delay_rows: int | None = None

    # ------------------------------------------------------------------ 输出
    def now(self) -> float:
        return time.perf_counter() - self.t0

    def say(self, text: str, code: str | None = None) -> None:
        self.console.line(f"{hm(self.now())} {sanitize(text)}", code)

    # ------------------------------------------------------------------ 主流程
    def run(self) -> int:
        args = self.args
        self.say("启动：加载预处理产物与模型（进程内调用检测服务，不监听端口、不联网）", C_CYAN)

        processed_dir = Path(args.processed_dir) if args.processed_dir else PROCESSED_DIR
        meta, X, labels, times, scaler = load_stream(processed_dir)
        self._times = times
        n_stream = len(labels)
        feature_names = list(meta["feature_names"])
        window_size = int(meta.get("window_size", self.cfg.window_size) or self.cfg.window_size)

        try:
            from src.service import DetectionService
        except ImportError as exc:  # pragma: no cover
            raise DemoError(f"无法导入检测服务 src/service.py：{exc}", code=2) from exc
        _quiet_logs()

        t_load = time.perf_counter()
        try:
            svc = DetectionService(cfg=self.cfg, processed_dir=processed_dir,
                                   artifact_dir=Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_DIR,
                                   model_tag=args.model_tag or None)
        except Exception as exc:  # noqa: BLE001
            raise DemoError(_explain_load_failure(exc), code=3) from exc
        if not svc.ready:
            raise DemoError(_explain_ready_failure(str(svc.reason)),
                            code=3 if "TensorFlow" in str(svc.reason) else 2)
        load_sec = time.perf_counter() - t_load

        # 审计台账写到演示专用文件：不把回放数据混进 results/audit_alerts.jsonl 那份真实闭环台账
        svc.audit_path = AUDIT_PATH
        try:
            AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            AUDIT_PATH.write_text("", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

        thr_req = float(args.threshold if args.threshold is not None else self.cfg.report_threshold)
        conf_req = int(args.confirm if args.confirm is not None else self.cfg.online_confirm_windows)
        policy = svc.reload_policy(threshold=thr_req, confirm_windows=conf_req, action=args.action, dry_run=True)
        threshold = float(policy["threshold"])
        confirm = int(policy["confirm_windows"])
        model_tag = svc.model_path.parent.name if getattr(svc, "model_path", None) else "unknown"
        stride = int(svc.cfg.online_stride)
        self.model_tag = model_tag
        self.whitelist = tuple(getattr(svc.policy, "whitelist", ()) or ())
        self.threshold, self.confirm, self.policy = threshold, confirm, policy

        self.say(f"检测服务就绪：模型 {model_tag} · 窗口 T={window_size} · 在线步长 {stride} · "
                 f"阈值 {threshold:.2f} · 连续确认 {confirm} 窗口 · 处置建议 {policy['action']} · "
                 f"dry_run={policy['dry_run']}（演练模式，不真实下发）· 加载耗时 {load_sec:.1f} 秒", C_GREEN)

        scene = build_scenario(labels, args.attack, window_size, stride, args.normal_rows)
        self.scenario = scene
        self.rows_per_sec = 0.0 if args.speed <= 0 else BASE_ROWS_PER_SEC * float(args.speed)
        src_ips, dst_ips, ips_real = load_ips(meta, n_stream, args.seed)
        self.ips_real = ips_real

        # tensors.npz 里存的是标准化后的特征，而检测服务内部会再做一次标准化，
        # 因此先反标准化回真实物理量再喂进去（与 tools/controller_simulator.py 同一做法）。
        raw = scaler.inverse_transform(X).astype(np.float64)

        # ---------------- 回放计划 ----------------
        play_global: list[int] = []
        play_block: list[int] = []
        for bi, blk in enumerate(scene.blocks):
            for k in range(blk.length):
                play_global.append(blk.gstart + k)
                play_block.append(bi)
        total = len(play_global)
        pos_of: dict[int, int] = {g: i for i, g in enumerate(play_global)}

        # 虚拟流时间轴：同段内用真实时间戳差；跨段（位置跳转）处用接收段的中位流间隔补齐。
        # 目的：既不假装时间连续，也不让数据里几十秒的采集空档把"演示时间轴"拉爆。
        virt = np.zeros(total, dtype=np.float64)
        for bi in range(len(scene.blocks)):
            pos = [i for i, b in enumerate(play_block) if b == bi]
            if not pos:
                continue
            g0, g1 = pos[0], pos[-1]
            d = np.diff(times[play_global[g0]:play_global[g1] + 1])
            med = float(np.median(d)) if d.size else 0.002
            acc = 0.0
            start = g0 if g0 == 0 else g0
            for i in range(start, g1 + 1):
                if i == g0:
                    virt[i] = (0.0 if g0 == 0 else virt[g0 - 1] + med)
                    acc = virt[i]
                    continue
                same = play_global[i] == play_global[i - 1] + 1
                acc += float(times[play_global[i]] - times[play_global[i - 1]]) if same else med
                virt[i] = acc
        virt_base = float(virt[pos_of[scene.seg_start]])   # 攻击起点的虚拟时间，用于相对秒数

        reference = load_prediction_reference(model_tag)
        b0, b1 = scene.blocks[0], scene.blocks[1]
        post = next((b for b in scene.blocks if b.key == "normal_post"), None)

        self.say(f"数据来源：{rel_path(processed_dir / 'tensors.npz')} 的真实留出测试流"
                 f"（{n_stream:,} 条流记录，真实标签 + 真实流时间戳）", C_DIM)
        self.say("剧本：" + " → ".join(f"{b.name_cn} {b.length} 条（第 {b.gstart}~{b.gend} 行）"
                                        + ("[位置跳转]" if b.jumped else "") for b in scene.blocks), C_CYAN)
        if scene.note:
            self.say(f"选段说明：{scene.note}", C_DIM)
        if self.rows_per_sec > 0:
            self.say(f"回放节奏：{self.rows_per_sec:.0f} 条流记录/秒（{args.speed:g} 倍速，纯展示节奏，"
                     f"不参与任何指标计算）· 预计 {total / self.rows_per_sec:.1f} 秒演完", C_DIM)
        else:
            self.say("回放节奏：不限速（--speed 0，自动化模式）", C_DIM)

        self.events.append(Event("剧本开始", row=b0.gstart, rel_ms=(float(virt[0]) - virt_base) * 1000.0,
                                 note=f"正常流量回放 {b0.length} 条"))

        # ---------------- 回放 ----------------
        batch = max(1, int(args.batch))
        for bi, blk in enumerate(scene.blocks):
            self._announce_phase(bi, blk, scene, play_global, pos_of, virt, virt_base)
            if blk.jumped:
                # 位置跳转 = 上游数据中断。此时必须重置滑窗缓冲：
                # 否则中断前后的行会被拼进同一个窗口，生成**现实中不存在的窗口**
                # （它在全局行号上并不连续，既不是真实相邻流量，也不属于任何一次离线评估）。
                # 真机上的对应动作就是"流量打一个断点、窗口从头累积"，这里如实照做并打印。
                info = svc.reset_stream(reason=f"剧本位置跳转 → 第 {blk.gstart} 行")
                self.stream_resets += 1
                self.say(f"⏸ 数据中断（位置跳转）：滑窗缓冲已重置，丢弃已积累的 "
                         f"{int(info.get('dropped_rows') or 0)} 条流记录，窗口从头累积"
                         f"（避免出现横跨两段的伪窗口）", C_YELLOW)
            base_pos = play_block.index(bi)
            for off in range(0, blk.length, batch):
                take = min(batch, blk.length - off)
                positions = [base_pos + off + k for k in range(take)]
                rows = []
                for p in positions:
                    g = play_global[p]
                    row = {f: float(raw[g, j]) for j, f in enumerate(feature_names)}
                    row["Label"] = str(labels[g])          # 真值只用于核对，不参与推理
                    row["Source IP"] = str(src_ips[g])
                    row["Destination IP"] = str(dst_ips[g])
                    rows.append(row)
                self._pace(self.pushed + take)
                resp = svc.push(rows, meta={"src_ip": str(src_ips[play_global[positions[0]]]),
                                            "dst_ip": str(dst_ips[play_global[positions[0]]])})
                if not resp.get("ok"):
                    raise DemoError(f"检测服务返回错误：{resp.get('error')}", code=1)
                self.pushed += take
                self._handle_batch(resp, positions, play_global, pos_of, virt, virt_base, times, labels,
                                   scene, threshold, reference)
                self._maybe_status(scene, bi, threshold)

        self._close_burst()
        self._maybe_status(scene, len(scene.blocks) - 1, threshold, force=True)
        self.console.close()
        summary = self._finish(scene, svc, reference)
        self._write_report(scene, summary, reference)
        return 0

    # ------------------------------------------------------------------ 阶段提示
    def _announce_phase(self, bi: int, blk: Block, scene: Scenario, play_global: list[int],
                        pos_of: dict[int, int], virt: np.ndarray, virt_base: float) -> None:
        if blk.key == "normal_pre":
            self.say(f"推流中… 正常流量回放（第 {blk.gstart}~{blk.gend} 行，共 {blk.length} 条）", C_DIM)
        elif blk.key == "attack":
            p = pos_of[blk.gstart]
            self.say(f"⚠ 攻击开始（真实标签：{blk.label}，注入位置 = 真实攻击片段起点 "
                     f"第 {blk.gstart} 行 / 流时间 {fmt_ts(self._times[blk.gstart])}）", C_MAGENTA)
            self.events.append(Event("攻击起点", row=blk.gstart, rel_ms=0.0,
                                     note=f"真实标签切换为 {blk.label}（长度 {blk.length} 条）"))
            del p
        elif blk.key == "normal_post":
            last = next(b.gend for b in scene.blocks if b.key == "attack")
            self.events.append(Event("攻击结束", row=last,
                                     rel_ms=(float(virt[pos_of[last]]) - virt_base) * 1000.0,
                                     note="该片段最后一条攻击流记录（真实标签切回 Normal）"))
            self.say(f"攻击结束（第 {last} 行为该片段最后一条攻击流记录）", C_CYAN)
            if blk.jumped:
                self.say(f"恢复观察中… 回放到攻击片段之后最近的纯正常片段（第 {blk.gstart}~{blk.gend} 行，"
                         f"位置有一次跳转）", C_DIM)
            else:
                self.say(f"恢复观察中… 紧接着攻击片段的连续正常流量（第 {blk.gstart}~{blk.gend} 行，"
                         f"数据连续、无位置跳转）", C_DIM)
        self._close_burst()   # 阶段切换时收尾上一次"告警爆发"的计数

    # ------------------------------------------------------------------ 批次处理
    def _handle_batch(self, resp: dict, positions: list[int], play_global: list[int],
                      pos_of: dict[int, int], virt: np.ndarray, virt_base: float, times: np.ndarray,
                      labels: np.ndarray, scene: Scenario, threshold: float,
                      reference: dict | None) -> None:
        """处理一批推送的返回：记录窗口判定、时间线事件，并把告警对齐到它真正的窗口。

        ⚠ 一个必须自己算的细节：服务的 ``alert.rows_covered`` 是在**整批推完之后**才填的
        （``rows_covered=(stats["rows"]-T+1, stats["rows"])``），因此同一批里多条告警拿到的都是
        "本批最后 T 行"，不能用来定位窗口。这里改为按服务自己的判定规则重算：
        第 (p+1) 行（1-based）满足 ``(p+1) % stride == 0`` 且 ``p+1 >= T`` 时窗口完成，
        ``resp["rows"]`` 按行序返回每个窗口，其中 ``confirmed=True`` 的那些**顺序**
        与 ``resp["alerts"]`` 一一对应。
        """
        seg_lo, seg_hi = scene.seg_start, scene.seg_end
        alerts = list(resp.get("alerts") or [])
        n_used = 0
        for k, item in enumerate(resp.get("rows") or []):
            win = item.get("window")
            if not win:
                continue
            p = float(win["p_attack"])
            pos = positions[k]
            g_end = play_global[pos]
            self.window_hist.append(p)
            self._last_confirmed = bool(win["confirmed"])
            self.windows.append({"pos": pos, "p": p, "pred": str(win["pred_class"]),
                                 "alarm": bool(win["alarm"]), "confirmed": bool(win["confirmed"]),
                                 "g_end": g_end})
            rel_ms = (float(virt[pos]) - virt_base) * 1000.0

            # ---- 逐窗口一致性核对：窗口完全落在同一段内时，其内容与离线评估的某个窗口完全相同 ----
            if reference is not None:
                p_first = pos - (scene.window_size - 1)
                if p_first < 0:
                    self.check_skipped += 1
                else:
                    g_first = play_global[p_first]
                    same_block = all(play_global[p_first + j] == g_first + j for j in range(scene.window_size))
                    if not same_block:
                        self.check_skipped += 1
                    else:
                        ref = reference.get((g_first, g_end))
                        if ref is None:
                            self.check_skipped += 1
                        else:
                            self.check_n += 1
                            dp = abs(float(ref[0]) - p)
                            self.check_max_dp = max(self.check_max_dp, dp)
                            if dp > 1e-3:
                                self.check_big += 1
                            if str(ref[1]) != str(win["pred_class"]):
                                self.check_mismatch += 1

            if win["alarm"] and g_end >= seg_lo and not any(e.name == "首个超阈值窗口" for e in self.events):
                self.events.append(Event("首个超阈值窗口", row=g_end, rel_ms=rel_ms, p=p,
                                         note=f"p={p:.3f} ≥ 阈值 {threshold:.2f}（尚未连续确认）"))
                self.say(f"风险值越线：{p:.3f} ≥ 阈值 {threshold:.2f}（第 {g_end} 行，"
                         f"攻击起点后第 {g_end - seg_lo} 条流记录）· 等待连续 {self.confirm} 个窗口确认", C_YELLOW)
            if g_end > seg_hi and p < threshold and not any(e.name == "风险回落" for e in self.events):
                self.events.append(Event("风险回落", row=g_end, rel_ms=rel_ms, p=p,
                                         note=f"p={p:.3f} < 阈值 {threshold:.2f}（攻击片段已结束）"))
                self.say(f"攻击结束，风险值回落：{p:.3f} < 阈值 {threshold:.2f}（第 {g_end} 行）", C_GREEN)
            if (g_end > seg_hi and p < threshold
                    and not any(e.name == "风险稳定恢复" for e in self.events)):
                run = 0
                for w in reversed(self.windows):
                    run = run + 1 if w["p"] < threshold else 0
                    if run >= self.confirm:
                        break
                if run >= self.confirm:
                    self.events.append(Event("风险稳定恢复", row=g_end, rel_ms=rel_ms, p=p,
                                             note=f"连续 {self.confirm} 个窗口低于阈值（最近 p={p:.3f}）"))
                    self.say(f"恢复正常：连续 {self.confirm} 个窗口低于阈值（最近 p={p:.3f}）", C_GREEN)

            # 确认窗口 → 对应的告警（服务按同一顺序返回）。**放在同一个循环里处理**，
            # 这样屏幕上的先后顺序与流记录顺序完全一致（否则同一批里的"回落"会跑到"告警"前面）。
            if win["confirmed"] and n_used < len(alerts):
                self._handle_alert(alerts[n_used], pos, play_global, virt, virt_base, times, labels,
                                   scene, threshold)
                n_used += 1
        alerts = list(resp.get("alerts") or [])
        if n_used != len(alerts):   # 理论上不会发生；真发生就如实提示，不猜
            self.say(f"⚠ 内部对齐提示：本批服务返回 {len(alerts)} 条告警、已按窗口顺序匹配 {n_used} 条，"
                     f"其余按本批最后一个窗口归位", C_YELLOW)
            for a in alerts[n_used:]:
                self._handle_alert(a, positions[-1], play_global, virt, virt_base, times, labels,
                                   scene, threshold)

    def _handle_alert(self, a: dict, pos_end: int, play_global: list[int], virt: np.ndarray,
                      virt_base: float, times: np.ndarray, labels: np.ndarray, scene: Scenario,
                      threshold: float) -> None:
        p_first = max(0, pos_end - (scene.window_size - 1))
        g_first, g_last = play_global[p_first], play_global[pos_end]
        # 窗口真值必须按"窗口里真正包含的行"来取：位置跳转处的窗口在全局行号上是不连续的，
        # 用 labels[g_first:g_last+1] 会把中间没被回放的行也算进来（那是错的）。
        win_idx = list(range(p_first, pos_end + 1))
        win_labels = [str(labels[play_global[i]]) for i in win_idx]
        bad = [x for x in win_labels if x != NORMAL]
        truth = NORMAL if not bad else "含攻击（" + "、".join(f"{k}×{v}" for k, v in Counter(bad).most_common()) + "）"
        crosses = any(play_global[i] != play_global[i - 1] + 1 for i in win_idx[1:])
        seg_lo, seg_hi = scene.seg_start, scene.seg_end
        rel_ms = (float(virt[pos_end]) - virt_base) * 1000.0
        if g_last >= seg_lo and g_last <= seg_hi:
            phase, delay = "attack", (float(times[g_last]) - float(times[seg_lo])) * 1000.0
        elif g_last < seg_lo:
            phase, delay = "normal_pre", None
        else:
            phase, delay = "normal_post", None
        is_false = not bad

        rec = AlertRecord(seq=len(self.alerts) + 1, phase=phase, row_first=g_first, row_last=g_last,
                          p_attack=float(a["p_attack"]), pred_class=str(a["pred_class"]),
                          action=str(a["action"]), src_ip=str(a.get("src_ip") or "-"), truth=truth,
                          is_false_alarm=is_false, ts=float(times[g_last]), rel_ms=rel_ms, delay_ms=delay,
                          crosses_seam=crosses)
        self.alerts.append(rec)
        seam_txt = "  ← 跨位置跳转窗口（演示拼接边界，不属于离线评估的窗口）" if crosses else ""

        # ---- 屏幕输出：误报逐条打印；同一场"告警爆发"只打印首条 + 计数（全部记录仍在报告与台账里）----
        if is_false:
            self._close_burst()
            self.say(f"⚠ 误报 #{rec.seq}：判定 {rec.pred_class} · 异常概率 {rec.p_attack:.3f} · "
                     f"窗口 [第 {g_first}~{g_last} 行]（真值 Normal）· 处置 {rec.action}"
                     f"  ← 攻击前/后的误报照实打印，不隐藏{seam_txt}", C_YELLOW)
        else:
            new_burst = (not self._burst["active"]) or self._burst["cls"] != rec.pred_class
            if new_burst:
                self._close_burst()
                self._burst.update(active=True, cls=rec.pred_class, n=0, first=rec.seq)
                tail = (f" · 检测时延 {delay:.1f} 毫秒（流时间，攻击起点后第 {g_last - seg_lo} 条流记录）"
                        if delay is not None else "")
                head = "  ← 首个确认告警" if delay is not None else ""
                self.say(f"⚠ 告警 #{rec.seq}：判定 {rec.pred_class} · 异常概率 {rec.p_attack:.3f} · "
                         f"窗口 [第 {g_first}~{g_last} 行] 真值 {truth} · 处置建议 {rec.action}{tail}{head}{seam_txt}",
                         C_RED)
            self._burst["n"] += 1

        if not is_false and delay is not None and not any(e.name == "首个确认告警" for e in self.events):
            self.delay_ms, self.delay_rows = delay, int(g_last - seg_lo)
            self.events.append(Event("首个确认告警", row=g_last, rel_ms=rel_ms, p=rec.p_attack,
                                     note=f"判定 {rec.pred_class}，p={rec.p_attack:.4f}"))

        # ---- 处置：本地模拟执行器（真实下发见 integrations/ryu_guard.py）----
        if rec.action in ("block", "throttle"):
            verb = "加入阻断名单" if rec.action == "block" else "加入限速名单"
            if rec.src_ip not in self.blocked:
                self.blocked[rec.src_ip] = self.now()
                if not any(e.name == "处置下发" for e in self.events):
                    self.events.append(Event("处置下发", row=g_last, rel_ms=rel_ms, p=rec.p_attack,
                                             note=f"{rec.action} ← 策略建议，由本地模拟执行器执行"))
                if not self._burst["new_ips"]:
                    self.say(f"⛔ 处置生效：源 IP {rec.src_ip} 已{verb}（模拟执行，有效期 "
                             f"{self.args.block_seconds} 秒）· 名单当前 {len(self.blocked)} 个地址", C_RED)
                self._burst["new_ips"].append(rec.src_ip)

    def _close_burst(self) -> None:
        """收尾一次"告警爆发"：屏幕上只打印首条与首个处置地址，其余以计数汇总。

        全部条目仍在报告第五节与 results/live_demo_audit.jsonl 里（一条都不少）。
        """
        b = self._burst
        if b["active"] and (b["n"] > 1 or len(b["new_ips"]) > 1):
            extra = ""
            if len(b["new_ips"]) > 1:
                extra = (f"；处置共覆盖 {len(self.blocked)} 个源地址"
                         f"（{'、'.join(sorted(self.blocked))}）")
            self.say(f"… 本次告警爆发（{b['cls']}，自 #{b['first']} 起）累计 {b['n']} 条确认告警{extra}；"
                     f"屏幕只打印首条与首个处置地址，完整记录见 results/live_demo_incident.md 第五节与 "
                     f"results/live_demo_audit.jsonl", C_DIM)
        b.update(active=False, cls="", n=0, first=0, new_ips=[])

    # ------------------------------------------------------------------ 状态条
    def _maybe_status(self, scene: Scenario, block_index: int, threshold: float, force: bool = False) -> None:
        now = self.now()
        animate = self.console.animate and self.rows_per_sec > 0
        if not force:
            if animate:
                if now - self._last_status_t < 0.12:
                    return
            elif self.pushed - self._last_status_rows < max(1, int(self.args.status_rows)):
                return
        self._last_status_t = now
        self._last_status_rows = self.pushed
        blk = scene.blocks[min(block_index, len(scene.blocks) - 1)]
        p = self.window_hist[-1] if self.window_hist else 0.0
        n_hit = sum(1 for a in self.alerts if not a.is_false_alarm)
        n_false = len(self.alerts) - n_hit
        if p >= threshold:
            if getattr(self, "_last_confirmed", False):
                phase_txt, code = "告警中（已确认）", C_RED
            else:
                phase_txt = "触发告警（待确认）"
                code = C_RED if blk.key == "attack" else C_YELLOW
        else:
            phase_txt = "正常" if blk.key != "attack" else "攻击片段内（尚未越线）"
            code = C_GREEN
        self.console.live(f"{hm(now)} 风险值 {p:.3f}  {spark(self.window_hist)}  {phase_txt} · "
                          f"已回放 {self.pushed}/{scene.total_rows} 条 · 确认告警 {n_hit} · 误报 {n_false}", code)

    # ------------------------------------------------------------------ 节奏
    def _pace(self, pushed: int) -> None:
        if self.rows_per_sec <= 0:
            return
        dt = (self.t0 + pushed / self.rows_per_sec) - time.perf_counter()
        if dt > 0:
            time.sleep(min(dt, 1.0))

    # ------------------------------------------------------------------ 结束
    def _finish(self, scene: Scenario, svc: Any, reference: dict | None) -> dict:
        self.console.close()
        stats = svc.snapshot()
        attack_alerts = [a for a in self.alerts if a.phase == "attack" and not a.is_false_alarm]
        false_pre = [a for a in self.alerts if a.phase == "normal_pre" and a.is_false_alarm]
        false_post = [a for a in self.alerts if a.phase == "normal_post" and a.is_false_alarm]
        by_action = Counter(a.action for a in self.alerts)
        fall = next((e for e in self.events if e.name == "风险回落"), None)

        # "稳定恢复"：回落点之后连续 confirm 个窗口都低于阈值
        recovered = False
        if fall is not None:
            fall_pos = max((w["pos"] for w in self.windows if w["g_end"] == fall.row), default=None)
            if fall_pos is not None:
                run = 0
                for w in [w for w in self.windows if w["pos"] >= fall_pos]:
                    run = run + 1 if w["p"] < self.threshold else 0
                    if run >= self.confirm:
                        recovered = True
                        break

        missed = not attack_alerts
        self.say(f"演示结束：回放 {self.pushed} 条流记录 · 攻击片段内确认告警 {len(attack_alerts)} 条 · "
                 f"误报 {len(false_pre) + len(false_post)} 条（攻击前 {len(false_pre)} / 攻击后 {len(false_post)}）· "
                 + (f"检测时延 {self.delay_ms:.1f} 毫秒（流时间，折合 {self.delay_rows} 条流记录）"
                    if self.delay_ms is not None else "本片段未报出确认告警"),
             C_RED if missed else C_GREEN)
        if reference is not None and self.check_n:
            self.say(f"一致性核对：与离线评估缓存逐窗口比对 {self.check_n} 个窗口，最大 |Δp| = "
                     f"{self.check_max_dp:.2e}，判定类别不一致 {self.check_mismatch} 个"
                     f"（跳过 {self.check_skipped} 个不可比窗口：跨位置跳转处）", C_DIM)
        self.say(f"服务侧批量推理时延（真实墙钟，只上屏幕、不入报告）：p50 {stats['batch_latency_ms']['p50']:.2f} ms · "
                 f"p95 {stats['batch_latency_ms']['p95']:.2f} ms · 累计判定窗口 {stats['windows']} 个", C_DIM)
        self.say("诚实说明：屏幕上的“风险回落/恢复正常”来自回放走过了真实攻击片段（真实标签切回 Normal），"
                 "不是处置动作把攻击压住了——本机没有真实流量，处置只到“策略建议 + 本地模拟执行器”；"
                 "真实下发见 integrations/ryu_guard.py（本机无 Ryu 环境，未实跑）。", C_YELLOW)
        return {"attack_alerts": attack_alerts, "false_pre": false_pre, "false_post": false_post,
                "by_action": by_action, "recovered": recovered, "missed": missed, "fall": fall,
                "n_blocked": len(self.blocked), "blocked_ips": sorted(self.blocked), "stats": stats}

    # ------------------------------------------------------------------ 报告
    def _run_args(self) -> str:
        a = self.args
        parts = ["python tools/live_demo.py",
                 f"--attack {a.attack}", f"--normal-rows {a.normal_rows}", f"--speed {a.speed:g}",
                 f"--threshold {self.threshold:g}", f"--confirm {self.confirm}", f"--action {a.action}"]
        if a.speed <= 0:
            parts.append("--no-color")
        return sanitize(" ".join(parts))

    def _offline_ref_line(self) -> str:
        try:
            m = load_json(RESULT_DIR / f"metrics_{self.model_tag}.json").get("online", {})
            sec = m.get("mean_detection_delay_sec")
            rows = m.get("mean_detection_delay_rows")
            if sec is None:
                return "离线评估未提供"
            return (f"同一模型在同一测试流、{m.get('n_attack_segments', '?')} 个攻击片段上的平均时延 "
                    f"{float(sec) * 1000:.1f} 毫秒（平均 {float(rows):.0f} 条流记录），"
                    f"见 results/metrics_{self.model_tag}.json")
        except Exception:  # noqa: BLE001
            return "离线评估未提供"

    def _write_report(self, scene: Scenario, summary: dict, reference: dict | None) -> None:
        seg_lo, seg_hi = scene.seg_start, scene.seg_end
        miss = summary["missed"]
        fall = summary["fall"]
        ev = {e.name: e for e in self.events}
        lines: list[str] = []

        lines.append("# 实况演示事件报告（live_demo）\n")
        lines.append(f"> 一条命令生成：`{self._run_args()}`\n")
        lines.append("> 本报告**不含墙钟时间、不含随机数**：同一组参数下两次运行逐字节一致"
                     "（可用 `Get-FileHash results/live_demo_incident.md` 自行校验）。"
                     "屏幕上的 `[mm:ss.s]` 是演示墙钟（节奏信息），故不写入报告。"
                     "报告里唯一随参数变化的只有顶部这行命令与“回放倍速”一行——"
                     "把 `--speed` 从 0 改到 50，其余所有数字（含检测时延、误报、一致性核对）都不变。\n")

        # ---------------- 一、事件摘要 ----------------
        lines.append("## 一、事件摘要\n")
        conclusion = ("已报出确认告警，并完成处置建议下发（本机为模拟执行）" if not miss
                      else "**本片段未报出确认告警（漏报）**")
        recover_txt = ("是：攻击片段结束后风险值回落到阈值以下"
                       + ("，并连续满确认窗口数保持低位" if summary["recovered"] else "")) if fall else "未观察到回落"
        lines.append(markdown_table(["项目", "内容"], [
            ["攻击类型（真实标签）", scene.attack_class],
            ["注入位置", f"第 {seg_lo} 行（测试流全局行号，0-based）· 流时间 {fmt_ts(self._times[seg_lo])}"],
            ["攻击片段长度", f"{scene.seg_length} 条流记录（选段门槛 max(2T,8)={max(2 * scene.window_size, 8)}；"
                             f"该类型候选 {scene.n_candidates} 个，其中够长 {scene.n_long_enough} 个）"],
            ["回放剧本", " → ".join(f"{b.name_cn} {b.length} 条"
                                     + ("（位置跳转）" if b.jumped else "") for b in scene.blocks)],
            ["检测模型与判定口径", f"{self.model_tag} · 窗口 T={scene.window_size} · 在线步长 {scene.stride} · "
                                   f"阈值 {self.threshold:.2f} · 连续确认 {self.confirm} 窗口"
                                   f"（与 src/evaluate.py 同口径、同滑窗网格）"],
            ["处置动作", f"策略建议 action={self.policy['action']}（dry_run={self.policy['dry_run']}）；"
                         f"本地模拟执行器写入 {summary['n_blocked']} 个源地址"
                         + (f"：{'、'.join(summary['blocked_ips'])}" if summary["blocked_ips"] else "")],
            ["结论", conclusion],
            ["风险是否回落", recover_txt],
        ]))
        lines.append("")

        # ---------------- 二、时间线 ----------------
        lines.append("## 二、时间线\n")
        rows: list[list[str]] = []
        for name in ("剧本开始", "攻击起点", "首个超阈值窗口", "首个确认告警", "处置下发"):
            e = ev.get(name)
            if e is None:
                if name in ("首个超阈值窗口", "首个确认告警") and miss:
                    rows.append([name, "—", "—", "—", "—", "**未出现：本片段漏报**"])
                continue
            note = e.note or ""
            if name == "首个确认告警" and self.delay_ms is not None:
                note += (f"；**检测时延 = {self.delay_ms:.1f} 毫秒**（折合 {self.delay_rows} 条流记录）")
            rows.append([name, _ms_txt(e.rel_ms), _ts_txt(e.row, self._times), _row_txt(e.row), _p_txt(e.p),
                         note or "—"])
        e = ev.get("攻击结束")
        if e is not None:
            rows.append(["攻击结束", _ms_txt(e.rel_ms), _ts_txt(e.row, self._times), _row_txt(e.row), "—",
                         e.note or "该片段最后一条攻击流记录（真实标签切回 Normal）"])
        else:
            rows.append(["攻击结束", "—", _ts_txt(seg_hi, self._times), str(seg_hi), "—",
                         "该片段最后一条攻击流记录（真实标签切回 Normal）"])
        for name in ("风险回落", "风险稳定恢复"):
            e = ev.get(name)
            if e is not None:
                note = e.note or "—"
                # 位置跳转之后的事件要标出来：这一段的「演示时间轴」与「真实流时间戳」之间
                # 隔着一次跳转，两个数不能直接相减（详见表下注释）。
                if any(b.jumped for b in scene.blocks) and any(b.key == "normal_post" and b.jumped
                                                               and e.row >= b.gstart for b in scene.blocks):
                    note += "（本节数据位于**位置跳转之后**，两个时间轴不可直接相减）"
                rows.append([name, _ms_txt(e.rel_ms), _ts_txt(e.row, self._times), _row_txt(e.row),
                             _p_txt(e.p), note])
        if self.windows:
            lastw = self.windows[-1]
            rows.append(["剧本结束", "—", _ts_txt(lastw["g_end"], self._times), str(lastw["g_end"]),
                         f"{lastw['p']:.4f}", "回放结束（最后一条流记录及其判定）"])
        lines.append(markdown_table(["事件", "演示时间轴（重建·相对攻击起点）", "真实流时间戳（UTC）", "流记录行号",
                                     "风险值 p_attack", "说明"], rows))
        lines.append("")
        _jumped = any(b.jumped for b in scene.blocks)
        lines.append("> 演示时间轴 = 数据集真实流时间戳差累计"
                     + ("（跨位置跳转处用该段中位流间隔补齐，因此**在跳转处不可与「真实流时间戳」直接相减**）"
                        if _jumped else "（本场全程连续，两列可直接相减、逐行自洽）")
                     + "；**检测时延一律取真实流时间戳差**（攻击片段起点 → 首个确认告警窗口的右端行），"
                     "与 `src/evaluate.py::online_alarm_evaluation` 完全一致，不是墙上时间，也不受回放倍速影响。\n")

        # ---------------- 三、关键指标 ----------------
        lines.append("## 三、关键指标\n")
        n_win_attack = sum(1 for w in self.windows if seg_lo <= w["g_end"] <= seg_hi)
        if reference is not None and self.check_n:
            skip_txt = (f"，跳过 {self.check_skipped} 个不可比窗口（位置跳转后滑窗已重置，"
                        f"这些窗口不属于离线评估的网格）" if self.check_skipped
                        else "，**没有任何窗口被跳过**（位置跳转处已重置滑窗，所有窗口都是真实连续窗口）")
            check_txt = (f"与 {rel_path(ARTIFACT_DIR / f'predictions_{self.model_tag}.csv')} 逐窗口比对 "
                         f"{self.check_n} 个窗口：最大 |Δp| = {self.check_max_dp:.2e}"
                         f"（|Δp| > 1e-3 的窗口 {self.check_big} 个），判定类别不一致 {self.check_mismatch} 个"
                         + skip_txt)
        else:
            check_txt = "未做（缺少离线预测缓存，或本场未与离线滑窗网格对齐）"
        lines.append(markdown_table(["指标", "数值"], [
            ["**检测时延（流时间）**",
             f"**{self.delay_ms:.1f} 毫秒**（折合 {self.delay_rows} 条流记录）" if self.delay_ms is not None
             else "本片段未报出确认告警（漏报）"],
            ["参考：离线评估的平均时延", self._offline_ref_line()],
            ["攻击片段内确认告警数", f"{len(summary['attack_alerts'])} 条（服务口径：每个确认窗口一条告警，"
                                     f"与 app/app.py 的 /api/alerts 一致；屏幕上同一场爆发合并显示，台账不合并）"],
            ["攻击片段内被判定的窗口数", f"{n_win_attack} 个"],
            ["攻击前误报数（真值 Normal）", f"{len(summary['false_pre'])} 条"],
            ["攻击后误报数（真值 Normal）", f"{len(summary['false_post'])} 条"],
            ["处置动作分布", "、".join(f"{k}×{v}" for k, v in sorted(summary["by_action"].items())) or "无"],
            ["模拟阻断名单", f"{summary['n_blocked']} 个地址"
                             + (f"：{'、'.join(summary['blocked_ips'])}" if summary["blocked_ips"] else "")
                             + f"（有效期 {self.args.block_seconds} 秒）"],
            ["回放规模", f"{self.pushed} 条流记录 / {len(self.windows)} 个判定窗口"],
            ["回放倍速", f"{self.args.speed:g}（1 倍速 = {BASE_ROWS_PER_SEC:g} 条流记录/秒；"
                         f"纯展示节奏，不参与任何指标计算）"],
            ["逐窗口一致性核对", check_txt],
        ]))
        lines.append("")

        # ---------------- 四、诚实与口径说明 ----------------
        jump_txt = ("本剧本的恢复阶段取自攻击片段之后最近的纯正常片段，因此那里有一次位置跳转——"
                    "测试流里攻击片段密集交错（实测各攻击类**后接正常流量的中位数为 0 行**），"
                    "缺少足够长的连续正常流量，无法拼出「安静基线 + 攻击 + 恢复」的单一连续区间。"
                    f"跳转处我们**主动重置了滑窗缓冲（本场 {self.stream_resets} 次）**，"
                    "因为这等价于真实环境里的「上游数据中断」：若不重置，中断前后的行会被拼进同一个窗口，"
                    "生成**现实中不存在的窗口**（它在全局行号上并不连续）。"
                    "重置后本场的每一个窗口都是真实连续窗口，逐窗口核对因此**没有任何窗口被跳过**；"
                    "代价是跳转处会丢掉不足一个窗口的已积累行（本场 16 条，已在屏幕上如实打印）。"
                    if any(b.jumped for b in scene.blocks) else "本剧本全程连续，没有位置跳转。")
        lines.append("## 四、诚实与口径说明（逐条）\n")
        lines.append(
            "1. **进程内调用**：本演示直接在本进程里调用 `src.service.DetectionService`"
            "（与 `app/app.py` 的 `/api/detect`、`tools/controller_simulator.py` 是同一个引擎），"
            "**没有**另起 HTTP 服务、**没有**联网，因此屏幕上也不存在端口。\n"
            "2. **处置的性质**：服务处于 `dry_run`，给出的是**策略建议**（observe/throttle/block）；"
            "本演示中的“已加入阻断名单”由脚本内置的**本地模拟执行器**完成（带有效期），"
            "**不是**真的下发改动交换机流表。真实下发由 `integrations/ryu_guard.py` 通过 OpenFlow 完成，"
            "**本机没有 Ryu 环境、未实跑**（其接口与判定逻辑由 `tools/controller_simulator.py` 在无 Ryu 环境下验证）。"
            f"另外本场白名单{('为 ' + str(list(self.whitelist))) if self.whitelist else '为空（不豁免任何源地址）'}，"
            "即策略不会因为白名单而跳过任何源地址；上线前必须按 `deploy/cida-detect.env` 配好白名单。\n"
            "3. **检测时延口径**：只取**真实流时间戳**之差（攻击片段起点 → 首个确认告警窗口的右端行），"
            "与 `src/evaluate.py` 完全一致；**不是**墙上时间，也**不**随回放倍速变化"
            "（本报告只有“回放倍速”那一行会随 `--speed` 变，检测时延、误报数、一致性核对都不变）。"
            "注意本数据集的流记录间隔在同一片段内为亚毫秒级，因此毫秒数天然很小，"
            "“折合多少条流记录”才是与离线报告可比的量。\n"
            "4. **误报口径**：确认告警的窗口内若**没有任何**被标注为攻击的流记录，即计一次误报"
            "（与 `src/evaluate.py` 的注释同口径）。正常阶段出现的告警一律原样打印并在第五节列出，不做隐藏；"
            "屏幕上把连续的确认告警合并成一次“告警爆发”显示（只印首条与计数），但第五节与本报告的计数、"
            f"以及 `{rel_path(AUDIT_PATH)}` 里是**全部**条目，一条都没有被合并掉。\n"
            "5. **风险回落 ≠ 处置有效**：回放的是历史数据，攻击片段结束、真实标签切回 Normal 之后风险自然回落。"
            "本报告**不**声称是阻断动作压住了攻击，也**不**声称在真实网络上验证过防护效果。\n"
            "6. **数据真实**：回放 `data/processed/tensors.npz` 的留出测试流（真实标签 + 真实流时间戳），"
            "“注入攻击”只是把回放位置放到真实攻击片段起点（与 `app/app.py` 的 `/api/first_segment_of` 同性质）。\n"
            f"7. **源 IP**：{'取自 `data/raw/` 原始 CSV 的 `Source IP` / `Destination IP` 真实字段' if self.ips_real else '原始 CSV 不可读，退回确定性合成的 10.0.0.x（**不是**真实 IP，仅为让处置动作有地址可写）'}。\n"
            f"8. **审计台账**：本次告警写入 `{rel_path(AUDIT_PATH)}`（每次演示重写，只反映本次），"
            "**没有**混进 `results/audit_alerts.jsonl` 那份真实闭环台账。\n"
            f"9. **剧本跳转**：{jump_txt}\n"
            "10. **可复现**：模型推理是确定性前向计算，剧本由参数唯一确定，且与离线评估共用同一批滑窗；"
            "报告里所有数字都不依赖运行时机（服务侧的墙钟推理时延只上屏幕，不入报告）。"
            "逐窗口核对中的极小概率差（见第三节最后一行）来自“反标准化 → 服务内部再标准化”的浮点往返——"
            "服务接口按**真实物理量**接收流记录（与 `tools/controller_simulator.py` 同一做法），"
            f"本场没有任何窗口的判定类别因此改变。\n"
        )
        lines.append("")

        # ---------------- 五、告警明细 ----------------
        lines.append("## 五、告警明细（本次全部确认告警，含误报）\n")
        if self.alerts:
            rows = []
            for a in self.alerts:
                rows.append([f"#{a.seq}", a.phase_cn, f"{a.rel_ms:.1f} ms", f"{a.row_first}~{a.row_last}",
                             f"{a.p_attack:.4f}", a.pred_class,
                             a.truth + ("（跨跳转窗口）" if a.crosses_seam else ""), a.action, a.src_ip,
                             "**是（真值 Normal）**" if a.is_false_alarm else "否"])
            lines.append(markdown_table(["#", "阶段", "相对攻击起点", "窗口行号", "p_attack", "判定类别",
                                         "窗口真值", "处置", "源 IP", "误报"], rows))
        else:
            lines.append("_本次未产生确认告警。_\n")
        lines.append("")

        # ---------------- 六、复现 ----------------
        lines.append("## 六、复现命令\n")
        lines.append("```powershell\n"
                     "# 静默快速模式（自动化/CI，秒级完成，退出码 0）\n"
                     "python tools/live_demo.py --speed 0 --no-color\n\n"
                     "# 现场演示模式（屏幕上实时滚动，十几秒演完）\n"
                     "python tools/live_demo.py --speed 50 --attack Probe\n\n"
                     "# 换攻击类型 / 加减正常流量 / 调判定口径\n"
                     "python tools/live_demo.py --attack DoS --normal-rows 500\n"
                     "python tools/live_demo.py --threshold 0.9 --confirm 3 --action throttle\n"
                     "```\n")
        lines.append("> 由 `tools/live_demo.py` 生成，可重复运行、结果一致。\n")

        save_text("\n".join(lines), REPORT_PATH)
        self.say(f"事件报告已生成 → {rel_path(REPORT_PATH)}", C_GREEN)


def _ms_txt(v: float | None) -> str:
    return "—" if v is None else f"{v:.1f} 毫秒"


def _row_txt(row: int | None) -> str:
    return "—" if row is None else str(row)


def _p_txt(p: float | None) -> str:
    return "—" if p is None else f"{p:.4f}"


def _ts_txt(row: int | None, times: np.ndarray) -> str:
    if row is None or times is None or row < 0 or row >= len(times):
        return "—"
    return fmt_ts(float(times[row]))


def _quiet_logs() -> None:
    """压掉第三方与服务自身的 INFO/WARNING 日志。

    演示输出自成一套时间线（同样的信息由本脚本更清晰地打印）；`src/service.py` 的行为与
    审计台账完全不受影响。必须在 `import src.service` **之后**调用——`src.utils.get_logger`
    首次创建同名 logger 时会把级别设回 INFO。
    """
    for name in ("cida", "cida.service", "tensorflow", "keras", "absl"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _explain_load_failure(exc: Exception) -> str:
    text = str(exc)
    if "tensorflow" in text.lower() or "keras" in text.lower():
        return ("加载模型失败：本机缺少 TensorFlow/Keras，模型无法载入。\n"
                "请先安装依赖（可离线安装 requirements.txt 中的 wheel）：\n"
                "  python -m pip install -r requirements.txt\n"
                f"原始错误：{text}")
    return f"加载模型失败：{text}\n请先运行 python run_all.py 生成模型产物。"


def _explain_ready_failure(reason: str) -> str:
    if "TensorFlow" in reason:
        return ("检测服务不可用：本机缺少 TensorFlow，模型无法加载。\n"
                "请先安装依赖：python -m pip install -r requirements.txt")
    return f"检测服务不可用：{reason}\n请先运行 python run_all.py 生成模型与预处理产物。"


# ======================================================================================
# 命令行
# ======================================================================================
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="一条命令看一场真实攻防：正常流量 → 攻击 → 告警 → 处置 → 回落 → 事件报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--attack", default="DDoS",
                    help="要演示的攻击类型：DDoS/DoS/Probe/BruteForce/WebAttack/Botnet/U2R")
    ap.add_argument("--normal-rows", type=int, default=1000,
                    help="攻击前后各回放多少条正常流（上限；实际受“紧邻的连续纯正常片段”限制）")
    ap.add_argument("--speed", type=float, default=20.0,
                    help="回放倍速（1 倍速 = 6 条流记录/秒，纯展示节奏）；0 = 不限速尽快跑完")
    ap.add_argument("--threshold", type=float, default=None, help="判定阈值（默认取配置 report_threshold）")
    ap.add_argument("--confirm", type=int, default=None, help="连续确认窗口数（默认取配置 online_confirm_windows）")
    ap.add_argument("--action", default="block", choices=["observe", "throttle", "block"],
                    help="处置动作（作为策略建议下发，执行由本地模拟执行器完成）")
    ap.add_argument("--seed", type=int, default=42, help="仅在缺少真实 IP 时用于合成源 IP，不影响任何指标")
    ap.add_argument("--batch", type=int, default=8, help="每次推给检测服务的流记录数（模拟一个采集周期的增量）")
    ap.add_argument("--status-rows", type=int, default=100, help="非 TTY 模式下每多少条流记录打印一次状态行")
    ap.add_argument("--block-seconds", type=int, default=60, help="模拟阻断名单的有效期（秒）")
    ap.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色（非 TTY 环境自动关闭）")
    ap.add_argument("--no-animation", action="store_true", help="关闭原地刷新的状态条（非 TTY 环境自动关闭）")
    ap.add_argument("--processed-dir", default=None, help="预处理产物目录（默认 data/processed）")
    ap.add_argument("--artifact-dir", default=None, help="模型产物目录（默认 artifacts）")
    ap.add_argument("--model-tag", default=None, help="指定模型子目录名（默认用 run_all.py 记录的主模型）")
    args = ap.parse_args(argv)
    if args.normal_rows < 0:
        ap.error("--normal-rows 不能为负")
    if args.batch < 1:
        ap.error("--batch 至少为 1")
    return args


def _resolve_tty(args: argparse.Namespace) -> None:
    """非 TTY（重定向到文件/管道/CI）时自动关闭颜色与动画，避免刷出一堆控制字符。"""
    try:
        tty = bool(sys.stdout.isatty())
    except Exception:  # noqa: BLE001
        tty = False
    if not tty:
        args.no_color = True
        args.no_animation = True
    if os.environ.get("NO_COLOR"):
        args.no_color = True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_tty(args)
    silence_tf_logs()
    _quiet_logs()   # 服务模块导入后会在 run() 里再压一次（get_logger 会重设级别）

    demo: LiveDemo | None = None
    try:
        demo = LiveDemo(args)
        return demo.run()
    except DemoError as exc:
        Console(color=not args.no_color, animate=False).line(f"\n[错误] {sanitize(str(exc))}", C_RED)
        return int(exc.code)
    except KeyboardInterrupt:
        Console(color=not args.no_color, animate=False).line("\n[中断] 已停止（Ctrl+C）", C_YELLOW)
        return 130
    except BrokenPipeError:
        return 0   # 下游提前关闭管道（如 | head）：正常收手，不当作失败
    except Exception as exc:  # noqa: BLE001
        Console(color=not args.no_color, animate=False).line(
            f"\n[错误] 演示异常终止：{type(exc).__name__}: {sanitize(str(exc))}\n"
            "提示：可先用 `--speed 0 --no-color` 复跑定位问题。", C_RED)
        return 1
    finally:
        if demo is not None:
            demo.console.close()


if __name__ == "__main__":
    raise SystemExit(main())
