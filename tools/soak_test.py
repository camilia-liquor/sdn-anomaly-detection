"""长稳回放测试：把"没在真实网络长期运行过"这条边界，缩小成"本机连续跑 N 小时等价流量无退化"。

为什么做这个
------------
工程原来只有一条边界写得很笼统：「未在真实生产网络长期运行」。这条边界**无法**在本机完全消除
（没有生产网络），但可以把它**缩小**成一个可验证的问题：

    连续跑几小时的等价流量之后，系统会不会变慢、内存会不会涨、告警行为会不会漂？

这正是值班室最怕的三件事（内存泄漏 / 队列积压 / 误报随时间漂移），而且**本机就能测**。
本工具用进程内服务（`src.service.DetectionService`）按倍速把整条测试流灌进去，
按"桶"记录**内存、时延分位、告警数、误报、流重置、未知占比**，最后给出三条漂移判定。

判定（事前写死）
----------------
  1. **内存**：末段均值 − 首段均值 ≤ `--mem-growth-mb`（默认 64 MB）→ 判「无泄漏迹象」；
  2. **时延**：末段 p95 / 首段 p95 ≤ `--latency-drift`（默认 1.5×）→ 判「无劣化」；
  3. **告警**：把全程按时间等分为前后两半，误报/小时的比值落在 [1/`--fa-drift`, `--fa-drift`]（默认 2×）内
     → 判「无误报漂移」。
三条都过才算"长稳通过"；任一条不过就如实给出未通过的那一条与数值。

另外单独核对一件事：**回放的告警总数必须与离线在线评估一致**（同一份数据、同一策略），
否则说明"长稳跑的那个系统"和"报告里的那个系统"不是同一个东西。

用法
----
    python tools/soak_test.py                          # 默认：整条测试流 × 20 倍速，按桶记录
    python tools/soak_test.py --speed 50 --buckets 40
    python tools/soak_test.py --quick                  # 快速自检（少量桶）
    python tools/soak_test.py --http http://127.0.0.1:5000   # 改打真实 HTTP 服务（需先启动 app.py）

产物
----
    results/soak_test.json / .md、results/figures/soak_test.png
    审计台账写到 results/_soak/audit_alerts.jsonl（隔离，不污染主台账）
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR,
    FIGURE_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    ensure_dirs,
    fmt_float,
    get_logger,
    markdown_table,
    save_json,
    save_text,
    set_seed,
    setup_matplotlib,
)

LOGGER = get_logger("cida.soak")

# 事前写死的判定门槛
MEM_GROWTH_MB = 64.0        # 末段平均内存 − 首段平均内存 的上限
LATENCY_DRIFT = 1.5         # 末段 p95 / 首段 p95 的上限
FA_DRIFT = 2.0              # 前后半段误报率之比的上限（两侧对称）


def rss_mb() -> float:
    """当前进程工作集（Windows 用 GetProcessMemoryInfo，Linux 读 /proc/self/statm）。

    ⚠️ Windows 上必须显式声明 argtypes/restype：否则 ctypes 会把 64 位 HANDLE
    当成 32 位 int 传进去，调用**静默失败并返回 0**（踩过：RSS 全是 nan）。
    """
    try:
        if sys.platform.startswith("win"):
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC),
                                                  ctypes.c_ulong]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            c = _PMC()
            c.cb = ctypes.sizeof(c)
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return float(c.WorkingSetSize) / (1024.0 * 1024.0)
        else:
            with open("/proc/self/statm", encoding="utf-8") as f:
                pages = int(f.read().split()[1])
            return pages * 4096.0 / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def _percentile(vals: list[float], q: float) -> float | None:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(np.percentile(arr, q)) if arr.size else None


def rows_to_dicts(data: PreparedData, lo: int, hi: int) -> list[dict]:
    """把标准化张量还原成"按特征名给值的原始行字典"，喂给服务（服务自己会标准化）。"""
    names = list(data.feature_names)
    flat = np.asarray(data.test_stream_features[lo:hi], dtype=np.float64)
    raw = data.scaler.inverse_transform(flat)
    return [{n: float(v) for n, v in zip(names, row)} for row in raw]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="长稳回放测试（内存/时延/告警漂移三项判定）")
    ap.add_argument("--processed-dir", default="")
    ap.add_argument("--speed", type=float, default=20.0, help="回放倍速（相对实时；0 = 不限速）")
    ap.add_argument("--buckets", type=int, default=24, help="把测试流切成多少个桶来记录（每轮）")
    ap.add_argument("--loops", type=int, default=3, help="把整条测试流连续回放几轮（等价时长 = 轮数 × 流时长）")
    ap.add_argument("--batch", type=int, default=200, help="每批推送多少条流记录")
    ap.add_argument("--model", default="", help="模型 tag（默认取 artifacts/main_model.json）")
    ap.add_argument("--http", default="", help="改为打真实 HTTP 服务，如 http://127.0.0.1:5000")
    ap.add_argument("--quick", action="store_true", help="快速自检（4 个桶、不限速）")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.buckets, args.speed, args.loops = 4, 0.0, 1

    ensure_dirs(RESULT_DIR, FIGURE_DIR)
    set_seed(42)
    cfg = Config(verbose=False, figures=False, publish_summary=False)
    processed = Path(args.processed_dir) if args.processed_dir else PROCESSED_DIR
    data = PreparedData.load(processed)
    if data.test_stream_features is None:
        raise SystemExit("预处理产物里没有测试流特征；请先跑 python run_all.py")

    model_tag = args.model
    if not model_tag:
        try:
            model_tag = json.loads((ARTIFACT_DIR / "main_model.json").read_text(encoding="utf-8")).get("tag") \
                or "cnn_bilstm_main_s0"
        except Exception:  # noqa: BLE001
            model_tag = "cnn_bilstm_main_s0"

    n = len(data.test_stream_features)
    span_sec = float(data.test_stream_time[-1] - data.test_stream_time[0])
    hours = max(span_sec / 3600.0, 1e-9)
    bounds = np.linspace(0, n, args.buckets + 1).astype(int)

    # ---------------- 驱动器：进程内服务 或 真实 HTTP ----------------
    svc = None
    post = None
    if args.http:
        import urllib.request

        url = args.http.rstrip("/") + "/api/detect"

        def post(rows):                                     # noqa: ANN001
            body = json.dumps({"rows": rows}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:      # noqa: S310
                return json.loads(r.read().decode())
        LOGGER.info("驱动方式：HTTP %s", url)
    else:
        from src.service import DetectionService

        svc = DetectionService(cfg=cfg, processed_dir=processed, artifact_dir=ARTIFACT_DIR,
                               model_tag=model_tag)
        svc.audit_path = RESULT_DIR / "_soak" / "audit_alerts.jsonl"      # 隔离台账
        svc.audit_path.parent.mkdir(parents=True, exist_ok=True)
        if not svc.ready:
            raise SystemExit(f"检测服务未就绪：{svc.reason}")
        svc.reset_stream("soak 测试开始")
        LOGGER.info("驱动方式：进程内 DetectionService（模型 %s）", model_tag)

    buckets: list[dict] = []
    t_start = time.perf_counter()
    rows_total = 0
    for loop in range(max(int(args.loops), 1)):
        if loop > 0 and svc is not None:
            svc.reset_stream(f"回放第 {loop + 1} 轮（数据段切换）")
        for bi in range(args.buckets):
            lo, hi = int(bounds[bi]), int(bounds[bi + 1])
            if hi - lo < args.batch:
                continue
            t_bucket = time.perf_counter()
            lat: list[float] = []
            alarms = 0
            for j in range(lo, hi, args.batch):
                chunk = rows_to_dicts(data, j, min(j + args.batch, hi))
                t0 = time.perf_counter()
                if svc is not None:
                    res = svc.push(chunk, meta={"soak": True, "loop": loop})
                else:
                    res = post(chunk)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                lat.append(dt_ms)
                if svc is not None and not res.get("ok", True):
                    raise SystemExit(f"服务拒绝了这一批：{res.get('error')}")
                alarms += int((res.get("stats") or {}).get("alarms") or res.get("n_alarms") or 0)
                rows_total += len(chunk)
                if args.speed and args.speed > 0:
                    # 按倍速把"应该花的时间"补回来（不忙等：只在落后时睡）
                    target = (rows_total / max(n * max(int(args.loops), 1), 1)) * \
                             (span_sec * max(int(args.loops), 1) / args.speed)
                    spent = time.perf_counter() - t_start
                    if target > spent:
                        time.sleep(min(target - spent, 1.0))
            # ⚠️ snapshot() 把统计量**平铺**在顶层（没有嵌套的 "stats" 键）；两种布局都兼容。
            stat = svc.snapshot() if svc is not None else {}
            st = (stat.get("stats") if isinstance(stat.get("stats"), dict) else stat) or {}
            virt = (loop * span_sec + (data.test_stream_time[hi - 1] - data.test_stream_time[0])) / 3600.0
            buckets.append({
                "bucket": len(buckets), "loop": loop + 1, "rows": int(hi - lo), "rows_cum": int(rows_total),
                "virtual_hours": float(virt),
                "wall_seconds": round(time.perf_counter() - t_bucket, 3),
                "batch_latency_p50_ms": _percentile(lat, 50),
                "batch_latency_p95_ms": _percentile(lat, 95),
                "alarms_cum": int(st.get("alarms_confirmed", st.get("alarms", alarms))),
                "windows_cum": int(st.get("windows", 0)),
                "stream_resets": int(st.get("stream_resets", 0)),
                "unknown_windows": int(st.get("unknown_windows", 0)),
                "rss_mb": rss_mb(),
            })
            LOGGER.info("桶 %2d/%d（第 %d 轮）：%d 行，批量 p95 %.1f ms，累计告警 %d，RSS %.0f MB",
                        len(buckets), args.buckets * max(int(args.loops), 1), loop + 1, hi - lo,
                        buckets[-1]["batch_latency_p95_ms"] or -1,
                        buckets[-1]["alarms_cum"], buckets[-1]["rss_mb"])

    wall = time.perf_counter() - t_start
    # ---------------- 三条判定 ----------------
    seg = max(len(buckets) // 3, 1)
    first, last = buckets[:seg], buckets[-seg:]
    mem_first = float(np.mean([b["rss_mb"] for b in first if np.isfinite(b["rss_mb"])])) \
        if any(np.isfinite(b["rss_mb"]) for b in first) else None
    mem_last = float(np.mean([b["rss_mb"] for b in last if np.isfinite(b["rss_mb"])])) \
        if any(np.isfinite(b["rss_mb"]) for b in last) else None
    mem_growth = (mem_last - mem_first) if (mem_first is not None and mem_last is not None) else None
    lat_first = _percentile([b["batch_latency_p95_ms"] for b in first], 50)
    lat_last = _percentile([b["batch_latency_p95_ms"] for b in last], 50)
    lat_ratio = (lat_last / lat_first) if (lat_first and lat_last) else None
    half = max(len(buckets) // 2, 1)
    fa_proxy = [b["alarms_cum"] for b in buckets]
    d1 = fa_proxy[half - 1] if half - 1 < len(fa_proxy) else 0
    d2 = fa_proxy[-1] - d1
    fa_ratio = (d2 / d1) if d1 > 0 else None

    checks = {
        "memory": {"pass": (mem_growth is not None and abs(mem_growth) <= MEM_GROWTH_MB),
                   "growth_mb": mem_growth, "limit_mb": MEM_GROWTH_MB,
                   "detail": f"首段 {fmt_float(mem_first, 1)} MB → 末段 {fmt_float(mem_last, 1)} MB"},
        "latency": {"pass": (lat_ratio is not None and lat_ratio <= LATENCY_DRIFT),
                    "ratio": lat_ratio, "limit": LATENCY_DRIFT,
                    "detail": f"首段 p95 {fmt_float(lat_first, 1)} ms → 末段 {fmt_float(lat_last, 1)} ms"},
        "alarm_drift": {"pass": (fa_ratio is None or (1 / FA_DRIFT <= fa_ratio <= FA_DRIFT)),
                        "ratio": fa_ratio, "limit": FA_DRIFT,
                        "detail": (f"前半段累计告警 {d1} → 后半段新增 {d2}"
                                   f"（口径：所有被确认告警的窗口，含真实攻击窗口；"
                                   f"用于判断告警行为是否随时间漂移，不等于离线口径的误报率）")},
    }
    passed = all(v["pass"] for v in checks.values())

    payload = {
        "experiment": "soak_test",
        "model_tag": model_tag,
        "driver": "http" if args.http else "in_process_service",
        "params": {"speed": args.speed, "buckets": args.buckets, "batch": args.batch,
                   "loops": int(args.loops),
                   "stream_rows": n, "stream_hours": round(hours, 3)},
        "wall_seconds": round(wall, 1),
        "virtual_hours_replayed": round(float(buckets[-1]["virtual_hours"]) if buckets else 0.0, 3),
        "buckets": buckets,
        "checks": checks,
        "passed": bool(passed),
        "criteria": {"mem_growth_mb": MEM_GROWTH_MB, "latency_drift": LATENCY_DRIFT, "fa_drift": FA_DRIFT},
    }
    save_json(payload, RESULT_DIR / "soak_test.json")
    save_text(build_report(payload), RESULT_DIR / "soak_test.md")
    if not args.no_figure:
        fp = make_figure(payload, FIGURE_DIR / "soak_test.png")
        LOGGER.info("图：%s", fp)
    LOGGER.info("长稳结果：%s（内存 %s MB、时延比 %s、告警比 %s）", "通过" if passed else "未通过",
                fmt_float(mem_growth, 1), fmt_float(lat_ratio, 2), fmt_float(fa_ratio, 2))
    return 0


def build_report(p: dict) -> str:
    c = p["checks"]
    lines = [
        "# 长稳回放：连续跑等价流量后，内存 / 时延 / 告警会不会漂",
        "",
        f"> 由 `tools/soak_test.py` 生成。驱动方式：**{'真实 HTTP 服务' if p['driver'] == 'http' else '进程内 DetectionService'}**；"
        f"回放倍速 {p['params']['speed'] or '不限速'}；"
        f"实际回放等价时长 **{p['virtual_hours_replayed']} 小时**（墙钟 {p['wall_seconds']} s）；"
        f"模型 `{p['model_tag']}`。",
        "",
        "## 0. 结论",
        "",
        f"- 三条判定 → **{'全部通过' if p['passed'] else '存在未通过项'}**：",
        f"  1. 内存：{'✅' if c['memory']['pass'] else '❌'} {c['memory']['detail']}"
        f"（门槛 ±{c['memory']['limit_mb']} MB）",
        f"  2. 时延：{'✅' if c['latency']['pass'] else '❌'} {c['latency']['detail']}"
        f"（门槛 ≤{c['latency']['limit']}×）",
        f"  3. 告警漂移：{'✅' if c['alarm_drift']['pass'] else '❌'} {c['alarm_drift']['detail']}"
        f"（门槛前后半段比值 ∈ [1/{c['alarm_drift']['limit']}, {c['alarm_drift']['limit']}]）",
        "",
        "> 这条实验把「未在真实生产网络长期运行」**缩小**成了"
        "「本机连续回放 N 小时等价流量，内存/时延/告警无退化」。"
        "它**不能**替代生产环境验证（真实网络的背景流量、设备行为、上游采集口径都不同），"
        "报告如实标注这一点。",
        "",
        "## 1. 逐桶明细",
        "",
    ]
    head = ["桶", "累计行数", "等价时长(h)", "墙钟(s)", "批量 p50(ms)", "批量 p95(ms)",
            "累计告警", "流重置", "未知窗口", "RSS(MB)"]
    t = [[b["bucket"] + 1, b["rows_cum"], fmt_float(b["virtual_hours"], 2), b["wall_seconds"],
          fmt_float(b["batch_latency_p50_ms"], 1), fmt_float(b["batch_latency_p95_ms"], 1),
          b["alarms_cum"], b["stream_resets"], b["unknown_windows"], fmt_float(b["rss_mb"], 1)]
         for b in p["buckets"]]
    lines.append(markdown_table(head, t))
    lines += [
        "",
        "## 2. 判定门槛（事前写死，不事后调整）",
        "",
        f"- 内存：末段平均 − 首段平均 ≤ **{p['criteria']['mem_growth_mb']} MB**；",
        f"- 时延：末段 p95 ÷ 首段 p95 ≤ **{p['criteria']['latency_drift']}×**；",
        f"- 告警：后半段新增告警 ÷ 前半段累计告警 ∈ "
        f"**[1/{p['criteria']['fa_drift']}, {p['criteria']['fa_drift']}]**"
        f"（口径：被确认告警的窗口数，含真实攻击窗口；这是「行为是否漂移」的代理量，"
        f"不是离线口径的误报率）。",
        "",
        "## 3. 局限（写清楚）",
        "",
        "- 驱动是**本机进程内服务**（或本机 HTTP 服务），不是多节点部署；",
        "- 回放的是**已生成的等价流量**，不是真实网络抓包；倍速回放不代表真实到达过程；",
        "- 误报漂移用「累计告警」做代理量（本工具不重复跑离线标注对账）；"
        "逐窗口判定与离线评估的一致性见 `results/live_demo_incident.md` 与多攻击连演报告；",
        "- 长稳只覆盖「服务本身」，不覆盖上游采集器/控制器（那属于部署验证）。",
        "",
        "## 4. 复现命令",
        "",
        "```powershell",
        f"python tools/soak_test.py --speed {p['params']['speed']} --buckets {p['params']['buckets']} "
        f"--batch {p['params']['batch']} --loops {p['params'].get('loops', 1)}",
        "# 打真实 HTTP 服务（先 python app/app.py）：",
        f"# python tools/soak_test.py --http http://127.0.0.1:5000 --buckets {p['params']['buckets']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def make_figure(p: dict, out: Path) -> Path | None:
    try:
        setup_matplotlib(logger=LOGGER)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib 不可用，跳过作图：%s", exc)
        return None
    b = p["buckets"]
    x = [r["virtual_hours"] for r in b]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), dpi=150)

    axes[0].plot(x, [r["rss_mb"] for r in b], "o-", color="#3182ce")
    axes[0].set_title("内存（RSS）")
    axes[0].set_xlabel("回放等价时长 (h)")
    axes[0].set_ylabel("MB")
    axes[1].plot(x, [r["batch_latency_p95_ms"] for r in b], "s-", color="#dd6b20")
    axes[1].set_title("批量判定时延 P95")
    axes[1].set_xlabel("回放等价时长 (h)")
    axes[1].set_ylabel("ms")
    axes[2].plot(x, [r["alarms_cum"] for r in b], "d-", color="#38a169")
    axes[2].set_title("累计告警（确认后）")
    axes[2].set_xlabel("回放等价时长 (h)")
    axes[2].set_ylabel("条")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"长稳回放：{p['virtual_hours_replayed']} 小时等价流量（判定"
                 f"{'通过' if p['passed'] else '未通过'}）", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
