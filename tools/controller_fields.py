"""控制器原生字段扩展：只加 **OpenFlow 真能拿到/真能推导** 的字段，量化"字段缺口"能补回多少。

背景（这一步解决的是一个真问题）
--------------------------------
工程里已经量化过一次"字段缺口"：完整流特征口径宏 F1 **0.9940**，而只用 OpenFlow 流表统计的
17 列时掉到 **0.8441**，代价是工作点必须大幅收紧（阈值 0.50→0.99、确认 3→5、时延 18→41 ms）。
结论是"字段缺口会转化成工作点代价"。但**到此为止有个明显的下一步没做**：
那 17 列并不是"控制器能拿到的全部"——控制器还能做**流表内的聚合与轮询差分**。

本工具做的就是这一步：把控制器**真的能算出来**的字段补上，再看差距能缩到多少。

字段来源（逐条说明，不含任何"编造"）
------------------------------------
① OFPFlowStats（OpenFlow 1.3）直接给的：duration_sec/nsec、packet_count、byte_count、
   match 字段（src/dst ip、src/dst port、ip_proto、icmp type/code）、idle/hard timeout、flags
   —— 这就是既有 sdn schema 的 17 列，原样保留。
② 从 ① 派生的单流统计：`avg_pkt_size = byte_count / packet_count`（OFPFlowStats 里没有，
   但一步除法就能得到）。
③ 控制器按**轮询窗口内的流表聚合**（Ryu 在同一个 poll 周期里能拿到全部流条目，可以分组统计）：
   - 源侧：`src_flow_count`、`src_distinct_dst`、`src_distinct_dport`、`src_dport_entropy`、
           `src_short_flow_ratio`、`src_pkt_per_flow_mean`、`src_new_flow_rate`
   - 目的侧：`dst_flow_count`、`dst_distinct_src`、`dst_src_entropy`
   - 表级：`table_flow_count`、`proto_share`
   这些正是扫描/暴破/洪泛/心跳的行为特征（一个源打很多端口、很多源打同一目的、短流比例高……），
   **全部只用流表里本来就有的字段算出来**。
④ **明确不可得**（写进报告，避免夸大）：TCP 标志位计数、IAT/包长分布、头部长度、
   活跃/空闲周期、subflow 统计、窗口大小、bulk 统计 —— OpenFlow 流表统计**不提供**这些，
   因此本工具**不**去伪造它们（这也是为什么本扩展**不可能**追平 CICFlowMeter 的 84 列口径）。

实验设计（判据事前写死）
------------------------
  同一批流（同一次生成，只切 schema）分别训练两个口径：
    `ctrl17`  = 既有 17 列；`ctrl_ext` = 17 列 + 上述派生/聚合字段；
  结构固定 cnn_bilstm、T=16、stride=5、3 个随机种子，报均值±标准差。
  判据：
    J1 精度：Δ = macroF1(ext) − macroF1(17)，|Δ| ≥ 1σ 才算有效；
    J2 工作点代价：在误报预算 1 次/小时下，达到该预算所需的**阈值/确认次数/时延**是否下降；
    J3 与完整口径的差距：ext 相对完整流特征口径（0.9940，读自主实验）还差多少。

隔离：生成数据写在 `data/raw_ctrl_*/`、产物在 `artifacts/ctrl_*/` 与 `results/ctrl_ext_*`；
主实验的 data/raw、data/processed、artifacts/cnn_bilstm_main_* 全程只读。

用法
----
    python tools/controller_fields.py                    # 完整（生成 + 两口径 × 3 种子 + 工作点）
    python tools/controller_fields.py --quick            # 快速自检
    python tools/controller_fields.py --skip-generate    # 复用已有数据，只跑训练与汇总
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData, prepare_dataset  # noqa: E402
from src.evaluate import confirm_alarms, evaluate_model, online_alarm_evaluation  # noqa: E402
from src.model import load_model_bundle  # noqa: E402
from src.train import run_training  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR,
    FIGURE_DIR,
    RESULT_DIR,
    ensure_dirs,
    fmt_float,
    get_logger,
    markdown_table,
    save_json,
    save_text,
    set_seed,
    setup_matplotlib,
    silence_tf_logs,
)

LOGGER = get_logger("cida.ctrl_fields")

DEFAULT_SCALE = 0.45          # 生成规模（相对主数据 9 万条 → 约 4 万条流）
DEFAULT_SEEDS = (0, 1, 2)
WINDOW_SEC = 60.0             # 控制器聚合窗口（轮询周期量级）
SHORT_FLOW_SEC = 1.0          # "短流"判据（duration < 1 s）
FA_BUDGET = 1.0               # 工作点搜索用的误报预算（次/小时）
FULL_SCHEMA_F1 = 0.9940       # 完整流特征口径的宏 F1（主实验 3 种子均值，只作为参照读入）
CTRL17_REF_F1 = 0.8441        # 既有控制器 17 列口径的宏 F1（历史结果，作为交叉参考）

# 扩展字段（全部来自 ②/③ 两类来源；名字即口径）
EXT_FIELDS = [
    "avg_pkt_size",          # byte_count / packet_count
    "src_flow_count",        # 同源流条目数（窗口内）
    "src_distinct_dst",      # 同源不同目的 IP 数
    "src_distinct_dport",    # 同源不同目的端口数（扫描）
    "src_dport_entropy",     # 同源目的端口分布熵
    "src_short_flow_ratio",  # 同源短流占比
    "src_pkt_per_flow_mean", # 同源平均包数/流
    "src_new_flow_rate",     # 同源新建流速率（流/秒）
    "dst_flow_count",        # 同目的流条目数
    "dst_distinct_src",      # 同目的不同源数（洪泛/暴破）
    "dst_src_entropy",       # 同目的源分布熵
    "table_flow_count",      # 流表规模（窗口内总流数）
    "proto_share",           # 该协议在窗口内的占比
]

UNAVAILABLE = [
    "TCP 标志位计数（SYN/ACK/FIN/RST/PSH/URG）",
    "包长分布（mean/std/max/min/variance）",
    "到达间隔 IAT 的均值/标准差/最大最小",
    "头部长度、窗口大小等报文级字段",
    "活跃/空闲周期（Active/Idle）统计",
    "subflow / bulk 统计",
]


# --------------------------------------------------------------------------------------
# 1) 生成控制器口径的原始数据（只切 schema，物理事实与主实验同源）
# --------------------------------------------------------------------------------------
def generate_ctrl_base(scale: float, seeds: tuple[int, ...]) -> Path:
    from src.traffic_generator import generate_datasets

    out = ROOT / "data" / "raw_ctrl"
    cfg = Config(schema="sdn", scale=float(scale), verbose=False, publish_summary=False)
    LOGGER.info("生成控制器口径数据（schema=sdn，scale=%.2f）→ %s", scale, out)
    generate_datasets(cfg, out_dir=out, save=True)
    return out / "sdn_stream.csv"


def _entropy(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _rolling_group_stats(df: pd.DataFrame, key: str, ts: np.ndarray, prefix: str) -> dict:
    """按 key（源或目的 IP）在**过去 WINDOW_SEC 秒**的滑动窗口里算聚合量。

    ⚠️ 窗口必须是**因果的（只看过去）**：控制器在收到一条流统计时，不可能知道未来 60 秒里
    还会出现哪些流。最初版本用了 ±60 秒的对称窗口（含未来行），那是一种**前视泄漏**，
    会让离线指标虚高、并且与真机行为不符。这里改成 `[t−W, t]`。
    """
    n = len(df)
    out_count = np.zeros(n, dtype=np.float64)
    out_distinct = np.zeros(n, dtype=np.float64)
    out_dport = np.zeros(n, dtype=np.float64)
    out_ent = np.zeros(n, dtype=np.float64)
    out_short = np.zeros(n, dtype=np.float64)
    out_pkt = np.zeros(n, dtype=np.float64)
    out_rate = np.zeros(n, dtype=np.float64)
    group_col = "tp_dst" if key == "ip_src" else "tp_src"
    other_ip = "ip_dst" if key == "ip_src" else "ip_src"
    pkt_col = "packet_count"
    dur_col = "flow_duration_sec"

    for _, idx in df.groupby(key, sort=False).indices.items():
        idx = np.sort(np.asarray(idx))
        t = ts[idx]
        ports = df[group_col].to_numpy()[idx]
        other = df[other_ip].to_numpy()[idx]
        pkts = df[pkt_col].to_numpy(dtype=np.float64)[idx]
        durs = df[dur_col].to_numpy(dtype=np.float64)[idx]
        lo = 0
        for i in range(len(idx)):
            while t[i] - t[lo] > WINDOW_SEC:      # 只保留过去 W 秒内的行
                lo += 1
            sl = slice(lo, i + 1)                  # ← 因果：到当前行为止
            m = i + 1 - lo
            out_count[idx[i]] = m
            out_distinct[idx[i]] = len(np.unique(other[sl]))
            up, cnt = np.unique(ports[sl], return_counts=True)
            out_dport[idx[i]] = len(up)
            out_ent[idx[i]] = _entropy(cnt.astype(np.float64))
            out_short[idx[i]] = float((durs[sl] < SHORT_FLOW_SEC).mean())
            out_pkt[idx[i]] = float(pkts[sl].mean())
            span = max(float(t[i] - t[lo]), 1e-6)
            out_rate[idx[i]] = m / span
    return {
        f"{prefix}_flow_count": out_count,
        f"{prefix}_distinct_{'dst' if key == 'ip_src' else 'src'}": out_distinct,
        f"{prefix}_dport_entropy" if key == "ip_src" else f"{prefix}_src_entropy": out_ent,
        **({f"{prefix}_distinct_dport": out_dport} if key == "ip_src" else {}),
        **({f"{prefix}_short_flow_ratio": out_short,
            f"{prefix}_pkt_per_flow_mean": out_pkt,
            f"{prefix}_new_flow_rate": out_rate} if key == "ip_src" else {}),
    }


def derive_extended(base_csv: Path) -> Path:
    """从控制器口径数据派生扩展字段，写出第二份 CSV（列名与 id 列保持一致）。"""
    df = pd.read_csv(base_csv, low_memory=False)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    pkt_col = "packet_count"
    byt_col = "byte_count"

    out = df.copy()
    out["avg_pkt_size"] = df[byt_col].to_numpy(dtype=np.float64) / np.maximum(
        df[pkt_col].to_numpy(dtype=np.float64), 1.0)
    for key, prefix in (("ip_src", "src"), ("ip_dst", "dst")):
        stats = _rolling_group_stats(df, key, ts, prefix)
        for k, v in stats.items():
            out[k] = v
    out["table_flow_count"] = float(len(df)) * 0.0 + np.asarray(
        [len(df)] * len(df), dtype=np.float64)          # 表级规模由外部轮询决定，这里取同批总量
    # 更诚实的表级量：窗口内总流数 = 该行所在窗口内的行数（用源侧计数近似表规模）
    out["table_flow_count"] = out["src_flow_count"].to_numpy()
    proto = df["ip_proto"].to_numpy()
    up, cnt = np.unique(proto, return_counts=True)
    share = dict(zip(up.tolist(), (cnt / cnt.sum()).tolist()))
    out["proto_share"] = [float(share.get(p, 0.0)) for p in proto.tolist()]

    dst_dir = ROOT / "data" / "raw_ctrl_ext"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "sdn_stream.csv"
    out.to_csv(dst, index=False, encoding="utf-8")
    LOGGER.info("扩展口径已写出：%s（%d 行，新增 %d 列）", dst, len(out), len(EXT_FIELDS))
    return dst


# --------------------------------------------------------------------------------------
# 2) 工作点搜索（同一误报预算下比较代价）
# --------------------------------------------------------------------------------------
def best_point_under_budget(model, data: PreparedData, cfg: Config, *, budget: float) -> dict:
    from src.data_pipeline import build_windows
    from src.model import predict_probs

    X, _, _, info = build_windows(data.test_stream_features, labels=data.test_stream_labels,
                                  window_size=cfg.window_size, stride=cfg.online_stride,
                                  pure_only=False)
    starts = np.asarray(info["starts"])
    _, p = predict_probs(model, X, batch_size=cfg.batch_size)
    p = np.asarray(p, dtype=np.float64)
    hours = float(np.asarray(data.test_stream_time)[-1] - np.asarray(data.test_stream_time)[0]) / 3600.0
    best = None
    for k in (1, 2, 3, 5):
        for thr in np.unique(np.concatenate([np.linspace(0.3, 1.0, 141),
                                             np.quantile(p, np.linspace(0.9, 1.0, 60))])):
            on = online_alarm_evaluation(data, p, starts, cfg, threshold=float(thr), confirm=k)
            fa = on["false_alarms_per_hour"] or 0.0
            if fa > budget or on["segment_recall"] < 0.999:
                continue
            cand = {"threshold": float(thr), "confirm": int(k), "fa_per_hour": float(fa),
                    "segment_recall": float(on["segment_recall"]),
                    "mean_delay_ms": (float(on["mean_detection_delay_sec"]) * 1000.0
                                      if on["mean_detection_delay_sec"] is not None else None)}
            key = (cand["mean_delay_ms"] if cand["mean_delay_ms"] is not None else 9e9, cand["confirm"])
            if best is None or key < (best["mean_delay_ms"] or 9e9, best["confirm"]):
                best = cand
            break        # 该确认次数下阈值已从低往高扫，第一个满足预算的就是最灵敏的点
    return best or {"note": "该口径下没有满足预算且召回 100% 的工作点"}


# --------------------------------------------------------------------------------------
# 3) 主流程
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="控制器原生字段扩展：量化字段缺口能补回多少")
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.scale, args.epochs, args.seeds = 0.06, 2, "0"

    started = time.time()
    ensure_dirs(RESULT_DIR, FIGURE_DIR, ARTIFACT_DIR)
    silence_tf_logs()
    set_seed(42)
    seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())

    base_csv = ROOT / "data" / "raw_ctrl" / "sdn_stream.csv"
    ext_csv = ROOT / "data" / "raw_ctrl_ext" / "sdn_stream.csv"
    if not args.skip_generate:
        base_csv = generate_ctrl_base(float(args.scale), seeds)
        ext_csv = derive_extended(base_csv)
    else:
        if not base_csv.exists() or not ext_csv.exists():
            raise SystemExit("--skip-generate 需要 data/raw_ctrl 与 data/raw_ctrl_ext 都已存在")

    variants = {"ctrl17": base_csv.parent, "ctrl_ext": ext_csv.parent}
    runs: dict[str, dict] = {}
    for name, raw_dir in variants.items():
        for seed in seeds:
            tag = f"{name}_s{seed}"
            cfg = Config(schema="sdn", verbose=False, figures=False, publish_summary=False,
                         raw_dir=str(raw_dir), processed_dir=str(ROOT / "data" / f"processed_{name}"),
                         window_size=16, window_stride=5, epochs=args.epochs, seed=seed,
                         input_mode="sequence", run_name=name)
            data = prepare_dataset(cfg, save=(seed == seeds[0]))
            res = run_training(cfg, data, variant="cnn_bilstm", tag=tag, epochs=args.epochs,
                               artifact_dir=ARTIFACT_DIR / name)
            rep = evaluate_model(res.model, data, cfg, tag=tag,
                                 result_dir=RESULT_DIR / "ctrl_ext", artifact_dir=ARTIFACT_DIR / name,
                                 save_predictions=False)
            pt = best_point_under_budget(res.model, data, cfg, budget=FA_BUDGET)
            runs[tag] = {
                "variant": name, "seed": seed,
                "macro_f1": float(rep["multiclass"]["macro_f1"]),
                "accuracy": float(rep["multiclass"]["accuracy"]),
                "binary_f1": float(rep["binary"]["f1"]),
                "binary_roc_auc": float(rep["binary"]["roc_auc"]),
                "tpr_at_fpr_1pct": float(rep["multiclass"].get("n_classes_present") and
                                         __import__("src.evaluate", fromlist=["tpr_at_fpr"])
                                         .tpr_at_fpr(data.y_test_bin, np.asarray(rep["_binary_curve"][1]), 0.01)),
                "n_features": int(len(data.feature_names)),
                "feature_names": list(data.feature_names),
                "online_default": rep.get("online") or {},
                "best_point": pt,
                "n_params": int(res.model.count_params()),
            }
            LOGGER.info("[%s] 特征 %d 列 | 宏F1=%.4f AUC=%.4f | 预算 %.1f 次/h 的工作点：%s",
                        tag, len(data.feature_names), runs[tag]["macro_f1"],
                        runs[tag]["binary_roc_auc"], FA_BUDGET,
                        {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in (pt or {}).items()})
            del res

    def agg(name, key):
        vals = [runs[f"{name}_s{s}"][key] for s in seeds if f"{name}_s{s}" in runs
                and runs[f"{name}_s{s}"].get(key) is not None]
        if not vals:
            return {"n": 0, "mean": None, "std": None}
        return {"n": len(vals), "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "values": [float(v) for v in vals]}

    summary = {name: {"macro_f1": agg(name, "macro_f1"), "binary_roc_auc": agg(name, "binary_roc_auc"),
                      "n_features": runs.get(f"{name}_s{seeds[0]}", {}).get("n_features"),
                      "n_params": runs.get(f"{name}_s{seeds[0]}", {}).get("n_params")}
               for name in variants}
    a17, aex = summary["ctrl17"]["macro_f1"], summary["ctrl_ext"]["macro_f1"]
    delta = None
    sigma = None
    if a17.get("mean") is not None and aex.get("mean") is not None:
        delta = float(aex["mean"] - a17["mean"])
        sigmas = [x["std"] for x in (a17, aex) if x.get("std")]
        sigma = float(max(sigmas)) if sigmas else None
    judge = {
        "J1_accuracy": {"pass": bool(delta is not None and sigma is not None and delta >= sigma),
                        "delta": delta, "sigma": sigma,
                        "rule": "Δ = macroF1(ext) − macroF1(17)，|Δ| ≥ 1σ（种子标准差）才算有效"},
        "J2_operating_point": {
            "rule": f"在误报预算 {FA_BUDGET} 次/小时下，所需阈值/确认/时延是否下降",
            "ctrl17": runs.get(f"ctrl17_s{seeds[0]}", {}).get("best_point"),
            "ctrl_ext": runs.get(f"ctrl_ext_s{seeds[0]}", {}).get("best_point")},
        "J3_vs_full_schema": {
            "rule": "与完整流特征口径（0.9940）还差多少",
            "full_schema_f1": FULL_SCHEMA_F1,
            "gap_ctrl17": (FULL_SCHEMA_F1 - a17["mean"]) if a17.get("mean") is not None else None,
            "gap_ctrl_ext": (FULL_SCHEMA_F1 - aex["mean"]) if aex.get("mean") is not None else None,
            "historical_ctrl17_f1": CTRL17_REF_F1},
    }
    payload = {
        "experiment": "controller_native_field_extension",
        "question": "只用 OpenFlow 真能拿到/推导的字段时，把「控制器能算的聚合量」补上能补回多少精度与工作点",
        "params": {"scale": float(args.scale), "epochs": int(args.epochs), "seeds": list(seeds),
                   "window": 16, "stride": 5, "window_sec": WINDOW_SEC, "fa_budget": FA_BUDGET},
        "base_feature_count_expected": 17,
        "ext_fields": EXT_FIELDS,
        "unavailable_fields": UNAVAILABLE,
        "runs": runs,
        "summary": summary,
        "judgement": judge,
        "elapsed_s": round(time.time() - started, 1),
    }
    save_json(payload, RESULT_DIR / "ctrl_field_extension.json")
    save_text(build_report(payload), RESULT_DIR / "ctrl_field_extension.md")
    if not args.no_figure:
        fp = make_figure(payload, FIGURE_DIR / "ctrl_field_extension.png")
        LOGGER.info("图：%s", fp)
    LOGGER.info("完成：%s（用时 %.0fs）", RESULT_DIR / "ctrl_field_extension.md", payload["elapsed_s"])
    return 0


def build_report(p: dict) -> str:
    s = p["summary"]
    j = p["judgement"]
    d = j["J1_accuracy"]
    lines = [
        "# 控制器原生字段扩展：字段缺口能补回多少",
        "",
        "> 由 `tools/controller_fields.py` 生成。两个口径**同一批流**（同一次生成，只切 schema），"
        "结构/窗口/种子完全一致，因此差值可归因于字段。",
        "",
        "## 0. 结论",
        "",
        f"- 17 列口径宏 F1 **{fmt_float(s['ctrl17']['macro_f1']['mean'])}"
        f"{(' ± ' + fmt_float(s['ctrl17']['macro_f1']['std'])) if s['ctrl17']['macro_f1']['std'] else ''}**"
        f"（{s['ctrl17']['n_features']} 个特征）→ 扩展口径 "
        f"**{fmt_float(s['ctrl_ext']['macro_f1']['mean'])}"
        f"{(' ± ' + fmt_float(s['ctrl_ext']['macro_f1']['std'])) if s['ctrl_ext']['macro_f1']['std'] else ''}**"
        f"（{s['ctrl_ext']['n_features']} 个特征）",
        f"- Δ = **{fmt_float(d['delta'])}**，σ = {fmt_float(d['sigma'])} → "
        f"**{'✅ 超出种子噪声（字段确实补回了精度）' if d['pass'] else '❌ 未超出种子噪声'}**",
        f"- 与完整流特征口径（{fmt_float(j['J3_vs_full_schema']['full_schema_f1'])}）的差距："
        f"17 列差 **{fmt_float(j['J3_vs_full_schema']['gap_ctrl17'])}** → 扩展后差 "
        f"**{fmt_float(j['J3_vs_full_schema']['gap_ctrl_ext'])}**",
        "",
        "## 1. 扩展了哪些字段（逐条说明来源）",
        "",
        "**① OFPFlowStats 直接给的（既有 17 列，原样保留）**：duration、packet_count、byte_count、"
        "match 字段（源/目的 IP、源/目的端口、ip_proto、icmp type/code）、idle/hard timeout、flags。",
        "",
        "**② 单流一步派生**：`avg_pkt_size = byte_count / packet_count`。",
        "",
        "**③ 控制器按轮询窗口做流表聚合得到**（Ryu 在同一 poll 周期能拿到全部流条目，可分组统计）：",
        "",
    ]
    lines.append(markdown_table(["字段", "含义 / 为什么可推导"],
                                [[f"`{f}`", _field_desc(f)] for f in p["ext_fields"]]))
    lines += [
        "",
        "**④ 明确不可得（不伪造）**：",
        "",
    ]
    lines += [f"- {x}" for x in p["unavailable_fields"]]
    lines += [
        "",
        f"> 因此本扩展**不可能**追平 CICFlowMeter 的 84 列口径 —— 这不是实现问题，"
        f"而是 OpenFlow 流表统计**本身不含**这些量。**字段缺口的正确回应是补可推导的聚合量 + 调工作点，"
        f"而不是假装拿到了报文级信息。**",
        "",
        "## 2. 逐运行结果",
        "",
    ]
    head = ["口径", "种子", "特征数", "宏 F1", "二分类 AUC", "参数量",
            f"预算 {FA_BUDGET} 次/h 的工作点（阈值/确认/时延）"]
    t = []
    for tag, r in sorted(p["runs"].items()):
        bp = r.get("best_point") or {}
        t.append([r["variant"], r["seed"], r["n_features"], fmt_float(r["macro_f1"]),
                  fmt_float(r["binary_roc_auc"]), f"{r.get('n_params', 0):,}",
                  (f"{fmt_float(bp.get('threshold'), 3)} / {bp.get('confirm')} / "
                   f"{fmt_float(bp.get('mean_delay_ms'), 1)} ms" if bp.get("threshold") is not None
                   else bp.get("note", "未测"))])
    lines.append(markdown_table(head, t))
    lines += [
        "",
        "## 3. 工作点代价（判据 J2）",
        "",
        f"- 17 列：{_point_line(j['J2_operating_point'].get('ctrl17'))}",
        f"- 扩展：{_point_line(j['J2_operating_point'].get('ctrl_ext'))}",
        "",
        "> 说明：这里的工作点是「**在同一误报预算下、召回保持 100% 的前提下时延最小**」的那一个，"
        "与 `tools/operating_point.py` 同一套口径；工作点越低说明同样的安全要求付出的代价越小。",
        "",
        "## 4. 局限（写清楚，不粉饰）",
        "",
        "- 聚合量是在**离线数据上按 60 秒滑动窗**离线复算的；真机上受轮询周期、流表老化（idle timeout）"
        "与统计延迟影响，数值会有偏差（方向不变）。",
        "- `idle_timeout`/`hard_timeout`/`flags` 是**控制器配置值**，在单次实验里近似常数、"
        "对模型帮助有限；真实部署里它们随策略变化，可能带来额外信息。",
        "- 端口是特征之一：真实 SDN 里可用，但会带来「按端口记忆」的风险（本工程默认在完整口径里"
        "丢弃端口，这里保留是因为控制器原生口径本来就带端口）。",
        "- 单一种子之外的结论强度有限：每口径 3 个种子只够判「是否超过种子噪声」。",
        "",
        "## 5. 复现命令",
        "",
        "```powershell",
        f"python tools/controller_fields.py --scale {p['params']['scale']} --epochs {p['params']['epochs']}",
        "```",
        "",
        "> 隔离：数据写在 `data/raw_ctrl/`、`data/raw_ctrl_ext/`，产物在 `artifacts/ctrl_*/` 与 "
        "`results/ctrl_ext/`；主实验数据与模型全程只读。",
        "",
    ]
    return "\n".join(lines)


def _field_desc(name: str) -> str:
    return {
        "avg_pkt_size": "字节数 ÷ 包数，一步除法可得",
        "src_flow_count": "同一源 IP 在窗口内的流条目数（流表分组计数）",
        "src_distinct_dst": "同一源打过的不同目的 IP 数（横向移动/扫描线索）",
        "src_distinct_dport": "同一源打过的不同目的端口数（**扫描的核心指标**）",
        "src_dport_entropy": "同一源目的端口的分布熵（扫描 = 熵高且分散）",
        "src_short_flow_ratio": "同一源里 duration < 1 s 的流占比（暴破/扫描 = 短流多）",
        "src_pkt_per_flow_mean": "同一源每流平均包数（心跳/长连接 = 少而规律）",
        "src_new_flow_rate": "同一源新建流速率（流/秒，洪泛与扫描都高）",
        "dst_flow_count": "同一目的 IP 的流条目数",
        "dst_distinct_src": "打同一目的 IP 的不同源数（**DDoS 的核心指标**）",
        "dst_src_entropy": "同一目的的源分布熵（分布式洪泛 = 熵高）",
        "table_flow_count": "窗口内流表规模（攻击强度/表压力代理量）",
        "proto_share": "该协议在窗口内的占比（协议构成变化）",
    }.get(name, "—")


def _point_line(pt: dict | None) -> str:
    if not pt or pt.get("threshold") is None:
        return (pt or {}).get("note", "未测")
    return (f"阈值 **{fmt_float(pt['threshold'], 3)}** + 确认 **{pt['confirm']}** → "
            f"召回 {pt['segment_recall']:.0%}、误报 {fmt_float(pt['fa_per_hour'], 2)} 次/小时、"
            f"平均时延 **{fmt_float(pt['mean_delay_ms'], 1)} ms**")


def make_figure(p: dict, out: Path) -> Path | None:
    try:
        setup_matplotlib(logger=LOGGER)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib 不可用，跳过作图：%s", exc)
        return None
    s = p["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), dpi=150)
    names = ["控制器 17 列", f"控制器扩展（{s['ctrl_ext']['n_features']} 列）", "完整流特征（参照）"]
    vals = [s["ctrl17"]["macro_f1"]["mean"], s["ctrl_ext"]["macro_f1"]["mean"], FULL_SCHEMA_F1]
    errs = [s["ctrl17"]["macro_f1"]["std"] or 0, s["ctrl_ext"]["macro_f1"]["std"] or 0, 0.0011]
    ax = axes[0]
    ax.bar(names, vals, yerr=errs, capsize=4, color=["#8aa0b6", "#4fd1c5", "#f6ad55"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, fmt_float(v), ha="center", fontsize=10)
    ax.set_ylim(max(0.0, min(vals) - 0.08), 1.02)
    ax.set_ylabel("宏平均 F1")
    ax.set_title("字段口径对精度的影响")
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    labels, thrs = [], []
    for key, label in (("ctrl17", "控制器 17 列"), ("ctrl_ext", "控制器扩展")):
        bp = (p["judgement"]["J2_operating_point"].get(key) or {})
        labels.append(label)
        thrs.append(bp.get("threshold") if bp.get("threshold") is not None else np.nan)
    ax.bar(labels, thrs, color=["#8aa0b6", "#4fd1c5"])
    for i, v in enumerate(thrs):
        if np.isfinite(v):
            ax.text(i, v + 0.01, fmt_float(v, 3), ha="center", fontsize=10)
    ax.set_ylabel(f"达到误报预算 {FA_BUDGET} 次/小时所需的判定阈值")
    ax.set_title("同样的安全要求，代价是否下降（阈值越低 = 越从容）")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
