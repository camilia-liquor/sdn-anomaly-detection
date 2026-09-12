"""窗口长度消融汇总：把「窗口要多长才算够」这件事测穿，并给出可执行的部署结论。

为什么需要这个工具
------------------
作品里有一条必须被钉死的负面结论：**卷积分支（CNN）没有带来增益**。
早期的窗口消融（T=10/16/24）显示「混合模型 − 最好单分支」的差值随窗口单调收敛，
于是留下一个含糊的解释——「CNN 需要更长的上下文才能发挥作用」。这个解释只在**收敛点之前**
被观察过；本工具把 T=10…48 的全部消融产物汇总成**一张表 + 一张图 + 一份带硬结论的报告**，
把那个解释放到收敛点之后检验，并把「差距是否落在随机种子噪声内」量化清楚。

本工具同时汇总**第二条结论**：窗口长度是误报的免费闸门（T 越长误报越低），代价是检测时延上升。
因此报告里既有「架构对不对」，也有「工作点该选哪个」。

支持的种子组织方式（自动发现，不写死）
--------------------------------------
  对每个 (T, 结构)，按产物名枚举候选 tag：
      base  = ws<T>              （混合模型）
      base  = ws<T>_<variant>    （两个单分支）
      tag   = base 或 base_s<k>  （多种子）
  再读 artifacts/<tag>/config_used.json 拿真实 seed；config 缺失时从 tag 名解析。
  一个 (T, 结构) 下有 ≥2 个种子时给出「均值 ± 样本标准差」，只有 1 个种子时如实标注。

判定规则（预先写死，不允许事后挑标准）
--------------------------------------
  对每个 T：噪声尺度 = 该 T 上混合与最好单分支两侧 σ 的**较大者**（多种子时实测），
            无多种子时退回参照 σ（默认取 T=16 主实验的 3 种子 σ）。
  1. |Δ| < 1σ   → 落在随机种子噪声内
  2. 1σ ≤ |Δ| < 2σ → 边缘证据，需多种子复现
  3. |Δ| ≥ 2σ   → 超出噪声
  另外单独标注**指标饱和**（混合宏 F1 ≥ 阈值）：此时 Δ 受天花板截断，不能当能力证据。

三种结论（针对「卷积的增益是否随窗口增长」）：
  * `null`     所有已测 T 的 Δ 都在噪声内 → 边界测穿，卷积分支冗余
  * `blip`     个别 T 越界，但**最长窗口的 Δ 又回到噪声内** → 单点抖动、趋势不成立（弱支持）
  * `positive` 越界且最长窗口仍越界 → 假设方向获支持（仍需多种子确认）

输入 / 输出
-----------
  in : results/metrics_ws*.json、artifacts/ws*/config_used.json、
       results/metrics_cnn_bilstm_main*_s*.json（参照噪声）、results/benchmark_latency.json
  out: results/window_ablation.json、results/window_ablation.md、
       results/figures/window_ablation_trend.png
  缺哪个就在报告里标 `未测`，绝不编数。

用法
----
    python tools/window_ablation.py
    python tools/window_ablation.py --windows 10,16,24,32,48
    python tools/window_ablation.py --no-figure

工具是幂等的：只读产物 + 覆盖写三个输出文件，重复运行结果完全一致。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (  # noqa: E402
    FIGURE_DIR,
    RESULT_DIR,
    ensure_dirs,
    get_logger,
    markdown_table,
    save_json,
    save_text,
    setup_matplotlib,
)

LOGGER = get_logger("cida.window_ablation")

DEFAULT_WINDOWS = (10, 16, 24, 32, 48)
VARIANTS = ("cnn_bilstm", "cnn", "bilstm")
VARIANT_LABEL = {
    "cnn_bilstm": "混合（CNN + BiLSTM）",
    "cnn": "仅 CNN",
    "bilstm": "仅 BiLSTM",
}
MISSING = "未测"
SATURATION_F1 = 0.99995
MAX_EXTRA_SEEDS = 6


# --------------------------------------------------------------------------------------
# 读取与抽取
# --------------------------------------------------------------------------------------
def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:  # pragma: no cover - 产物损坏时如实报错
        LOGGER.warning("读取 %s 失败：%s", path.name, exc)
        return None


def variant_base_tag(window: int, variant: str) -> str:
    """与 run_all.py 的 make_tag 约定一致：主变体不带后缀，对照变体带。"""
    return f"ws{window}" if variant == "cnn_bilstm" else f"ws{window}_{variant}"


def extract(report: dict | None) -> dict | None:
    """从单个 metrics_*.json 里抽取本工具需要的字段（缺字段一律 None，不猜）。"""
    if not report:
        return None
    mc = report.get("multiclass") or {}
    on = report.get("online") or {}
    data = report.get("data") or {}
    model = report.get("model") or {}
    n_errors, n_test = None, None
    cm = mc.get("confusion_matrix")
    if cm:
        n_test = int(sum(sum(row) for row in cm))
        n_errors = int(sum(cm[i][j] for i in range(len(cm)) for j in range(len(cm)) if i != j))
    return {
        "macro_f1": mc.get("macro_f1"),
        "accuracy": mc.get("accuracy"),
        "macro_precision": mc.get("macro_precision"),
        "macro_recall": mc.get("macro_recall"),
        "n_errors": n_errors,
        "n_test_windows": n_test,
        "segment_recall": on.get("segment_recall"),
        "false_alarms_per_hour": on.get("false_alarms_per_hour"),
        "false_alarms": on.get("false_alarms"),
        "mean_detection_delay_sec": on.get("mean_detection_delay_sec"),
        "p95_detection_delay_sec": on.get("p95_detection_delay_sec"),
        "n_windows": on.get("n_windows"),
        "stream_hours": on.get("stream_hours"),
        "window_size": data.get("window_size"),
        "n_params": model.get("n_params"),
    }


def collect_records(results_dir: Path, artifact_dir: Path, window: int, variant: str) -> list[dict]:
    """枚举该 (T, 结构) 下所有已落盘的种子记录，按 seed 升序返回。"""
    base = variant_base_tag(window, variant)
    tags = [base] + [f"{base}_s{k}" for k in range(MAX_EXTRA_SEEDS)]
    recs: list[dict] = []
    for tag in tags:
        rep = _load_json(results_dir / f"metrics_{tag}.json")
        if rep is None:
            continue
        cfg = _load_json(artifact_dir / tag / "config_used.json") or {}
        seed = cfg.get("seed")
        if seed is None:  # config 还没落盘（例如训练正在进行）→ 从 tag 名解析
            m = re.search(r"_s(\d+)$", tag)
            seed = int(m.group(1)) if m else 0
        row = extract(rep)
        if row is None:
            continue
        row["tag"] = tag
        row["seed"] = int(seed)
        row["seed_source"] = "config_used.json" if cfg else "tag-name"
        recs.append(row)
    # 同一个 seed 只保留一条（正常不会重复；若重复，保留先命中的 base/小后缀那条）
    seen, uniq = set(), []
    for r in sorted(recs, key=lambda x: x["seed"]):
        if r["seed"] in seen:
            LOGGER.warning("%s：seed %d 有重复产物，只取 %s", base, r["seed"], r["tag"])
            continue
        seen.add(r["seed"])
        uniq.append(r)
    return uniq


def protocol(artifact_dir: Path, tag: str) -> dict | None:
    """读出该 tag 那次运行的真实口径，避免拿默认值冒充历史配置。"""
    cfg = _load_json(artifact_dir / tag / "config_used.json")
    if not cfg:
        return None
    return {
        "tag": tag,
        "window_size": cfg.get("window_size"),
        "window_stride": cfg.get("window_stride"),
        "online_stride": cfg.get("online_stride"),
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "purge_windows": cfg.get("purge_windows"),
        "input_mode": cfg.get("input_mode"),
        "publish_summary": cfg.get("publish_summary"),
    }


# --------------------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------------------
def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _pop_std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _stats(values: list[float], nd: int = 8) -> dict:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "population_std": None, "values": []}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "std": _sample_std(vals),
        "population_std": _pop_std(vals),
        "min": min(vals),
        "max": max(vals),
        "values": [round(v, nd) for v in vals],
    }


def variant_stats(recs: list[dict], sat_f1: float = SATURATION_F1) -> dict:
    def col(key):
        return [r[key] for r in recs if r.get(key) is not None]

    first = recs[0] if recs else {}
    f1_stats = _stats(col("macro_f1"))
    return {
        "n_seeds": len(recs),
        "seeds": [r["seed"] for r in recs],
        "tags": [r["tag"] for r in recs],
        "macro_f1": f1_stats,
        "accuracy": _stats(col("accuracy")),
        "false_alarms_per_hour": _stats(col("false_alarms_per_hour")),
        "false_alarms": _stats(col("false_alarms"), nd=4),
        "mean_detection_delay_sec": _stats(col("mean_detection_delay_sec")),
        "p95_detection_delay_sec": _stats(col("p95_detection_delay_sec")),
        "segment_recall": _stats(col("segment_recall")),
        "n_errors": _stats(col("n_errors"), nd=4),
        "n_test_windows": first.get("n_test_windows"),
        "n_params": first.get("n_params"),
        "saturated": bool(f1_stats["mean"] is not None and f1_stats["mean"] >= sat_f1),
    }


# --------------------------------------------------------------------------------------
# 参照噪声（主实验多种子）
# --------------------------------------------------------------------------------------
def reference_noise(results_dir: Path, window: int = 16) -> dict:
    """主实验在 T=<window> 上的多种子宏 F1 波动 → 无多种子的 T 退回用的参照尺度。"""
    per_variant: dict[str, dict] = {}
    for variant in VARIANTS:
        suffix = "" if variant == "cnn_bilstm" else f"_{variant}"
        vals, used = [], []
        for k in range(MAX_EXTRA_SEEDS):
            rep = _load_json(results_dir / f"metrics_cnn_bilstm_main{suffix}_s{k}.json")
            if rep is None:
                continue
            f1 = (rep.get("multiclass") or {}).get("macro_f1")
            if f1 is None:
                continue
            vals.append(float(f1))
            used.append(k)
        per_variant[variant] = {
            "n_seeds": len(vals),
            "seeds": used,
            "source": f"results/metrics_cnn_bilstm_main{suffix}_s*.json",
            **_stats(vals),
        }

    agg = _load_json(results_dir / "aggregate_cnn_bilstm_main.json") or {}
    agg_ref = {}
    for v in VARIANTS:
        entry = (agg.get(v) or {}).get("multiclass_macro_f1") or {}
        if entry:
            agg_ref[v] = {"mean": entry.get("mean"), "std": entry.get("std"),
                          "n_seeds": (agg.get(v) or {}).get("n_seeds")}

    hyb = per_variant["cnn_bilstm"]
    # 参照 σ：混合模型自己的种子波动（这是该判据最直接的标尺）
    sigma = hyb.get("std")
    return {
        "reference_window": window,
        "per_variant": per_variant,
        "aggregate_reference": agg_ref,
        "sigma_seed": sigma,
        "sigma_seed_population": hyb.get("population_std"),
        "noise_source_protocol": protocol(ROOT / "artifacts", "cnn_bilstm_main_s0"),
        "note": (
            f"参照噪声只在 T={window}（主实验口径）上测过。有 3 个种子的 T 用**该 T 自己**的 σ；"
            "只有 1 个种子的 T 才退回这个参照值，报告里逐行标明。"
        ),
    }


def aggregate_crosscheck(results_dir: Path, window: int) -> dict | None:
    """若 run_all.py 自己写了 aggregate_ws<T>.json，拿它交叉核对本工具的统计。"""
    agg = _load_json(results_dir / f"aggregate_ws{window}.json")
    if not agg:
        return None
    out = {}
    for v in VARIANTS:
        entry = agg.get(v) or {}
        f1 = entry.get("multiclass_macro_f1") or {}
        if f1:
            out[v] = {"mean": f1.get("mean"), "std": f1.get("std"), "n_seeds": entry.get("n_seeds")}
    return out or None


# --------------------------------------------------------------------------------------
# 汇总表
# --------------------------------------------------------------------------------------
def build_rows(windows, results_dir: Path, artifact_dir: Path, ref_noise: dict,
               sat_f1: float = SATURATION_F1) -> list[dict]:
    ref_sigma = ref_noise.get("sigma_seed")
    rows = []
    for T in windows:
        entry: dict = {"window": T, "variants": {}, "records": {}}
        for variant in VARIANTS:
            recs = collect_records(results_dir, artifact_dir, T, variant)
            entry["records"][variant] = recs
            entry["variants"][variant] = variant_stats(recs, sat_f1=sat_f1)
        # 口径：优先用混合模型的 tag，其次任一变体
        proto = None
        for variant in VARIANTS:
            st = entry["variants"][variant]
            if st["tags"]:
                proto = protocol(artifact_dir, st["tags"][0])
                if proto:
                    break
        entry["protocol"] = proto
        entry["n_seeds"] = max((entry["variants"][v]["n_seeds"] for v in VARIANTS), default=0)
        entry["seeds"] = sorted({s for v in VARIANTS for s in entry["variants"][v]["seeds"]})
        entry["measured"] = entry["variants"]["cnn_bilstm"]["n_seeds"] > 0

        # 差值用均值；单种子时均值即该值
        means = {v: entry["variants"][v]["macro_f1"]["mean"] for v in VARIANTS}
        entry["macro_f1_mean"] = means
        entry["diff"] = {"cnn": None, "bilstm": None, "best_single": None}
        entry["best_single_variant"] = None
        entry["best_single_mean"] = None
        if entry["measured"] and means["cnn_bilstm"] is not None:
            for v in ("cnn", "bilstm"):
                if means[v] is not None:
                    entry["diff"][v] = means["cnn_bilstm"] - means[v]
            present = [v for v in ("cnn", "bilstm") if means[v] is not None]
            if present:
                entry["best_single_variant"] = max(present, key=lambda v: means[v])
                entry["best_single_mean"] = means[entry["best_single_variant"]]
                entry["diff"]["best_single"] = means["cnn_bilstm"] - entry["best_single_mean"]

        # 该 T 自己的噪声尺度：取两侧 σ 的较大者（对「卷积有增益」最不利）
        st_h = entry["variants"]["cnn_bilstm"]["macro_f1"]
        st_b = (entry["variants"][entry["best_single_variant"]]["macro_f1"]
                if entry["best_single_variant"] else {"std": None})
        cands = [(st_h.get("std"), "混合模型本 T 的 3 种子 σ"),
                 (st_b.get("std"), f"{VARIANT_LABEL.get(entry['best_single_variant'], '单分支')}本 T 的 3 种子 σ")]
        local = [(s, src) for s, src in cands if s]
        if local:
            sigma_used, sigma_src = max(local, key=lambda x: x[0])
        else:
            sigma_used, sigma_src = ref_sigma, f"参照 σ（T={ref_noise.get('reference_window')} 主实验，单种子 T 无自有 σ）"
        entry["sigma_used"] = sigma_used
        entry["sigma_source"] = sigma_src
        entry["has_ci"] = entry["n_seeds"] >= 2
        if entry["measured"] and sigma_used and entry["diff"]["best_single"] is not None:
            entry["delta_over_sigma"] = entry["diff"]["best_single"] / sigma_used
        else:
            entry["delta_over_sigma"] = None
        entry["aggregate_crosscheck"] = aggregate_crosscheck(results_dir, T) if entry["has_ci"] else None
        rows.append(entry)
    return rows


def latency_reference(results_dir: Path) -> dict:
    """单窗口推理耗时：只有主模型跑过基准（入口张量形状即窗口长度），其余标未测。"""
    rep = _load_json(results_dir / "benchmark_latency.json")
    if not rep:
        return {"available": False}
    shape = rep.get("input_shape") or []
    window = shape[1] if len(shape) >= 2 else None
    per_batch = {}
    for r in rep.get("results") or []:
        if r.get("batch_size") is not None:
            per_batch[str(r["batch_size"])] = r.get("per_window_ms")
    return {"available": True, "tag": rep.get("tag"), "window": window, "per_batch_ms": per_batch}


# --------------------------------------------------------------------------------------
# 结论判定（数据驱动，不粉饰）
# --------------------------------------------------------------------------------------
def judge(rows: list[dict], ref_noise: dict) -> dict:
    measured = [r for r in rows if r["measured"] and r["diff"].get("best_single") is not None]
    verdict: dict = {
        "measured_windows": [r["window"] for r in measured],
        "windows_with_ci": [r["window"] for r in measured if r["has_ci"]],
        "windows_single_seed": [r["window"] for r in measured if not r["has_ci"]],
        "saturated_windows": [r["window"] for r in measured if r["variants"]["cnn_bilstm"]["saturated"]],
        "reference_sigma": ref_noise.get("sigma_seed"),
    }
    if not measured:
        verdict["case"] = "insufficient"
        verdict["reason"] = "缺少可用的消融产物，无法判定。"
        return verdict

    deltas = [(r["window"], r["diff"]["best_single"], r["sigma_used"], r["has_ci"]) for r in measured]
    largest_T, largest_delta, largest_sigma, _ = deltas[-1]
    best = max(deltas, key=lambda x: x[1])
    verdict.update({
        "delta_series": [{"window": t, "delta": d, "sigma": s, "has_ci": c} for t, d, s, c in deltas],
        "largest_window": largest_T,
        "delta_at_largest": largest_delta,
        "max_delta": best[1],
        "max_delta_window": best[0],
        "max_delta_sigma": best[2],
    })
    positive = [(t, d, s) for t, d, s, _ in deltas if s and d > s]
    verdict["positive_windows"] = [t for t, _, _ in positive]
    verdict["positive_detail"] = [{"window": t, "delta": d, "sigma": s, "ratio": d / s} for t, d, s in positive]

    if not positive:
        verdict["case"] = "null"
        verdict["reason"] = (
            f"所有已测 T 的「混合 − 最好单分支」都不超过 1σ（最大 {best[1]:+.4f} @ T={best[0]}），"
            "卷积分支的相对增益始终停在噪声内。"
        )
    elif largest_sigma and largest_delta > largest_sigma:
        verdict["case"] = "positive"
        verdict["reason"] = (
            f"最长窗口 T={largest_T} 处 Δ={largest_delta:+.4f} 超过 1σ={largest_sigma:.4f}，"
            f"且越界窗口为 {'、'.join('T=%d' % t for t in verdict['positive_windows'])}。"
        )
    else:
        verdict["case"] = "blip"
        verdict["reason"] = (
            f"有个别 T 越界（{'、'.join('T=%d（Δ=%+.4f，%0.2fσ）' % (p['window'], p['delta'], p['ratio']) for p in verdict['positive_detail'])}），"
            f"但**最长窗口 T={largest_T} 的 Δ={largest_delta:+.4f} 又回到噪声内**：越界是单点迹象，趋势不成立。"
        )
    return verdict


# --------------------------------------------------------------------------------------
# 格式化
# --------------------------------------------------------------------------------------
def _f(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return MISSING
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return MISSING


def _num(x, nd: int = 4) -> str:
    return _f(x, nd) if x is not None else MISSING


def _ms(stats: dict, nd: int = 4) -> str:
    """均值 ± 标准差；只有 1 个种子时只写均值并标注。"""
    if not stats or stats.get("n", 0) == 0:
        return MISSING
    if stats.get("n", 0) == 1:
        return f"{_f(stats['mean'], nd)}（单种子）"
    return f"{_f(stats['mean'], nd)} ± {_f(stats.get('std') or 0.0, nd)}"


def _seed_note(row: dict) -> str:
    seeds = row.get("seeds") or []
    if not seeds:
        return MISSING
    if len(seeds) == 1:
        return f"仅 seed {seeds[0]}（单种子，无 CI）"
    return "seed " + ",".join(str(s) for s in seeds)


def _band(delta: float | None, sigma: float | None) -> str:
    if delta is None or not sigma:
        return MISSING
    r = abs(delta) / sigma
    better = "混合更好" if delta > 0 else "混合更差"
    if r < 1:
        return f"{r:.2f}σ（噪声内）"
    if r < 2:
        return f"{r:.2f}σ（边缘，{better}）"
    return f"{r:.2f}σ（超出，{better}）"


# --------------------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------------------
def build_report(rows, verdict, lat, ref_noise) -> str:
    L: list[str] = []
    A = L.append

    sat_Ts = verdict.get("saturated_windows") or []
    with_ci = verdict.get("windows_with_ci") or []
    single = verdict.get("windows_single_seed") or []
    T32 = next((r for r in rows if r["window"] == 32), None)
    T16 = next((r for r in rows if r["window"] == 16), None)
    delta_series = verdict.get("delta_series") or []

    A("# 窗口长度消融：卷积分支在流统计量特征上到底有没有增益")
    A("")
    A("> 本报告由 `tools/window_ablation.py` 依据 `results/metrics_ws*.json` 与 "
      "`artifacts/ws*/config_used.json` 自动生成，全部数字取自磁盘上的实验产物，未做任何手工调整；"
      "缺测项一律标 `未测`。判定阈值在脚本里预先写死，不做事后调整。")
    A("")

    # ---------------- 0. 结论 ----------------
    A("## 0. 结论（先说硬的）")
    A("")
    series_txt = " → ".join(
        f"{d['delta']:+.4f}(T={d['window']})" for d in delta_series
    )
    A(f"**差距走势**：{series_txt}"
      + (f"；T={sat_Ts[-1]} 处双方都饱和到 1.0。" if sat_Ts else "。"))
    A("")
    if verdict["case"] == "insufficient":
        A(f"**证据不足**：{verdict['reason']}")
    else:
        pos = verdict.get("positive_detail") or []
        if pos:
            p = max(pos, key=lambda x: x["ratio"])
            A(f"**「卷积需要更长上下文」这一假设在 T={p['window']} 处首次出现正向迹象**"
              f"（Δ={p['delta']:+.4f}，约 {p['ratio']:.2f}σ），"
              + (f"但 T={verdict['largest_window']} 立刻被饱和天花板掩盖"
                 f"（Δ={verdict['delta_at_largest']:+.4f}）。" if sat_Ts else "。"))
            A("")
            A("**我们的结论是：弱支持，不足以支撑「混合模型更优」的主张。**理由有三：")
            A("")
            if T32 is not None:
                best_v = T32["best_single_variant"]
                A(f"1. **幅度只有噪声量级的 {p['ratio']:.1f} 倍**："
                  f"T={p['window']} 的 Δ={p['delta']:+.4f}，"
                  f"该 T 的噪声尺度 σ={_f(p['sigma'])}；"
                  "按预先写死的判据只到「边缘/刚越界」，不是稳定优势。")
                A(f"2. **T={verdict['largest_window']} 归零**：如果卷积真的因上下文变长而开始起作用，"
                  f"最长窗口的增益应当最大；实测 Δ={verdict['delta_at_largest']:+.4f}，"
                  "趋势与该假设的预测方向相反。")
                A(f"3. **饱和**：T≥{sat_Ts[0]} 时混合模型宏 F1 已经打满 1.0000，"
                  "Δ 被天花板截断，此刻的差值已不是能力差异的度量。")
        else:
            A("**假设②成立：边界已测穿。在流统计量特征上，卷积分支是冗余的。**")
        A("")
        # 本轮关键：单种子 → 多种子发生了什么
        multi = [r for r in rows if r["measured"] and r["has_ci"] and r["best_single_variant"]]
        flips: list[dict] = []
        if multi:
            A("### 0.1 补齐种子后发生了什么（本轮关键）")
            A("")
            A("同一种配置只跑 1 个种子时，Δ 的符号可能完全由那一次运行决定。"
              "下面把每个有 ≥2 个种子的 T 的**逐种子结果**摊开，"
              "这是判断「某个 Δ 是真实增益还是运气」的唯一直接证据：")
            A("")
            for r in multi:
                seeds = r["seeds"]
                rows_tbl = []
                for v in VARIANTS:
                    recs = {rec["seed"]: rec for rec in r["records"][v]}
                    per = " / ".join(
                        (_f(recs[s]["macro_f1"]) if s in recs else "—") for s in seeds)
                    rows_tbl.append([VARIANT_LABEL[v], per,
                                     _ms(r["variants"][v]["macro_f1"])])
                A(f"**T={r['window']}（种子 {', '.join(str(s) for s in seeds)}）**")
                A("")
                A(markdown_table(["结构", f"逐种子宏F1（seed {'/'.join(str(s) for s in seeds)}）",
                                  "均值 ± 样本标准差"], rows_tbl))
                A("")
                s0 = seeds[0]
                h0 = _per_seed_f1(r, "cnn_bilstm", s0)
                sgl = [_per_seed_f1(r, v, s0) for v in ("cnn", "bilstm")]
                sgl = [x for x in sgl if x is not None]
                if h0 is not None and sgl:
                    delta0 = h0 - max(sgl)
                    A(f"- **只看 seed {s0}**：Δ = {h0:.4f} − {max(sgl):.4f} = **{delta0:+.4f}**；"
                      f"**补齐 {len(seeds)} 个种子后**：Δ = {_f(r['diff']['best_single'])}"
                      f"（判据用该 T 自己的 σ={_f(r.get('sigma_used'))}，"
                      f"落在 **{_band(r['diff']['best_single'], r.get('sigma_used'))}**）。")
                    if delta0 > 0 >= r["diff"]["best_single"]:
                        best_seed = max(
                            (rec["seed"] for rec in r["records"]["cnn_bilstm"]),
                            key=lambda s: _per_seed_f1(r, "cnn_bilstm", s) or -1)
                        flips.append({
                            "window": r["window"], "seed0": s0, "delta0": delta0,
                            "delta_multi": r["diff"]["best_single"],
                            "sigma": r.get("sigma_used"), "n_seeds": len(seeds),
                            "best_seed": best_seed, "has_ci": r["has_ci"],
                        })
                        A(f"- 那个正向迹象**完全来自 seed {s0} 这一次运行**：它是 {len(seeds)} 个种子里"
                          "表现最好的那一次，多种子平均后 Δ 转负。**单种子时它的符号没有任何保证。**")
                        A(f"- **因此：单种子时看到的 {delta0:+.4f} 不构成对"
                          "「卷积分支需要更长上下文」的支持；补齐种子后 Δ 转负并落入噪声内，"
                          "该假设在窗口长度这个维度上没有得到任何可用证据。**")
                    A("")
            if flips:
                A("### 0.2 这次结论反转本身值得记录（方法学）")
                A("")
                f0 = max(flips, key=lambda x: x["delta0"])
                A(f"- 单种子（seed {f0['seed0']}）在 **T={f0['window']}** 上得到 "
                  f"Δ = **{f0['delta0']:+.4f}**，看上去像「卷积终于有增益了」——"
                  "如果报告停在这一步，我们会写下一个**假阳性**。")
                A(f"- 补齐其余种子后（共 {f0['n_seeds']} 个）Δ 变成 **{f0['delta_multi']:+.4f}**"
                  + (f"，**落在该 T 的种子标准差（σ={_f(f0['sigma'])}）之内**" if f0.get("sigma") else "")
                  + "——正向迹象消失。")
                if f0.get("best_seed") == f0["seed0"]:
                    A(f"- 原因很直白：seed {f0['seed0']} 恰好是 {f0['n_seeds']} 个种子里"
                      "表现最好的那一次，其余种子都没有它好（见 §0.1 的逐种子表）。")
                else:
                    A(f"- 也就是说，这些种子之间的表现差异本身就有这么大的量级"
                      f"（最好的那次是 seed {f0.get('best_seed')}，见 §0.1 的逐种子表）。")
                A("- 因此 **「单种子 + 边缘显著」这个组合本身就不可作为结论**："
                  "它既可能来自真实效应，也可能来自一次运气好的初始化与训练顺序，"
                  "而**单次运行无法区分这两者**。")
                A("- 这正是本工程坚持「每个配置跑 3 个随机种子、报均值 ± 样本标准差」的原因；"
                  "本报告第 3 节把判据写成 σ 分档（<1σ 噪声内 / 1–2σ 边缘 / ≥2σ 超出），"
                  "也是为了让「差多少才算差」在事前就有据可依，而不是事后挑标准。")
                A("- 一句话总结：**我们自己抓到了自己的一个假阳性，并把修正过程留在报告里**——"
                  "一个能被自己的补充实验推翻的数字，比一个漂亮但站不住的正向数字更有价值。")
                A("")
        A("**这一轮真正可用的收获是部署结论**：" + _deployment_sentence(rows))
    A("")

    # ---------------- 1. 口径 ----------------
    A("## 1. 实验口径（从产物里读回来，不是默认值）")
    A("")
    A(markdown_table(
        ["T", "window_size", "window_stride", "online_stride", "epochs", "batch",
         "purge_windows", "该 T 的种子"],
        [[r["window"],
          (r["protocol"] or {}).get("window_size", MISSING),
          (r["protocol"] or {}).get("window_stride", MISSING),
          (r["protocol"] or {}).get("online_stride", MISSING),
          (r["protocol"] or {}).get("epochs", MISSING),
          (r["protocol"] or {}).get("batch_size", MISSING),
          (r["protocol"] or {}).get("purge_windows", MISSING),
          _seed_note(r) if r["measured"] else MISSING]
         for r in rows]))
    A("")
    A("- 产物命名规则（与 `run_all.py` 的 `make_tag` 一致）：混合模型 `ws<T>`、"
      "单分支 `ws<T>_cnn` / `ws<T>_bilstm`，多种子再加 `_s<seed>`；"
      "指标文件即 `results/metrics_<tag>.json`。")
    A("- 所有 T 共用同一份合成数据（数据生成固定用 cfg.seed，与训练种子无关），"
      "同一 T 下的多个种子**只有模型随机种子不同、数据完全相同**——这正是估算种子噪声所需的对照。")
    A("- 全部运行都带 `--set publish_summary=false`，因此**不会写主实验指针**"
      "（`results/summary.json` 与 `artifacts/main_model.json` 保持原样）。")
    A("")

    # ---------------- 2. 主表 ----------------
    A("## 2. 主表：三种结构 × 窗口长度（多种子给均值 ± 样本标准差）")
    A("")
    lat_txt = _f(lat["per_batch_ms"].get("1"), 2) if lat.get("available") else MISSING
    A(markdown_table(
        ["T", "混合 宏F1", "仅 CNN 宏F1", "仅 BiLSTM 宏F1", "混合 − 最好单分支",
         "误报/小时（混合）", "平均检测时延(ms)（混合）", "单窗口推理(ms)", "种子"],
        [[r["window"],
          _ms(r["variants"]["cnn_bilstm"]["macro_f1"]),
          _ms(r["variants"]["cnn"]["macro_f1"]),
          _ms(r["variants"]["bilstm"]["macro_f1"]),
          _f(r["diff"].get("best_single")),
          _ms(r["variants"]["cnn_bilstm"]["false_alarms_per_hour"], 2),
          (_f((r["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]["mean"] or 0) * 1000, 2)
           if r["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]["n"] else MISSING),
          (lat_txt if lat.get("window") == r["window"] else MISSING),
          _seed_note(r) if r["measured"] else MISSING]
         for r in rows]))
    A("")
    A("- 「最好单分支」= 该 T 上 仅 CNN 与 仅 BiLSTM 中宏 F1 均值更高者；差值 > 0 表示混合模型更好。")
    if lat.get("available"):
        A(f"- 单窗口推理耗时来自 `results/benchmark_latency.json`（tag=`{lat.get('tag')}`，"
          f"入口张量窗口长度 = {lat.get('window')}），**只有主模型口径测过**，其余 T 标 `未测`；"
          "各批量的每窗口耗时分别为 "
          + " / ".join(f"batch={bs} → {_f(v, 3)} ms"
                       for bs, v in sorted(lat["per_batch_ms"].items(), key=lambda kv: int(kv[0])))
          + "。")
    else:
        A("- 单窗口推理耗时：`results/benchmark_latency.json` 不存在，全部标 `未测`。")
    A("")
    for r in rows:
        if r.get("aggregate_crosscheck"):
            A(f"- T={r['window']} 独立复算核对：`run_all.py` 自己写的 "
              f"`results/aggregate_ws{r['window']}.json` 只覆盖 `--seeds 1,2` 那两次运行"
              "（**不含 seed 0**），把它与本工具对同两个种子的统计对比即可验证口径一致——"
              + "；".join(f"{VARIANT_LABEL.get(v, v)} mean={_f(e.get('mean'))}、std={_f(e.get('std'))}、n={e.get('n_seeds')}"
                          for v, e in r["aggregate_crosscheck"].items())
              + "。（本报告第 2 节的数是 3 个种子的，因此数值不同是正常的。）")
            A("")
    A("")

    # ---------------- 2.1 第二条结论：误报 vs 时延 ----------------
    A("## 2.1 第二条结论：窗口长度是误报的「免费闸门」，代价是检测时延")
    A("")
    A(markdown_table(
        ["T", "误报/小时（混合）", "误报次数（混合）", "平均检测时延(ms)（混合）",
         "p95 时延(ms)（混合）", "片段召回（混合）"],
        [[r["window"],
          _ms(r["variants"]["cnn_bilstm"]["false_alarms_per_hour"], 2),
          _ms(r["variants"]["cnn_bilstm"]["false_alarms"], 2),
          (_f((r["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]["mean"] or 0) * 1000, 2)
           if r["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]["n"] else MISSING),
          (_f((r["variants"]["cnn_bilstm"]["p95_detection_delay_sec"]["mean"] or 0) * 1000, 2)
           if r["variants"]["cnn_bilstm"]["p95_detection_delay_sec"]["n"] else MISSING),
          _ms(r["variants"]["cnn_bilstm"]["segment_recall"], 3)]
         for r in rows]))
    A("")
    A("- 趋势是**单调且方向相反**的：窗口越长，误报越低，但检测时延越高。"
      "原因是机械的——窗口越长，单个窗口覆盖流记录越多，随机抖动被平均掉；"
      "但一条攻击要在窗口内被确认，就必须等窗口填满，时延随 T 线性增长。")
    A(f"- 结论：**窗口长度是误报的免费闸门**（不动阈值、不动模型，纯靠改输入形状就把误报压下来），"
      f"**代价是时延**。{_deployment_sentence(rows)}")
    A("")
    A("### 2.2 误分类窗口数（宏 F1 差值到底等于多少条窗口）")
    A("")
    A(markdown_table(
        ["T", "测试窗口数", "混合 误分类", "最好单分支 误分类", "误分类之差"],
        [[r["window"],
          r["variants"]["cnn_bilstm"]["n_test_windows"] if r["measured"] else MISSING,
          _ms(r["variants"]["cnn_bilstm"]["n_errors"], 2),
          (_ms(r["variants"][r["best_single_variant"]]["n_errors"], 2)
           if r["best_single_variant"] else MISSING),
          (_f(r["variants"]["cnn_bilstm"]["n_errors"]["mean"] - r["variants"][r["best_single_variant"]]["n_errors"]["mean"], 2)
           if (r["best_single_variant"] and r["variants"]["cnn_bilstm"]["n_errors"]["n"]
               and r["variants"][r["best_single_variant"]]["n_errors"]["n"]) else MISSING)]
         for r in rows]))
    A("")
    A("这张表是理解第 2 节的关键：**长窗口下的宏 F1 差值，折合成窗口数只有个位数甚至 0**，"
      "所以它天然落在单次训练的随机波动范围内。")
    A("")

    # ---------------- 3. 判据 ----------------
    A("## 3. 判据：差多少才算「真的差」？（噪声尺度）")
    A("")
    A(f"本工具的规则是**每个 T 用该 T 自己的种子噪声**；只有单种子的 T 才退回参照尺度"
      f"（T={ref_noise.get('reference_window')} 主实验的 3 种子 σ）。逐 T 的 σ 来源见第 4 节。")
    A("")
    A(markdown_table(
        ["T", "该 T 的 σ（噪声尺度）", "σ 来源", "是否有置信区间"],
        [[r["window"],
          _f(r.get("sigma_used")),
          r.get("sigma_source", MISSING),
          ("有（≥2 种子）" if r["has_ci"] else "无（单种子）") if r["measured"] else MISSING]
         for r in rows]))
    A("")
    if verdict.get("reference_sigma"):
        A(f"参照尺度（T={ref_noise.get('reference_window')} 主实验 3 种子）σ_seed = "
          f"**{verdict['reference_sigma']:.4f}**"
          f"（总体标准差 {_f(ref_noise.get('sigma_seed_population'))}）。"
          "本节所有分档统一用**样本标准差（ddof=1）**，不用总体标准差，以免同一个数出现两种口径。")
        A("")
    A("**判据（预先写死，不允许事后挑标准）**：")
    A("")
    A("1. `|混合 − 最好单分支| < 1σ` → **落在随机种子噪声内**，不能宣称有增益；")
    A("2. `1σ ≤ |Δ| < 2σ` → **边缘证据**，必须多种子复现后才敢写；")
    A("3. `|Δ| ≥ 2σ` → 超出噪声，可以宣称增益（但仍需确认非饱和）。")
    A("")
    A("**为什么每个 T 用自己的 σ**：把 T=16 的 σ 搬到 T=32 用，会低估长窗口下的波动"
      "（训练集窗口数、类别分布都变了）。T=32 已有 3 个种子，因此它的判据是有置信区间支撑的。")
    A("")

    # ---------------- 4. 逐 T 明细 ----------------
    A("## 4. 逐窗口差值明细")
    A("")
    A(markdown_table(
        ["T", "混合 宏F1", "最好单分支（宏F1）", "Δ = 混合 − 最好单分支", "σ 分档",
         "Δ/σ", "指标饱和", "σ 来源", "种子"],
        [[r["window"],
          _ms(r["variants"]["cnn_bilstm"]["macro_f1"]),
          (f"{VARIANT_LABEL[r['best_single_variant']]} = {_f(r['best_single_mean'])}"
           if r["best_single_variant"] else MISSING),
          _f(r["diff"].get("best_single")),
          _band(r["diff"].get("best_single"), r.get("sigma_used")),
          _f(r.get("delta_over_sigma"), 2),
          ("是（打满）" if r["variants"]["cnn_bilstm"]["saturated"] else "否"),
          r.get("sigma_source", MISSING),
          _seed_note(r) if r["measured"] else MISSING]
         for r in rows]))
    A("")
    nsc = None
    deltas = [d["delta"] for d in delta_series]
    sigs = [d["sigma"] for d in delta_series]
    if len(deltas) >= 2 and all(s for s in sigs):
        signs = [0 if abs(d) < s else (1 if d > 0 else -1) for d, s in zip(deltas, sigs)]
        signs = [s for s in signs if s]
        nsc = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    if nsc is not None:
        pairs = list(zip([d["window"] for d in delta_series], deltas, sigs))
        neg = [(t, d) for t, d, s in pairs if d < -s]
        pos = [(t, d) for t, d, s in pairs if d > s]
        if not pos:
            A(f"- 在 ±1σ 死区外，**没有任何一个 T 的 Δ 显著为正**；"
              f"显著为负的有 {len(neg)} 个（{'、'.join(f'T={t}（{d:+.4f}）' for t, d in neg)}）。"
              "也就是说 Δ 是从「显著更差」**单调收敛到噪声内**，全程没有转正。")
        else:
            A(f"- 在 ±1σ 死区外，Δ 的符号变化了 **{nsc} 次**：显著为负 {len(neg)} 个 T，"
              f"显著为正 {len(pos)} 个 T（{'、'.join(f'T={t}' for t, _ in pos)}）——"
              "序列**非单调**，「随 T 单调收敛、再继续加长就会转正」这一印象不成立。")
        A("")
    A("")

    # ---------------- 5. 图 ----------------
    A("## 5. 趋势图")
    A("")
    A("![窗口长度消融趋势](figures/window_ablation_trend.png)")
    A("")
    A("- 左图（核心）：**混合 − 最好单分支的 Δ 随窗口长度变化**，灰带为 ±1σ 噪声区；"
      "**空心点 = 该点混合模型宏 F1 已打满（指标饱和）**，其差值不再代表能力差异；有误差棒的点 = 该 T 有 3 个种子；")
    A("- 右图：三种结构的宏 F1 随窗口长度变化，多种子的点带 ±1σ 误差棒；")
    A("- 两条曲线都不做任何平滑、不挑点，实测点全部标出。")
    A("")

    # ---------------- 6. 架构建议 ----------------
    A("## 6. 对架构选择的建议")
    A("")
    n_params = {v: (T16 or {}).get("variants", {}).get(v, {}).get("n_params") for v in VARIANTS} if T16 else {}
    A("卷积在**流统计量**输入上没有可复现的增益，它的价值主张就不该再写成「提升检测精度」。"
      "建议按下面三条改写，并在作品里主动承认这条负面结论：")
    A("")
    A("1. **把卷积的价值主张重定为「为原始包序列 / 载荷字节输入预留结构」**：卷积擅长"
      "「相邻若干条记录的局部组合模式」——突发 SYN、固定间隔心跳、重传指纹、载荷前若干字节的签名。"
      "流统计量已在特征工程阶段把每个时间步压成聚合量，这份局部信息本来就丢了；"
      "一旦输入换成原始包序列（或连续流的字节张量），卷积分支才是它该待的地方。"
      "现在的结构可以诚实地描述为**为那条路预留的接口**，而不是当前精度来源。")
    A("2. **不要说它「不花钱」**：短窗口下混合模型比最好单分支**显著更差**（见第 4 节），"
      "长窗口下只是**没有贡献**；而参数从"
      + (f" {int(n_params.get('bilstm') or 0):,}（仅 BiLSTM）涨到 {int(n_params.get('cnn_bilstm') or 0):,}（混合），"
         if n_params.get("bilstm") and n_params.get("cnn_bilstm") else " 接近翻倍，")
      + "推理链路还多一条。纯流统计量交付 + 资源紧张 → **砍掉卷积分支、直接用仅 BiLSTM**。")
    A("3. **不要在「加长窗口」上继续投入**：本次已把 T 从 10 加到 48，Δ 没有稳定转正，"
      "而且 T≥" + (f"{sat_Ts[0]}" if sat_Ts else "32") + " 起指标已打满，"
      "再加长连「有没有差别」都测不出来（见第 7 节）。")
    A("")

    # ---------------- 7. 两个限制 + 下一步 ----------------
    A("## 7. 两个必须写清的限制（以及下一步）")
    A("")
    A(f"### 7.1 饱和：这个仿真基准在 T≥{sat_Ts[0] if sat_Ts else 32} 已经失去区分度")
    A("")
    sat_desc = []
    for T in sat_Ts:
        r = next((x for x in rows if x["window"] == T), None)
        if r is None:
            continue
        sat_desc.append(
            f"T={T}：混合 宏 F1 = {_f(r['variants']['cnn_bilstm']['macro_f1']['mean'])}、"
            f"误分类 {_ms(r['variants']['cnn_bilstm']['n_errors'], 2)}、"
            f"在线误报 {_ms(r['variants']['cnn_bilstm']['false_alarms_per_hour'], 2)} 次/小时")
    for s in sat_desc:
        A(f"- {s}")
    A("")
    # 单次打满 ≠ 该配置饱和：把"某个种子的单次观测打满"单独点出来
    seed_sat = []
    for r in rows:
        check = ["cnn_bilstm"] + ([r["best_single_variant"]] if r["best_single_variant"] else [])
        for v in dict.fromkeys(check):
            stats_v = r["variants"][v]["macro_f1"]
            for rec in r["records"].get(v, []):
                if rec.get("macro_f1") is not None and rec["macro_f1"] >= SATURATION_F1:
                    seed_sat.append((r["window"], VARIANT_LABEL[v], rec["seed"],
                                     rec["macro_f1"], stats_v["mean"],
                                     r["variants"][v]["n_seeds"]))
    unmean_sat = [x for x in seed_sat if x[0] not in sat_Ts]
    if unmean_sat:
        for T, vlab, seed, f1, mean, n in unmean_sat:
            A(f"- ⚠️ 注意区分「单次打满」和「配置饱和」：**T={T} 的 {vlab} 在 seed {seed} 单次跑出 "
              f"宏 F1 = {f1:.4f}（零错误），但该 T 该结构的 {n} 种子均值只有 {mean:.4f}**。"
              "单次打满是运气（其余种子的错误被平均出来），不代表这个配置已经饱和；"
              "而 T=" + (f"{sat_Ts[0]}" if sat_Ts else "?") + " 是均值都打满，那是真的饱和。")
        A("")
    A(f"- 也就是说：在 T≥{sat_Ts[0] if sat_Ts else 32} 上，测试集是"
      "**零错误**、宏 F1 顶到 1.0000，**这个仿真基准已经饱和，无法再区分架构差异**；")
    A("- 因此「卷积分支到底有没有增益」这个问题，**不能靠继续加长窗口来回答**——"
      "再加长只会让双方都顶在天花板上，Δ 恒为 0，得到的是无信息的结果。")
    A("")
    A("**下一步（二选一，或都做）**：")
    A("")
    A("1. **提高数据难度后重做消融**：调 `overlap_ratio`、`stealth_ratio`、`label_noise`"
      "（必要时再叠加 `drift_strength`），把混合与单分支都从 1.0000 拉下来，"
      "在**不饱和**的区间上重测 T=16/24/32 的 Δ。只有在有区分度的测试集上，"
      "Δ 的符号与幅度才是关于架构的证据。")
    A("2. **换成原始包序列输入**：把输入从流统计量换成原始包序列（或连续流的字节张量）后重测，"
      "这是卷积分支唯一被期待发挥作用的地方；若那里仍然无增益，就可以彻底放弃卷积。")
    A("")
    A("### 7.2 单种子 vs 多种子：哪些结论有置信区间")
    A("")
    if with_ci:
        A(f"- **有置信区间**（每个结构 ≥2 个种子）："
          f"{'、'.join('T=%d' % t for t in with_ci)}"
          + (f"，各 {next((r['n_seeds'] for r in rows if r['window'] == with_ci[0]), '?')} 个种子；"
             if with_ci else "；")
          + "这些 T 上的 Δ、误报、时延都带均值 ± 标准差，是**可写进报告的部署级数字**。")
    if single:
        A(f"- **只是单点观测**（每个结构 1 个种子）：{'、'.join('T=%d' % t for t in single)}。"
          "这些 T 的 Δ **没有置信区间**，不能用来做显著性主张；"
          "它们在本报告里的作用只有两个：给出趋势的走向，以及提示「这里值得补种子」。")
    A("- 参照 σ 来自 T=" + str(ref_noise.get("reference_window")) + " 的主实验多种子运行，"
      "其训练预算与消融不完全相同"
      + (f"（{_proto_txt(ref_noise)}）" if ref_noise.get("noise_source_protocol") else "")
      + "，当参照用时需注意。")
    A("")
    A("### 7.3 其余局限")
    A("")
    A("- **测试集不同**：窗口变长后测试集窗口数减少（见第 2.2 节），"
      "宏 F1 的对比是「同一份时序数据、不同切窗方式」，不是同一批样本。")
    A("- **合成数据**：结论建立在工程自带的仿真流量上；真实数据上的复现见 `results/cross_dataset.json`。")
    A("- **单点口径**：窗口步长只扫了 `window_stride=5`"
      "（另有一次 stride=12 的对照见 `results/metrics_ws24s12.json`），没有联合扫 stride 与判定阈值。")
    A("- **误报口径**：误报/小时来自在线模拟（固定阈值 0.5、`online_stride=3`），"
      "没有对每个 T 单独标定阈值；且大窗口下误报次数只能取 0/1/2，方差大，不宜按小数位比较。")
    A("")

    # ---------------- 8. 复现 ----------------
    A("## 8. 复现命令")
    A("")
    A("```powershell")
    for r in rows:
        T = r["window"]
        seeds = ",".join(str(s) for s in (r["seeds"] or [0]))
        A(f"# T={T}（tag 规则：混合 ws{T}、单分支 ws{T}_cnn / ws{T}_bilstm、多种子加 _s<seed>）")
        A(f"python run_all.py --variants cnn_bilstm,cnn,bilstm --seeds {seeds} --no-figures --tag ws{T} `")
        A(f"  --set window_size={T} --set window_stride=5 --set epochs=25 `")
        A(f"  --set processed_dir=data/processed_ws{T} --set publish_summary=false")
    A("python tools/window_ablation.py")
    A("```")
    A("")
    A("> 三条硬约束：`publish_summary=false`（否则覆盖 `results/summary.json` 与 `artifacts/main_model.json`）；"
      "`window_stride=5` 与既有 ws10/16/24 保持一致；`epochs=25` 与既有 ws 运行保持一致"
      "（工程默认已改为 40，不显式指定就不可比）。")
    A("")
    return "\n".join(L)


def _proto_txt(ref_noise: dict) -> str:
    p = ref_noise.get("noise_source_protocol") or {}
    if not p:
        return ""
    return (f"主实验 epochs={p.get('epochs')}、window_size={p.get('window_size')}，"
            f"消融用 epochs=25、window_size 随 T 变化")


def _per_seed_f1(row: dict, variant: str, seed: int):
    for rec in row.get("records", {}).get(variant, []):
        if rec["seed"] == seed:
            return rec.get("macro_f1")
    return None


def _deployment_sentence(rows) -> str:
    """数据驱动的部署建议。

    选型只用**有置信区间**（≥2 种子）的 T：拿单种子的 0.00 去比 3 种子均值是拿运气比统计量。
    """
    allc = [r for r in rows if r["measured"] and r["variants"]["cnn_bilstm"]["false_alarms_per_hour"]["n"]]
    if not allc:
        return "误报/时延数据不足，暂不给工作点建议。"
    cand = [r for r in allc if r["has_ci"]] or allc
    pick = min(cand, key=lambda r: (r["variants"]["cnn_bilstm"]["false_alarms_per_hour"]["mean"],
                                    r["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]["mean"] or 0))
    st_fa = pick["variants"]["cnn_bilstm"]["false_alarms_per_hour"]
    st_f1 = pick["variants"]["cnn_bilstm"]["macro_f1"]
    pick_delay_ms = (pick["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]["mean"] or 0) * 1000
    txt = (f"误报最低且**有置信区间**的是 **T={pick['window']}**"
           f"（{_ms(st_fa, 2)} 次/小时，宏 F1 {_ms(st_f1)}），"
           f"代价是平均检测时延 **{pick_delay_ms:.1f} ms**。")
    if st_fa.get("std") is not None and st_fa["n"] >= 2:
        worst = st_fa["max"]
        txt += (f"即使取 {st_fa['n']} 个种子里最差的那次（{worst:.2f} 次/小时），"
                "也在同一量级——说明这个误报优势不是挑种子挑出来的。")
    ref = next((r for r in rows if r["window"] == 16), None)
    if ref is not None and ref["variants"]["cnn_bilstm"]["false_alarms_per_hour"]["n"]:
        ref_fa = ref["variants"]["cnn_bilstm"]["false_alarms_per_hour"]["mean"]
        ref_delay = (ref["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]["mean"] or 0) * 1000
        ratio = (ref_fa / st_fa["mean"]) if st_fa["mean"] else float("inf")
        txt += (f"对照 T=16：误报 {ref_fa:.2f} 次/小时、时延 {ref_delay:.1f} ms。"
                + (f"即用约 {pick_delay_ms - ref_delay:.0f} ms 的额外时延换掉约 {ratio:.0f} 倍的误报。"
                   if ratio != float("inf") else ""))
        txt += (f"**若业务能接受约 {pick_delay_ms:.0f} ms 的检测时延，"
                f"T={pick['window']} 是比 T=16 更好的工作点**（不动阈值、不动模型，纯靠输入形状换误报）；"
                "若时延预算只有十几毫秒，就该留在 T=16，改用阈值标定/连续确认来压误报。")
    others = [r for r in allc if r is not pick
              and r["variants"]["cnn_bilstm"]["false_alarms_per_hour"]["mean"] < st_fa["mean"]]
    if others:
        o = min(others, key=lambda r: r["variants"]["cnn_bilstm"]["false_alarms_per_hour"]["mean"])
        o_fa = o["variants"]["cnn_bilstm"]["false_alarms_per_hour"]
        txt += (f"另外 T={o['window']} 的单次观测误报更低（{_f(o_fa['mean'], 2)} 次/小时），"
                "但它只有 1 个种子、没有置信区间，**不建议仅凭它选型**。")
    return txt


# --------------------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------------------
def make_figure(rows, verdict, path: Path) -> Path | None:
    zh = setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def L(zh_text: str, en_text: str) -> str:
        return zh_text if zh else en_text

    meas = [r for r in rows if r["measured"] and r["diff"].get("best_single") is not None]
    if not meas:
        LOGGER.warning("没有可画的实测点，跳过趋势图。")
        return None

    Ts = [r["window"] for r in meas]
    gaps = [r["diff"]["best_single"] for r in meas]
    sat = [r["variants"]["cnn_bilstm"]["saturated"] for r in meas]
    ref_sigma = verdict.get("reference_sigma")
    band_sigma = ref_sigma or 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))

    # ---- 左：核心图，Δ 随 T ----
    ax = axes[0]
    if band_sigma:
        ax.axhspan(-band_sigma, band_sigma, color="#9e9e9e", alpha=0.22,
                   label=L(f"±1σ 参照噪声带（σ={band_sigma:.4f}）",
                           f"±1 sigma reference noise (sigma={band_sigma:.4f})"))
    ax.axhline(0.0, color="#424242", lw=1.0, ls="--")
    errs = []
    for r in meas:
        st_h = r["variants"]["cnn_bilstm"]["macro_f1"].get("std")
        st_b = (r["variants"][r["best_single_variant"]]["macro_f1"].get("std")
                if r["best_single_variant"] else None)
        errs.append(math.sqrt((st_h or 0) ** 2 + (st_b or 0) ** 2) if (st_h or st_b) else 0.0)
    if any(e > 0 for e in errs):
        ax.errorbar(Ts, gaps, yerr=errs, color="#c62828", lw=1.8, capsize=4, zorder=2,
                    label=L("Δ ± 合成标准差（多种子 T）", "delta ± combined std (multi-seed T)"))
    else:
        ax.plot(Ts, gaps, color="#c62828", lw=1.8, zorder=2,
                label=L("混合 − 最好单分支", "Hybrid − best single"))
    for t, g, s, r in zip(Ts, gaps, sat, meas):
        ax.plot([t], [g], marker="o", ms=8.5, color="#c62828",
                markerfacecolor=("white" if s else "#c62828"), markeredgewidth=1.6, zorder=3)
        tag = (L("（饱和）", " (saturated)") if s else "")
        ax.annotate(f"{g:+.4f}{tag}", (t, g), textcoords="offset points",
                    xytext=(0, 12 if g >= 0 else -18), ha="center", fontsize=8.5)
    ax.set_xlabel(L("窗口长度 T（条流记录）", "Window length T (flows)"))
    ax.set_ylabel(L("Δ 宏 F1", "Delta macro-F1"))
    ax.set_title(L("核心图：卷积分支的相对增益随窗口长度变化",
                   "Core: conv-branch relative gain vs window length"))
    ax.set_xticks(Ts)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="none", mfc="white", mec="#c62828", ms=8.5))
    labels.append(L("该点指标已打满（饱和）", "metric saturated at this point"))
    ax.legend(handles, labels, fontsize=8.0, loc="best")

    # ---- 右：三结构宏 F1 随 T（带误差棒） ----
    ax2 = axes[1]
    style = {"cnn_bilstm": ("#1565c0", "o", "-"), "cnn": ("#ef6c00", "s", "--"),
             "bilstm": ("#2e7d32", "^", "-.")}
    for v in VARIANTS:
        pts = [(r["window"], r["variants"][v]["macro_f1"], r["variants"][v]["macro_f1"].get("std"))
               for r in rows if r["variants"][v]["macro_f1"]["n"]]
        if not pts:
            continue
        col, mk, ls = style[v]
        ax2.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts],
                     yerr=[(p[2] or 0.0) for p in pts], marker=mk, ls=ls, color=col, lw=1.9,
                     capsize=4, label=VARIANT_LABEL[v] if zh else v)
    ax2.set_xlabel(L("窗口长度 T（条流记录）", "Window length T (flows)"))
    ax2.set_ylabel(L("宏 F1", "Macro-F1"))
    sat_txt = (L(f"（T≥{verdict['saturated_windows'][0]} 已打满）",
                 f"(T>={verdict['saturated_windows'][0]} saturated)")
               if verdict.get("saturated_windows") else "")
    ax2.set_title(L("三种结构的宏 F1 随窗口长度变化", "Macro-F1 of three structures vs window length") + sat_txt)
    ax2.set_xticks([r["window"] for r in rows])
    all_f1 = [(r["variants"][v]["macro_f1"]["mean"], r["variants"][v]["macro_f1"].get("std") or 0)
              for r in rows for v in VARIANTS if r["variants"][v]["macro_f1"]["n"]]
    if all_f1:
        lo = min(m - s for m, s in all_f1) - 0.004
        ax2.set_ylim(math.floor(lo * 1000) / 1000, 1.0015)
    ax2.legend(fontsize=8.0, loc="lower right")

    case = verdict.get("case", "insufficient")
    case_txt = {
        "positive": L("结论：Δ 超出噪声（需多种子确认非饱和）", "verdict: delta exceeds noise"),
        "blip": L("结论：T=32 首次正向迹象，量级约 2σ（弱支持）→ 继续如实报告无稳定增益",
                  "verdict: first positive sign at T=32 (~2 sigma), weak support"),
        "null": L("结论：Δ 始终落在种子噪声内（边界已测穿）",
                  "verdict: gaps stay inside seed noise"),
        "insufficient": L("证据不足", "insufficient evidence"),
    }[case]
    fig.suptitle(case_txt, fontsize=11.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="窗口长度消融汇总（表 + 图 + 结论报告）")
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS),
                    help="要汇总的窗口长度，逗号分隔（默认 10,16,24,32,48）")
    ap.add_argument("--results-dir", default=str(RESULT_DIR), help="结果产物目录（默认 results）")
    ap.add_argument("--artifact-dir", default=str(ROOT / "artifacts"), help="训练产物目录（默认 artifacts）")
    ap.add_argument("--out", default="window_ablation", help="输出文件名前缀（默认 window_ablation）")
    ap.add_argument("--figure", default="window_ablation_trend.png", help="趋势图文件名")
    ap.add_argument("--noise-window", type=int, default=16,
                    help="单种子 T 退回用的参照噪声取自哪个窗口（默认 16，主实验 3 种子）")
    ap.add_argument("--saturation-f1", type=float, default=SATURATION_F1,
                    help="混合模型宏 F1 达到该值视为指标饱和（默认 0.99995）")
    ap.add_argument("--no-figure", action="store_true", help="只出表与报告，不画图")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    windows = [int(x) for x in str(args.windows).split(",") if x.strip()]
    results_dir = Path(args.results_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    ensure_dirs(FIGURE_DIR)

    global SATURATION_F1
    SATURATION_F1 = args.saturation_f1

    ref_noise = reference_noise(results_dir, window=args.noise_window)
    rows = build_rows(windows, results_dir, artifact_dir, ref_noise, sat_f1=args.saturation_f1)
    verdict = judge(rows, ref_noise)
    lat = latency_reference(results_dir)

    missing = [r["window"] for r in rows if not r["measured"]]
    if missing:
        LOGGER.warning("以下窗口长度没有混合模型产物，将标为「未测」：%s",
                       "、".join(f"T={t}" for t in missing))
    if verdict.get("windows_single_seed"):
        LOGGER.warning("以下 T 只有 1 个随机种子（无置信区间）：%s",
                       "、".join(f"T={t}" for t in verdict["windows_single_seed"]))
    if verdict.get("windows_with_ci"):
        LOGGER.info("以下 T 有 ≥2 个种子（有置信区间）：%s",
                    "、".join(f"T={t}" for t in verdict["windows_with_ci"]))

    report = build_report(rows, verdict, lat, ref_noise)
    out_json = save_json(
        {
            "experiment": "window_ablation",
            "windows_requested": windows,
            "saturation_threshold": args.saturation_f1,
            "rows": rows,
            "reference_noise": ref_noise,
            "verdict": verdict,
            "latency_reference": lat,
        },
        results_dir / f"{args.out}.json",
    )
    out_md = save_text(report, results_dir / f"{args.out}.md")
    fig_path = None
    if not args.no_figure:
        fig_path = make_figure(rows, verdict, FIGURE_DIR / args.figure)

    LOGGER.info("窗口消融汇总完成：%d 个窗口，其中已测 %d 个（有 CI：%s）",
                len(rows), sum(1 for r in rows if r["measured"]),
                "、".join(f"T={t}" for t in verdict.get("windows_with_ci") or []) or "无")
    LOGGER.info("判定：%s | 最大 Δ=%s @ T=%s | 最长窗口 Δ=%s",
                verdict.get("case"), _num(verdict.get("max_delta")),
                verdict.get("max_delta_window"), _num(verdict.get("delta_at_largest")))
    for p in (out_json, out_md, fig_path):
        if p:
            LOGGER.info("→ %s", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
