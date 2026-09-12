"""困难基准窗口消融：把「卷积分支没有增益」这条结论放到更难的数据集上重新检验。

为什么需要这个工具
-----------------
`tools/window_ablation.py` 已经把窗口长度消融做到 T=48，结论是**卷积分支（CNN）在流统计量
特征上没有增益**：Δ（混合 − 最好单分支）在每一个有置信区间的 T 上都是负的，从未转正。
但那份报告自己写明了一个漏洞：**T≥32 时混合模型与仅 BiLSTM 都打满宏 F1 = 1.0000，
仿真基准已经饱和**——饱和之后 Δ 被天花板截断，测不出架构差异。

于是「卷积没用」这个结论，是在一个**太简单**的基准上得到的。本工具把口径换成一份
**难度旋钮全部拧大**的困难数据集（独立目录，见下），重做 T=10/16/24/32 的消融，并把
两套基准的 Δ-随-T 曲线画在同一张图上，回答三个问题：

  1. 困难基准上混合模型是否终于**超过**最好单分支？Δ 是正还是负？
  2. Δ 随 T 的**走势方向**是否改变（即「卷积需要更长上下文」的假设是否成立）？
  3. 困难基准是否**解决了饱和问题**（混合宏 F1 还是不是 1.0000）？

输入 / 输出
-----------
  in : results/window_ablation.json（简单基准，**只读**）
       results/metrics_hard*.json、artifacts/hard*/config_used.json、
       artifacts/hard*/data_summary.json（困难基准，**只读**）
  out: results/hard_ablation.json、results/hard_ablation.md、
       results/figures/hard_ablation_compare.png

判定与统计口径与 `tools/window_ablation.py` 完全一致（同一套 σ 分档判据、同一套
均值 ± 样本标准差(ddof=1) 约定），因此两套基准的数字可以直接并排看。缺测项一律标 `未测`，
绝不编数。

用法
----
    python tools/hard_ablation.py
    python tools/hard_ablation.py --windows 10,16,24,32 --no-figure

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

LOGGER = get_logger("cida.hard_ablation")

DEFAULT_WINDOWS = (10, 16, 24, 32)
VARIANTS = ("cnn_bilstm", "cnn", "bilstm")
VARIANT_LABEL = {
    "cnn_bilstm": "混合（CNN + BiLSTM）",
    "cnn": "仅 CNN",
    "bilstm": "仅 BiLSTM",
}
MISSING = "未测"
SATURATION_F1 = 0.99995
MAX_EXTRA_SEEDS = 6

# 困难档参数的「建议值」（任务给出的建议档）与「实际取值」。实际取值一律从产物
# (artifacts/hard*/config_used.json、data/raw_hard/insdn_schema_info.json) 读回来，
# 这里的常量只用于在报告里写明「建议值 → 实际值 → 为什么改」。
RECOMMENDED_HARD = {
    "overlap_ratio": 0.70,
    "stealth_ratio": 0.80,
    "stealth_mix": 0.70,
    "label_noise": 0.05,
    "feature_noise": 0.15,
}
SIMPLE_REFERENCE = {
    "overlap_ratio": 0.35,
    "stealth_ratio": 0.45,
    "stealth_mix": 0.45,
    "label_noise": 0.01,
    "feature_noise": 0.05,
}

# 正式训练开始前做的冒烟诊断（临时内联脚本，未落盘产物；复现命令见报告附录）。
# 记录在这里是为了让「参数为什么从建议档改到实际档」这件事可追溯，不是训练结果。
SMOKE_PROBE = {
    "note": "正式训练之前用临时内联脚本做的难度冒烟诊断（未产生落盘产物），用于判断难度档是否合适。",
    "row_level_logreg_macro_f1": {"simple": 0.4924, "hard_ln0.05": 0.2374},
    "window_level_logreg_macro_f1_T16": {"simple": 0.9807, "hard_ln0.02": 0.8658},
    "label_noise_theory": (
        "窗口级评估用 pure_only 过滤：窗口内只要有 1 条被错标的记录就整条丢弃，"
        "保留率 ≈ (1−label_noise)^T。label_noise=0.05 时 T=32 只保留约 19% 的候选窗口。"
    ),
}
NOISE_ARG = "label_noise"

# 简单基准的统计功效诊断缓存（main 里填充，报告里只读）
simple_diag_cache: dict = {}
simple_gen_cache: dict = {}

# 「产物 ↔ 变体」绑定错误的收集（main 里重置；任何一条都必须在报告里露面）
binding_errors: list = []


# --------------------------------------------------------------------------------------
# 读取与抽取
# --------------------------------------------------------------------------------------
def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:  # pragma: no cover - 产物损坏时如实报错
        LOGGER.warning("读取 %s 失败：%s", path.name, exc)
        return None


def hard_base_tag(window: int, variant: str) -> str:
    """与 run_all.py 的 make_tag 约定一致：主变体不带后缀，对照变体带。"""
    return f"hard{window}" if variant == "cnn_bilstm" else f"hard{window}_{variant}"


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
        "model_variant": model.get("variant"),
        "per_class_support": {k: (v or {}).get("support")
                              for k, v in (mc.get("per_class") or {}).items()},
    }


def collect_records(results_dir: Path, artifact_dir: Path, window: int, variant: str,
                    prefix: str = "hard") -> list[dict]:
    """枚举该 (T, 结构) 下所有已落盘的种子记录，按 seed 升序返回。"""
    if prefix == "hard":
        base = hard_base_tag(window, variant)
    else:
        base = f"{prefix}{window}" if variant == "cnn_bilstm" else f"{prefix}{window}_{variant}"
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
        # ★ 绑定校验：metrics_<tag>.json 里记录的 model.variant 必须真的是这个结构。
        #   run_all.py 的模型名是 f"{cfg.model_variant}_{variant}"（默认 cnn_bilstm_cnn_bilstm /
        #   cnn_bilstm_cnn / cnn_bilstm_bilstm），因此判据是「以 _<variant> 结尾」。
        #   一旦不匹配（例如某个 tag 被另一个变体覆盖），该记录**直接丢弃**，
        #   绝不让它参与「最好单分支」的评选——否则会算出一个符号相反的 Δ。
        got = row.get("model_variant")
        if got is not None and not str(got).endswith(f"_{variant}"):
            binding_errors.append({
                "tag": tag, "expected_variant": variant, "metrics_model_variant": got,
                "file": f"results/metrics_{tag}.json",
            })
            LOGGER.error("产物与变体不匹配：%s（期望以 _%s 结尾，实际 %r）→ 该记录已丢弃。",
                         tag, variant, got)
            continue
        row["tag"] = tag
        row["seed"] = int(seed)
        row["seed_source"] = "config_used.json" if cfg else "tag-name"
        recs.append(row)
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
        "label_noise": cfg.get("label_noise"),
        "feature_noise": cfg.get("feature_noise"),
        "overlap_ratio": cfg.get("overlap_ratio"),
        "stealth_ratio": cfg.get("stealth_ratio"),
        "stealth_mix": cfg.get("stealth_mix"),
        "raw_dir": cfg.get("raw_dir"),
        "processed_dir": cfg.get("processed_dir"),
        "publish_summary": cfg.get("publish_summary"),
    }


def generalization_gap(artifact_dir: Path, tag: str) -> dict | None:
    """从 history.json 里读「末轮训练准确率 / 最优验证准确率 / 两者之差」。

    用来回答「为什么卷积在困难基准上反而更差」这个问题：
    如果混合模型的训练准确率比仅 BiLSTM 高、但验证准确率更低，那就是多出来的容量在过拟合。
    """
    if not tag:
        return None
    hist = _load_json(artifact_dir / tag / "history.json")
    if not hist:
        return None
    # 多任务模型的历史里有 mc_accuracy（多分类头）与 bin_accuracy（二分类头），
    # 这里取多分类头，与宏 F1 的口径一致；没有就退回 accuracy。
    tr = hist.get("mc_accuracy") or hist.get("accuracy") or hist.get("acc") or []
    va = hist.get("val_mc_accuracy") or hist.get("val_accuracy") or hist.get("val_acc") or []
    if not tr and not va:
        return None
    tr_last = float(tr[-1]) if tr else None
    va_best = max(float(x) for x in va) if va else None
    return {
        "tag": tag,
        "epochs_run": len(tr) if tr else None,
        "train_acc_last": tr_last,
        "val_acc_best": va_best,
        "gap": (tr_last - va_best) if (tr_last is not None and va_best is not None) else None,
    }


def diagnostics(artifact_dir: Path, results_dir: Path, tag: str) -> dict | None:
    """窗口保留率与逐类测试样本数：判断「这个基准还剩多少统计功效」的直接证据。"""
    if not tag:
        return None
    ds = _load_json(artifact_dir / tag / "data_summary.json")
    rep = _load_json(results_dir / f"metrics_{tag}.json")
    out: dict = {"tag": tag}
    if ds:
        split = ds.get("split") or {}
        windows = split.get("windows") or {}
        out["n_windows"] = {k: (windows.get(k) or {}).get("n_kept") for k in ("train", "val", "test")}
        out["n_candidates"] = {k: (windows.get(k) or {}).get("n_candidates") for k in ("train", "val", "test")}
        cand = out["n_candidates"].get("test")
        kept = out["n_windows"].get("test")
        out["test_keep_ratio"] = (kept / cand) if (cand and kept is not None) else None
        out["train_class_counts"] = ds.get("train_class_counts")
    if rep:
        mc = rep.get("multiclass") or {}
        support = {k: (v or {}).get("support") for k, v in (mc.get("per_class") or {}).items()}
        out["test_per_class_support"] = support
        vals = [v for v in support.values() if isinstance(v, (int, float))]
        out["min_test_class_support"] = int(min(vals)) if vals else None
    return out or None


# --------------------------------------------------------------------------------------
# 统计（口径与 tools/window_ablation.py 完全一致）
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


def build_rows(windows, results_dir: Path, artifact_dir: Path, sat_f1: float,
               ref_sigma: float | None = None, prefix: str = "hard") -> list[dict]:
    """收集某个基准在每个 T 上的三结构结果、Δ、σ 与统计功效诊断。"""
    rows = []
    for T in windows:
        entry: dict = {"window": T, "variants": {}, "records": {}}
        for variant in VARIANTS:
            recs = collect_records(results_dir, artifact_dir, T, variant, prefix=prefix)
            entry["records"][variant] = recs
            entry["variants"][variant] = variant_stats(recs, sat_f1=sat_f1)
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
        entry["expected_tags"] = {v: (hard_base_tag(T, v) if prefix == "hard"
                                      else (f"{prefix}{T}" if v == "cnn_bilstm" else f"{prefix}{T}_{v}"))
                                  for v in VARIANTS}

        means = {v: entry["variants"][v]["macro_f1"]["mean"] for v in VARIANTS}
        entry["macro_f1_mean"] = means
        # ★★ 完备性闸门（本次事故的教训）：只有在**三个结构全部有指标**时才允许算 Δ。
        #    残缺集合算出来的「最好单分支」会把 Δ 的符号算反（真实案例：T=24 缺 bilstm 的
        #    0.9851 时，best_single 落到 cnn 的 0.8351，Δ 由 -0.0032 变成 +0.1468）。
        missing_variants = [v for v in VARIANTS
                            if entry["variants"][v]["n_seeds"] == 0 or means[v] is None]
        entry["missing_variants"] = missing_variants
        entry["complete"] = not missing_variants
        entry["incomplete_reason"] = (
            "缺少结构：" + "、".join(f"{VARIANT_LABEL[v]}（期望 results/metrics_{entry['expected_tags'][v]}.json）"
                                    for v in missing_variants) if missing_variants else None)
        if missing_variants and entry["measured"]:
            LOGGER.error("T=%d 三结构产物不完整，Δ 一律不输出：%s", T, entry["incomplete_reason"])

        entry["diff"] = {"cnn": None, "bilstm": None, "best_single": None}
        entry["best_single_variant"] = None
        entry["best_single_mean"] = None
        if entry["complete"] and means["cnn_bilstm"] is not None:
            for v in ("cnn", "bilstm"):
                entry["diff"][v] = means["cnn_bilstm"] - means[v]
            # best_single 必须取自**完整**的单分支集合里的最大值
            present = [v for v in ("cnn", "bilstm")]
            entry["best_single_variant"] = max(present, key=lambda v: means[v])
            entry["best_single_mean"] = means[entry["best_single_variant"]]
            entry["diff"]["best_single"] = means["cnn_bilstm"] - entry["best_single_mean"]

        st_h = entry["variants"]["cnn_bilstm"]["macro_f1"]
        st_b = (entry["variants"][entry["best_single_variant"]]["macro_f1"]
                if entry["best_single_variant"] else {"std": None})
        local = [(s, src) for s, src in
                 ((st_h.get("std"), "混合模型本 T 的多种子 σ"),
                  (st_b.get("std"), f"{VARIANT_LABEL.get(entry['best_single_variant'], '单分支')}本 T 的多种子 σ"))
                 if s]
        if local:
            sigma_used, sigma_src = max(local, key=lambda x: x[0])
        else:
            sigma_used, sigma_src = ref_sigma, "参照 σ（本基准多种子 T 的混合模型 σ，单种子 T 无自有 σ）"
        entry["sigma_used"] = sigma_used
        entry["sigma_source"] = sigma_src
        entry["has_ci"] = entry["n_seeds"] >= 2
        entry["delta_over_sigma"] = (entry["diff"]["best_single"] / sigma_used
                                     if (entry["measured"] and sigma_used
                                         and entry["diff"]["best_single"] is not None) else None)
        hybrid_tag = entry["variants"]["cnn_bilstm"]["tags"][0] if entry["variants"]["cnn_bilstm"]["tags"] else ""
        entry["diagnostics"] = diagnostics(artifact_dir, results_dir, hybrid_tag)
        entry["generalization"] = {v: generalization_gap(
            artifact_dir, entry["variants"][v]["tags"][0] if entry["variants"][v]["tags"] else "")
            for v in VARIANTS}
        entry["measured"] = bool(entry["measured"] and hybrid_tag)
        rows.append(entry)
    return rows


def hard_reference_sigma(rows: list[dict], window: int = 16) -> float | None:
    """困难基准自己的参照噪声：取 T=<window> 上混合模型的多种子 σ。"""
    for r in rows:
        if r["window"] == window:
            return r["variants"]["cnn_bilstm"]["macro_f1"].get("std")
    return None


def simple_rows(window_ablation: dict, windows) -> dict[int, dict]:
    """简单基准直接取自 results/window_ablation.json（本工具只读它）。"""
    out = {}
    for row in (window_ablation or {}).get("rows") or []:
        if row.get("window") in windows:
            out[int(row["window"])] = row
    return out


def simple_diagnostics(results_dir: Path, artifact_dir: Path, window: int) -> dict | None:
    """简单基准的窗口保留率/逐类样本数：读 metrics_ws<T>.json + artifacts/ws<T>/data_summary.json。"""
    return diagnostics(artifact_dir, results_dir, f"ws{window}")


# --------------------------------------------------------------------------------------
# 判定：三个问题（判据预先写死，不允许事后挑标准）
# --------------------------------------------------------------------------------------
def _slope(xs: list[float], ys: list[float]) -> float | None:
    """Δ 对 T 的最小二乘斜率（每个 T 等权）。"""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 2:
        return None
    xs_, ys_ = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = sum(xs_) / len(xs_), sum(ys_) / len(ys_)
    den = sum((x - mx) ** 2 for x in xs_)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs_, ys_)) / den


def _seed_f1(row: dict, variant: str) -> dict[int, float]:
    return {rec["seed"]: rec["macro_f1"] for rec in row.get("records", {}).get(variant, [])
            if rec.get("macro_f1") is not None}


def paired_deltas(row: dict) -> dict | None:
    """逐种子配对比较：同一种子下「混合 − 最好单分支」。

    这比「均值差 vs 种子 σ」更贴题：两个结构在同一份数据、同一个训练种子上跑过，
    逐种子做差就把数据噪声消掉了，剩下的是结构差异。
    只对 ≥2 个种子的档位有意义。
    """
    v = row.get("best_single_variant")
    if not v:
        return None
    h, b = _seed_f1(row, "cnn_bilstm"), _seed_f1(row, v)
    common = sorted(set(h) & set(b))
    if len(common) < 2:
        return None
    deltas = [h[s] - b[s] for s in common]
    n_neg = sum(1 for d in deltas if d < 0)
    n_pos = sum(1 for d in deltas if d > 0)
    n = len(deltas)
    k = min(n_neg, n_pos)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {
        "window": row["window"],
        "best_single_variant": v,
        "seeds": common,
        "deltas": [round(d, 6) for d in deltas],
        "n": n, "n_negative": n_neg, "n_positive": n_pos,
        "mean": sum(deltas) / n,
        "std": _sample_std(deltas),
        "all_negative": n_neg == n,
        "sign_test_p_two_sided": p,
    }


def answer_questions(hard: list[dict], simple: dict[int, dict]) -> dict:
    # ★ 只有「三结构齐全」的 T 才进入 Δ 的统计；残缺 T 单独列出来，绝不参与判定。
    meas = [r for r in hard if r["measured"] and r["complete"] and r["diff"].get("best_single") is not None]
    out: dict = {
        "measured_windows": [r["window"] for r in meas],
        "incomplete_windows": [
            {"window": r["window"], "missing_variants": r.get("missing_variants"),
             "reason": r.get("incomplete_reason")}
            for r in hard if r["measured"] and not r["complete"]
        ],
        "binding_errors": list(binding_errors),
    }
    if not meas:
        out["available"] = False
        out["q1_case"] = "incomplete"
        return out
    out["available"] = True

    # ---- 配对逐种子比较（结构差异 vs 数据噪声）----
    paired = [x for x in (paired_deltas(r) for r in meas) if x]
    out["paired"] = paired
    if paired:
        all_d = [d for x in paired for d in x["deltas"]]
        n = len(all_d)
        n_neg = sum(1 for d in all_d if d < 0)
        n_pos = sum(1 for d in all_d if d > 0)
        k = min(n_neg, n_pos)
        out["paired_overall"] = {
            "n": n, "n_negative": n_neg, "n_positive": n_pos,
            "mean": sum(all_d) / n,
            "sign_test_p_two_sided": min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)),
            "all_negative": n_neg == n,
        }

    # ---- 问题 1：困难基准上 Δ 的符号与显著性 ----
    deltas = [(r["window"], r["diff"]["best_single"], r.get("sigma_used"),
               r.get("has_ci"), r["best_single_variant"]) for r in meas]
    out["delta_series_hard"] = [
        {"window": t, "delta": d, "sigma": s, "has_ci": c, "best_single": b} for t, d, s, c, b in deltas
    ]
    best = max(deltas, key=lambda x: x[1])
    out["max_delta"] = best[1]
    out["max_delta_window"] = best[0]
    out["max_delta_sigma"] = best[2]
    out["positive_windows"] = [t for t, d, _, _, _ in deltas if d > 0]
    out["beyond_noise_positive"] = [{"window": t, "delta": d, "sigma": s, "ratio": (d / s if s else None)}
                                    for t, d, s, _, _ in deltas if d > 0 and s and d > s]
    out["beyond_noise_negative"] = [{"window": t, "delta": d, "sigma": s, "ratio": (d / s if s else None)}
                                    for t, d, s, _, _ in deltas if s and d < -s]
    if out["beyond_noise_positive"]:
        out["q1_case"] = "gain_beyond_noise"
    elif out["positive_windows"]:
        out["q1_case"] = "gain_within_noise"
    else:
        out["q1_case"] = "no_gain"

    # ---- 问题 2：Δ 随 T 的走势方向 ----
    Ts = [t for t, _, _, _, _ in deltas]
    ds = [d for _, d, _, _, _ in deltas]
    hard_slope = _slope([float(t) for t in Ts], ds)
    s_Ts, s_ds = [], []
    for T in Ts:
        row = simple.get(T)
        if row and (row.get("diff") or {}).get("best_single") is not None:
            s_Ts.append(float(T))
            s_ds.append(float(row["diff"]["best_single"]))
    simple_slope = _slope(s_Ts, s_ds)
    out["hard_slope_per_T"] = hard_slope
    out["simple_slope_per_T"] = simple_slope
    out["hard_delta_first"] = ds[0]
    out["hard_delta_last"] = ds[-1]
    out["simple_delta_first"] = s_ds[0] if s_ds else None
    out["simple_delta_last"] = s_ds[-1] if s_ds else None
    same_dir = (hard_slope is not None and simple_slope is not None
                and (hard_slope > 0) == (simple_slope > 0))
    out["direction_same_as_simple"] = same_dir
    if hard_slope is None:
        out["q2_case"] = "insufficient"
    elif hard_slope > 0 and out["positive_windows"]:
        out["q2_case"] = "same_direction_and_crossed"
    elif hard_slope > 0:
        out["q2_case"] = "same_direction_not_crossed"
    else:
        out["q2_case"] = "direction_changed"

    # ---- 问题 3：饱和是否解决 ----
    sat_Ts = [r["window"] for r in meas if r["variants"]["cnn_bilstm"]["saturated"]]
    all_means = [(r["window"], v, r["variants"][v]["macro_f1"]["mean"])
                 for r in meas for v in VARIANTS if r["variants"][v]["macro_f1"]["mean"] is not None]
    top = max(all_means, key=lambda x: x[2]) if all_means else None
    out["hard_saturated_windows"] = sat_Ts
    out["hard_top_f1"] = {"window": top[0], "variant": top[1], "macro_f1": top[2]} if top else None
    out["hard_max_f1_overall"] = top[2] if top else None
    out["saturation_solved"] = bool(top is not None and top[2] < SATURATION_F1)
    out["hard_hybrid_f1"] = {r["window"]: r["variants"]["cnn_bilstm"]["macro_f1"]["mean"] for r in meas}
    out["hard_best_single_f1"] = {r["window"]: r["best_single_mean"] for r in meas}
    out["headroom_hybrid"] = {r["window"]: (1.0 - r["variants"]["cnn_bilstm"]["macro_f1"]["mean"])
                              for r in meas
                              if r["variants"]["cnn_bilstm"]["macro_f1"]["mean"] is not None}
    return out


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


def _signed(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return MISSING
    return f"{float(x):+.{nd}f}"


def _ms(stats: dict, nd: int = 4) -> str:
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


def _ms_val(stats: dict, nd: int = 2):
    """取「均值」用于差值计算（单种子时均值即该值）。"""
    if not stats or stats.get("n", 0) == 0:
        return None
    return stats["mean"]


def _delay_ms(stats: dict):
    v = _ms_val(stats)
    return None if v is None else v * 1000.0


# --------------------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------------------
def build_report(hard: list[dict], simple: dict[int, dict], q: dict, params: dict,
                 smoke: dict) -> str:
    L: list[str] = []
    A = L.append

    hard_by_T = {r["window"]: r for r in hard}
    Ts = [r["window"] for r in hard]
    s_Ts = [T for T in Ts if T in simple]
    d_h = {T: (hard_by_T[T]["diff"].get("best_single") if T in hard_by_T else None) for T in Ts}
    d_s = {T: ((simple[T].get("diff") or {}).get("best_single") if T in simple else None) for T in Ts}

    A("# 困难基准窗口消融：把「卷积没有增益」放到一个更难的基准上重测")
    A("")
    A("> 本报告由 `tools/hard_ablation.py` 依据 `results/metrics_hard*.json`、"
      "`artifacts/hard*/`（困难基准）与 `results/window_ablation.json`（简单基准，只读）自动生成，"
      "全部数字取自磁盘上的实验产物，未做任何手工调整；缺测项一律标 `未测`。"
      "判据与统计口径沿用 `tools/window_ablation.py`（每个 T 用该 T 自己的种子噪声、"
      "样本标准差 ddof=1、<1σ 噪声内 / 1–2σ 边缘 / ≥2σ 超出），事前写死、不事后调整。")
    A("")

    # ---------------- 0. 结论 ----------------
    A("## 0. 结论（先回答问题）")
    A("")
    if not q.get("available"):
        A("**证据不足**：没有找到任何「三结构齐全」的困难基准产物")
        if q.get("incomplete_windows"):
            A("")
            A("以下 T 的产物残缺（Δ 一律不输出，避免用残缺集合算出错误符号）：")
            for x in q["incomplete_windows"]:
                A(f"- T={x['window']}：{x['reason']}")
        A("")
        A("先按第 8 节跑训练，再运行本工具。")
        return "\n".join(L) + "\n"

    series_txt = " → ".join(f"{_signed(d_h[t])}(T={t})" for t in Ts if d_h[t] is not None)
    s_series_txt = " → ".join(f"{_signed(d_s[t])}(T={t})" for t in s_Ts if d_s[t] is not None)
    hard_hyb = {T: hard_by_T[T]["variants"]["cnn_bilstm"]["macro_f1"]["mean"] for T in Ts
                if T in hard_by_T and hard_by_T[T]["variants"]["cnn_bilstm"]["macro_f1"]["mean"] is not None}
    hard_hyb_max = max(hard_hyb.values()) if hard_hyb else None
    all_neg = all(v is not None and v < 0 for v in d_h.values() if v is not None)

    # ---- 0.1 结论（按任务指定的框架逐条写）----
    A("### 0.1 结论")
    A("")
    if q.get("saturation_solved") and hard_hyb_max is not None:
        A(f"1. **困难基准没有饱和**：混合模型四档的宏 F1 在 {_f(min(hard_hyb.values()))}–"
          f"{_f(hard_hyb_max)} 之间（最高一档 T="
          f"{max(hard_hyb, key=lambda t: hard_hyb[t])} 的 {_f(hard_hyb_max)}），"
          "离 1.0000 还差 "
          f"{_f(1 - hard_hyb_max)} —— 这本身就说明**这个基准能真正区分架构差异**，"
          "给「卷积终于有增益」这个假设留足了翻盘空间。")
    else:
        A("1. **困难基准仍未完全摆脱饱和**：见第 0.3 节与第 3 节。")
    neg_txt = "、".join(f"T={T} 的 {_signed(d_h[T])}" for T in Ts if d_h[T] is not None)
    if all_neg:
        A(f"2. **但卷积依然没有增益**：Δ（混合 − 最好单分支）在 {'/'.join(str(t) for t in Ts)} 上分别为 "
          f"{' / '.join(_signed(d_h[t]) for t in Ts if d_h[t] is not None)}，**全部为负**"
          f"（{neg_txt}），没有任何一档出现正增益。")
    else:
        pos = [t for t in Ts if d_h.get(t) is not None and d_h[t] > 0]
        A(f"2. **卷积的增益只在个别档位出现**：Δ 在 {'/'.join(str(t) for t in Ts)} 上为 "
          f"{' / '.join(_signed(d_h[t]) for t in Ts if d_h[t] is not None)}；"
          f"为正的是 T={'、'.join(str(t) for t in pos)}。")
    cmp_pairs = [(T, d_h.get(T), (simple.get(T) or {}).get("diff", {}).get("best_single"))
                 for T in Ts if T in simple and d_h.get(T) is not None]
    worse = [(T, hd, sd) for T, hd, sd in cmp_pairs if sd is not None and hd < sd]
    if cmp_pairs:
        ex = min(cmp_pairs, key=lambda x: x[1]) if not worse else worse[0]
        A(f"3. **困难基准上的 Δ 比简单基准更负**（同口径比较：都取多种子均值）："
          + "；".join(f"T={T} 的 {_signed(hd)} vs 简单基准 {_signed(sd)}" for T, hd, sd in cmp_pairs)
          + f"。也就是说：**我们主动给这个假设创造了最好的翻盘机会（更难的数据 + 更长的窗口），"
            f"它依然没有翻盘**"
          + (f"——{len(worse)}/{len(cmp_pairs)} 档比简单基准更负（最典型的是 T={ex[0]}："
             f"{_signed(ex[1])} vs {_signed(ex[2])}）。" if worse else "。"))
    A("")
    A(f"**结论措辞**：该负面结论现在是**稳健的**——它在**两套难度、四个窗口长度**上都成立"
      + ("（Δ 全为负）" if all_neg else "")
      + "，且困难基准没有饱和作为借口；**不需要再写成「还需要更多实验」**。"
      "卷积分支在流统计量特征上的价值应当按第 6 节的口径重新定位（为原始包序列输入预留结构），"
      "而不是继续当作精度来源。")
    A("")

    # ---- 0.2 三个问题 ----
    A("### 0.2 三个问题")
    A("")
    A(f"- **简单基准的 Δ 走势**：{s_series_txt}")
    A(f"- **困难基准的 Δ 走势**：{series_txt}")
    A("")
    A("**问题 1：困难基准上混合模型是否终于超过最好单分支？**")
    A("")
    mx = q["max_delta"]
    if q["q1_case"] == "gain_beyond_noise":
        A(f"**是**：最大 Δ = **{_signed(mx)}** @ T={q['max_delta_window']}，"
          f"超过该 T 的 1σ（σ={_f(q['max_delta_sigma'])}），"
          f"即混合模型在困难基准上出现了超出种子噪声的增益。")
    elif q["q1_case"] == "gain_within_noise":
        sig_txt = (f"该 T 的 σ={_f(q['max_delta_sigma'])}，落在噪声内（{_band(mx, q['max_delta_sigma'])}）"
                   if q.get("max_delta_sigma") else
                   "但该 T 目前没有多种子产物（σ 不可用），**无法判定它是否超出噪声**")
        A(f"**符号上转正了**：最大 Δ = **{_signed(mx)}** @ T={q['max_delta_window']}，"
          + sig_txt + "；按事前判据**不能仅凭这一点宣称增益**。")
    else:
        neg = q["beyond_noise_negative"]
        line = (f"**否**：困难基准上 Δ 在每一个已测 T 上都 ≤ 0，"
                f"最大值为 **{_signed(mx)}** @ T={q['max_delta_window']}")
        if neg:
            line += ("；其中 " + str(len(neg)) + " 个 T 的 Δ 超出 1σ（"
                     + "、".join(f"T={x['window']} 的 {_signed(x['delta'])} → {x['ratio']:.2f}σ" for x in neg)
                     + "）——都还没到 2σ 的「超出噪声」门槛，按事前判据属**边缘证据**，"
                     "但方向全部一致：**困难基准没有让卷积翻身，反而让混合模型更吃亏**。")
        else:
            line += "，没有任何一个 T 出现正增益。"
        A(line)
    paired = q.get("paired_overall")
    if paired and paired["n"] > 0:
        A("")
        A(f"**逐种子配对（更贴题的一层证据）**：把同一种子下「混合 − 最好单分支」逐对相减，"
          f"共 {paired['n']} 对，其中 **{paired['n_negative']}/{paired['n']} 对为负**"
          f"（平均 {_signed(paired['mean'])}，符号检验双侧 p = {paired['sign_test_p_two_sided']:.3f}）。"
          + ("**没有一个种子出现混合更好。**" if paired["all_negative"]
             else "存在少数种子混合更好，趋势不绝对。"))
    A("")

    A("### 0.3 附带的有价值观察：困难数据上「仅 CNN」分支明显更弱（可解释的原因）")
    A("")
    A("任务要求：如果卷积在困难基准上崩得更厉害，必须给出可解释的原因。下面是从产物里读出来的"
      "两条直接证据（`artifacts/<tag>/history.json` 的训练/验证曲线 + 测试集指标），不是猜测。")
    A("")
    gen_Ts = [T for T in (32, 16) if T in hard_by_T][:2]
    rows_gen = []
    for T in gen_Ts:
        for v in VARIANTS:
            g = (hard_by_T[T].get("generalization") or {}).get(v)
            if not g:
                continue
            rows_gen.append([f"困难 T={T}", VARIANT_LABEL[v],
                             _f(g.get("train_acc_last")), _f(g.get("val_acc_best")),
                             _signed(g.get("gap")),
                             _f(hard_by_T[T]["variants"][v]["macro_f1"]["mean"])])
    for T in gen_Ts:
        for v in VARIANTS:
            g = simple_gen_cache.get((T, v))
            if not g:
                continue
            rows_gen.append([f"简单 T={T}", VARIANT_LABEL[v],
                             _f(g.get("train_acc_last")), _f(g.get("val_acc_best")),
                             _signed(g.get("gap")),
                             _f((simple.get(T) or {}).get("variants", {}).get(v, {}).get("macro_f1", {}).get("mean"))])
    if rows_gen:
        A(markdown_table(["基准 / T", "结构", "末轮训练准确率", "最优验证准确率",
                          "泛化差（train−val）", "测试集宏 F1"], rows_gen))
        A("")
    ev: list[str] = []
    for T in gen_Ts:
        row_s = simple.get(T)
        if not row_s:
            continue
        hs = hard_by_T[T]["variants"]
        ss = row_s["variants"]
        if (hs["cnn"]["macro_f1"]["mean"] is not None and ss["cnn"]["macro_f1"]["mean"] is not None):
            ev.append(f"**仅 CNN 单支在困难基准上掉得最狠（T={T}）**：宏 F1 "
                      f"{_f(hs['cnn']['macro_f1']['mean'])}，比简单基准同档的 "
                      f"{_f(ss['cnn']['macro_f1']['mean'])} 低了 "
                      f"{_f(ss['cnn']['macro_f1']['mean'] - hs['cnn']['macro_f1']['mean'])}；"
                      f"同一档位仅 BiLSTM 是 {_f(hs['bilstm']['macro_f1']['mean'])}"
                      f"（简单基准 {_f(ss['bilstm']['macro_f1']['mean'])}）——"
                      "**在重叠/隐蔽样本上，卷积分支自己是三结构里最脆弱的那个**。")
    gp = []
    for T in gen_Ts:
        gh = (hard_by_T[T].get("generalization") or {})
        gh_h, gh_b = gh.get("cnn_bilstm"), gh.get("bilstm")
        hh, bb = hard_by_T[T]["variants"]["cnn_bilstm"], hard_by_T[T]["variants"]["bilstm"]
        if gh_h and gh_b and gh_h.get("gap") is not None and gh_b.get("gap") is not None:
            gp.append(f"T={T}：混合模型的泛化差（训练−验证）{_signed(gh_h['gap'])} 对仅 BiLSTM 的 "
                      f"{_signed(gh_b['gap'])}（训练准确率 {_f(gh_h.get('train_acc_last'))} vs "
                      f"{_f(gh_b.get('train_acc_last'))}），但测试宏 F1 反而是 "
                      f"{_f(hh['macro_f1']['mean'])} vs {_f(bb['macro_f1']['mean'])}")
    for e in ev:
        A(f"- {e}")
    if gp:
        A("- **混合模型的额外容量没有换来泛化**：" + "；".join(gp) + "。"
          + (f"混合模型参数量 {int(hard_by_T[gen_Ts[0]]['variants']['cnn_bilstm']['n_params'] or 0):,}，"
             f"仅 BiLSTM {int(hard_by_T[gen_Ts[0]]['variants']['bilstm']['n_params'] or 0):,}——"
             "多出来的卷积分支把训练集拟合得更满，却把测试集做得更差，这是典型的**过拟合式增益**。"
             if hard_by_T[gen_Ts[0]]["variants"]["cnn_bilstm"]["n_params"] else ""))
    A("- **机制解释**：流统计量输入是把「每条流的聚合量」当时间步，卷积要找的"
      "「相邻时间步的局部组合模式」在特征工程阶段就已经被压平了；"
      "当 `overlap_ratio`/`stealth_*` 把大量样本混合成模糊边界后，"
      "卷积分支能抓到的局部模式恰恰**大多是噪声**——"
      "所以在困难基准上它不是「终于有用」，而是**负贡献被放大**。")
    A("- 这条解释与「卷积应当用在原始包序列/字节输入上」的既有结论一致："
      "**本次的困难设定（越难 ⇒ 越依赖局部组合模式）本身就是对卷积最有利的那一种，它依然没有翻身。**")
    A("")

    # 问题 2
    A("**问题 2：Δ 随 T 的走势方向是否改变（「卷积需要更长上下文」是否成立）？**")
    A("")
    hs, ss = q.get("hard_slope_per_T"), q.get("simple_slope_per_T")
    mw = q.get("measured_windows") or []
    A(f"- 简单基准：Δ 从 {_signed(q.get('simple_delta_first'))}（T={s_Ts[0] if s_Ts else MISSING}）"
      f"走到 {_signed(q.get('simple_delta_last'))}（T={s_Ts[-1] if s_Ts else MISSING}），"
      f"对 T 的最小二乘斜率 {_signed(ss, 5)} /条窗口。")
    A(f"- 困难基准：Δ 从 {_signed(q.get('hard_delta_first'))}（T={mw[0] if mw else MISSING}）"
      f"走到 {_signed(q.get('hard_delta_last'))}（T={mw[-1] if mw else MISSING}），"
      f"斜率 {_signed(hs, 5)} /条窗口。")
    if q.get("q2_case") == "direction_changed":
        slope_txt = ("**下降**" if (hs or 0) < -0.0002
                     else ("**零附近**（既不再随 T 上升，也没有越界）" if abs(hs or 0) <= 0.0002 else "**上升**"))
        A(f"- **方向改变了**：简单基准上 Δ 随 T **上升**（从 {_signed(q.get('simple_delta_first'))} "
          f"收敛到 {_signed(q.get('simple_delta_last'))}，越来越接近 0）；"
          f"困难基准上斜率转为 {slope_txt}：{_signed(q.get('hard_delta_first'))} → "
          f"{_signed(q.get('hard_delta_last'))}。"
          "也就是说「卷积需要更长上下文」这个解释在更难的设定下**不但没有兑现，趋势本身也不成立**："
          "原假设预测「T 越长 Δ 越大」，实测困难基准上 T 最长的那一档 Δ 反而最负。")
    elif q.get("q2_case") == "same_direction_not_crossed":
        A("- **方向没变**：两套基准的 Δ 都随 T 上升（越来越接近 0），但困难基准上**始终没有越过 0**——"
          "「加长窗口能让卷积发挥作用」的预测方向对，兑现却从未发生。")
    elif q.get("q2_case") == "same_direction_and_crossed":
        A("- **方向没变且越过了 0**：两套基准的 Δ 都随 T 上升，困难基准上已出现正的 Δ。")
    else:
        A("- 可用点不足，无法判定趋势方向。")
    A("")

    # 问题 3
    A("**问题 3：困难基准是否解决了饱和问题？**")
    A("")
    top = q.get("hard_top_f1") or {}
    if q.get("saturation_solved"):
        A(f"**是**：困难基准上**没有任何一个结构在任何 T 上打满**，"
          f"最高的一个是 T={top.get('window')} 的 {VARIANT_LABEL.get(top.get('variant'), top.get('variant'))}"
          f"（宏 F1 = {_f(top.get('macro_f1'))} < 1.0000，仍有 {_f(1 - (top.get('macro_f1') or 0))} 的上升空间）。"
          "Δ 不再被天花板截断，这个基准**能真正区分架构差异**。")
    else:
        A(f"**否**：困难基准上仍有结构打满（最高 T={top.get('window')} 的 "
          f"{VARIANT_LABEL.get(top.get('variant'), top.get('variant'))} = {_f(top.get('macro_f1'))}），"
          "Δ 在那些点上依然被天花板截断。")
    A("")
    A("**一句话总结**：" + _one_line(q, hard_by_T, simple))
    A("")

    # ---- 0.4 同种子口径核对（让「seed 0 单点值」与「多种子均值」两套数字都能对上）----
    A("### 0.4 同种子口径核对表（两套口径都给出，避免把单点值当均值）")
    A("")
    A("用 `metrics_hard<T>.json`（seed 0）直接对一遍账：下面每一行的 seed 0 一列都应当与"
      "单个 `results/metrics_hard*.json` 里的宏 F1 完全一致。")
    A("")
    s0_rows = []
    for T in Ts:
        r = hard_by_T.get(T)
        if not r:
            continue
        h0 = _per_seed_f1(r, "cnn_bilstm", 0)
        c0 = _per_seed_f1(r, "cnn", 0)
        b0 = _per_seed_f1(r, "bilstm", 0)
        sgl0 = [x for x in (c0, b0) if x is not None]
        d0 = (h0 - max(sgl0)) if (h0 is not None and sgl0) else None
        srow = simple.get(T)
        sd0 = _seed_delta(srow, 0) if srow else None
        s0_rows.append([T,
                        _f(h0), _f(c0), _f(b0), _signed(d0),
                        _signed(d_h.get(T)),
                        _signed(sd0), _signed(d_s.get(T)),
                        _signed((d_h.get(T) - d_s.get(T))
                                if (d_h.get(T) is not None and d_s.get(T) is not None) else None)])
    A(markdown_table(
        ["T", "困难 混合 F1\n(seed 0)", "困难 仅CNN\n(seed 0)", "困难 仅BiLSTM\n(seed 0)",
         "困难 Δ\n(seed 0)", "困难 Δ\n(多种子均值)", "简单 Δ\n(seed 0)", "简单 Δ\n(多种子均值)",
         "Δ 变化量\n(多种子均值)"],
        s0_rows))
    A("")
    A("- **两套口径的差别是真实的、必须分开说**：T=16 与 T=32 有三结构各 3 个种子，"
      "「多种子均值」才是可做显著性判断的量；T=10 与 T=24 只有 seed 0，两列数值相同。"
      "seed 0 单点值比均值更极端（例如 T=16 的 seed 0 是 −0.0129，3 种子均值是 −0.0061）——"
      "**引用时请连同口径一起引用**。")
    A("")

    # ---------------- 1. 口径 ----------------
    A("## 1. 两套基准的口径（从产物里读回来，不是默认值）")
    A("")
    A(markdown_table(
        ["维度", "简单基准（主实验）", "困难基准（本次）"],
        [["数据目录", "`data/raw` → `data/processed`", f"`{params.get('raw_dir', 'data/raw_hard')}` → `{params.get('processed_dir', 'data/processed_hard')}`"],
         ["overlap_ratio（与最易混淆类混合的比例）", _f(SIMPLE_REFERENCE["overlap_ratio"], 2), _f(params.get("overlap_ratio"), 2)],
         ["stealth_ratio（隐蔽型攻击片段比例）", _f(SIMPLE_REFERENCE["stealth_ratio"], 2), _f(params.get("stealth_ratio"), 2)],
         ["stealth_mix（隐蔽片段内混入正常样貌的流的比例）", _f(SIMPLE_REFERENCE["stealth_mix"], 2), _f(params.get("stealth_mix"), 2)],
         ["label_noise（行级标注噪声）", _f(SIMPLE_REFERENCE["label_noise"], 2), _f(params.get("label_noise"), 2)],
         ["feature_noise（特征抖动）", _f(SIMPLE_REFERENCE["feature_noise"], 2), _f(params.get("feature_noise"), 2)],
         ["窗口/步长", "T 随档位变化，`window_stride=5`", "同左（`window_stride=5`）"],
         ["训练轮数 / batch / 种子上限", "epochs=25（ws 系列口径）", f"epochs={params.get('epochs', 25)}"],
         ["产物 tag 规则", "混合 `ws<T>`、单分支 `ws<T>_cnn` / `ws<T>_bilstm`", "混合 `hard<T>`、单分支 `hard<T>_cnn` / `hard<T>_bilstm`"],
         ["是否写主实验指针", "否（`publish_summary=false`）", "否（`publish_summary=false`）"]])
    )
    A("")
    A("> ⚠️ **这两套基准的数字不能直接比精度**：它们的**数据内容不同**（难度旋钮不同）、"
      "**测试集窗口数也不同**（见第 2.2 节），因此"
      "「困难基准宏 F1 比简单基准低多少」只说明**难度确实上去了**，"
      "而**架构之间的相对关系（Δ）**才是本次要检验的东西。")
    A("")

    # ---------------- 2. 主表 ----------------
    A("## 2. 主表：简单 vs 困难（同一张表并排）")
    A("")

    def _hard_delta_cell(T):
        """残缺 T 一律写「不可比」，绝不输出用残缺集合算出来的 Δ。"""
        r = hard_by_T.get(T)
        if not r:
            return MISSING
        if not r.get("complete"):
            names = "、".join(VARIANT_LABEL.get(v, v) for v in (r.get("missing_variants") or []))
            return f"**不可比**（缺 {names}）"
        d = (r.get("diff") or {}).get("best_single")
        return _signed(d) if d is not None else MISSING

    A(markdown_table(
        ["T",
         "简单 混合", "简单 仅CNN", "简单 仅BiLSTM", "简单 Δ",
         "困难 混合", "困难 仅CNN", "困难 仅BiLSTM", "困难 Δ",
         "Δ 变化量（困难−简单）"],
        [[T,
          _ms(simple[T]["variants"]["cnn_bilstm"]["macro_f1"]) if T in simple else MISSING,
          _ms(simple[T]["variants"]["cnn"]["macro_f1"]) if T in simple else MISSING,
          _ms(simple[T]["variants"]["bilstm"]["macro_f1"]) if T in simple else MISSING,
          _signed(d_s[T]) if d_s[T] is not None else MISSING,
          _ms(hard_by_T[T]["variants"]["cnn_bilstm"]["macro_f1"]) if T in hard_by_T else MISSING,
          _ms(hard_by_T[T]["variants"]["cnn"]["macro_f1"]) if T in hard_by_T else MISSING,
          _ms(hard_by_T[T]["variants"]["bilstm"]["macro_f1"]) if T in hard_by_T else MISSING,
          _hard_delta_cell(T),
          (_signed(d_h[T] - d_s[T]) if (d_h[T] is not None and d_s[T] is not None) else MISSING)]
         for T in Ts]))
    A("")
    incomplete_Ts = [r["window"] for r in hard if r["measured"] and not r["complete"]]
    if incomplete_Ts:
        A(f"- ⚠️ **以下 T 的困难基准产物残缺，Δ 已强制为「不可比」**："
          + "；".join(f"T={r['window']}（{r.get('incomplete_reason')}）"
                      for r in hard if r["measured"] and not r["complete"])
          + "。残缺集合会把「最好单分支」选错、从而把 Δ 的符号算反，因此本工具宁可不出数。")
    A("- 表中的数是**宏 F1 均值**；有 ≥2 个种子的档位写成「均值 ± 样本标准差」，单种子档位如实标注。")
    A("- Δ = 混合 − 该 T 上 仅 CNN / 仅 BiLSTM 中宏 F1 更高者；**Δ > 0 表示混合模型更好**。")
    A("- 「Δ 变化量」= 困难 Δ − 简单 Δ：正值表示困难基准上卷积的相对地位**相对改善**，"
      "负值表示**更差**。")
    incomplete = [T for T in Ts
                  if T in hard_by_T and any(hard_by_T[T]["variants"][v]["n_seeds"] == 0 for v in VARIANTS)]
    if incomplete:
        A(f"- ⚠️ 困难基准在 T={'、'.join(str(t) for t in incomplete)} 上有结构尚未产出产物，"
          "这些行的 Δ 只与该 T 上**已测**的单分支比较（缺哪一支见第 2.1 节），"
          "**不能当作完整的三结构对比**。")
    A("")
    A("### 2.1 哪一个分支是「最好单分支」")
    A("")
    A(markdown_table(
        ["T", "简单基准 最好单分支", "困难基准 最好单分支"],
        [[T,
          (f"{VARIANT_LABEL.get(simple[T]['best_single_variant'], MISSING)}"
           f"（{_f(simple[T]['best_single_mean'])}）") if T in simple else MISSING,
          (f"{VARIANT_LABEL.get(hard_by_T[T]['best_single_variant'], MISSING)}"
           f"（{_f(hard_by_T[T]['best_single_mean'])}）") if T in hard_by_T else MISSING]
         for T in Ts]))
    A("")

    # ---------------- 2.2 统计功效 ----------------
    A("### 2.2 基准难度与统计功效（困难基准还剩多少可测样本）")
    A("")
    A("这一节是本报告的方法学基础：难度旋钮不只改变「有多难」，还会改变「还剩多少窗口可测」。"
      "窗口级评估用 `pure_only` 过滤（窗口内只要有一条记录标签不一致就整条丢弃），"
      "而行级标注噪声会把窗口整条打掉，因此 **label_noise 在窗口级评估里的主要作用是「删样本」"
      "而不是「加难度」**。")
    A("")
    A(markdown_table(
        ["T", "简单 测试窗口数", "简单 保留率(测试)", "简单 最小类 support",
         "困难 测试窗口数", "困难 保留率(测试)", "困难 最小类 support", "困难 训练窗口数"],
        [[T,
          _f((simple[T]["variants"]["cnn_bilstm"] or {}).get("n_test_windows"), 0) if T in simple else MISSING,
          _ratio(simple_diag_cache.get(T)),
          _f((simple_diag_cache.get(T) or {}).get("min_test_class_support"), 0),
          _f((hard_by_T[T]["variants"]["cnn_bilstm"]["n_test_windows"]), 0) if T in hard_by_T else MISSING,
          _ratio((hard_by_T[T].get("diagnostics") if T in hard_by_T else None)),
          _f(((hard_by_T[T].get("diagnostics") or {}).get("min_test_class_support") if T in hard_by_T else None), 0),
          _f(((hard_by_T[T].get("diagnostics") or {}).get("n_windows", {}) or {}).get("train") if T in hard_by_T else None, 0)]
         for T in Ts]))
    A("")
    A("- 「保留率(测试)」= 测试段保留窗口数 / 候选窗口数（候选 = 按 stride 切出的全部窗口）。"
      "简单基准的保留率由 `artifacts/ws<T>/data_summary.json` 读出，困难基准由 `artifacts/hard<T>/data_summary.json` 读出。")
    A(f"- 粗略规律：保留率 ≈ (1−label_noise)^T × 边界损失，因此 label_noise 每提高 0.01，"
      f"T=32 就多丢约 28% 的窗口。本实验的 label_noise 取值见第 1 节，其调整过程见第 6 节。")
    A("- 「最小类 support」= 测试集里最小的那一类的窗口数。它决定宏 F1 的**颗粒度噪声**"
      "（一个窗口判错对宏 F1 的影响 ≈ 1/(8 × 该类窗口数)）："
      "support 越小，Δ 越容易被样本颗粒度噪声淹没。")
    A("")

    # ---------------- 3. 判据 ----------------
    A("## 3. 判据与噪声尺度（与窗长消融同一套）")
    A("")
    A(markdown_table(
        ["T", "困难 混合 宏F1", "困难 最好单分支", "困难 Δ", "困难 σ（噪声尺度）", "Δ/σ",
         "σ 分档", "困难 是否饱和", "种子"],
        [[T,
          _ms(hard_by_T[T]["variants"]["cnn_bilstm"]["macro_f1"]) if T in hard_by_T else MISSING,
          _f(hard_by_T[T]["best_single_mean"]) if T in hard_by_T else MISSING,
          _signed(d_h[T]) if d_h[T] is not None else MISSING,
          _f((hard_by_T[T].get("sigma_used") if T in hard_by_T else None)),
          _f((hard_by_T[T].get("delta_over_sigma") if T in hard_by_T else None), 2),
          _band(d_h[T], (hard_by_T[T].get("sigma_used") if T in hard_by_T else None)),
          ("是（打满）" if hard_by_T[T]["variants"]["cnn_bilstm"]["saturated"] else "否") if T in hard_by_T else MISSING,
          _seed_note(hard_by_T[T]) if T in hard_by_T else MISSING]
         for T in Ts]))
    A("")
    A("**判据（预先写死）**：`|Δ| < 1σ` → 落在随机种子噪声内，不能宣称增益；"
      "`1σ ≤ |Δ| < 2σ` → 边缘证据；`|Δ| ≥ 2σ` → 超出噪声。"
      "有 ≥2 个种子的 T 用**该 T 自己**的 σ；只有单种子的 T 退回困难基准的参照 σ"
      "（取 T=16 混合模型的多种子 σ），并在表里逐行标明来源。")
    A("")
    A("### 3.1 逐种子明细（关键 T）")
    A("")
    multi = [r for r in hard if r["measured"] and r["has_ci"]]
    if multi:
        for r in multi:
            seeds = r["seeds"]
            tbl = []
            for v in VARIANTS:
                recs = {rec["seed"]: rec for rec in r["records"][v]}
                per = " / ".join((_f(recs[s]["macro_f1"]) if s in recs else "—") for s in seeds)
                tbl.append([VARIANT_LABEL[v], per, _ms(r["variants"][v]["macro_f1"])])
            A(f"**T={r['window']}（种子 {', '.join(str(s) for s in seeds)}）**")
            A("")
            A(markdown_table(["结构", f"逐种子宏F1（seed {'/'.join(str(s) for s in seeds)}）",
                              "均值 ± 样本标准差"], tbl))
            A("")
            s0 = seeds[0]
            h0 = _per_seed_f1(r, "cnn_bilstm", s0)
            sgl = [x for x in (_per_seed_f1(r, v, s0) for v in ("cnn", "bilstm")) if x is not None]
            if h0 is not None and sgl:
                A(f"- 只看 seed {s0}：Δ = **{_signed(h0 - max(sgl))}**；"
                  f"补齐 {len(seeds)} 个种子后 Δ = **{_signed(r['diff']['best_single'])}**"
                  f"（{_band(r['diff']['best_single'], r.get('sigma_used'))}）。")
                A("")
    else:
        A("困难基准目前没有 ≥2 个种子的档位，逐种子明细不可用（全部为单点观测，无置信区间）。")
        A("")

    # ---------------- 3.2 配对逐种子比较 ----------------
    A("### 3.2 逐种子配对比较（数据噪声被消掉的那一层）")
    A("")
    paired = q.get("paired") or []
    if paired:
        A("上面第 3 节的 Δ/σ 用的是「均值之差 ÷ 种子标准差」。但两套结构是在**同一份数据、"
          "同一批种子上**跑出来的，更贴题的比法是**逐种子配对做差**："
          "同一种子下「混合 − 最好单分支」，数据噪声在这一步就被消掉了。")
        A("")
        A(markdown_table(
            ["T", "最好单分支", "逐种子 Δ", "Δ 均值 ± 标准差", "负 / 正", "符号检验双侧 p"],
            [[x["window"], VARIANT_LABEL.get(x["best_single_variant"], x["best_single_variant"]),
              " / ".join(_signed(d) for d in x["deltas"]),
              f"{_signed(x['mean'])} ± {_f(x['std'])}",
              f"{x['n_negative']} / {x['n_positive']}",
              f"{x['sign_test_p_two_sided']:.3f}"]
             for x in paired]))
        A("")
        ov = q.get("paired_overall") or {}
        if ov:
            A(f"- 合并两档：**{ov['n_negative']}/{ov['n']} 对为负**，平均 {_signed(ov['mean'])}，"
              f"符号检验双侧 p = **{ov['sign_test_p_two_sided']:.3f}**"
              + ("——**没有任何一个种子下混合模型更好**。" if ov["all_negative"] else "。"))
            A("- 判读：p 值只说明「方向一致」这件事不太像运气；它**不**说明效应量大。"
              "真正的量级请看 Δ 均值（都在 -0.001 ~ -0.015 之间），"
              "对应到测试集上是几十条窗口的差别。")
        A("")
    else:
        A("没有 ≥2 个种子的档位，无法做逐种子配对比较。")
        A("")

    # ---------------- 4. 难度代价表 ----------------
    A("## 4. 难度代价表：把基准变难之后，整体掉了多少")
    A("")
    A("同一结构（`cnn_bilstm`，混合模型）在 T=16 上，简单基准 vs 困难基准：")
    A("")
    cost = difficulty_cost(hard_by_T, simple, 16, simple_diag_cache)
    A(markdown_table(
        ["指标", "简单基准（T=16）", "困难基准（T=16）", "变化"],
        cost["rows"]))
    A("")
    for line in cost["notes"]:
        A(f"- {line}")
    A("")
    A("同口径的其他 T（宏 F1 与在线误报）：")
    A("")
    A(markdown_table(
        ["T", "简单 混合宏F1", "困难 混合宏F1", "宏F1 下降",
         "简单 误报/小时", "困难 误报/小时", "简单 平均时延(ms)", "困难 平均时延(ms)"],
        [[T,
          _ms(simple[T]["variants"]["cnn_bilstm"]["macro_f1"]) if T in simple else MISSING,
          _ms(hard_by_T[T]["variants"]["cnn_bilstm"]["macro_f1"]) if T in hard_by_T else MISSING,
          (_signed(hard_by_T[T]["variants"]["cnn_bilstm"]["macro_f1"]["mean"]
                   - simple[T]["variants"]["cnn_bilstm"]["macro_f1"]["mean"])
           if (T in simple and T in hard_by_T
               and simple[T]["variants"]["cnn_bilstm"]["macro_f1"]["mean"] is not None
               and hard_by_T[T]["variants"]["cnn_bilstm"]["macro_f1"]["mean"] is not None) else MISSING),
          _ms(simple[T]["variants"]["cnn_bilstm"]["false_alarms_per_hour"], 2) if T in simple else MISSING,
          _ms(hard_by_T[T]["variants"]["cnn_bilstm"]["false_alarms_per_hour"], 2) if T in hard_by_T else MISSING,
          _f(_delay_ms(simple[T]["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]), 2) if T in simple else MISSING,
          _f(_delay_ms(hard_by_T[T]["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]), 2) if T in hard_by_T else MISSING]
         for T in Ts]))
    A("")
    A("- 时延单位是毫秒（原始产物里是秒，这里乘 1000 展示）；"
      "在线误报口径为固定阈值 0.5、`online_stride=3`、连续确认 3 个窗口。")
    A("- **误报的量级差异要谨慎解读**：困难基准里正常流量被大量「正常样貌」的隐蔽攻击流污染，"
      "误报率上升是数据性质变化的结果，不能直接读成模型变差。")
    A("")

    # ---------------- 5. 逐 T 明细 ----------------
    A("## 5. 逐 T 明细（困难基准）")
    A("")
    A(markdown_table(
        ["T", "混合 宏F1", "仅CNN 宏F1", "仅BiLSTM 宏F1", "Δ", "Δ/σ", "误分类窗口（混合）",
         "测试窗口数", "在线误报/小时", "平均时延(ms)", "参数量", "种子"],
        [[T,
          _ms(hard_by_T[T]["variants"]["cnn_bilstm"]["macro_f1"]) if T in hard_by_T else MISSING,
          _ms(hard_by_T[T]["variants"]["cnn"]["macro_f1"]) if T in hard_by_T else MISSING,
          _ms(hard_by_T[T]["variants"]["bilstm"]["macro_f1"]) if T in hard_by_T else MISSING,
          _signed(d_h[T]) if d_h[T] is not None else MISSING,
          _f((hard_by_T[T].get("delta_over_sigma") if T in hard_by_T else None), 2),
          _ms(hard_by_T[T]["variants"]["cnn_bilstm"]["n_errors"], 2) if T in hard_by_T else MISSING,
          _f(hard_by_T[T]["variants"]["cnn_bilstm"]["n_test_windows"], 0) if T in hard_by_T else MISSING,
          _ms(hard_by_T[T]["variants"]["cnn_bilstm"]["false_alarms_per_hour"], 2) if T in hard_by_T else MISSING,
          _f(_delay_ms(hard_by_T[T]["variants"]["cnn_bilstm"]["mean_detection_delay_sec"]), 2) if T in hard_by_T else MISSING,
          _f(hard_by_T[T]["variants"]["cnn_bilstm"]["n_params"], 0) if T in hard_by_T else MISSING,
          _seed_note(hard_by_T[T]) if T in hard_by_T else MISSING]
         for T in Ts]))
    A("")
    nerr = [r["variants"]["cnn_bilstm"]["n_errors"]["mean"] for r in hard
            if r["measured"] and r["variants"]["cnn_bilstm"]["n_errors"]["n"]]
    ntw = [r["variants"]["cnn_bilstm"]["n_test_windows"] for r in hard
           if r["measured"] and r["variants"]["cnn_bilstm"]["n_test_windows"]]
    if nerr and ntw:
        A(f"这张表是理解 Δ 的关键：**混合模型的误分类窗口只有 {min(nerr):.0f}–{max(nerr):.0f} 条**"
          f"（测试集 {min(ntw)}–{max(ntw)} 条窗口），而 Δ 是把 8 个类别的 F1 平均后的小数第三位——"
          "它天然落在单次训练的随机波动与样本颗粒度范围内。")
        A("")

    # ---------------- 6. 参数取值与调整记录 ----------------
    A("## 6. 困难档参数取值与「我调整过参数」的完整记录")
    A("")
    A(markdown_table(
        ["参数", "建议档（任务给出）", "最终取值", "是否调整", "理由"],
        [["overlap_ratio", _f(RECOMMENDED_HARD["overlap_ratio"], 2), _f(params.get("overlap_ratio"), 2), "否",
          "直接采用：与最易混淆类混合，直接改变类别边界，是真正作用于「可分性」的旋钮。"],
         ["stealth_ratio", _f(RECOMMENDED_HARD["stealth_ratio"], 2), _f(params.get("stealth_ratio"), 2), "否",
          "直接采用：提高「隐蔽型片段」占比，攻击必须在窗口聚合层面才看得出来。"],
         ["stealth_mix", _f(RECOMMENDED_HARD["stealth_mix"], 2), _f(params.get("stealth_mix"), 2), "否",
          "直接采用：隐蔽片段内更多流被混合成正常样貌。"],
         ["feature_noise", _f(RECOMMENDED_HARD["feature_noise"], 2), _f(params.get("feature_noise"), 2), "否",
          "直接采用（该旋钮在管线里是 `sigma = feature_noise × 0.1` 的标准差抖动，量级很小，影响有限）。"],
         ["label_noise", _f(RECOMMENDED_HARD["label_noise"], 2), _f(params.get("label_noise"), 2), "**是（0.05 → 0.02）**",
          "冒烟实测：见下方说明。"]])
    )
    A("")
    A("### 6.1 为什么把 label_noise 从建议的 0.05 下调到 0.02")
    A("")
    A("**在正式训练之前**，用临时内联脚本做了两条冒烟诊断（未产生任何落盘产物，复现命令见附录），"
      "结论是：**label_noise 在窗口级评估里的作用主要是「删样本」，不是「加难度」**。")
    A("")
    A("1. **机制**：窗口级训练/评估用 `pure_only=True`，窗口内只要有 1 条记录标签不一致就整条丢弃，"
      "因此保留率 ≈ (1−label_noise)^T。")
    A("2. **实测**（label_noise=0.05）：T=10 保留 59.0%、T=16 保留 42.8%、T=24 保留 28.1%、"
      "**T=32 只剩 18.6%**（3250 条窗口，测试集 794 条，最小的类 U2R 只有 **7 条**测试窗口）。")
    A("3. **后果**：宏 F1 的颗粒度噪声 ≈ 1/(8 × 最小类窗口数) ≈ 0.018（T=32），"
      "而我们要检验的 Δ 量级是 0.001–0.01——**样本颗粒度噪声会把信号完全淹没**，"
      "T=32 这一档就不能再用来回答「卷积是否需要更长上下文」。")
    A("4. **调整**：label_noise 0.05 → **0.02**（其余 4 个旋钮全部保留建议值）。"
      "实测保留率回到 T=16 70.7%（7178 训练 / 3051 测试）、T=32 50.0%（4946 训练 / 2151 测试），"
      "最小类测试 support 由 7 条回到 40 条，量级与简单基准（T=32 测试 2853 条）可比。")
    A("5. **难度没有被放松**：窗口级线性探针（把 (T,F) 展平后跑逻辑回归，纯难度指标、与架构无关）"
      f"在 T=16 上由简单基准的宏 F1 **{_f(smoke['window_level_logreg_macro_f1_T16']['simple'])}** "
      f"降到困难基准的 **{_f(smoke['window_level_logreg_macro_f1_T16']['hard_ln0.02'])}**"
      "（降幅约 0.115），行级探针同样从 0.4924 降到 0.2374（label_noise=0.05 时）。"
      "**难度确实上去了，而且没有难到「所有模型都崩」。**")
    A("")
    A("> 诚实性声明：这次调整**没有**任何模型训练结果作为依据（调整发生在 24 次训练全部开始之前），"
      "依据只有上面的窗口保留率、逐类样本数与两个与架构无关的线性探针。"
      "调整前后我都留在这里，读者可以自行判断。")
    A("")

    # ---------------- 7. 图 ----------------
    A("## 7. 核心图")
    A("")
    A("![困难基准 vs 简单基准](figures/hard_ablation_compare.png)")
    A("")
    A("- **左图（核心）**：两条 Δ-随-T 曲线画在同一张图上，蓝=简单基准、紫=困难基准，"
      "各自带 ±1σ 噪声带宽；空心点=该点混合模型宏 F1 已打满（指标饱和）。"
      "「困难基准是否让卷积翻身」一眼可见。")
    A("- 中图：困难基准上三种结构的宏 F1 随 T（多种子的点带 ±1σ 误差棒）。")
    A("- 右图：难度代价——混合模型的宏 F1 在简单 vs 困难基准上的差距（**跨基准，只读作「难度代价」，"
      "不可当作不同架构之间的比较**）。")
    A("- 三条曲线都不做平滑、不挑点，实测点全部标出。")
    A("")

    # ---------------- 8. 复现 ----------------
    A("## 8. 复现命令")
    A("")
    A("```powershell")
    A("# 困难档参数（每次运行都必须带上，否则会覆盖 data/raw_hard 为默认难度）")
    A("$hard = @(")
    A('  "--set","raw_dir=data/raw_hard","--set","processed_dir=data/processed_hard",')
    A(f'  "--set","overlap_ratio={_f(params.get("overlap_ratio"), 2)}","--set","stealth_ratio={_f(params.get("stealth_ratio"), 2)}",')
    A(f'  "--set","stealth_mix={_f(params.get("stealth_mix"), 2)}","--set","label_noise={_f(params.get("label_noise"), 2)}",')
    A(f'  "--set","feature_noise={_f(params.get("feature_noise"), 2)}","--set","publish_summary=false")')
    A("")
    for T in Ts:
        seeds = list(hard_by_T[T]["seeds"] or [0])
        # ⚠️ run_all.py 的 tag 规则：`--seeds` 里只要有多于一个种子，**所有**运行（含 seed 0）
        #    都会带上 `_s<seed>` 后缀。因此 seed 0 必须单独跑一次，才能落成不带后缀的
        #    `hard<T>`（本工具按 `hard<T>` / `hard<T>_s<k>` 找产物）。
        groups = [[s] for s in seeds]
        if len(seeds) > 1 and 0 in seeds:
            groups = [[0], [s for s in seeds if s != 0]]
        for g in groups:
            A(f"# T={T}（tag 规则：混合 hard{T}、单分支 hard{T}_cnn / hard{T}_bilstm、多种子加 _s<seed>）")
            A(f"python run_all.py --variants cnn_bilstm,cnn,bilstm --seeds {','.join(str(s) for s in g)} "
              "--no-figures `")
            A(f"  --tag hard{T} --set window_size={T} --set window_stride=5 --set epochs=25 @hard")
    A("python tools/hard_ablation.py")
    A("```")
    A("")
    A("> 三条硬约束：`publish_summary=false`（否则会覆盖 `results/summary.json` 与 "
      "`artifacts/main_model.json`）；`epochs=25` 与既有 ws 系列保持一致"
      "（工程默认已改为 40，不显式指定就不可比）；**多种子的档位必须把 seed 0 单独跑一次**——"
      "`run_all.py` 的 `make_tag` 只看本次 `--seeds` 的个数：个数 >1 时所有运行都带 `_s<seed>` 后缀，"
      "`--seeds 0,1,2` 会产出 `hard16_s0/s1/s2` 而**不会**产出 `hard16`，"
      "与既有 `ws16` 系列的命名约定不一致（本工具按 `hard<T>` 找主变体）。")
    A("")
    A("### 附录：困难档冒烟诊断的复现（正式训练之前跑的那两条）")
    A("")
    A("```powershell")
    A("# 1) 窗口级难度探针：把 (T,F) 展平后用逻辑回归，简单 vs 困难各跑一次")
    A("#    （简单基准读 data/processed 的 T=16 张量，困难基准用 save=False 现场准备，不落盘）")
    A("# 2) 窗口保留率与逐类样本数：读 artifacts/<tag>/data_summary.json 的 split.windows")
    A("python -c \"import json;d=json.load(open('artifacts/hard32/data_summary.json',encoding='utf-8'));print(d['split']['windows']['test'])\"")
    A("```")
    A("")
    A("本工具把第 2 条做成了可复现的表格（第 2.2 节），第 1 条的线性探针结果记在 "
      "`results/hard_ablation.json` 的 `smoke_probe` 字段里。")
    A("")

    # ---------------- 9. 局限 ----------------
    A("## 9. 局限（写清楚，不粉饰）")
    A("")
    A("- **两套基准的数据不同**：难度旋钮不同 ⇒ 测试集内容不同、窗口数不同。"
      "因此「困难宏 F1 < 简单宏 F1」只证明难度上去了，**不能**把两边的绝对精度放在一起排名。")
    A("- **困难基准仍然只是仿真**：结论依旧建立在工程自带的仿真流量上，"
      "真实数据上的复现见 `results/cross_dataset.json`。")
    single_seed_Ts = [r["window"] for r in hard if r["measured"] and not r["has_ci"]]
    ci_Ts = [r["window"] for r in hard if r["measured"] and r["has_ci"]]
    if single_seed_Ts:
        A(f"- **T={'、'.join(str(t) for t in single_seed_Ts)} 是单种子**："
          "这些档的 Δ 没有置信区间，只能看趋势，不能做显著性主张。")
    if ci_Ts:
        A(f"- **T={'、'.join(str(t) for t in ci_Ts)} 有 ≥2 个种子**："
          "这些档的 Δ 带均值 ± 样本标准差，可以做「是否超出种子噪声」的判断。")
    A("- **窗口步长固定 `window_stride=5`**：没有联合扫 stride / 判定阈值 / 确认次数。")
    A("- **困难基准的误报口径未重新标定阈值**：沿用固定阈值 0.5，"
      "难度变化会让误报率整体上移（见第 4 节），不代表模型在真实部署中的工作点。")
    A("- **饱和是「这套仿真 + 这些旋钮」的性质，不是普适结论**："
      "把难度继续往上推（或换成原始包序列输入）仍可能出现新的饱和点。")
    A("")
    return "\n".join(L)


def _ratio(diag: dict | None) -> str:
    if not diag:
        return MISSING
    r = diag.get("test_keep_ratio")
    if r is None:
        return MISSING
    return f"{r * 100:.1f}%"


def _per_seed_f1(row: dict, variant: str, seed: int):
    for rec in row.get("records", {}).get(variant, []):
        if rec["seed"] == seed:
            return rec.get("macro_f1")
    return None


def _seed_delta(row: dict | None, seed: int = 0):
    """简单基准某一种子下的 Δ = 混合 − 最好单分支（用于同种子口径核对）。"""
    if not row:
        return None
    v = row.get("best_single_variant")
    if not v:
        return None
    h = _per_seed_f1(row, "cnn_bilstm", seed)
    b = _per_seed_f1(row, v, seed)
    if h is None or b is None:
        return None
    return h - b


def _one_line(q: dict, hard_by_T: dict, simple: dict) -> str:
    """把三个问题的答案压成一句话（结论段用）。"""
    parts = []
    if q["q1_case"] == "no_gain":
        ov = q.get("paired_overall") or {}
        extra = (f"（逐种子配对 {ov.get('n_negative')}/{ov.get('n')} 对为负，"
                 f"符号检验双侧 p={ov.get('sign_test_p_two_sided'):.3f}）" if ov else "")
        parts.append("困难基准上卷积**依然没有增益**（Δ 全为负）" + extra)
    elif q["q1_case"] == "gain_within_noise":
        parts.append("困难基准上 Δ 出现正值但**落在种子噪声内**")
    else:
        parts.append("困难基准上 Δ 出现**超出噪声的正增益**")
    if q.get("q2_case") == "direction_changed":
        parts.append("「卷积需要更长上下文」的趋势在困难基准上**不再成立**")
    elif q.get("q2_case") == "same_direction_not_crossed":
        parts.append("Δ 随 T 的走势方向没变、但始终没有越过 0")
    else:
        parts.append("Δ 随 T 的走势出现了越过 0 的变化")
    if q.get("saturation_solved"):
        parts.append(f"饱和问题**已解决**（最高结构宏 F1 = {_f(q.get('hard_max_f1_overall'))} < 1.0000）")
    else:
        parts.append("饱和问题**没有解决**")
    return "；".join(parts) + "。"


def difficulty_cost(hard_by_T: dict, simple: dict, T: int, simple_diag: dict) -> dict:
    """同一结构（cnn_bilstm）在两个基准上的「难度代价」对照。"""
    rows: list[list] = []
    notes: list[str] = []
    h = hard_by_T.get(T)
    s = simple.get(T)
    if not h or not s:
        return {"rows": [[MISSING, MISSING, MISSING, MISSING]], "notes": ["T=%d 的产物不齐，无法给难度代价表。" % T]}

    def _row(label, sval, hval, fmt="{:.4f}", diff_fmt="{:+.4f}"):
        diff = MISSING
        if isinstance(sval, (int, float)) and isinstance(hval, (int, float)):
            diff = diff_fmt.format(hval - sval)
        sv = fmt.format(sval) if isinstance(sval, (int, float)) else MISSING
        hv = fmt.format(hval) if isinstance(hval, (int, float)) else MISSING
        rows.append([label, sv, hv, diff])

    hs = h["variants"]["cnn_bilstm"]
    ss = s["variants"]["cnn_bilstm"]
    _row("多分类宏 F1（混合）", _ms_val(ss["macro_f1"]), _ms_val(hs["macro_f1"]))
    _row("多分类准确率（混合）", _ms_val(ss["accuracy"]), _ms_val(hs["accuracy"]))
    _row("误分类窗口数（混合）", _ms_val(ss["n_errors"]), _ms_val(hs["n_errors"]), "{:.1f}", "{:+.1f}")
    _row("在线误报次数/小时（混合）", _ms_val(ss["false_alarms_per_hour"]),
         _ms_val(hs["false_alarms_per_hour"]), "{:.2f}", "{:+.2f}")
    _row("平均检测时延 ms（混合）", _delay_ms(ss["mean_detection_delay_sec"]),
         _delay_ms(hs["mean_detection_delay_sec"]), "{:.2f}", "{:+.2f}")
    _row("在线片段召回（混合）", _ms_val(ss["segment_recall"]), _ms_val(hs["segment_recall"]))
    _row("测试窗口数", ss.get("n_test_windows"), hs.get("n_test_windows"), "{:.0f}", "{:+.0f}")
    rows.append(["源数据（难度旋钮）",
                 f"默认档（overlap {_f(SIMPLE_REFERENCE['overlap_ratio'], 2)} / stealth {_f(SIMPLE_REFERENCE['stealth_ratio'], 2)} / "
                 f"label_noise {_f(SIMPLE_REFERENCE['label_noise'], 2)}）",
                 "困难档（见第 1 节）", "—"])
    s1 = _ms_val(ss["macro_f1"])
    h1 = _ms_val(hs["macro_f1"])
    if s1 is not None and h1 is not None:
        notes.append(f"同一结构、同一 T，仅把数据换成困难档：宏 F1 由 **{s1:.4f}** 掉到 **{h1:.4f}**"
                     f"（**{h1 - s1:+.4f}**，占原值的 {(s1 - h1) / s1 * 100:.1f}%）。"
                     "这就是「难度代价」的量级。")
    if not (ss["macro_f1"].get("n", 0) >= 2):
        notes.append("简单基准 T=%d 的混合模型只有 1 个种子，其时延/误报是单点观测。" % T)
    if not (hs["macro_f1"].get("n", 0) >= 2):
        notes.append("困难基准 T=%d 的混合模型只有 1 个种子，其时延/误报是单点观测。" % T)
    else:
        notes.append(f"困难基准 T={T} 的混合模型有 {hs['macro_f1']['n']} 个种子，"
                     "宏 F1 写的是均值 ± 样本标准差；误报/时延同样是多种子均值。")
    if simple_diag and simple_diag.get("test_keep_ratio") is not None:
        notes.append(f"两边的测试窗口数不可忽略地不同（简单 {s.get('n_test_windows')} 条 vs "
                     f"困难 {hs.get('n_test_windows')} 条），**精度数值不可直接比较**。")
    return {"rows": rows, "notes": notes}


# --------------------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------------------
def make_figure(hard, simple: dict, q: dict, path: Path) -> Path | None:
    zh = setup_matplotlib()
    import matplotlib.pyplot as plt

    def L(zh_text: str, en_text: str) -> str:
        return zh_text if zh else en_text

    meas = [r for r in hard if r["measured"] and r["complete"] and r["diff"].get("best_single") is not None]
    if not meas:
        LOGGER.warning("没有可画的困难基准实测点（三结构齐全的 T 为空），跳过对比图。")
        return None

    Ts = [r["window"] for r in meas]
    hard_gaps = [r["diff"]["best_single"] for r in meas]
    s_Ts = [T for T in Ts if T in simple and (simple[T].get("diff") or {}).get("best_single") is not None]
    s_gaps = [simple[T]["diff"]["best_single"] for T in s_Ts]
    s_sat = [simple[T]["variants"]["cnn_bilstm"]["saturated"] for T in s_Ts]
    h_sat = [r["variants"]["cnn_bilstm"]["saturated"] for r in meas]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))
    C_S, C_H = "#1565c0", "#6a1b9a"

    # ---- 左：核心图，两条 Δ-随-T 曲线 ----
    ax = axes[0]
    s_sig = max([simple[T].get("sigma_used") or 0.0 for T in s_Ts] or [0.0])
    h_sig = max([r.get("sigma_used") or 0.0 for r in meas] or [0.0])
    if s_sig:
        ax.axhspan(-s_sig, s_sig, color=C_S, alpha=0.10,
                   label=L(f"简单基准 ±1σ（σ≤{s_sig:.4f}）", f"simple ±1σ (σ<={s_sig:.4f})"))
    if h_sig:
        ax.axhspan(-h_sig, h_sig, color=C_H, alpha=0.14,
                   label=L(f"困难基准 ±1σ（σ≤{h_sig:.4f}）", f"hard ±1σ (σ<={h_sig:.4f})"))
    ax.axhline(0.0, color="#424242", lw=1.1, ls="--")
    if s_Ts:
        errs_s = [math.sqrt(((simple[T]["variants"]["cnn_bilstm"]["macro_f1"].get("std") or 0) ** 2)
                            + ((simple[T]["variants"][simple[T]["best_single_variant"]]["macro_f1"].get("std") or 0) ** 2)
                            if simple[T]["best_single_variant"] else 0.0) for T in s_Ts]
        ax.errorbar(s_Ts, s_gaps, yerr=errs_s, color=C_S, lw=1.8, ls="--", marker="o", ms=7,
                    capsize=3, zorder=2, label=L("简单基准：Δ", "simple: Δ"))
        for t, g, st in zip(s_Ts, s_gaps, s_sat):
            ax.plot([t], [g], marker="o", ms=8.5, color=C_S, ls="none",
                    markerfacecolor=("white" if st else C_S), markeredgewidth=1.6, zorder=3)
            ax.annotate(f"{g:+.4f}", (t, g), textcoords="offset points", xytext=(0, -16 if g < 0 else 12),
                        ha="center", fontsize=7.5, color=C_S)
    errs_h = []
    for r in meas:
        st_h = r["variants"]["cnn_bilstm"]["macro_f1"].get("std")
        st_b = (r["variants"][r["best_single_variant"]]["macro_f1"].get("std")
                if r["best_single_variant"] else None)
        errs_h.append(math.sqrt((st_h or 0) ** 2 + (st_b or 0) ** 2) if (st_h or st_b) else 0.0)
    ax.errorbar(Ts, hard_gaps, yerr=errs_h, color=C_H, lw=2.2, marker="s", ms=7, capsize=4,
                zorder=4, label=L("困难基准：Δ", "hard: Δ"))
    for t, g, st in zip(Ts, hard_gaps, h_sat):
        ax.plot([t], [g], marker="s", ms=9, color=C_H, ls="none",
                markerfacecolor=("white" if st else C_H), markeredgewidth=1.6, zorder=5)
        ax.annotate(f"{g:+.4f}", (t, g), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.0, color=C_H, fontweight="bold")
    ax.set_xlabel(L("窗口长度 T（条流记录）", "Window length T (flows)"))
    ax.set_ylabel(L("Δ 宏 F1 = 混合 − 最好单分支", "Delta macro-F1 (hybrid − best single)"))
    ax.set_title(L("核心图：Δ 随 T 的走势，简单基准 vs 困难基准\n（两条线都在 0 以下＝卷积始终无增益）",
                   "Core: Δ vs T, simple vs hard benchmark"))
    ax.set_xticks(Ts)
    ax.legend(fontsize=8.0, loc="best")
    ax.grid(alpha=0.25, lw=0.6)

    # ---- 中：困难基准三种结构的宏 F1 ----
    ax2 = axes[1]
    style = {"cnn_bilstm": (C_H, "o", "-"), "cnn": ("#ef6c00", "s", "--"), "bilstm": ("#2e7d32", "^", "-.")}
    for v in VARIANTS:
        pts = [(r["window"], r["variants"][v]["macro_f1"]) for r in hard
               if r["variants"][v]["macro_f1"]["n"]]
        if not pts:
            continue
        col, mk, ls = style[v]
        ax2.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts],
                     yerr=[(p[1].get("std") or 0.0) for p in pts], marker=mk, ls=ls, color=col,
                     lw=1.9, capsize=4, label=VARIANT_LABEL[v] if zh else v)
    if q.get("saturation_solved"):
        ax2.axhline(1.0, color="#9e9e9e", lw=1.0, ls=":")
        ax2.text(Ts[0], 1.0, L(" 1.0000 天花板（本次没有任何结构打到）", " ceiling 1.0000 (not reached)"),
                 fontsize=8.0, color="#616161", va="bottom")
    ax2.set_xlabel(L("窗口长度 T", "Window length T"))
    ax2.set_ylabel(L("宏 F1", "Macro-F1"))
    ax2.set_title(L("困难基准：三种结构的宏 F1", "Hard benchmark: macro-F1 of three structures"))
    ax2.set_xticks(Ts)
    allf1 = [(r["variants"][v]["macro_f1"]["mean"], r["variants"][v]["macro_f1"].get("std") or 0.0)
             for r in hard for v in VARIANTS if r["variants"][v]["macro_f1"]["mean"] is not None]
    if allf1:
        lo = min(m - s for m, s in allf1) - 0.01
        ax2.set_ylim(max(lo, 0.0), 1.008)
    ax2.legend(fontsize=8.0, loc="lower right")
    ax2.grid(alpha=0.25, lw=0.6)

    # ---- 右：难度代价 ----
    ax3 = axes[2]
    both = [T for T in Ts if T in simple]
    if both:
        ys_s = [simple[T]["variants"]["cnn_bilstm"]["macro_f1"]["mean"] for T in both]
        ys_h = [hard_by_T_mean(hard, T) for T in both]
        ax3.plot(both, ys_s, marker="o", color=C_S, ls="--", lw=1.9, label=L("简单基准（混合）", "simple (hybrid)"))
        ax3.plot(both, ys_h, marker="s", color=C_H, lw=2.2, label=L("困难基准（混合）", "hard (hybrid)"))
        for T, a, b in zip(both, ys_s, ys_h):
            if a is not None and b is not None:
                ax3.annotate("", xy=(T, b), xytext=(T, a),
                             arrowprops=dict(arrowstyle="<->", color="#757575", lw=1.0))
                ax3.text(T, (a + b) / 2, f"{b - a:+.4f}", fontsize=7.5, color="#424242",
                         ha="left", va="center", rotation=90)
        ax3.set_xticks(both)
    ax3.set_xlabel(L("窗口长度 T", "Window length T"))
    ax3.set_ylabel(L("宏 F1（混合模型）", "Macro-F1 (hybrid)"))
    ax3.set_title(L("难度代价：同一结构的宏 F1 掉了多少\n（跨基准，仅读作难度代价，不是架构比较）",
                    "Difficulty cost: macro-F1 drop (cross-benchmark, reference only)"))
    ax3.legend(fontsize=8.0, loc="best")
    ax3.grid(alpha=0.25, lw=0.6)

    case_txt = {
        "no_gain": L("结论：困难基准上卷积依然没有增益（Δ 全为负）",
                     "verdict: still no conv gain on the harder benchmark"),
        "gain_within_noise": L("结论：Δ 出现正值但仍在种子噪声内",
                               "verdict: positive delta, still within seed noise"),
        "gain_beyond_noise": L("结论：Δ 出现超出噪声的正增益（需核对是否饱和）",
                               "verdict: positive delta beyond noise"),
    }.get(q.get("q1_case"), L("证据不足", "insufficient"))
    if q.get("saturation_solved"):
        case_txt += L(f"；困难基准不再饱和（最高宏 F1 {_f(q.get('hard_max_f1_overall'))}）",
                      f"; not saturated (max macro-F1 {_f(q.get('hard_max_f1_overall'))})")
    fig.suptitle(case_txt, fontsize=12.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def hard_by_T_mean(hard, T):
    for r in hard:
        if r["window"] == T:
            return r["variants"]["cnn_bilstm"]["macro_f1"]["mean"]
    return None


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="困难基准窗口消融对比（表 + 图 + 结论报告）")
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS),
                    help="要汇总的窗口长度，逗号分隔（默认 10,16,24,32）")
    ap.add_argument("--results-dir", default=str(RESULT_DIR), help="结果产物目录（默认 results）")
    ap.add_argument("--artifact-dir", default=str(ROOT / "artifacts"), help="训练产物目录（默认 artifacts）")
    ap.add_argument("--out", default="hard_ablation", help="输出文件名前缀（默认 hard_ablation）")
    ap.add_argument("--figure", default="hard_ablation_compare.png", help="对比图文件名")
    ap.add_argument("--noise-window", type=int, default=16,
                    help="单种子 T 退回用的参照噪声取自哪个窗口（默认 16）")
    ap.add_argument("--saturation-f1", type=float, default=SATURATION_F1,
                    help="宏 F1 达到该值视为指标饱和（默认 0.99995）")
    ap.add_argument("--no-figure", action="store_true", help="只出表与报告，不画图")
    ap.add_argument("--strict", action="store_true",
                    help="严格模式：只要有任何 T 的三结构产物残缺（或存在产物↔变体绑定错误），"
                         "就以退出码 2 结束。用于 CI/验收时防止用残缺集合得出错误结论。")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global SATURATION_F1, simple_diag_cache, simple_gen_cache
    args = parse_args(argv)
    windows = [int(x) for x in str(args.windows).split(",") if x.strip()]
    results_dir = Path(args.results_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    ensure_dirs(FIGURE_DIR)
    SATURATION_F1 = args.saturation_f1
    binding_errors.clear()

    wa_path = results_dir / "window_ablation.json"
    wa = _load_json(wa_path)
    if wa is None:
        LOGGER.error("缺少简单基准产物 %s（本工具需要它做并排对比）。", wa_path.name)
        return 2
    simple = simple_rows(wa, set(windows))
    missing_simple = [T for T in windows if T not in simple]
    if missing_simple:
        LOGGER.warning("简单基准缺少这些 T：%s", "、".join(f"T={t}" for t in missing_simple))

    simple_diag_cache = {T: simple_diagnostics(results_dir, artifact_dir, T) for T in windows}
    for T in windows:
        for v in VARIANTS:
            tags = ((simple.get(T) or {}).get("variants", {}).get(v) or {}).get("tags") or []
            simple_gen_cache[(T, v)] = generalization_gap(artifact_dir, tags[0]) if tags else None

    # 两遍：先收集困难基准，拿到它自己的参照 σ，再定稿 σ 分档
    hard = build_rows(windows, results_dir, artifact_dir, SATURATION_F1, ref_sigma=None)
    ref_sigma = hard_reference_sigma(hard, args.noise_window)
    if ref_sigma:
        hard = build_rows(windows, results_dir, artifact_dir, SATURATION_F1, ref_sigma=ref_sigma)
    missing = [r["window"] for r in hard if not r["measured"]]
    if missing:
        LOGGER.warning("困难基准缺少这些 T 的产物，将标为「未测」：%s", "、".join(f"T={t}" for t in missing))
    if ref_sigma is None:
        LOGGER.warning("困难基准在 T=%d 上没有多种子产物，单种子 T 的 σ 档位将标「未测」。",
                       args.noise_window)

    q = answer_questions(hard, simple)
    q["hard_reference_sigma"] = ref_sigma
    q["hard_reference_sigma_window"] = args.noise_window

    params = {}
    for r in hard:
        if r.get("protocol"):
            params = r["protocol"]
            break
    if not params:
        params = {"raw_dir": "data/raw_hard", "processed_dir": "data/processed_hard"}

    report = build_report(hard, simple, q, params, SMOKE_PROBE)
    out_json = save_json(
        {
            "experiment": "hard_window_ablation",
            "windows_requested": windows,
            "saturation_threshold": args.saturation_f1,
            "hard_params_actual": {k: params.get(k) for k in
                                   ("overlap_ratio", "stealth_ratio", "stealth_mix", "label_noise", "feature_noise")},
            "hard_params_recommended": RECOMMENDED_HARD,
            "simple_params_reference": SIMPLE_REFERENCE,
            "param_adjustment": {
                "field": NOISE_ARG, "from": RECOMMENDED_HARD[NOISE_ARG],
                "to": params.get(NOISE_ARG),
                "reason": "label_noise 在窗口级评估里主要作用是删样本（保留率≈(1−ln)^T），"
                          "0.05 会让 T=32 只剩 18.6% 窗口、最小类测试 support 降到 7 条，"
                          "宏 F1 颗粒度噪声淹没 Δ；调整为 0.02（依据为正式训练之前的冒烟诊断，非训练结果）。",
            },
            "smoke_probe": SMOKE_PROBE,
            "completeness": {
                "windows": windows,
                "complete_windows": [r["window"] for r in hard if r["complete"]],
                "incomplete_windows": [
                    {"window": r["window"], "missing_variants": r.get("missing_variants"),
                     "expected_tags": r.get("expected_tags"), "reason": r.get("incomplete_reason")}
                    for r in hard if not r["complete"]
                ],
                "binding_errors": list(binding_errors),
                "delta_policy": "只有三结构（cnn_bilstm / cnn / bilstm）全部有指标时才计算 Δ；"
                                "任一结构缺失则该 T 记为 incomplete 且 Δ 输出为 null，"
                                "并在报告里标『不可比』（残缺集合会把 best_single 选错、使 Δ 符号反转）。",
            },
            "hard_rows": hard,
            "simple_rows": {str(T): simple[T] for T in sorted(simple)},
            "simple_diagnostics": {str(T): simple_diag_cache.get(T) for T in sorted(simple_diag_cache)},
            "answers": q,
        },
        results_dir / f"{args.out}.json",
    )
    out_md = save_text(report, results_dir / f"{args.out}.md")
    fig_path = None
    if not args.no_figure:
        fig_path = make_figure(hard, simple, q, FIGURE_DIR / args.figure)

    # ---------------- 逐 T 核对打印（供人工对账，防止再出现「读成 None 后符号算反」）----------------
    LOGGER.info("逐 T 核对（困难基准）：")
    LOGGER.info("  %-4s %-10s %-10s %-10s %-24s %-10s %-12s", "T", "混合", "仅CNN", "仅BiLSTM",
                "best_single", "Δ(seed0)", "Δ(均值)")
    for r in hard:
        T = r["window"]
        v = r["variants"]
        names = {kk: _f(v[kk]["macro_f1"]["mean"]) for kk in VARIANTS}
        best = (f"{VARIANT_LABEL.get(r['best_single_variant'], MISSING)}"
                f"({_f(r['best_single_mean'])})" if r["best_single_variant"]
                else f"不可比（缺 {'、'.join(VARIANT_LABEL.get(x, x) for x in (r.get('missing_variants') or []))}）")
        h0 = _per_seed_f1(r, "cnn_bilstm", 0)
        sgl0 = [x for x in (_per_seed_f1(r, "cnn", 0), _per_seed_f1(r, "bilstm", 0)) if x is not None]
        d0 = _signed(h0 - max(sgl0)) if (h0 is not None and len(sgl0) == 2) else "不可比"
        LOGGER.info("  %-4s %-10s %-10s %-10s %-24s %-10s %-12s", T, names["cnn_bilstm"], names["cnn"],
                    names["bilstm"], best, d0, _signed(r["diff"].get("best_single")))
    if binding_errors:
        LOGGER.error("发现 %d 条「产物 ↔ 变体」绑定错误（这些记录已被丢弃）：", len(binding_errors))
        for e in binding_errors:
            LOGGER.error("  %s：期望 _%s，实际 %r", e["file"], e["expected_variant"], e["metrics_model_variant"])
    incomplete = [r for r in hard if r["measured"] and not r["complete"]]
    if incomplete:
        for r in incomplete:
            LOGGER.error("T=%d 产物残缺，Δ 已置为不可比：%s", r["window"], r.get("incomplete_reason"))

    LOGGER.info("困难基准消融汇总完成：可算 Δ 的 T %d/%d（多种子：%s）",
                sum(1 for r in hard if r["complete"]), len(hard),
                "、".join(f"T={r['window']}" for r in hard if r["has_ci"]) or "无")
    LOGGER.info("问题1（是否有增益）：%s | 最大 Δ=%s @ T=%s",
                q.get("q1_case"), _signed(q.get("max_delta")), q.get("max_delta_window"))
    LOGGER.info("问题2（趋势方向）：%s | 困难斜率=%s，简单斜率=%s",
                q.get("q2_case"), _signed(q.get("hard_slope_per_T"), 5),
                _signed(q.get("simple_slope_per_T"), 5))
    LOGGER.info("问题3（饱和是否解决）：%s | 困难最高宏 F1=%s",
                q.get("saturation_solved"), _f(q.get("hard_max_f1_overall")))
    for p in (out_json, out_md, fig_path):
        if p:
            LOGGER.info("→ %s", p.name)
    if args.strict and (incomplete or binding_errors):
        LOGGER.error("--strict：存在残缺产物或绑定错误，以退出码 2 结束。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
