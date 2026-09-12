"""与已发表方法对标：RNN / LSTM / BiLSTM / Transformer / CNN / 本项目双分支，同一套口径。

为什么做这个
------------
对标论文（曹鲍垒等. 基于 PLO-BiLSTM 的网络异常流量检测方法. 通信技术, 2026, 59(8): 950-960）
的对比实验是「本文方法 vs RNN / LSTM / BiLSTM / Transformer」，其图 7 报的是
准确率 / 精确率 / 召回率 / F1 四项。本项目原来只有 cnn / bilstm / mlp 三个结构消融变体，
**缺 RNN / LSTM / Transformer 三个外部对照**，因此"我们比它们好在哪"这句话一直没有本地证据。

本工具把三种模型补进 `src/model.py` 的变体表（见 `VARIANTS`），并**在同一份数据、
同一套切分、同一套训练与评估代码**下逐种子跑一遍，产出可直接放进报告的对标表。

口径说明（为什么数字不能和论文直接比）
--------------------------------------
* 论文：CICIDS2017 / UNSW_2018_IoT_Botnet，7:3 随机划分，Min-max 归一化，Accuracy 为主指标。
* 本项目：自研数据集（或控制器原生字段口径），**时序切分 + purge gap**（防泄漏），
  宏平均 F1 + 在线告警口径（片段召回 / 时延 / 误报每小时）。
数据集、划分方式、指标定义三处都不同，因此**只能比"方法之间的相对排序"，不能比绝对数值**。
本工具同时报准确率与宏平均 F1，便于读者看清"指标口径如何影响结论"。

用法
----
    python tools/published_baselines.py --quick                  # 冒烟
    python tools/published_baselines.py --seeds 0,1,2 --epochs 25 \
        --variants rnn,lstm,bilstm,transformer,cnn,cnn_bilstm

产物：`results/published_baselines.{json,md}`、`results/figures/published_baselines.png`；
训练产物落 `artifacts/_pub_baselines/`（可再生的临时产物，不进版本库）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData  # noqa: E402
from src.evaluate import evaluate_model  # noqa: E402
from src.train import run_training  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR, FIGURE_DIR, RESULT_DIR, ensure_dirs, get_logger, save_json, save_text,
    set_seed, setup_matplotlib, silence_tf_logs,
)

LOGGER = get_logger("cida.pubbase")

#: 默认跑全部 6 个变体：3 个对标论文 + 3 个本项目结构（含主模型）
DEFAULT_VARIANTS = ("rnn", "lstm", "bilstm", "transformer", "cnn", "cnn_bilstm")

#: 人读名字（报告里用，兼顾中英）
DISPLAY = {
    "rnn": "RNN（双向朴素循环）",
    "lstm": "LSTM（单向）",
    "bilstm": "BiLSTM（双向）",
    "transformer": "Transformer（2 层编码器）",
    "cnn": "CNN（本项目单分支对照）",
    "cnn_bilstm": "**CNN‖BiLSTM 双分支（本项目主模型）**",
}


def _agg(vals: list[float]) -> dict:
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {"n": len(v), "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else None, "values": v}


def _ms(a: dict | None, nd: int = 4) -> str:
    if not a or a.get("mean") is None:
        return "—"
    return (f"{a['mean']:.{nd}f}" if a.get("std") is None
            else f"{a['mean']:.{nd}f} ± {a['std']:.{nd}f}")


def _run_one(dataset: str, variant: str, seed: int, data: PreparedData, *, args) -> dict:
    cfg = Config(verbose=False, figures=False, publish_summary=False,
                 processed_dir=str(ROOT / "data" / dataset),
                 window_size=int(data.window_size), input_mode=str(data.input_mode),
                 epochs=int(args.epochs), seed=int(seed), run_name=f"pubbase_{dataset}_{variant}")
    art = ARTIFACT_DIR / "_pub_baselines" / dataset
    tag = f"pubbase_{dataset}_{variant}_s{seed}"
    t0 = time.time()
    res = run_training(cfg, data, variant=variant, tag=tag, epochs=int(args.epochs),
                       artifact_dir=art)
    rep = evaluate_model(res.model, data, cfg, tag=tag, result_dir=RESULT_DIR / "_pub_baselines",
                         artifact_dir=art, save_predictions=False)
    mc, bi, on = rep["multiclass"], rep["binary"], (rep.get("online") or {})
    rec = {
        "dataset": dataset, "variant": variant, "seed": int(seed),
        "n_features": int(data.n_features), "n_params": int(res.model.count_params()),
        "accuracy": float(mc["accuracy"]),
        "macro_precision": float(mc["macro_precision"]),
        "macro_recall": float(mc["macro_recall"]),
        "macro_f1": float(mc["macro_f1"]),
        "weighted_f1": float(mc["weighted_f1"]),
        "cohen_kappa": float(mc["cohen_kappa"]),
        "macro_roc_auc_ovr": float(mc["macro_roc_auc_ovr"]),
        "binary_f1": float(bi["f1"]),
        "binary_roc_auc": float(bi["roc_auc"]),
        "binary_pr_auc": float(bi["pr_auc"]),
        "tpr_at_fpr_1pct": float(bi.get("tpr_at_fpr_0.01")) if bi.get("tpr_at_fpr_0.01") is not None else None,
        "segment_recall": (float(on["segment_recall"]) if on.get("segment_recall") is not None else None),
        "mean_delay_ms": (float(on["mean_detection_delay_sec"]) * 1000.0
                          if on.get("mean_detection_delay_sec") is not None else None),
        "p95_delay_ms": (float(on["p95_detection_delay_sec"]) * 1000.0
                         if on.get("p95_detection_delay_sec") is not None else None),
        "false_alarms_per_hour": (float(on["false_alarms_per_hour"])
                                  if on.get("false_alarms_per_hour") is not None else None),
        "seconds": round(time.time() - t0, 1),
    }
    del res
    return rec


def build_report(p: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# 与已发表方法对标：RNN / LSTM / BiLSTM / Transformer / CNN / 本项目双分支")
    A("")
    A(f"> 由 `tools/published_baselines.py` 生成。口径：`{p['args']['dataset']}`，"
      f"窗口 T={p['window_size']}、特征 {p['n_features']} 维、"
      f"{len(p['args']['seeds'])} 个随机种子、epochs={p['args']['epochs']}。")
    A("> 六个模型**共用同一套输入张量、同一套时序切分（含 purge gap）、同一份训练与评估代码**，"
      "因此相互之间可以直接比。")
    A("")

    # ---- 主表：论文图 7 的四项指标 + 我们更看重的在线指标 ----
    L.append("## 1. 主表（对标论文图 7 的四项指标，并补上在线告警口径）")
    L.append("")
    L.append("| 模型 | 参数量 | 准确率 | 宏精确率 | 宏召回率 | 宏F1 | 二分类 ROC-AUC | "
             "Tpr@Fpr1% | 片段召回 | 平均时延(ms) | 误报(次/h) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for v in p["args"]["variants"]:
        s = p["summary"].get(v)
        if not s:
            continue
        A(f"| {DISPLAY.get(v, v)} | {s['n_params']:,} | {_ms(s['accuracy'])} | "
          f"{_ms(s['macro_precision'])} | {_ms(s['macro_recall'])} | **{_ms(s['macro_f1'])}** | "
          f"{_ms(s['binary_roc_auc'])} | {_ms(s['tpr_at_fpr_1pct'])} | "
          f"{_ms(s['segment_recall'])} | {_ms(s['mean_delay_ms'], 2)} | "
          f"{_ms(s['false_alarms_per_hour'], 2)} |")
    L.append("")

    # ---- 结论 ----
    v = p["verdict"]
    L.append("## 2. 结论")
    L.append("")
    A(f"- 主模型（CNN‖BiLSTM 双分支）的宏平均 F1 为 **{_ms(p['summary']['cnn_bilstm']['macro_f1'])}**；"
      f"最接近的对照是 {v['closest']['variant']}（{_ms(p['summary'][v['closest']['variant']]['macro_f1'])}，"
      f"Δ={v['closest']['delta']:+.4f}）。")
    for name, st in v["vs"].items():
        A(f"- 相对 **{DISPLAY.get(name, name)}**：平均 Δ宏F1 = {st['mean_delta']:+.4f}"
          f"（{st['positive']} 正 / {st['negative']} 负"
          + (f"，符号检验 p={st['p_value']:.4f}" if st.get("p_value") is not None else "")
          + "）。")
    A(f"- **{v['statement']}**")
    L.append("")
    L.append("> 论文的图 7 结论是其方法在四项指标上均优于 RNN/LSTM/BiLSTM/Transformer。"
             "本表能回答的问题**不是**「我们的数字比它高」（数据集与划分都不同，不可比），"
             "而是：**在同一份数据和同一套防泄漏口径下，双分支融合相对这些经典时序结构到底强多少**。")
    L.append("")
    return "\n".join(L) + "\n"


def _fmt1(x, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def make_figure(p: dict, out: Path) -> Path | None:
    try:
        setup_matplotlib()
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("画图不可用：%s", exc)
        return None
    variants = [v for v in p["args"]["variants"] if p["summary"].get(v)]
    metrics = [("accuracy", "准确率"), ("macro_precision", "宏精确率"),
               ("macro_recall", "宏召回率"), ("macro_f1", "宏F1")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    width = 0.8 / len(metrics)
    for k, (key, label) in enumerate(metrics):
        ys = [p["summary"][v][key]["mean"] for v in variants]
        es = [p["summary"][v][key]["std"] or 0.0 for v in variants]
        ax.bar(np.arange(len(variants)) + k * width - 0.4 + width / 2, ys, width=width,
               yerr=es, capsize=2.5, label=label)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=18, fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("对标论文图 7 的四项指标（误差棒 = 种子间标准差）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax2 = axes[1]
    ys = [p["summary"][v]["binary_roc_auc"]["mean"] for v in variants]
    es = [p["summary"][v]["binary_roc_auc"]["std"] or 0.0 for v in variants]
    ax2.barh(range(len(variants)), ys, xerr=es, capsize=2.5)
    ax2.set_yticks(range(len(variants)))
    ax2.set_yticklabels(variants, fontsize=8)
    ax2.set_xlim(0.5, 1.02)
    ax2.set_title("二分类 ROC-AUC")
    ax2.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOGGER.info("对标图已生成 → %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="与已发表方法对标（RNN/LSTM/BiLSTM/Transformer/CNN/双分支）")
    ap.add_argument("--dataset", default="processed")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--redraw", action="store_true",
                    help="不重训：读已有 results/published_baselines.json，只重新生成报告与图")
    args = ap.parse_args(argv)
    if args.redraw:
        p = json.loads((RESULT_DIR / "published_baselines.json").read_text(encoding="utf-8"))
        save_text(build_report(p), RESULT_DIR / "published_baselines.md")
        if not args.no_figure:
            make_figure(p, FIGURE_DIR / "published_baselines.png")
        LOGGER.info("已按现有 JSON 重新生成报告与图（未重训）")
        return 0
    if args.quick:
        args.variants, args.seeds, args.epochs = "rnn,cnn_bilstm", "0", 2

    args.variants = tuple(x.strip() for x in str(args.variants).split(",") if x.strip())
    args.seeds = tuple(int(x) for x in str(args.seeds).split(",") if x.strip())

    ensure_dirs(RESULT_DIR, FIGURE_DIR, ARTIFACT_DIR)
    silence_tf_logs()
    set_seed(42)
    started = time.time()

    pdir = ROOT / "data" / args.dataset
    if not (pdir / "tensors.npz").exists():
        raise SystemExit(f"找不到预处理产物 {pdir}/tensors.npz（本项目不重新生成数据）")
    data = PreparedData.load(pdir)
    LOGGER.info("=== 对标实验：口径 %s（%d 维 × T=%d），变体 %s，种子 %s ===",
                args.dataset, data.n_features, data.window_size, args.variants, args.seeds)

    records: list[dict] = []
    for variant in args.variants:
        for seed in args.seeds:
            rec = _run_one(args.dataset, variant, seed, data, args=args)
            records.append(rec)
            LOGGER.info("[%s|s%d] acc=%.4f 宏F1=%.4f AUC=%.4f 片段召回=%s 时延=%sms 误报=%s/h（%.0fs）",
                        variant, seed, rec["accuracy"], rec["macro_f1"], rec["binary_roc_auc"],
                        _fmt1(rec["segment_recall"], 3), _fmt1(rec["mean_delay_ms"], 2),
                        _fmt1(rec["false_alarms_per_hour"], 2), rec["seconds"])

    keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "cohen_kappa", "macro_roc_auc_ovr", "binary_f1", "binary_roc_auc", "binary_pr_auc",
            "tpr_at_fpr_1pct", "segment_recall", "mean_delay_ms", "p95_delay_ms",
            "false_alarms_per_hour")
    summary = {}
    for variant in args.variants:
        recs = [r for r in records if r["variant"] == variant]
        if not recs:
            continue
        s = {"n_params": int(recs[0]["n_params"]), "n_features": int(recs[0]["n_features"]),
             "n_seeds": len(recs)}
        s.update({k: _agg([r.get(k) for r in recs]) for k in keys})
        summary[variant] = s

    # 逐种子配对：主模型 − 每个对照
    def paired(other: str) -> dict:
        deltas = []
        for seed in args.seeds:
            a = next((r for r in records if r["variant"] == "cnn_bilstm" and r["seed"] == seed), None)
            b = next((r for r in records if r["variant"] == other and r["seed"] == seed), None)
            if a and b:
                deltas.append(a["macro_f1"] - b["macro_f1"])
        d = [x for x in deltas if x is not None and np.isfinite(x)]
        pos, neg = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
        p_value = None
        if pos + neg:
            from scipy import stats
            p_value = float(stats.binomtest(pos, pos + neg, 0.5).pvalue)
        return {"n": len(d), "positive": pos, "negative": neg,
                "mean_delta": float(np.mean(d)) if d else None, "p_value": p_value}

    vs = {v: paired(v) for v in args.variants if v != "cnn_bilstm" and v in summary}
    closest = None
    for v, st in vs.items():
        if st["mean_delta"] is None:
            continue
        if closest is None or st["mean_delta"] < closest["delta"]:
            closest = {"variant": v, "delta": st["mean_delta"]}
    statement = "在本口径下，双分支融合相对所有对照都有正增益" if (vs and all(
        (st["mean_delta"] or 0) > 0 and st["negative"] == 0 for st in vs.values())) else (
        "在本口径下，双分支融合并非对所有对照都有正增益——按项目纪律如实保留这个结果")

    payload = {
        "experiment": "published_baselines",
        "reference_paper": {"title": "基于 PLO-BiLSTM 的网络异常流量检测方法",
                            "authors": "曹鲍垒, 孙知信, 付虹蕾",
                            "venue": "通信技术, 2026, 59(8): 950-960"},
        "args": {"dataset": args.dataset, "variants": list(args.variants),
                 "seeds": list(args.seeds), "epochs": int(args.epochs)},
        "window_size": int(data.window_size), "n_features": int(data.n_features),
        "summary": summary, "records": records,
        "verdict": {"vs": vs, "closest": closest or {"variant": None, "delta": None},
                    "statement": statement},
        "seconds_total": round(time.time() - started, 1),
    }
    (RESULT_DIR / "_pub_baselines").mkdir(parents=True, exist_ok=True)
    save_json(payload, RESULT_DIR / "published_baselines.json")
    save_text(build_report(payload), RESULT_DIR / "published_baselines.md")
    if not args.no_figure:
        make_figure(payload, FIGURE_DIR / "published_baselines.png")
    LOGGER.info("完成：%s（%d 条运行记录，总耗时 %.0fs）",
                RESULT_DIR / "published_baselines.md", len(records), payload["seconds_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
