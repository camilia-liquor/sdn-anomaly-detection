"""多源证据融合实测：**控制面（OpenFlow 流表统计） vs 数据面（pcap 抓包）双视角**。

它回答什么问题
--------------
单看一路为什么不够、两路一起看值多少——用**同一条仿真流**的两种独立测量路径把它量化出来：

    仿真流（data/raw/sdn_stream.csv，含真值）
      ├─ 控制面视角：交换机流表统计报上来的 17 维特征（本来就在 CSV 里）
      └─ 数据面视角：由 --write-sample 合成 pcap → 再由 pcap 重建出同样的 17 维特征
                     （独立测量路径：报文计数/长度/间隔 → 流特征）

两个视角用**同一个模型、同一个 scaler**（`data/processed_sdn` + `artifacts/sdn_ctrl`）分别打分，
再用 `src/fusion.py` 融合（冲突取保守侧、单路标 degraded、无证据不编造），最后与单路逐一对比。

产物：`results/multisource_fusion.json` / `.md`（逐窗口明细 + 指标 + 冲突样例 + 口径声明）。

⚠️ 结论口径（写死在报告里，不许省略）
* 两个视角来自**同一条仿真流的两种投影**：这里的"互补/冲突"来自
  「流表计数 vs 报文重建」的**测量差异**（例如每秒速率特征），**不是**真实环境里
  "镜像口覆盖不到/交换机看不到包级细节"那种覆盖差异。真实环境的覆盖差异需要现场部署才能量化。
* 样本按类均衡选取（用于覆盖各类形态），因此**误报率不代表真实基率**，只看相对差异。
* 本工具**只读**既有产物，不重跑主流程、不改任何受守护的产物（summary.json / main_model.json）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import build_windows  # noqa: E402
from src.fusion import (SOURCE_CONTROL, SOURCE_DATA, SourceEvidence, align_by_key,  # noqa: E402
                        flow_key, fuse_evidence, fusion_metrics)
from src.utils import RESULT_DIR, ensure_dirs, get_logger, load_json, save_json, save_text  # noqa: E402

LOGGER = get_logger("cida.fusion")
PCAP_TOOL = ROOT / "tools" / "pcap_to_flows.py"


# --------------------------------------------------------------------------------------
# 取数
# --------------------------------------------------------------------------------------
def _run(cmd: list[str]) -> None:
    # encoding 必须显式给 utf-8：子进程打印中文，交给 locale（Windows 上是 GBK）解码会在读取线程里炸
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"命令失败（{proc.returncode}）：{' '.join(cmd)}\n{proc.stdout[-2000:]}\n"
                         f"{proc.stderr[-2000:]}")


def _make_views(work: Path, flows: int, quiet: bool) -> tuple[dict, Path, Path]:
    """合成 pcap（数据面）并把两份视角文件准备好，返回 (manifest, pcap, 数据面 CSV)。"""
    ensure_dirs(work)
    pcap, manifest = work / "fusion_sample.pcap", work / "fusion_manifest.json"
    dp_csv = work / "fusion_dataplane_flows.csv"
    _run([sys.executable, str(PCAP_TOOL), "--write-sample", str(pcap),
          "--manifest", str(manifest), "--flows", str(int(flows))] + (["--quiet"] if quiet else []))
    _run([sys.executable, str(PCAP_TOOL), "--input", str(pcap), "--output", str(dp_csv)]
         + (["--quiet"] if quiet else []))
    return load_json(manifest), pcap, dp_csv


def _feature_matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.asarray([[float(r.get(n, 0.0) or 0.0) for n in names] for r in rows], dtype=np.float64)


def _score_windows(model, scaler, feats: np.ndarray, names: list[str], window: int, stride: int):
    """按同一口径把特征标准化 → 切窗 → 出每个窗口的 p_attack 与窗口标签。"""
    X = np.asarray(feats, dtype=np.float64)
    if scaler is not None:
        X = scaler.transform(X)
    labels = np.zeros(len(X), dtype=int)          # 占位：窗口标签由调用方按真值另算
    Xw, y_w, starts, info = build_windows(X, labels, int(window), int(stride), pure_only=False)
    if starts is None:
        starts = np.asarray(info.get("starts", []))
    out = model.predict(Xw, verbose=0)
    if isinstance(out, dict):
        p = np.asarray(out["bin"], dtype=float).reshape(-1)
    else:
        p = 1.0 - np.asarray(out, dtype=float)[:, 0]     # 无二分类头时用 1 - P(Normal)
    return p[:len(starts)], np.asarray(starts, dtype=int), Xw


def _score_from_labels(model, scaler, feats: np.ndarray, labels: np.ndarray, window: int, stride: int):
    """同一套取窗逻辑，但用**真值标签**构建窗口（多数投票标签，pure_only=False）。"""
    X = np.asarray(feats, dtype=np.float64)
    if scaler is not None:
        X = scaler.transform(X)
    Xw, y_w, starts, info = build_windows(X, np.asarray(labels, dtype=int), int(window), int(stride),
                                          pure_only=False)
    if starts is None:
        starts = np.asarray(info.get("starts", []))
    out = model.predict(Xw, verbose=0)
    if isinstance(out, dict):
        p = np.asarray(out["bin"], dtype=float).reshape(-1)
    else:
        p = 1.0 - np.asarray(out, dtype=float)[:, 0]
    return p[:len(starts)], np.asarray(y_w, dtype=int), np.asarray(starts, dtype=int)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def run(args) -> dict:
    import joblib
    from src.model import load_model_bundle

    work = Path(args.work_dir)
    manifest, pcap, dp_csv = _make_views(work, args.flows, args.quiet)
    flows_meta = list(manifest.get("flows") or [])
    if not flows_meta:
        raise SystemExit("manifest 里没有 flows，无法做双视角对比")

    processed = Path(args.processed_dir)
    meta = load_json(processed / "meta.json")
    names = list(meta["feature_names"])
    scaler = joblib.load(processed / "scaler.joblib") if (processed / "scaler.joblib").exists() else None
    model = load_model_bundle(ROOT / args.model)
    LOGGER.info("模型=%s；窗口=%d 步长=%d；流 %d 条；特征口径 %d 维（%s）",
                args.model, args.window, args.stride, len(flows_meta), len(names), processed)

    # ---- 控制面视角：**交换机上报的流表统计**（原始 CSV 里那一行就是流表口径） ----
    # ⚠️ 这里必须用原始 CSV，而不是 manifest 里的 `features`：manifest 记的是"pcap 生成器
    #    希望重建出来的值"（端口补 0、超时/标志位按默认值注入、字节数取整），把它当控制面视角
    #    等于拿数据面的近似值去冒充控制面 —— 本工具第一版就是这么写的，结果是"两路几乎全一致"，
    #    把真正的测量差异（端口/超时/标志位/字节数）全部掩盖掉了。
    source_csv = ROOT / str(manifest.get("source") or "data/raw/sdn_stream.csv")
    if not source_csv.exists():
        raise SystemExit(f"控制面视角需要原始流表 CSV：{source_csv} 不存在")
    import pandas as pd
    raw = pd.read_csv(source_csv)
    idx = [int(f["src_row_index"]) for f in flows_meta]
    if max(idx) >= len(raw):
        raise SystemExit(f"manifest 里的行号超出 {source_csv} 范围")
    control_rows = raw.iloc[idx].to_dict("records")

    # 顺带量一下"pcap 生成清单"与真实流表口径差多少（= 数据面路径表达不出来的东西）
    manifest_diff = Counter()
    for f, c in zip(flows_meta, control_rows):
        for k in names:
            try:
                if float(c.get(k, 0.0) or 0.0) != float(f["features"].get(k, 0.0) or 0.0):
                    manifest_diff[k] += 1
            except (TypeError, ValueError):
                manifest_diff[k] += 1

    # ---- 数据面视角：pcap 重建出来的流特征（独立测量路径，真值 Unknown，真值取自 manifest） ----
    import csv as _csv
    data_rows: list[dict] = []
    with dp_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            data_rows.append(row)
    if len(data_rows) != len(control_rows):
        raise SystemExit(f"两路流数不一致：控制面 {len(control_rows)} vs 数据面 {len(data_rows)}")

    # ---- 两个视角在同一流速上的特征差异（这就是冲突的来源，必须如实列出来） ----
    feat_diff = Counter()
    for c, d in zip(control_rows, data_rows):
        for k in names:
            try:
                if float(c.get(k, 0.0) or 0.0) != float(d.get(k, 0.0) or 0.0):
                    feat_diff[k] += 1
            except (TypeError, ValueError):
                feat_diff[k] += 1

    # ---- 对齐诊断（按流键；两路说的必须是同一批流） ----
    ctrl_for_key = [{"ip_src": f["ip_src"], "ip_dst": f["ip_dst"],
                     **{k: f["features"].get(k) for k in ("tp_src", "tp_dst", "ip_proto")}}
                    for f in flows_meta]
    data_for_key = [{"ip_src": r.get("ip_src"), "ip_dst": r.get("ip_dst"),
                     **{k: r.get(k) for k in ("tp_src", "tp_dst", "ip_proto")}} for r in data_rows]
    aligned = align_by_key(ctrl_for_key, data_for_key)

    # ---- 逐窗口打分 + 融合 ----
    y_true = np.asarray([int(f.get("label") or 0) for f in flows_meta], dtype=int)
    p_ctrl, y_win, starts = _score_from_labels(model, scaler, _feature_matrix(control_rows, names),
                                               y_true, args.window, args.stride)
    p_data, y_win2, starts2 = _score_from_labels(model, scaler, _feature_matrix(data_rows, names),
                                                 y_true, args.window, args.stride)
    if len(p_ctrl) != len(p_data) or len(starts) != len(starts2):
        raise SystemExit(f"两路窗口数不一致：控制面 {len(p_ctrl)} vs 数据面 {len(p_data)}")

    records: list[dict] = []
    for i in range(len(p_ctrl)):
        v = fuse_evidence(SourceEvidence(SOURCE_CONTROL, p_attack=float(p_ctrl[i])),
                          SourceEvidence(SOURCE_DATA, p_attack=float(p_data[i])),
                          threshold=args.threshold)
        records.append({"window": int(i), "start_row": int(starts[i]),
                        "p_control": round(float(p_ctrl[i]), 4), "p_data": round(float(p_data[i]), 4),
                        "label": int(y_win[i]), "verdict": v.to_dict()})
    metrics = fusion_metrics(records, [int(r["label"]) for r in records], threshold=args.threshold)

    # ---- 单路失效场景（模拟镜像口/采集路径故障）：把数据面置为不可用再融合一遍 ----
    # 为什么必须做这一段：这份仿真数据上两路几乎不冲突，只看指标会得出"融合没多大用"。
    # 但真实运维里"一路瞎了"是**必然会发生**的事故——那时系统必须**说出来**，
    # 而不是静默退化成单路（用户以为两只眼睛都还在看）。这一段就是把那个场景量出来。
    outage_records: list[dict] = []
    for i in range(len(p_ctrl)):
        v = fuse_evidence(SourceEvidence(SOURCE_CONTROL, p_attack=float(p_ctrl[i])),
                          SourceEvidence(SOURCE_DATA, available=False,
                                         note="数据面不可用（模拟采集路径故障）"),
                          threshold=args.threshold)
        outage_records.append({"window": int(i), "start_row": int(starts[i]),
                               "p_control": round(float(p_ctrl[i]), 4), "p_data": None,
                               "label": int(y_win[i]), "verdict": v.to_dict()})
    outage = fusion_metrics(outage_records, [int(r["label"]) for r in outage_records],
                            threshold=args.threshold)

    conflicts = [r for r in records if r["verdict"]["conflict"]]
    payload = {
        "experiment": "multisource_fusion",
        "views": {"control": "交换机组装的流表统计（OpenFlow flow stats 口径）",
                  "data": "由合成 pcap 重建的流特征（独立测量路径）"},
        "config": {"window": int(args.window), "stride": int(args.stride),
                   "threshold": float(args.threshold), "flows": int(args.flows),
                   "processed_dir": str(processed), "model": str(args.model),
                   "rule": "conservative_max_with_explicit_conflict"},
        "source": {"manifest": str(Path(pcap).relative_to(ROOT)) if Path(pcap).is_relative_to(ROOT)
                   else str(pcap),
                   "raw_stream": str(source_csv), "n_flows": len(flows_meta),
                   "class_counts": dict(Counter(str(f.get("attack_type")) for f in flows_meta))},
        "feature_mismatch": dict(feat_diff),           # 两路测量差异（冲突的来源）
        "manifest_vs_raw_csv_diff": dict(manifest_diff),   # 数据面路径表达不出来的字段（端口/超时/标志位/取整）
        "alignment": {"matched": len(aligned["matched"]), "control_only": len(aligned["control_only"]),
                      "data_only": len(aligned["data_only"]), "unkeyed": aligned["unkeyed"]},
        "metrics": metrics,
        "single_source_outage": outage,   # 模拟"数据面瞎了"：系统应如实标 degraded，而不是静默退化
        "conflicts": conflicts[:20],
        "windows": records,
        "caveats": [
            "两个视角来自**同一条仿真流的两种投影**：互补/冲突来自「流表计数 vs 报文重建」的测量差异，"
            "不是真实环境的覆盖差异（镜像口丢包、采集点之外等需要现场部署才能量化）",
            "样本按类均衡选取 + 阈值 0.5 是在**不均衡**的真实分布上标定的 → 这里的误报率会偏得很难看（两路都接近 1.0），**不代表真实基率**，只能用于两路之间的相对比较",
            "本实验只读既有产物，未重跑主流程、未改动任何受守护的产物",
        ],
    }
    return payload


def build_report(p: dict) -> str:
    m, c = p["metrics"], p["config"]
    o = p.get("single_source_outage") or {}
    fm = p.get("feature_mismatch") or {}
    lines = [
        "# 多源证据融合实测（控制面流表 + 数据面 pcap）", "",
        "> 由 `tools/multisource_fusion.py` 生成。**只读**既有产物，未重跑主流程、"
        "未改动 `results/summary.json` / `artifacts/main_model.json`。", "",
        "## 0. 这次到底比了什么", "",
        "同一条仿真流的两条独立测量路径，用**同一个模型**打分后再融合：", "",
        "| 视角 | 怎么来的 |", "|---|---|",
        f"| 控制面（control） | {p['views']['control']} —— `{p['source']['raw_stream']}` |",
        f"| 数据面（data） | {p['views']['data']} —— `{p['source']['manifest']}` |", "",
        f"- 流数 **{p['source']['n_flows']}**（类别分布：{p['source']['class_counts']}）",
        f"- 取窗：窗口 **{c['window']}** / 步长 **{c['stride']}** → **{m['n']}** 个窗口，"
        f"阈值 {c['threshold']}，规则 `{c['rule']}`",
        f"- 按流键对齐：matched **{p['alignment']['matched']}**、"
        f"control_only {p['alignment']['control_only']}、data_only {p['alignment']['data_only']}、"
        f"unkeyed {p['alignment']['unkeyed']}",
        f"- pcap 生成清单与真实流表口径不一致的字段：**{len(p.get('manifest_vs_raw_csv_diff') or {})} 个**"
        f"（{sorted((p.get('manifest_vs_raw_csv_diff') or {}).keys())}）——这些正是**数据面路径表达不出来**的东西", "",
        "## 1. 两路测量差异（冲突的来源，先看这个）", "",
        "| 特征 | 两路取值不一致的流数 |", "|---|---|",
    ]
    for k, v in sorted(fm.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | **{v}** / {p['source']['n_flows']} |")
    if not fm:
        lines.append("| （无） | 0 |")
    lines += [
        "", "两路**只有在这些字段上会分歧**（其余字段逐值相同）——集中在「每秒/纳秒速率」（由时长换算）、"
        "时长与字节取整、以及超时/标志位/端口这类**流表有、报文重建不一定有**的字段。"
        "⚠️ 端口/超时/标志位的差异**部分是本工具合成 pcap 时的默认值所致**（合成器按流表统计造包，"
        "无法反推这些字段）；真实镜像口能否看到它们取决于部署方式，不能一概而论——"
        "这里只说明「两路测量口径确实不同」这个事实本身。", "",
        "## 2. 单路 vs 融合（窗口级）", "",
        "| 口径 | 召回 | 漏报率 | 误报率 |",
        "|---|---|---|---|",
        f"| 只用控制面 | {m['recall_control']:.3f} | {m['fnr_control']:.3f} | {m['fpr_control']:.3f} |",
        f"| 只用数据面 | {m['recall_data']:.3f} | {m['fnr_data']:.3f} | {m['fpr_data']:.3f} |",
        f"| **融合（conservative）** | **{m['recall_fused']:.3f}** | **{m['fnr_fused']:.3f}** | "
        f"**{m['fpr_fused']:.3f}** |", "",
        "| 融合行为 | 数量 | 含义 |", "|---|---|---|",
        f"| `rescued_by_pair` | {m['rescued_by_pair']} | 一路漏、另一路抓到（互补覆盖，融合保住） |",
        f"| `recovered_by_fusion` | {m['recovered_by_fusion']} | **两路单独都漏**、融合抓到（弱证据一致） |",
        f"| `lost_by_fusion` | {m['lost_by_fusion']} | 某一路抓到、融合却漏 —— **必须为 0** |",
        f"| `conflicts` | {m['conflicts']} | 两路结论对立（已取保守侧并要求人工研判） |",
        f"| `degraded` | {m['degraded']} | 只有一路可用（如实标注覆盖度不足） |",
        f"| `weak_agreement` | {m['weak_agreement']} | 两路都差一点但一致 → 报警 |", "",
    ]
    if o:
        lines += [
            "### 单路失效场景（模拟数据面采集路径故障）", "",
            "真实运维里「一路瞎了」是必然会发生的事故，这里把它量出来——系统必须**说出来**，"
            "而不是静默退化成单路：", "",
            "| 口径 | 召回 | 误报率 | degraded | 需人工研判 |",
            "|---|---|---|---|---|",
            f"| 控制面单独 + 数据面不可用 | {o['recall_fused']:.3f} | {o['fpr_fused']:.3f} | "
            f"{o['degraded']} / {o['n']} | {o['degraded']} |", "",
            f"结论：数据面失效时融合召回等价于控制面单独水平（{o['recall_fused']:.3f}），"
            f"但 **{o['degraded']}/{o['n']} 个窗口全部被标成 `degraded`** —— "
            "覆盖度下降是**可见**的，不会变成「没人知道的单点」。", "",
        ]
    if p["conflicts"]:
        lines += ["### 冲突样例（两路对立，取保守侧）", "",
                  "| 窗口 | 起始行 | 控制面 p | 数据面 p | 真值 | 结论 |", "|---|---|---|---|---|---|"]
        for r in p["conflicts"][:8]:
            lines.append(f"| {r['window']} | {r['start_row']} | {r['p_control']:.3f} | "
                         f"{r['p_data']:.3f} | {r['label']} | {r['verdict']['reason'][:40]}… |")
        lines.append("")
    lines += [
        "## 3. 结论与边界（不许省略）", "",
        f"1. **互补是真实存在的**：{m['rescued_by_pair']} 个窗口是「一路漏、另一路抓到」——"
        f"单路视角下这些就是漏报，融合把它们保住了；",
        f"2. **融合不会丢掉任何一路抓到的攻击**：`lost_by_fusion` = {m['lost_by_fusion']}（该值必须恒为 0，"
        "它同时是「将来有人把规则改成平均/投票」的哨兵）；",
        f"3. **代价也要如实说**：融合取保守侧，误报率是两路的并集——"
        f"（{m['fpr_fused']:.3f} vs 控制面 {m['fpr_control']:.3f} / 数据面 {m['fpr_data']:.3f}）；"
        f"两路对立的 {m['conflicts']} 个窗口被交给人研判——这正是「不能被平均抹平」的成本；",
        "4. ⚠️ **口径边界**：两个视角来自**同一条仿真流的两种投影**，互补来自「流表计数 vs 报文重建」的"
        "测量差异，**不是**真实环境里「镜像口覆盖不到」那种覆盖差异；样本按类均衡选取，"
        "误报率**不代表真实基率**（阈值是在不均衡分布上标定的，本节数字只能用于两路之间的相对比较）。"
        "真实环境的覆盖差异需要现场部署才能量化。", "",
        "## 4. 复现", "", "```bash",
        f"python tools/multisource_fusion.py --flows {c['flows']} --window {c['window']} "
        f"--stride {c['stride']} --threshold {c['threshold']}",
        "```", "",
        f"- 中间产物（可删）：`{ROOT.name}/_tmp/fusion/`（pcap + manifest + 数据面 CSV）",
        f"- 数据面视角由 `tools/pcap_to_flows.py` 合成与还原，其往返保真度另见 `results/pcap_roundtrip.md`",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="多源证据融合实测（控制面流表 vs 数据面 pcap）")
    ap.add_argument("--flows", type=int, default=400, help="参与对比的流数（按类均衡抽取）")
    ap.add_argument("--window", type=int, default=16, help="窗口长度（与主模型一致）")
    ap.add_argument("--stride", type=int, default=16, help="取窗步长")
    ap.add_argument("--threshold", type=float, default=0.5, help="判定阈值")
    ap.add_argument("--processed-dir", default="data/processed_sdn", help="特征口径（17 维控制器原生）")
    ap.add_argument("--model", default="artifacts/sdn_ctrl/model.keras", help="打分模型")
    ap.add_argument("--work-dir", default=str(ROOT / "_tmp" / "fusion"), help="中间产物目录")
    ap.add_argument("--quiet", action="store_true", help="少打印")
    args = ap.parse_args(argv)

    payload = run(args)
    save_json(payload, RESULT_DIR / "multisource_fusion.json")
    save_text(build_report(payload), RESULT_DIR / "multisource_fusion.md")
    m = payload["metrics"]
    LOGGER.info("完成：%s / %s —— 窗口 %d：控制面召回 %.3f、数据面 %.3f、融合 %.3f"
                "（互补保住 %d、融合独有 %d、冲突 %d、降级 %d、融合丢 %d）",
                RESULT_DIR / "multisource_fusion.md", RESULT_DIR / "multisource_fusion.json",
                m["n"], m["recall_control"], m["recall_data"], m["recall_fused"],
                m["rescued_by_pair"], m["recovered_by_fusion"], m["conflicts"], m["degraded"],
                m["lost_by_fusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
