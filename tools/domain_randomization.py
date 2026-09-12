"""域随机化跨域实验：把"零样本跨域不可用"从一句失败结论，升级成一个**有解方案 + 量化证据**。

问题的来历
----------
工程里已经做过一次诚实的失败实验（`results/cross_dataset_report.md`）：
自研数据训练、直接推理真实公开数据集 → 宏 F1 0.5579、FPR 20.4%；**只重拟合标准化器甚至把 AUC
从 0.6697 打到 0.4742**；诊断出的根因是「**采集口径/域分布不匹配**」，并给出了正确的下一步
——"用目标域统计量反向标定生成器参数（域随机化）"。本工具就是去把这一步**做完并量化**。

实验设计（判据事前写死）
------------------------
造 K 个"域"，每个域=一组不同的**网络/采集条件**（全部用工程自己的生成器旋钮，物理量自洽，
不做任何后处理的伪造字段）：
    D1 基准分布；D2 隐蔽攻击为主；D3 类别边界模糊；D4 强噪声+标注更乱；
    D5 老式明显攻击；D6 业务构成变化（攻击片段数大幅改变）。
对每个**留出域** k，用另外 K−1 个域训练，再对域 k 做**零样本**评估（目标域一个样本都不给）。
三种训练设定，用于把"多样性"和"数据量"分开：

    multi            : 其余 K−1 个域（多样性 + 数据量都增加）
    single           : **只用一个域**（其余域中的第一个）
    multi_matched    : 用 K−1 个域、但每域只取中间一段，使**总行数与 single 相同**
                       （等预算对照：多样性增加、数据量不变）

判据（事前写死）
    1. `J1 多样性有效`：multi 的零样本宏 F1 均值 > single，且逐折配对符号检验 p < 0.05；
    2. `J2 增益来自多样性而非数据量`：multi 优于 multi_matched（同样本量对照）；
    3. `J3 代价`：multi 在**训练域自身**的域内精度（in-domain）相对 single 的下降；
       —— 只有当 J1/J2 成立且 J3 的代价可接受时，才写"域随机化是可行方案"。
    三条都不成立就如实写"在本仿真框架内不奏效"，不修饰。

跨域评估口径（关键）
--------------------
  * 标准化器**只用训练域的 train 段 fit**，目标域原样复用（不做任何目标域标定，这才是零样本）；
  * 目标域的特征列按**训练域的特征名**对齐（同一 schema，列名一致）；
  * 类别编码沿用训练域的编码器（8 类完全相同）；
  * 目标域窗口用与训练完全相同的窗口长度/步长/pure_only 规则。

隔离与产物
----------
  只**新建** `data/raw_domain*/`（各领域的原始 CSV，被 `data/raw*/*.csv` 忽略）与
  `artifacts/dr_*/`，从不触碰 `data/raw`、`data/processed`、主实验 artifacts；
  结果写 `results/domain_randomization.json / .md`、`results/figures/domain_randomization.png`。

用法
----
    python tools/domain_randomization.py                # 完整：6 域 × 6 折 × 3 设定
    python tools/domain_randomization.py --quick        # 快速自检（3 域、2 epoch）
    python tools/domain_randomization.py --folds 0,1    # 只跑指定折（续跑用）
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
from src.data_pipeline import PreparedData, build_windows, prepare_dataset  # noqa: E402
from src.evaluate import binary_metrics, multiclass_metrics, tpr_at_fpr  # noqa: E402
from src.model import predict_probs  # noqa: E402
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

LOGGER = get_logger("cida.domain_random")

# --------------------------------------------------------------------------------------
# 域定义（事前写死）：每个域 = 一组网络/采集条件，全部用生成器自带旋钮
# --------------------------------------------------------------------------------------
DOMAINS: list[dict] = [
    {"key": "D1", "name": "基准分布", "seed_offset": 0, "overrides": {}},
    {"key": "D2", "name": "隐蔽攻击为主", "seed_offset": 11,
     "overrides": {"stealth_ratio": 0.75, "stealth_mix": 0.60}},
    {"key": "D3", "name": "类别边界模糊", "seed_offset": 22,
     "overrides": {"overlap_ratio": 0.65}},
    {"key": "D4", "name": "强噪声+标注更乱", "seed_offset": 33,
     "overrides": {"feature_noise": 0.15, "label_noise": 0.02}},
    {"key": "D5", "name": "老式明显攻击", "seed_offset": 44,
     "overrides": {"stealth_ratio": 0.15, "stealth_mix": 0.20, "overlap_ratio": 0.15}},
    {"key": "D6", "name": "业务构成变化", "seed_offset": 55,
     "overrides": {"episodes_scale": {"attack": 0.45, "normal": 1.8}}},
]
SETTINGS = ("multi", "single", "multi_matched")
P_THRESHOLD = 0.05          # J1 的显著性门槛（事前写死）
DEFAULT_DOMAIN_FLOWS = 0.09  # 每个域的规模（相对主数据 9 万条 → 约 8 千条流）
DEFAULT_EPOCHS = 25


def domain_config(base: Config, spec: dict, *, scale: float) -> Config:
    """按域定义造一个生成器配置（只改旋钮，不改代码路径）。"""
    ov = dict(spec.get("overrides") or {})
    episodes = dict(base.episodes_per_class)
    ep_scale = ov.pop("episodes_scale", None)
    if ep_scale:
        for k in list(episodes):
            factor = ep_scale["normal"] if k == "Normal" else ep_scale["attack"]
            episodes[k] = max(int(round(episodes[k] * factor)), 4)
    return replace(base, seed=base.seed + int(spec["seed_offset"]), scale=float(scale),
                   episodes_per_class=episodes, **ov)


def generate_domains(base: Config, *, scale: float, only: tuple[str, ...] = ()) -> dict[str, Path]:
    """为每个域生成原始 CSV（独立目录，绝不覆盖 data/raw）。"""
    from src.traffic_generator import generate_datasets

    out: dict[str, Path] = {}
    for spec in DOMAINS:
        if only and spec["key"] not in only:
            continue
        d = ROOT / "data" / f"raw_domain_{spec['key']}"
        csv = d / "insdn_stream.csv"
        if csv.exists():
            n = sum(1 for _ in csv.open("r", encoding="utf-8")) - 1
            LOGGER.info("域 %s 已存在（%d 行），跳过生成：%s", spec["key"], n, csv)
        else:
            cfg_k = domain_config(base, spec, scale=scale)
            LOGGER.info("生成域 %s（%s）：旋钮 %s", spec["key"], spec["name"], spec["overrides"])
            generate_datasets(cfg_k, out_dir=d, save=True)
        out[spec["key"]] = csv
    return out


def combine_domains(csvs: list[Path], dst_dir: Path, *, rows_per_domain: int = 0) -> Path:
    """把若干域的原始 CSV 按**时间顺序分块**拼成一份训练数据（域之间加时间偏移）。

    为什么要加偏移：不同域是"不同条件下不同时段的流量"，直接按各自 Timestamp 排序会让多个域的
    行交错在一起，切窗口时会把不同域混进同一个窗口。加一个大于单域时长的时间偏移，
    各域就变成时间轴上的**连续区块**——既符合物理含义，也保住了滑窗语义。

    ``rows_per_domain`` > 0 时，每个域只取**中间连续一段**：这是标准的"等预算"对照
    （总行数与单域基线相同，唯一差异是域的数量/多样性）。
    注意用"中间段"而不是"前 N 行"：片段是分三个阶段排布的，取前 N 行会偏掉类别构成。

    ⚠️ 踩过的坑：最初用"把同一个域重复拼接 K−1 次"来对齐数据量，结果与单域基线**逐位相同**——
    因为数据管线会 `drop_duplicates` 掉完全重复的行，重复拼接等于没加数据。
    等预算对照必须靠"多域 + 每域抽样"，不能靠复制。
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    offset = 0.0
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        df = df.copy()
        if rows_per_domain and len(df) > rows_per_domain:
            mid = len(df) // 2
            half = rows_per_domain // 2
            df = df.iloc[max(mid - half, 0): max(mid - half, 0) + rows_per_domain].reset_index(drop=True)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(0.0) + offset
            offset = float(df["Timestamp"].max()) + 3600.0     # 域之间留 1 小时空档
        frames.append(df)
    # ⚠️ 绝不能把"域标记/轮次"这类辅助列写进训练 CSV：数值型的辅助列会被当成**特征**
    # （踩过：__rep 让特征数从 71 变成 72，两种训练设定因此不可比，还构成轻度泄漏）。
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[:, [c for c in combined.columns if not str(c).startswith("__")]]
    out = dst_dir / "insdn_stream.csv"
    combined.to_csv(out, index=False, encoding="utf-8")
    return out


def target_windows(csv: Path, data: PreparedData, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用**训练域**的标准化器/编码器/窗口规则，把留出域切成可评估的窗口（零样本口径）。"""
    df = pd.read_csv(csv, low_memory=False)
    feats = list(data.feature_names)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"留出域缺少训练域的特征列：{missing[:5]}…")
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    lab_col = "attack_type" if "attack_type" in df.columns and df["attack_type"].nunique() > 2 else "Label"
    labels = df[lab_col].astype(str).str.strip().replace({"BENIGN": "Normal", "benign": "Normal", "normal": "Normal"})
    known = set(data.class_names)
    unknown = sorted(set(labels.unique()) - known)
    if unknown:
        raise SystemExit(f"留出域出现训练域没见过的类别 {unknown}；零样本口径下无法评估")
    X_raw = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    X_scaled = data.scaler.transform(X_raw).astype(np.float32)
    X, y_names, _, info = build_windows(X_scaled, labels.to_numpy(), cfg.window_size,
                                        cfg.window_stride, pure_only=True)
    y = data.label_encoder.transform(np.asarray(y_names).astype(str)).astype(np.int64)
    LOGGER.info("留出域窗口：%d 个（候选 %d，纯标签保留 %d）", X.shape[0],
                info["n_candidates"], info["n_kept"])
    return X, y, np.asarray(y_names)


def eval_on(model, X: np.ndarray, y: np.ndarray, class_names: list[str], cfg: Config) -> dict:
    p_mc, p_bin = predict_probs(model, X, batch_size=cfg.batch_size)
    mc = multiclass_metrics(y, p_mc, class_names)
    normal_id = class_names.index("Normal") if "Normal" in class_names else 0
    y_bin = (y != normal_id).astype(np.int64)
    # 目标域窗口里可能只有单一类别（特别是早期小规模自检）：此时二分类指标无定义，如实返回 None
    if len(np.unique(y_bin)) < 2:
        bm = {"f1": None, "roc_auc": None}
        tpr1 = None
    else:
        bm = binary_metrics(y_bin, p_bin, cfg.report_threshold, cfg)
        tpr1 = float(tpr_at_fpr(y_bin, p_bin, 0.01))
    return {
        "macro_f1": float(mc["macro_f1"]),
        "accuracy": float(mc["accuracy"]),
        "binary_f1": (float(bm["f1"]) if bm.get("f1") is not None else None),
        "binary_roc_auc": (float(bm["roc_auc"]) if bm.get("roc_auc") is not None else None),
        "tpr_at_fpr_1pct": tpr1,
        "n_windows": int(len(y)),
        "per_class_f1": {k: float(v.get("f1", 0.0)) for k, v in (mc.get("per_class") or {}).items()},
    }


def sign_test(deltas: list[float]) -> dict:
    from math import comb

    vals = [d for d in deltas if d is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    neg = sum(1 for d in vals if d < 0)
    pos = sum(1 for d in vals if d > 0)
    k = min(neg, pos)
    return {"n": n, "n_positive": int(pos), "n_negative": int(neg),
            "mean": float(np.mean(vals)), "deltas": [float(v) for v in vals],
            "sign_test_p_two_sided": float(min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="域随机化跨域实验（多域训练 → 留一域零样本）")
    ap.add_argument("--domain-flows", type=float, default=DEFAULT_DOMAIN_FLOWS,
                    help="每个域的规模系数（相对主数据 9 万条）")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--folds", default="", help="只跑这些折（域 key，逗号分隔）；默认全部")
    ap.add_argument("--quick", action="store_true", help="快速自检：3 域 + 2 epoch + 规模很小")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.domain_flows, args.epochs = 0.012, 2
        global DOMAINS
        DOMAINS = DOMAINS[:3]
    started = time.time()
    ensure_dirs(RESULT_DIR, FIGURE_DIR, ARTIFACT_DIR)
    silence_tf_logs()
    set_seed(42)

    base = Config(verbose=False, figures=False, publish_summary=False, epochs=args.epochs)
    keys = [d["key"] for d in DOMAINS]
    only = tuple(x.strip() for x in str(args.folds).split(",") if x.strip())
    csvs = generate_domains(base, scale=float(args.domain_flows), only=only)
    n_per_domain = int(sum(1 for _ in csvs[keys[0]].open("r", encoding="utf-8")) - 1) if csvs else 0
    per_domain_rows = max(n_per_domain // max(len(keys) - 1, 1), 200) if n_per_domain else 0

    folds = [k for k in keys if k in csvs]
    rows: list[dict] = []
    runs: dict[str, dict] = {}
    for k in folds:
        train_keys = [x for x in keys if x != k]
        single_key = train_keys[0]                       # 事前写死：单域基线 = 其余域中的第一个
        LOGGER.info("=" * 78)
        LOGGER.info("留出域 %s（%s）：multi 用 %s 训练；single 用 %s", k,
                    next(d["name"] for d in DOMAINS if d["key"] == k), train_keys, single_key)
        LOGGER.info("=" * 78)
        combos = {
            "multi": ([csvs[x] for x in train_keys], 0),
            "single": ([csvs[single_key]], 0),
            # 等预算对照：域数一样多，但每个域只取中间 (1/(K-1)) 段 → 总行数与单域基线相同
            "multi_matched": ([csvs[x] for x in train_keys], per_domain_rows),
        }
        ctrl_dir = ROOT / "data" / "processed_dr"
        for setting, (src_csvs, rows_each) in combos.items():
            tag = f"dr_{k}_{setting}"
            art = ARTIFACT_DIR / f"dr_{k}"
            combined = combine_domains(src_csvs, ROOT / "data" / f"raw_dr_{k}_{setting}",
                                       rows_per_domain=rows_each)
            cfg_s = replace(base, schema="insdn", source="synthetic", raw_dir=str(combined.parent),
                            processed_dir=str(ctrl_dir), input_mode="sequence",
                            window_size=16, window_stride=5, publish_summary=False, figures=False,
                            batch_size=256)
            data = None
            try:
                data = prepare_dataset(cfg_s, save=False)
            except ValueError as exc:
                LOGGER.error("[%s] 训练数据准备失败（%s）；跳过该设定", tag, exc)
                runs[tag] = {"fold": k, "setting": setting, "error": str(exc)}
                continue
            res = run_training(cfg_s, data, variant="cnn_bilstm", tag=tag, epochs=args.epochs,
                               artifact_dir=art)
            in_dom = eval_on(res.model, data.X_test, data.y_test_mc, data.class_names, cfg_s)
            try:
                Xt, yt, _ = target_windows(csvs[k], data, cfg_s)
                out_dom = eval_on(res.model, Xt, yt, data.class_names, cfg_s)
            except (SystemExit, ValueError) as exc:
                LOGGER.warning("[%s] 留出域窗口不可用（%s）；该折的零样本指标记未测", tag, exc)
                out_dom = {"macro_f1": None, "binary_roc_auc": None, "tpr_at_fpr_1pct": None,
                           "n_windows": 0, "error": str(exc)}
            runs[tag] = {"fold": k, "setting": setting, "in_domain": in_dom, "zero_shot": out_dom,
                         "n_train_rows": int(len(data.X_train)),
                         "n_train_windows": int(data.X_train.shape[0]),
                         "train_seconds": round(res.duration_s, 1)}
            LOGGER.info("[%s] 域内宏F1=%s | 零样本（留出域 %s）宏F1=%s AUC=%s TPR@1%%FPR=%s",
                        tag, fmt_float(in_dom["macro_f1"]), k, fmt_float(out_dom["macro_f1"]),
                        fmt_float(out_dom["binary_roc_auc"]), fmt_float(out_dom["tpr_at_fpr_1pct"]))
            del res
        r = {"fold": k,
             "domain": next(d["name"] for d in DOMAINS if d["key"] == k),
             "single_source": single_key,
             "multi_train_domains": train_keys}
        for setting in SETTINGS:
            t = runs.get(f"dr_{k}_{setting}") or {}
            r[f"{setting}_zero_shot_f1"] = (t.get("zero_shot") or {}).get("macro_f1")
            r[f"{setting}_zero_shot_auc"] = (t.get("zero_shot") or {}).get("binary_roc_auc")
            r[f"{setting}_zero_shot_tpr1"] = (t.get("zero_shot") or {}).get("tpr_at_fpr_1pct")
            r[f"{setting}_in_domain_f1"] = (t.get("in_domain") or {}).get("macro_f1")
        rows.append(r)

    # ---- 判据（事前写死）----
    j1 = sign_test([(r["multi_zero_shot_f1"] or 0) - (r["single_zero_shot_f1"] or 0) for r in rows])
    j2 = sign_test([(r["multi_zero_shot_f1"] or 0) - (r["multi_matched_zero_shot_f1"] or 0)
                    for r in rows])
    cost = sign_test([(r["multi_in_domain_f1"] or 0) - (r["single_in_domain_f1"] or 0) for r in rows])
    judge = {
        "J1_diversity_helps": {
            "pass": bool(j1.get("n") and j1["mean"] > 0 and j1["sign_test_p_two_sided"] < P_THRESHOLD),
            "detail": j1, "rule": f"multi 零样本宏 F1 均值 > single，且符号检验 p < {P_THRESHOLD}"},
        "J2_gain_beyond_volume": {
            "pass": bool(j2.get("n") and j2["mean"] > 0 and j2["sign_test_p_two_sided"] < P_THRESHOLD),
            "detail": j2, "rule": "multi 优于同数据量的单域（multi_matched），排除「只是数据变多」"},
        "J3_in_domain_cost": {
            "detail": cost, "rule": "multi 相对 single 的域内精度变化（负值 = 有代价）"},
    }
    verdict = ("domain_randomization_works"
               if judge["J1_diversity_helps"]["pass"] and judge["J2_gain_beyond_volume"]["pass"]
               else "not_supported")
    payload = {
        "experiment": "domain_randomization_cross_domain",
        "question": "把训练数据换成多个网络/采集条件下的域，零样本跨域是否从「不可用」变成「可用」？",
        "domains": DOMAINS,
        "settings": list(SETTINGS),
        "params": {"domain_flows": float(args.domain_flows), "epochs": int(args.epochs),
                   "window": 16, "stride": 5, "variant": "cnn_bilstm"},
        "folds": rows,
        "runs": runs,
        "judgement": judge,
        "verdict": verdict,
        "elapsed_s": round(time.time() - started, 1),
    }
    save_json(payload, RESULT_DIR / "domain_randomization.json")
    save_text(build_report(payload), RESULT_DIR / "domain_randomization.md")
    if not args.no_figure:
        fp = make_figure(payload, FIGURE_DIR / "domain_randomization.png")
        LOGGER.info("图：%s", fp)
    LOGGER.info("完成：判定 %s（用时 %.0fs）", verdict, payload["elapsed_s"])
    return 0


def build_report(p: dict) -> str:
    rows = p["folds"]
    j = p["judgement"]
    verdict_text = {
        "domain_randomization_works": "**可行**：多域训练显著提升零样本跨域，且增益来自多样性而非数据量。",
        "not_supported": "**在本仿真框架内不成立**：多域训练没有带来显著、可归因于多样性的零样本增益。",
    }[p["verdict"]]
    lines = [
        "# 域随机化跨域：把「零样本跨域不可用」变成一个有解方案（或如实说它不成）",
        "",
        "> 由 `tools/domain_randomization.py` 生成。**判据事前写死**；跨域评估严格零样本"
        "（标准化器只用训练域 fit，目标域一个样本都不参与）。",
        "",
        "## 0. 结论",
        "",
        f"- 判定：{verdict_text}",
        f"- J1（多样性有效）：{j['J1_diversity_helps']['detail'].get('mean')} 的均值提升，"
        f"逐折 {j['J1_diversity_helps']['detail'].get('n_positive')}/"
        f"{j['J1_diversity_helps']['detail'].get('n')} 折为正，"
        f"符号检验 p={fmt_float(j['J1_diversity_helps']['detail'].get('sign_test_p_two_sided'), 3)}"
        f" → {'✅ 通过' if j['J1_diversity_helps']['pass'] else '❌ 未通过'}",
        f"- J2（增益不只是因为数据变多）：同数据量单域对照下均值提升 "
        f"{j['J2_gain_beyond_volume']['detail'].get('mean')}，"
        f"p={fmt_float(j['J2_gain_beyond_volume']['detail'].get('sign_test_p_two_sided'), 3)}"
        f" → {'✅ 通过' if j['J2_gain_beyond_volume']['pass'] else '❌ 未通过'}",
        f"- J3（代价）：域内宏 F1 变化 {j['J3_in_domain_cost']['detail'].get('mean')}"
        f"（负值表示多域训练牺牲了域内精度）",
        "",
        "## 1. 域定义（全部用工程生成器旋钮，物理量自洽）",
        "",
    ]
    head = ["域", "条件", "旋钮"]
    t = [[d["key"], d["name"], "、".join(f"{k}={v}" for k, v in (d["overrides"] or {}).items()) or "默认"]
         for d in p["domains"]]
    lines.append(markdown_table(head, t))
    lines += ["", "## 2. 逐折结果（留一域零样本）", "",
              "> 三种训练设定：`multi` = 其余 K−1 个域；`single` = 只用一个域；"
              "`multi_matched` = 用 K−1 个域、每域只取中间一段，使总行数与 single 相同（等预算对照）。", ""]
    head2 = ["留出域", "条件", "multi 零样本 F1", "single 零样本 F1", "multi_matched",
             "multi AUC", "multi TPR@1%FPR", "multi 域内 F1", "single 域内 F1"]
    t2 = [[r["fold"], r["domain"], fmt_float(r.get("multi_zero_shot_f1")),
           fmt_float(r.get("single_zero_shot_f1")), fmt_float(r.get("multi_matched_zero_shot_f1")),
           fmt_float(r.get("multi_zero_shot_auc")), fmt_float(r.get("multi_zero_shot_tpr1")),
           fmt_float(r.get("multi_in_domain_f1")), fmt_float(r.get("single_in_domain_f1"))]
          for r in rows]
    lines.append(markdown_table(head2, t2))
    multi_mean = np.mean([r["multi_zero_shot_f1"] for r in rows if r.get("multi_zero_shot_f1")])
    single_mean = np.mean([r["single_zero_shot_f1"] for r in rows if r.get("single_zero_shot_f1")])
    lines += [
        "",
        f"- 平均：multi **{fmt_float(multi_mean)}** vs single **{fmt_float(single_mean)}**"
        f"（零样本宏 F1）。",
        "",
        "## 3. 与既有跨域失败实验的关系",
        "",
        "`results/cross_dataset_report.md` 里的失败实验是**自研仿真 → 真实公开数据集**，"
        "诊断出的根因是采集口径/域分布不匹配，并给出下一步建议「域随机化」。"
        "本实验把那个建议**做完**：在仿真框架内造出多个采集条件并做留一域零样本。"
        "两者的区别必须讲清楚：**本实验证明的是「多域训练能否缓解域漂移」这一机制**，"
        "**不能**替代「真实目标域」上的验证（真实数据的口径差异比这里的域间差异更极端）。",
        "",
        "## 4. 局限（写清楚，不粉饰）",
        "",
        "- 域是**用同一生成器旋钮造出来的**，域间差异小于「仿真 vs 真实」的差异；"
        "因此这里的结论是**机制层面的**（多样性有没有用），不是「跨域问题已解决」。",
        "- 每折只有 3 个训练设定、单一模型结构（cnn_bilstm）与单一种子：量级可信，单点数值不宜外推。",
        "- `single` 基线固定取「其余域中的第一个」（事前写死），没有对全部单域求平均"
        "（那样会让算力翻数倍，但结论更稳）；`multi_matched` 提供了「同样本量、更多多样性」这一侧的对照。",
        "- **等预算对照不能靠「复制同一域」实现**：数据管线会 `drop_duplicates` 掉完全重复的行，"
        "复制等于没加数据（本工具最初就踩了这个坑：两次结果逐位相同才发现）；"
        "等预算只能靠「多域 + 每域抽样」。",
        "- 目标域评估用的是与训练相同的窗口规则与 pure_only 过滤；真实部署还需要考虑在线口径"
        "（阈值/确认次数需要按目标域重新标定，见 `tools/calibrate_online.py`）。",
        "",
        "## 5. 复现命令",
        "",
        "```powershell",
        f"python tools/domain_randomization.py --domain-flows {p['params']['domain_flows']} "
        f"--epochs {p['params']['epochs']}",
        "```",
        "",
        "> 隔离：域数据写在 `data/raw_domain_*/`（各域一份 CSV），训练用 `data/raw_dr_*/`，"
        "产物在 `artifacts/dr_*/`，主实验的 `data/raw`、`data/processed` 与 artifacts 全程只读。",
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
    rows = p["folds"]
    labels = [f"{r['fold']}\n{r['domain']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.27
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=150)
    ax = axes[0]
    ax.bar(x - w, [r.get("multi_zero_shot_f1") or 0 for r in rows], w, label="multi（多域训练）",
           color="#4fd1c5")
    ax.bar(x, [r.get("single_zero_shot_f1") or 0 for r in rows], w, label="single（单域）",
           color="#8aa0b6")
    ax.bar(x + w, [r.get("multi_matched_zero_shot_f1") or 0 for r in rows], w,
           label="multi_matched（等预算·多域）", color="#f6ad55")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("零样本宏 F1（留出域）")
    ax.set_title("留一域零样本：多域训练 vs 单域训练")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    delta = [(r.get("multi_zero_shot_f1") or 0) - (r.get("single_zero_shot_f1") or 0) for r in rows]
    delta_m = [(r.get("multi_zero_shot_f1") or 0) - (r.get("multi_matched_zero_shot_f1") or 0)
               for r in rows]
    ax.bar(x - w / 2, delta, w, label="multi − single", color="#3182ce")
    ax.bar(x + w / 2, delta_m, w, label="multi − multi_matched（同数据量）", color="#805ad5")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("零样本宏 F1 提升")
    ax.set_title("增益是否来自「多样性」而非「数据量」")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
