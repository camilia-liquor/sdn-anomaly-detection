"""一键跑通全流程：数据生成 → 数据准备 → 训练 → 评估 → 出图 → 汇总报告。

常用命令
--------
    python run_all.py                      # 完整流程（默认约 6 万条仿真流记录 + 40 epoch）
    python run_all.py --smoke              # 小规模快跑自检（几分钟内出结果，验证环境）
    python run_all.py --ablation           # 额外跑 4 个模型变体做消融实验
    python run_all.py --set train_epochs=60 --set window_size=12
    python run_all.py --config my.json --skip-data      # 复用已生成的数据
    python run_all.py --source real --real-data-dir D:\\datasets\\inSDN   # 换成真实数据集

产物
----
    data/raw/                 原始仿真 CSV（两种 schema 可选）
    data/processed/           张量 / scaler / 标签编码器
    artifacts/<variant>/      模型、训练历史、逐窗口预测缓存
    results/                  指标 JSON/CSV、图片、汇总报告
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 允许 `python run_all.py` 直接运行（把项目根目录加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.data_pipeline import prepare_dataset
from src.evaluate import evaluate_model
from src.train import run_training
from src.utils import (
    ARTIFACT_DIR,
    FIGURE_DIR,
    LOG_DIR,
    RESULT_DIR,
    banner,
    ensure_dirs,
    get_logger,
    markdown_table,
    resolve_dir,
    save_json,
    save_text,
    set_seed,
    timestamp_tag,
)
from src.visualize import make_all_figures, plot_ablation

LOGGER = get_logger("cida.run")


# --------------------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNN+BiLSTM SDN 异常检测：一键实验流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="", help="JSON 配置文件路径")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="覆盖任意配置项，可重复使用")
    p.add_argument("--smoke", action="store_true", help="小规模快跑（自动缩小数据量与 epoch）")
    p.add_argument("--ablation", action="store_true", help="跑全部模型变体做消融实验")
    p.add_argument("--baselines", nargs="?", const="random_forest,extra_trees,logistic_regression",
                   default="", metavar="LIST",
                   help="额外训练传统机器学习基线（默认 RF/ET/LR，可与消融一起对比）")
    p.add_argument("--variants", type=str, default="", help="逗号分隔的模型变体列表，如 cnn,lstm")
    p.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    p.add_argument("--scale", type=float, default=None, help="覆盖仿真数据规模系数")
    p.add_argument("--input-mode", type=str, default=None, choices=["sequence", "sample"])
    p.add_argument("--source", type=str, default=None, choices=["synthetic", "real"])
    p.add_argument("--real-data-dir", type=str, default=None)
    p.add_argument("--skip-data", action="store_true", help="跳过数据生成与预处理，复用 data/processed")
    p.add_argument("--skip-train", action="store_true", help="跳过训练，直接评估已有模型")
    p.add_argument("--no-figures", action="store_true", help="不出图")
    p.add_argument("--tag", type=str, default="", help="本次运行标签（产物目录名）")
    p.add_argument("--seeds", type=str, default="",
                   help="多种子重复实验，逗号分隔，如 0,1,2（数据固定、只变模型随机种子，报告均值±标准差）")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_json(args.config) if args.config else Config()
    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式：{item}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    if overrides:
        cfg.apply_overrides(overrides)
    if args.scale is not None:
        cfg.scale = float(args.scale)
    if args.input_mode:
        cfg.input_mode = args.input_mode
    if args.source:
        cfg.source = args.source
    if args.real_data_dir:
        cfg.real_data_dir = args.real_data_dir
    if args.tag:
        cfg.run_name = args.tag

    if args.smoke:
        # 快跑模式：约 1 分钟跑完，但仍要保证"每类在三个时间段都有样本"、
        # 且训练步数足够让模型真的学起来（否则评委看到的就是一堆没意义的数字）。
        cfg.scale = min(cfg.scale, 0.5)
        cfg.episodes_per_class = {
            "Normal": 30, "DDoS": 10, "DoS": 10, "Probe": 12,
            "BruteForce": 12, "WebAttack": 12, "Botnet": 12, "U2R": 12,
        }
        cfg.epochs = min(cfg.epochs, 25)
        cfg.early_stop_patience = min(cfg.early_stop_patience, 6)
        # ★ 快跑使用**独立的数据目录**：否则跑一次快跑就会把正式实验的 data/raw、
        #   data/processed 换成小规模版本，演示大屏与后续复评都会跟着错。
        cfg.raw_dir = cfg.raw_dir or "data/raw_smoke"
        cfg.processed_dir = cfg.processed_dir or "data/processed_smoke"
        cfg.run_name = cfg.run_name + "_smoke" if not cfg.run_name.endswith("_smoke") else cfg.run_name
    return cfg


def make_tag(run_name: str, variant: str, seed: int, n_variants: int, n_seeds: int,
             main_variant: str | None = None) -> str:
    """产物目录命名：主模型不带变体后缀，对照变体带；多种子再带 _s<seed>。"""
    parts = [run_name]
    if n_variants > 1 and variant != main_variant:
        parts.append(variant)
    if n_seeds > 1:
        parts.append(f"s{seed}")
    return "_".join(parts)


def aggregate_runs(runs: list[dict]) -> dict[str, dict]:
    """按模型变体聚合多种子结果，返回 {variant: {metric: {mean, std, n}}}。"""
    import numpy as np

    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["variant"], []).append(r["report"])
    out: dict[str, dict] = {}
    metrics = {
        "multiclass_accuracy": lambda x: x["multiclass"]["accuracy"],
        "multiclass_macro_f1": lambda x: x["multiclass"]["macro_f1"],
        "binary_f1": lambda x: x["binary"]["f1"],
        "binary_roc_auc": lambda x: x["binary"]["roc_auc"],
        "binary_pr_auc": lambda x: x["binary"]["pr_auc"],
        "tpr_at_fpr_1pct": lambda x: x["binary"].get("tpr_at_fpr_0.01", float("nan")),
        "segment_recall": lambda x: (x.get("online") or {}).get("segment_recall", float("nan")),
        "false_alarms_per_hour": lambda x: (x.get("online") or {}).get("false_alarms_per_hour", float("nan")),
        "mean_detection_delay_sec": lambda x: (x.get("online") or {}).get("mean_detection_delay_sec", float("nan")),
        "n_params": lambda x: x["model"]["n_params"],
    }
    for variant, reps in groups.items():
        entry = {"n_seeds": len(reps)}
        for name, fn in metrics.items():
            vals = np.asarray([fn(r) for r in reps], dtype=float)
            entry[name] = {
                "mean": float(np.nanmean(vals)) if len(vals) else float("nan"),
                "std": float(np.nanstd(vals)) if len(vals) else float("nan"),
            }
        out[variant] = entry
    return out


def _aggregate_table(aggregate: dict[str, dict]) -> str:
    rows = []
    for variant, m in aggregate.items():
        rows.append(
            f"  {variant:<16} acc={m['multiclass_accuracy']['mean']:.4f}±{m['multiclass_accuracy']['std']:.4f}  "
            f"macroF1={m['multiclass_macro_f1']['mean']:.4f}±{m['multiclass_macro_f1']['std']:.4f}  "
            f"binF1={m['binary_f1']['mean']:.4f}±{m['binary_f1']['std']:.4f}  "
            f"AUC={m['binary_roc_auc']['mean']:.4f}±{m['binary_roc_auc']['std']:.4f}  "
            f"片段召回={m['segment_recall']['mean']:.3f}±{m['segment_recall']['std']:.3f}"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------------------
# 汇总报告
# --------------------------------------------------------------------------------------
def write_summary(cfg: Config, report: dict, train_summary: dict | None, figures: list[Path],
                  ablation: dict[str, dict] | None, out_dir: Path | None = None,
                  aggregate: dict[str, dict] | None = None, main_tag: str = "") -> Path:
    out_dir = Path(out_dir or RESULT_DIR)
    mc = report["multiclass"]
    bn = report["binary"]
    online = report.get("online") or {}
    main_tag = main_tag or cfg.run_name

    lines = [
        "# 实验汇总（自动生成）",
        "",
        f"- 运行标签：`{main_tag}`",
        f"- 数据来源：`{cfg.source}`（schema=`{cfg.schema}`，输入模式=`{cfg.input_mode}`，窗口={cfg.window_size}）",
        f"- 随机种子：`{cfg.seed}`（固定种子 + 时序切分，结果可复现）",
        "",
        "## 1. 数据规模",
        "",
        f"- 训练/验证/测试窗口数：{report['data']['train']} / {report['data']['val']} / {report['data']['test']}",
        f"- 特征数：{report['data']['n_features']}，类别数：{report['data']['n_classes']}",
        f"- 类别：{', '.join(report['data']['class_names'])}",
        "",
        "## 2. 多分类指标",
        "",
        markdown_table(
            ["指标", "数值"],
            [
                ["准确率 Accuracy", f"{mc['accuracy']:.4f}"],
                ["宏平均精确率 Precision(macro)", f"{mc['macro_precision']:.4f}"],
                ["宏平均召回率 Recall(macro)", f"{mc['macro_recall']:.4f}"],
                ["宏平均 F1 F1(macro)", f"{mc['macro_f1']:.4f}"],
                ["加权 F1 F1(weighted)", f"{mc['weighted_f1']:.4f}"],
                ["Top-2 准确率", f"{mc['top2_accuracy']:.4f}"],
                ["Kappa 系数", f"{mc['cohen_kappa']:.4f}"],
                ["宏平均 AUC(OvR)", f"{mc.get('macro_roc_auc_ovr', float('nan')):.4f}"],
            ],
        ),
        "",
        "### 逐类表现",
        "",
        markdown_table(
            ["类别", "精确率", "召回率", "F1", "样本数"],
            [
                [n, f"{mc['per_class'][n]['precision']:.4f}", f"{mc['per_class'][n]['recall']:.4f}",
                 f"{mc['per_class'][n]['f1']:.4f}", mc['per_class'][n]['support']]
                for n in report["data"]["class_names"] if n in mc["per_class"]
            ],
        ),
        "",
        "## 3. 二分类异常检测（正常 vs 异常）",
        "",
        markdown_table(
            ["指标", "数值"],
            [
                ["判定阈值", f"{bn['threshold']:.2f}"],
                ["准确率", f"{bn['accuracy']:.4f}"],
                ["平衡准确率", f"{bn['balanced_accuracy']:.4f}"],
                ["精确率", f"{bn['precision']:.4f}"],
                ["召回率", f"{bn['recall']:.4f}"],
                ["F1", f"{bn['f1']:.4f}"],
                ["ROC-AUC", f"{bn['roc_auc']:.4f}"],
                ["PR-AUC", f"{bn['pr_auc']:.4f}"],
                ["MCC", f"{bn['mcc']:.4f}"],
                ["TPR@FPR=1%", f"{bn.get('tpr_at_fpr_0.01', float('nan')):.4f}"],
                ["TPR@FPR=0.1%", f"{bn.get('tpr_at_fpr_0.001', float('nan')):.4f}"],
                ["最优阈值 / 最优 F1", f"{report['binary_best_threshold']['threshold']:.3f} / {report['binary_best_threshold']['f1']:.4f}"],
            ],
        ),
    ]
    if online:
        lines += [
            "",
            "## 4. 在线（告警级）指标 —— 贴近真实部署",
            "",
            markdown_table(
                ["指标", "数值"],
                [
                    ["测试流长度", f"{online.get('stream_rows')} 条流记录 / {online.get('stream_hours', 0):.2f} 小时"],
                    ["攻击片段数", online.get("n_attack_segments")],
                    ["成功检出片段", online.get("n_segments_detected")],
                    ["片段召回率", f"{online.get('segment_recall', float('nan')):.4f}"],
                    ["平均检测时延", f"{online.get('mean_detection_delay_sec')} 秒"],
                    ["P95 检测时延", f"{online.get('p95_detection_delay_sec')} 秒"],
                    ["误报次数", online.get("false_alarms")],
                    ["误报率", f"{online.get('false_alarms_per_hour')} 次/小时"],
                ],
            ),
        ]
    if train_summary:
        lines += [
            "",
            "## 5. 训练信息",
            "",
            markdown_table(
                ["项目", "数值"],
                [
                    ["模型变体", train_summary.get("variant")],
                    ["参数量", f"{train_summary.get('n_params', 0):,}"],
                    ["实际训练轮数", train_summary.get("epochs_run")],
                    ["最优 epoch", train_summary.get("best_epoch", 0) + 1],
                    ["最优验证损失", f"{train_summary.get('best_val_loss', float('nan')):.4f}"],
                    ["训练耗时", f"{train_summary.get('train_seconds')} 秒"],
                ],
            ),
        ]
    if ablation:
        lines += ["", "## 6. 消融实验（各结构贡献）", ""]
        rows = []
        for tag, rep in ablation.items():
            rows.append([
                tag,
                f"{rep['multiclass']['accuracy']:.4f}",
                f"{rep['multiclass']['macro_f1']:.4f}",
                f"{rep['binary']['f1']:.4f}",
                f"{rep['binary']['roc_auc']:.4f}",
                f"{rep['model']['n_params']:,}",
            ])
        lines += [markdown_table(["变体", "多分类准确率", "多分类宏F1", "二分类F1", "二分类AUC", "参数量"], rows)]

    if aggregate:
        lines += [
            "",
            f"## 6b. 多种子重复实验（均值 ± 标准差，n={max(v.get('n_seeds', 1) for v in aggregate.values())}）",
            "",
            "同一份数据、只改变模型随机种子，用于确认结论不是「运气好」跑出来的。",
            "",
        ]
        rows = []
        for variant, m in aggregate.items():
            rows.append([
                variant,
                f"{m['multiclass_accuracy']['mean']:.4f} ± {m['multiclass_accuracy']['std']:.4f}",
                f"{m['multiclass_macro_f1']['mean']:.4f} ± {m['multiclass_macro_f1']['std']:.4f}",
                f"{m['binary_f1']['mean']:.4f} ± {m['binary_f1']['std']:.4f}",
                f"{m['binary_roc_auc']['mean']:.4f} ± {m['binary_roc_auc']['std']:.4f}",
                f"{m['segment_recall']['mean']:.3f} ± {m['segment_recall']['std']:.3f}",
                f"{m['n_params']['mean']:,.0f}",
            ])
        lines += [markdown_table(
            ["变体", "多分类准确率", "多分类宏F1", "二分类F1", "二分类AUC", "在线片段召回", "参数量"], rows
        )]

    if figures:
        lines += ["", "## 7. 图表清单", ""]
        lines += [f"- `{p.relative_to(p.parents[1]) if p.parents else p.name}`" for p in figures]

    text = "\n".join(lines) + "\n"
    path = out_dir / f"summary_{cfg.run_name}.md"
    save_text(text, path)
    # 同时在项目根目录留一份"最新汇总"方便直接打开；
    # 但快跑（--smoke）与补充实验（--set publish_summary=false）不能覆盖正式实验的汇总。
    if "smoke" not in cfg.run_name.lower() and getattr(cfg, "publish_summary", True):
        save_text(text, RESULT_DIR / "汇总报告.md")
        save_json(
            {
                "tag": main_tag,
                "run_name": cfg.run_name,
                "multiclass": {k: v for k, v in mc.items() if not k.startswith("confusion") and k != "per_class"},
                "binary": bn,
                "online": online,
                "train": train_summary,
                "aggregate": aggregate,
                "config": cfg.to_dict(),
            },
            out_dir / "summary.json",
        )
    return path


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    ensure_dirs()
    log_path = LOG_DIR / f"run_{cfg.run_name}_{timestamp_tag()}.log"
    global LOGGER
    LOGGER = get_logger("cida.run", log_file=log_path)
    set_seed(cfg.seed)

    banner(LOGGER, "CNN + BiLSTM  SDN 异常检测流水线")
    LOGGER.info("配置：%s", {k: v for k, v in cfg.to_dict().items() if k in
                            ("schema", "source", "scale", "input_mode", "window_size", "window_stride",
                             "model_variant", "epochs", "batch_size", "learning_rate", "seed")})
    LOGGER.info("日志文件：%s", log_path)
    t_start = time.time()

    # ---------- 1. 数据 ----------
    data = None
    if args.skip_data:
        from src.data_pipeline import PreparedData

        LOGGER.info("跳过数据生成，直接读取 data/processed ……")
        data = PreparedData.load()
    else:
        banner(LOGGER, "步骤 1/5  生成并预处理数据")
        data = prepare_dataset(cfg, force_regenerate=not args.skip_data)

    # ---------- 2. 训练 ----------
    variants = [cfg.model_variant]
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    elif args.ablation:
        variants = ["cnn_bilstm", "cnn", "bilstm", "mlp"]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()] or [cfg.seed]

    banner(LOGGER, f"步骤 2/5  训练模型：{', '.join(variants)}（随机种子：{seeds}）")
    runs: list[dict] = []
    for v in variants:
        for seed in seeds:
            tag = make_tag(cfg.run_name, v, seed, len(variants), len(seeds), cfg.model_variant)
            cfg_i = Config.from_dict(cfg.to_dict())
            cfg_i.seed = int(seed)
            if args.skip_train:
                from src.model import load_model_bundle

                model_path = ARTIFACT_DIR / tag / "model.keras"
                if not model_path.exists():
                    LOGGER.error("未找到已训练模型 %s，跳过该变体", model_path)
                    continue
                model, result = load_model_bundle(model_path), None
            else:
                result = run_training(cfg_i, data, variant=v, tag=tag, epochs=args.epochs)
                model = result.model
            report = evaluate_model(model, data, cfg_i, tag=tag)
            runs.append({"variant": v, "seed": int(seed), "tag": tag,
                         "report": report, "result": result})

    if not runs:
        LOGGER.error("没有任何模型完成评估，请检查上面的报错。")
        return 1

    # ---------- 2.5 传统机器学习基线 ----------
    if args.baselines:
        banner(LOGGER, "步骤 2.5/5  训练传统机器学习基线：" + args.baselines)
        from src.baselines import run_baselines

        names = [n.strip() for n in str(args.baselines).split(",") if n.strip()]
        for name, bl_model in run_baselines(data, cfg, names=names).items():
            tag = f"{cfg.run_name}_{name}"
            # 基线模型也落盘，便于之后用 tools/reevaluate.py 重新评估（不用重训）
            try:
                import joblib

                out = ARTIFACT_DIR / tag
                out.mkdir(parents=True, exist_ok=True)
                # 压缩保存：未剪枝的随机森林/极端随机树模型动辄上百 MB
                joblib.dump(bl_model, out / "baseline.joblib", compress=3)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("基线模型保存失败：%s", exc)
            try:
                report_bl = evaluate_model(bl_model, data, cfg, tag=tag, save_predictions=False)
                runs.append({"variant": name, "seed": cfg.seed, "tag": tag,
                             "report": report_bl, "result": None})
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("基线 %s 评估失败：%s", name, exc)

    # 主模型 = 第一个变体的第一个种子（演示界面、图表都以它为准）
    main_run = runs[0]
    main_tag = main_run["tag"]
    report = main_run["report"]
    main_result = main_run["result"]
    reports = {r["tag"]: r["report"] for r in runs}
    ablation = {k: v for k, v in reports.items() if k != main_tag} or None
    aggregate = aggregate_runs(runs) if len(seeds) > 1 else None
    is_smoke = "smoke" in cfg.run_name.lower()
    # 记录"哪一个是主模型"：演示大屏与报告都据此取对应的预测缓存与指标，避免张冠李戴。
    # 快跑与补充实验（publish_summary=false）不写这个指针，免得把主模型指向别的实验。
    if not is_smoke and getattr(cfg, "publish_summary", True):
        save_json(
            {"tag": main_tag, "variant": main_run["variant"], "seed": main_run["seed"],
             "predictions": f"predictions_{main_tag}.csv",
             "metrics": f"metrics_{main_tag}.json",
             "attention": f"attention_{main_tag}.npy"},
            ARTIFACT_DIR / "main_model.json",
        )
    if aggregate:
        save_json(aggregate, RESULT_DIR / f"aggregate_{cfg.run_name}.json")
        LOGGER.info("多种子聚合结果（同一份数据，只变模型随机种子）：\n%s", _aggregate_table(aggregate))

    # ---------- 3. 出图 ----------
    figures: list[Path] = []
    figure_dir = (
        (RESULT_DIR / "_smoke" / "figures") if is_smoke
        else resolve_dir(getattr(cfg, "figure_dir", ""), FIGURE_DIR)
    )
    if not args.no_figures:
        banner(LOGGER, "步骤 3/5  生成图表")
        pred_csv = ARTIFACT_DIR / f"predictions_{main_tag}.csv"
        history = main_result.history if main_result is not None else None
        attn = ARTIFACT_DIR / f"attention_{main_tag}.npy"
        figures = make_all_figures(
            cfg, data, report, pred_csv=pred_csv,
            attn_path=attn if attn.exists() else None,
            history=history, tag=main_tag, out_dir=figure_dir,
        )
        if len(reports) > 1:
            p = plot_ablation(reports, out_dir=figure_dir)
            if p:
                figures.append(p)

    # ---------- 4. 汇总 ----------
    banner(LOGGER, "步骤 4/5  生成汇总报告")
    train_summary = main_result.summary() if main_result is not None else None
    path = write_summary(cfg, report, train_summary, figures, ablation,
                         out_dir=(RESULT_DIR / "_smoke") if is_smoke else None,
                         aggregate=aggregate, main_tag=main_tag)

    # ---------- 5. 成绩单 ----------
    banner(LOGGER, "步骤 5/5  完成")
    mc, bn, online = report["multiclass"], report["binary"], report.get("online") or {}
    LOGGER.info("多分类：准确率 %.4f | 宏F1 %.4f | Kappa %.4f", mc["accuracy"], mc["macro_f1"], mc["cohen_kappa"])
    LOGGER.info("二分类：F1 %.4f | ROC-AUC %.4f | TPR@1%%FPR %.4f", bn["f1"], bn["roc_auc"], bn.get("tpr_at_fpr_0.01", float("nan")))
    if online:
        LOGGER.info(
            "在线：片段召回 %.2f | 平均时延 %s 秒 | 误报 %s 次/小时",
            online["segment_recall"], online.get("mean_detection_delay_sec"), online.get("false_alarms_per_hour"),
        )
    LOGGER.info("汇总报告：%s", path)
    LOGGER.info("图表目录：%s", FIGURE_DIR)
    LOGGER.info("总耗时：%.1f 秒", time.time() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
