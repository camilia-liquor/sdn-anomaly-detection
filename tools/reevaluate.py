"""重新评估已有模型（改了评估口径/阈值后，不必重新训练）。

典型场景：
    # 改了在线片段的判定口径、或想换个报告阈值，直接重算指标
    python tools/reevaluate.py --tag cnn_bilstm_main --threshold 0.7
    python tools/reevaluate.py --all          # 把所有已训练模型都重算一遍
    python tools/reevaluate.py --tag x --figures   # 顺便重画图

支持的模型来源：
    artifacts/<tag>/model.keras       （CNN+BiLSTM 及其变体）
    artifacts/<tag>/baseline.joblib   （传统机器学习基线）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData  # noqa: E402
from src.evaluate import evaluate_model  # noqa: E402
from src.model import load_model_bundle  # noqa: E402
from src.utils import ARTIFACT_DIR, ensure_dirs, get_logger, set_seed  # noqa: E402

LOGGER = get_logger("cida.reeval")


def find_models() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for d in sorted(ARTIFACT_DIR.iterdir() if ARTIFACT_DIR.exists() else []):
        if not d.is_dir():
            continue
        for name in ("model.keras", "baseline.joblib"):
            p = d / name
            if p.exists():
                out.append((d.name, p))
                break
    return out


def load_any(path: Path):
    if path.suffix == ".joblib":
        return joblib.load(path)
    return load_model_bundle(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="重新评估已训练模型")
    ap.add_argument("--tag", default="", help="artifacts 下的模型目录名")
    ap.add_argument("--all", action="store_true", help="重新评估所有已训练模型")
    ap.add_argument("--threshold", type=float, default=None, help="覆盖告警阈值")
    ap.add_argument("--min-segment", type=int, default=None, help="在线片段最小长度（行）")
    ap.add_argument("--figures", action="store_true", help="顺便重画图表")
    args = ap.parse_args(argv)

    ensure_dirs()
    cfg = Config()
    set_seed(cfg.seed)
    if args.threshold is not None:
        cfg.report_threshold = float(args.threshold)
    if args.min_segment is not None:
        cfg.online_min_segment_rows = int(args.min_segment)

    data = PreparedData.load()
    targets = find_models() if (args.all or not args.tag) else [(args.tag, None)]
    if not targets:
        LOGGER.error("没有找到任何已训练模型，请先运行 python run_all.py")
        return 1

    for tag, path in targets:
        if path is None:
            path = next((p for p in (ARTIFACT_DIR / tag / "model.keras",
                                     ARTIFACT_DIR / tag / "baseline.joblib") if p.exists()), None)
            if path is None:
                LOGGER.warning("跳过 %s：目录下没有模型文件", tag)
                continue
        try:
            model = load_any(path)
            report = evaluate_model(model, data, cfg, tag=tag)
            on = report.get("online") or {}
            LOGGER.info(
                "[%s] 多分类 acc=%.4f macroF1=%.4f | 二分类 F1=%.4f AUC=%.4f | "
                "在线：片段 %d 个，召回 %.3f，误报 %.2f 次/小时，时延 %s 秒",
                tag, report["multiclass"]["accuracy"], report["multiclass"]["macro_f1"],
                report["binary"]["f1"], report["binary"]["roc_auc"],
                on.get("n_attack_segments", 0), on.get("segment_recall", float("nan")),
                on.get("false_alarms_per_hour") or 0.0, on.get("mean_detection_delay_sec"),
            )
            if args.figures:
                from src.visualize import make_all_figures

                history_path = ARTIFACT_DIR / tag / "history.json"
                history = None
                if history_path.exists():
                    import json

                    history = json.loads(history_path.read_text(encoding="utf-8"))
                make_all_figures(
                    cfg, data, report,
                    pred_csv=ARTIFACT_DIR / f"predictions_{tag}.csv",
                    attn_path=ARTIFACT_DIR / f"attention_{tag}.npy",
                    history=history, tag=tag,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("[%s] 重新评估失败：%s", tag, exc)
    LOGGER.info("完成。别忘了重新生成报告： python tools/make_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
