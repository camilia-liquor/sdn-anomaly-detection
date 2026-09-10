"""晚融合（概率集成）评估：把多个已训练模型的预测概率平均后再判定。

为什么做这个
------------
消融实验会显示：单独 BiLSTM 分支与双分支混合模型的指标非常接近
（差异在随机种子波动量级内）。这引出一个很自然会追问的问题：
**那 CNN 分支到底有没有用？**

答案是"有，但要换个融合方式看"：两个分支在不同样本上犯错的位置并不完全重叠，
把它们的**预测概率平均**（晚融合）之后，宏 F1 / AUC / 误报率通常都能超过任一单分支。
本脚本用已经训练好的模型直接算，**不需要重新训练**。

用法：
    python tools/ensemble_eval.py                      # 自动挑同名不同变体的模型
    python tools/ensemble_eval.py --tags a b c         # 指定要集成的 tag
    python tools/ensemble_eval.py --group seed         # 同一变体的多种子集成
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData, build_windows  # noqa: E402
from src.evaluate import (  # noqa: E402
    _save_per_class,
    binary_metrics,
    multiclass_metrics,
    online_alarm_evaluation,
)
from src.model import load_model_bundle, predict_probs  # noqa: E402
from src.utils import ARTIFACT_DIR, RESULT_DIR, ensure_dirs, get_logger, save_json, set_seed  # noqa: E402

LOGGER = get_logger("cida.ensemble")


class EnsembleModel:
    """把多个模型包成一个"取平均概率"的模型，接口与单模型一致。"""

    def __init__(self, models: list, names: list[str]):
        self.models = models
        self.names = names
        self.name = "ensemble(" + "+".join(names) + ")"
        self.cida_has_attention = False

    def predict(self, X, batch_size: int = 512, verbose: int = 0) -> dict:
        p_mc, p_bin = [], []
        for m in self.models:
            a, b = predict_probs(m, X, batch_size=batch_size)
            p_mc.append(a)
            p_bin.append(b)
        mc = np.mean(p_mc, axis=0)
        bn = np.mean(p_bin, axis=0)
        mc = mc / np.clip(mc.sum(axis=1, keepdims=True), 1e-9, None)
        return {"mc": mc.astype("float32"), "bin": np.clip(bn, 1e-7, 1 - 1e-7)[:, None].astype("float32")}

    def count_params(self) -> int:
        total = 0
        for m in self.models:
            try:
                total += int(m.count_params())
            except Exception:
                total += 0
        return total


def discover_models(explicit: list[str] | None, group: str) -> list[tuple[str, Path]]:
    if explicit:
        out = []
        for t in explicit:
            p = ARTIFACT_DIR / t
            for name in ("model.keras", "baseline.joblib"):
                if (p / name).exists():
                    out.append((t, p / name))
                    break
        return out
    # 自动发现：同一批实验下的不同变体（tag 里含 _cnn / _bilstm / _mlp）
    found = []
    for d in sorted(ARTIFACT_DIR.iterdir() if ARTIFACT_DIR.exists() else []):
        if not d.is_dir():
            continue
        variants = [v for v in ("_cnn", "_bilstm", "_mlp") if d.name.endswith(v + "_s0") or d.name.endswith(v)]
        if variants and (d / "model.keras").exists():
            found.append((d.name, d / "model.keras"))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="晚融合（概率集成）评估")
    ap.add_argument("--tags", nargs="*", default=None, help="要集成的模型 tag 列表")
    ap.add_argument("--group", default="variant", choices=["variant", "seed", "all"],
                    help="集成分组方式（默认把不同变体集成到一起）")
    ap.add_argument("--out-tag", default="ensemble", help="结果的 tag 名")
    args = ap.parse_args(argv)

    ensure_dirs()
    cfg = Config()
    set_seed(cfg.seed)
    data = PreparedData.load()

    targets = discover_models(args.tags, args.group)
    if len(targets) < 2:
        LOGGER.error("至少需要 2 个模型才能集成；当前找到 %d 个（先跑 run_all.py --ablation）", len(targets))
        return 1

    models, names = [], []
    for tag, path in targets:
        try:
            models.append(joblib.load(path) if path.suffix == ".joblib" else load_model_bundle(path))
            names.append(tag)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("跳过 %s：%s", tag, exc)
    LOGGER.info("集成 %d 个模型：%s", len(models), ", ".join(names))

    ens = EnsembleModel(models, names)
    p_mc, p_bin = predict_probs(ens, data.X_test, batch_size=cfg.batch_size)
    mc = multiclass_metrics(data.y_test_mc, p_mc, data.class_names)
    bn = binary_metrics(data.y_test_bin, p_bin, cfg.report_threshold, cfg)
    LOGGER.info("[集成] 多分类 acc=%.4f macroF1=%.4f | 二分类 F1=%.4f AUC=%.4f",
                mc["accuracy"], mc["macro_f1"], bn["f1"], bn["roc_auc"])

    online = {}
    if data.test_stream_features is not None and data.input_mode == "sequence":
        X_stream, _, _, info = build_windows(
            data.test_stream_features, labels=data.test_stream_labels,
            window_size=cfg.window_size, stride=cfg.online_stride, pure_only=False,
        )
        _, p_bin_s = predict_probs(ens, X_stream, batch_size=cfg.batch_size)
        online = online_alarm_evaluation(data, p_bin_s, np.asarray(info["starts"]), cfg)
        LOGGER.info("[集成] 在线：片段召回 %.3f，误报 %.2f 次/小时，时延 %s 秒",
                    online["segment_recall"], online["false_alarms_per_hour"],
                    online["mean_detection_delay_sec"])

    report = {
        "tag": args.out_tag,
        "model": {"variant": ens.name, "n_params": ens.count_params(),
                  "input_shape": list(data.input_shape), "members": names},
        "data": data.summary(),
        "multiclass": mc,
        "binary": bn,
        "online": online,
        "ensemble_members": names,
    }
    save_json(report, RESULT_DIR / f"metrics_{args.out_tag}.json")
    _save_per_class(mc, data.class_names, RESULT_DIR / f"per_class_{args.out_tag}.csv")
    np.savetxt(RESULT_DIR / f"confusion_{args.out_tag}.csv",
               np.asarray(mc["confusion_matrix"]), fmt="%d", delimiter=",")
    print(f"集成结果已保存 → results/metrics_{args.out_tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
