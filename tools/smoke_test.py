"""不依赖 TensorFlow 的自检脚本（环境没问题时也能先验证数据链路）。

它做三件事：
1. 小规模生成仿真数据 → 预处理 → 打印形状/类别/窗口统计；
2. 用一个「假模型」（直接读标签 + 加噪声）跑完整评估流程，
   验证指标计算、在线告警评估、各项 CSV/JSON 落盘是否正确；
3. 生成全部图表，验证字体与绘图代码。

跑法：``python tools/smoke_test.py``（约 20~60 秒）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

from src.config import Config
from src.data_pipeline import PreparedData, prepare_dataset, rolling_windows
from src.evaluate import evaluate_model
from src.utils import RESULT_DIR, ensure_dirs, get_logger, set_seed

LOGGER = get_logger("cida.smoke")

# 自检的所有产物都写在独立目录里，保证**不会污染正式实验的 data/ 与 results/**
SMOKE_RAW = ROOT / "data" / "raw_smoke"
SMOKE_PROCESSED = ROOT / "data" / "processed_smoke"
SMOKE_RESULT = ROOT / "results" / "_smoke"
SMOKE_FIGURE = SMOKE_RESULT / "figures"
SMOKE_ARTIFACT = ROOT / "artifacts" / "_smoke"


class DummyModel:
    """假模型：以真值为中心加噪声产生概率，用来验证下游评估/绘图链路。

    异常越明显（例如 DDoS）预测越准；正常流量噪声更大 —— 模拟真实模型的行为。
    """

    def __init__(self, class_names: list[str], y_hint: np.ndarray | None = None, noise: float = 0.9):
        self.name = "dummy_model"
        self.class_names = list(class_names)
        self.noise = noise
        self.y_hint = y_hint
        self.cida_has_attention = False

    def count_params(self) -> int:
        return 12345

    def predict(self, X, batch_size: int = 512, verbose: int = 0):
        n = len(X)
        rng = np.random.default_rng(0)
        # 用输入特征的均值当"判别信号"，保证同一段流量拿到相近的概率
        strength = np.tanh(np.abs(X.reshape(n, -1)).mean(axis=1))
        p_attack = np.clip(strength + rng.normal(0, self.noise * 0.25, n), 1e-3, 1 - 1e-3)
        p_mc = np.tile((1 - p_attack)[:, None] / max(len(self.class_names) - 1, 1),
                       (1, len(self.class_names)))
        p_mc[:, 0] = 1 - p_attack
        p_mc += rng.random(p_mc.shape) * 0.02
        p_mc /= p_mc.sum(axis=1, keepdims=True)
        return {"mc": p_mc.astype("float32"), "bin": p_attack[:, None].astype("float32")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="无 TF 依赖的自检")
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--keep", action="store_true", help="保留生成的产物（默认也保留，便于查看）")
    args = ap.parse_args(argv)

    ensure_dirs()
    cfg = Config()
    cfg.scale = args.scale
    cfg.run_name = "smoke"
    cfg.epochs = 1
    # ★ 自检使用**独立的数据目录**，绝不覆盖正式实验的 data/raw、data/processed
    #   （否则跑一次自检就会把正式结果的数据集换成小规模版本，演示和复评都会跟着错）
    cfg.raw_dir = "data/raw_smoke"
    cfg.processed_dir = "data/processed_smoke"
    set_seed(cfg.seed)
    t0 = time.time()

    LOGGER.info("=== 1/4 生成数据 + 预处理 ===")
    data: PreparedData = prepare_dataset(cfg, force_regenerate=True, save=True,
                                         out_dir=ROOT / "data" / "processed_smoke")
    LOGGER.info("输入形状 %s，类别 %s", data.input_shape, data.class_names)
    LOGGER.info("训练/验证/测试窗口数：%d / %d / %d", len(data.X_train), len(data.X_val), len(data.X_test))
    assert data.X_train.ndim == 3, "序列模式应为 (N, T, F)"
    assert not np.isnan(data.X_train).any(), "训练张量存在 NaN"
    assert not np.isinf(data.X_train).any(), "训练张量存在 inf"
    assert len(set(data.class_names)) == data.n_classes

    LOGGER.info("=== 2/4 检查滑窗与在线取窗 ===")
    X_roll, starts = rolling_windows(data.test_stream_features, cfg.window_size, cfg.online_stride)
    LOGGER.info("在线窗口 %d 个，起始行 %s ...", len(X_roll), starts[:5])
    assert X_roll.shape[1:] == (cfg.window_size, data.n_features)
    assert (starts + cfg.window_size <= len(data.test_stream_features)).all()

    LOGGER.info("=== 3/4 用假模型跑完整评估 ===")
    dummy = DummyModel(data.class_names)
    report = evaluate_model(dummy, data, cfg, tag="smoke",
                            result_dir=SMOKE_RESULT, artifact_dir=SMOKE_ARTIFACT)
    for key in ("multiclass", "binary", "online"):
        assert key in report, f"评估报告缺少 {key}"
    assert 0.0 <= report["multiclass"]["accuracy"] <= 1.0
    assert -1.0 <= report["binary"]["mcc"] <= 1.0
    LOGGER.info("多分类 acc=%.4f，二分类 F1=%.4f，在线误报率=%s 次/小时",
                report["multiclass"]["accuracy"], report["binary"]["f1"],
                report["online"].get("false_alarms_per_hour"))

    LOGGER.info("=== 4/4 生成图表 ===")
    from src.visualize import make_all_figures

    pred_csv = SMOKE_ARTIFACT / "predictions_smoke.csv"
    figs = make_all_figures(cfg, data, report, pred_csv=pred_csv, attn_path=None,
                            history=None, tag="smoke", out_dir=SMOKE_FIGURE)
    for p in figs:
        LOGGER.info("  ✓ %s", p.name)
    assert figs, "没有生成任何图表"

    LOGGER.info("自检通过 ✅  总耗时 %.1f 秒", time.time() - t0)
    LOGGER.info("（自检产物都在 %s 与 %s 下，正式实验结果未受影响）",
                SMOKE_RESULT.relative_to(ROOT), SMOKE_RAW.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
