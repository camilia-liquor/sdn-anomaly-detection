"""推理命令行工具：对新的流记录 CSV 做检测（演示"训练完怎么上线用"）。

用法
----
    # 1) 对某个 CSV 批量检测（逐条流 → 滚动滑窗 → 输出每条流的异常概率）
    python tools/predict_cli.py --input data/raw/insdn_stream.csv --limit 2000

    # 2) 只看汇总
    python tools/predict_cli.py --input new_traffic.csv --quiet

    # 3) 指定模型与产物目录
    python tools/predict_cli.py --input x.csv --model artifacts/cnn_bilstm/model.keras

输出：``results/predictions_cli.csv``，含时间戳、真实标签（若有）、预测类别、异常概率、是否告警。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data_pipeline import PreparedData, clean_dataframe, order_by_time
from src.model import load_model_bundle, predict_probs
from src.utils import ARTIFACT_DIR, RESULT_DIR, ensure_dirs, get_logger

LOGGER = get_logger("cida.predict")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对新流量 CSV 做异常检测")
    ap.add_argument("--input", required=True, help="待检测的 CSV 路径")
    ap.add_argument("--processed", default="", help="预处理产物目录（默认 data/processed）")
    ap.add_argument("--model", default="", help="模型路径（默认自动找 artifacts 下最新的 .keras）")
    ap.add_argument("--threshold", type=float, default=None, help="告警阈值（默认用配置值）")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条（调试用）")
    ap.add_argument("--stride", type=int, default=0, help="滑窗步长（默认用配置值）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ensure_dirs()
    cfg = Config()
    data = PreparedData.load(Path(args.processed) if args.processed else None)
    model_path = Path(args.model) if args.model else _latest_model()
    if model_path is None or not model_path.exists():
        LOGGER.error("找不到模型文件，请先运行 python run_all.py 或显式指定 --model")
        return 1
    model = load_model_bundle(model_path)
    threshold = float(args.threshold if args.threshold is not None else cfg.report_threshold)
    stride = int(args.stride or cfg.online_stride)

    df = pd.read_csv(args.input, low_memory=False)
    if args.limit:
        df = df.head(int(args.limit))
    df, _, _, mc_col = clean_dataframe(df, cfg)
    df = order_by_time(df, cfg)

    missing = [f for f in data.feature_names if f not in df.columns]
    if missing:
        LOGGER.error("输入数据缺少 %d 个训练时用到的特征列，例如：%s", len(missing), missing[:5])
        return 2
    X = df[data.feature_names].to_numpy(dtype=np.float64)
    X = data.scaler.transform(X).astype(np.float32)

    T, s = int(data.window_size), stride
    n = len(X)
    if n < T:
        LOGGER.error("输入只有 %d 条记录，少于窗口长度 %d", n, T)
        return 2
    starts = np.arange(0, n - T + 1, s)
    windows = np.stack([X[i : i + T] for i in starts]).astype(np.float32)
    p_mc, p_bin = predict_probs(model, windows, batch_size=cfg.batch_size)
    pred_cls = [data.class_names[i] for i in np.argmax(p_mc, axis=1)]

    # 把窗口结论广播到窗口覆盖的每条流（线上就是这么用的：窗口满即出结论）
    rows = []
    truth = df[mc_col].astype(str).to_numpy() if mc_col in df.columns else np.array(["?"] * n)
    for k, st in enumerate(starts):
        for i in range(int(st), int(st) + T):
            rows.append({
                "row": i,
                "pred_class": pred_cls[k],
                "p_attack": float(p_bin[k]),
                "alarm": bool(p_bin[k] >= threshold),
                "true_class": truth[i],
                **{f"p_{c}": float(p_mc[k, j]) for j, c in enumerate(data.class_names)},
            })
    out = pd.DataFrame(rows).drop_duplicates(subset=["row"], keep="last").sort_values("row")
    out_path = RESULT_DIR / "predictions_cli.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    n_alarm = int(out["alarm"].sum())
    if not args.quiet:
        LOGGER.info("检测完成：%d 条流记录 → %d 个窗口，告警 %d 个（阈值 %.2f）",
                    n, len(starts), n_alarm, threshold)
        if (truth != "?").any():
            acc = float((out["pred_class"].to_numpy() == out["true_class"].to_numpy()).mean())
            LOGGER.info("与真实标签逐条对比准确率：%.4f", acc)
        LOGGER.info("结果已保存 → %s", out_path)
    else:
        print(f"rows={n} windows={len(starts)} alarms={n_alarm} out={out_path}")
    return 0


def _latest_model() -> Path | None:
    cands = sorted(ARTIFACT_DIR.glob("*/model.keras"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


if __name__ == "__main__":
    raise SystemExit(main())
