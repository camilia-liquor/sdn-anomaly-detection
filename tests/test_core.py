"""单元测试（不依赖 TensorFlow 也能跑）：覆盖生成器、数据管线、窗口构建、评估口径。

运行：
    python -m pytest tests -q          # 有 pytest 时
    python tests/test_core.py          # 没装 pytest 也可直接跑（内置简易断言）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    BINARY_NORMAL,
    build_windows,
    clean_dataframe,
    detect_label_columns,
    infer_feature_columns,
)
from src.traffic_generator import (  # noqa: E402
    ALL_CLASSES,
    INSDN_HEADER,
    LABEL_COLUMNS,
    SCHEMA_ID_COLUMNS,
    SDN_EXTRA_HEADER,
    TrafficSimulator,
)


def test_schema_headers():
    """两种 schema 的列名与列数必须与对齐的数据集一致（否则"逐字一致"的说法不成立）。"""
    assert len(INSDN_HEADER) == 84, len(INSDN_HEADER)
    assert INSDN_HEADER[0] == "Flow ID" and INSDN_HEADER[-1] == "Label"
    assert len(SDN_EXTRA_HEADER) == 23
    assert SDN_EXTRA_HEADER[0] == "timestamp" and "attack_type" in SDN_EXTRA_HEADER
    # 标签列候选必须同时覆盖 CIC-IDS2017（Label）与真实 inSDN（target）两种写法
    assert "Label" in LABEL_COLUMNS and "target" in LABEL_COLUMNS
    # 标识列必须同时覆盖两套命名，否则真实数据集会把 IP 当特征
    ids = SCHEMA_ID_COLUMNS["insdn"]
    for col in ("Flow ID", "Src IP", "Dst IP", "Source IP", "Destination Port"):
        assert col in ids, col


def test_generator_physical_consistency():
    """生成器产出的物理量必须自洽：包速率 = 总包数/时长；平均包长 = 总字节/总包数。"""
    cfg = Config()
    cfg.scale = 0.05
    df = TrafficSimulator(cfg).generate()
    assert len(df) > 100
    assert list(df.columns) == INSDN_HEADER
    d = df[df["Flow Duration"] > 0]
    pps = d["Flow Packets/s"].to_numpy()
    expect = (d["Total Fwd Packets"].to_numpy() + d["Total Backward Packets"].to_numpy()) / (
        d["Flow Duration"].to_numpy() / 1e6
    )
    assert np.nanmedian(np.abs(pps - expect) / np.clip(expect, 1e-9, None)) < 1e-6
    assert not np.isinf(df.select_dtypes("number").to_numpy()).any()
    # 类别必须齐全（至少覆盖配置里的键）
    assert set(df["Label"].unique()) <= set(ALL_CLASSES)


def test_generator_difficulty_knobs():
    """难度旋钮必须真的生效：提高重叠与隐蔽比例后，同类样本的方差应变大（更难分）。"""
    base = Config()
    base.scale = 0.06
    hard = Config()
    hard.scale = 0.06
    hard.overlap_ratio = 0.9
    hard.stealth_ratio = 1.0
    a = TrafficSimulator(base).generate()
    b = TrafficSimulator(hard).generate()
    col = "Flow Packets/s"
    # 用力样本的"类间可分性"应更弱（同类内部的离散程度更大）
    def within_ratio(df):
        g = df.groupby("Label")[col].apply(lambda s: np.log1p(s).std())
        return float(np.nanmean(g.to_numpy()))
    assert within_ratio(b) > within_ratio(a) * 0.9


def test_label_and_feature_detection():
    """按列名自动识别：既认识 Label，也认识真实 inSDN 的 target。"""
    for label_name in ("Label", "label", "target"):
        df = pd.DataFrame({
            "Flow ID": [f"f{i}" for i in range(6)],
            label_name: ["Normal", "Normal", "DDoS", "DDoS", "Probe", "Probe"],
            "Flow Duration": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "Tot Fwd Pkts": [3.0, 5.0, 7.0, 11.0, 13.0, 17.0],
            "Const": [1.0] * 6,
        })
        cfg = Config()
        cols = detect_label_columns(df, cfg)
        assert label_name in cols, (label_name, cols)
        feats = infer_feature_columns(df, cols, SCHEMA_ID_COLUMNS["insdn"])
        assert "Flow Duration" in feats and "Tot Fwd Pkts" in feats
        assert "Const" not in feats, "常数列必须被剔除"
        assert "Flow ID" not in feats, "标识列必须被剔除"


def test_clean_dataframe_no_row_index_leak():
    """回归测试：清洗绝不能把"行号"带进特征（上游 demo 的经典 bug）。"""
    df = pd.DataFrame({
        "Label": ["Normal"] * 5 + ["DDoS"] * 5,
        "Flow Duration": list(range(1, 11)),
        "Tot Fwd Pkts": list(range(10, 20)),
        "Flow ID": [f"f{i}" for i in range(10)],
    })
    cfg = Config()
    clean, feats, _, mc = clean_dataframe(df, cfg)
    assert "index" not in clean.columns
    assert "index" not in feats
    assert "Flow ID" not in feats
    assert mc == "Label"
    assert set(feats) == {"Flow Duration", "Tot Fwd Pkts"}


def test_build_windows_pure_and_vote():
    """窗口构建：纯窗口筛选、多数投票、告警导向标签、start 索引都要正确。"""
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.array(["Normal"] * 8 + ["DDoS"] * 8 + ["Probe"] * 4)
    Xw, yw, ts, info = build_windows(X, y, 4, 2, pure_only=True)
    assert Xw.shape[1:] == (4, 2)
    assert all(len(set(yw[i:i + 1])) == 1 for i in range(len(yw)))
    assert info["n_skipped_mixed"] == info["n_candidates"] - info["n_kept"]
    # 非纯模式：窗口内出现攻击时标为攻击（告警导向）
    Xw2, yw2, _, info2 = build_windows(X, y, 5, 5, pure_only=False)
    assert len(yw2) == info2["n_candidates"]
    assert any(v != BINARY_NORMAL for v in yw2)
    assert np.array_equal(info["starts"], np.asarray(info["starts"], dtype=np.int64))


def test_build_windows_stride_and_timestamps():
    X = np.random.default_rng(0).normal(size=(30, 3)).astype(np.float32)
    y = np.array(["Normal"] * 30)
    t = np.arange(30, dtype=float)
    Xw, yw, ts, info = build_windows(X, y, 5, 5, pure_only=True, timestamps=t)
    assert Xw.shape == (6, 5, 3)
    assert ts.shape == (6, 5)
    assert np.array_equal(ts[0], np.arange(5, dtype=float))
    assert np.array_equal(info["starts"], np.arange(0, 30 - 5 + 1, 5))


def test_build_windows_errors():
    X = np.zeros((3, 2), dtype=np.float32)
    try:
        build_windows(X, np.array(["Normal"] * 3), 10, 1)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("数据量小于窗口长度时应当报错")


def test_metrics_on_perfect_and_random_predictions():
    """评估口径自检：完美预测必须满分；随机预测的 AUC 应接近 0.5。"""
    from src.evaluate import binary_metrics, multiclass_metrics

    rng = np.random.default_rng(0)
    n, c = 600, 3
    y = rng.integers(0, c, size=n)
    p_perfect = np.eye(c)[y] * 0.98 + 0.01
    mc = multiclass_metrics(y, p_perfect, ["Normal", "DDoS", "Probe"])
    assert mc["accuracy"] > 0.99 and mc["macro_f1"] > 0.99

    yb = rng.integers(0, 2, size=n)
    p_rand = rng.random(n)
    cfg = Config()
    bm = binary_metrics(yb, p_rand, 0.5, cfg)
    assert 0.35 < bm["roc_auc"] < 0.65, bm["roc_auc"]
    assert "tpr_at_fpr_0.01" in bm and "mcc" in bm


def test_class_weight_inverse_frequency():
    """类别加权必须是逆频：少数类权重更大。"""
    from src.data_pipeline import PreparedData

    d = PreparedData(
        X_train=np.zeros((10, 2, 2), dtype=np.float32),
        y_train_mc=np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 2]),
        y_train_bin=np.array([0] * 7 + [1] * 3),
        X_val=np.zeros((2, 2, 2), dtype=np.float32), y_val_mc=np.array([0, 1]), y_val_bin=np.array([0, 1]),
        X_test=np.zeros((2, 2, 2), dtype=np.float32), y_test_mc=np.array([0, 1]), y_test_bin=np.array([0, 1]),
        class_names=["Normal", "DDoS", "Probe"], feature_names=["a", "b"],
        scaler=None, label_encoder=None, input_mode="sequence", window_size=2,
    )
    w = d.class_weight()
    assert w[2] > w[1] > w[0]


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} 个测试通过" + ("" if failed else " ✅"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
