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


# ---------------------------------------------------------------------------
# 对抗训练增广（可选能力，默认关）
# ---------------------------------------------------------------------------
def _tiny_dict_model():
    """造一个与主模型**输出结构一致**（dict：mc + bin）的极小模型，用于验证对抗增广函数。

    为什么不直接 `build_model`：本测试要验证的是增广函数（形状 / L∞ 边界 / 标签配对），
    用真模型会把单元测试从秒级拖到分钟级；而"多分类头 + 二分类头"的结构一致性才是关键。
    """
    import tensorflow as tf

    inp = tf.keras.Input(shape=(4, 3))
    h = tf.keras.layers.Flatten()(inp)
    h = tf.keras.layers.Dense(8, activation="relu")(h)
    return tf.keras.Model(
        inputs=inp,
        outputs={"mc": tf.keras.layers.Dense(2, activation="softmax", name="mc")(h),
                 "bin": tf.keras.layers.Dense(1, activation="sigmoid", name="bin")(h)},
    )


def test_adversarial_augment_default_off_and_bounded():
    """对抗训练增广：默认关（原样返回，主流程与产物不动）；开启时只做 L∞ 有界扰动且标签逐行配对。

    守的是三件事：① 默认配置下这个可选能力**不产生任何影响**（既有产物 sha256 不被破）；
    ② 开启后扰动必须同时满足 L∞ 预算与训练集物理可行域；③ 标签不能被拼错位
    ——把攻击样本配上正常标签，比不做对抗训练更糟。
    """
    from src.train import ADV_OBJECTIVES, adversarial_augment, fgsm_perturb

    model = _tiny_dict_model()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4, 3)).astype(np.float32)
    y_mc = np.array([0, 1] * 6, dtype=np.int64)
    y_bin = (y_mc != 0).astype(np.float32)

    # ① 默认（eps=0）原样返回：不复制样本、不改写数据
    X0 = X.copy()
    Xa, ma, ba, info = adversarial_augment(model, X, y_mc, y_bin, eps=0.0)
    assert info["augmented"] is False and info["n_adv"] == 0, info
    assert np.array_equal(Xa, X) and np.array_equal(ma, y_mc) and np.array_equal(ba, y_bin)
    assert len(Xa) == len(X0) and np.array_equal(X, X0), "默认关时不得改动输入数据"

    # ② 开启：样本数 = 干净 + 对抗，形状/类型不变，干净部分原样保留
    eps, seed = 0.3, 3
    Xa, ma, ba, info = adversarial_augment(model, X, y_mc, y_bin, eps=eps, ratio=0.5, seed=seed)
    assert info["augmented"] is True and info["n_adv"] == len(X), info
    assert Xa.shape == (2 * len(X), 4, 3) and Xa.dtype == np.float32
    assert ma.shape == (len(Xa),) and ba.shape == (len(Xa),)
    assert np.array_equal(Xa[:len(X)], X), "干净样本必须原样保留（对抗副本只能追加在后面）"
    assert np.array_equal(ma[:len(X)], y_mc) and np.array_equal(ba[:len(X)], y_bin)

    # ③ 扰动必须同时满足：L∞ 预算 eps、训练集逐特征物理可行域 [lo, hi]
    adv = Xa[len(X):]
    assert np.all(np.abs(adv - X) <= eps + 1e-6), "扰动必须落在 L∞ 球面内"
    lo = X.reshape(-1, X.shape[-1]).min(axis=0)
    hi = X.reshape(-1, X.shape[-1]).max(axis=0)
    assert np.all(adv >= lo - 1e-6) and np.all(adv <= hi + 1e-6), "扰动后必须仍在训练集特征可行域内"
    assert float(np.mean(np.abs(adv - X) > 0)) > 0.5, "FGSM 必须真的产生扰动，不能退化成恒等映射"

    # ④ 标签逐行配对不被打乱（本工程对"标签错位"零容忍）
    assert np.array_equal(ba, (ma != 0).astype(np.float32)), "增广后二分类标签必须仍与多分类标签一致"

    # ⑤ 定向目标真的生效：对抗副本被判得更像 Normal（不是"加了点噪声"）
    def _normal_loss(arr):
        p_normal = model.predict(arr, verbose=0)["mc"][:, 0]
        return float(-np.log(p_normal + 1e-12).mean())

    assert _normal_loss(adv) < _normal_loss(X), "FGSM(target_normal) 必须让对抗样本更像正常流量"

    # ⑥ 非法目标必须显式报错，而不是悄悄按默认目标跑（否则报告口径会对不上）
    try:
        fgsm_perturb(model, X, eps=eps, objective="whatever")
    except ValueError:
        pass
    else:
        raise AssertionError(f"非法对抗目标必须报错，可选目标为 {ADV_OBJECTIVES}")


# ---------------------------------------------------------------------------
# 多源证据融合（控制面流表 + 数据面 pcap 双路）
# ---------------------------------------------------------------------------
def test_multisource_fusion_policy():
    """双源融合：一路静默否掉另一路是**不允许**的；单路/无路必须如实标注而不是编造。

    这一项守的是融合层最容易犯的错：把"两个视角的矛盾"用平均/投票悄悄抹平
    （看起来指标很好看，实际上单点风险被藏进了融合层）。
    """
    from src.fusion import (SOURCE_CONTROL, SOURCE_DATA, SourceEvidence, align_by_key,
                            flow_key, fuse_evidence, fusion_metrics)

    # ① 两种 schema 的字段名必须归一到同一个键（否则两路永远配不上对）
    ctrl = {"Src IP": "10.0.0.9", "Dst IP": "10.0.0.1", "tp_src": 1234, "tp_dst": 80, "ip_proto": 6}
    data = {"Source IP": "10.0.0.9", "Destination IP": "10.0.0.1", "Source Port": 1234,
            "Destination Port": 80, "Protocol": 6}
    assert flow_key(ctrl) == flow_key(data), (flow_key(ctrl), flow_key(data))
    # 磁盘上的 sdn 流表口径用的是 ip_src/ip_dst（不是 Src IP）——别名表漏一个，
    # 那一侧就会"全取不到键"→ 对齐变成零匹配，而且**不会报错**（本工程踩过一次）
    assert flow_key({"ip_src": "10.0.0.9", "ip_dst": "10.0.0.1", "tp_src": 1234,
                     "tp_dst": 80, "ip_proto": 6}) == flow_key(ctrl)
    assert flow_key({"Src IP": "10.0.0.9"}) is None, "缺目的 IP 时不得猜一个键"

    # ② 对齐：只在一路出现的流必须被单独列出（它们正是融合要补的那部分）
    alone = {"Src IP": "10.0.0.77", "Dst IP": "10.0.0.1", "tp_src": 5, "tp_dst": 443, "ip_proto": 6}
    al = align_by_key([ctrl, alone], [data])
    assert len(al["matched"]) == 1 and len(al["control_only"]) == 1 and not al["data_only"], al
    assert al["unkeyed"] == {"control": 0, "data": 0}

    def fuse(c=None, d=None, **kw):
        ev_c = SourceEvidence(SOURCE_CONTROL, p_attack=c, available=c is not None)
        ev_d = SourceEvidence(SOURCE_DATA, p_attack=d, available=d is not None)
        return fuse_evidence(ev_c, ev_d, threshold=0.5, **kw)

    # ③ 两路一致：不冲突、不需要额外人工研判
    v = fuse(0.9, 0.8)
    assert v.alarm and not v.conflict and not v.review_required and not v.degraded, v.to_dict()

    # ④ 两路对立：取更保守（更高风险）的一侧，且**必须**标记冲突并交人看
    v = fuse(0.9, 0.1)
    assert v.p_attack == 0.9 and v.alarm is True, v.to_dict()
    assert v.conflict is True and v.review_required is True, v.to_dict()
    assert "不允许" in v.reason or "对立" in v.reason, v.reason

    # ⑤ 阈值附近的分差算"两路都不确定"，不算冲突（否则冲突标记会被噪声淹没）
    v = fuse(0.51, 0.45)
    assert v.conflict is False, v.to_dict()

    # ⑥ 只有一路可用：用它的分数，但必须标 degraded（如实说明这次只有一只眼睛）
    v = fuse(0.95, None)
    assert v.p_attack == 0.95 and v.degraded is True and v.review_required is True, v.to_dict()
    v = fuse(None, 0.2)
    assert v.p_attack == 0.2 and v.alarm is False and v.degraded is True, v.to_dict()

    # ⑦ 两路都没有：不给分数（**不编造**），也不假装报过警
    v = fuse(None, None)
    assert v.p_attack is None and v.alarm is False and v.degraded is True, v.to_dict()

    # ⑧ 互补增益必须能单独数出来，且"融合反而漏掉某一路抓到的攻击"必须为 0
    def rec(c, d):
        v = fuse(c, d)
        return {"verdict": v.to_dict()}

    records = [
        rec(0.9, 0.9),      # 两路都抓到
        rec(0.1, 0.8),      # 控制面漏、数据面抓到 → 互补覆盖
        rec(0.05, None),    # 只有控制面、且漏（数据面没覆盖到）→ degraded
        rec(0.1, 0.1),      # 两路都判正常（真·漏报）
        rec(0.9, 0.1),      # 冲突样本（一路报警、一路判正常）
        rec(0.30, 0.32),    # 两路都差一点但一致 → 融合独有的一种报警（弱证据一致）
        rec(0.30, 0.05),    # 只有一路弱 → 不报
    ]
    y = [1, 1, 1, 1, 0, 1, 1]
    m = fusion_metrics(records, y, threshold=0.5)
    assert m["recall_control"] == round(1 / 6, 4) and m["recall_data"] == round(2 / 6, 4), m
    assert m["recall_fused"] == round(3 / 6, 4), m          # 融合把第 6 条（两路都差一点）也抓到了
    assert m["rescued_by_pair"] == 1, m                     # 控制面漏的那条被数据面保住
    assert m["recovered_by_fusion"] == 1, m                 # 两路单看都漏、融合抓到（弱证据一致）
    assert m["weak_agreement"] == 1, m
    assert m["lost_by_fusion"] == 0, "融合不得丢掉任何一路抓到的攻击"
    assert m["conflicts"] == 2 and m["degraded"] == 1, m
    assert m["fused_fp"] >= 1, m                            # 冲突样本被保守判成攻击（代价如实计入）
    assert m["rule"] and m["threshold"] == 0.5



def _review_service(tmpdir):
    """构造可确定复现的检测服务，并把落盘重定向到临时目录。

    为什么必须重定向 `audit_path` / `feedback_path`：这两个文件在正式运行时写进 `results/`，
    测试若照写会把测试数据混进真实审计台账与回流文件里。
    """
    from src.service import DetectionService

    svc = DetectionService()
    if not svc.ready:
        raise AssertionError(f"检测服务未就绪，无法验证复核闭环：{svc.reason}")
    svc.audit_path = tmpdir / "audit_test.jsonl"
    svc.feedback_path = tmpdir / "review_feedback.jsonl"
    svc.review_evidence_path = tmpdir / "review_evidence.jsonl"
    return svc


def _patch_infer(svc, p_attack: float, uniform: bool) -> None:
    """替换推理输出：uniform=True → 均匀分布（高熵 → 判未知形态）；False → 低熵（不判未知）。"""
    n_cls = max(len(svc.class_names), 2)

    def _fake(raw):
        n = raw.shape[0]
        if uniform:
            p_mc = np.full((n, n_cls), 1.0 / n_cls)
        else:
            p_mc = np.full((n, n_cls), 0.01)
            p_mc[:, 0] = 1.0 - 0.01 * (n_cls - 1)
        return p_mc, np.full(n, float(p_attack))

    svc._infer = _fake


def _push_until_windows(svc, target: int, src_ip: str = "10.0.0.9", extra: dict | None = None) -> list:
    """再产生 target 个窗口后返回这批告警（**相对**当前窗口数，便于在同一服务上分阶段测试）。

    窗口由 `online_stride` 决定，不能"推一次就假设出了窗口"——那会让测试在不同配置下静默失效。
    也不要用绝对窗口数：同一服务二次调用时会因为窗口数已达标而空转。
    """
    row = {f: 0.0 for f in svc.feature_names}
    row["Src IP"] = src_ip
    if extra:
        row.update(extra)
    start = int(svc.stats["windows"])
    want = start + max(1, int(target))
    alerts: list = []
    for _ in range(40):
        if svc.stats["windows"] >= want:
            break
        res = svc.push([dict(row) for _ in range(int(svc.cfg.window_size))], {"src_ip": src_ip})
        assert res.get("ok"), res
        alerts.extend(res.get("alerts", []))
    assert svc.stats["windows"] >= want, f"未能再产生 {target} 个窗口（当前 {svc.stats['windows']}）"
    return alerts


def test_unknown_action_guardrail_never_blocks():
    """未知形态的动作护栏：即便全局策略是 block，也绝不直接封禁未知流量。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        svc = _review_service(Path(td))
        svc.reload_policy(action="block", unknown_action="block")
        assert svc._decide_action("10.0.0.1", unknown=True) == "throttle", "未知分支不得升级为 block"
        assert svc._decide_action("10.0.0.1", unknown=False) == "block", "已知攻击仍按全局策略封禁"
        svc.reload_policy(action="observe")
        assert svc._decide_action("10.0.0.1", unknown=True) == "observe"
        svc.reload_policy(action="block", unknown_action="throttle", whitelist=["10.0.0.7"])
        assert svc._decide_action("10.0.0.7", unknown=True) == "observe", "白名单永远优先"
        # 非法动作必须被拒绝（否则策略快照里会出现 ryu_guard 不认识的动作）
        svc.reload_policy(unknown_action="drop-everything")
        assert svc.policy.unknown_action == "throttle"


def test_observation_pool_and_replay_buffer():
    """未被确认的窗口必须留痕：观察池记证据，回看缓冲记全过程。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        svc = _review_service(Path(td))
        # 确认窗口数放到很大 → 任何窗口都不会被确认，从而与 online_stride 无关地确定复现
        svc.reload_policy(threshold=0.5, confirm_windows=99, unknown_threshold=0.0, action="observe")
        _patch_infer(svc, p_attack=0.9, uniform=False)
        _push_until_windows(svc, 2)
        n_win = int(svc.stats["windows"])
        assert n_win >= 2
        assert svc.stats["alarms_confirmed"] == 0, "确认窗口数不足，不应产生已确认告警"
        assert len(svc.observations) == n_win, "每个未确认窗口都应进观察池留痕"
        ob = svc.list_observations(1)[0]
        assert ob["why"] == "near_threshold", ob
        assert ob["p_attack"] >= 0.5 * 0.5, "观察池下界应为 observe_margin × threshold"
        # 回看缓冲：每个窗口都留痕，才能回答"这段为什么没报警"
        assert len(svc.recent) == n_win
        last = list(svc.recent)[-1]
        assert last["alarm"] is True and last["confirmed"] is False, last
        # 容量必须有界，长期运行不能吃内存
        assert svc.observations.maxlen == int(svc.policy.observe_pool_size)
        # 观察池条目可被提升进待研判队列
        promoted = svc.promote_observation()
        assert promoted["ok"], promoted
        assert svc.list_review(status="pending")[-1]["kind"] == "observation"
        assert not svc.promote_observation(index=99999)["ok"], "越界下标必须报错而不是静默成功"


def test_unknown_alert_enters_review_and_label_flows_back():
    """未知形态：进待研判队列 → 人工给结论 → 结论回流落盘（「未知 → 已知」闭环的落地点）。"""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        svc = _review_service(Path(td))
        svc.reload_policy(threshold=0.5, confirm_windows=1, unknown_threshold=1.0,
                          action="block", dry_run=True)
        _patch_infer(svc, p_attack=0.9, uniform=True)
        alerts = _push_until_windows(svc, 1)
        assert alerts, "应产生一条告警"
        a = alerts[0]
        assert a["unknown"] is True and str(a["pred_class"]).startswith("Unknown"), a
        assert a["action"] == "throttle", f"未知形态最多限速，实际 {a['action']}"
        assert isinstance(a.get("review_id"), int), "未知告警必须进待研判队列"
        assert any(x["review_id"] == a["review_id"] and x["why"] == "unknown_shape"
                   for x in svc.list_review(status="pending"))
        # 人工研判 → 回流落盘
        res = svc.label_review(a["review_id"], "attack:Botnet", note="C2 心跳特征")
        assert res["ok"], res
        assert svc.feedback_path.exists(), "标注结论必须回流落盘"
        rec = json.loads(svc.feedback_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert rec["verdict"] == "attack:Botnet" and rec["review_id"] == a["review_id"], rec
        assert svc.stats["review_confirmed_attack"] == 1
        assert svc.list_review(status="labeled"), "已研判项状态应变为 labeled"
        # 非法输入必须被拒绝，而不是静默写入脏数据
        assert not svc.label_review(a["review_id"], "maybe")["ok"]
        assert not svc.label_review(999999, "normal")["ok"]

        # review_id 必须在**服务重启后仍然唯一**：它是 feedback 与 evidence 之间唯一的关联键。
        # 若每次启动都从 1 重来，旧结论会与新的窗口特征按同一个 id 配对，
        # 拼出张冠李戴甚至自相矛盾的训练样本（同一窗口既有 attack 又有 normal）。
        svc2 = _review_service(Path(td))            # 复用同一份落盘文件 = 模拟重启
        recovered = int(svc2._recover_review_seq())  # 恢复是惰性的，直接验证恢复结果
        assert recovered >= int(a["review_id"]), \
            f"重启后必须能从历史记录恢复 review_id（恢复得到 {recovered}，历史最大 {a['review_id']}）"
        # 重启后的实例是默认策略（unknown_threshold=0=关闭），要显式开启才会产生待研判项
        svc2.reload_policy(threshold=0.5, confirm_windows=1, unknown_threshold=1.0,
                           action="observe", dry_run=True)
        _patch_infer(svc2, p_attack=0.9, uniform=True)
        fresh = _push_until_windows(svc2, 1, src_ip="10.0.0.11")
        assert fresh and fresh[0]["review_id"] > int(a["review_id"]), \
            f"新告警必须拿到更大的 review_id：{fresh[0].get('review_id') if fresh else None}"

        # ---- 关键：结论必须能用于重训（否则"回流"只是话术）----
        # 光有结论没有特征，人工标完也训不了模型。证据与结论按 review_id 关联。
        assert svc.review_evidence_path.exists(), "必须落盘窗口特征证据"
        ev = [json.loads(x) for x in
              svc.review_evidence_path.read_text(encoding="utf-8").strip().splitlines()]
        mine = [e for e in ev if e["review_id"] == a["review_id"]]
        assert mine, "该告警应有对应的证据记录"
        e0 = mine[-1]
        assert e0["feature_names"] == svc.feature_names, "证据文件必须自描述（带特征名与列序）"
        assert e0["n_features"] == len(svc.feature_names)
        assert len(e0["X"]) == int(svc.cfg.window_size) * len(svc.feature_names), "向量长度应为 T×F"
        # 流级证据也要随盘落下来：否则复核时看不到"当时是哪些流"，
        # 而且下游就无法核验/忽略回放数据自带的真实标签（会退化成无根据的保证）
        assert e0.get("flow_evidence"), "证据记录必须随附流级定位字段"
        # 证据 + 结论可以拼成一条带标签的训练样本（这就是重训的输入）
        fb = json.loads(svc.feedback_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        joined = {**e0, "verdict": fb["verdict"]}
        assert joined["verdict"] == "attack:Botnet" and len(joined["X"]) > 0


def test_bounded_alerts_and_jsonl_rotation():
    """长期运行的资源边界：告警在内存里有上界，审计/回流文件写满即轮转并丢弃最旧。"""
    import json as _json
    import tempfile
    from collections import deque as _deque

    from src.utils import append_jsonl

    # 1) JSONL 轮转：历史文件数不超过 backups，且当前文件始终可解析（不能写出半行）
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "audit.jsonl"
        for i in range(60):
            append_jsonl(p, {"i": i, "pad": "x" * 40}, max_bytes=200, backups=2)
        files = sorted(x.name for x in Path(td).glob("audit.jsonl*"))
        assert p.exists(), "轮转后当前文件必须仍然存在"
        assert len(files) <= 3, f"最多 1 个当前文件 + 2 个备份，实际 {files}"
        for line in p.read_text(encoding="utf-8").strip().splitlines():
            _json.loads(line)
        kept = "".join((Path(td) / f).read_text(encoding="utf-8") for f in files)
        assert '"i": 0,' not in kept, "最旧的记录应已被轮转丢弃（保留的是最近的）"
        assert '"i": 59' in kept or '"i": 58' in kept, "最新记录必须保留"

    # 2) 内存告警有上界，但累计计数器与保留条数解耦
    with tempfile.TemporaryDirectory() as td:
        svc = _review_service(Path(td))
        assert svc.alerts.maxlen == int(svc.policy.alerts_retained), "告警列表必须是有界 deque"
        svc.reload_policy(threshold=0.5, confirm_windows=1, action="observe")
        _patch_infer(svc, p_attack=0.9, uniform=False)
        _push_until_windows(svc, 1)
        assert svc.stats["alerts_total"] >= 1
        svc.alerts = _deque(svc.alerts, maxlen=1)     # 人为把保留上限压到 1 条
        total_before = int(svc.stats["alerts_total"])
        _push_until_windows(svc, 2)
        assert len(svc.alerts) == 1, "保留条数必须受 maxlen 约束"
        assert int(svc.stats["alerts_total"]) > total_before, "累计计数不受保留上限影响"


def test_alarm_escalation_state_machine():
    """同源反复告警自动升级；但安全设置（白名单 / 全局 observe）永远压过自动升级。

    注意基线必须是 throttle 而不是 observe：observe 是运维明确的"只观察、不处置"，
    自动升级**不允许**把它变成限速或阻断（这正是本测试第 3 节在守的东西）。
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        svc = _review_service(Path(td))
        svc.reload_policy(threshold=0.5, confirm_windows=1, unknown_threshold=0.0,
                          action="throttle", escalate_after=3, escalate_window_sec=300)
        _patch_infer(svc, p_attack=0.9, uniform=False)
        alerts = _push_until_windows(svc, 3)
        acts = [a["action"] for a in alerts]
        assert len(acts) >= 3, acts
        assert acts[0] == "throttle" and acts[1] == "throttle", f"前两次不应升级：{acts}"
        assert acts[2] == "block", f"第 3 次应升级为阻断：{acts}"
        assert alerts[2]["escalated"] is True
        assert int(svc.stats["escalations"]) >= 1

        # 未知形态：即便反复告警也只能到 throttle，绝不升到 block
        svc.reload_policy(threshold=0.5, confirm_windows=1, unknown_threshold=1.0,
                          action="block", unknown_action="throttle",
                          escalate_after=2, escalate_window_sec=300)
        svc.reset_stream("测试换段")
        assert svc._escalation == {}, "流重置后升级计数必须归零，否则换段后会被误升级"
        _patch_infer(svc, p_attack=0.9, uniform=True)
        u = _push_until_windows(svc, 3, src_ip="10.0.0.77")
        assert u, "应产生未知告警"
        assert all(a["action"] == "throttle" for a in u), [a["action"] for a in u]

        # 安全设置必须压过自动升级：白名单源与「全局只观察」都不允许被升级成实际处置。
        # 这两条是文档里明确承诺的"永不处置"，自动升级不能悄悄推翻它们。
        svc.reload_policy(threshold=0.5, confirm_windows=1, unknown_threshold=0.0,
                          action="block", escalate_after=2, escalate_window_sec=300,
                          whitelist=["10.0.0.7"])
        svc.reset_stream("换源测白名单")
        _patch_infer(svc, p_attack=0.9, uniform=False)
        w = _push_until_windows(svc, 4, src_ip="10.0.0.7")
        assert w, "应产生告警"
        assert all(a["action"] == "observe" for a in w), \
            f"白名单源永不被处置，自动升级也不得越过：{[a['action'] for a in w]}"

        svc.reload_policy(action="observe", whitelist=[],
                          escalate_after=2, escalate_window_sec=300)
        svc.reset_stream("换全局 observe")
        o = _push_until_windows(svc, 4, src_ip="10.0.0.8")
        assert o, "应产生告警"
        assert all(a["action"] == "observe" for a in o), \
            f"全局 observe 表示只观察，升级不得把它变成限速/阻断：{[a['action'] for a in o]}"

        # executed 表示"真的下发了规则"，不是"策略不是演练模式"：
        # observe 没有规则可下发，必须记 False，否则审计台账在说谎。
        svc.reload_policy(action="observe", dry_run=False, whitelist=[],
                          escalate_after=99, escalate_window_sec=300)
        svc.reset_stream("测 executed 语义")
        e = _push_until_windows(svc, 1, src_ip="10.0.0.9")
        assert e, "应产生告警"
        assert all(a["executed"] is False for a in e), \
            f"observe 无规则可下发，executed 必须为 False：{[a['executed'] for a in e]}"
        svc.reload_policy(action="throttle", dry_run=False)
        svc.reset_stream("测 throttle executed")
        t2 = _push_until_windows(svc, 1, src_ip="10.0.0.10")
        assert t2, "应产生告警"
        assert all(a["executed"] is True for a in t2), \
            f"throttle + 非演练模式应记为已执行：{[a['executed'] for a in t2]}"


def test_autotighten_requires_switch_and_human_verdicts():
    """自动放松（自动降级）：默认关；开启后只由**人工结论**驱动，有占比门槛、有上限、有审计。

    这一项直接复用本工程已建的复核闭环（`results/review_feedback.jsonl`）：
    系统此前只会自动升级（越报越狠），长期高误报没有自愈路径。
    """
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        svc = _review_service(Path(td))
        svc.reload_policy(threshold=0.5, confirm_windows=1, unknown_threshold=1.0, action="observe")
        _patch_infer(svc, p_attack=0.9, uniform=True)
        alerts = _push_until_windows(svc, 5, src_ip="10.0.0.21")
        rids = [a["review_id"] for a in alerts if isinstance(a.get("review_id"), int)]
        assert len(rids) >= 5, f"应有 5 条待研判项，实际 {rids}"

        # ① 默认关：即便人工把告警判成 normal，阈值也不许动（可选能力不得悄悄改变工作点）
        for rid in rids[:3]:
            assert svc.label_review(rid, "normal")["ok"]
        assert svc.policy.autotighten is False
        r = svc._maybe_autotighten()
        assert r["applied"] is False and "未开启" in r["reason"], r
        assert svc.policy.threshold == 0.5 and int(svc.stats.get("autotighten", 0)) == 0
        assert svc.stats["review_labeled_normal"] == 3

        # ② 显式开启，但已研判结论条数不足 → 不动
        svc.reload_policy(autotighten=True, autotighten_min_labeled=5,
                          autotighten_normal_ratio=0.8, autotighten_step=0.05,
                          autotighten_max_threshold=0.9)
        r = svc._maybe_autotighten()
        assert r["applied"] is False and r["n_labeled"] == 3 and "不足" in r["reason"], r
        assert svc.policy.threshold == 0.5

        # ③ normal 占比不达标（3/4 = 75% < 80%）→ 说明不是"普遍误报"，不得放松
        assert svc.label_review(rids[3], "attack:Botnet")["ok"]
        assert svc.policy.threshold == 0.5, "normal 占比未达门槛不得上调阈值"
        assert int(svc.stats.get("autotighten", 0)) == 0
        assert svc.stats["review_confirmed_attack"] == 1

        # ④ 达标即自动放松一档：触发点是 label_review 内部那次检查（不需要人工再点一次）
        assert svc.label_review(rids[4], "normal")["ok"]
        assert abs(svc.policy.threshold - 0.55) < 1e-9, svc.policy.threshold
        assert int(svc.stats["autotighten"]) == 1
        snap = svc.snapshot()
        assert snap["autotighten_total"] == 1 and snap["review_labeled_normal"] == 4, snap
        assert snap["autotighten_last"]["applied"] is True

        # 阈值变更必须留在审计台账里，且能回答"凭什么"（人工结论条数与 normal 占比）
        audit = [json.loads(x) for x in
                 svc.audit_path.read_text(encoding="utf-8").strip().splitlines()]
        ev = [x for x in audit if x.get("event") == "autotighten"]
        assert ev, "阈值被自动上调必须写审计台账"
        assert ev[-1]["from_threshold"] == 0.5 and ev[-1]["to_threshold"] == 0.55, ev[-1]
        assert ev[-1]["n_labeled"] == 5 and ev[-1]["n_normal"] == 4, ev[-1]
        prom = svc.prometheus()
        assert "sdn_ids_autotighten_total 1" in prom, "Prometheus 必须暴露自动放松次数"
        assert "sdn_ids_threshold 0.55" in prom, "Prometheus 必须暴露当前阈值"

        # ⑤ 有上限：反复满足条件也只能爬到 autotighten_max_threshold（不许推到 1.0 把检测关掉）
        for _ in range(30):
            r = svc._maybe_autotighten()
        assert abs(svc.policy.threshold - 0.9) < 1e-9, svc.policy.threshold
        assert r["applied"] is False and "上限" in r["reason"], r

        # ⑥ 口径必须是"人工已给结论的条数"：未研判的队列项不得计入
        extra = _push_until_windows(svc, 2, src_ip="10.0.0.31")
        assert extra, "应继续产生告警"
        assert svc.list_review(status="pending"), "新告警仍在待研判状态"
        assert svc._maybe_autotighten()["n_labeled"] == 5, "未研判项不得计入放松依据"

        # ⑦ 开关也支持字符串形式关闭（前端表单传 "0"）
        svc.reload_policy(autotighten="0")
        assert svc.policy.autotighten is False
        assert "未开启" in svc._maybe_autotighten()["reason"]

        # ⑧ 轮转备份里装的是**旧内容**：同一 review_id 的改判必须以新记录为准。
        #    若按文件名顺序（.1 → 当前）读，旧结论会覆盖改判，阈值就会按错误的误报率被调整。
        import os
        import time

        assert svc.label_review(rids[0], "attack:Probe")["ok"]       # 改判，写进当前文件
        rot = svc.feedback_path.with_name(svc.feedback_path.name + ".1")
        rot.write_text(json.dumps({"review_id": int(rids[0]), "verdict": "normal"}) + "\n",
                       encoding="utf-8")
        older = time.time() - 600
        os.utime(rot, (older, older))                                # 备份 = 更早的内容
        svc2 = _review_service(Path(td))                             # 模拟重启：只看文件、不看内存
        assert svc2._review_verdicts()[int(rids[0])] == "attack:Probe", \
            "轮转备份里的旧结论不得覆盖当前文件里的改判"


def test_observation_pool_carries_evidence():
    """观察池条目必须带可追溯的流证据（五元组级），而不是只有一个分数。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        svc = _review_service(Path(td))
        svc.reload_policy(threshold=0.5, confirm_windows=99, action="observe")
        _patch_infer(svc, p_attack=0.9, uniform=False)
        _push_until_windows(svc, 1, src_ip="10.1.2.3",
                            extra={"Dst IP": "10.9.9.9", "Dst Port": 80})
        ob = svc.list_observations(1)[0]
        ev = ob.get("evidence") or []
        assert ev, "观察池条目必须附流证据"
        assert all(r.get("Src IP") == "10.1.2.3" for r in ev), ev[:2]
        assert any(r.get("Dst IP") == "10.9.9.9" for r in ev), ev[:2]
        assert len(ev) <= int(svc.cfg.window_size), "证据条数不应超过窗口长度"
        # 存的是定位字段，不是全部 71 维特征（否则观察池会吃到几十 MB）
        assert all(len(r) <= 6 for r in ev), [len(r) for r in ev]


def test_feedback_dataset_builder_guards():
    """回流数据集构建器：只认人工 verdict、硬校验特征列序、检出并忽略真实标签、拒绝退化输入。"""
    import importlib.util

    tool = Path(__file__).resolve().parents[1] / "tools" / "retrain_from_feedback.py"
    spec = importlib.util.spec_from_file_location("cida_retrain_tool", tool)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    feats = ["f0", "f1"]
    classes = ["Normal", "DDoS", "Botnet"]
    win = 2
    x_ok = [0.1, 0.2, 0.3, 0.4]          # T×F = 2×2

    def ev(rid, names=None, x=None, gt=False):
        flow = [{"Src IP": "10.0.0.1", "Dst IP": "10.0.0.2"}]
        if gt:
            flow[0]["Label"] = "WebAttack"       # 回放数据自带的真值：必须被忽略
        return {"review_id": rid, "X": x_ok if x is None else x,
                "feature_names": feats if names is None else names,
                "window_size": win, "flow_evidence": flow}

    fb = [
        {"review_id": 1, "verdict": "attack:Botnet"},
        {"review_id": 2, "verdict": "normal"},
        {"review_id": 3, "verdict": "maybe"},           # 非法结论 → 跳过
        {"review_id": 4, "verdict": "attack:Probe"},    # 类别不在模型类别表 → 拒绝
        {"review_id": 5, "verdict": "normal"},          # 有结论但缺证据 → 跳过
        {"review_id": 6, "verdict": "normal"},          # 特征列序不同 → 跳过
        {"review_id": 7, "verdict": "normal"},          # 向量长度不对 → 跳过
    ]
    evs = [ev(1, gt=True), ev(2), ev(4), ev(6, names=["f1", "f0"]), ev(7, x=[0.1])]

    ds = mod.build_feedback_dataset(evs, fb, feats, classes, win)
    assert ds["X"].shape == (2, win, len(feats)), ds["X"].shape
    assert list(ds["y_mc"]) == [2, 0], ds["y_mc"]        # Botnet=2；normal → Normal=0
    assert list(ds["y_bin"]) == [1.0, 0.0], ds["y_bin"]
    assert ds["normal_index"] == 0
    assert ds["evidence_with_gt_label"] == 1, "必须检出证据里的真实标签（并已忽略）"
    assert len(ds["bad_class"]) == 1 and "Probe" in ds["bad_class"][0], ds["bad_class"]
    skipped = " ".join(ds["skipped"])
    for frag in ("verdict 非法", "缺证据", "特征名"):
        assert frag in skipped, f"应跳过「{frag}」：{ds['skipped']}"
    assert any("向量长度" in s for s in ds["skipped"]), ds["skipped"]

    # 没有任何可用样本时必须拒绝，而不是产出一个空数据集装作成功
    try:
        mod.build_feedback_dataset([], [], feats, classes, win)
    except SystemExit as exc:
        assert "没有" in str(exc)
    else:
        raise AssertionError("空输入必须被拒绝")


def _patch_infer_sequence(svc, scores):
    """按**窗口顺序**逐个给出 p_attack 的推理假件（用于精确验证连续确认语义）。"""
    seq = list(scores)
    n_cls = max(len(svc.class_names), 2)

    def _fake(raw):
        n = raw.shape[0]
        got = [float(seq.pop(0)) if seq else 0.0 for _ in range(n)]
        p_mc = np.full((n, n_cls), 0.01)
        p_mc[:, 0] = 1.0 - 0.01 * (n_cls - 1)
        return p_mc, np.asarray(got, dtype=float)

    svc._infer = _fake


def _collect_window_flags(svc, n_windows: int, max_batches: int = 80):
    """推流并收集逐窗口的 (alarm, confirmed) 标记（窗口由 stride 决定，故循环到够为止）。"""
    row = {f: 0.0 for f in svc.feature_names}
    flags: list = []
    for _ in range(max_batches):
        if len(flags) >= n_windows:
            break
        res = svc.push([dict(row) for _ in range(int(svc.cfg.window_size))], {})
        assert res.get("ok"), res
        for item in res["rows"]:
            w = item.get("window")
            if w is not None:
                flags.append((bool(w["alarm"]), bool(w["confirmed"])))
    assert len(flags) >= n_windows, f"只产生 {len(flags)} 个窗口，期望 {n_windows}"
    return flags[:n_windows]


def test_confirmation_semantics_agree_across_implementations():
    """连续确认（去抖）语义在三个实现里必须**完全一致**——用断言钉住，而不是靠"读过觉得一致"。

    同一规则被写了三遍：服务端 `src/service.py::push`、`src/evaluate.py::confirm_alarms`、
    `tools/real_eval.py::online_metrics`。本工程已两次栽在"同一语义多处各判一次、后来只改一处"
    （白名单 vs 升级、服务端升级 vs 适配层幂等），所以这里对同一串窗口分数要求三处给出同一结果。
    """
    import importlib.util
    import tempfile

    from src.evaluate import confirm_alarms

    tool = Path(__file__).resolve().parents[1] / "tools" / "real_eval.py"
    spec = importlib.util.spec_from_file_location("cida_real_eval_for_confirm", tool)
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)  # type: ignore[union-attr]

    thr = 0.5
    # 覆盖：连续命中、中断后重来、刚好达阈值次数、长连击
    seq = [0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.2]
    above = np.asarray([s >= thr for s in seq])

    for k in (1, 2, 3, 5):
        ref = list(confirm_alarms(above, k))

        # ② real_eval 的实现：全 1 标签 → 所有告警都在片段内，故 alarms 数即确认数
        r = real.online_metrics(np.arange(len(seq), dtype=float) * 60.0,
                               np.ones(len(seq), dtype=int), np.asarray(seq, dtype=float), thr, k)
        assert int(r["alarms"]) == int(sum(ref)), \
            f"real_eval 的确认数与 evaluate 不一致（N={k}）：{r['alarms']} vs {sum(ref)}"

        # ③ 服务端的实现：逐窗口标记必须与参考实现逐位相同
        with tempfile.TemporaryDirectory() as td:
            svc = _review_service(Path(td))
            svc.reload_policy(threshold=thr, confirm_windows=k, unknown_threshold=0.0,
                              action="observe", dry_run=True)
            _patch_infer_sequence(svc, seq)
            flags = _collect_window_flags(svc, len(seq))
        got_confirmed = [f[1] for f in flags]
        assert got_confirmed == ref, \
            f"服务端确认语义与 evaluate 不一致（N={k}）：\n  服务端 {got_confirmed}\n  参考   {ref}"
        assert [f[0] for f in flags] == list(above), \
            f"服务端报警语义与阈值比较不一致（N={k}）"


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
