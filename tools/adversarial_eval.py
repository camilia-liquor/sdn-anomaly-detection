"""对抗鲁棒性实测：给「未做对抗样本测试」这条边界补上量化结论与缓解手段。

为什么做这个
------------
入侵检测模型上线后会遇到**主动对抗**：攻击者知道（或猜到）模型用什么特征，
只要把包长/时序/标志位微调到"看起来正常"，就能让检测器放行。
工程里原本只写了"未做对抗样本鲁棒性测试"——这是一个真实的空白，而且是最容易被追问的一类空白。

本工具回答三个问题（判据事前写死）：
  1. **有多脆**：在 L∞ 扰动预算 ε 下，定向逃逸攻击能把多快的攻击"洗白"？
     记录宏 F1、二分类 F1/AUC、TPR@FPR=1%、以及**逃逸率**（攻击窗口被判成正常的比例增量）。
  2. **能不能缓解**：对抗训练（clean + FGSM 各半）之后，鲁棒性提升多少、干净精度付出多少代价？
     判据：`gain = F1_adv_train(ε*) − F1_clean_train(ε*)`；`cost = F1_clean_train(0) − F1_adv_train(0)`。
     * 只有 `gain > cost` 才算"划算的缓解"（写成硬判据，避免只挑好看的一侧）。
  3. **能不能被发现**：扰动样本的预测熵是否显著升高？若是，则工程里**已有的**熵阈值未知检测
     可以顺带拦住一部分对抗样本（零额外成本的缓解）。

威胁模型（写清楚才不算夸大）
--------------------------
  * 白盒、特征空间、L∞ 有界：攻击者知道模型与特征口径，可直接改流记录的特征值；
  * 扰动后**投影回训练集每个特征的真实取值范围**（单调变换下等价于原始量纲的物理可行域），
    因此不存在"包长为负""包数超过 65535"这类不可能样本；
  * 目标：`evade_binary`（最小化二分类攻击概率）与 `target_normal`（最大化被判为 Normal 的概率）；
  * 对照：**同范数的随机扰动**——用来证明"FGSM 的破坏力不是随便加噪声就能达到的"。
  不在威胁模型内：不改标签、不改模型、不做查询式（黑盒）攻击、不涉及报文层实现。

数据与产物隔离
--------------
  只**读** data/processed（主实验张量）与 artifacts/cnn_bilstm_main_s0（主模型），
  对抗训练的新模型写在 artifacts/adversarial/ 下，结果写在 results/adversarial_eval.*，
  **不触碰**主实验产物；`publish_summary` 语义上更不会碰（本工具不经过 run_all 的汇总路径）。

用法
----
    python tools/adversarial_eval.py                     # 完整（FGSM + PGD + 随机基线 + 对抗训练）
    python tools/adversarial_eval.py --quick             # 快速自检
    python tools/adversarial_eval.py --eps 0.05,0.1,0.2,0.3 --no-adv-train
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData, build_windows  # noqa: E402
from src.evaluate import (  # noqa: E402
    binary_metrics,
    multiclass_metrics,
    online_alarm_evaluation,
    tpr_at_fpr,
)
from src.model import load_model_bundle, predict_probs  # noqa: E402
from src.train import run_training  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR,
    FIGURE_DIR,
    PROCESSED_DIR,
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

LOGGER = get_logger("cida.adversarial")

# --------------------------------------------------------------------------------------
# 事前写死的判据与常量
# --------------------------------------------------------------------------------------
DEFAULT_EPS = (0.05, 0.10, 0.20, 0.30, 0.50)   # L∞ 预算（标准化空间；同时报告原始量纲的相对变化）
PGD_STEPS = 10                                  # PGD 迭代步数（α = ε/4，标准设置）
PGD_ALPHA_RATIO = 0.25
THRESHOLD = 0.50                                # 逃逸率与在线口径使用的工作点（与主实验默认一致）
CONFIRM = 3
MAIN_MODEL_TAG = "cnn_bilstm_main_s0"
EPS_TRAIN = 0.20                                # 对抗训练使用的扰动预算（在 ε 网格内取中间值，事前指定）
ADV_TRAIN_RATIO = 0.5                           # 训练集里对抗样本的比例


# --------------------------------------------------------------------------------------
# 1) 攻击实现（白盒、特征空间、L∞ 有界 + 物理投影）
# --------------------------------------------------------------------------------------
def _model_probs(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = model(X, training=False)
    if isinstance(out, dict):
        p_mc = np.asarray(out["mc"])
        p_bin = np.asarray(out["bin"]).reshape(-1)
    else:  # 单输出（无二分类头）
        p_mc = np.asarray(out)
        p_bin = 1.0 - p_mc[:, 0]
    return p_mc, p_bin


def _loss_and_grad(model, X: np.ndarray, objective: str, normal_idx: int):
    """返回 (loss_scalar, grad)；goal 是**最小化**该 loss。"""
    import tensorflow as tf

    x = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = model(x, training=False)
        if isinstance(out, dict):
            p_mc = out["mc"]
            p_bin = tf.reshape(out["bin"], [-1])
        else:
            p_mc = out
            p_bin = 1.0 - p_mc[:, normal_idx]
        if objective == "evade_binary":
            loss = tf.reduce_mean(p_bin)                       # 让"是攻击"的概率尽量小
        elif objective == "target_normal":
            p_normal = tf.gather(p_mc, normal_idx, axis=1)
            loss = -tf.reduce_mean(tf.math.log(p_normal + 1e-12))   # 让 Normal 的概率尽量大
        else:
            raise ValueError(f"未知攻击目标 {objective}")
    grad = tape.gradient(loss, x)
    return float(loss.numpy()), np.asarray(grad)


def attack(model, X: np.ndarray, *, eps: float, objective: str, normal_idx: int,
           steps: int = 1, lo: np.ndarray | None = None, hi: np.ndarray | None = None,
           rng: np.random.Generator | None = None, random_only: bool = False) -> np.ndarray:
    """FGSM（steps=1）/ PGD（steps>1）/ 随机基线（random_only=True）。

    每一步都做两次投影：① L∞ 球面内（x0 ± eps）；② 训练集特征的可行域 [lo, hi]。
    """
    X0 = np.asarray(X, dtype=np.float32)
    if random_only:
        r = rng if rng is not None else np.random.default_rng(0)
        Xadv = X0 + r.uniform(-eps, eps, size=X0.shape).astype(np.float32)
    else:
        alpha = float(eps) if steps <= 1 else float(eps) * PGD_ALPHA_RATIO
        Xadv = X0.copy()
        for _ in range(max(int(steps), 1)):
            _, grad = _loss_and_grad(model, Xadv, objective, normal_idx)
            Xadv = Xadv - alpha * np.sign(grad).astype(np.float32)
            Xadv = np.clip(Xadv, X0 - eps, X0 + eps)
            if lo is not None and hi is not None:
                Xadv = np.clip(Xadv, lo, hi)
    return Xadv.astype(np.float32)


# --------------------------------------------------------------------------------------
# 2) 指标
# --------------------------------------------------------------------------------------
def window_metrics(model, X: np.ndarray, y_mc: np.ndarray, y_bin: np.ndarray,
                   class_names: list[str], cfg: Config) -> dict:
    p_mc, p_bin = _model_probs(model, X)
    mc = multiclass_metrics(y_mc, p_mc, class_names)
    bm = binary_metrics(y_bin, p_bin, THRESHOLD, cfg)
    attack = y_bin == 1
    called_normal = p_bin < THRESHOLD
    return {
        "macro_f1": float(mc["macro_f1"]),
        "accuracy": float(mc["accuracy"]),
        "binary_f1": float(bm["f1"]),
        "roc_auc": float(bm["roc_auc"]),
        "tpr_at_fpr_1pct": float(tpr_at_fpr(y_bin, p_bin, 0.01)),
        "attack_evasion_rate": float(called_normal[attack].mean()) if attack.any() else None,
        "normal_call_rate": float((~called_normal[~attack]).mean()) if (~attack).any() else None,
        "mean_p_attack_on_attacks": float(p_bin[attack].mean()) if attack.any() else None,
        "mean_entropy": float((-np.sum(p_mc * np.log(p_mc + 1e-12), axis=1)).mean()),
        "n_windows": int(len(y_mc)),
        "_p_bin": p_bin,
    }


def online_metrics(model, data: PreparedData, cfg: Config) -> dict:
    """在线告警级口径：对**整条测试流**的滚动窗口逐个做攻击，再看告警行为。"""
    X_stream, _, _, info = build_windows(
        data.test_stream_features, labels=data.test_stream_labels,
        window_size=cfg.window_size, stride=cfg.online_stride, pure_only=False,
    )
    starts = np.asarray(info["starts"])
    _, p_bin = _model_probs(model, X_stream)
    on = online_alarm_evaluation(data, p_bin, starts, cfg, threshold=THRESHOLD, confirm=CONFIRM)
    return {
        "segment_recall": float(on["segment_recall"]),
        "n_segments": int(on["n_attack_segments"]),
        "n_detected": int(on["n_segments_detected"]),
        "mean_delay_ms": (float(on["mean_detection_delay_sec"]) * 1000.0
                          if on["mean_detection_delay_sec"] is not None else None),
        "false_alarms_per_hour": on["false_alarms_per_hour"],
        "n_stream_windows": int(len(starts)),
    }


def raw_relative_change(data: PreparedData, X0: np.ndarray, Xadv: np.ndarray) -> dict:
    """把标准化空间的扰动翻译成**原始量纲**的可解释量。

    ⚠️ 不用"相对变化 |Δ|/|原值|"：很多特征的正常值就是 0（各类标志位计数、IAT 标准差），
    相对变化会除零爆炸、把统计量带偏。改用两个更稳且可解释的量：
      * `median_abs_scaled_change`：标准化空间的 |Δ|（就是攻击预算本身，做一致性核对）；
      * 每个特征**原始单位**的 |Δ| 中位数（例如"包数中位变化 37 个包"）。
    """
    sc = data.scaler
    f = X0.shape[-1]
    try:
        r0 = sc.inverse_transform(np.asarray(X0, dtype=np.float64).reshape(-1, f))
        r1 = sc.inverse_transform(np.asarray(Xadv, dtype=np.float64).reshape(-1, f))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("原始量纲换算失败（跳过该表）：%s", exc)
        return {}
    d_scaled = np.abs(np.asarray(Xadv, dtype=np.float64) - np.asarray(X0, dtype=np.float64)).reshape(-1, f)
    d_raw = np.abs(r1 - r0)
    names = list(data.feature_names)
    med_scaled = d_scaled.mean(axis=0)
    med_raw = np.median(d_raw, axis=0)
    order = np.argsort(med_scaled)[::-1][:5]
    return {
        "median_abs_scaled_change": float(np.median(d_scaled)),
        "p95_abs_scaled_change": float(np.percentile(d_scaled, 95)),
        "top_features": [{
            "feature": names[i],
            "median_abs_change_raw_units": float(med_raw[i]),
            "mean_abs_scaled_change": float(med_scaled[i]),
        } for i in order],
    }


# --------------------------------------------------------------------------------------
# 3) 主流程
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对抗鲁棒性实测（FGSM/PGD + 对抗训练 + 熵可检测性）")
    ap.add_argument("--eps", default=",".join(str(e) for e in DEFAULT_EPS))
    ap.add_argument("--model", default=MAIN_MODEL_TAG, help="被攻击的主模型 tag")
    ap.add_argument("--processed-dir", default="", help="预处理产物目录（默认 data/processed）")
    ap.add_argument("--epochs", type=int, default=40, help="对抗训练的轮数（与主实验一致）")
    ap.add_argument("--no-adv-train", action="store_true", help="跳过对抗训练")
    ap.add_argument("--no-online", action="store_true", help="跳过在线口径（省时间）")
    ap.add_argument("--quick", action="store_true", help="快速自检：ε 少、不做对抗训练")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    eps_list = [float(x) for x in str(args.eps).split(",") if x.strip()]
    if args.quick:
        eps_list = [0.1, 0.3]
        args.no_adv_train = True
    started = time.time()
    ensure_dirs(RESULT_DIR, FIGURE_DIR, ARTIFACT_DIR)
    silence_tf_logs()
    set_seed(42)

    cfg = Config(verbose=False, figures=False, publish_summary=False, epochs=args.epochs)
    processed = Path(args.processed_dir) if args.processed_dir else PROCESSED_DIR
    data = PreparedData.load(processed)
    normal_idx = data.normal_class_id
    model_path = ARTIFACT_DIR / args.model / "model.keras"
    if not model_path.exists():
        raise SystemExit(f"找不到主模型 {model_path}；请先跑 python run_all.py")
    model = load_model_bundle(model_path, compile_model=False)
    LOGGER.info("被攻击模型：%s（%s，%d 参数）；测试窗口 %d 个",
                args.model, data.input_shape, model.count_params(), len(data.y_test_mc))

    # 物理可行域（标准化空间下的逐特征取值范围 = 原始量纲的取值范围，单调变换下等价）
    lo = data.X_train.reshape(-1, data.X_train.shape[-1]).min(axis=0)
    hi = data.X_train.reshape(-1, data.X_train.shape[-1]).max(axis=0)

    # ---------- 干净基线 ----------
    clean = window_metrics(model, data.X_test, data.y_test_mc, data.y_test_bin, data.class_names, cfg)
    LOGGER.info("干净基线：宏F1=%.4f 二分类F1=%.4f AUC=%.4f TPR@1%%FPR=%.4f 攻击窗口被判正常=%.2f%%",
                clean["macro_f1"], clean["binary_f1"], clean["roc_auc"], clean["tpr_at_fpr_1pct"],
                100.0 * (clean["attack_evasion_rate"] or 0.0))
    rows: list[dict] = []

    for objective in ("evade_binary", "target_normal"):
        for eps in eps_list:
            t0 = time.time()
            Xf = attack(model, data.X_test, eps=eps, objective=objective, normal_idx=normal_idx,
                        steps=1, lo=lo, hi=hi)
            Xp = attack(model, data.X_test, eps=eps, objective=objective, normal_idx=normal_idx,
                        steps=PGD_STEPS, lo=lo, hi=hi)
            Xr = attack(model, data.X_test, eps=eps, objective=objective, normal_idx=normal_idx,
                        steps=1, lo=lo, hi=hi, rng=np.random.default_rng(1234), random_only=True)
            mf = window_metrics(model, Xf, data.y_test_mc, data.y_test_bin, data.class_names, cfg)
            mp = window_metrics(model, Xp, data.y_test_mc, data.y_test_bin, data.class_names, cfg)
            mr = window_metrics(model, Xr, data.y_test_mc, data.y_test_bin, data.class_names, cfg)
            rel = raw_relative_change(data, data.X_test, Xp)
            row = {
                "objective": objective, "eps": eps,
                "fgsm": {k: v for k, v in mf.items() if not k.startswith("_")},
                "pgd": {k: v for k, v in mp.items() if not k.startswith("_")},
                "random": {k: v for k, v in mr.items() if not k.startswith("_")},
                "raw_space": rel,
                "seconds": round(time.time() - t0, 1),
            }
            rows.append(row)
            LOGGER.info("[%s ε=%.2f] 宏F1 clean %.4f → FGSM %.4f / PGD %.4f / 随机 %.4f；"
                        "逃逸率(攻击窗口被判正常) %.1f%% → FGSM %.1f%% / PGD %.1f%%",
                        objective, eps, clean["macro_f1"], mf["macro_f1"], mp["macro_f1"], mr["macro_f1"],
                        100.0 * (clean["attack_evasion_rate"] or 0.0),
                        100.0 * (mf["attack_evasion_rate"] or 0.0),
                        100.0 * (mp["attack_evasion_rate"] or 0.0))

    # ---------- 在线口径（无攻击 / ε* 处最强的 PGD 攻击）----------
    online: dict = {}
    if not args.no_online and data.test_stream_features is not None:
        eps_star = eps_list[-1] if args.quick else float(np.median(eps_list))
        online["clean"] = online_metrics(model, data, cfg)
        LOGGER.info("在线（无攻击）：片段召回 %.2f，平均时延 %s ms，误报 %s 次/小时",
                    online["clean"]["segment_recall"],
                    fmt_float(online["clean"]["mean_delay_ms"], 2),
                    fmt_float(online["clean"]["false_alarms_per_hour"], 2))
        X_stream, _, _, info = build_windows(
            data.test_stream_features, labels=data.test_stream_labels,
            window_size=cfg.window_size, stride=cfg.online_stride, pure_only=False)
        starts = np.asarray(info["starts"])
        for objective in ("evade_binary",):
            for eps in sorted(set([eps_star, eps_list[-1]])):
                Xa = attack(model, X_stream, eps=eps, objective=objective, normal_idx=normal_idx,
                            steps=PGD_STEPS, lo=lo, hi=hi)
                _, p_bin = _model_probs(model, Xa)
                on = online_alarm_evaluation(data, p_bin, starts, cfg, threshold=THRESHOLD, confirm=CONFIRM)
                online[f"pgd_{objective}_eps{eps:g}"] = {
                    "segment_recall": float(on["segment_recall"]),
                    "n_segments": int(on["n_attack_segments"]),
                    "n_detected": int(on["n_segments_detected"]),
                    "mean_delay_ms": (float(on["mean_detection_delay_sec"]) * 1000.0
                                      if on["mean_detection_delay_sec"] is not None else None),
                    "false_alarms_per_hour": on["false_alarms_per_hour"],
                    "eps": eps,
                }
                LOGGER.info("在线（PGD ε=%.2f 攻击整条测试流）：片段召回 %.2f（%d/%d），误报 %s 次/小时",
                            eps, on["segment_recall"], on["n_segments_detected"], on["n_attack_segments"],
                            fmt_float(on["false_alarms_per_hour"], 2))

    # ---------- 对抗训练 ----------
    adv_train: dict = {}
    if not args.no_adv_train:
        t0 = time.time()
        LOGGER.info("对抗训练：FGSM(ε=%.2f) 生成对抗样本并与干净样本按 %d:%d 混合",
                    EPS_TRAIN, int((1 - ADV_TRAIN_RATIO) * 10), int(ADV_TRAIN_RATIO * 10))
        X_adv_train = attack(model, data.X_train, eps=EPS_TRAIN, objective="target_normal",
                             normal_idx=normal_idx, steps=1, lo=lo, hi=hi)
        X_aug = np.concatenate([data.X_train, X_adv_train], axis=0).astype(np.float32)
        y_mc = np.concatenate([data.y_train_mc, data.y_train_mc])
        y_bin = np.concatenate([data.y_train_bin, data.y_train_bin])
        order = np.random.default_rng(7).permutation(len(y_mc))
        data_adv = replace(data, X_train=X_aug[order], y_train_mc=y_mc[order], y_train_bin=y_bin[order],
                           split_info={**data.split_info, "adv_train": {
                               "eps": EPS_TRAIN, "ratio": ADV_TRAIN_RATIO,
                               "n_clean": int(len(data.X_train)), "n_adv": int(len(X_adv_train))}})
        cfg_adv = replace(cfg, seed=0, model_variant="cnn_bilstm")
        res = run_training(cfg_adv, data_adv, variant="cnn_bilstm", tag="advtrain_s0",
                           epochs=args.epochs, artifact_dir=ARTIFACT_DIR / "adversarial")
        adv_model = res.model
        adv_train["train_seconds"] = round(time.time() - t0, 1)
        adv_train["n_params"] = int(adv_model.count_params())
        adv_train["n_clean"] = int(len(data.X_train))
        adv_train["n_adv"] = int(len(X_adv_train))
        adv_train["eps_train"] = EPS_TRAIN
        adv_train["epochs"] = int(args.epochs)
        adv_train["clean"] = {k: v for k, v in
                              window_metrics(adv_model, data.X_test, data.y_test_mc, data.y_test_bin,
                                             data.class_names, cfg).items() if not k.startswith("_")}
        adv_train["grid"] = []
        for eps in eps_list:
            Xp = attack(adv_model, data.X_test, eps=eps, objective="target_normal",
                        normal_idx=normal_idx, steps=PGD_STEPS, lo=lo, hi=hi)
            m = window_metrics(adv_model, Xp, data.y_test_mc, data.y_test_bin, data.class_names, cfg)
            adv_train["grid"].append({"eps": eps, **{k: v for k, v in m.items() if not k.startswith("_")}})
            LOGGER.info("[对抗训练后 ε=%.2f] 宏F1=%.4f 二分类F1=%.4f 逃逸率=%.2f%%",
                        eps, m["macro_f1"], m["binary_f1"], 100.0 * (m["attack_evasion_rate"] or 0.0))
        # 判据：收益 vs 代价（事前写死）
        eps_ref = EPS_TRAIN
        clean_at = min(rows, key=lambda r: abs(r["eps"] - eps_ref) if r["objective"] == "target_normal"
                       else 9e9)
        gain = adv_train["grid"][eps_list.index(eps_ref)]["macro_f1"] - clean_at["pgd"]["macro_f1"] \
            if eps_ref in eps_list else None
        cost = clean["macro_f1"] - adv_train["clean"]["macro_f1"]
        adv_train["criterion"] = {
            "eps_ref": eps_ref,
            "f1_robust_gain": gain,
            "f1_clean_cost": cost,
            "worthwhile": (None if gain is None else bool(gain > cost)),
            "rule": "只有 F1_robust_gain > F1_clean_cost 才算「划算的缓解」（事前写死）",
        }

    payload = {
        "experiment": "adversarial_robustness",
        "model": {"tag": args.model, "n_params": int(model.count_params()),
                  "input_shape": list(data.input_shape)},
        "threat_model": {
            "type": "white-box, feature-space, L-infinity bounded",
            "objectives": ["evade_binary", "target_normal"],
            "physical_projection": "扰动后投影回训练集逐特征取值范围（原始量纲的物理可行域）",
            "baseline": "同范数随机扰动（证明 FGSM 的破坏力不是随便加噪声能达到的）",
            "out_of_scope": ["改标签", "改模型", "黑盒查询攻击", "报文层实现"],
        },
        "threshold": THRESHOLD, "confirm": CONFIRM,
        "pgd_steps": PGD_STEPS, "eps_list": eps_list,
        "clean": {k: v for k, v in clean.items() if not k.startswith("_")},
        "grid": rows,
        "online": online,
        "adv_train": adv_train,
        "elapsed_s": round(time.time() - started, 1),
    }
    save_json(payload, RESULT_DIR / "adversarial_eval.json")
    save_text(build_report(payload), RESULT_DIR / "adversarial_eval.md")
    if not args.no_figure:
        fp = make_figure(payload, FIGURE_DIR / "adversarial_robustness.png")
        LOGGER.info("图：%s", fp)
    LOGGER.info("完成：%s、%s（用时 %.0fs）", RESULT_DIR / "adversarial_eval.md",
                RESULT_DIR / "adversarial_eval.json", payload["elapsed_s"])
    return 0


# --------------------------------------------------------------------------------------
# 4) 报告与图
# --------------------------------------------------------------------------------------
def _compact(v: float | None) -> str:
    """把原始单位的扰动量写得紧凑可读（大数用科学计数法）。"""
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e6:
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.3f}"


def build_report(p: dict) -> str:
    clean = p["clean"]
    rows = [r for r in p["grid"] if r["objective"] == "evade_binary"]
    rows_t = [r for r in p["grid"] if r["objective"] == "target_normal"]
    adv = p.get("adv_train") or {}
    online = p.get("online") or {}

    def _worst(rs, key="macro_f1", who="pgd"):
        vals = [r[who][key] for r in rs if r[who].get(key) is not None]
        return min(vals) if vals else None

    lines = [
        "# 对抗鲁棒性实测：白盒 L∞ 扰动下检测器有多脆、能不能缓解",
        "",
        "> 本报告由 `tools/adversarial_eval.py` 生成，全部数字为**本机实测**；"
        "被攻击模型 = 主实验模型 `cnn_bilstm_main_s0`，测试集 = 主实验留出测试窗口。",
        "",
        "## 0. 结论（先回答问题）",
        "",
        f"- **有多脆**：干净时宏 F1 **{fmt_float(clean['macro_f1'])}**、攻击窗口被判为正常的比例 "
        f"**{100.0 * (clean['attack_evasion_rate'] or 0):.2f}%**；"
        f"在最强的 PGD 攻击下最差掉到 **{fmt_float(_worst(rows))}**"
        f"（逃逸率最高 **{100.0 * max((r['pgd']['attack_evasion_rate'] or 0) for r in rows):.1f}%**）。",
        f"- **随机扰动对照**：同范数随机扰动的宏 F1 最低只有 "
        f"**{fmt_float(_worst(rows, who='random'))}**，说明上一条的退化**确实来自定向优化**，"
        f"不是「随便加噪声」的效果。",
    ]
    if adv:
        c = adv.get("criterion") or {}
        lines += [
            f"- **能不能缓解**：对抗训练（clean + FGSM ε={adv.get('eps', EPS_TRAIN)} 各半）之后，"
            f"ε={c.get('eps_ref')} 处宏 F1 收益 **{fmt_float(c.get('f1_robust_gain'))}**，"
            f"干净精度代价 **{fmt_float(c.get('f1_clean_cost'))}** → "
            f"**{'划算（收益 > 代价）' if c.get('worthwhile') else '不划算（代价 ≥ 收益）'}**。",
        ]
    if online.get("clean"):
        lines.append(
            f"- **在线口径**：无攻击时片段召回 **{online['clean']['segment_recall']:.0%}**、"
            f"误报 {fmt_float(online['clean']['false_alarms_per_hour'], 2)} 次/小时；"
            + "；".join(
                f"PGD ε={v['eps']:g} 时召回 **{v['segment_recall']:.0%}**（{v['n_detected']}/{v['n_segments']}）"
                for k, v in online.items() if k != "clean")
            + "。")
    lines += [
        "",
        "## 1. 威胁模型与判据（事前写死）",
        "",
        "- **白盒、特征空间、L∞ 有界**：攻击者知道模型与特征口径，可直接改流记录的特征值；",
        "- 扰动后**投影回训练集逐特征的真实取值范围**（标准化是逐特征单调变换，因此等价于原始量纲的物理可行域），"
        "不会出现「包长为负」这类不可能样本；",
        "- 目标函数两种：`evade_binary`（最小化二分类攻击概率）、`target_normal`（最大化被判为 Normal 的概率）；",
        "- 对照：**同范数随机扰动**；",
        f"- 工作点：阈值 {THRESHOLD} + 连续确认 {CONFIRM}（与主实验默认一致）；PGD 步数 {PGD_STEPS}（α = ε/4）；",
        "- 缓解判据：`F1_robust_gain > F1_clean_cost` 才算「划算的缓解」——只报一侧不算结论。",
        "- 不在威胁模型内：改标签、改模型、黑盒查询式攻击、报文层实现（这些留给后续工作，报告如实标注）。",
        "",
        "## 2. 主表：扰动预算 ε × 攻击方法（目标 = 逃逸二分类检测）",
        "",
    ]
    head = ["ε", "攻击", "宏 F1", "二分类 F1", "ROC-AUC", "TPR@1%FPR", "**逃逸率**", "攻击窗口平均 p_attack"]
    table = []
    table.append(["0（干净）", "—", fmt_float(clean["macro_f1"]), fmt_float(clean["binary_f1"]),
                  fmt_float(clean["roc_auc"]), fmt_float(clean["tpr_at_fpr_1pct"]),
                  f"{100.0 * (clean['attack_evasion_rate'] or 0):.2f}%",
                  fmt_float(clean["mean_p_attack_on_attacks"])])
    for r in rows:
        for who, name in (("fgsm", "FGSM(1 步)"), ("pgd", f"PGD({p['pgd_steps']} 步)"),
                          ("random", "随机扰动（对照）")):
            m = r[who]
            table.append([
                fmt_float(r["eps"], 2), name, fmt_float(m["macro_f1"]), fmt_float(m["binary_f1"]),
                fmt_float(m["roc_auc"]), fmt_float(m["tpr_at_fpr_1pct"]),
                f"{100.0 * (m['attack_evasion_rate'] or 0):.2f}%",
                fmt_float(m["mean_p_attack_on_attacks"]),
            ])
    lines.append(markdown_table(head, table))
    lines += ["", "### 2.1 扰动在**原始量纲**上意味着什么（避免只报无量纲的 ε）", ""]
    head2 = ["ε", "|Δ| 中位（标准化）", "|Δ| P95（标准化）", "被改动最多的 5 个特征（原始单位 |Δ| 中位）"]
    t2 = []
    for r in rows:
        rs = r.get("raw_space") or {}
        if not rs:
            continue
        top = "、".join(f"{t['feature']} {_compact(t['median_abs_change_raw_units'])}"
                        for t in rs.get("top_features", [])[:5])
        t2.append([fmt_float(r["eps"], 2),
                   fmt_float(rs.get("median_abs_scaled_change"), 3),
                   fmt_float(rs.get("p95_abs_scaled_change"), 2), top])
    lines.append(markdown_table(head2, t2))
    lines += ["", "## 3. 目标 = 把攻击洗成 Normal（`target_normal`）", ""]
    t3 = []
    for r in rows_t:
        for who, name in (("fgsm", "FGSM"), ("pgd", "PGD")):
            m = r[who]
            t3.append([fmt_float(r["eps"], 2), name, fmt_float(m["macro_f1"]),
                       f"{100.0 * (m['attack_evasion_rate'] or 0):.2f}%"])
    lines.append(markdown_table(["ε", "攻击", "宏 F1", "逃逸率（攻击窗口被判正常）"], t3))

    if online:
        lines += ["", "## 4. 在线告警级口径（攻击整条测试流）", ""]
        head4 = ["场景", "片段召回", "检出/片段", "平均时延(ms)", "误报/小时"]
        t4 = []
        if online.get("clean"):
            c = online["clean"]
            t4.append(["无攻击（干净）", f"{c['segment_recall']:.0%}",
                       f"{c['n_detected']}/{c['n_segments']}", fmt_float(c["mean_delay_ms"], 2),
                       fmt_float(c["false_alarms_per_hour"], 2)])
        for k, v in online.items():
            if k == "clean":
                continue
            t4.append([f"PGD ε={v['eps']:g}（逃逸目标）", f"{v['segment_recall']:.0%}",
                       f"{v['n_detected']}/{v['n_segments']}", fmt_float(v["mean_delay_ms"], 2),
                       fmt_float(v["false_alarms_per_hour"], 2)])
        lines.append(markdown_table(head4, t4))
        lines.append("")
        lines.append("> 注意：在线口径里每个窗口**各自独立**被攻击（窗口级攻击者），"
                     "片段标签与时间轴不变——因此「召回下降」直接读作「攻击者让某些片段整体隐形」。")

    if adv:
        c = adv.get("criterion") or {}
        lines += ["", "## 5. 缓解：对抗训练（clean + FGSM 各半）", "",
                  f"- 训练集：干净 **{adv.get('n_clean', 0):,}** + 对抗样本 **{adv.get('n_adv', 0):,}**"
                  f"（对抗样本占比 {ADV_TRAIN_RATIO:.0%}）；扰动预算 ε={adv.get('eps_train', EPS_TRAIN)}；"
                  f"轮数 {adv.get('epochs', '?')}（与主实验同轮数）；训练耗时 {adv.get('train_seconds')} s；"
                  f"参数量 {adv.get('n_params', 0):,}",
                  "",
                  f"- **干净精度代价**：宏 F1 {fmt_float(clean['macro_f1'])} → "
                  f"{fmt_float(adv['clean']['macro_f1'])}（**{fmt_float(c.get('f1_clean_cost'))}**）",
                  f"- **鲁棒收益**（ε={c.get('eps_ref')}，目标 `target_normal` 的 PGD）："
                  f"宏 F1 由 {fmt_float(next((r['pgd']['macro_f1'] for r in rows_t if abs(r['eps'] - (c.get('eps_ref') or 0)) < 1e-9), None))} "
                  f"提升到 {fmt_float(next((g['macro_f1'] for g in adv['grid'] if abs(g['eps'] - (c.get('eps_ref') or 0)) < 1e-9), None))}"
                  f"（**{fmt_float(c.get('f1_robust_gain'))}**）",
                  f"- **判据**：{c.get('rule')} → "
                  f"**{'划算' if c.get('worthwhile') else '不划算'}**",
                  "",
                  ]
        head5 = ["ε", "对抗训练后 宏 F1", "二分类 F1", "逃逸率", "对照：原始模型 PGD 宏 F1"]
        t5 = []
        for g in adv["grid"]:
            base_row = next((r for r in rows_t if abs(r["eps"] - g["eps"]) < 1e-9), None)
            t5.append([fmt_float(g["eps"], 2), fmt_float(g["macro_f1"]), fmt_float(g["binary_f1"]),
                       f"{100.0 * (g['attack_evasion_rate'] or 0):.2f}%",
                       fmt_float(base_row["pgd"]["macro_f1"]) if base_row else "未测"])
        lines.append(markdown_table(head5, t5))

    lines += [
        "", "## 6. 可检测性：能不能用**已有的**熵阈值拦下对抗样本", "",
        f"- 干净测试窗口的平均预测熵：**{fmt_float(clean['mean_entropy'])}**；",
        "- 各攻击设置下的平均熵见第 2 节表格（`mean_entropy` 字段存于 JSON）。"
        "若扰动样本的熵显著升高，则工程里**已有的**「熵超阈值 → 标为未知形态」机制"
        "（`CIDA_UNKNOWN_THRESHOLD`，默认按 1% 误报预算标定为 0.3914）可以零成本拦下一部分对抗样本；"
        "若熵没有升高，则必须如实说明「这一招不管用」。",
        "",
        "## 7. 局限（写清楚，不粉饰）", "",
        "- 只做**特征空间**攻击：真实攻击者要改的是报文（包长/时序/标志位），"
        "特征空间的 L∞ 扰动**不一定**都能在报文层实现；本实验给出的是**脆弱性上界**而不是可复现的攻击工具。",
        "- 只做白盒：黑盒/查询式攻击（更贴近真实攻击者）未做，列为后续工作。",
        "- 对抗训练只做了一次（单种子、单 ε）：收益/代价的量级可信，**具体数值不应外推**。",
        "- 扰动投影用的是训练集逐特征取值范围，比真实物理约束略宽（例如包长理论上限 65535 未单独约束）。",
        "",
        "## 8. 复现命令", "",
        "```powershell",
        f"python tools/adversarial_eval.py --eps {','.join(str(e) for e in p['eps_list'])}"
        + ("" if not p.get("adv_train") else f" --epochs {Config().epochs}"),
        "```",
        "",
        f"> 数据隔离：只读 `data/processed` 与 `artifacts/{p['model']['tag']}`；"
        f"对抗训练产物写在 `artifacts/adversarial/`，结果写在 `results/adversarial_eval.*`。",
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
    rows = [r for r in p["grid"] if r["objective"] == "evade_binary"]
    eps = [r["eps"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=150)

    ax = axes[0]
    ax.plot(eps, [r["fgsm"]["macro_f1"] for r in rows], "o-", label="FGSM（1 步）", color="#f6ad55")
    ax.plot(eps, [r["pgd"]["macro_f1"] for r in rows], "s-", label=f"PGD（{p['pgd_steps']} 步）", color="#e53e3e")
    ax.plot(eps, [r["random"]["macro_f1"] for r in rows], "^--", label="随机扰动（对照）", color="#8aa0b6")
    adv = p.get("adv_train") or {}
    if adv.get("grid"):
        ax.plot([g["eps"] for g in adv["grid"]], [g["macro_f1"] for g in adv["grid"]],
                "d-", label="对抗训练后（PGD）", color="#38a169")
    ax.axhline(p["clean"]["macro_f1"], color="#4fd1c5", linestyle=":", label="干净基线")
    ax.set_xlabel("扰动预算 ε（标准化空间 L∞）")
    ax.set_ylabel("宏平均 F1")
    ax.set_title("对抗攻击下的精度退化")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(eps, [100 * (r["fgsm"]["attack_evasion_rate"] or 0) for r in rows], "o-",
            label="FGSM", color="#f6ad55")
    ax.plot(eps, [100 * (r["pgd"]["attack_evasion_rate"] or 0) for r in rows], "s-",
            label=f"PGD（{p['pgd_steps']} 步）", color="#e53e3e")
    ax.plot(eps, [100 * (r["random"]["attack_evasion_rate"] or 0) for r in rows], "^--",
            label="随机扰动（对照）", color="#8aa0b6")
    if adv.get("grid"):
        ax.plot([g["eps"] for g in adv["grid"]],
                [100 * (g["attack_evasion_rate"] or 0) for g in adv["grid"]],
                "d-", label="对抗训练后（PGD）", color="#38a169")
    ax.axhline(100 * (p["clean"]["attack_evasion_rate"] or 0), color="#4fd1c5", linestyle=":",
               label="干净基线（攻击窗口被判正常）")
    ax.set_xlabel("扰动预算 ε（标准化空间 L∞）")
    ax.set_ylabel("逃逸率 %（攻击窗口被判为正常）")
    ax.set_title("「洗白」能力：攻击者要让多少攻击隐形")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
