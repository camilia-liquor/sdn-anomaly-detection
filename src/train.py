"""训练流程：类别加权 + 回调 + 产物落盘，全流程可复现。

原始 demo 训练部分的问题（本模块逐一修正）
------------------------------------------
1. 只跑 ``epochs=10`` 且 ``EarlyStopping(monitor='loss')`` 监控的是**训练损失**，
   训练损失一直在降，等于早停永远不会触发；这里监控 ``val_loss`` 并 ``restore_best_weights``。
2. ``ReduceLROnPlateau(min_lr=0.001)`` > 初始学习率 0.0005，学习率永远不衰减；这里 min_lr < lr。
3. 没有类别权重：攻击样本远少于正常流量，模型会退化成"全都判成正常"，召回率极低；
   这里按逆频加权（并用 sample_weight 适配双头多任务模型）。
4. 没有固定随机种子、没有保存 scaler/编码器 → 结果不可复现、模型无法上线推理；
   这里全部落盘到 ``artifacts/``。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .data_pipeline import PreparedData, inverse_frequency_weights, prepare_dataset
from .model import build_model, load_model_bundle
from .utils import ARTIFACT_DIR, get_logger, save_json, set_seed

LOGGER = get_logger("cida.train")


@dataclass
class TrainResult:
    model: object
    history: dict
    model_path: Path
    weights_path: Path
    history_path: Path
    variant: str
    duration_s: float
    best_epoch: int
    best_val_loss: float
    n_params: int
    tag: str = ""
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        hist = self.history
        out = {
            "variant": self.variant,
            "tag": self.tag,
            "epochs_run": len(hist.get("loss", [])),
            "best_epoch": int(self.best_epoch),
            "best_val_loss": float(self.best_val_loss),
            "final_train_acc": float(hist.get("mc_accuracy", hist.get("accuracy", [float("nan")]))[-1]),
            "best_val_acc": float(np.nanmax(hist.get("val_mc_accuracy", hist.get("val_accuracy", [float("nan")])))),
            "n_params": int(self.n_params),
            "train_seconds": round(self.duration_s, 1),
            "model_path": str(self.model_path),
        }
        out.update(self.extra)
        return out


# --------------------------------------------------------------------------------------
# 回调
# --------------------------------------------------------------------------------------
def build_callbacks(cfg: Config, ckpt_path: Path, log_path: Path | None = None):
    from tensorflow.keras import callbacks

    cbs = [
        callbacks.ModelCheckpoint(
            filepath=str(ckpt_path), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=False, verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=cfg.early_stop_patience,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=cfg.lr_factor,
            patience=cfg.lr_patience, min_lr=min(cfg.min_learning_rate, cfg.learning_rate * 0.1),
            verbose=1,
        ),
        callbacks.TerminateOnNaN(),
    ]
    if log_path is not None:
        cbs.append(callbacks.CSVLogger(str(log_path), append=False))
    return cbs


# --------------------------------------------------------------------------------------
# 对抗训练增广（可选能力，默认关 —— 见 Config.adv_train_eps）
# --------------------------------------------------------------------------------------
ADV_OBJECTIVES = ("target_normal", "evade_binary")


def fgsm_perturb(model, X, *, eps: float, objective: str = "target_normal", normal_idx: int = 0,
                 batch_size: int = 512, lo=None, hi=None) -> np.ndarray:
    """单步 FGSM（L∞ 有界）：返回被推向「更像正常流量」的扰动样本。

    为什么在 ``src`` 内自己实现，而不是 import ``tools/adversarial_eval.py``：
    工具是**评测脚本**（依赖 CLI 参数、数据目录、报告产物），上线训练主流程反向依赖评测脚本，
    会让"能上线的路径"被"做实验的路径"绑住（改一次实验脚本就可能改坏训练）。
    这里只保留最小内核，且与工具语义逐项一致：
      ① 目标函数：``target_normal`` = 最大化 Normal 类概率（报告中被评测的威胁模型）；
         ``evade_binary`` = 最小化「是攻击」的概率（对照用）；
      ② 投影一：L∞ 球面 ``x0 ± eps``；
      ③ 投影二：训练集逐特征取值范围 ``[lo, hi]``（原始量纲的物理可行域，避免造出物理上不存在的流）。

    注意：这里**不做就地修改**，输入 ``X`` 不被改写（训练集可能被多处复用）。
    """
    import tensorflow as tf

    X0 = np.asarray(X, dtype=np.float32)
    eps = float(eps)
    if objective not in ADV_OBJECTIVES:
        raise ValueError(f"未知对抗目标 {objective}，可选 {ADV_OBJECTIVES}")
    if eps <= 0.0 or len(X0) == 0:
        return X0.copy()
    lo_t = None if lo is None else tf.constant(np.asarray(lo, dtype=np.float32))
    hi_t = None if hi is None else tf.constant(np.asarray(hi, dtype=np.float32))
    out = np.empty_like(X0)
    step = max(1, int(batch_size))
    for s in range(0, len(X0), step):
        xb = tf.convert_to_tensor(X0[s:s + step], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xb)
            pred = model(xb, training=False)
            if isinstance(pred, dict):
                p_mc, p_bin = pred["mc"], tf.reshape(pred["bin"], [-1])
            else:   # 只有多分类头的模型（如消融变体）
                p_mc = pred
                p_bin = 1.0 - tf.gather(pred, int(normal_idx), axis=1)
            if objective == "evade_binary":
                loss = tf.reduce_mean(p_bin)
            else:
                p_normal = tf.gather(p_mc, int(normal_idx), axis=1)
                loss = -tf.reduce_mean(tf.math.log(p_normal + 1e-12))
        grad = tape.gradient(loss, xb)
        adv = xb - eps * tf.sign(grad)
        adv = tf.clip_by_value(adv, xb - eps, xb + eps)
        if lo_t is not None and hi_t is not None:
            adv = tf.clip_by_value(adv, lo_t, hi_t)
        out[s:s + step] = np.asarray(adv)
    return out.astype(np.float32)


def adversarial_augment(model, X, y_mc, y_bin, *, eps: float, ratio: float = 0.5,
                        objective: str = "target_normal", normal_idx: int = 0,
                        batch_size: int = 512, seed: int = 0):
    """把对抗副本拼进训练集，返回 ``(X_aug, y_mc_aug, y_bin_aug, info)``。

    ``ratio`` = 对抗样本在**增广后**训练集里的占比（0.5 → 干净:对抗 = 1:1，与
    ``tools/adversarial_eval.py`` 的口径一致）。为 <0.5 时按固定种子随机抽子集——
    不能取"前 k 条"：时间序切分下训练集是按时间排的，取前缀会把某个攻击类整段抽掉或整段重复。
    标签按同一索引复制，因此**逐行配对关系不变**（多分类标签与二分类标签不会错位）。

    ``eps<=0`` 或 ``ratio<=0`` 时原样返回（``info["augmented"]=False``）——这正是默认配置下的行为：
    不复制样本、不动数据，主流程与既有产物 sha256 完全不变。
    """
    X = np.asarray(X, dtype=np.float32)
    y_mc = np.asarray(y_mc)
    y_bin = np.asarray(y_bin)
    n = int(len(X))
    eps, ratio = float(eps), float(ratio)
    info: dict = {"augmented": False, "eps": eps, "ratio": ratio, "objective": objective,
                  "n_clean": n, "n_adv": 0}
    if eps <= 0.0 or ratio <= 0.0 or n == 0:
        info["reason"] = "eps≤0 或 ratio≤0：不做增广（默认关，不改变主流程与产物）"
        return X, y_mc, y_bin, info

    ratio = min(0.9, max(0.0, ratio))
    k = int(round(n * ratio / (1.0 - ratio)))
    k = max(1, min(k, n))
    idx = np.arange(n) if k >= n else np.sort(np.random.default_rng(int(seed)).permutation(n)[:k])
    # 物理可行域 = 训练集自身逐特征取值范围（与 adversarial_eval 的投影口径一致）
    lo = X.reshape(-1, X.shape[-1]).min(axis=0)
    hi = X.reshape(-1, X.shape[-1]).max(axis=0)
    X_adv = fgsm_perturb(model, X[idx], eps=eps, objective=objective, normal_idx=normal_idx,
                         batch_size=batch_size, lo=lo, hi=hi)
    X_aug = np.concatenate([X, X_adv], axis=0).astype(np.float32)
    y_mc_aug = np.concatenate([y_mc, y_mc[idx]])
    y_bin_aug = np.concatenate([y_bin, y_bin[idx]])
    info.update({
        "augmented": True,
        "n_adv": int(len(X_adv)),
        "ratio": round(len(X_adv) / float(n + len(X_adv)), 4),
        "seed": int(seed),
        "projected_to_train_range": True,
    })
    return X_aug, y_mc_aug, y_bin_aug, info


# --------------------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------------------
def run_training(
    cfg: Config,
    data: PreparedData | None = None,
    *,
    variant: str | None = None,
    tag: str = "",
    epochs: int | None = None,
    artifact_dir: Path | None = None,
) -> TrainResult:
    """训练一个模型变体并保存全部产物。"""
    import tensorflow as tf  # noqa: F401  (确认依赖存在，并让日志里出现版本号)

    set_seed(cfg.seed)
    if data is None:
        data = prepare_dataset(cfg)

    variant = (variant or cfg.model_variant).lower()
    tag = tag or variant
    out_dir = Path(artifact_dir or ARTIFACT_DIR) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    weights_path = out_dir / "model.weights.h5"
    history_path = out_dir / "history.json"
    ckpt_path = out_dir / "best.keras"

    model = build_model(data.input_shape, data.n_classes, cfg, variant=variant)
    # 模型结构写进文件备查，而不是刷满整屏日志
    try:
        lines: list[str] = []
        model.summary(print_fn=lines.append)
        (out_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    n_epochs = int(epochs if epochs is not None else cfg.epochs)

    def _labels(y_mc, y_bin) -> dict:
        """按模型实际输出头裁剪标签字典（消融变体可能没有二分类头）。"""
        d = {"mc": y_mc, "bin": y_bin}
        if isinstance(model.output, dict):
            d = {k: v for k, v in d.items() if k in model.output}
        return d

    y_val = _labels(data.y_val_mc, data.y_val_bin)

    # ---------------- 对抗训练增广（默认关；见 Config.adv_train_eps） ----------------
    X_train = data.X_train
    y_mc_train = data.y_train_mc
    y_bin_train = data.y_train_bin
    adv_info: dict = {"augmented": False,
                      "eps": float(getattr(cfg, "adv_train_eps", 0.0) or 0.0)}
    adv_eps = float(getattr(cfg, "adv_train_eps", 0.0) or 0.0)
    if adv_eps > 0.0:
        # 先暖机再生成对抗样本：刚 build 出来的模型是随机权重，它的梯度方向没有意义
        #（对随机模型做 FGSM ≈ 加噪声）。暖机的轮数计入总轮数预算，保证总训练量不变。
        # 暖机段**不挂回调、不加类别权重**：它只为拿到一个有意义的梯度方向，不是交付产物；
        # 最优权重 / 早停 / 学习率衰减全部由后面的主训练段决定。
        warmup = max(1, int(getattr(cfg, "adv_train_warmup_epochs", 2) or 2))
        warmup = max(1, min(warmup, max(1, n_epochs // 3)))
        LOGGER.info("对抗训练开启：先干净训练 %d 轮暖机，再用 FGSM(ε=%.3f、目标=Normal) 生成对抗样本混合训练",
                    warmup, adv_eps)
        model.fit(X_train, _labels(y_mc_train, data.y_train_bin),
                  epochs=warmup, batch_size=cfg.batch_size, verbose=0)
        X_train, y_mc_train, y_bin_train, adv_info = adversarial_augment(
            model, X_train, y_mc_train, data.y_train_bin, eps=adv_eps,
            normal_idx=int(getattr(data, "normal_class_id", 0)), batch_size=cfg.batch_size,
            seed=int(cfg.seed),
        )
        n_epochs = max(1, n_epochs - warmup)
        adv_info["warmup_epochs"] = warmup
        LOGGER.info("对抗增广完成：干净 %d 条 + 对抗 %d 条（对抗占比 %.0f%%），主训练剩 %d 轮",
                    adv_info["n_clean"], adv_info["n_adv"], 100.0 * adv_info["ratio"], n_epochs)

    y_train = _labels(y_mc_train, y_bin_train)

    sample_weight = None
    if cfg.use_class_weight:
        # 逆频权重按**实际用于训练的那份标签**算（开启对抗增广时含对抗副本）：
        # 公式本身只有一处实现（src/data_pipeline.py::inverse_frequency_weights），
        # 避免"同一规则两处各算一次、后来只改一处"。
        cw = inverse_frequency_weights(y_mc_train, data.n_classes)
        sw = np.asarray([cw[int(c)] for c in y_mc_train], dtype="float32")
        LOGGER.info("类别权重：%s", {data.class_names[k]: round(v, 3) for k, v in cw.items()})
        if isinstance(model.output, dict):
            sample_weight = {k: sw for k in model.output}
        else:
            sample_weight = sw

    callbacks = build_callbacks(cfg, ckpt_path, out_dir / "epoch_log.csv")
    started = time.time()
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(data.X_val, y_val),
            epochs=n_epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=2 if cfg.verbose else 0,
        )
    except (TypeError, ValueError) as exc:
        # 某些 Keras 版本不接受 dict 形式的 sample_weight；退化为不加权，但保留其余流程
        LOGGER.warning("sample_weight 不被当前 Keras 接受（%s），改为不使用类别权重重跑。", exc)
        history = model.fit(
            X_train, y_train,
            validation_data=(data.X_val, y_val),
            epochs=n_epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=2 if cfg.verbose else 0,
        )
    duration = time.time() - started

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    val_loss_key = "val_loss" if "val_loss" in hist else list(hist)[-1]
    best_epoch = int(np.argmin(hist[val_loss_key])) if hist.get(val_loss_key) else 0
    best_val_loss = float(hist[val_loss_key][best_epoch]) if hist.get(val_loss_key) else float("nan")

    # 用验证集最优权重重新加载一次，确保落盘的是最优模型
    if ckpt_path.exists():
        try:
            model = load_model_bundle(ckpt_path, compile_model=False)
            LOGGER.info("已载入验证集最优权重（epoch %d）", best_epoch + 1)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("载入最优权重失败，沿用当前权重：%s", exc)

    save_ok = True
    try:
        model.save(model_path)
        model.save_weights(weights_path)
    except Exception as exc:  # pragma: no cover
        save_ok = False
        LOGGER.warning("模型保存失败：%s", exc)

    save_json(hist, history_path)
    save_json(cfg.to_dict(), out_dir / "config_used.json")
    save_json(data.summary(), out_dir / "data_summary.json")

    result = TrainResult(
        model=model, history=hist, model_path=model_path, weights_path=weights_path,
        history_path=history_path, variant=variant, duration_s=duration,
        best_epoch=best_epoch, best_val_loss=best_val_loss,
        n_params=int(model.count_params()), tag=tag,
        extra={"model_saved": save_ok, "checkpoint": str(ckpt_path),
               **({"adv_train": adv_info} if adv_info.get("augmented") else {})},
    )
    LOGGER.info("训练完成：%s", result.summary())
    return result


def load_history(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
