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
from .data_pipeline import PreparedData, prepare_dataset
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
    y_train = {"mc": data.y_train_mc, "bin": data.y_train_bin}
    y_val = {"mc": data.y_val_mc, "bin": data.y_val_bin}
    if isinstance(model.output, dict):
        y_train = {k: v for k, v in y_train.items() if k in model.output}
        y_val = {k: v for k, v in y_val.items() if k in model.output}

    sample_weight = None
    if cfg.use_class_weight:
        cw = data.class_weight()
        sw = np.asarray([cw[int(c)] for c in data.y_train_mc], dtype="float32")
        LOGGER.info("类别权重：%s", {data.class_names[k]: round(v, 3) for k, v in cw.items()})
        if isinstance(model.output, dict):
            sample_weight = {k: sw for k in model.output}
        else:
            sample_weight = sw

    callbacks = build_callbacks(cfg, ckpt_path, out_dir / "epoch_log.csv")
    started = time.time()
    try:
        history = model.fit(
            data.X_train, y_train,
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
            data.X_train, y_train,
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
        extra={"model_saved": save_ok, "checkpoint": str(ckpt_path)},
    )
    LOGGER.info("训练完成：%s", result.summary())
    return result


def load_history(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
