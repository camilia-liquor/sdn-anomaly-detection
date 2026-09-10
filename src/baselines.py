"""传统机器学习基线：评委一定会问"为什么不用随机森林 / 和 XGBoost 比呢"。

本模块把 sklearn 的经典模型包装成与 Keras 模型**同样的接口**，
因此可以直接复用 ``src/evaluate.py`` 里的全部评估流程（多分类 + 二分类 + 在线告警级），
保证"深度模型 vs 传统模型"是在完全相同的划分与指标下对比的。

支持：随机森林、极端随机树、逻辑回归、线性 SVM、KNN
（若环境里装了 xgboost / lightgbm 也会自动纳入）。

用法：
    python run_all.py --baselines                 # 一键跑 RF/ET/LR 并与主模型对比
    python tools/run_baselines.py                 # 也可单独跑
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import Config
from .utils import get_logger

LOGGER = get_logger("cida.baseline")

BASELINES = ("random_forest", "extra_trees", "logistic_regression", "svm", "knn")


def _make_estimator(name: str, seed: int, n_jobs: int = -1):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    name = name.lower()
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            n_jobs=n_jobs, random_state=seed, class_weight="balanced_subsample",
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200, n_jobs=n_jobs, random_state=seed, class_weight="balanced_subsample",
        )
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=400, n_jobs=n_jobs, random_state=seed, class_weight="balanced", C=1.0,
        )
    if name == "svm":
        return LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=3000)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=7, n_jobs=n_jobs, weights="distance")
    if name == "xgboost":
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.9, tree_method="hist", n_jobs=n_jobs,
                             random_state=seed, eval_metric="mlogloss")
    if name == "lightgbm":
        from lightgbm import LGBMClassifier  # type: ignore

        return LGBMClassifier(n_estimators=300, learning_rate=0.1, n_jobs=n_jobs, random_state=seed)
    raise ValueError(f"未知基线模型：{name}")


def _fit_estimator(est, X, y):
    """训练估计器；并行训练在受限环境（无命名管道/无多进程权限）会失败，自动回退单进程。"""
    try:
        est.fit(X, y)
        return est
    except Exception as exc:  # noqa: BLE001
        params = est.get_params() if hasattr(est, "get_params") else {}
        if "n_jobs" in params and params.get("n_jobs") not in (None, 1):
            from sklearn.base import clone

            LOGGER.warning("并行训练失败（%s），自动回退为单进程重试……", exc)
            est2 = clone(est)
            est2.set_params(n_jobs=1)
            est2.fit(X, y)
            return est2
        raise


def available_baselines() -> list[str]:
    names = list(BASELINES)
    for extra in ("xgboost", "lightgbm"):
        try:
            __import__(extra)
            names.append(extra)
        except Exception:
            pass
    return names


class SklearnModelWrapper:
    """把 sklearn 估计器包装成"像 Keras 模型"的对象，供统一评估流程调用。"""

    def __init__(self, estimator, name: str, n_classes: int, normal_id: int = 0):
        self.estimator = estimator
        self.name = f"baseline_{name}"
        self.n_classes = int(n_classes)
        self.normal_id = int(normal_id)
        self.cida_has_attention = False

    # --- 接口对齐 ---
    def fit(self, X, y, **kwargs):
        Xf = flatten(X)
        self.estimator = _fit_estimator(self.estimator, Xf, np.asarray(y).reshape(-1))
        return self

    def predict(self, X, batch_size: int = 512, verbose: int = 0) -> dict[str, Any]:
        Xf = flatten(X)
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(Xf)
            # 补齐缺失类别（某些类在训练子集中可能完全没有）
            classes = getattr(self.estimator, "classes_", np.arange(proba.shape[1]))
            full = np.zeros((len(proba), self.n_classes), dtype=np.float64)
            for j, c in enumerate(classes):
                full[:, int(c)] = proba[:, j]
            p_mc = full
        else:
            from scipy.special import softmax  # type: ignore

            dec = self.estimator.decision_function(Xf)
            p_mc = softmax(dec, axis=1) if dec.ndim > 1 else np.stack([1 - dec, dec], axis=1)
        p_attack = 1.0 - p_mc[:, self.normal_id]
        return {"mc": p_mc.astype("float32"), "bin": np.clip(p_attack, 1e-7, 1 - 1e-7)[:, None].astype("float32")}

    def count_params(self) -> int:
        """尽量给出一个有意义的"规模"数字：森林=树节点总数，线性模型=系数个数。"""
        est = self.estimator
        try:
            if hasattr(est, "estimators_"):
                total = 0
                for t in np.ravel(est.estimators_):
                    if hasattr(t, "tree_"):
                        total += int(t.tree_.node_count)
                if total:
                    return total
            if hasattr(est, "tree_"):
                return int(est.tree_.node_count)
            if hasattr(est, "coef_"):
                return int(np.size(est.coef_)) + int(np.size(getattr(est, "intercept_", [])))
        except Exception:
            pass
        for attr in ("n_estimators", "max_iter"):
            if hasattr(est, attr):
                return int(getattr(est, attr))
        return 1000

    def summary(self) -> dict:
        return {"name": self.name, "estimator": type(self.estimator).__name__}


def flatten(X) -> np.ndarray:
    """(N, T, F) → (N, T*F)，传统模型吃不了三维张量。"""
    X = np.asarray(X)
    if X.ndim == 3:
        return X.reshape(len(X), -1)
    return X


def train_baseline(name: str, data, cfg: Config) -> SklearnModelWrapper:
    """训练一个基线模型。"""
    import time

    kind = name.lower()
    est = _make_estimator(kind, cfg.seed, n_jobs=int(getattr(cfg, "baseline_n_jobs", -1)))
    wrapper = SklearnModelWrapper(est, kind, data.n_classes, normal_id=data.normal_class_id)
    Xf = flatten(data.X_train)
    LOGGER.info("训练基线 %s：输入 %s（展平后 %d 维）……", kind, data.X_train.shape, Xf.shape[1])
    t0 = time.time()
    wrapper.fit(data.X_train, data.y_train_mc)
    LOGGER.info("基线 %s 训练完成，耗时 %.1f 秒", kind, time.time() - t0)
    return wrapper


def run_baselines(data, cfg: Config, names: list[str] | None = None) -> dict[str, SklearnModelWrapper]:
    names = names or ["random_forest", "extra_trees", "logistic_regression"]
    out: dict[str, SklearnModelWrapper] = {}
    for n in names:
        try:
            out[n] = train_baseline(n, data, cfg)
        except ImportError as exc:
            LOGGER.warning("跳过基线 %s：%s", n, exc)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("基线 %s 训练失败：%s", n, exc)
    return out
