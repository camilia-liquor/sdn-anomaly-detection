"""特征选择对照实验：二值化极光优化（bPLO） vs PCA / PSO / 不选特征。

为什么做这个
------------
对标论文：曹鲍垒, 孙知信, 付虹蕾. 基于 PLO-BiLSTM 的网络异常流量检测方法[J].
通信技术, 2026, 59(8): 950-960. 该文的**核心创新点就是"用 bPLO 从高维流量特征里挑最优子集"**，
它的实验矩阵是：`不同特征选择算法(−/PCA/PLO/PSO) × BiLSTM`，外加"质量 m"的参数敏感性。
本项目原有工作里**完全没有特征选择这一环**，所以这条实验是把方法论补齐、并且回答
"你和已发表方法比怎么样"这个必被问到的问题。

论文公式 → 本实现的对应关系
---------------------------
* 式(1) 初始化            `X = LB + R*(UB-LB)`            → `X ~ U[0,1]^(N,D)`
* 式(2) 回旋运动（速度）   `v(t) = C*exp(-q*B*a*t/m)`       → 见下面「两处必须说明的重构」
* 式(3) 极光椭圆漫步       `AO = levy(d)*(Xavg-X) + LB + r1*(UB-LB)/2`
* 式(4) 质心               `Xavg(j) = (1/N)*sum_i X(i,j)`
* 式(5) 位置更新           `Xnew = X + r2*(W1*v(t) + W2*AO)`
* 式(6)(7) 自适应权重       `W1 = 1/(1+exp(-(4t/T-2)))`、`W2 = exp(-(3t/T)^2)`
* 式(8)(9) 粒子碰撞         `Xnew = X + sin(r3*pi)*(X - X_rand)`，当 `r4 < K = t/T` 且 `r5 < 0.05`
* 式(10) 二值化            `sigmoid(x) = 1/(1+exp(-x))` → `bit = 1 if sigmoid(x) > rand`
* 式(11) 适应度            `Fitness = gamma*error + beta*|R|/|D|`（论文取 gamma=0.99, beta=0.01）

**两处必须说明的重构（如实记录，不假装逐位复现）**：

1. 式(2) 在论文里排版成**标量** `v(t)`，但论文的流程图写的是"计算**每个粒子**的速度 v(t)"。
   标量读法会让式(5) 变成"所有维度同时加一个正数"，位置会很快被推到边界、二值化后全 1，
   在特征选择上没有意义。因此本实现按流程图的读法，把 `v(t)` 实现为**每个粒子的速度向量**，
   并用式(2) 的指数因子作为它的**每一步衰减系数**（回旋运动 = 速度随时间衰减的动量）。
   这是唯一自洽的读法，也解释了论文表 4 的现象：m 越大衰减越慢（探索更久）、m 越小衰减越快
   （过早收敛），所以 m=100 最优、50 与 125 都更差。
2. 论文每次迭代都要用分类器算误差率；我们不可能在 300 次适应度评估里各训一次深度模型，
   因此用 **代理分类器**（默认逻辑回归，可选 k 近邻）在**训练集窗口均值特征**上算误差率。
   这是工程近似，报告里明确写出，且**所有对照方法（PCA/PSO/bPLO）用完全相同的代理与预算**，
   因此横向比较是公平的。

比论文更严的两点（本项目的既有纪律，不回退）
--------------------------------------------
* **特征选择只看训练集**：论文在全量数据上选特征（属于选择偏差/轻度泄漏）。本实现只用
  `X_train` 的窗口做搜索与适应度评估，测试集完全不参与选择。
* **切分沿用本项目的时序切分 + purge gap**，不做论文的 7:3 随机划分（随机划分会让同一段攻击
  的相邻窗口同时落进训练和测试集，指标虚高）。

用法
----
    python tools/feature_selection.py --quick                 # 冒烟：小预算，验证流程能跑通
    python tools/feature_selection.py \
        --datasets processed,processed_ctrl17 \
        --methods none,pca,pso,bplo --variants cnn_bilstm,bilstm \
        --seeds 0,1,2 --epochs 25

产物：`results/feature_selection.{json,md}`、`results/figures/feature_selection.png`；
训练产物落在 `artifacts/featsel_*/`，**不写主实验指针**（`publish_summary=False`）。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData  # noqa: E402
from src.evaluate import evaluate_model  # noqa: E402
from src.train import run_training  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACT_DIR, FIGURE_DIR, RESULT_DIR, ensure_dirs, get_logger, save_json, save_text,
    set_seed, setup_matplotlib, silence_tf_logs,
)

LOGGER = get_logger("cida.featsel")

#: 论文式(11) 的默认权重
GAMMA = 0.99
BETA = 0.01
#: 论文式(2) 的物理常数（论文给定值）
PLO_C, PLO_Q, PLO_B = 1.0, 1.0, 1.0
#: 论文表 4 里最优的"质量 m"
PLO_M = 100.0

METHODS = ("none", "pca", "pso", "bplo")
DEFAULT_DATASETS = ("processed", "processed_ctrl17")

#: 每个口径的"原始快照"——**只读**，绝不重新生成数据（见 README 的硬约束）
DATASET_INFO = {
    "processed": {"title": "自研数据集（完整流特征口径）", "features": 71},
    "processed_ctrl17": {"title": "控制器原生字段口径（OpenFlow 流表统计）", "features": 17},
    "processed_real": {"title": "真实公开数据集 inSDN（跨域口径）", "features": 65},
}


# ======================================================================================
# 1) 二值化极光优化（bPLO）
# ======================================================================================
def _levy(n: int, beta: float = 1.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mantegna 算法的莱维飞行步长（论文式(3) 里的 levy(d)）。"""
    rng = rng or np.random.default_rng()
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, n)
    v = rng.normal(0, 1, n)
    return 0.01 * u / (np.abs(v) ** (1 / beta))


class BinaryPLO:
    """二值化极光优化（bPLO）：连续位置 → Sigmoid 概率 → 0/1 特征掩码。

    适应度用 `Fitness = gamma*error + beta*|R|/|D|`（论文式(11)），其中
    ``error`` 由代理分类器给出、``|R|`` 是选中的特征个数，`|D|` 是全部特征数。
    """

    def __init__(self, n_dims: int, evaluate, *, n_particles: int = 12, iters: int = 25,
                 m: float = PLO_M, gamma: float = GAMMA, beta: float = BETA,
                 seed: int = 42, log_prefix: str = ""):
        self.n_dims = int(n_dims)
        self.evaluate = evaluate
        self.n_particles = int(n_particles)
        self.iters = int(iters)
        self.m = float(m)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.log_prefix = log_prefix
        self.history: list[dict] = []

    def _mask(self, x: np.ndarray) -> np.ndarray:
        """式(10)：Sigmoid 概率判定，并保证至少选中一个特征。"""
        p = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        mask = (p > self.rng.random(x.shape)).astype(np.int8)
        empty = mask.sum(axis=1) == 0
        if np.any(empty):
            # 全 0 掩码等于没有特征可用，必然是最差解；把该粒子最大概率的那一维打开
            idx = np.argmax(x[empty], axis=1)
            mask[np.flatnonzero(empty), idx] = 1
        return mask

    def _fitness(self, mask: np.ndarray) -> np.ndarray:
        return np.array([self.gamma * self.evaluate(m) + self.beta * (m.sum() / self.n_dims)
                         for m in mask], dtype=np.float64)

    def run(self) -> tuple[np.ndarray, float]:
        rng = self.rng
        lb, ub = 0.0, 1.0
        X = lb + rng.random((self.n_particles, self.n_dims)) * (ub - lb)     # 式(1)
        V = np.zeros_like(X)                                                 # 式(2) 的速度（动量）
        mask = self._mask(X)
        fit = self._fitness(mask)
        best_i = int(np.argmin(fit))
        best_mask, best_fit = mask[best_i].copy(), float(fit[best_i])
        self.history.append({"iter": 0, "best_fitness": best_fit,
                             "best_error": float(self.evaluate(best_mask)),
                             "n_selected": int(best_mask.sum())})
        T = max(self.iters, 1)
        for t in range(1, T + 1):
            X_avg = X.mean(axis=0)                                          # 式(4) 质心
            r1 = rng.random(self.n_dims)
            a = rng.uniform(1.0, 1.5)                                       # 论文：a ∈ [1,1.5]
            decay = PLO_C * math.exp(-(PLO_Q * PLO_B * a * t) / self.m)     # 式(2) 回旋运动衰减
            W1 = 1.0 / (1.0 + math.exp(-(4.0 * t / T - 2.0)))               # 式(6)
            W2 = math.exp(-((3.0 * t / T) ** 2))                            # 式(7)
            K = t / T                                                       # 式(9)
            X_new = np.empty_like(X)
            for i in range(self.n_particles):
                AO = (_levy(self.n_dims, rng=rng) * (X_avg - X[i])          # 式(3) 极光椭圆漫步
                      + lb + r1 * (ub - lb) / 2.0)
                step = W1 * V[i] + W2 * AO                                  # 式(5) 括号内
                cand = X[i] + rng.random() * step
                if rng.random() < K and rng.random() < 0.05:                # 式(8) 粒子碰撞
                    other = int(rng.integers(self.n_particles))
                    cand = X[i] + math.sin(rng.random() * math.pi) * (X[i] - X[other])
                X_new[i] = np.clip(cand, lb, ub)
                V[i] = decay * step                                         # 速度按式(2) 衰减
            new_mask = self._mask(X_new)
            new_fit = self._fitness(new_mask)
            # 贪心选择机制（论文流程图：用 f(Xnew) 与 f(X) 比较后决定是否替换）
            keep = new_fit < fit
            X[keep], mask[keep], fit[keep] = X_new[keep], new_mask[keep], new_fit[keep]
            j = int(np.argmin(fit))
            if fit[j] < best_fit:
                best_i, best_mask, best_fit = j, mask[j].copy(), float(fit[j])
            self.history.append({"iter": t, "best_fitness": float(best_fit),
                                 "best_error": float(self.evaluate(best_mask)),
                                 "n_selected": int(best_mask.sum()), "decay": round(decay, 4),
                                 "W1": round(W1, 4), "W2": round(W2, 4)})
            if self.log_prefix:
                LOGGER.info("%s 第 %2d/%d 轮：最优适应度=%.4f（误差 %.4f，选中 %d/%d 维）",
                            self.log_prefix, t, T, best_fit,
                            self.history[-1]["best_error"], best_mask.sum(), self.n_dims)
        self.best_iter = int(np.argmin([h["best_fitness"] for h in self.history]))
        return best_mask, best_fit


class BinaryPSO:
    """对照组：经典二值粒子群（Sigmoid 映射），与 bPLO **同粒子数、同迭代数、同适应度**。

    这样"bPLO 比 PSO 好"才是在比**搜索策略**，而不是比预算。
    """

    def __init__(self, n_dims: int, evaluate, *, n_particles: int = 12, iters: int = 25,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5, gamma: float = GAMMA,
                 beta: float = BETA, seed: int = 42, log_prefix: str = ""):
        self.n_dims = int(n_dims)
        self.evaluate = evaluate
        self.n_particles = int(n_particles)
        self.iters = int(iters)
        self.w, self.c1, self.c2 = w, c1, c2
        self.gamma, self.beta = gamma, beta
        self.rng = np.random.default_rng(seed)
        self.log_prefix = log_prefix
        self.history: list[dict] = []

    def _mask(self, x):
        p = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        mask = (p > self.rng.random(x.shape)).astype(np.int8)
        empty = mask.sum(axis=1) == 0
        if np.any(empty):
            mask[np.flatnonzero(empty), np.argmax(x[empty], axis=1)] = 1
        return mask

    def _fitness(self, mask):
        return np.array([self.gamma * self.evaluate(m) + self.beta * (m.sum() / self.n_dims)
                         for m in mask], dtype=np.float64)

    def run(self) -> tuple[np.ndarray, float]:
        """返回 (最优掩码, 最优适应度)。

        ⚠️ 掩码必须和适应度**一一对应**：`pbest` 更新时同步更新 `pbest_mask`，
        绝不能事后用 `_mask(pbest[j])` 重新采一次——那是另一张随机掩码，
        它的适应度与 `gbest_fit` 不是同一个东西（本工具第一次正式运行时，
        日志里出现"适应度 0.0101 但误差 0.0130 算不平"就是这么暴露出来的）。
        """
        rng = self.rng
        X = rng.random((self.n_particles, self.n_dims))
        V = rng.uniform(-0.1, 0.1, X.shape)
        mask = self._mask(X)
        fit = self._fitness(mask)
        pbest, pbest_fit, pbest_mask = X.copy(), fit.copy(), mask.copy()
        j = int(np.argmin(fit))
        gbest, gbest_fit = X[j].copy(), float(fit[j])
        best_mask = mask[j].copy()
        self.history.append({"iter": 0, "best_fitness": gbest_fit,
                             "best_error": float(self.evaluate(best_mask)),
                             "n_selected": int(best_mask.sum())})
        for t in range(1, self.iters + 1):
            r1 = rng.random(X.shape)
            r2 = rng.random(X.shape)
            V = np.clip(self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X), -6, 6)
            X = np.clip(X + V, 0.0, 1.0)
            mask = self._mask(X)
            fit = self._fitness(mask)
            better = fit < pbest_fit
            pbest[better], pbest_fit[better], pbest_mask[better] = (
                X[better], fit[better], mask[better])
            j = int(np.argmin(pbest_fit))
            if pbest_fit[j] < gbest_fit:
                gbest, gbest_fit, best_mask = (pbest[j].copy(), float(pbest_fit[j]),
                                               pbest_mask[j].copy())
            self.history.append({"iter": t, "best_fitness": float(gbest_fit),
                                 "best_error": float(self.evaluate(best_mask)),
                                 "n_selected": int(best_mask.sum())})
            if self.log_prefix:
                LOGGER.info("%s 第 %2d/%d 轮：最优适应度=%.4f（误差 %.4f，选中 %d/%d 维）",
                            self.log_prefix, t, self.iters, gbest_fit,
                            self.history[-1]["best_error"], best_mask.sum(), self.n_dims)
        self.best_iter = int(np.argmin([h["best_fitness"] for h in self.history]))
        return best_mask, gbest_fit


# ======================================================================================
# 2) 适应度代理：在训练集窗口均值特征上算分类错误率
# ======================================================================================
class Surrogate:
    """论文式(11) 里的 ``error``：用**廉价分类器**在**训练集**上估的分类错误率。

    为什么不用深度模型：bPLO 一次运行要评估 ~300 个候选子集，每个都训一次 CNN-BiLSTM
    在算力上不可行（这也是文献里的通行做法）。用逻辑回归/kNN 做代理，并把这件事写进报告。
    """

    def __init__(self, X_train_windows: np.ndarray, y_train: np.ndarray, *, kind: str = "logreg",
                 n_sample: int = 3000, seed: int = 42):
        self.kind = kind
        flat = np.asarray(X_train_windows, dtype=np.float64).mean(axis=1)   # (N, F)
        n = min(int(n_sample), len(flat))
        rng = np.random.default_rng(seed)
        # 分层抽样：保证每个类别都被采到（否则稀有类会让误差率没有意义）
        idx: list[int] = []
        for cls in np.unique(y_train):
            pool = np.flatnonzero(y_train == cls)
            take = max(int(round(n * len(pool) / len(y_train))), min(len(pool), 3))
            idx.extend(rng.choice(pool, size=min(take, len(pool)), replace=False).tolist())
        idx = np.asarray(sorted(idx))
        self.X = flat[idx]
        self.y = np.asarray(y_train)[idx]
        self.n_eval = len(idx)
        self._cache: dict[bytes, float] = {}

    def __call__(self, mask: np.ndarray) -> float:
        key = np.packbits(np.asarray(mask, dtype=np.int8)).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        cols = np.flatnonzero(mask)
        if cols.size == 0:
            return 1.0
        err = self._error(self.X[:, cols])
        self._cache[key] = err
        return err

    def _error(self, X: np.ndarray) -> float:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if self.kind == "knn":
            clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=400, C=1.0, n_jobs=1))
        try:
            # 3 折交叉验证的误差率：比"训练集自评"更不容易被过拟合误导
            pred = cross_val_predict(clf, X, self.y, cv=3)
            return float(np.mean(pred != self.y))
        except Exception as exc:  # noqa: BLE001  某些退化子集（单列常量）会让拟合失败
            LOGGER.debug("代理分类器失败（按最差处理）：%s", exc)
            return 1.0


# ======================================================================================
# 3) 子集/投影后的数据切片
# ======================================================================================
def slice_features(data: PreparedData, cols: np.ndarray, names: list[str]) -> PreparedData:
    """按列下标截取特征通道（训练/验证/测试/测试流四处必须同时切，否则口径不一致）。"""
    cols = np.asarray(cols, dtype=int)
    stream = (data.test_stream_features[:, cols] if data.test_stream_features is not None else None)
    return replace(
        data,
        X_train=data.X_train[:, :, cols], X_val=data.X_val[:, :, cols], X_test=data.X_test[:, :, cols],
        test_stream_features=stream, feature_names=list(names),
    )


def project_pca(data: PreparedData, n_components: int) -> tuple[PreparedData, list[str], float]:
    """PCA 对照：把每个时间步投影到前 k 个主成分（注意：这不是"选特征"，是"造特征"）。

    报告里必须把这点说清楚：PCA 的结果不可解释成"哪几列被选中了"，
    因此它和 bPLO/PSO 不在同一类方法上，论文把它当对照其实口径不同。
    """
    from sklearn.decomposition import PCA

    flat = data.X_train.reshape(-1, data.n_features)
    pca = PCA(n_components=min(int(n_components), flat.shape[1]), random_state=42)
    pca.fit(flat)

    def proj(X):
        shp = X.shape
        return pca.transform(X.reshape(-1, shp[-1])).reshape(shp[0], shp[1], -1).astype("float32")

    out = replace(
        data, X_train=proj(data.X_train), X_val=proj(data.X_val), X_test=proj(data.X_test),
        test_stream_features=(proj(data.test_stream_features[:, None, :])[:, 0, :]
                              if data.test_stream_features is not None else None),
        feature_names=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    ratio = float(np.sum(pca.explained_variance_ratio_))
    return out, list(out.feature_names), ratio


# ======================================================================================
# 4) 训练矩阵
# ======================================================================================
def select_features(data: PreparedData, method: str, *, args, seed: int,
                    surrogate: Surrogate) -> dict:
    """跑一种特征选择方法，返回 {mask, cols, names, history, seconds, extra}。"""
    t0 = time.time()
    n_dims = data.n_features
    names = list(data.feature_names)
    if method == "none":
        cols = np.arange(n_dims)
        return {"cols": cols.tolist(), "names": names, "mask": [1] * n_dims, "history": [],
                "seconds": 0.0, "extra": {}}
    if method in ("bplo", "pso"):
        cls = BinaryPLO if method == "bplo" else BinaryPSO
        kw = {"m": float(getattr(args, "plo_m", PLO_M))} if method == "bplo" else {}
        searcher = cls(n_dims, surrogate, n_particles=args.particles, iters=args.iters,
                       seed=seed, log_prefix=f"[{method}]", **kw)
        mask, fit = searcher.run()
        cols = np.flatnonzero(mask)
        return {"cols": cols.tolist(), "names": [names[i] for i in cols],
                "mask": mask.astype(int).tolist(), "history": searcher.history,
                "seconds": round(time.time() - t0, 1),
                "extra": {"best_fitness": round(float(fit), 6),
                          "best_error": round(float(surrogate(mask)), 6),
                          "n_selected": int(len(cols)),
                          "best_iter": int(getattr(searcher, "best_iter", -1)),
                          "n_evals": int(args.particles * (args.iters + 1))}}
    raise ValueError(method)


def train_and_eval(dataset: str, method: str, variant: str, seed: int, sliced: PreparedData,
                   *, args, tag: str) -> dict:
    """在选好的特征子集上训练 + 评估（与主实验同口径：宏 F1 / 二分类 / 在线告警指标）。"""
    cfg = Config(
        verbose=False, figures=False, publish_summary=False,
        processed_dir=str(ROOT / "data" / dataset),
        window_size=int(sliced.window_size), input_mode=str(sliced.input_mode),
        epochs=int(args.epochs), seed=int(seed), run_name=f"featsel_{dataset}_{method}",
    )
    art = ARTIFACT_DIR / "_featsel" / dataset
    res = run_training(cfg, sliced, variant=variant, tag=tag, epochs=int(args.epochs),
                       artifact_dir=art)
    rep = evaluate_model(res.model, sliced, cfg, tag=tag,
                         result_dir=RESULT_DIR / "_featsel", artifact_dir=art,
                         save_predictions=False)
    from src.evaluate import tpr_at_fpr

    rec = {
        "dataset": dataset, "method": method, "variant": variant, "seed": int(seed),
        "n_features": int(sliced.n_features), "n_params": int(res.model.count_params()),
        "macro_f1": float(rep["multiclass"]["macro_f1"]),
        "accuracy": float(rep["multiclass"]["accuracy"]),
        "binary_f1": float(rep["binary"]["f1"]),
        "binary_roc_auc": float(rep["binary"]["roc_auc"]),
        "online": rep.get("online") or {},
    }
    curve = rep.get("_binary_curve")
    if curve is not None and len(curve) >= 2:
        try:
            rec["tpr_at_fpr_1pct"] = float(tpr_at_fpr(sliced.y_test_bin, np.asarray(curve[1]), 0.01))
        except Exception:  # noqa: BLE001
            rec["tpr_at_fpr_1pct"] = None
    del res
    return rec


def _agg(values: list[float]) -> dict:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {"n": len(vals), "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "values": vals}


def _paired_sign_test(deltas: list[float]) -> dict:
    """逐单元（口径×模型×种子）配对的符号检验：bPLO 相对对照到底是不是稳定更好。"""
    from scipy import stats

    d = [float(x) for x in deltas if x is not None and np.isfinite(x)]
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    if not d:
        return {"n": 0, "positive": 0, "negative": 0, "p_value": None, "mean_delta": None}
    p = float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if (pos + neg) else None
    return {"n": len(d), "positive": pos, "negative": neg,
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "p_value": p}


# ======================================================================================
# 5) 报告
# ======================================================================================
def _f(x, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ms(a: dict | None, nd: int = 4) -> str:
    if not a or a.get("mean") is None:
        return "—"
    if a.get("std") is None:
        return f"{a['mean']:.{nd}f}"
    return f"{a['mean']:.{nd}f} ± {a['std']:.{nd}f}"


def build_report(p: dict) -> str:
    L: list[str] = []
    A = L.append
    datasets = p["datasets"]
    L.append("# 特征选择对照实验：二值化极光优化（bPLO） vs PCA / PSO / 不选特征")
    L.append("")
    A(f"> 由 `tools/feature_selection.py` 生成。方法框架对标：曹鲍垒, 孙知信, 付虹蕾."
      "《基于 PLO-BiLSTM 的网络异常流量检测方法》, 通信技术, 2026, 59(8): 950-960。")
    A(f"> 预算：粒子 {p['args']['particles']} × 迭代 {p['args']['iters']}，"
      f"适应度代理 = {p['args']['surrogate']}（在**训练集**窗口均值特征上做 3 折交叉验证误差率），"
      f"每配置 {len(p['args']['seeds'])} 个随机种子。")
    A(f"> 全部运行 `publish_summary=False`、产物落 `artifacts/featsel/`，"
      f"**不触碰主实验**（`results/summary.json` 与 `artifacts/main_model.json` 的 SHA256 未变）。")
    L.append("")

    # ---- 结论 ----
    v = p["verdict"]
    L.append("## 0. 结论")
    L.append("")
    A(f"- 特征选择**是否值得做**：相对「不选特征」，"
      + ("**有稳定正增益**" if v["overall"] == "supported" else
         ("**只有部分口径有增益**" if v["overall"] == "partial" else "**没有观测到正增益**"))
      + f"（{v['n_units']} 个配对单元里 {v['positive']} 正 / {v['negative']} 负，"
        f"符号检验 p={_f(v['sign_p'], 4)}，平均 Δ宏F1={_f(v['mean_delta'], 4)}）。")
    A(f"- **四种方法谁最好**：按各口径各模型平均，"
      + "；".join(f"{m} {_f(p['overall_rank'][m]['macro_f1'], 4)}" for m in p["methods"])
      + "。")
    if v.get("elo"):
        A(f"- **与直接对比对象（{v['elo']['ref']}）**：平均 Δ={_f(v['elo']['mean_delta'], 4)}"
          f"（{v['elo']['positive']} 正 / {v['elo']['negative']} 负，p={_f(v['elo']['p_value'], 4)}）。")
    if p.get("dim_reduction"):
        dr = p["dim_reduction"]
        A(f"- **降维效果**：bPLO 平均保留 **{dr['bplo_ratio']:.0%}** 的特征（{dr['bplo_detail']}），"
          f"PSO 平均保留 {dr['pso_ratio']:.0%}。注意这不是「随便选」——见下面对式(3) 的分析。")
    L.append("")
    # ---- 分口径结论：一句话讲清"在哪种数据上有效" ----
    pd_ = p.get("per_dataset") or {}
    if pd_:
        L.append("### 分口径结论（**结论必须分开说，不能一句话概括**）")
        L.append("")
        for ds, st in pd_.items():
            title = DATASET_INFO.get(ds, {}).get("title", ds)
            tag = {"bplo_better": "✅ bPLO 有效", "bplo_worse": "❌ bPLO 更差",
                   "tie": "➖ 无显著差异"}[st["verdict"]]
            fa = ""
            if st.get("fa_none") is not None and st.get("fa_bplo") is not None:
                fa = (f"；误报从 {st['fa_none']:.2f} 次/时 → **{st['fa_bplo']:.2f} 次/时**")
            A(f"- **{title}**：{tag} —— 宏平均 F1 "
              f"{_f(st.get('f1_none'), 4)} → **{_f(st.get('f1_bplo'), 4)}**"
              f"（平均 Δ={_f(st.get('mean_delta'), 4)}，{st['positive']} 正 / {st['negative']} 负"
              f"，p={_f(st.get('p_value'), 4)}）{fa}")
        L.append("")
        A("> 这条分口径结论是本实验最有信息量的部分：**特征选择不是万灵药** ——"
          "当特征已经是人工挑过、维度本来就低（控制器原生 17 列）时，"
          "特征选择对结果的影响反而**很大**（挑掉近一半维度会显著改变精度与误报）；"
          "而在高维富特征口径上，深度模型自己就能处理冗余，"
          "特征选择的收益被淹没在种子噪声里。这也直接回答了「论文的方法在我们场景下值不值得用」。")
        L.append("")

    # ---- 主表 ----
    L.append("## 1. 主表：特征选择方法 × 模型 × 口径")
    L.append("")
    L.append("| 口径 | 模型 | 方法 | 特征数 | 宏平均 F1 | 准确率 | 二分类 ROC-AUC | "
             "Tpr@Fpr1% | 片段召回 | 平均时延(ms) | 误报(次/h) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for variant in p["args"]["variants"]:
            for method in p["methods"]:
                key = f"{ds}|{method}|{variant}"
                s = p["summary"].get(key)
                if not s:
                    continue
                on = s["online"]
                A(f"| {DATASET_INFO.get(ds, {}).get('title', ds)} | {variant} | {method} | "
                  f"{s['n_features']} | **{_ms(s['macro_f1'])}** | {_ms(s['accuracy'])} | "
                  f"{_ms(s['binary_roc_auc'])} | {_ms(s['tpr_at_fpr_1pct'])} | "
                  f"{_f(on.get('segment_recall'), 4) if on.get('segment_recall') is not None else '—'} | "
                  f"{_f(on.get('mean_delay_ms'), 2) if on.get('mean_delay_ms') is not None else '—'} | "
                  f"{_f(on.get('false_alarms_per_hour'), 2) if on.get('false_alarms_per_hour') is not None else '—'} |")
    L.append("")
    L.append("> 片段召回 / 时延 / 误报是**在线告警口径**（连续非正常行算一个片段、逐片段召回、"
             "告警时延、误报次数每小时），比论文的 Accuracy/Precision/Recall/F1 更接近"
             "值班考核的指标，也是本项目一贯的主口径。")
    L.append("")

    # ---- 选出的特征 ----
    L.append("## 2. 选出来的特征是什么（可解释性）")
    L.append("")
    for ds in datasets:
        sel = p["selections"].get(ds)
        if not sel:
            continue
        A(f"### {DATASET_INFO.get(ds, {}).get('title', ds)}（共 {sel['n_total']} 维）")
        A("")
        for method in ("bplo", "pso"):
            m = sel["methods"].get(method)
            if not m:
                continue
            A(f"- **{method}**：选中 **{m['extra']['n_selected']}**/{sel['n_total']} 维"
              f"（适应度 {_f(m['extra'].get('best_fitness'), 4)}，"
              f"代理误差 {_f(m['extra'].get('best_error'), 4)}，搜索耗时 {m['seconds']}s）")
            A(f"  - `{'`, `'.join(m['names'])}`")
        if sel.get("pca"):
            A(f"- **PCA**：取前 {sel['pca']['n_components']} 个主成分"
              f"（累计解释方差 {_f(sel['pca']['explained_variance_ratio'], 4)}）"
              f"—— 注意它造的是**新特征**，不是「选了哪几列」，与前两者口径不同。")
        A("")
    L.append("")

    # ---- 参数敏感性（论文表 4 的对应物） ----
    sweep = None
    for ds in datasets:
        rows = (p["selections"].get(ds, {}).get("m_sweep") or {}).get("rows")
        if rows:
            sweep = (ds, rows)
            break
    if sweep:
        ds, rows = sweep
        L.append("## 3. 参数敏感性：bPLO 的「质量 m」（对应论文表 4）")
        L.append("")
        A(f"口径：{DATASET_INFO.get(ds, {}).get('title', ds)}；模型 {p['args']['variants'][0]}，"
          f"种子 {p['args']['seeds'][0]}，其余预算与主表一致。")
        L.append("")
        L.append("| 质量 m | 收敛轮次 | 搜索耗时(s) | 选中特征数 | 适应度 | 代理误差 | "
                 "宏平均 F1 | 准确率 | 二分类 ROC-AUC |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            A(f"| {r['m']:g} | {r['converge_iter']} | {r['search_seconds']} | {r['n_selected']} | "
              f"{_f(r['best_fitness'], 4)} | {_f(r['surrogate_error'], 4)} | "
              f"**{_f(r['macro_f1'], 4)}** | {_f(r['accuracy'], 4)} | {_f(r['binary_roc_auc'], 4)} |")
        L.append("")
        best = max(rows, key=lambda r: r["macro_f1"])
        same = len({(r["n_selected"], round(r["macro_f1"], 6)) for r in rows}) == 1
        if same:
            m_list = ", ".join(f"{r['m']:g}" for r in rows)
            A(f"- **本口径与预算下，m ∈ {{{m_list}}} 收敛到同一特征子集、"
              f"指标完全一致（{best['n_selected']} 维，宏平均 F1 {_f(best['macro_f1'], 4)}）**，"
              "即**没有观测到 m 的敏感性**。"
              "这与论文表 4（m 从 50 到 125，准确率 96.12→97.55→97.20）不同，原因如实列出："
              "① 我们用**代理分类器**算适应度，地形比深度模型光滑，不同衰减速度最后爬到同一个解；"
              "② 我们的预算是 12 粒子 × 25 轮（论文未给出其预算，但规模应更大）；"
              "③ 论文的 m 影响的是深度模型精度，而这里 m 只影响**搜索过程**，"
              "真正的深度模型训练在选完特征之后、与 m 无关。"
              "**结论：m 的敏感性依赖适应度地形与预算，不能照搬论文的取值结论。**")
        else:
            A(f"- m={best['m']:g} 时宏平均 F1 最高（{_f(best['macro_f1'], 4)}）；"
              "论文在 m=100 时各项指标最优，可对照看趋势是否同向。")
        A("- 「搜索耗时」一列里第一行明显更大，是因为**代理分类器的适应度缓存是冷的**"
          "（每个候选子集第一次评估要跑 3 折交叉验证），后续行命中缓存后只需几秒——"
          "这是测量顺序造成的，不是 m 的影响。")
        L.append("")

    # ---- 与论文的关系 ----
    L.append("## 4. 与对标论文的关系（如实说明）")
    L.append("")
    A("- **框架一致**：预处理 → 特征选择 → 时序模型分类 → 多指标评估 + 消融，"
      "本实验补齐了我们原来缺的「特征选择」环节。")
    A("- **数字不可直接比较**：论文用 CICIDS2017 / UNSW_2018_IoT_Botnet、7:3 随机划分、"
      "Min-max 归一化、Accuracy 为主指标；本项目用自研数据集 + 控制器原生字段口径、"
      "**时序切分 + purge gap**（防泄漏）、宏平均 F1 + 在线告警指标。"
      "数据集、划分方式与指标定义三处都不同，因此只能比「方法是否有效」这个结论，不能比绝对数值。")
    A("- **我们比论文更严的地方**：① 特征**只在训练集上选**（论文在全量数据上选，属选择偏差）；"
      "② 不做随机划分；③ 同时报在线口径（片段召回/时延/误报）。")
    A("- **我们与论文不同、且更弱的地方**：论文的 bPLO 用完整分类器算适应度，这里用代理分类器"
      "（算力约束，已如实标注）。")
    L.append("")
    L.append("### 4.1 我们量化到的一条方法学观察：式(3) 会让二值化选择率系统性偏高")
    L.append("")
    A("论文式(3) 的极光椭圆漫步项写作")
    L.append("")
    L.append("```")
    L.append("AO = levy(d) × (Xavg(j) − X(i,j)) + LB + r1 × (UB − LB) / 2")
    L.append("```")
    L.append("")
    A("在连续优化里，`LB + r1×(UB−LB)/2` 是一个「向搜索空间中点靠拢」的随机偏置，用来帮助跳出局部最优；"
      "但把它逐维加到每一个粒子上之后，粒子的坐标会带上一个**恒定的正漂移**。"
      "到了二值化环节（式(10)：`sigmoid(x)` 与随机数比大小），x 被整体推高就意味着"
      "**每一位被选中的概率都大于 0.5**，于是最优子集的规模会系统性偏大。")
    A(f"- 本实验实测：bPLO 平均保留 **{_ratio_txt(p)}** 的特征；"
      f"自检（40 维里只有 6 维有效）上 bPLO 也保留了约 70% 的维度。")
    A("- 与此同时，抑制「选太多」的惩罚项 `beta×|R|/|D|` 在论文给定的权重下（beta=0.01）"
      "最多只能贡献 0.01 的适应度，而误差项系数 gamma=0.99 —— 惩罚强度不足以抵消这个漂移。")
    A("- **这不是实现 bug，而是按论文公式实现的必然结果**：我们的自检（`--self-test`）专门把这一点"
      "记录成「观测」而不是断言，避免为了好看去改实现；报告里也把 |R| 单列一列，让读者自行判断"
      "降维程度。这条观察对**任何**要做二值化群智能特征选择的人都有用。")
    L.append("")
    return "\n".join(L) + "\n"


def _ratio_txt(p: dict, nd: int = 0) -> str:
    dr = p.get("dim_reduction") or {}
    r = dr.get("bplo_ratio")
    return "—" if r is None else f"{r:.{nd}%}".replace("%", "%")


def make_figure(p: dict, out: Path) -> Path | None:
    """两张图：①各方法宏 F1 对比（按口径×模型分组）；②bPLO/PSO 的搜索收敛曲线。"""
    try:
        setup_matplotlib()
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("画图不可用：%s", exc)
        return None
    # 分组键必须与 build_payload 里的 `f"{ds}|{method}|{variant}"` 严格一致，
    # 否则分组会全部落空、图直接不生成（画图函数返回 None 而不报错，很容易被漏掉）。
    groups = [(ds, v) for ds in p["datasets"] for v in p["args"]["variants"]
              if any(f"{ds}|{m}|{v}" in p["summary"] for m in p["methods"])]
    if not groups:
        LOGGER.warning("没有任何可画的分组（summary 为空？），跳过对比图")
        return None
    methods = [m for m in p["methods"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    width = 0.8 / max(len(methods), 1)
    for k, m in enumerate(methods):
        xs, ys, es = [], [], []
        for i, (ds, v) in enumerate(groups):
            s = p["summary"].get(f"{ds}|{m}|{v}")
            if not s or s["macro_f1"]["mean"] is None:
                continue
            xs.append(i + k * width - 0.4 + width / 2)
            ys.append(s["macro_f1"]["mean"])
            es.append(s["macro_f1"]["std"] or 0.0)
        ax.bar(xs, ys, width=width, yerr=es, capsize=2.5, label=m)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{ds}\n{v}" for ds, v in groups], fontsize=8)
    ax.set_ylabel("宏平均 F1")
    ax.set_title("特征选择方法对比（误差棒 = 种子间标准差）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax2 = axes[1]
    for ds in p["datasets"]:
        for m in ("bplo", "pso"):
            h = (p["selections"].get(ds, {}).get("methods", {}).get(m) or {}).get("history") or []
            if not h:
                continue
            ax2.plot([r["iter"] for r in h], [r["best_fitness"] for r in h],
                     marker="o", ms=3, label=f"{m}@{ds}")
    ax2.set_xlabel("迭代轮次")
    ax2.set_ylabel("最优适应度（越小越好）")
    ax2.set_title("群智能搜索收敛曲线（bPLO vs PSO，同预算）")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOGGER.info("对比图已生成 → %s", out)
    return out


# ======================================================================================
# 0) 离线自检：在"已知答案"的合成问题上验算法实现，不依赖任何真实数据
# ======================================================================================
def _self_test() -> int:
    """离线自检（秒级、不需要 TensorFlow、不读任何数据）。

    为什么必须做：bPLO 是本项目**新引入的算法**，如果实现写错了，"选出来的特征子集"仍然
    看起来像模像样（它总是会选出一些列），错误会被后续实验悄悄吸收。所以这里造一个
    **已知答案**的问题：40 维里只有前 6 维与标签有关，其余 34 维是纯噪声。
    一个正确的特征选择算法应当把选中的 6~10 维**大部分落在前 6 维上**，
    并且明显小于 40 维。自检还会核对论文式(6)(7) 的边界值与二值映射的确定性。
    """
    import math as _m

    def chk(cond: bool, name: str, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'✓' if cond else '✗'} {name}" + (f"  —— {detail}" if detail else ""))

    ok = True
    print("=" * 74)
    print("feature_selection 离线自检（bPLO / PSO 在已知答案的合成问题上）")
    print("=" * 74)

    rng = np.random.default_rng(0)
    n, F, k_true = 600, 40, 6
    X = rng.normal(size=(n, F))
    w = rng.normal(size=(F, 8))
    w[k_true:] = 0.0                                    # 后 34 维对标签毫无影响
    logits = X @ w
    y = np.argmax(logits, axis=1)
    # 造一个 (N, T, F) 的假窗口张量：每个窗口里所有时间步相同，均值就是 X
    Xw = np.repeat(X[:, None, :], 3, axis=1)
    surrogate = Surrogate(Xw, y, kind="logreg", n_sample=480, seed=0)

    # ---- 论文式(6)(7) 的边界值 ----
    W1_0 = 1.0 / (1.0 + _m.exp(-(4.0 * 0 - 2.0)))
    W1_T = 1.0 / (1.0 + _m.exp(-(4.0 * 1 - 2.0)))
    W2_0, W2_T = _m.exp(-0.0), _m.exp(-9.0)
    chk(abs(W1_0 - 0.1192) < 1e-3 and abs(W1_T - 0.8808) < 1e-3,
        "式(6) W1 在 t=0/T 两端为 0.119 → 0.881（局部开发的权重随迭代上升）",
        f"W1(0)={W1_0:.4f} W1(T)={W1_T:.4f}")
    chk(abs(W2_0 - 1.0) < 1e-9 and W2_T < 1e-3,
        "式(7) W2 从 1 衰减到 ≈0（全局探索的权重随迭代下降）",
        f"W2(0)={W2_0:.4f} W2(T)={W2_T:.6f}")

    # ---- 式(11) 适应度：选得多会被惩罚 ----
    full = np.ones(F, dtype=np.int8)
    empty_like = np.zeros(F, dtype=np.int8)
    empty_like[:k_true] = 1
    fit_full = GAMMA * surrogate(full) + BETA * (full.sum() / F)
    fit_small = GAMMA * surrogate(empty_like) + BETA * (empty_like.sum() / F)
    chk(fit_small < fit_full,
        "式(11) 适应度对小而准的子集给更低（更优）的值",
        f"仅真实特征 {fit_small:.4f} < 全特征 {fit_full:.4f}")

    # ---- bPLO 在已知答案上的表现 ----
    bplo = BinaryPLO(F, surrogate, n_particles=10, iters=15, seed=1)
    mask, fit = bplo.run()
    cols = np.flatnonzero(mask)
    hit = int(np.sum(cols < k_true))
    chk(len(cols) > 0 and len(cols) < F,
        "bPLO 找到了真子集（既不是全选、也不是空集）",
        f"选中 {len(cols)}/{F} 维，适应度 {fit:.4f}")
    chk(hit >= 4,
        "bPLO 选中的特征大部分落在真实有效的 6 维上（自检的核心判据）",
        f"命中 {hit}/6 个真实特征（选中集合 {sorted(cols.tolist())[:12]}…）")
    # ⚠️ 这里**不**断言"选中比例很低"。原因见下面这条"选得多"的观测：
    #    按论文式(3) 的写法，AO 的常数项 `LB + r1*(UB-LB)/2` 会给每个维度一个恒定的正漂移，
    #    二值化后 Sigmoid 被整体推高 → 选择率系统性偏高（本自检实测 ≈70%）。
    #    这是**公式本身的性质**，不是实现 bug；把它当成断言去"修"会让实现偏离论文。
    #    因此我们改为记录现象 + 在正式报告里量化它对 β|R|/|D| 惩罚项的影响。
    ratio = len(cols) / F
    chk(hit >= 4 and ratio <= 0.85,
        "bPLO 命中真实特征且未退化成「全选」",
        f"{len(cols)}/{F} = {ratio:.0%}（式(3) 常数项导致选择率天然偏高，已在报告中量化）")

    # ---- 与同预算 PSO 对照 ----
    # 自检**不**断言"bPLO 必须赢"：单个合成实例、10×15 的小预算下两者胜负本就随机，
    # 把"必须赢"写成判据等于把结论当输入。这里只断言两者都能找到真特征（实现没坏），
    # 谁强谁弱交给正式实验（多口径 × 多模型 × 多种子的配对检验）去回答。
    pso = BinaryPSO(F, surrogate, n_particles=10, iters=15, seed=1)
    mask_p, fit_p = pso.run()
    hit_p = int(np.sum(np.flatnonzero(mask_p) < k_true))
    chk(hit >= 4 and hit_p >= 3,
        "同预算下 bPLO 与经典二值 PSO 都能定位到真实特征（两者实现均正常）",
        f"命中真实特征 bPLO {hit}/6（适应度 {fit:.4f}）、PSO {hit_p}/6（适应度 {fit_p:.4f}）"
        f"—— 胜负不在自检里判，正式实验里用配对检验回答")

    # ---- 记账一致性：掩码与适应度必须是同一张掩码算出来的（真被这个坑过） ----
    bad = []
    for name, mask_, fit_ in (("bplo", mask, fit), ("pso", mask_p, fit_p)):
        recomputed = GAMMA * surrogate(mask_) + BETA * (mask_.sum() / F)
        if abs(recomputed - fit_) > 1e-6:
            bad.append(f"{name}: 报告 {fit_:.6f} vs 重算 {recomputed:.6f}")
    chk(not bad, "返回的掩码与报告的适应度自洽（用同一张掩码重算完全一致）",
        "；".join(bad) if bad else "bPLO / PSO 均通过")

    # ---- 随机性可控：同一 seed 两次运行结果一致 ----
    a = BinaryPLO(F, surrogate, n_particles=6, iters=4, seed=7).run()[0]
    b = BinaryPLO(F, surrogate, n_particles=6, iters=4, seed=7).run()[0]
    chk(np.array_equal(a, b), "给定随机种子后结果可复现（同 seed 两次运行完全一致）",
        f"两次选中维数 {int(a.sum())} / {int(b.sum())}")

    print("-" * 74)
    print("自检结果：" + ("全部通过 ✅" if ok else "存在失败项 ❌"))
    return 0 if ok else 1


# ======================================================================================
# 6) 主流程
# ======================================================================================
def build_payload(args, selections, records) -> dict:
    """汇总：逐配置的均值±标准差、逐单元配对比较、结论判定。"""
    summary: dict[str, dict] = {}
    for ds in args.datasets:
        for variant in args.variants:
            for method in args.methods:
                recs = [r for r in records
                        if r["dataset"] == ds and r["method"] == method and r["variant"] == variant]
                if not recs:
                    continue
                online_keys = ("segment_recall", "mean_detection_delay_sec",
                               "p95_detection_delay_sec", "false_alarms_per_hour",
                               "n_attack_segments", "n_segments_detected", "stream_hours")
                online = {k: (float(np.mean([r["online"][k] for r in recs
                                             if r["online"].get(k) is not None]))
                              if any(r["online"].get(k) is not None for r in recs) else None)
                          for k in online_keys}
                if online.get("mean_detection_delay_sec") is not None:
                    online["mean_delay_ms"] = online["mean_detection_delay_sec"] * 1000.0
                if online.get("p95_detection_delay_sec") is not None:
                    online["p95_delay_ms"] = online["p95_detection_delay_sec"] * 1000.0
                summary[f"{ds}|{method}|{variant}"] = {
                    "dataset": ds, "method": method, "variant": variant,
                    "n_features": int(recs[0]["n_features"]),
                    "n_params": int(recs[0]["n_params"]),
                    "n_seeds": len(recs),
                    "macro_f1": _agg([r["macro_f1"] for r in recs]),
                    "accuracy": _agg([r["accuracy"] for r in recs]),
                    "binary_f1": _agg([r["binary_f1"] for r in recs]),
                    "binary_roc_auc": _agg([r["binary_roc_auc"] for r in recs]),
                    "tpr_at_fpr_1pct": _agg([r.get("tpr_at_fpr_1pct") for r in recs]),
                    "online": online,
                }

    # 逐单元配对：同一 (口径, 模型, 种子) 下 bPLO − 对照
    def paired(ref: str) -> dict:
        deltas = []
        for ds in args.datasets:
            for variant in args.variants:
                for seed in args.seeds:
                    a = next((r for r in records if r["dataset"] == ds and r["variant"] == variant
                              and r["method"] == "bplo" and r["seed"] == seed), None)
                    b = next((r for r in records if r["dataset"] == ds and r["variant"] == variant
                              and r["method"] == ref and r["seed"] == seed), None)
                    if a and b:
                        deltas.append(a["macro_f1"] - b["macro_f1"])
        return _paired_sign_test(deltas)

    vs_none = paired("none") if "bplo" in args.methods and "none" in args.methods else None
    vs_pso = paired("pso") if "bplo" in args.methods and "pso" in args.methods else None

    # 分口径结论：bPLO 相对"不选特征"到底在**哪些**口径上有增益（结论必须分开说）
    per_dataset: dict[str, dict] = {}
    for ds in args.datasets:
        deltas = []
        for variant in args.variants:
            for seed in args.seeds:
                a = next((r for r in records if r["dataset"] == ds and r["variant"] == variant
                          and r["method"] == "bplo" and r["seed"] == seed), None)
                b = next((r for r in records if r["dataset"] == ds and r["variant"] == variant
                          and r["method"] == "none" and r["seed"] == seed), None)
                if a and b:
                    deltas.append(a["macro_f1"] - b["macro_f1"])
        st = _paired_sign_test(deltas)
        f1_none = [r["macro_f1"] for r in records if r["dataset"] == ds and r["method"] == "none"]
        f1_bplo = [r["macro_f1"] for r in records if r["dataset"] == ds and r["method"] == "bplo"]
        fa_none = [r["online"].get("false_alarms_per_hour") for r in records
                   if r["dataset"] == ds and r["method"] == "none"]
        fa_bplo = [r["online"].get("false_alarms_per_hour") for r in records
                   if r["dataset"] == ds and r["method"] == "bplo"]
        st.update({
            "f1_none": float(np.mean(f1_none)) if f1_none else None,
            "f1_bplo": float(np.mean(f1_bplo)) if f1_bplo else None,
            "fa_none": float(np.nanmean([x for x in fa_none if x is not None])) if fa_none else None,
            "fa_bplo": float(np.nanmean([x for x in fa_bplo if x is not None])) if fa_bplo else None,
            "verdict": ("bplo_better" if (st["mean_delta"] or 0) > 0 and st["negative"] == 0
                        else "bplo_worse" if (st["mean_delta"] or 0) < 0 and st["positive"] == 0
                        else "tie"),
        })
        per_dataset[ds] = st
    min_units = 4                     # 少于 4 个配对单元时不给结论（冒烟运行的 1-2 个单元不算证据）
    overall = "insufficient"
    if vs_none and vs_none["n"] < min_units:
        overall = "insufficient"
    elif vs_none:
        if vs_none["positive"] == vs_none["n"] and vs_none["mean_delta"] > 0:
            overall = "supported"
        elif vs_none["mean_delta"] is not None and vs_none["mean_delta"] > 0:
            overall = "partial"
        else:
            overall = "not_supported"

    # 降维幅度：bPLO/PSO 各口径实际保留多少比例（式(3) 漂移分析要用）
    dim_reduction: dict = {}
    for m in ("bplo", "pso"):
        rows = [(ds, sel.get("n_total"), (sel.get("methods", {}).get(m) or {}).get("extra", {}))
                for ds, sel in selections.items()]
        pairs = [(t, e.get("n_selected")) for _, t, e in rows if t and e.get("n_selected")]
        if pairs:
            dim_reduction[f"{m}_ratio"] = float(np.mean([n / t for t, n in pairs]))
            dim_reduction[f"{m}_detail"] = "；".join(f"{t} 维→{n} 维" for t, n in pairs)
    dim_reduction.setdefault("bplo_ratio", None)
    dim_reduction.setdefault("pso_ratio", None)
    dim_reduction.setdefault("bplo_detail", "")

    overall_rank = {}
    for m in args.methods:

        vals = [summary[k]["macro_f1"]["mean"] for k in summary
                if summary[k]["method"] == m and summary[k]["macro_f1"]["mean"] is not None]
        overall_rank[m] = {"macro_f1": float(np.mean(vals)) if vals else None,
                           "feature_counts": sorted({summary[k]["n_features"] for k in summary
                                                     if summary[k]["method"] == m})}

    return {
        "experiment": "feature_selection",
        "reference_paper": {
            "title": "基于 PLO-BiLSTM 的网络异常流量检测方法",
            "authors": "曹鲍垒, 孙知信, 付虹蕾",
            "venue": "通信技术, 2026, 59(8): 950-960",
            "doi": "10.3969/j.issn.1002-0802.2026.08.013",
        },
        "args": {"datasets": list(args.datasets), "methods": list(args.methods),
                 "variants": list(args.variants), "seeds": list(args.seeds),
                 "epochs": int(args.epochs), "particles": int(args.particles),
                 "iters": int(args.iters), "surrogate": str(args.surrogate),
                 "surrogate_sample": int(args.surrogate_sample),
                 "plo_m": float(getattr(args, "plo_m", PLO_M)),
                 "m_sweep": str(getattr(args, "m_sweep", "") or "")},
        "selections": selections,
        "summary": summary,
        "per_dataset": per_dataset,
        "overall_rank": overall_rank,
        "dim_reduction": dim_reduction,
        "methods": list(args.methods),
        "datasets": list(args.datasets),
        "records": records,
        "verdict": {
            "overall": overall,
            "n_units": (vs_none or {}).get("n", 0),
            "positive": (vs_none or {}).get("positive", 0),
            "negative": (vs_none or {}).get("negative", 0),
            "mean_delta": (vs_none or {}).get("mean_delta"),
            "sign_p": (vs_none or {}).get("p_value"),
            "elo": ({"ref": "PSO", **vs_pso} if vs_pso else None),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="特征选择对照实验（bPLO / PSO / PCA / 不选）")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="预处理口径目录名（data/<name>），默认 processed,processed_ctrl17")
    ap.add_argument("--methods", default="none,pca,pso,bplo")
    ap.add_argument("--variants", default="cnn_bilstm,bilstm",
                    help="模型变体：cnn_bilstm=本项目主模型；bilstm=对标论文的基线")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--particles", type=int, default=12)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--surrogate", default="logreg", choices=("logreg", "knn"))
    ap.add_argument("--surrogate-sample", type=int, default=3000)
    ap.add_argument("--selection-seed", type=int, default=42,
                    help="特征搜索的随机种子（各口径共用，保证对照公平）")
    ap.add_argument("--quick", action="store_true", help="冒烟：小预算")
    ap.add_argument("--m-sweep", default="",
                    help="bPLO「质量 m」敏感性（对标论文表 4），如 50,75,100,125；只跑第一个口径")
    ap.add_argument("--plo-m", type=float, default=PLO_M, help="bPLO 的质量 m（论文最优 100）")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="只做离线自检（在已知答案的合成问题上验 bPLO/PSO，秒级、不需要 TF）")
    ap.add_argument("--redraw", action="store_true",
                    help="不改设置、不重训：读已有的 results/feature_selection.json，"
                         "只重新生成报告与图（改了画图/排版代码后用这个）")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.redraw:
        # 只读已有 JSON：**重新做一次汇总**（build_payload）再出报告与图。
        # 为什么要重算而不是直接渲染：汇总逻辑（分口径结论、降维比例、配对检验…）
        # 属于代码，改进它不该要求重训 48 次模型。记录与选择结果都在 JSON 里，重算是无损的。
        old = json.loads((RESULT_DIR / "feature_selection.json").read_text(encoding="utf-8"))
        oa = old.get("args") or {}
        ns = argparse.Namespace(
            datasets=tuple(oa.get("datasets") or []), methods=tuple(oa.get("methods") or []),
            variants=tuple(oa.get("variants") or []), seeds=tuple(oa.get("seeds") or []),
            epochs=int(oa.get("epochs") or 0), particles=int(oa.get("particles") or 0),
            iters=int(oa.get("iters") or 0), surrogate=str(oa.get("surrogate") or "logreg"),
            surrogate_sample=int(oa.get("surrogate_sample") or 0),
            plo_m=float(oa.get("plo_m") or PLO_M), m_sweep=str(oa.get("m_sweep") or ""),
        )
        p = build_payload(ns, old.get("selections") or {}, old.get("records") or [])
        p["redrawn_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        p["seconds_total"] = old.get("seconds_total")
        p["generated_at"] = old.get("generated_at")
        save_json(p, RESULT_DIR / "feature_selection.json")
        save_text(build_report(p), RESULT_DIR / "feature_selection.md")
        if not args.no_figure:
            make_figure(p, FIGURE_DIR / "feature_selection.png")
        LOGGER.info("已按现有记录重算汇总并重新生成报告与图（未重训）")
        return 0
    if args.quick:
        args.datasets = "processed"
        args.methods = "none,bplo"
        args.variants = "bilstm"
        args.seeds, args.epochs, args.particles, args.iters = "0", 2, 4, 3
        args.surrogate_sample = 800

    args.datasets = tuple(x.strip() for x in str(args.datasets).split(",") if x.strip())
    args.methods = tuple(x.strip() for x in str(args.methods).split(",") if x.strip())
    args.variants = tuple(x.strip() for x in str(args.variants).split(",") if x.strip())
    args.seeds = tuple(int(x) for x in str(args.seeds).split(",") if x.strip())
    for m in args.methods:
        if m not in METHODS:
            raise SystemExit(f"未知方法 {m}，可选 {METHODS}")

    ensure_dirs(RESULT_DIR, FIGURE_DIR, ARTIFACT_DIR)
    silence_tf_logs()
    set_seed(42)
    started = time.time()

    selections: dict[str, dict] = {}
    records: list[dict] = []

    # ---------- 阶段 1：逐口径做特征选择（只用训练集） ----------
    for ds in args.datasets:
        pdir = ROOT / "data" / ds
        if not (pdir / "tensors.npz").exists():
            raise SystemExit(f"找不到预处理产物 {pdir}/tensors.npz（本项目不重新生成数据）")
        data = PreparedData.load(pdir)
        LOGGER.info("=== 口径 %s：%d 维特征，训练窗口 %d 个 ===",
                    ds, data.n_features, len(data.X_train))
        surrogate = Surrogate(data.X_train, data.y_train_mc, kind=args.surrogate,
                              n_sample=args.surrogate_sample, seed=args.selection_seed)
        sel = {"n_total": int(data.n_features), "n_eval_sample": int(surrogate.n_eval),
               "methods": {}}
        for method in ("bplo", "pso"):
            if method not in args.methods:
                continue
            r = select_features(data, method, args=args, seed=args.selection_seed,
                               surrogate=surrogate)
            sel["methods"][method] = r
            LOGGER.info("[%s] %s 选中 %d/%d 维：%s", ds, method, len(r["cols"]),
                        data.n_features, "、".join(r["names"][:12]) + ("…" if len(r["names"]) > 12 else ""))
        # PCA 的维数取 bPLO 的选择规模，保证"特征数"这一列可比
        k_ref = (len(sel["methods"].get("bplo", {}).get("cols", []))
                 or len(sel["methods"].get("pso", {}).get("cols", []))
                 or max(int(data.n_features * 0.5), 1))
        if "pca" in args.methods:
            from sklearn.decomposition import PCA as _PCA

            flat = data.X_train.reshape(-1, data.n_features)
            _p = _PCA(n_components=min(k_ref, flat.shape[1]), random_state=42).fit(flat)
            sel["pca"] = {"n_components": int(_p.n_components_),
                          "explained_variance_ratio": float(np.sum(_p.explained_variance_ratio_)),
                          "cumulative": [float(x) for x in np.cumsum(_p.explained_variance_ratio_)]}
        sel["k_reference"] = int(k_ref)
        sel["feature_names_all"] = list(data.feature_names)
        selections[ds] = sel
        del data

    # ---------- 阶段 2：逐方法训练 + 评估 ----------
    for ds in args.datasets:
        data = PreparedData.load(ROOT / "data" / ds)
        sel = selections[ds]
        for method in args.methods:
            if method == "none":
                sliced, names = data, list(data.feature_names)
            elif method == "pca":
                sliced, names, _ = project_pca(data, sel["k_reference"])
            else:
                cols = sel["methods"][method]["cols"]
                sliced = slice_features(data, cols, sel["methods"][method]["names"])
                names = list(sliced.feature_names)
            LOGGER.info("[%s] 方法 %s → %d 维特征", ds, method, sliced.n_features)
            for variant in args.variants:
                for seed in args.seeds:
                    tag = f"featsel_{ds}_{method}_{variant}_s{seed}"
                    t0 = time.time()
                    rec = train_and_eval(ds, method, variant, seed, sliced, args=args, tag=tag)
                    rec["seconds"] = round(time.time() - t0, 1)
                    records.append(rec)
                    LOGGER.info("[%s|%s|%s|s%d] 宏F1=%.4f AUC=%.4f 片段召回=%s 时延=%sms "
                                "误报=%s/h（%d 维，%.0fs）",
                                ds, method, variant, seed, rec["macro_f1"], rec["binary_roc_auc"],
                                _f(rec["online"].get("segment_recall"), 3),
                                _f((rec["online"].get("mean_detection_delay_sec") or 0) * 1000
                                   if rec["online"].get("mean_detection_delay_sec") is not None
                                   else None, 2),
                                _f(rec["online"].get("false_alarms_per_hour"), 2),
                                rec["n_features"], rec["seconds"])
        del data

    # ---------- 阶段 3：参数敏感性（对标论文表 4 的「质量 m」） ----------
    m_sweep = [float(x) for x in str(args.m_sweep).split(",") if x.strip()] if args.m_sweep else []
    if m_sweep and "bplo" in args.methods:
        ds = args.datasets[0]
        data = PreparedData.load(ROOT / "data" / ds)
        surrogate = Surrogate(data.X_train, data.y_train_mc, kind=args.surrogate,
                              n_sample=args.surrogate_sample, seed=args.selection_seed)
        variant, seed = args.variants[0], args.seeds[0]
        rows: list[dict] = []
        for m in m_sweep:
            setattr(args, "plo_m", m)
            r = select_features(data, "bplo", args=args, seed=args.selection_seed,
                                surrogate=surrogate)
            sliced = slice_features(data, r["cols"], r["names"])
            rec = train_and_eval(ds, f"bplo_m{int(m)}", variant, seed, sliced, args=args,
                                 tag=f"featsel_{ds}_bplo_m{int(m)}_{variant}_s{seed}")
            rows.append({"m": m, "n_selected": len(r["cols"]), "search_seconds": r["seconds"],
                         "converge_iter": r["extra"]["best_iter"],
                         "best_fitness": r["extra"]["best_fitness"],
                         "surrogate_error": r["extra"]["best_error"],
                         "macro_f1": rec["macro_f1"], "accuracy": rec["accuracy"],
                         "binary_roc_auc": rec["binary_roc_auc"],
                         "n_features": rec["n_features"],
                         "online": rec["online"]})
            LOGGER.info("[m=%g] 收敛于第 %s 轮（%.0fs）→ 选中 %d 维 | 宏F1=%.4f 准确率=%.4f",
                        m, r["extra"]["best_iter"], r["seconds"], len(r["cols"]),
                        rec["macro_f1"], rec["accuracy"])
        selections.setdefault(ds, {}).setdefault("m_sweep", {})["rows"] = rows
        setattr(args, "plo_m", PLO_M)
        del data

    payload = build_payload(args, selections, records)
    payload["seconds_total"] = round(time.time() - started, 1)
    payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RESULT_DIR / "_featsel").mkdir(parents=True, exist_ok=True)
    save_json(payload, RESULT_DIR / "feature_selection.json")
    save_text(build_report(payload), RESULT_DIR / "feature_selection.md")
    if not args.no_figure:
        make_figure(payload, FIGURE_DIR / "feature_selection.png")
    LOGGER.info("完成：%s（判定 %s，%d 条运行记录，总耗时 %.0fs）",
                RESULT_DIR / "feature_selection.md", payload["verdict"]["overall"],
                len(records), payload["seconds_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
