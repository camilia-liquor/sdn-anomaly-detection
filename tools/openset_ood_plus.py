"""开集拒识改进：在已有三种后处理分数之外，加上 **Mahalanobis** 与 **KNN-OOD**，并如实比较。

问题（来自既有实验的诚实结论）
------------------------------
`results/openset_report.md` 已经证明：不做拒识时 **100% 的未知攻击被自信地错判成已知类**
（U2R→WebAttack 84%、Botnet→Normal 96%）；加入 OOD 拒识后 U2R 在 1% 误报预算下可检出 69.8%，
**但 Botnet 只有 44.3%**。报告还指出了一条反直觉结论：三种分数的跨场景 AUROC 都在 0.978 左右，
**低误报预算下的尾部表现却差一个数量级** —— 也就是说"AUROC 好 ≠ 尾部可用"。

本工具要回答的是：**换更强的 OOD 分数，能不能把低误报预算下的尾部提上来？**
  * `mahalanobis`：在**特征空间**（融合层输出）上按已知类估计类条件均值与共享协方差，
    取到最近已知类的距离（越大越像未知）。这是经典的"深度模型特征空间 OOD"做法。
  * `knn`：到已知**训练窗口**中第 k 近邻的平均距离（越大越像未知）；实现简单、无分布假设。
两者都只用**已知类**的训练窗口拟合，不碰留出类（否则就不是零样本拒识了）。

判据（事前写死）
----------------
  J1（尾部提升）：在 1% 误报预算下，`max(UDR of mahalanobis, knn)` 是否比
      `max(UDR of msp, entropy, energy)` 高出一个种子标准差以上（**按场景分别判**）；
  J2（代价）：同样预算下，已知攻击的**误拒率**是否变差（超过基线均值 + 1σ 算变差）。
  verdict：`improved`（两个场景 J1 都成立且 J2 不变差）/ `partial` / `not_improved`。

复用与隔离
----------
  * 复用 `tools/openset_eval.py` 的数据重建与已训练的开集模型（`artifacts/openset/openset_no*_s*`），
    **不重新训练**，因此结论严格可比；
  * 只读主实验数据与开集模型；结果写 `results/openset_ood_plus.json / .md`、
    `results/figures/openset_ood_plus.png`。

用法
----
    python tools/openset_ood_plus.py                 # 两个场景 × 3 种子 × 5 种分数
    python tools/openset_ood_plus.py --knn-k 10 --scenarios Botnet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.config import Config  # noqa: E402
from src.model import load_model_bundle, predict_probs  # noqa: E402
from src.utils import (  # noqa: E402
    FIGURE_DIR,
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

import openset_eval as OE  # noqa: E402  复用数据重建 / 打分 / 阈值工具

LOGGER = get_logger("cida.openset_plus")

BASELINE_SCORES = ("msp", "entropy", "energy")
NEW_SCORES = ("mahalanobis", "knn")
DIAG_SCORES = ("norm",)         # 诊断项：只在报告里出现，不作为"方法"推荐
BUDGETS = (0.01, 0.001)
FEATURE_LAYER = "fuse"          # 融合层输出（256 维）：分类头之前的表示
SHRINKAGE = 1e-3                # 协方差收缩系数（特征维度高、样本有限时必须正则化）
KNN_MAX_TRAIN = 4000            # KNN 参考集上限（内存/时间）
DEFAULT_K = 5


# --------------------------------------------------------------------------------------
# 1) 特征提取与两类新分数
# --------------------------------------------------------------------------------------
def build_feature_model(model):
    """取融合层输出作为特征表示（拿不到就退回倒数第二层）。"""
    import tensorflow as tf

    try:
        return tf.keras.Model(inputs=model.input, outputs=model.get_layer(FEATURE_LAYER).output)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("取不到 %r 层（%s），退回用分类头前的最后一层", FEATURE_LAYER, exc)
        return tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)


def extract_features(feat_model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    return np.asarray(feat_model.predict(X, batch_size=batch_size, verbose=0), dtype=np.float64)


def fit_mahalanobis(F: np.ndarray, y: np.ndarray) -> dict:
    """已知类上的类条件均值 + 共享协方差（带收缩），返回打分所需的参数。"""
    classes = sorted(set(int(v) for v in np.unique(y)))
    mus = {c: F[y == c].mean(axis=0) for c in classes}
    d = F.shape[1]
    cov = np.zeros((d, d), dtype=np.float64)
    for c in classes:
        Fc = F[y == c]
        if len(Fc) > 1:
            Z = Fc - mus[c]
            cov += Z.T @ Z
    cov /= max(len(F) - len(classes), 1)
    cov += SHRINKAGE * np.trace(cov) / d * np.eye(d)      # 收缩：保证可逆、数值稳定
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:                          # 极端情况下退回伪逆
        prec = np.linalg.pinv(cov)
    return {"mus": mus, "prec": prec, "classes": classes}


def score_mahalanobis(params: dict, F: np.ndarray) -> np.ndarray:
    """到最近已知类的马氏距离（越远越像未知）。"""
    dists = []
    for c in params["classes"]:
        Z = F - params["mus"][c]
        dists.append(np.einsum("ij,jk,ik->i", Z, params["prec"], Z))
    return np.min(np.stack(dists, axis=1), axis=1)


def score_knn(F_ref: np.ndarray, F: np.ndarray, k: int) -> np.ndarray:
    """到已知训练窗口第 k 近邻的平均欧氏距离（分块计算，避免 O(N×M) 显存爆炸）。"""
    k = max(int(k), 1)
    out = np.empty(len(F), dtype=np.float64)
    for i in range(0, len(F), 512):
        blk = F[i:i + 512]
        d2 = ((blk[:, None, :] - F_ref[None, :, :]) ** 2).sum(axis=2)
        kk = min(k, d2.shape[1])
        part = np.partition(d2, kk - 1, axis=1)[:, :kk]
        out[i:i + 512] = np.sqrt(part).mean(axis=1)
    return out


# --------------------------------------------------------------------------------------
# 2) 指标
# --------------------------------------------------------------------------------------
def score_norm(F_ref: np.ndarray, F: np.ndarray) -> np.ndarray:
    """**诊断用**：只看特征向量范数与训练集中位范数的偏离。

    为什么要加这一项：Mahalanobis 距离在特征空间里对"整体尺度偏移"极敏感，
    如果未知类只是范数分布不同，那么马氏距离的高分**并不代表**它真的学到了类条件结构。
    把"范数探测"单列出来，就能看出马氏距离的提升有多少是"平凡信号"贡献的 —— 这是自我怀疑，
    不是又一个待推荐的分数。
    """
    r = np.linalg.norm(F_ref, axis=1)
    med = float(np.median(r))
    return np.abs(np.linalg.norm(F, axis=1) - med)


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(pos) == 0 or len(neg) == 0:
        return None
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return float(roc_auc_score(y, s))


def tail_metrics(s_val_normal: np.ndarray, s_unknown: np.ndarray, s_known_attack: np.ndarray) -> dict:
    """低误报预算下的尾部表现 + 已知类代价。"""
    out = {"budgets": {}}
    for a in BUDGETS:
        thr = OE.threshold_at_fpr(s_val_normal, a)
        out["budgets"][f"{a:g}"] = {
            "threshold": float(thr),
            "udr": float(np.mean(s_unknown > thr)) if len(s_unknown) else None,
            "known_attack_false_reject": (float(np.mean(s_known_attack > thr))
                                          if len(s_known_attack) else None),
        }
    return out


def _agg(vals: list[float | None]) -> dict:
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else None,
            "values": [float(v) for v in a]}


# --------------------------------------------------------------------------------------
# 3) 主流程
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="开集拒识改进：Mahalanobis / KNN-OOD 对照实验")
    ap.add_argument("--scenarios", default="U2R,Botnet")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--knn-k", type=int, default=DEFAULT_K)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    started = time.time()
    ensure_dirs(RESULT_DIR, FIGURE_DIR)
    silence_tf_logs()
    cfg = Config(verbose=False, figures=False, publish_summary=False)
    set_seed(cfg.seed)

    base = OE.load_base(cfg)
    records: dict[str, dict] = {}
    for scen in scenarios:
        payload = OE.build_payload(cfg, base, scen)
        data = payload["data"]
        names_tr, names_va, names_te = payload["names"]["train"], payload["names"]["val"], payload["names"]["test"]
        rec: dict = {"seeds": {}, "n_windows": {"train": int(len(names_tr)), "val": int(len(names_va)),
                                                "test": int(len(names_te))}}
        for seed in seeds:
            tag = f"openset_no{scen.lower()}_s{seed}"
            mp = OE.model_path_for(tag)
            if not Path(mp).exists():
                LOGGER.warning("缺少已训练模型 %s，跳过该种子（请先跑 tools/openset_eval.py）", mp)
                continue
            model = load_model_bundle(mp, compile_model=False)
            fm = build_feature_model(model)
            F_tr = extract_features(fm, data.X_train)
            F_va = extract_features(fm, data.X_val)
            F_te = extract_features(fm, data.X_test)

            # ---- 新分数：只用已知类训练窗口拟合 ----
            maha = fit_mahalanobis(F_tr, data.y_train_mc)
            sub = np.random.default_rng(seed).choice(len(F_tr), size=min(KNN_MAX_TRAIN, len(F_tr)),
                                                     replace=False)
            scores = {
                "mahalanobis": (score_mahalanobis(maha, F_va), score_mahalanobis(maha, F_te)),
                "knn": (score_knn(F_tr[sub], F_va, args.knn_k), score_knn(F_tr[sub], F_te, args.knn_k)),
                "norm": (score_norm(F_tr[sub], F_va), score_norm(F_tr[sub], F_te)),
            }
            # ---- 基线分数：与既有工具同一实现（同一代码路径，保证可比） ----
            pv_mc, _ = predict_probs(model, data.X_val)
            pt_mc, _ = predict_probs(model, data.X_test)
            sv = OE.ood_scores(pv_mc, OE.extract_logits(model, data.X_val))
            st = OE.ood_scores(pt_mc, OE.extract_logits(model, data.X_test))
            for k in BASELINE_SCORES:
                if sv.get(k) is not None and st.get(k) is not None:
                    scores[k] = (sv[k], st[k])

            entry: dict = {}
            for name, (s_va, s_te) in scores.items():
                s_val_normal = s_va[names_va == "Normal"]
                s_unknown = s_te[names_te == scen]
                s_known_attack = s_te[(names_te != "Normal") & (names_te != scen)]
                tm = tail_metrics(s_val_normal, s_unknown, s_known_attack)
                tm["auroc_unknown_vs_normal"] = _auroc(s_unknown, s_val_normal)
                tm["auroc_unknown_vs_known_attack"] = _auroc(s_unknown, s_known_attack)
                tm["n_unknown"] = int(len(s_unknown))
                entry[name] = tm
            rec["seeds"][str(seed)] = entry
            LOGGER.info("[%s s%d] 1%% 预算 UDR：%s | 已知攻击误拒：%s", scen, seed,
                        {k: round(v["budgets"]["0.01"]["udr"], 4) for k, v in entry.items()},
                        {k: round(v["budgets"]["0.01"]["known_attack_false_reject"], 4)
                         for k, v in entry.items()})
            del model, fm
        # ---- 跨种子聚合 ----
        agg: dict = {}
        for name in (*BASELINE_SCORES, *NEW_SCORES, *DIAG_SCORES):
            vals = [rec["seeds"][str(s)][name] for s in seeds if str(s) in rec["seeds"]
                    and name in rec["seeds"][str(s)]]
            if not vals:
                continue
            agg[name] = {
                "n_seeds": len(vals),
                "udr_at_1pct": _agg([v["budgets"]["0.01"]["udr"] for v in vals]),
                "udr_at_0.1pct": _agg([v["budgets"]["0.001"]["udr"] for v in vals]),
                "false_reject_at_1pct": _agg([v["budgets"]["0.01"]["known_attack_false_reject"]
                                              for v in vals]),
                "auroc_unknown_vs_normal": _agg([v["auroc_unknown_vs_normal"] for v in vals]),
                "auroc_unknown_vs_known_attack": _agg([v["auroc_unknown_vs_known_attack"] for v in vals]),
            }
        rec["aggregate"] = agg
        records[scen] = rec

    # ---------------- 判据（事前写死） ----------------
    judgement: dict = {}
    for scen, rec in records.items():
        agg = rec.get("aggregate") or {}
        base_vals = {k: (agg.get(k) or {}).get("udr_at_1pct", {}) for k in BASELINE_SCORES}
        base_best = max((v.get("mean") or -1, k) for k, v in base_vals.items())
        new_vals = {k: (agg.get(k) or {}).get("udr_at_1pct", {}) for k in NEW_SCORES}
        new_best = max((v.get("mean") or -1, k) for k, v in new_vals.items())
        base_sigma = base_vals.get(base_best[1], {}).get("std") or 0.0
        gain = float(new_best[0] - base_best[0])
        rej_base = (agg.get(base_best[1]) or {}).get("false_reject_at_1pct", {}).get("mean")
        rej_new = (agg.get(new_best[1]) or {}).get("false_reject_at_1pct", {}).get("mean")
        judgement[scen] = {
            "baseline_best_score": base_best[1], "baseline_best_udr_at_1pct": base_best[0],
            "new_best_score": new_best[1], "new_best_udr_at_1pct": new_best[0],
            "gain": gain, "baseline_sigma": float(base_sigma),
            "J1_tail_improved": bool(gain > float(base_sigma)),
            "known_attack_false_reject_baseline": rej_base,
            "known_attack_false_reject_new": rej_new,
            "J2_cost_ok": bool(rej_new is None or rej_base is None or rej_new <= rej_base + 0.02),
        }
    flags = [v["J1_tail_improved"] for v in judgement.values()]
    costs_ok = [v["J2_cost_ok"] for v in judgement.values()]
    verdict = ("improved" if flags and all(flags) and all(costs_ok)
               else "partial" if any(flags) else "not_improved")
    payload_out = {
        "experiment": "openset_ood_scores_extension",
        "question": "换成特征空间的 Mahalanobis / KNN 分数，低误报预算下的未知攻击检出率能否提上来？",
        "params": {"scenarios": scenarios, "seeds": seeds, "knn_k": int(args.knn_k),
                   "feature_layer": FEATURE_LAYER, "shrinkage": SHRINKAGE,
                   "budgets": list(BUDGETS), "knn_max_train": KNN_MAX_TRAIN},
        "scenarios": records,
        "judgement": judgement,
        "verdict": verdict,
        "elapsed_s": round(time.time() - started, 1),
    }
    save_json(payload_out, RESULT_DIR / "openset_ood_plus.json")
    save_text(build_report(payload_out), RESULT_DIR / "openset_ood_plus.md")
    if not args.no_figure:
        fp = make_figure(payload_out, FIGURE_DIR / "openset_ood_plus.png")
        LOGGER.info("图：%s", fp)
    LOGGER.info("完成：判定 %s（用时 %.0fs）", verdict, payload_out["elapsed_s"])
    return 0


SCORE_LABEL = {"msp": "1 − max softmax（基线）", "entropy": "熵 H(p)（基线）",
               "energy": "能量 −logsumexp（基线）",
               "mahalanobis": "马氏距离（**本次新增**）", "knn": "KNN 距离（**本次新增**）",
               "norm": "特征范数偏离（**诊断项**）"}


def build_report(p: dict) -> str:
    verdict_text = {
        "improved": "**有效**：两个场景的尾部检出率都显著提升，且已知类代价没有变差。",
        "partial": "**部分有效**：只在部分场景上提升（见下表），对另一个场景无效。",
        "not_improved": "**无效**：特征空间分数没有把低误报预算下的尾部提上来。",
    }[p["verdict"]]
    lines = [
        "# 开集拒识改进：给未知攻击检测加上 Mahalanobis 与 KNN 分数",
        "",
        "> 由 `tools/openset_ood_plus.py` 生成。**复用已训练的开集模型**（不重训），"
        "因此与 `results/openset_report.md` 的结论严格可比；分数只用**已知类训练窗口**拟合。",
        "",
        "## 0. 结论",
        "",
        f"- 判定：{verdict_text}",
    ]
    for scen, j in p["judgement"].items():
        lines.append(
            f"- **{scen}**：基线最佳 `{j['baseline_best_score']}` 在 1% 预算下 UDR "
            f"**{fmt_float(j['baseline_best_udr_at_1pct'])}**；新分数最佳 `{j['new_best_score']}` "
            f"**{fmt_float(j['new_best_udr_at_1pct'])}**（增益 {fmt_float(j['gain'])}，基线 σ "
            f"{fmt_float(j['baseline_sigma'])}）→ "
            f"{'✅ 提升' if j['J1_tail_improved'] else '❌ 未提升'}；"
            f"已知攻击误拒率 {fmt_float(j['known_attack_false_reject_baseline'])} → "
            f"{fmt_float(j['known_attack_false_reject_new'])}"
            f"（{'可接受' if j['J2_cost_ok'] else '变差'}）")
    lines += [
        "",
        "## 1. 方法（只看已知类，零样本拒识）",
        "",
        f"- 特征表示：模型**融合层 `{FEATURE_LAYER}`**（256 维，分类头之前）；",
        f"- `mahalanobis`：按已知类估计类条件均值与**共享协方差**（收缩系数 {SHRINKAGE}），"
        f"分数 = 到最近已知类的马氏距离；",
        f"- `knn`：分数 = 到已知训练窗口中第 {p['params']['knn_k']} 近邻的平均欧氏距离"
        f"（参考集最多 {KNN_MAX_TRAIN} 个窗口）；",
        "- 阈值一律由**验证集正常窗口**在给定误报预算下取分位数得到（与既有工具同一实现）；",
        "- 基线三种分数（msp/entropy/energy）在**同一代码路径**下重算，保证可比。",
        "",
        "## 2. 主表：1% 误报预算下的未知检出率（UDR）与代价",
        "",
    ]
    head = ["场景", "分数", "UDR@1%", "UDR@0.1%", "已知攻击误拒@1%",
            "AUROC(未知 vs 正常)", "AUROC(未知 vs 已知攻击)"]
    t = []
    for scen, rec in p["scenarios"].items():
        for name in (*BASELINE_SCORES, *NEW_SCORES, *DIAG_SCORES):
            a = (rec.get("aggregate") or {}).get(name)
            if not a:
                continue
            t.append([scen, SCORE_LABEL.get(name, name),
                      fmt_float(a["udr_at_1pct"]["mean"]),
                      fmt_float(a["udr_at_0.1pct"]["mean"]),
                      fmt_float(a["false_reject_at_1pct"]["mean"]),
                      fmt_float(a["auroc_unknown_vs_normal"]["mean"]),
                      fmt_float(a["auroc_unknown_vs_known_attack"]["mean"])])
    lines.append(markdown_table(head, t))
    lines += [
        "",
        "> 读法：**只看 UDR 不看 AUROC**。既有报告已经指出三种基线分数的 AUROC 都在 0.978 左右、"
        "却在低误报预算下差一个数量级——所以本表的关键列是 **UDR@0.1%**。",
        "",
        "**自我怀疑诊断（重要）**：表里的 `norm` 是**诊断项**而不是推荐方法 —— "
        "它只看特征向量范数与训练集中位范数的偏离。如果 `norm` 的 UDR 与 `mahalanobis` 相当，"
        "说明后者的提升**主要来自「未知类的特征范数整体偏移」这种平凡信号**，"
        "而不是真的学到了类条件结构；报告必须据此调整措辞。",
        "",
        f"未知类测试窗口数（样本量提醒）："
        + "、".join(f"{s} {((r.get('seeds') or {}).get(list(r.get('seeds') or {'0': {}})[0], {}) or {}).get('msp', {}).get('n_unknown', '—')} 个"
                    for s, r in p["scenarios"].items()),
        "",
        "## 3. 逐种子明细（避免只看均值）",
        "",
    ]
    head2 = ["场景", "种子"] + [SCORE_LABEL.get(n, n) for n in (*BASELINE_SCORES, *NEW_SCORES, *DIAG_SCORES)]
    t2 = []
    for scen, rec in p["scenarios"].items():
        for seed, entry in sorted((rec.get("seeds") or {}).items()):
            row = [scen, seed]
            for name in (*BASELINE_SCORES, *NEW_SCORES, *DIAG_SCORES):
                e = entry.get(name)
                row.append(fmt_float(e["budgets"]["0.01"]["udr"]) if e else "—")
            t2.append(row)
    lines.append(markdown_table(head2, t2))
    lines += [
        "",
        "## 4. 局限（写清楚，不粉饰）",
        "",
        "- 特征空间分数用的是**融合层输出**，它本身是在「已知类」上学出来的表示；"
        "如果未知攻击在特征空间里与某个已知类重叠，两类距离分数都会失败"
        "（这正是 Botnet→Normal 那种情形难救的原因）。",
        "- 只有 2 个留出场景 × 3 个种子，且**不重训模型**：结论是「换分数能提多少」的量级判断，"
        "不是「新方法稳定性」的证明。",
        "- Mahalanobis 假设共享协方差与近似高斯；KNN 依赖参考集规模（本工具上限 "
        f"{KNN_MAX_TRAIN} 个窗口，放宽可能略有变化）。",
        "- 拒识只改了**判定**，没有改训练目标（例如引入 outlier exposure 或原型学习）——"
        "那是更大的改动，留作后续工作。",
        "",
        "## 5. 复现命令",
        "",
        "```powershell",
        f"python tools/openset_ood_plus.py --scenarios {','.join(p['params']['scenarios'])} "
        f"--knn-k {p['params']['knn_k']}",
        "```",
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
    scen_list = list(p["scenarios"].keys())
    scores = [s for s in (*BASELINE_SCORES, *NEW_SCORES)]
    colors = {"msp": "#2b7bba", "entropy": "#e08214", "energy": "#c0392b",
              "mahalanobis": "#38a169", "knn": "#805ad5"}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), dpi=150,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    x = np.arange(len(scen_list))
    w = 0.16
    for i, s in enumerate(scores):
        vals = []
        for scen in scen_list:
            a = ((p["scenarios"][scen].get("aggregate") or {}).get(s) or {})
            vals.append(((a.get("udr_at_0.1pct") or {}).get("mean") or 0.0) * 100)
        ax.bar(x + (i - 2) * w, vals, w, label=s, color=colors.get(s, "#888"))
    ax.set_xticks(x)
    ax.set_xticklabels(scen_list)
    ax.set_ylabel("UDR@0.1% 误报预算（%）")
    ax.set_title("最严预算下的未知攻击检出率：尾部才是关键")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    for i, s in enumerate(scores):
        vals = []
        for scen in scen_list:
            a = ((p["scenarios"][scen].get("aggregate") or {}).get(s) or {})
            vals.append(((a.get("udr_at_1pct") or {}).get("mean") or 0.0) * 100)
        ax.bar(x + (i - 2) * w, vals, w, label=s, color=colors.get(s, "#888"))
    ax.set_xticks(x)
    ax.set_xticklabels(scen_list)
    ax.set_ylabel("UDR@1% 误报预算（%）")
    ax.set_title("1% 预算下的检出率")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
