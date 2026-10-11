#!/usr/bin/env python
"""把人工研判结论回流成**可训练的标注数据集**——"未知 → 已知"闭环的最后一公里。

为什么需要这个工具：复核人给出结论后，`results/review_feedback.jsonl` 里只有 verdict、
分数和源 IP，**没有特征**；特征在 `results/review_evidence.jsonl` 里。两者按 `review_id`
关联起来，才能拼成 (X, y) 训练样本。缺了这一步，"回流"永远只是话术。

用法：
    python tools/retrain_from_feedback.py                # 只构建数据集（快，不训练）
    python tools/retrain_from_feedback.py --retrain      # 再把数据集混入训练，产出候选模型
    python tools/retrain_from_feedback.py --retrain --epochs 20

安全约束（都写死在这里，不靠使用者自觉）：
  1. **绝不覆盖主模型**：候选模型落在 `artifacts/feedback/`，主模型与 `results/summary.json` 不受影响；
  2. **特征列序必须与模型完全一致**：顺序错了不会报错、只会静默训出垃圾模型，所以这里硬校验；
  3. **绝不使用证据里的真实标签**：演示/回放数据在流记录里带 `Label`（ground truth），
     用它当训练真值就是自欺欺人——本工具只认人工 verdict，并会把忽略掉的 Label 数量如实报出来；
  4. 没有已研判样本、或 verdict 里的类别不在模型类别表里，一律拒绝并说明原因。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import NORMAL_CANDIDATES  # noqa: E402
from src.utils import (ARTIFACT_DIR, PROCESSED_DIR, RESULT_DIR, get_logger, human_bytes,  # noqa: E402
                       load_json, resolve_dir, save_json, save_text)

LOGGER = get_logger("cida.retrain")
# 正常类别的别名以 `src/data_pipeline.NORMAL_CANDIDATES` 为**单一事实来源**。
# 本工程已有过两次"同一规则在多处各判一次、后来只改了一处"的事故，所以这里不再自己抄一份。
# 额外补上 CIC-IoT2023 的 "BenignTraffic"（该数据集标签即如此，pipeline 的候选表未含）。
NORMAL_ALIASES = tuple(NORMAL_CANDIDATES) + ("BenignTraffic",)


# --------------------------------------------------------------------------- 读取
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("跳过无法解析的一行（%s）：%s…", path.name, line[:60])
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _model_contract(cfg: Config) -> tuple[list[str], list[str], int]:
    """从预处理产物读取模型契约：特征名顺序、类别顺序、窗口长度。"""
    # 走 resolve_dir 与其它工具/服务保持一致：cfg.processed_dir 可能是空串（= 默认目录）
    meta_path = resolve_dir(cfg.processed_dir, PROCESSED_DIR) / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"缺少预处理产物 {meta_path}，先运行 python run_all.py")
    meta = load_json(meta_path)
    feats = list(meta.get("feature_names") or [])
    classes = list(meta.get("class_names") or [])
    win = int(meta.get("window_size") or cfg.window_size)
    if not feats or not classes:
        raise SystemExit(f"{meta_path} 里缺少 feature_names / class_names，无法确定模型契约")
    return feats, classes, win


# --------------------------------------------------------------------------- 构建
def build_feedback_dataset(evidence: list[dict], feedback: list[dict],
                           feature_names: list[str], class_names: list[str],
                           window_size: int) -> dict:
    """把证据与结论按 `review_id` 拼成 (X, y)。所有拒绝条件都在这里集中判定。"""
    feats = list(feature_names)
    classes = list(class_names)
    n_expect = int(window_size) * len(feats)

    normal_idx = next((i for i, c in enumerate(classes) if c in NORMAL_ALIASES), None)
    if normal_idx is None:
        raise SystemExit(f"模型类别表里找不到 Normal 类（{classes}），无法把 'normal' 结论映射成标签")

    ev_by_id: dict[int, dict] = {}
    for e in evidence:
        try:
            rid = int(e.get("review_id"))
        except (TypeError, ValueError):
            continue
        ev_by_id[rid] = e                      # 同一 review_id 后写的覆盖先写的

    X: list[np.ndarray] = []
    y: list[int] = []
    rids: list[int] = []
    used_labels = 0          # 证据里带真实标签的条数（**不使用**，只如实汇报）
    skipped: list[str] = []
    bad_class: list[str] = []

    for fb in feedback:
        try:
            rid = int(fb.get("review_id"))
        except (TypeError, ValueError):
            continue
        verdict = str(fb.get("verdict") or "").strip()
        # 先判结论是否合法，再去找证据：顺序反了会把"非法结论"误报成"缺证据"，
        # 把人引到错误的排查方向（这是测试抓出来的）。
        if verdict == "normal":
            label = normal_idx
        elif verdict.startswith("attack:"):
            cls = verdict.split(":", 1)[1].strip()
            if cls not in classes:
                bad_class.append(f"#{rid} attack:{cls}（不在模型类别表）")
                continue
            label = classes.index(cls)
        else:
            skipped.append(f"#{rid} verdict 非法：{verdict!r}")
            continue

        ev = ev_by_id.get(rid)
        if ev is None:
            skipped.append(f"#{rid} 有结论但缺证据（无法重训）")
            continue
        arr = np.asarray(ev.get("X") or [], dtype=np.float32).ravel()
        if arr.size != n_expect:
            skipped.append(f"#{rid} 向量长度 {arr.size} ≠ T×F={n_expect}")
            continue
        # 证据自带的 feature_names 必须与模型完全一致，否则列序错位会静默训出垃圾模型
        if list(ev.get("feature_names") or []) != feats:
            skipped.append(f"#{rid} 证据的特征名与当前模型不一致")
            continue
        # 真实标签检测：证据文件里随附的流级记录可能带 Label（回放/演示数据自带真值）。
        # 只统计、**绝不使用**——拿它当训练真值等于自欺欺人。
        flow_ev = ev.get("flow_evidence")
        if flow_ev is None:
            flow_ev = ev.get("evidence")      # 兼容早期只写 evidence 的记录
        if any(isinstance(r, dict) and r.get("Label") is not None for r in (flow_ev or [])):
            used_labels += 1

        X.append(arr.reshape(int(window_size), len(feats)))
        y.append(int(label))
        rids.append(rid)

    if not X:
        raise SystemExit(
            "没有任何可用的已研判样本，拒绝生成数据集。\n"
            "提示：先在大屏「复核台」对待研判队列给出结论（或在 /api/review 提交 verdict），"
            "确认 results/review_feedback.jsonl 与 results/review_evidence.jsonl 都有内容。")

    Xa = np.stack(X).astype(np.float32)
    ya = np.asarray(y, dtype=np.int64)
    y_bin = (ya != normal_idx).astype(np.float32)
    dist = {classes[i]: int((ya == i).sum()) for i in range(len(classes)) if int((ya == i).sum()) > 0}
    return {
        "X": Xa, "y_mc": ya, "y_bin": y_bin, "review_ids": np.asarray(rids, dtype=np.int64),
        "class_names": classes, "feature_names": feats, "window_size": int(window_size),
        "normal_index": int(normal_idx),
        "skipped": skipped, "bad_class": bad_class, "evidence_with_gt_label": used_labels,
        "distribution": dist,
    }


# --------------------------------------------------------------------------- 训练
def retrain(cfg: Config, ds: dict, epochs: int | None) -> dict:
    """把回流样本**混入**原训练集后训练一个候选模型（隔离目录，绝不覆盖主模型）。"""
    from dataclasses import replace

    from src.data_pipeline import prepare_dataset
    from src.train import run_training

    base = prepare_dataset(cfg)
    if list(base.class_names) != list(ds["class_names"]):
        raise SystemExit("类别顺序与基础数据不一致，拒绝训练（会造成标签错位）")
    if int(base.window_size) != int(ds["window_size"]):
        raise SystemExit(f"窗口长度不一致：基础数据 {base.window_size}，回流样本 {ds['window_size']}")

    logger = LOGGER
    n_new = int(ds["X"].shape[0])
    logger.info("混入 %d 条人工研判样本（原训练集 %d 条）", n_new, int(base.X_train.shape[0]))
    mixed = replace(
        base,
        X_train=np.concatenate([base.X_train, ds["X"]], axis=0),
        y_train_mc=np.concatenate([base.y_train_mc, ds["y_mc"]], axis=0),
        y_train_bin=np.concatenate([base.y_train_bin, ds["y_bin"]], axis=0),
    )
    res = run_training(cfg, mixed, tag="feedback_candidate", epochs=epochs,
                       artifact_dir=ARTIFACT_DIR / "feedback")
    return {"tag": "feedback_candidate", "artifact_dir": str(ARTIFACT_DIR / "feedback" / "feedback_candidate"),
            "macro_f1": float(getattr(res, "macro_f1", float("nan"))),
            "n_new_samples": n_new}


# --------------------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(description="把人工研判结论回流成可训练的标注数据集")
    p.add_argument("--evidence", default=str(RESULT_DIR / "review_evidence.jsonl"))
    p.add_argument("--feedback", default=str(RESULT_DIR / "review_feedback.jsonl"))
    p.add_argument("--out", default=str(RESULT_DIR / "feedback_dataset.npz"))
    p.add_argument("--retrain", action="store_true", help="再把数据集混入训练，产出候选模型（耗时）")
    p.add_argument("--epochs", type=int, default=None, help="候选模型的训练轮数（默认用配置值）")
    args = p.parse_args(argv)

    feats, classes, win = _model_contract(cfg)
    evidence = _read_jsonl(Path(args.evidence))
    feedback = _read_jsonl(Path(args.feedback))
    LOGGER.info("读入证据 %d 条 / 研判结论 %d 条", len(evidence), len(feedback))
    if not feedback:
        raise SystemExit(f"{args.feedback} 里没有任何研判结论，无可回流的数据")

    ds = build_feedback_dataset(evidence, feedback, feats, classes, win)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=ds["X"], y_mc=ds["y_mc"], y_bin=ds["y_bin"], review_ids=ds["review_ids"],
        class_names=np.asarray(ds["class_names"]), feature_names=np.asarray(ds["feature_names"]),
        window_size=np.asarray([ds["window_size"]]),
    )

    summary: dict[str, Any] = {
        "samples": int(ds["X"].shape[0]),
        "shape": list(ds["X"].shape),
        "distribution": ds["distribution"],
        "normal_index": ds["normal_index"],
        "review_ids": [int(x) for x in ds["review_ids"]],
        "skipped": ds["skipped"],
        "bad_class": ds["bad_class"],
        "evidence_with_ground_truth_label_ignored": ds["evidence_with_gt_label"],
        "dataset": str(out),
        "size_bytes": int(out.stat().st_size),
        "note": "标签只来自人工 verdict；证据里出现的真实标签 Label 一律忽略（不参与训练）",
    }
    save_json(summary, RESULT_DIR / "feedback_dataset.json")

    # 人可读小结：谁在什么情况下能直接用这份数据集
    lines = [
        "# 回流数据集（人工研判 → 训练样本）",
        "",
        f"> 由 `tools/retrain_from_feedback.py` 生成。**{summary['samples']} 条**样本，"
        f"形状 {summary['shape']}（条数 × 窗口 × 特征），{human_bytes(summary['size_bytes'])}。",
        "",
        "## 类别分布",
        "",
        "| 类别 | 样本数 |",
        "|---|---|",
    ]
    for k, v in sorted(ds["distribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## 标签来源与诚实边界",
        "",
        "- 标签**只来自人工研判结论**（`review_feedback.jsonl` 的 `verdict`）。",
        f"- 证据里有 **{ds['evidence_with_gt_label']}** 条携带真实标签 `Label`（演示/回放数据自带），"
        "**已全部忽略**——用它当训练真值等于自欺欺人。若该数为 0，说明证据里确实没有真值可用，"
        "而不是「检查通过」。",
        f"- 被跳过的条目 **{len(ds['skipped'])}** 条"
        + ("：" + "；".join(ds["skipped"][:5]) if ds["skipped"] else "。"),
        f"- 类别不在模型类别表里而拒绝的 **{len(ds['bad_class'])}** 条"
        + ("：" + "；".join(ds["bad_class"][:5]) if ds["bad_class"] else "。"),
        "- 样本量通常很小：它的作用是**持续性微调**，不是替代主训练集。"
        "候选模型必须经 `results/real_eval.md` 同口径评测后才可替换主模型。",
        "",
        "## 下一步",
        "",
        "```bash",
        "# 1) 用回流样本混入原训练集，训练一个**候选**模型（隔离在 artifacts/feedback/，不动主模型）",
        "python tools/retrain_from_feedback.py --retrain",
        "# 2) 候选模型必须过同一套无泄漏评测与验收自检，再决定是否替换",
        "python tools/real_eval.py --dataset insdn --protocol block",
        "python tools/acceptance_check.py",
        "```",
    ]
    save_text("\n".join(lines) + "\n", RESULT_DIR / "feedback_dataset.md")

    LOGGER.info("回流数据集已写出：%s（%d 条）", out, summary["samples"])
    for k, v in sorted(ds["distribution"].items(), key=lambda kv: -kv[1]):
        LOGGER.info("  %-12s %d", k, v)
    if ds["skipped"]:
        LOGGER.warning("跳过 %d 条：%s", len(ds["skipped"]), "；".join(ds["skipped"][:3]))
    if ds["bad_class"]:
        LOGGER.warning("类别不在模型类别表而拒绝 %d 条：%s",
                       len(ds["bad_class"]), "；".join(ds["bad_class"][:3]))

    if args.retrain:
        info = retrain(cfg, ds, args.epochs)
        summary["retrain"] = info
        save_json(summary, RESULT_DIR / "feedback_dataset.json")
        LOGGER.info("候选模型已产出：%s（宏 F1 %.4f）", info["artifact_dir"], info["macro_f1"])
    else:
        LOGGER.info("未训练。加 --retrain 可把这份数据集混入训练，产出隔离的候选模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
