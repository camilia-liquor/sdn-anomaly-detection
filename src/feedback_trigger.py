"""回流触发策略：**什么时候该重训**（"未知 → 已知"闭环的最后一步）。

为什么单独成一个模块
--------------------
数据链路早已闭合：`results/review_evidence.jsonl`（窗口特征）+ `results/review_feedback.jsonl`
（人工结论）按 `review_id` 就能拼出带标签样本，`tools/retrain_from_feedback.py` 也能一键产出
**隔离的候选模型**（绝不覆盖主模型）。缺的一直是"**什么时候**该跑"——过去靠人记，
于是闭环的触发环节事实上是断的。

这里把触发判定做成**纯函数 + 一份状态文件**，让它可以被定时任务安全地反复调用：
每次只看"上次重训之后**新增**的结论"，而不是累计条数（累计条数一旦越过门槛就会每次都重训，
在同样的数据上反复训同一个模型，既浪费算力也会让人对报告里的"候选模型"失去信任）。

判据（写死在代码里，全部可在命令行覆盖，但不允许静默放松）
----------------------------------------------------------
1. **新结论条数** ≥ `min_new_labels`（默认 10）：样本太少没有统计意义；
2. 其中**新确认为攻击**的条数 ≥ `min_new_attack`（默认 3）：闭环的价值主要是把"未知"变成
   "已知攻击"；全是 `normal`（普通误报）时，该动的是工作点（`autotighten`），不是在正常样本上重训；
3. 距上次重训 ≥ `cooldown_hours`（默认 24）：防止一次告警风暴把所有算力吃掉；
4. 状态文件损坏时**拒绝自动重训**（宁可少跑一次，也不要在错误的状态上跑）。

⚠️ 本模块**只做判定**：不训练、不写模型、不碰 `results/summary.json` 等受守护的产物。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

STATE_VERSION = 1
# 允许的"没得跑"原因（写进日志与产物，便于回答"为什么今天没重训"）
REASON_OK = "ok"
REASON_NO_NEW = "no_new_labels"
REASON_TOO_FEW_ATTACK = "too_few_new_attacks"
REASON_COOLDOWN = "cooldown"
REASON_BAD_STATE = "bad_state"


@dataclass
class RetrainState:
    """一次重训触发器的持久状态（`results/retrain_state.json`）。

    `last_review_id` 是**增量游标**：只认比它更大的 review_id，因此同一批结论
    （包括轮转后重新读到的旧结论）不会被重复计入——这和 `review_id` 跨重启唯一的
    设计是配套的（它本来就是结论与证据之间唯一的关联键）。
    """

    version: int = STATE_VERSION
    last_run_ts: float = 0.0
    last_review_id: int = 0
    n_runs: int = 0
    last_exit_code: int | None = None
    history: list = field(default_factory=list)      # 最近若干次触发记录（调用方负责裁剪）

    def to_dict(self) -> dict:
        return asdict(self)


def load_state(path: str | Path) -> tuple[RetrainState, bool]:
    """读状态文件，返回 (状态, 是否可用)。文件缺失 = 全新状态（可用）；损坏 = 不可用。"""
    p = Path(path)
    if not p.exists():
        return RetrainState(), True
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return RetrainState(
            version=int(data.get("version", STATE_VERSION)),
            last_run_ts=float(data.get("last_run_ts", 0.0) or 0.0),
            last_review_id=int(data.get("last_review_id", 0) or 0),
            n_runs=int(data.get("n_runs", 0) or 0),
            last_exit_code=(None if data.get("last_exit_code") is None
                            else int(data["last_exit_code"])),
            history=list(data.get("history") or []),
        ), True
    except (OSError, ValueError, TypeError) as exc:
        return RetrainState(), False


def save_state(path: str | Path, state: RetrainState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)          # 原子替换：定时任务可能被中断，不能让状态只写一半


def new_labels(records: list[dict], since_review_id: int) -> list[dict]:
    """取出比 `since_review_id` 更新的结论记录（按 review_id 去重，同一 id 取最后一次）。"""
    latest: dict[int, dict] = {}
    for rec in records or []:
        try:
            rid = int(rec.get("review_id"))
        except (TypeError, ValueError):
            continue
        if rid > int(since_review_id):
            latest[rid] = rec
    return [latest[k] for k in sorted(latest)]


def decide(records: list[dict], state: RetrainState, *, min_new_labels: int = 10,
           min_new_attack: int = 3, cooldown_hours: float = 24.0,
           now: float | None = None, state_ok: bool = True) -> dict:
    """判定"现在该不该重训"，返回带原因的结论（**不产生任何副作用**）。

    返回：``{"run": bool, "reason": str, "n_new": int, "n_new_attack": int,
    "since_review_id": int, "next_review_id": int, "cooldown_left_h": float, ...}``
    """
    now = time.time() if now is None else float(now)
    since = int(state.last_review_id)
    fresh = new_labels(records, since)
    n_new = len(fresh)
    n_attack = sum(1 for r in fresh if str(r.get("verdict") or "").startswith("attack:"))
    cooldown = max(0.0, float(cooldown_hours)) * 3600.0
    elapsed = now - float(state.last_run_ts or 0.0)
    left_h = max(0.0, (cooldown - elapsed) / 3600.0) if cooldown else 0.0
    info = {
        "since_review_id": since,
        "n_new": int(n_new),
        "n_new_attack": int(n_attack),
        "min_new_labels": int(min_new_labels),
        "min_new_attack": int(min_new_attack),
        "cooldown_hours": float(cooldown_hours),
        "cooldown_left_h": round(left_h, 3),
        "last_run_ts": float(state.last_run_ts or 0.0),
        "n_runs": int(state.n_runs),
        "next_review_id": (max([since] + [int(r.get("review_id") or 0) for r in fresh])
                           if fresh else since),
    }

    def out(run: bool, reason: str) -> dict:
        return {"run": bool(run), "reason": reason, **info}

    if not state_ok:
        return out(False, REASON_BAD_STATE)
    if n_new == 0:
        return out(False, REASON_NO_NEW)
    if n_attack < max(0, int(min_new_attack)):
        return out(False, REASON_TOO_FEW_ATTACK)
    if n_new < max(1, int(min_new_labels)):
        return out(False, REASON_NO_NEW)
    if left_h > 0:
        return out(False, REASON_COOLDOWN)
    return out(True, REASON_OK)


def mark_ran(state: RetrainState, decision: dict, *, exit_code: int = 0,
             now: float | None = None, keep_history: int = 20) -> RetrainState:
    """记录一次**真正执行**的触发（成功或失败都要记，否则失败会被每小时重试一次而无人察觉）。"""
    now = time.time() if now is None else float(now)
    state.last_run_ts = now
    state.last_review_id = int(decision.get("next_review_id") or state.last_review_id)
    state.n_runs = int(state.n_runs) + 1
    state.last_exit_code = int(exit_code)
    state.history = (list(state.history) + [{
        "ts": round(now, 3), "n_new": int(decision.get("n_new") or 0),
        "n_new_attack": int(decision.get("n_new_attack") or 0),
        "review_id_upto": int(state.last_review_id), "exit_code": int(exit_code),
    }])[-max(1, int(keep_history)):]
    return state
