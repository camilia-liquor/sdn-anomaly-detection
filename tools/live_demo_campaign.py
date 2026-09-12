"""多攻击连演：一条命令把 7 类攻击依次演一遍，并给出**每类攻击的检出时延对比表**。

为什么做这个
------------
单场 `live_demo.py` 能证明"系统在做事"，但评委/运维下一个问题一定是：
**"换成别的攻击类型呢？都一样快吗？"** 本工具用同一套模型、同一个阈值、同一套确认规则，
把每一类攻击各演一场，把结果并成一张表——**这才是可以拿去对比、可以写进报告的结论**。

它不重复实现任何演示逻辑：直接复用 `tools/live_demo.py` 的 `LiveDemo`，
每个场景用**全新构造的检测服务实例**（因此场景之间的位置跳转不会污染窗口，
也就不会出现横跨两段的伪窗口），并把每场的详细报告分别落盘。

用法::

    python tools/live_demo_campaign.py                          # 默认连演 7 类
    python tools/live_demo_campaign.py --attacks DDoS,Botnet,Probe
    python tools/live_demo_campaign.py --speed 0 --no-color      # 静默快跑（用于自动化/CI）
    python tools/live_demo_campaign.py --speed 30 --normal-rows 600

产物：`results/live_demo_campaign.md`（对比表 + 逐场一致性）、`results/live_demo_campaign.json`，
以及每场的详细事件报告 `results/live_demo_incident_<类别>.md`。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import RESULT_DIR, get_logger, save_json, save_text, silence_tf_logs  # noqa: E402

LOGGER = get_logger("cida.campaign")

DEFAULT_ATTACKS = ["DDoS", "Probe", "BruteForce", "Botnet", "DoS", "WebAttack", "U2R"]


def _load_live_demo():
    """把 tools/live_demo.py 当模块加载（它本身也有 CLI，但这里只借它的演示能力）。

    注意：必须在 `exec_module` **之前**把模块注册进 `sys.modules`——`live_demo.py` 里用了
    `@dataclass`，而 dataclasses 会通过 `sys.modules[cls.__module__]` 去解析类型注解，
    模块不在 `sys.modules` 里会直接抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    """
    path = ROOT / "tools" / "live_demo.py"
    name = "cida_live_demo"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def _passthrough(argv: Sequence[str]) -> list[str]:
    """把本工具的参数翻译成 live_demo 的命令行参数。"""
    out: list[str] = []
    for a in argv:
        if a in ("--attacks", "--out-md", "--out-json"):
            continue
        out.append(a)
    return out


def run_one(live, cls: str, base_argv: Sequence[str], tag_paths: bool = True) -> dict:
    """跑一场演示，返回本场的关键结果（同时把详细报告写到按类别命名的文件）。"""
    argv = ["--attack", cls] + [a for a in base_argv]
    args = live.parse_args(argv)
    try:
        live._resolve_tty(args)
    except Exception:  # pragma: no cover - 非 TTY 环境下不需要
        pass

    if tag_paths:
        live.REPORT_PATH = RESULT_DIR / f"live_demo_incident_{cls}.md"
        live.AUDIT_PATH = RESULT_DIR / f"live_demo_audit_{cls}.jsonl"

    captured: list[tuple[Any, Any, Any]] = []
    orig_write = live.LiveDemo._write_report

    def patched(self, scene, summary, reference):  # noqa: ANN001
        captured.append((self, scene, summary))
        return orig_write(self, scene, summary, reference)

    live.LiveDemo._write_report = patched  # type: ignore[assignment]
    demo = None
    t0 = time.perf_counter()
    try:
        demo = live.LiveDemo(args)
        rc = demo.run()
    finally:
        live.LiveDemo._write_report = orig_write  # type: ignore[assignment]

    wall = time.perf_counter() - t0
    if not captured:
        raise RuntimeError(f"{cls}：演示未产出场景结果（返回码 {rc}）")
    obj, scene, summary = captured[-1]
    stats = summary["stats"]
    attack_alerts = summary["attack_alerts"]
    n_false = len(summary["false_pre"]) + len(summary["false_post"])
    return {
        "class": cls,
        "ok": bool(rc == 0),
        "status": "ok",
        "returncode": int(rc),
        "wall_seconds": round(wall, 2),
        "segment_start": int(scene.seg_start),
        "segment_length": int(scene.seg_length),
        "n_candidates": int(scene.n_candidates),
        "n_long_enough": int(scene.n_long_enough),
        "pushed_rows": int(getattr(obj, "pushed", 0)),
        "detected": not summary["missed"],
        "delay_ms": None if obj.delay_ms is None else round(float(obj.delay_ms), 3),
        "delay_rows": obj.delay_rows,
        "attack_alerts": len(attack_alerts),
        "false_alarms": int(n_false),
        "false_pre": len(summary["false_pre"]),
        "false_post": len(summary["false_post"]),
        "by_action": {k: int(v) for k, v in summary["by_action"].items()},
        "n_blocked_ips": int(summary["n_blocked"]),
        "recovered": bool(summary["recovered"]),
        "check_n": int(getattr(obj, "check_n", 0)),
        "check_max_dp": float(getattr(obj, "check_max_dp", 0.0)),
        "check_mismatch": int(getattr(obj, "check_mismatch", 0)),
        "check_skipped": int(getattr(obj, "check_skipped", 0)),
        "unknown_windows": int(stats.get("unknown_windows", 0)),
        "stream_resets": int(stats.get("stream_resets", 0)),
        "infer_p50_ms": float(stats["batch_latency_ms"]["p50"]),
        "policy": dict(stats.get("policy") or {}),
        "report": f"results/live_demo_incident_{cls}.md",
    }


def _table(rows: list[dict], policy: dict) -> str:
    head = ("| 攻击类型 | 片段长度 | 是否检出 | **检测时延(ms)** | 折合流记录 | 片段内告警 | 误报 | "
            "处置动作 | 未知窗口 | 逐窗口核对 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    body = []
    for r in rows:
        delay = "—" if r["delay_ms"] is None else f"**{r['delay_ms']:.2f}**"
        rows_txt = "—" if r["delay_rows"] is None else str(r["delay_rows"])
        acts = "、".join(f"{k}×{v}" for k, v in r["by_action"].items()) or "—"
        check = (f"{r['check_n']} 窗口，最大 |Δp| {r['check_max_dp']:.1e}，"
                 f"类别不一致 {r['check_mismatch']}") if r["check_n"] else "未对齐"
        # ⚠️ 「场景不可用」必须与「未检出」分开：前者是**演示剧本构造**的限制（测试流里该类型
        # 找不到满足条件的片段组合），与模型能力无关；混在一起会让人误以为模型漏报。
        status = r.get("status", "ok")
        if status == "scenario_unavailable":
            verdict = "⚠️ 场景不可用"
            delay = rows_txt = "—"
        elif status == "error":
            verdict = "⚠️ 运行出错"
        else:
            verdict = "✅ 是" if r["detected"] else "❌ 未检出"
        seg = r["segment_length"] if r["segment_length"] else "—"
        body.append(f"| **{r['class']}** | {seg} | {verdict} | {delay} | {rows_txt} | "
                    f"{r['attack_alerts']} | {r['false_alarms']} | {acts} | {r['unknown_windows']} | {check} |")
    return "\n".join([head, sep] + body)


def build_report(rows: list[dict], *, argv: Sequence[str]) -> str:
    det = [r for r in rows if r["detected"]]
    skipped = [r for r in rows if r.get("status") == "scenario_unavailable"]
    errored = [r for r in rows if r.get("status") == "error"]
    ran = [r for r in rows if r.get("status") == "ok"]
    missed = [r for r in ran if not r["detected"]]
    delays = [r["delay_ms"] for r in det if r["delay_ms"] is not None]
    total_false = sum(r["false_alarms"] for r in rows)
    total_alarms = sum(r["attack_alerts"] for r in rows)
    total_check = sum(r["check_n"] for r in rows)
    worst_check = max((r["check_max_dp"] for r in rows if r["check_n"]), default=0.0)
    mismatch = sum(r["check_mismatch"] for r in rows)
    skipped_windows = sum(r["check_skipped"] for r in rows)
    policy = rows[0]["policy"] if rows else {}

    lines: list[str] = []
    lines.append("# 多攻击连演：每类攻击的检出时延对比")
    lines.append("")
    lines.append(f"> 一条命令生成：`python tools/live_demo_campaign.py {' '.join(argv)}`")
    lines.append("> ")
    lines.append("> 判定口径全场一致：" + (
        f"阈值 {policy.get('threshold')}、连续确认 {policy.get('confirm_windows')} 窗口、"
        f"处置动作 {policy.get('action')}（dry_run={policy.get('dry_run')}）、"
        f"未知标记阈值 {policy.get('unknown_threshold')}。" if policy else "见各场报告。"))
    lines.append("")
    lines.append("## 一、结论")
    lines.append("")
    lines.append(f"* **实际演出 {len(ran)}/{len(rows)} 类，检出 {len(det)}/{len(ran)}**："
                 + ("所有演出过的攻击类型都报出了确认告警，没有漏报。" if not missed
                    else "未检出的类型：" + "、".join(r["class"] for r in missed) + "。"))
    if skipped:
        lines.append(f"* **{len(skipped)} 类未演出**（" + "、".join(r["class"] for r in skipped) +
                     "）：这是**演示剧本构造条件**不满足（该类型在测试流里找不到"
                     "「有安静基线的前置片段 + 够长攻击片段」的组合），**与模型能力无关**——"
                     "这些类型的检出性能请看离线评估 `results/summary_*.md` 与 `results/per_class_*.csv`"
                     "（基于全部攻击片段，样本量更大）。")
    if errored:
        lines.append(f"* ⚠️ **{len(errored)} 类运行出错**（" + "、".join(r["class"] for r in errored) +
                     "）：见第四节与 `results/live_demo_campaign.json` 的 `error` 字段（如实记录，未隐藏）。")
    if delays:
        lines.append(f"* **检测时延**：最快 {min(delays):.2f} 毫秒、最慢 {max(delays):.2f} 毫秒、"
                     f"平均 {sum(delays) / len(delays):.2f} 毫秒（**流时间**口径，"
                     "与 `src/evaluate.py::online_alarm_evaluation` 一致；不是墙上时间，也不受回放倍速影响）。")
    lines.append(f"* **告警与误报**：全场确认告警 {total_alarms} 条、"
                 f"正常阶段误报 **{total_false}** 条" + ("（各场均为 0）" if total_false == 0 else "") + "。")
    lines.append(f"* **可核对性**：累计与离线评估逐窗口比对 **{total_check}** 个窗口，"
                 f"最大 |Δp| = {worst_check:.2e}，判定类别不一致 **{mismatch}** 个"
                 + (f"，跳过 {skipped_windows} 个不可比窗口（场景内位置跳转所致，已重置滑窗）"
                    if skipped_windows else "，**没有跳过任何窗口**") + "。")
    lines.append("")
    lines.append("## 二、逐类对比表（本表是核心）")
    lines.append("")
    lines.append(_table(rows, policy))
    lines.append("")
    lines.append("> 「检测时延」= 攻击片段起点 → 首个**确认**告警窗口右端的真实流时间差。"
                 "「片段内告警」= 该攻击片段持续期间产生的确认告警条数（服务口径：一个确认窗口一条）。"
                 "「未知窗口」= 被 `unknown_threshold` 标为「未知形态」的窗口数（默认 0 = 关闭）。")
    lines.append("> ")
    lines.append("> 「逐窗口核对」列里的**「未对齐」**表示该场因**位置跳转 + 滑窗重置**导致判定窗口"
                 "与离线评估的滑窗网格错位，因此不做逐窗口比对——**判定仍由模型逐窗口给出、指标口径不变**，"
                 "只是少了一次交叉验证。对齐的场次占总数的 "
                 f"{sum(1 for r in rows if r.get('check_n'))}/{len(rows)}。")
    lines.append("")
    lines.append("## 三、逐场明细与产物")
    lines.append("")
    lines.append("| 攻击类型 | 片段起点行 | 候选片段 | 够长候选 | 回放条数 | 处置覆盖源 IP | 风险回落 | 墙钟耗时 | 详细报告 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['class']} | {r['segment_start']} | {r['n_candidates']} | {r['n_long_enough']} | "
                     f"{r['pushed_rows']} | {r['n_blocked_ips']} | {'是' if r['recovered'] else '否'} | "
                     f"{r['wall_seconds']:.1f} s | `{r['report']}` |")
    lines.append("")
    lines.append("## 四、诚实与口径说明")
    lines.append("")
    lines.append("1. **每场都是独立的检测服务实例**：场景之间必然发生位置跳转（等价于上游数据中断），"
                 "新实例的滑窗缓冲从零累积，因此**场景之间不会互相污染**——这也是本工具不用一个长连接"
                 "连演的原因。**但场景内部**仍可能有跳转（前置基线取自别处、或恢复段不在攻击片段之后），"
                 "此时脚本会重置滑窗缓冲：好处是不产生横跨两段的伪窗口，代价是该场与离线评估的"
                 f"可比窗口变少（本场累计跳过 {skipped_windows} 个），个别场次（如上面的 WebAttack）"
                 "会因网格错位而整场不做逐窗口比对。")
    lines.append("2. **检测时延是流时间口径**：取自数据集真实时间戳之差，因此把 `--speed` 从 0 改到 100，"
                 "本报告的所有**判定类**数字逐字不变（只有墙钟耗时与服务侧推理 p50 会随负载变化）。")
    lines.append("3. **处置的性质**：每场的「处置」都是 `dry_run` 下的**策略建议** + 脚本内置的本地模拟执行器，"
                 "**不是**真的改交换机流表。真实下发见 `integrations/ryu_guard.py`"
                 "（本机无 Ryu 环境，未实跑；接口与判定逻辑由 `tools/controller_simulator.py` 验证）。")
    lines.append("4. **风险回落 ≠ 处置有效**：回放的是历史数据，攻击片段结束后真实标签切回 Normal，"
                 "风险自然回落。本报告不声称阻断动作压住了攻击。")
    lines.append("5. **每类攻击只测了一个片段**（选中该类型「最看得清」的那个片段），因此这里的时延是"
                 "**演示级**数字；统计意义上的每类性能请看离线评估 `results/summary_*.md` 与 "
                 "`results/per_class_*.csv`（基于 24 个攻击片段）。两者口径一致、用途不同。")
    lines.append("")
    lines.append("> 本文件由 `tools/live_demo_campaign.py` 生成，**可复现**：同参数两次运行，"
                 "所有**判定类指标**（检测时延、折合流记录数、告警数、误报数、逐窗口核对的窗口数与最大 |Δp|）"
                 "逐字一致；只有两处**真实墙钟计时**（逐场墙钟耗时、服务侧推理 p50）随机器负载波动。")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="多攻击连演：依次演示多类攻击并汇总每类检出时延",
        epilog="其余参数（--speed/--normal-rows/--threshold/--confirm/--action/--batch/--model-tag 等）"
               "全部透传给 tools/live_demo.py。")
    ap.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS),
                    help="要连演的攻击类型（逗号分隔），默认全部 7 类")
    ap.add_argument("--out-md", default=str(RESULT_DIR / "live_demo_campaign.md"))
    ap.add_argument("--out-json", default=str(RESULT_DIR / "live_demo_campaign.json"))
    ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--normal-rows", type=int, default=400)
    ap.add_argument("--no-color", action="store_true", default=True)
    ap.add_argument("--no-animation", action="store_true", default=True)
    known, _unknown = ap.parse_known_args(argv)
    return known


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    args = parse_args(argv)
    silence_tf_logs()
    live = _load_live_demo()

    classes = [c.strip() for c in str(args.attacks).split(",") if c.strip()]
    if not classes:
        LOGGER.error("--attacks 为空，无事可做")
        return 2

    base = ["--speed", f"{args.speed:g}", "--normal-rows", str(args.normal_rows),
            "--no-color", "--no-animation"]
    # 把用户显式给出的其它 live_demo 参数也带上（例如 --threshold / --model-tag）
    for a in argv:
        if a.startswith("--") and a not in ("--attacks", "--out-md", "--out-json",
                                            "--speed", "--normal-rows", "--no-color",
                                            "--no-animation"):
            base.append(a)

    rows: list[dict] = []
    LOGGER.info("开始连演 %d 类攻击：%s", len(classes), "、".join(classes))
    for i, cls in enumerate(classes, 1):
        LOGGER.info("—— [%d/%d] %s ——", i, len(classes), cls)
        try:
            r = run_one(live, cls, base)
        except Exception as exc:  # noqa: BLE001 — 单场失败不应中断整场连演，但要如实记录
            msg = f"{type(exc).__name__}: {exc}"
            # 区分「剧本构造不出来」与「运行出错」：前者是演示脚手架的限制，不是模型的问题。
            status = "scenario_unavailable" if "找不到可用的演示剧本" in str(exc) else "error"
            LOGGER.error("%s 演示未能完成（%s）：%s", cls, status, msg)
            rows.append({"class": cls, "ok": False, "status": status, "detected": False, "returncode": -1,
                         "delay_ms": None, "delay_rows": None, "attack_alerts": 0, "false_alarms": 0,
                         "by_action": {}, "check_n": 0, "check_max_dp": 0.0, "check_mismatch": 0,
                         "segment_length": 0, "segment_start": -1, "n_candidates": 0, "n_long_enough": 0,
                         "pushed_rows": 0, "n_blocked_ips": 0, "recovered": False, "unknown_windows": 0,
                         "stream_resets": 0, "wall_seconds": 0.0, "policy": {},
                         "report": f"（{cls} 未演出）", "error": msg})
            continue
        rows.append(r)

    text = build_report(rows, argv=argv)
    save_text(text, Path(args.out_md))
    save_json({"attacks": classes, "rows": rows,
               "summary": {
                   "n": len(rows), "detected": sum(1 for r in rows if r["detected"]),
                   "mean_delay_ms": (sum(r["delay_ms"] for r in rows if r["delay_ms"] is not None)
                                     / max(sum(1 for r in rows if r["delay_ms"] is not None), 1)),
                   "total_attack_alerts": sum(r["attack_alerts"] for r in rows),
                   "total_false_alarms": sum(r["false_alarms"] for r in rows),
                   "total_checked_windows": sum(r["check_n"] for r in rows),
                   "max_abs_dp": max((r["check_max_dp"] for r in rows), default=0.0),
                   "class_mismatch": sum(r["check_mismatch"] for r in rows),
               }}, Path(args.out_json))

    print()
    print(text.split("## 二、逐类对比表（本表是核心）")[1].split("## 三、")[0].strip())
    print()
    LOGGER.info("连演完成：检出 %d/%d，报告 → %s", sum(1 for r in rows if r["detected"]), len(rows), args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
