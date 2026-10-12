#!/usr/bin/env python
"""CIDA 统一入口：把散落的 40 个工具/多个进程收成一条命令 + 六个子命令。

为什么需要它：这套系统内部已经闭环（检测→分级→确认→处置→留痕→复核→回流→评测→验收），
但对外一直是"39 个工具、6 个入口脚本，你得知道该敲哪个"。**内部是系统、门面是散件**，
这个差距对评委和接手的人都不友好。

    python tools/cida.py doctor      环境 + 依赖 + 产物 + 数据 的完整体检
    python tools/cida.py status      全系统状态：检测服务 / 适配层 / 产物新鲜度 / 磁盘 / 待研判积压
    python tools/cida.py demo        起演示大屏（自动避开浏览器不安全端口）
    python tools/cida.py guard       起 Ryu 适配层（带白名单与演练模式检查）
    python tools/cida.py verify      三套测试 + 130 项交付自检
    python tools/cida.py inventory   产物与数据清单（哪些当前有效、哪些可清）

设计取舍：
  · doctor/verify **不重写逻辑**，直接调用既有工具（避免出现"两份实现、只改一处"——
    这个工程已经踩过两次同源判定分叉的坑）。
  · status 把适配层纳进来的方式是**读它的心跳文件**，而不是让它开 HTTP 端口：
    适配层是跑在另一个进程的 Ryu 应用，多开一个监听面就多一份攻击面，心跳文件足够判断存活。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import (ARTIFACT_DIR, FIGURE_DIR, PROCESSED_DIR, RESULT_DIR,  # noqa: E402
                       get_logger, human_bytes, load_json, resolve_dir)

LOGGER = get_logger("cida")

HEARTBEAT = RESULT_DIR / "guard_heartbeat.json"
STALE_SERVICE_S = 15.0        # 服务探活超时
STALE_GUARD_S = 300.0         # 适配层心跳超过 5 分钟视为失联
STALE_ARTIFACT_S = 7 * 86400  # 产物超过 7 天提示可能过期

# Chromium 会直接拒绝这些端口（ERR_UNSAFE_PORT），演示时别踩
UNSAFE_PORTS = {2049, 3659, 4045, 5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669,
                6697, 10080}


# --------------------------------------------------------------------------- 小工具
def _ok(b): return "✅" if b else "❌"
def _warn(b): return "✅" if b else "⚠️ "


def _age(path: Path) -> tuple[float, str]:
    if not path.exists():
        return float("inf"), "缺失"
    sec = max(0.0, time.time() - path.stat().st_mtime)
    if sec < 90:
        return sec, f"{sec:.0f} 秒前"
    if sec < 5400:
        return sec, f"{sec / 60:.0f} 分钟前"
    if sec < 3 * 86400:
        return sec, f"{sec / 3600:.1f} 小时前"
    return sec, f"{sec / 86400:.1f} 天前"


def _get_json(url: str, timeout: float = STALE_SERVICE_S):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - 探活失败是正常分支（服务没起）
        return None


def _service_url() -> str:
    host = os.environ.get("CIDA_HOST", "127.0.0.1")
    port = os.environ.get("CIDA_PORT", "5000")
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _run(desc: str, argv: list[str], cwd: Path | None = None) -> int:
    """调用既有工具（不重写逻辑），stdio 直通。"""
    print(f"\n▶ {desc}\n  $ {' '.join(argv)}")
    return subprocess.call(argv, cwd=str(cwd or ROOT))


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


# --------------------------------------------------------------------------- status
def cmd_status(args) -> int:
    """全系统状态一屏看完：这是"整套系统健康吗"的唯一答案。"""
    url = _service_url()
    print("=" * 74)
    print(f"CIDA 系统状态  ·  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)

    # 1) 检测服务
    health = _get_json(f"{url}/api/health")
    stats = _get_json(f"{url}/api/stats") if health is not None else None
    if health is None:
        print(f"\n检测服务      ❌ 探活失败（{url}）—— 未启动或不可达")
    else:
        ready = bool(health.get("detector_ready"))
        line = f"\n检测服务      {_ok(ready)} {'就绪' if ready else '未就绪'}"
        if stats:
            pol = stats.get("policy") or {}
            line += (f" · 已判窗口 {stats.get('windows', 0):,}"
                     f" · 确认告警 {stats.get('alarms_confirmed', 0):,}"
                     f" · 策略 {pol.get('action')}{'（演练）' if pol.get('dry_run') else '（真实下发）'}")
        print(line)
        if not ready:
            print(f"              原因：{health.get('detector_reason') or '缺少模型/预处理产物'}")

    # 2) 适配层（读心跳，不开端口）
    hb, hb_age = (load_json(HEARTBEAT) if HEARTBEAT.exists() else None), _age(HEARTBEAT)[0]
    if hb is None:
        print(f"适配层        ⚠️  无心跳（未启动过，或从未下达处置）")
    else:
        alive = hb_age <= STALE_GUARD_S
        print(f"适配层        {_warn(alive)} 心跳 {_age(HEARTBEAT)[1]}"
              f" · 模式 {'真实处置' if hb.get('execute') else '演练'}"
              f" · 白名单 {len(hb.get('whitelist') or [])} 个"
              f" · 封禁中 {hb.get('blocked_now', 0)}"
              f" · 已处置 {hb.get('alerts_handled', 0)}")
        if not alive:
            print(f"              心跳已超过 {STALE_GUARD_S:.0f} 秒 —— 适配层可能已退出")
        if not (hb.get("whitelist") or []):
            print("              ⚠️  白名单为空：上线前必须加入控制器与管理网段")

    # 3) 产物新鲜度：判据是「模型是否比产生它的代码还旧」，而不是「模型多久没更新」——
    #    后者会在"模型本来就对、只是没重训"时误报，而误报会训练人忽略这个工具。
    src_files = [ROOT / "src" / n for n in ("model.py", "train.py", "data_pipeline.py")]
    src_files = [p for p in src_files if p.exists()]
    model_p = ARTIFACT_DIR / "main_model.json"
    model_m = model_p.stat().st_mtime if model_p.exists() else 0.0
    code_m = max((p.stat().st_mtime for p in src_files), default=0.0)
    stale = bool(model_m and code_m and model_m < code_m)
    _, age_model = _age(model_p)
    _, age_sum = _age(RESULT_DIR / "summary.json")
    meta = resolve_dir("", PROCESSED_DIR) / "meta.json"
    print(f"\n产物新鲜度    {_warn(not stale)} 主模型 {age_model} · summary {age_sum}"
          f" · meta.json {'就绪' if meta.exists() else '缺失'}")
    if stale:
        newest = max(src_files, key=lambda p: p.stat().st_mtime)
        print(f"              ⚠️  模型早于代码改动（{newest.name} 更新过）"
              f"：改动可能在推理链路生效但未反映到模型，请重跑 python run_all.py")

    # 4) 交付验收（读自检产物，不写死）
    acc = RESULT_DIR / "acceptance_summary.json"
    if acc.exists():
        d = load_json(acc) or {}
        bad = int(d.get("failed", 0))
        print(f"交付验收      {_ok(bad == 0)} 通过 {d.get('passed', '?')} 项"
              f" / 警告 {d.get('warned', '?')} / 失败 {bad}（{d.get('ts', '')}）")
    else:
        print("交付验收      ⚠️  尚无结果，请跑 python tools/cida.py verify")

    # 5) 待研判积压（值不值得关注，看这一行）
    if stats:
        pending = int(stats.get("review_pending", 0) or 0)
        print(f"待研判积压    {_warn(pending == 0)} {pending} 条"
              f" · 观察池 {stats.get('observations', 0)}"
              f" · 已回流 {stats.get('review_labeled', 0)}")
        if pending:
            print("              ⚠️  有人在等结论：未消化的未知告警不会自己变成训练样本")

    # 6) 磁盘
    lines = []
    for name, p in (("results", RESULT_DIR), ("logs", ROOT / "logs")):
        if p.exists():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            lines.append(f"{name} {human_bytes(total)}")
    du = shutil.disk_usage(str(ROOT))
    print(f"\n磁盘          {_warn(du.free > 2 * 1024**3)} 可用 {human_bytes(du.free)}"
          f" · {' · '.join(lines) if lines else '无 results/logs'}")

    print("\n" + "-" * 74)
    print("提示：一行命令体检用 doctor；复现主实验用 run_all.py；提交前用 verify。")
    return 0


# --------------------------------------------------------------------------- 其他子命令
def cmd_doctor(args) -> int:
    rc = _run("环境与依赖体检", _py("tools/check_env.py"))
    meta = resolve_dir("", PROCESSED_DIR) / "meta.json"
    print(f"\n▶ 关键产物")
    for label, p in (("预处理 meta.json", meta),
                     ("主模型指针", ARTIFACT_DIR / "main_model.json"),
                     ("主结果 summary.json", RESULT_DIR / "summary.json"),
                     ("验收自检结果", RESULT_DIR / "acceptance_summary.json")):
        print(f"  {_ok(p.exists())} {label}  {_age(p)[1]}")
    real = ROOT / "data" / "real"
    n_real = len([d for d in real.glob("*.csv")]) + len([d for d in real.glob("*.zip")])
    print(f"  {_warn(n_real > 0)} data/real 数据文件 {n_real} 个（真实数据评测用）")
    print(f"\n▶ 统一入口自检：{_ok(True)} cida 可用（doctor/status/demo/guard/verify/inventory）")
    return rc


def cmd_demo(args) -> int:
    port = int(args.port)
    if port in UNSAFE_PORTS or port < 1024:
        print(f"✗ 端口 {port} 会被 Chromium 拒绝（ERR_UNSAFE_PORT）或被系统占用限制，"
              f"请换一个，例如 5070")
        return 2
    print(f"▶ 启动演示大屏： http://127.0.0.1:{port}/")
    print("  提示：想看到「复核台」有数据，进页面后把「未知形态阈值」从 0 拉到 0.30 再点应用策略。")
    return _run("起大屏", _py("app/app.py", "--port", str(port)))


def cmd_guard(args) -> int:
    wl = [x for x in (args.whitelist or "").replace(",", " ").split() if x]
    print("▶ 启动 Ryu 适配层前的检查：")
    print(f"  {_ok(bool(wl))} 白名单 {len(wl)} 个 {wl if wl else '（空 —— 上线前必须加入控制器/管理网段/网关）'}")
    print(f"  {_ok(not args.execute)} 模式：{'演练（只打印，不改流表）' if not args.execute else '真实处置（会改流表）'}")
    if not shutil.which("ryu-manager"):
        print("  ❌ 未找到 ryu-manager（需先安装 Ryu；离线自检可用 "
              "python tools/ryu_live_check.py --self-test）")
        return 2
    cfg = ROOT / "deploy" / "guard-args.txt"
    if wl:
        cfg.write_text(" ".join([f"--whitelist={','.join(wl)}", f"--execute={args.execute}"]),
                       encoding="utf-8")
    argv = ["ryu-manager", "integrations/ryu_guard.py"]
    if wl:
        argv += ["--user-flags", str(cfg)]
    return _run("起适配层", argv)


def cmd_verify(args) -> int:
    rc = 0
    for name in ("tests/test_core.py", "tests/test_ryu_guard.py", "tests/test_pcap_tools.py"):
        rc |= _run(f"跑 {name}", _py(name))
    rc |= _run("交付验收自检（130 项）", _py("tools/acceptance_check.py"))
    print(f"\n{'✅ 全部通过' if rc == 0 else '❌ 有失败项，详见上面输出'}")
    return rc


def cmd_inventory(args) -> int:
    """产物与数据清单：哪些当前有效、哪些可清（②先给最小可用版）。"""
    print("=" * 74)
    print("产物与数据清单")
    print("=" * 74)

    def scan(p: Path, note: str, keep: list[str] | None = None) -> None:
        if not p.exists():
            print(f"\n{note}：不存在")
            return
        subs = sorted([d for d in p.iterdir() if d.is_dir()], key=lambda d: -d.stat().st_mtime)
        print(f"\n{note}（{len(subs)} 个目录，按最近修改排序，取前 12）")
        for d in subs[:12]:
            _, age = _age(d)
            mark = "★" if (keep and d.name in keep) else " "
            print(f"  {mark} {d.name:38} {age:>10}")

    keep_art = []
    ptr = ARTIFACT_DIR / "main_model.json"
    if ptr.exists():
        keep_art = [str((load_json(ptr) or {}).get("tag") or "")]
    print(f"\n★ = 当前主模型相关（{keep_art or '未记录'}）；其余多为消融/对标/试跑的中间产物")
    scan(ARTIFACT_DIR, "artifacts/（模型产物）", keep_art)
    scan(ROOT / "data", "data/（数据集与预处理产物）", ["processed", "real", "raw"])
    print(f"\nresults/ 共 {len(list(RESULT_DIR.glob('*')))} 个条目")
    print("  要清理时请先确认没有文档引用：grep -r \"<文件名>\" docs competition README.md")
    print("  约定：results/*.md 与 *.json 多为交付产物，*_smoke* / _* 为试跑中间产物（已 gitignore）")
    return 0


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cida", description="CIDA 统一入口（环境/状态/演示/适配层/验收/产物）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="环境与产物体检（调用 tools/check_env.py）")
    sub.add_parser("status", help="全系统状态：服务/适配层/产物/验收/积压/磁盘")
    d = sub.add_parser("demo", help="起演示大屏")
    d.add_argument("--port", type=int, default=5070, help="端口（默认 5070，避开浏览器不安全端口）")
    g = sub.add_parser("guard", help="起 Ryu 适配层")
    g.add_argument("--whitelist", default="", help="永不处置的源 IP，逗号分隔")
    g.add_argument("--execute", action="store_true", help="开启真实处置（默认仅演练）")
    sub.add_parser("verify", help="三套测试 + 交付验收自检")
    sub.add_parser("inventory", help="产物与数据清单（哪些当前有效、哪些可清）")

    args = p.parse_args(argv)
    return {"doctor": cmd_doctor, "status": cmd_status, "demo": cmd_demo,
            "guard": cmd_guard, "verify": cmd_verify, "inventory": cmd_inventory}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
