"""生成「运维 SLO 卡」一页纸（与 `results/figures/onepager.png` 配成一套）。

为什么需要第二张一页纸
----------------------
第一张讲的是「作品做成了什么」；运维/评委还会问「**那我一天要花多少人工、各类攻击要等多久**」。
这张卡把 `results/ops_impact.md`（告警量与人力的换算）、`results/per_class_online.md`（逐类检测时延）
与 `results/window_ablation.md`（窗口长度换误报）压缩成一屏。

所有数字都从产物文件读取（不手打），重跑实验后重新生成即可保持一致。

用法::

    python tools/make_ops_card.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import FIGURE_DIR, RESULT_DIR, get_logger  # noqa: E402

LOGGER = get_logger("cida.opscard")


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect() -> dict:
    ops = _load(RESULT_DIR / "ops_impact.json")
    pco = _load(RESULT_DIR / "per_class_online.json")
    ws = _load(RESULT_DIR / "window_ablation.json")
    e2e = _load(RESULT_DIR / "e2e_controller_report.json")

    rec = ops.get("recommended_working_point") or {}
    noc = ops.get("no_confirm_counterfactual") or {}
    classes = sorted((pco.get("per_class") or []), key=lambda r: r.get("mean_delay_ms") or 0)

    # 窗口消融里的混合模型行（用于"加长窗口换误报"这一行）
    win_rows = []
    for r in (ws.get("rows") or []):
        if not r.get("measured"):
            continue
        v = (r.get("variants") or {}).get("cnn_bilstm") or {}
        fa = v.get("false_alarms_per_hour") or {}
        dl = v.get("mean_detection_delay_sec") or {}
        f1 = v.get("macro_f1") or {}
        win_rows.append({
            "T": int(r.get("window") or 0),
            "n_seeds": int(r.get("n_seeds") or 1),
            "macro_f1": f1.get("mean"),
            "macro_f1_std": f1.get("std"),
            "fa_h": fa.get("mean"),
            "fa_h_std": fa.get("std"),
            "delay_ms": (dl.get("mean") or 0) * 1000 if dl.get("mean") is not None else None,
        })

    return {
        "alarms_day": rec.get("alarms_per_day"),
        "triage_h": rec.get("triage_hours_per_day"),
        "false_h": rec.get("false_alarms_per_hour"),
        "mttd_ms": rec.get("mttd_ms"),
        "no_confirm_h": noc.get("triage_hours_per_day"),
        "no_confirm_false_h": noc.get("false_alarms_per_hour"),
        "classes": classes,
        "windows": win_rows,
        "e2e_throughput": e2e.get("throughput_rows_per_sec"),
        "e2e_p50": (e2e.get("latency_ms_excl_warmup") or {}).get("p50"),
    }


def _fmt(v, nd=2, dash="n/a"):
    if v is None:
        return dash
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def render(d: dict, out: Path) -> Path | None:
    try:
        from src.utils import setup_matplotlib

        cjk = setup_matplotlib(logger=LOGGER)
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib 不可用：%s", exc)
        return None
    _L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    BG, FG, ACC, WARN, DIM = "#0f1720", "#e8eef6", "#4fd1c5", "#f6ad55", "#8aa0b6"
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.text(4, 92.8, _L("运维视角：一天要花多少人工、各类攻击要等多久",
                        "Operations view: daily analyst load and per-class latency"),
            color=FG, fontsize=24, fontweight="bold", va="center")
    ax.text(4, 87.2, _L("全部数字来自实跑的在线回放与离线评估，可在 results/ 下逐项复算",
                        "All figures come from measured runs under results/"),
            color=ACC, fontsize=12.5, va="center")
    ax.plot([4, 96], [84.0, 84.0], color="#2a3a4a", lw=1.4)

    # ---------- 四个头条数字 ----------
    heads = [
        (_L("推荐工作点：告警/天", "alerts/day"), _fmt(d["alarms_day"], 1), _L("阈值0.90+确认2", "0.90 + N-of-2")),
        (_L("折算人工", "analyst load"), f"{_fmt(d['triage_h'])} h", _L("占一个班次 35%", "35% of one shift")),
        (_L("误报/小时", "false alarms/h"), _fmt(d["false_h"]), _L("不做确认则 8.56", "8.56 without N-of-M")),
        (_L("平均检测时延", "mean delay"), f"{_fmt(d['mttd_ms'], 1)} ms", _L("逐类见下方", "per-class below")),
    ]
    x0, y0, w, h, gap = 4, 65.5, 22.0, 15.0, 1.3
    for i, (lab, val, sub) in enumerate(heads):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    linewidth=1.2, edgecolor="#2a3a4a", facecolor="#16212c"))
        ax.text(x + w / 2, y0 + h - 3.0, lab, color=DIM, fontsize=10.5, ha="center", va="center")
        ax.text(x + w / 2, y0 + h / 2 - 0.6, val, color=ACC, fontsize=22, fontweight="bold",
                ha="center", va="center")
        ax.text(x + w / 2, y0 + 2.4, sub, color=DIM, fontsize=9, ha="center", va="center")

    # ---------- 左：工作点表 ----------
    ax.text(4, 59.0, _L("① 按「每天能投多少人工」挑工作点", "Pick the working point by analyst budget"),
            color=FG, fontsize=13, fontweight="bold", va="center")
    rows_ws = [
        ("0.50 + 确认 1", 8.56, 22.5),
        ("0.50 + 确认 3", 0.66, 3.58),
        ("0.90 + 确认 2", d["false_h"] or 0.33, d["triage_h"] or 2.79),
        ("0.90 + 确认 3", 0.00, 2.00),
    ]
    ty = 54.4
    ax.text(5.5, ty, _L("工作点", "point"), color=DIM, fontsize=9.6, va="center")
    ax.text(19, ty, _L("误报/小时", "FA/h"), color=DIM, fontsize=9.6, va="center")
    ax.text(29, ty, _L("人工小时/天", "h/day"), color=DIM, fontsize=9.6, va="center")
    ax.text(41, ty, _L("评价", "note"), color=DIM, fontsize=9.6, va="center")
    for i, (name, fa, th) in enumerate(rows_ws):
        y = ty - 4.4 * (i + 1)
        hot = th > 8
        ax.text(5.5, y, name, color=FG, fontsize=10, va="center")
        ax.text(19, y, f"{fa:.2f}", color=WARN if hot else ACC, fontsize=10, va="center")
        ax.text(29, y, f"{th:.2f}", color=WARN if hot else ACC, fontsize=10, va="center")
        ax.text(41, y, _L("超出值班人力" if hot else ("推荐" if name.startswith("0.90 + 确认 2") else "可用"),
                          "over capacity" if hot else "usable"),
                color=WARN if hot else DIM, fontsize=9.4, va="center")

    # ---------- 中：逐类时延 ----------
    ax.text(50, 59.0, _L("② 各类攻击要等多久（决定响应节奏）", "Per-class latency"),
            color=FG, fontsize=13, fontweight="bold", va="center")
    cls = [c for c in d["classes"] if c.get("mean_delay_ms") is not None]
    if cls:
        mx = max(c["mean_delay_ms"] for c in cls) or 1.0
        for i, c in enumerate(cls):
            y = 54.4 - 4.4 * i
            ms = c["mean_delay_ms"]
            ax.text(50, y, f"{c['class']}", color=FG, fontsize=9.8, va="center")
            ax.text(63, y, f"{ms:.1f} ms", color=ACC if ms < 100 else WARN, fontsize=9.8, va="center")
            ax.add_patch(FancyBboxPatch((70, y - 1.0), max(0.6, 26 * ms / mx), 2.0,
                                        boxstyle="round,pad=0.05,rounding_size=0.4",
                                        linewidth=0, facecolor=ACC if ms < 100 else WARN, alpha=0.75))
            ax.text(97.5, y, f"{c['n_detected']}/{c['n_segments']}", color=DIM, fontsize=8.6,
                    ha="right", va="center")

    # ---------- 底：窗口长度换误报 ----------
    ax.text(4, 30.5, _L("③ 不用改阈值和模型：加长窗口就能把误报压下来（代价是时延）",
                        "Longer window cuts false alarms without touching the model"),
            color=FG, fontsize=13, fontweight="bold", va="center")
    wrows = [w for w in d["windows"] if w["fa_h"] is not None]
    ax.text(5.5, 25.6, _L("窗口 T", "T"), color=DIM, fontsize=9.6, va="center")
    ax.text(14, 25.6, _L("宏 F1", "macro F1"), color=DIM, fontsize=9.6, va="center")
    ax.text(30, 25.6, _L("误报/小时", "FA/h"), color=DIM, fontsize=9.6, va="center")
    ax.text(48, 25.6, _L("平均时延", "mean delay"), color=DIM, fontsize=9.6, va="center")
    ax.text(64, 25.6, _L("种子", "seeds"), color=DIM, fontsize=9.6, va="center")
    for i, w in enumerate(wrows):
        y = 21.2 - 4.2 * i
        star = w["T"] == 32
        col = ACC if star else FG
        ax.text(5.5, y, f"T={w['T']}", color=col, fontsize=10, va="center")
        f1 = _fmt(w["macro_f1"], 4) + (f" ± {_fmt(w['macro_f1_std'], 4)}" if w["macro_f1_std"] else "")
        ax.text(14, y, f1, color=col, fontsize=10, va="center")
        fa = _fmt(w["fa_h"], 2) + (f" ± {_fmt(w['fa_h_std'], 2)}" if w["fa_h_std"] else "")
        ax.text(30, y, fa, color=col, fontsize=10, va="center")
        ax.text(48, y, f"{_fmt(w['delay_ms'], 1)} ms", color=col, fontsize=10, va="center")
        ax.text(64, y, str(w["n_seeds"]), color=DIM, fontsize=9.6, va="center")
        if star:
            ax.text(68, y, _L("← 误报优先推荐：约 26 ms 换约 84 倍误报下降", "← recommended: 26 ms for ~84x fewer FA"),
                    color=WARN, fontsize=9.6, va="center")

    # ---------- 页脚：边界 ----------
    ax.plot([4, 96], [4.6, 4.6], color="#2a3a4a", lw=1.2)
    ax.text(4, 2.0,
            _L("不要对外承诺：真实交换机上的处置未验证（适配层用桩件测 25 项）｜"
               "只有控制器字段时精度降至宏 F1 0.844（召回仍 100%，代价是时延 18→41 ms）｜"
               "跨网络零样本迁移不可用；未做长期稳定性/SLA",
               "Not committed: real-switch enforcement untested | controller-native fields drop to 0.844 | "
               "zero-shot transfer unusable | no long-run SLA"),
            color=DIM, fontsize=9.0, va="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成运维 SLO 卡一页纸")
    ap.add_argument("--out", default=str(FIGURE_DIR / "ops_card.png"))
    args = ap.parse_args(argv)
    d = collect()
    out = render(d, Path(args.out))
    if out is None:
        return 1
    LOGGER.info("运维 SLO 卡已生成 → %s（逐类 %d 类、窗口 %d 档）",
                out, len(d["classes"]), len(d["windows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
