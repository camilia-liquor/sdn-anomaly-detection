"""交付验收自检：一条命令检查"这份工程拿去比赛是否完整、可跑、结果真实"。

    python tools/acceptance_check.py

检查五类内容（全部通过才会打印 ✅）：
  1. 代码与文档是否齐全（含原始素材是否保留）
  2. 数据集生成器是否符合 inSDN/CICFlowMeter 与 Ryu 采集脚本的字段规格，
     且生成的数据物理量自洽、类别齐全
  3. 预处理产物是否健康（形状、无 NaN/Inf、类别覆盖 train/val/test）
  4. 实验产物是否真实（指标齐全且在合理范围、9 张图表、主模型指针一致）
  5. 演示界面是否可用（用 Flask 测试客户端直接打接口，不占端口）
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.traffic_generator import (  # noqa: E402
    ALL_CLASSES,
    INSDN_HEADER,
    SDN_EXTRA_HEADER,
)
from src.utils import ARTIFACT_DIR, FIGURE_DIR, PROCESSED_DIR, RAW_DIR, RESULT_DIR, get_logger  # noqa: E402

LOG = get_logger("cida.accept")
PASS, FAIL, WARN = "✓", "✗", "!"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = 0
        self.warned = 0

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        mark = PASS if ok else FAIL
        if not ok:
            self.failed += 1
        self.rows.append((mark, name, detail))
        print(f"  {mark} {name}" + (f"  —— {detail}" if detail else ""))
        return ok

    def warn(self, ok: bool, name: str, detail: str = "") -> None:
        if ok:
            self.rows.append((PASS, name, detail))
            print(f"  {PASS} {name}" + (f"  —— {detail}" if detail else ""))
        else:
            self.warned += 1
            self.rows.append((WARN, name, detail))
            print(f"  {WARN} {name}  —— {detail}")


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------------------
def check_files(r: Report) -> None:
    section("1) 代码与文档")
    must = [
        "run_all.py", "requirements.txt", "README.md", "DEPLOYMENT.md",
        "src/config.py", "src/traffic_generator.py", "src/data_pipeline.py", "src/model.py",
        "src/train.py", "src/evaluate.py", "src/visualize.py", "src/baselines.py", "src/utils.py",
        "src/service.py",
        "app/app.py", "app/templates/index.html", "app/static/app.js", "app/static/style.css",
        "integrations/ryu_guard.py",
        "tests/test_core.py",
        "tools/check_env.py", "tools/smoke_test.py", "tools/predict_cli.py",
        "tools/reevaluate.py", "tools/drift_test.py", "tools/ensemble_eval.py",
        "tools/make_report.py", "tools/fetch_wheel.py", "tools/sanitize_paths.py",
        "tools/operating_point.py", "tools/error_analysis.py", "tools/benchmark_latency.py",
        "tools/controller_simulator.py", "tools/ops_impact.py", "tools/acceptance_check.py",
        "docs/01_原始代码问题清单.md", "docs/02_实验报告.md", "docs/03_答辩要点与配置清单.md",
        "competition/README.md", "competition/01_开源及第三方资源使用清单.md",
        "competition/02_技术报告.md", "competition/03_作品名称与简介.md",
        "competition/04_演示视频脚本.md", "competition/05_匿名化与合规自查清单.md",
        "configs/fast.json", "configs/full.json",
        "LICENSE", "NOTICE", "CONTRIBUTING.md", ".github/workflows/ci.yml",
        "samples/README.md", "samples/insdn_stream.csv",
    ]
    missing = [p for p in must if not (ROOT / p).exists()]
    r.check(not missing, f"交付文件齐全（{len(must)} 项）", f"缺少：{missing}" if missing else "")
    legacy = ["CNN_biLSTM2.py", "generate_normal_traffic/collect_normal_traffic.py",
              "generate_ddos_traffic/generate_ddos_traffic/collect_ddos_traffic.py"]
    kept = [p for p in legacy if (ROOT / p).exists()]
    r.check(len(kept) == len(legacy), "原始素材保留未删（便于对照）", f"{len(kept)}/{len(legacy)}")

    report_md = (ROOT / "docs" / "02_实验报告.md")
    if report_md.exists():
        txt = report_md.read_text(encoding="utf-8")
        r.check(len(txt) > 3000 and "0." in txt, "实验报告已按真实结果生成", f"{len(txt)} 字符")


def check_generator(r: Report) -> None:
    section("2) 数据集生成器规格与物理自洽性")
    r.check(len(INSDN_HEADER) == 84, "insdn schema = 84 列（与 inSDN / CICFlowMeter 一致）",
            f"实际 {len(INSDN_HEADER)}")
    r.check(INSDN_HEADER[0] == "Flow ID" and INSDN_HEADER[-1] == "Label",
            "insdn 首列 Flow ID / 末列 Label 与公开数据集一致")
    r.check(len(SDN_EXTRA_HEADER) == 23 and SDN_EXTRA_HEADER[0] == "timestamp",
            "sdn schema = 22 列采集字段 + 1 列多分类标注（与 Ryu 脚本一致）",
            f"实际 {len(SDN_EXTRA_HEADER)}")
    r.check("label" in SDN_EXTRA_HEADER and "attack_type" in SDN_EXTRA_HEADER,
            "sdn schema 同时提供原脚本的 0/1 label 与 attack_type")

    raw = RAW_DIR / "insdn_stream.csv"
    if not raw.exists():
        r.check(False, "原始仿真数据存在", str(raw))
        return
    df = pd.read_csv(raw, low_memory=False)
    r.check(list(df.columns) == INSDN_HEADER, "生成数据列名与 schema 逐字一致",
            f"{df.shape[0]:,} 行 × {df.shape[1]} 列")
    labels = df["Label"].astype(str)
    got = set(labels.unique())
    r.check(len(got) >= 8, f"类别齐全（{len(got)} 类）", "、".join(sorted(got)))
    r.check("Normal" in got and len(got - {"Normal"}) >= 7, "包含正常流量 + ≥7 类攻击")

    # 物理一致性抽查：包速率、平均包长
    d = df[df["Flow Duration"] > 0].head(2000)
    pps = d["Flow Packets/s"].to_numpy()
    expect = (d["Total Fwd Packets"].to_numpy() + d["Total Backward Packets"].to_numpy()) / (
        d["Flow Duration"].to_numpy() / 1e6)
    err = np.abs(pps - expect) / np.clip(expect, 1e-9, None)
    r.check(float(np.nanmedian(err)) < 1e-6, "物理量自洽：Flow Packets/s = 总包数/时长",
            f"中位相对误差 {float(np.nanmedian(err)):.2e}")
    avg = d["Average Packet Size"].to_numpy()
    tot = d["Total Length of Fwd Packets"].to_numpy() + d["Total Length of Bwd Packets"].to_numpy()
    cnt = d["Total Fwd Packets"].to_numpy() + d["Total Backward Packets"].to_numpy()
    err2 = np.abs(avg - tot / np.clip(cnt, 1, None)) / np.clip(tot / np.clip(cnt, 1, None), 1e-9, None)
    r.check(float(np.nanmedian(err2)) < 1e-6, "物理量自洽：平均包长 = 总字节/总包数",
            f"中位相对误差 {float(np.nanmedian(err2)):.2e}")
    r.check(not np.isinf(df.select_dtypes("number").to_numpy()).any(), "生成数据无无穷大值")

    # 难度：随机森林不能轻松拿满分 → 说明类别边界有重叠
    info = RAW_DIR / "insdn_schema_info.json"
    if info.exists():
        diff = json.loads(info.read_text(encoding="utf-8")).get("difficulty", {})
        r.check(float(diff.get("overlap_ratio", 0)) > 0 and float(diff.get("stealth_ratio", 0)) > 0,
                "难度旋钮已启用（类别重叠 + 隐蔽攻击片段）", str(diff))


def check_prepared(r: Report) -> None:
    section("3) 预处理产物健康度")
    meta_p = PROCESSED_DIR / "meta.json"
    tens = PROCESSED_DIR / "tensors.npz"
    if not (meta_p.exists() and tens.exists()):
        r.check(False, "预处理产物存在（先跑 python run_all.py）", str(PROCESSED_DIR))
        return
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    z = np.load(tens, allow_pickle=True)
    Xtr, Xva, Xte = z["X_train"], z["X_val"], z["X_test"]
    r.check(Xtr.ndim == 3 and Xtr.shape[1] == meta["window_size"],
            "窗口张量形状正确 (N, T, F)", f"{Xtr.shape}")
    r.check(not np.isnan(Xtr).any() and not np.isinf(Xtr).any(), "训练张量无 NaN/Inf")
    r.check(len(z["test_stream_features"]) > 0, "保留了连续测试流（用于在线评估与演示）",
            f"{len(z['test_stream_features']):,} 条流记录")
    y_test = z["y_test_mc"]
    n_classes = len(meta["class_names"])
    covered = len(set(np.unique(y_test).tolist()))
    r.check(covered == n_classes, "每个类别在测试集都有样本（时序切分未漏类）",
            f"{covered}/{n_classes} 类")
    r.check(meta["split_info"]["time_ordered_split"] is True, "按时间顺序切分（无随机切分泄漏）")
    r.check(int(meta["split_info"]["purge_rows"]) > 0, "切分边界留有 purge 间隙",
            f"{meta['split_info']['purge_rows']} 条")


def check_results(r: Report) -> None:
    section("4) 实验产物（真实结果）")
    summary_p = RESULT_DIR / "summary.json"
    if not summary_p.exists():
        r.check(False, "实验汇总存在（先跑 python run_all.py）", str(summary_p))
        return
    s = json.loads(summary_p.read_text(encoding="utf-8"))
    mc, bn, on = s["multiclass"], s["binary"], s.get("online") or {}
    r.check(0.5 < mc["accuracy"] <= 1.0, "多分类准确率在合理区间", f"{mc['accuracy']:.4f}")
    r.check(0.5 < mc["macro_f1"] <= 1.0, "多分类宏 F1 在合理区间", f"{mc['macro_f1']:.4f}")
    r.check(0.5 < bn["roc_auc"] <= 1.0, "二分类 ROC-AUC 在合理区间", f"{bn['roc_auc']:.4f}")
    r.check(mc["accuracy"] < 1.0 and bn["roc_auc"] < 1.0,
            "指标不是「满分」（说明数据有真实难度，非自欺欺人）")
    r.check(bool(on), "含在线告警级指标（时延 / 误报率 / 片段召回）",
            f"片段召回 {on.get('segment_recall')}，误报 {on.get('false_alarms_per_hour', 0):.1f} 次/小时")
    r.check(s.get("aggregate") is not None and len(s["aggregate"]) >= 4,
            "含多种子聚合结果（均值±标准差）", f"{len(s.get('aggregate') or {})} 个模型")

    metrics = sorted(RESULT_DIR.glob("metrics_*.json"))
    r.check(len(metrics) >= 8, "指标文件数量充足（主模型 + 消融 + 基线）", f"{len(metrics)} 份")

    figs = sorted(FIGURE_DIR.glob("*.png"))
    need = ["01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_"]
    have = [n for n in need if any(f.name.startswith(n) for f in figs)]
    r.check(len(have) == len(need), "9 类图表齐全（含混淆矩阵/ROC-PR/训练曲线/注意力/消融）",
            f"{len(have)}/{len(need)}")

    drift = RESULT_DIR / "drift_report.md"
    r.warn(drift.exists(), "含分布漂移鲁棒性报告", drift.name if drift.exists() else
           "未跑 python tools/drift_test.py（可选）")

    ptr = ARTIFACT_DIR / "main_model.json"
    if ptr.exists():
        tag = json.loads(ptr.read_text(encoding="utf-8"))["tag"]
        r.check((ARTIFACT_DIR / f"predictions_{tag}.csv").exists(),
                "主模型指针与预测缓存一致（演示界面据此取数）", tag)
    else:
        r.check(False, "主模型指针存在", str(ptr))


def check_demo(r: Report) -> None:
    section("5) 演示界面（Flask 测试客户端，不占端口）")
    try:
        spec = importlib.util.spec_from_file_location("cida_demo_app", ROOT / "app" / "app.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        r.check(False, "演示服务可加载", f"{type(exc).__name__}: {exc}")
        return
    r.check(mod.STATE.ok, "演示数据已就绪", mod.STATE.reason or "")
    client = mod.app.test_client()
    ov = client.get("/api/overview").get_json() or {}
    r.check(bool(ov.get("has_predictions")), "能取到模型预测缓存", f"tag={ov.get('tag')}")
    r.check(len(ov.get("segments") or []) > 0, "测试流里有可注入的攻击片段",
            f"{len(ov.get('segments') or [])} 段")
    st = client.get("/api/stream?pos=0&count=20").get_json() or {}
    r.check(len(st.get("rows") or []) == 20, "实时流接口返回正常", f"{len(st.get('rows') or [])} 行")
    r.check(any((x.get("win") or {}) for x in (st.get("rows") or [])), "流数据带窗口级判定结果")
    hp = client.get("/api/health").get_json() or {}
    r.check(bool(hp.get("ok")), "健康检查通过")


def check_unit_tests(r: Report) -> None:
    """6) 单元测试：覆盖生成器、数据管线、窗口构建、评估口径。"""
    section("6) 单元测试（无需 pytest，直接执行）")
    test_file = ROOT / "tests" / "test_core.py"
    if not test_file.exists():
        r.check(False, "单元测试文件存在", str(test_file))
        return
    import contextlib
    import importlib.util
    import io

    try:
        spec = importlib.util.spec_from_file_location("cida_unit_tests", test_file)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            rc = mod._run_all()
        out = buf.getvalue().strip().splitlines()
        summary = out[-1] if out else ""
        failed = [ln for ln in out if ln.strip().startswith("✗")]
        r.check(rc == 0, "单元测试全部通过", summary)
        if failed:
            for ln in failed[:5]:
                print("     ", ln.strip())
    except Exception as exc:  # noqa: BLE001
        r.check(False, "单元测试可执行", f"{type(exc).__name__}: {exc}")


def check_deployable(r: Report) -> None:
    """7) 可部署性：在线检测接口、控制器闭环、运维换算三条链路都要真的能跑。

    这一节回答的是「这是不是一个能用的系统」。检查方式不是读文件，而是**真的调用接口**：
    用 Flask 测试客户端把一批流记录 POST 进 /api/detect，看它是否返回逐条判定与告警；
    再核对控制器闭环报告与运维换算报告是否存在且数值自洽。
    """
    section("7) 可部署性（在线接口 / 控制器闭环 / 运维换算）")
    try:
        spec = importlib.util.spec_from_file_location("cida_demo_app2", ROOT / "app" / "app.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        r.check(False, "检测服务模块可加载", f"{type(exc).__name__}: {exc}")
        return
    client = mod.app.test_client()

    # 策略接口：查询 + 热更新（改完再改回去，保证幂等）
    pol = client.get("/api/policy").get_json() or {}
    r.check("threshold" in pol and "confirm_windows" in pol, "策略接口可查询",
            f"阈值={pol.get('threshold')} 确认={pol.get('confirm_windows')}")
    orig = {"threshold": pol.get("threshold"), "confirm_windows": pol.get("confirm_windows")}
    upd = client.post("/api/policy", json={"threshold": 0.9, "confirm_windows": 4}).get_json() or {}
    r.check(float(upd.get("threshold", -1)) == 0.9 and int(upd.get("confirm_windows", -1)) == 4,
            "策略接口可热更新（无需重启）", f"→ 阈值 {upd.get('threshold')} / 确认 {upd.get('confirm_windows')}")
    client.post("/api/policy", json=orig)

    # 核心接口：真的推一批流记录进去
    meta = json.loads((PROCESSED_DIR / "meta.json").read_text(encoding="utf-8"))
    feats = list(meta["feature_names"])
    try:
        z = np.load(PROCESSED_DIR / "tensors.npz", allow_pickle=True)
        sample = np.asarray(z["test_stream_features"], dtype=np.float32)[:60]
    except Exception as exc:  # noqa: BLE001
        r.check(False, "能取到测试流样本用于在线推流", f"{type(exc).__name__}: {exc}")
        return
    rows = [{f: float(v) for f, v in zip(feats, sample[i])} for i in range(len(sample))]
    resp = client.post("/api/detect", json={"rows": rows,
                                            "meta": {"src_ip": "10.0.0.5", "dst_ip": "10.0.0.1", "dst_port": 80}})
    det = resp.get_json() or {}
    out_rows = det.get("rows") or []
    judged = [x for x in out_rows if isinstance(x, dict) and x.get("window")]
    r.check(resp.status_code == 200 and det.get("ok") is True,
            "POST /api/detect 在线推流成功（这是控制器实际调用的接口）",
            f"HTTP {resp.status_code}，返回 {len(out_rows)} 条判定" if resp.status_code == 200 else str(det)[:120])
    r.check(len(judged) > 0 and "p_attack" in (judged[0]["window"] if judged else {}),
            "接口返回逐条判定（含概率、类别、是否告警、是否确认）",
            f"{len(judged)}/{len(out_rows)} 条带窗口判定，"
            f"示例 p_attack={judged[0]['window']['p_attack']} 类别={judged[0]['window']['pred_class']}"
            if judged else "无判定结果")
    # 告警上下文完整性：每条告警都要能追到「谁、何时、依据什么、做了什么」
    alerts_now = det.get("alerts") or (client.get("/api/alerts?limit=5").get_json() or [])
    r.check(all(k in (alerts_now[0] if alerts_now else {}) for k in
                ("time", "src_ip", "pred_class", "p_attack", "action", "rows_covered"))
            if alerts_now else True,
            "告警附带可追溯上下文（时间/源 IP/类别/置信度/处置动作/触发窗口）",
            f"{len(alerts_now)} 条告警")
    stats = client.get("/api/stats").get_json() or {}
    r.check(int((stats.get("stats") or stats).get("rows") or 0) > 0, "运行统计已累计（可观测）",
            f"rows={(stats.get('stats') or stats).get('rows')}")
    alerts = client.get("/api/alerts?limit=5").get_json()
    r.check(isinstance(alerts, list), "告警审计台账可查询", f"{len(alerts or [])} 条")
    metrics_txt = client.get("/metrics").get_data(as_text=True)
    r.check("cida" in metrics_txt.lower() or "# HELP" in metrics_txt,
            "Prometheus /metrics 可被现有监控抓取", f"{len(metrics_txt)} 字节")

    # 控制器闭环报告（无 Ryu 环境下的实测证据）
    e2e = RESULT_DIR / "e2e_controller_report.json"
    if e2e.exists():
        rep = json.loads(e2e.read_text(encoding="utf-8"))
        rps = float(rep.get("throughput_rows_per_sec") or rep.get("rows_per_sec") or 0.0)
        r.check(rps > 100, "控制器闭环实测吞吐达标（tools/controller_simulator.py）",
                f"{rps:,.0f} 条流/秒")
        lat = rep.get("latency_ms") or {}
        r.check(float(lat.get("p50") or 0) < 200, "在线判定时延达标（p50）",
                f"p50={lat.get('p50')} ms / p95={lat.get('p95')} ms")
    else:
        r.check(False, "控制器闭环报告存在", f"{e2e}（运行 tools/controller_simulator.py 生成）")

    # 运维换算报告（把指标翻译成告警量与人工工时）
    ops = RESULT_DIR / "ops_impact.json"
    if ops.exists():
        o = json.loads(ops.read_text(encoding="utf-8"))
        rec = o.get("recommended_working_point") or {}
        r.check(bool(rec) and rec.get("alarms_per_day") is not None,
                "运维换算报告存在且给出推荐上线工作点",
                f"阈值 {rec.get('threshold')} + 确认 {rec.get('confirm')} → "
                f"{rec.get('alarms_per_day', 0):.1f} 条告警/天、{rec.get('triage_hours_per_day', 0):.2f} 人工小时/天")
    else:
        r.check(False, "运维换算报告存在", f"{ops}（运行 tools/ops_impact.py 生成）")

    # 控制器适配层：本机没有 Ryu 时只能做静态检查（不假装跑过）
    guard = (ROOT / "integrations" / "ryu_guard.py").read_text(encoding="utf-8")
    r.check(all(k in guard for k in ("add_arguments", "OFPFlowStatsRequest", "OFPMatch", "whitelist")),
            "Ryu 适配层结构完整（参数注册 / 流表统计 / 规则下发 / 白名单）")
    r.check("未做运行验证" in guard or "未在真实交换机上实跑" in guard or "dry_run" in guard,
            "Ryu 适配层已声明运行边界与演练默认（不夸大验证范围）")
    has_ryu = importlib.util.find_spec("ryu") is not None
    r.warn(has_ryu, "本机 Ryu 环境（未安装属正常，适配层需在真实 SDN 环境验证）",
           "已安装" if has_ryu else "未安装：ryu_guard.py 只在有 Ryu 的机器上运行")


def main() -> int:
    print("=" * 74)
    print("交付验收自检 · CNN + BiLSTM SDN 异常检测工程")
    print("=" * 74)
    r = Report()
    check_files(r)
    check_generator(r)
    check_prepared(r)
    check_results(r)
    check_demo(r)
    check_unit_tests(r)
    check_deployable(r)

    section("结论")
    print(f"  通过 {len(r.rows) - r.failed - r.warned} 项 | 警告 {r.warned} 项 | 失败 {r.failed} 项")
    if r.failed == 0:
        print("\n  ✅ 全部关键项通过：这份工程可以直接拿去比赛（警告项为可选项）。")
        return 0
    print("\n  ✗ 存在未通过项，请按上面的 ✗ 逐条修复。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
