"""Ryu 适配层的行为测试（**本机不需要安装 Ryu**）。

为什么要有这个文件
------------------
`integrations/ryu_guard.py` 需要 Ryu + OpenFlow 交换机才能在真机上跑，
本工程在无 Ryu 的机器上开发，因此它无法用常规方式验证。但"无法在真机跑"
不等于"无法验证"——本文件用**最小桩件**把 `ryu` 相关模块替换掉，然后真的调用
适配层的核心逻辑，把下面这些**接口约定与安全默认值**锁死：

1. 适配层组装出的每一条流记录，字段必须**覆盖控制器原生口径模型所需的全部特征名**
   （直接读 `data/processed_sdn/meta.json`）。字段名对不上，检测服务会直接拒绝服务——
   这是"控制器接得进去吗"的硬性接口契约。
2. **默认必须是演练模式**：不加 `--execute` 时绝不能发出任何流表消息。
3. **白名单必须真的生效**：白名单内的源 IP 即使开了 `--execute` 也不下发规则。
4. **`block` 必须真的下发规则**：要能观察到一条带 `hard_timeout` 的 `OFPFlowMod`
   （早期实现把 `block` 写成了只记账不下发，本测试就是为了防这个回归）。
5. 审计台账必须逐条落盘，且字段可追溯。

用法：``python tests/test_ryu_guard.py``（退出码 0 = 全部通过）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL = "✓", "✗"
_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  —— {detail}" if detail else ""))
    return bool(ok)


# ======================================================================================
#  最小桩件：把 ryu 相关模块塞进 sys.modules，让适配层可以在没有 Ryu 的情况下被导入
# ======================================================================================
class _StubLogger:
    def __init__(self) -> None:
        self._log = logging.getLogger("ryu_guard_stub")

    def info(self, *a, **k):
        self._log.debug(*a)

    def warning(self, *a, **k):
        self._log.debug(*a)

    def error(self, *a, **k):
        self._log.error(*a)


class _StubSimpleSwitch13:
    """替代 ryu.app.simple_switch_13.SimpleSwitch13。"""

    def __init__(self, *args, **kwargs):
        self.logger = _StubLogger()
        self.sent: list = []


def _install_ryu_stubs():
    """把桩件装进 sys.modules，并返回一个「还原」函数。

    ⚠️ 必须还原：否则桩件会污染后续代码（例如 `importlib.util.find_spec("ryu")`
    会因为假模块没有 `__spec__` 直接抛 ValueError）。适配层在导入时就已经把
    `set_ev_cls` / `hub` 等对象绑到自己的全局命名空间，因此导入完成后即可安全还原。
    """
    def _noop_decorator(*dargs, **dkwargs):
        def wrap(fn):
            return fn
        return wrap

    ryu = types.ModuleType("ryu")
    ryu_app = types.ModuleType("ryu.app")
    simple = types.ModuleType("ryu.app.simple_switch_13")
    simple.SimpleSwitch13 = _StubSimpleSwitch13
    ctrl = types.ModuleType("ryu.controller")
    ofp_event = types.ModuleType("ryu.controller.ofp_event")
    ofp_event.EventOFPStateChange = type("EventOFPStateChange", (), {})
    ofp_event.EventOFPFlowStatsReply = type("EventOFPFlowStatsReply", (), {})
    handler = types.ModuleType("ryu.controller.handler")
    handler.MAIN_DISPATCHER = 1
    handler.DEAD_DISPATCHER = 2
    handler.set_ev_cls = _noop_decorator
    lib = types.ModuleType("ryu.lib")
    hub = types.ModuleType("ryu.lib.hub")
    hub.spawn = lambda fn, *a, **k: None          # 不启动后台线程，避免测试里空转
    hub.sleep = lambda *_a, **_k: None
    # 回读校验用 hub.Event 做异步等待；替身用标准库 Event（wait(timeout) 语义一致），
    # 这样测试既能模拟"回复到了"，也能模拟"超时"（不 set 即超时）。
    import threading as _threading
    hub.Event = _threading.Event
    packet = types.ModuleType("ryu.lib.packet")
    packet.ether_types = types.SimpleNamespace(ETH_TYPE_IP=0x0800)

    ryu.app = ryu_app
    ryu.controller = ctrl
    ryu.lib = lib
    stubs = {
        "ryu": ryu, "ryu.app": ryu_app, "ryu.app.simple_switch_13": simple,
        "ryu.controller": ctrl, "ryu.controller.ofp_event": ofp_event,
        "ryu.controller.handler": handler, "ryu.lib": lib, "ryu.lib.hub": hub,
        "ryu.lib.packet": packet,
    }
    names = list(stubs)
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules.update(stubs)

    def restore() -> None:
        for n, old in saved.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old

    return restore


# ======================================================================================
#  假的 OpenFlow 消息对象
# ======================================================================================
class _FakeParser:
    """替代 datapath.ofproto_parser：只记录被构造的参数，便于断言。"""

    def __init__(self) -> None:
        self.made: list[tuple[str, dict]] = []

    def _mk(self, name):
        def factory(*args, **kwargs):
            # Ryu 的部分消息构造函数支持位置参数（如 OFPInstructionActions(type, actions)），
            # 桩件两种调用方式都要能接住，否则会误报成被测代码的问题。
            self.made.append((name, {"args": args, **kwargs}))
            return {"__kind__": name, "args": args, **kwargs}
        return factory

    def __getattr__(self, item):
        return self._mk(item)


class _FakeOfproto:
    OFPIT_APPLY_ACTIONS = 4
    OFPIT_METER = 6
    OFPFC_ADD = 0
    OFPMC_ADD = 0
    OFPMF_KBPS = 1
    # 回滚用到的常量（OF1.3）
    OFPFC_DELETE_STRICT = 4
    OFPMC_DELETE = 2
    OFPP_ANY = 0xffffffff
    OFPG_ANY = 0xffffffff


class _FakeDatapath:
    def __init__(self, dp_id: int = 0x1) -> None:
        self.id = dp_id
        self.ofproto = _FakeOfproto()
        self.ofproto_parser = _FakeParser()
        self.sent: list = []
        self._xid = 1000

    def send_msg(self, msg) -> None:
        # 真实 Ryu 的 Datapath.send_msg 会给未分配 xid 的消息分配一个（回复必须沿用同一 xid），
        # 桩件必须复现这一步：回读校验正是靠 xid 把回复与请求配对的。
        if isinstance(msg, dict) and not msg.get("xid"):
            self._xid += 1
            msg["xid"] = self._xid
        self.sent.append(msg)

    def flowmods(self) -> list[dict]:
        return [m for m in self.sent if isinstance(m, dict) and m.get("__kind__") == "OFPFlowMod"]

    def flow_stats_requests(self) -> list[dict]:
        return [m for m in self.sent
                if isinstance(m, dict) and m.get("__kind__") == "OFPFlowStatsRequest"]

    def deletes(self) -> list[dict]:
        """回滚用的 DELETESTRICT 流表删除（command=OFPFC_DELETE_STRICT）。"""
        return [m for m in self.flowmods() if int(m.get("command", -1)) == _FakeOfproto.OFPFC_DELETE_STRICT]


class _FakeInstruction:
    """流表项里的一条指令（回读校验要据此判断"这条规则到底做了什么"）。"""

    def __init__(self, itype: int, actions=None) -> None:
        self.type = itype
        self.actions = actions if actions is not None else []


class _FakeFlowStat:
    def __init__(self, **kw) -> None:
        self.priority = kw.pop("priority", 1)
        self.match = kw.pop("match", {})
        self.instructions = kw.pop("instructions", [])
        self.duration_sec = kw.pop("duration_sec", 10)
        self.duration_nsec = kw.pop("duration_nsec", 0)
        self.idle_timeout = kw.pop("idle_timeout", 0)
        self.hard_timeout = kw.pop("hard_timeout", 0)
        self.flags = kw.pop("flags", 0)
        self.packet_count = kw.pop("packet_count", 100)
        self.byte_count = kw.pop("byte_count", 6400)


class _FakeMsg:
    def __init__(self, datapath, body, xid: int | None = None) -> None:
        self.datapath = datapath
        self.body = body
        self.xid = xid


class _FakeEvent:
    def __init__(self, datapath, body, xid: int | None = None) -> None:
        self.msg = _FakeMsg(datapath, body, xid)


def _load_guard(tmpdir: Path):
    """加载适配层模块（已装好 Ryu 桩件），导入完成后立即还原 sys.modules。"""
    restore = _install_ryu_stubs()
    try:
        spec = importlib.util.spec_from_file_location("cida_ryu_guard",
                                                      ROOT / "integrations" / "ryu_guard.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        restore()
    return mod


# ======================================================================================
#  测试用例
# ======================================================================================
def test_arguments_registered(mod) -> None:
    ap = argparse.ArgumentParser()
    mod.add_arguments(ap)
    args = ap.parse_args([])
    check(args.execute is False, "默认不开启真实处置（演练模式为默认值）",
          f"--execute 缺省 = {args.execute}")
    check(hasattr(args, "detect_url") and args.detect_url.startswith("http"),
          "提供 --detect-url 参数", args.detect_url)
    check(int(args.block_seconds) > 0, "处置有有效期（--block-seconds > 0，到期自动解除）",
          f"{args.block_seconds} 秒")
    check(isinstance(args.whitelist, list), "提供 --whitelist 参数（默认空，需上线时填）")
    check(getattr(args, "verify_readback", None) is False,
          "回读校验默认关闭（--verify-readback 需显式打开）",
          f"verify_readback={getattr(args, 'verify_readback', None)}")
    check(float(getattr(args, "verify_timeout", 0)) > 0 and getattr(args, "rollback_on_fail", None) is True,
          "回读校验有超时且默认失败即回滚（--no-rollback-on-fail 可关）",
          f"verify_timeout={getattr(args, 'verify_timeout', None)} "
          f"rollback_on_fail={getattr(args, 'rollback_on_fail', None)}")


def _make_guard(mod, tmpdir: Path, *, execute: bool, whitelist=(), verify: bool = False,
                verify_timeout: float = 2.0, rollback: bool = True):
    app = mod.GuardSwitch.__new__(mod.GuardSwitch)      # 绕过需要 RyuApp 的 __init__
    app.datapaths = {}
    app.detect_url = "http://127.0.0.1:0"
    app.execute = execute
    app.block_seconds = 60
    app.whitelist = set(whitelist)
    app.audit_path = str(tmpdir / "audit_ryu.jsonl")
    app.blocked = {}
    app.blocked_action = {}       # 与 __init__ 保持一致：幂等要能区分"重复"与"升级"
    # 与 __init__ 保持一致：回读校验的可选字段（桩件绕过 __init__，这类漂移本身是隐患）
    app.verify_readback = verify
    app.verify_timeout = verify_timeout
    app.rollback_on_fail = rollback
    app._pending_verify = {}
    app._n_verified = 0
    app._n_rollbacks = 0
    app.logger = _StubLogger()
    return app


def _sample_body():
    return [
        _FakeFlowStat(match={"ip_proto": 6, "tcp_src": 1234, "tcp_dst": 80,
                             "ipv4_src": "10.0.0.5", "ipv4_dst": "10.0.0.1"},
                      packet_count=1200, byte_count=76800, duration_sec=12, duration_nsec=0),
        _FakeFlowStat(match={"ip_proto": 17, "udp_src": 53, "udp_dst": 5353,
                             "ipv4_src": "10.0.0.9", "ipv4_dst": "10.0.0.1"},
                      packet_count=8, byte_count=640, duration_sec=2, duration_nsec=0),
        _FakeFlowStat(match={"ip_proto": 1, "icmpv4_type": 8, "icmpv4_code": 0,
                             "ipv4_src": "10.0.0.7", "ipv4_dst": "10.0.0.1"},
                      packet_count=4, byte_count=256, duration_sec=1, duration_nsec=0),
    ]


def test_payload_matches_model_features(mod, tmpdir: Path) -> None:
    """最关键的一条：控制器组装的字段必须覆盖模型要求的全部特征名。"""
    meta_path = ROOT / "data" / "processed_sdn" / "meta.json"
    if not meta_path.exists():
        check(False, "控制器原生口径的 meta.json 存在", str(meta_path))
        return
    want = set(json.loads(meta_path.read_text(encoding="utf-8"))["feature_names"])

    guard = _make_guard(mod, tmpdir, execute=False)
    captured: dict = {}

    def fake_call(rows, datapath=None):
        captured["rows"] = rows
        return {"ok": True, "alerts": []}

    guard._call_detect = fake_call  # type: ignore[assignment]
    dp = _FakeDatapath()
    guard._flow_stats_reply_handler(_FakeEvent(dp, _sample_body()))

    rows = captured.get("rows") or []
    check(len(rows) == 3, "每条流表统计都组装成一条流记录", f"{len(rows)} 条")
    if not rows:
        return
    got = set(rows[0].keys())
    missing = sorted(want - got)
    check(not missing, f"记录字段覆盖模型全部 {len(want)} 个特征名（接口契约）",
          f"缺失：{missing}" if missing else "无缺失（额外字段为上下文信息，服务会忽略）")
    # 数值可解析（服务端会 float() 转换）
    bad = [k for k in want if rows[0].get(k) in (None, "")]
    check(not bad, "特征值都非空（可被服务端安全 float 转换）", f"空值字段：{bad}" if bad else "")
    check(all(isinstance(rows[0][k], (int, float)) for k in want if k in rows[0]),
          "特征值为数值类型")
    check("Src IP" in rows[0] and rows[0]["Src IP"] == "10.0.0.5",
          "附带审计所需的上下文（源 IP）")


def test_dry_run_sends_nothing(mod, tmpdir: Path) -> None:
    guard = _make_guard(mod, tmpdir, execute=False)
    dp = _FakeDatapath()
    guard._handle_alert({"src_ip": "10.0.0.5", "pred_class": "DDoS", "p_attack": 0.99,
                         "action": "block"}, dp)
    check(dp.sent == [], "演练模式（execute=False）下不发任何流表消息",
          f"实际下发 {len(dp.sent)} 条")
    rec = json.loads(Path(guard.audit_path).read_text(encoding="utf-8").strip().splitlines()[-1])
    check(rec["dry_run"] is True and rec["executed"] is False,
          "审计台账如实标注「演练 / 未执行」")
    check(rec["src_ip"] == "10.0.0.5" and rec["pred_class"] == "DDoS" and rec["p_attack"] == 0.99,
          "审计台账含可追溯字段（源 IP / 类别 / 置信度）")


def test_whitelist_blocks_enforcement(mod, tmpdir: Path) -> None:
    guard = _make_guard(mod, tmpdir, execute=True, whitelist=["10.0.0.5"])
    dp = _FakeDatapath()
    guard._handle_alert({"src_ip": "10.0.0.5", "pred_class": "DDoS", "p_attack": 0.99,
                         "action": "block"}, dp)
    check(dp.flowmods() == [], "白名单内的源 IP 即使开启真实处置也不会被下发规则",
          f"实际下发 {len(dp.flowmods())} 条")


def test_block_actually_pushes_rule(mod, tmpdir: Path) -> None:
    """防回归：block 必须真的调用 _drop() 下发 OFPFlowMod。"""
    guard = _make_guard(mod, tmpdir, execute=True)
    dp = _FakeDatapath()
    guard._handle_alert({"src_ip": "10.0.0.42", "pred_class": "DDoS", "p_attack": 0.99,
                         "action": "block"}, dp)
    mods = dp.flowmods()
    check(len(mods) == 1, "block 动作真的下发了 1 条 OpenFlow 规则（防「只记账不下发」回归）",
          f"实际 {len(mods)} 条")
    if not mods:
        return
    kw = mods[0]
    check(int(kw.get("hard_timeout", 0)) == 60, "下发的规则带有效期（hard_timeout=60s，到期自动解除）",
          f"hard_timeout={kw.get('hard_timeout')}")
    check(int(kw.get("priority", 0)) >= 100, "规则优先级高于普通转发流表项",
          f"priority={kw.get('priority')}")
    match = kw.get("match") or {}
    check(match.get("ipv4_src") == "10.0.0.42", "规则按攻击源 IP 精确匹配",
          f"match={match}")
    rec = json.loads(Path(guard.audit_path).read_text(encoding="utf-8").strip().splitlines()[-1])
    check(rec["executed"] is True and rec["dry_run"] is False and rec["action"] == "block",
          "审计台账标注「已执行」且与动作一致")


def test_throttle_falls_back_to_drop(mod, tmpdir: Path) -> None:
    guard = _make_guard(mod, tmpdir, execute=True)
    dp = _FakeDatapath()
    guard._handle_alert({"src_ip": "10.0.0.77", "pred_class": "Probe", "p_attack": 0.9,
                         "action": "throttle"}, dp)
    kinds = [m.get("__kind__") for m in dp.sent if isinstance(m, dict)]
    check("OFPMeterMod" in kinds, "throttle 下发了 Meter 限速", f"消息序列：{kinds}")
    check(any(isinstance(m, dict) and m.get("__kind__") == "OFPMeterBandDrop" for m in dp.sent)
          or any("bands" in m for m in dp.sent if isinstance(m, dict)),
          "Meter 携带限速 band（rate/burst_size）")
    # 防回归：只创建 Meter 而不挂到流表项上是**不会限速**的
    flowmods = dp.flowmods()
    check(len(flowmods) == 1, "限速规则被下发为流表项（只建 Meter 不限速，必须挂到流上）",
          f"实际 {len(flowmods)} 条流表项")
    if flowmods:
        insts = flowmods[0].get("instructions") or []
        has_meter = any(
            (isinstance(i, dict) and i.get("__kind__") == "OFPInstructionMeter")
            or "OFPInstructionMeter" in str(i) for i in insts
        )
        check(has_meter, "流表项通过 OFPInstructionMeter 引用该 Meter（否则限速不生效）",
              f"instructions={insts}")
        check(int(flowmods[0].get("hard_timeout", 0)) == 60,
              "限速规则同样带有效期（到期自动解除）",
              f"hard_timeout={flowmods[0].get('hard_timeout')}")


def test_repeated_alerts_are_idempotent(mod, tmpdir: Path) -> None:
    """防回归：同一条攻击流的连续告警（滑窗会连报十几个窗口）只下发一次规则。

    早期版本的 `self.blocked` 只写不读 —— 一次 60 s 的攻击会往交换机塞进几十条内容
    完全相同的流表项（真机联调时暴露：10 条告警 = 10 条重复 DROP 规则）。
    """
    guard = _make_guard(mod, tmpdir, execute=True)
    dp = _FakeDatapath()
    alert = {"src_ip": "10.0.0.99", "pred_class": "DDoS", "p_attack": 0.99, "action": "block"}
    for _ in range(5):
        guard._handle_alert(dict(alert), dp)
    mods = dp.flowmods()
    check(len(mods) == 1, "5 次连续告警只下发 1 条规则（封禁窗口内幂等）",
          f"实际下发 {len(mods)} 条")
    recs = [json.loads(x) for x in
            Path(guard.audit_path).read_text(encoding="utf-8").strip().splitlines() if x.strip()]
    recs = [r for r in recs if r.get("src_ip") == "10.0.0.99"]     # 台账文件在用例间复用
    check(len(recs) == 5 and sum(1 for r in recs if r.get("executed")) == 1
          and sum(1 for r in recs if r.get("suppressed")) == 4,
          "台账逐条留痕：5 条告警里 1 条真正执行、4 条标注为幂等抑制",
          f"executed={sum(1 for r in recs if r.get('executed'))} "
          f"suppressed={sum(1 for r in recs if r.get('suppressed'))}")

    # 封禁窗口过期后，新的告警必须重新下发（幂等不能变成"永久失灵"）
    guard.blocked["10.0.0.99"] = 0.0
    guard._handle_alert(dict(alert), dp)
    check(len(dp.flowmods()) == 2, "封禁窗口过期后同源 IP 的新告警会重新下发规则",
          f"实际下发 {len(dp.flowmods())} 条")

    # ---- 幂等只能抑制"同级或更弱"，更强动作必须放行 ----
    # 否则服务端「同源反复告警自动升级」（throttle→block）会被幂等吃掉：
    # 交换机上永远只挂着第一条弱规则，升级能力形同虚设，而账面上一切正常。
    g2 = _make_guard(mod, tmpdir, execute=True)
    dp2 = _FakeDatapath()
    g2._handle_alert({"src_ip": "10.0.0.50", "pred_class": "Probe", "p_attack": 0.9,
                      "action": "throttle"}, dp2)
    n1 = len(dp2.sent)
    g2._handle_alert({"src_ip": "10.0.0.50", "pred_class": "Probe", "p_attack": 0.95,
                      "action": "throttle"}, dp2)
    check(len(dp2.sent) == n1, "同级动作在封禁窗口内仍被幂等抑制",
          f"sent {n1} → {len(dp2.sent)}")
    g2._handle_alert({"src_ip": "10.0.0.50", "pred_class": "Probe", "p_attack": 0.99,
                      "action": "block"}, dp2)
    check(len(dp2.flowmods()) >= 2, "升级为更强动作（throttle→block）必须真的下发，不被幂等吃掉",
          f"flowmods={len(dp2.flowmods())}")

    # ---- 异常隔离：一条坏告警不能拖垮同批其它告警 ----
    g3 = _make_guard(mod, tmpdir, execute=True)
    dp3 = _FakeDatapath()
    g3._handle_alert({"src_ip": "10.0.0.60", "pred_class": "DDoS", "p_attack": 0.99,
                      "action": "block"}, None)          # datapath=None 必然抛错
    g3._handle_alert({"src_ip": "10.0.0.61", "pred_class": "DDoS", "p_attack": 0.99,
                      "action": "block"}, dp3)           # 后续告警仍应被正常处置
    check(len(dp3.flowmods()) == 1, "一条坏告警抛错后，同批后续告警仍被正常处置",
          f"实际下发 {len(dp3.flowmods())} 条")

    # ---- self.blocked 必须有界（伪造源 IP 洪泛不能把它撑爆）----
    g4 = _make_guard(mod, tmpdir, execute=True)
    dp4 = _FakeDatapath()
    g4.blocked = {f"10.8.0.{i}": 0.0 for i in range(600)}          # 全部已过期
    g4.blocked_action = {f"10.8.0.{i}": "block" for i in range(600)}
    g4._handle_alert({"src_ip": "10.8.9.9", "pred_class": "DDoS", "p_attack": 0.99,
                      "action": "block"}, dp4)
    check(len(g4.blocked) < 100, "已过期的封禁记录会被清理（不再只写不清理）",
          f"blocked={len(g4.blocked)}")

    old_thr, old_cap = mod._BLOCKED_PRUNE_THRESHOLD, mod._BLOCKED_HARD_CAP
    mod._BLOCKED_PRUNE_THRESHOLD, mod._BLOCKED_HARD_CAP = 5, 10
    try:
        g5 = _make_guard(mod, tmpdir, execute=True)
        dp5 = _FakeDatapath()
        for i in range(40):
            g5._handle_alert({"src_ip": f"10.7.{i // 250}.{i % 250}", "pred_class": "DDoS",
                              "p_attack": 0.99, "action": "block"}, dp5)
        check(len(g5.blocked) <= 10, "大量**未过期**的源 IP 也不会让 blocked 无界增长（硬上限）",
              f"blocked={len(g5.blocked)}")
    finally:
        mod._BLOCKED_PRUNE_THRESHOLD, mod._BLOCKED_HARD_CAP = old_thr, old_cap


def test_observe_action_is_inert(mod, tmpdir: Path) -> None:
    guard = _make_guard(mod, tmpdir, execute=True)
    dp = _FakeDatapath()
    guard._handle_alert({"src_ip": "10.0.0.8", "pred_class": "Normal", "p_attack": 0.3,
                         "action": "observe"}, dp)
    check(dp.sent == [], "observe 动作不会下发任何规则（只记录）",
          f"实际下发 {len(dp.sent)} 条")

    # 动作强度排序在两处各写了一遍：服务端 `src/service.py::ACTIONS` 的顺序 +
    # 适配层的 `_ACTION_RANK`。若有人给服务端加了新动作却忘了同步适配层，
    # `_ACTION_RANK.get(action, 0)` 会**静默**按最低级处理（升级对该动作失效）——
    # 正是本工程已两次踩中的"同一语义多处各判一次、只改一处"。
    try:
        from src.service import ACTIONS
    except Exception as exc:  # noqa: BLE001 - 缺少依赖时不让本用例失败
        check(True, f"跳过动作表一致性检查（无法导入 src.service：{type(exc).__name__}）")
    else:
        ranks = [mod._ACTION_RANK.get(a) for a in ACTIONS]
        check(set(mod._ACTION_RANK) == set(ACTIONS) and ranks == list(range(len(ACTIONS))),
              "适配层动作强度表与服务端 ACTIONS 完全一致且严格单调（新增动作必须两处同步）",
              f"guard={sorted(mod._ACTION_RANK)} service={list(ACTIONS)} ranks={ranks}")


def test_readback_verification_and_rollback(mod, tmpdir: Path) -> None:
    """下发后回读校验 + 失败回滚（可选能力，默认关）。

    守三件事：
      ① 默认关时**不发**额外的流表请求、行为与产物不变；
      ② 开启后必须真的用 OFPFlowStatsRequest 把流表读回来，并按**动作是否正确**判定
         （只"找到同源规则"不算通过——本该丢弃却变成限速同样是账实不符）；
      ③ 判定失败/超时 → 回滚（删规则 + 删 Meter）并**解开幂等窗口**，否则一次失败的处置
         会被永久固化成"看起来在处理"。
    """
    src = "10.0.0.42"
    alert = {"src_ip": src, "pred_class": "DDoS", "p_attack": 0.99, "action": "block"}

    # ---- ① 默认关：不发额外请求，台账如实标注"未校验" ----
    g0 = _make_guard(mod, tmpdir, execute=True)
    dp0 = _FakeDatapath()
    g0._handle_alert(dict(alert), dp0)
    check(len(dp0.flow_stats_requests()) == 0,
          "默认关（verify_readback=False）时不下发任何额外的流表回读请求",
          f"实际发出 {len(dp0.flow_stats_requests())} 条 OFPFlowStatsRequest")
    rec0 = [json.loads(x) for x in Path(g0.audit_path).read_text(encoding="utf-8").strip().splitlines()]
    rec0 = [r for r in rec0 if r.get("src_ip") == src and r.get("action") == "block"]
    check(bool(rec0) and (rec0[-1].get("verify") or {}).get("checked") is False,
          "台账不谎报校验：未开启时 verify.checked=False",
          f"verify={rec0[-1].get('verify') if rec0 else None}")

    # ---- ② 开启 + 交换机确实生效 → 校验通过、不回滚 ----
    g = _make_guard(mod, tmpdir, execute=True, verify=True)
    dp = _FakeDatapath()
    detects: list = []
    g._call_detect = lambda rows, datapath=None: (detects.append(rows), {"ok": True, "alerts": []})[1]
    g._handle_alert(dict(alert), dp)
    reqs = dp.flow_stats_requests()
    check(len(reqs) == 1, "开启后下发成功会发 1 条 OFPFlowStatsRequest（回读流表确认生效）",
          f"实际 {len(reqs)} 条")
    check(len(dp.flowmods()) == 1 and len(dp.deletes()) == 0, "下发阶段只有 1 条 ADD 规则、没有回滚动作")
    if not reqs:
        return
    xid = reqs[0].get("xid")
    check(isinstance(xid, int), "回读请求携带 xid（回复必须沿用同一 xid，才能与周期采集区分开）",
          f"xid={xid}")

    # 交换机回：该源上确实挂着 priority=100、APPLY_ACTIONS 且动作为空的丢弃规则
    ok_body = [_FakeFlowStat(priority=1, match={"ipv4_src": "10.0.0.1"}),
               _FakeFlowStat(priority=100, match={"ipv4_src": src},
                             instructions=[_FakeInstruction(_FakeOfproto.OFPIT_APPLY_ACTIONS, [])])]
    g._flow_stats_reply_handler(_FakeEvent(dp, ok_body, xid=xid))
    check(g._n_verified == 1 and g._n_rollbacks == 0, "回读确认规则生效后计入「已验证」且不回滚",
          f"verified={g._n_verified} rollbacks={g._n_rollbacks}")
    check(len(g._pending_verify) == 0, "校验完成即从待校验表中移除（不留悬挂条目）",
          f"pending={len(g._pending_verify)}")
    check(detects == [], "校验回复不得被当成新一轮流量喂给检测器（否则会重复告警）",
          f"检测调用次数={len(detects)}")
    recs = [json.loads(x) for x in Path(g.audit_path).read_text(encoding="utf-8").strip().splitlines()]
    ver = [r for r in recs if r.get("event") == "verify" and r.get("src_ip") == src]
    check(bool(ver) and ver[-1]["ok"] is True and ver[-1]["reason"] == "ok",
          "台账单独记录校验结论（event=verify，含期望动作与判定依据）",
          f"{ver[-1] if ver else None}")

    # 采集回复（xid 不匹配）必须照旧走检测路径
    g._flow_stats_reply_handler(_FakeEvent(dp, [_FakeFlowStat(priority=1,
                                                             match={"ip_proto": 6, "ipv4_src": "10.0.0.5"})],
                                          xid=999999))
    check(len(detects) == 1, "xid 不匹配的流表回复仍按周期采集处理（继续喂检测器）",
          f"检测调用次数={len(detects)}")

    # ---- ③ 动作不符：本该丢弃、交换机上却是限速 → 回滚 ----
    g2 = _make_guard(mod, tmpdir, execute=True, verify=True)
    dp2 = _FakeDatapath()
    g2._handle_alert({"src_ip": "10.0.0.43", "pred_class": "DDoS", "p_attack": 0.99,
                      "action": "block"}, dp2)
    xid2 = dp2.flow_stats_requests()[0].get("xid")
    wrong_body = [_FakeFlowStat(priority=100, match={"ipv4_src": "10.0.0.43"},
                                instructions=[_FakeInstruction(_FakeOfproto.OFPIT_METER)])]
    g2._flow_stats_reply_handler(_FakeEvent(dp2, wrong_body, xid=xid2))
    check(len(dp2.deletes()) == 1, "回读发现「动作与预期不符」时下发 DELETESTRICT 回滚",
          f"回滚报文 {len(dp2.deletes())} 条")
    if dp2.deletes():
        check(int(dp2.deletes()[0].get("command", -1)) == _FakeOfproto.OFPFC_DELETE_STRICT
              and (dp2.deletes()[0].get("match") or {}).get("ipv4_src") == "10.0.0.43",
              "回滚按源 IP 精确删除刚下发的规则（DELETE_STRICT + 同 match + 同优先级）")
    check("10.0.0.43" not in g2.blocked, "回滚后解开该源的封禁窗口（否则幂等会把重试永久抑制掉）")
    recs2 = [json.loads(x) for x in Path(g2.audit_path).read_text(encoding="utf-8").strip().splitlines()]
    v2 = [r for r in recs2 if r.get("event") == "verify" and r.get("src_ip") == "10.0.0.43"]
    check(bool(v2) and v2[-1]["ok"] is False and v2[-1].get("rolled_back") is True,
          "回滚写进台账（ok=false + rolled_back=true + 原因）", f"{v2[-1] if v2 else None}")

    # 回滚之后：同一源再来告警必须重新尝试下发（不能因为幂等而沉默）
    n_before = len(dp2.flowmods())
    g2._handle_alert({"src_ip": "10.0.0.43", "pred_class": "DDoS", "p_attack": 0.99,
                      "action": "block"}, dp2)
    check(len(dp2.flowmods()) > n_before, "回滚后同源新告警会重新尝试下发（幂等窗口已解开）",
          f"flowmods {n_before} → {len(dp2.flowmods())}")

    # ---- ④ 超时 = 未确认：同样回滚 ----
    g3 = _make_guard(mod, tmpdir, execute=True, verify=True, verify_timeout=0.01)
    dp3 = _FakeDatapath()
    g3._handle_alert({"src_ip": "10.0.0.44", "pred_class": "Probe", "p_attack": 0.9,
                      "action": "block"}, dp3)
    check(len(g3._pending_verify) == 1, "未收到回复时条目挂在待校验表里（异步等待，不阻塞事件循环）",
          f"pending={len(g3._pending_verify)}")
    time.sleep(0.15)          # 超时下限是 0.1s（见 _schedule_verify 的 max(0.1, …)）
    done = g3._sweep_verifications(now=time.time() + 10.0)   # 显式给"未来时刻"，判定可确定复现
    check(len(done) == 1 and done[0]["reason"] == "timeout" and done[0]["rolled_back"] is True,
          "超时扫描把「等不到回复」判定为未确认并回滚（不确定 ≠ 生效）",
          f"{done[0] if done else None}")
    check(len(g3._pending_verify) == 0 and len(dp3.deletes()) == 1,
          "超时判定后清空待校验表并下发回滚报文")

    # ---- ⑤ throttle 回滚要连 Meter 一起删（否则限速残留在交换机上） ----
    g4 = _make_guard(mod, tmpdir, execute=True, verify=True)
    dp4 = _FakeDatapath()
    g4._handle_alert({"src_ip": "10.0.0.45", "pred_class": "Probe", "p_attack": 0.9,
                      "action": "throttle"}, dp4)
    xid4 = dp4.flow_stats_requests()[0].get("xid")
    g4._flow_stats_reply_handler(_FakeEvent(dp4, [], xid=xid4))     # 流表里什么都没有
    meter_deletes = [m for m in dp4.sent if isinstance(m, dict)
                     and m.get("__kind__") == "OFPMeterMod"
                     and int(m.get("command", -1)) == _FakeOfproto.OFPMC_DELETE]
    check(len(dp4.deletes()) == 1 and len(meter_deletes) == 1,
          "限速规则回滚时同时删除 Meter（只删流表项会留下孤儿 Meter）",
          f"deletes={len(dp4.deletes())} meter_deletes={len(meter_deletes)}")

    # ---- ⑥ 待校验表必须有界（交换机完全不回包 + 大量告警也不能无界增长） ----
    g5 = _make_guard(mod, tmpdir, execute=True, verify=True, verify_timeout=3600.0)
    dp5 = _FakeDatapath()
    cap = int(mod._PENDING_VERIFY_CAP)
    for i in range(cap + 5):
        g5._handle_alert({"src_ip": f"10.9.{i // 250}.{i % 250}", "pred_class": "DDoS",
                          "p_attack": 0.99, "action": "block"}, dp5)
    check(len(g5._pending_verify) <= cap, "待校验条目有硬上限（不会因为交换机不回包而无限增长）",
          f"pending={len(g5._pending_verify)} ≤ cap={cap}")
    recs5 = [json.loads(x) for x in Path(g5.audit_path).read_text(encoding="utf-8").strip().splitlines()]
    skipped = [r for r in recs5 if (r.get("verify") or {}).get("reason") == "pending_cap"]
    check(bool(skipped), "超限时如实记账「本次未校验」，而不是假装校验过", f"{len(skipped)} 条")

    # ---- ⑦ 关闭回滚开关时只记账不移除规则（把选择权留给运维） ----
    g6 = _make_guard(mod, tmpdir, execute=True, verify=True, rollback=False)
    dp6 = _FakeDatapath()
    g6._handle_alert({"src_ip": "10.0.0.46", "pred_class": "DDoS", "p_attack": 0.99,
                      "action": "block"}, dp6)
    xid6 = dp6.flow_stats_requests()[0].get("xid")
    g6._flow_stats_reply_handler(_FakeEvent(dp6, [], xid=xid6))
    check(len(dp6.deletes()) == 0 and len(g6._pending_verify) == 0,
          "rollback_on_fail=False 时校验失败只记账、不回滚（fail-closed 由运维选择）")


def _run_all() -> int:
    print("=" * 74)
    print("Ryu 适配层行为测试（用桩件替代 Ryu，验证接口契约与安全默认值）")
    print("=" * 74)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        try:
            mod = _load_guard(tmpdir)
        except Exception as exc:  # noqa: BLE001
            check(False, "适配层可在无 Ryu 环境下被加载（桩件生效）", f"{type(exc).__name__}: {exc}")
            mod = None
        if mod is not None:
            check(True, "适配层可在无 Ryu 环境下被加载（桩件生效）",
                  "ryu_guard.py 导入成功，核心逻辑可被直接调用")
            test_arguments_registered(mod)
            test_payload_matches_model_features(mod, tmpdir)
            test_dry_run_sends_nothing(mod, tmpdir)
            test_whitelist_blocks_enforcement(mod, tmpdir)
            test_block_actually_pushes_rule(mod, tmpdir)
            test_repeated_alerts_are_idempotent(mod, tmpdir)
            test_throttle_falls_back_to_drop(mod, tmpdir)
            test_observe_action_is_inert(mod, tmpdir)
            test_readback_verification_and_rollback(mod, tmpdir)
    n_fail = sum(1 for ok, _, _ in _results if not ok)
    print("-" * 74)
    print(f"{len(_results) - n_fail}/{len(_results)} 项通过" + ("" if n_fail == 0 else f"，{n_fail} 项失败 ❌"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
