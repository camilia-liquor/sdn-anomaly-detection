"""Ryu 控制器对接脚本：把「检测服务」真正接进 SDN 控制平面（采集 → 检测 → 处置）。

与仓库里原有的 `collect_normal_traffic.py`（只采集）相比，本脚本多做三件事：
1. **在线检测**：每轮流表统计不再只写 CSV，而是直接 POST 给检测服务（`/api/detect`）；
2. **自动处置**：检测服务返回告警后，按策略下发 OpenFlow 流表规则（丢弃 / 限速）；
3. **审计台账**：每条告警与处置动作写入 JSONL，可事后追溯（谁、何时、依据什么、做了什么）。

⚠️ 运行前提与安全说明（**务必先读**）
------------------------------------
* 本脚本需要真实 Ryu + OpenFlow 交换机环境（如 Mininet）。**已验证范围**：
  ① 与检测服务之间的接口、判定与处置逻辑由 `tools/controller_simulator.py` 端到端验证
  （`results/e2e_controller_report.md`）；
  ② 本适配层已在**真实 `ryu-manager` 进程**下、通过**真实 OF1.3 线协议**完成
  握手 → 流表统计上报 → 告警 → 下发 DROP 规则 → 落台账的全链路验证（16 项检查全过，
  见 `results/ryu_live_check.md`，由 `tools/ryu_live_check.py` 复现）。
  **仍未覆盖**：物理交换机、真实攻击流量、厂商私有扩展。
* **默认演练模式（dry_run=True）**：只打印将要下发的规则，不真正改流表。
  确认无误后再用 `--execute` 打开真实处置。
* **白名单**：控制器自身、管理网段、网关必须在 `--whitelist` 里，否则可能把管理链路断掉。
* 处置动作有超时（`--block-seconds`），到期自动解除，避免误封长时间影响业务。
* **处置幂等**：同一源 IP 在封禁窗口内重复告警时**不重复下发规则**（只写台账），
  避免一次持续攻击往流表里塞进几十条相同规则。
* **下发后回读校验（可选，默认关）**：`verify_readback = true` 时，规则下发后会再发
  `OFPFlowStatsRequest` 把流表**读回来**，确认该源上确实挂着动作正确的规则；
  不一致或超时 → **回滚**（删掉刚下发的规则 + 解开幂等窗口以便重试）并写 `event=verify` 台账。
  为什么默认关：有些交换机对带 match 的 stats 请求响应很慢甚至不回，默认开启会把"不确定"
  变成"回滚"，反而让处置形同虚设；而 `executed=true` 只代表"消息已发出"，**不代表交换机上真的生效**。

用法（在 Mininet 环境中）::

    用 **oslo.config 的 user-flags 机制**传自定义参数（Ryu ≥ 4.34 的正确做法）：
    先写一个 flag 文件，例如 `ryu_guard.conf`::

        [DEFAULT]
        detect_url = http://127.0.0.1:5000
        execute = true
        block_seconds = 60
        whitelist = 10.0.0.1,10.0.0.254
        audit = results/audit_alerts_ryu.jsonl

    然后::

        # 1) 先起检测服务（可跑在控制器同一台机器上）
        #    python app/app.py --port 5000 --processed-dir data/processed      # CICFlowMeter 字段模型
        #    python app/app.py --port 5000 --processed-dir data/processed_sdn  # 控制器原生字段模型
        # 2) 起 Ryu 控制器
        ryu-manager --user-flags ryu_guard.conf integrations/ryu_guard.py
        # 3) 也可以完全不带参数启动（用下面的默认值：演练模式 + 本机 5000 端口）

⚠️ **参数机制的真实情况（本机真机通电后才发现的）**：Ryu 4.34 的 `ryu-manager` **不再调用**
各 app 的 `add_arguments()` —— 老教程（以及本文件早期版本）依赖的那个钩子已经失效，
自定义 `--xxx` 会被 oslo.config 判为 `unrecognized arguments` 而启动失败。
正确做法是用 `--user-flags <conf>` 把选项喂给 oslo.config。
本模块因此**同时注册两种方式**：`RYU_GUARD_OPTS`（oslo.config 选项，供 user-flags 使用）
与 `add_arguments()`（保留给仍调用它的旧版 Ryu / 第三方启动器）。
命令行参数会被解析进 `DEFAULT` 组，`GuardSwitch.__init__` 会优先读它们。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ether_types  # noqa: F401  (保留以便扩展)

# 特征名必须与检测服务所使用的 schema 完全一致（见 data/processed*/meta.json）
#   默认按"控制器原生字段"（sdn schema）组装；若服务加载的是 insdn schema 模型，请用 --schema insdn
SDN_FEATURES = [
    "datapath_id", "tp_src", "tp_dst", "ip_proto", "icmp_code", "icmp_type",
    "flow_duration_sec", "flow_duration_nsec", "idle_timeout", "hard_timeout", "flags",
    "packet_count", "byte_count", "packet_count_per_second", "packet_count_per_nsecond",
    "byte_count_per_second", "byte_count_per_nsecond",
]


# 动作强度排序：幂等抑制只在"同级或更弱"时生效，更强动作必须放行（否则升级形同虚设）
_ACTION_RANK = {"observe": 0, "throttle": 1, "block": 2}
# self.blocked 的清理阈值与硬上限（防止大量伪造源 IP 把它撑爆）
_BLOCKED_PRUNE_THRESHOLD = 512
_BLOCKED_HARD_CAP = 4096

# 指令类型常量（OF1.3，见 ofproto_v1_3）：回读判定"这条规则到底做了什么"时用。
# 不 import ryu.ofproto 模块级常量，是为了让桩件测试（无 Ryu 环境）也能加载本文件。
_OFPIT_APPLY_ACTIONS = 4
_OFPIT_METER = 6
# 待校验条目上限：每条 = 一次下发，正常在 verify_timeout 内就会被回复或超时清理；
# 上限只是防止"交换机完全不回包 + 极端告警速率"时无界增长（超限的新校验直接跳过并如实记账）。
_PENDING_VERIFY_CAP = 256


class GuardSwitch(simple_switch_13.SimpleSwitch13):
    """带异常检测与自动处置的 SDN 控制器应用。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        # 参数可以在三个地方给（按优先级）：① ryu-manager 传进来的 kwargs（旧机制）
        # ② oslo.config DEFAULT 组（--user-flags 文件 或 命令行，Ryu 4.34 的正确姿势）
        # ③ 模块内默认值。这样"带参数"和"不带参数直接起"都能工作。
        def pick(key, default, conf_key=None):
            if key in kwargs:
                return kwargs[key]
            v = _conf_get(conf_key or key, None)
            return default if v in (None, "") else v

        self.detect_url = pick("detect_url", "http://127.0.0.1:5000")
        self.execute = bool(pick("execute", False))
        self.block_seconds = int(pick("block_seconds", 60))
        wl = pick("whitelist", [])
        if isinstance(wl, str):
            wl = [x.strip() for x in wl.split(",") if x.strip()]
        self.whitelist = set(wl or [])
        self.audit_path = pick("audit", os.path.join("results", "audit_alerts_ryu.jsonl"))
        # ---------------- 下发后回读校验 + 失败回滚（可选，默认关） ----------------
        # 为什么需要：`_drop()` / `_meter_or_limit()` 只要"消息发出去了"就返回 True，
        # 而**消息发出去 ≠ 交换机上真的生效**（权限、table 满、指令不支持、meter 超限、
        # 甚至中间有设备吞包）。这就是本工程反复强调的"账实不符"：台账说已下发，交换机上什么都没有。
        # 打开后：下发 → 发 OFPFlowStatsRequest 回读 → 确认规则真的在表里且动作正确；
        # 确认不了（不一致 / 超时）→ **回滚**（删掉刚下发的规则，并解开幂等窗口以便重试）。
        self.verify_readback = bool(pick("verify_readback", False))
        self.verify_timeout = float(pick("verify_timeout", 2.0) or 2.0)
        self.rollback_on_fail = bool(pick("rollback_on_fail", True))
        self._pending_verify: dict = {}      # xid → 待校验条目（异步等待，见 _schedule_verify）
        self._n_verified = 0
        self._n_rollbacks = 0
        self.blocked: dict[str, float] = {}
        # 每个源当前生效的动作级别：幂等要能区分"重复的同级告警"与"升级为更强动作"
        self.blocked_action: dict[str, str] = {}
        self._n_handled = 0          # 已处理告警数（进心跳，供 cida status 显示）
        os.makedirs(os.path.dirname(self.audit_path) or ".", exist_ok=True)
        self.logger.info("[guard] 检测服务=%s 真实处置=%s 白名单=%s",
                         self.detect_url, self.execute, sorted(self.whitelist) or "（空，建议至少加入控制器与管理网段）")
        if not self.execute:
            self.logger.warning("[guard] 当前为演练模式：只输出将要下发的规则，不改动流表（加 --execute 开启真实处置）")
        if self.verify_readback:
            self.logger.info("[guard] 已开启「下发后回读校验」：超时 %.1fs，校验失败%s",
                             self.verify_timeout, "回滚" if self.rollback_on_fail else "仅记账不回滚")
        self.monitor_thread = hub.spawn(self._monitor)
        if self.verify_readback:
            self.verify_thread = hub.spawn(self._verify_loop)

    # ------------------------------------------------------------------ 采集
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.logger.info("[guard] 交换机上线：%016x", dp.id)
        elif ev.state == DEAD_DISPATCHER and dp.id in self.datapaths:
            del self.datapaths[dp.id]
            self.logger.info("[guard] 交换机下线：%016x", dp.id)

    def _monitor(self):
        """周期性拉取流表统计（周期与控制器采集策略一致）+ 写心跳。"""
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            self._write_heartbeat()
            hub.sleep(10)

    def _write_heartbeat(self):
        """写心跳文件，让 `python tools/cida.py status` 能判断适配层是否还活着。

        为什么用心跳文件而不是开 HTTP 端口：适配层是跑在另一个进程的 Ryu 应用，
        为"被监控"而多开一个监听面，等于白送一份攻击面；心跳文件足够判断存活。
        """
        try:
            now = time.time()
            data = {
                "ts": now, "pid": os.getpid(),
                "execute": bool(self.execute),
                "block_seconds": int(self.block_seconds),
                "whitelist": sorted(self.whitelist),
                "datapaths": [f"{d:016x}" for d in self.datapaths],
                "blocked_now": sum(1 for exp in self.blocked.values() if exp > now),
                "alerts_handled": int(getattr(self, "_n_handled", 0)),
                "detect_url": self.detect_url,
                # 回读校验的健康度：待确认积压 > 0 说明交换机回复变慢/不回；
                # rollbacks > 0 说明"下发没生效"真的发生过（这是过去的盲区，现在可见了）
                "verify_readback": bool(getattr(self, "verify_readback", False)),
                "verified": int(getattr(self, "_n_verified", 0)),
                "rollbacks": int(getattr(self, "_n_rollbacks", 0)),
                "verify_pending": len(getattr(self, "_pending_verify", {}) or {}),
            }
            p = Path(os.path.join("results", "guard_heartbeat.json"))
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)          # 原子替换：避免 status 读到写了一半的文件
        except Exception as exc:  # noqa: BLE001 - 心跳失败绝不能影响处置主链路
            self.logger.debug("[guard] 心跳写入失败：%s", exc)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPFlowStatsRequest(datapath))

    # ------------------------------------------------------------------ 回读校验
    # ⚠️ 设计要点：**绝不能**在下发路径上同步等待回读回复。
    #    Ryu 的一个 RyuApp 只有一个事件派发循环（EventDispatcher 的 greenlet），
    #    回复事件也排在同一个队列里；若在 `_handle_alert` 里 wait 回复，回复永远轮不到被处理
    #    → 每次都超时 → 每次都被判"没生效" → 把已经正确的规则全部回滚掉。
    #    因此这里做成异步：发请求登记 xid → 回复到达时（另一个事件）判定 → 未到达则由
    #    独立的 `_verify_loop` 绿线程按 deadline 扫描超时。
    def _verify_loop(self):
        """独立绿线程：按 deadline 扫描超时的回读校验（不阻塞事件派发循环）。"""
        while True:
            hub.sleep(0.2)
            try:
                self._sweep_verifications()
            except Exception as exc:  # noqa: BLE001 - 校验循环本身绝不能把控制器带崩
                self.logger.error("[guard] 回读校验扫描异常：%s", exc)

    def _sweep_verifications(self, now: float | None = None) -> list:
        """把已超时的待校验项判定为「未确认」并走同一套回滚路径。返回本次处理的条目。"""
        now = time.time() if now is None else now
        done = []
        for xid, slot in list(self._pending_verify.items()):
            if now >= float(slot.get("deadline", 0.0)):
                self._pending_verify.pop(xid, None)
                done.append(self._resolve_verify(slot, None, reason="timeout"))
        return done

    def _schedule_verify(self, datapath, src: str, expect: str) -> dict:
        """下发成功后登记一次回读校验：发 OFPFlowStatsRequest 并记住它的 xid。

        返回给台账的**即时**状态（`state=pending`）——最终结论由回复或超时给出，
        并作为单独一条 `event=verify` 审计记录落盘。**不允许**把"已发送校验请求"记成"已验证"。
        """
        parser = datapath.ofproto_parser
        if len(self._pending_verify) >= _PENDING_VERIFY_CAP:
            # 有界：超限时宁可"不校验"并如实记账，也不能让内存无界增长
            self.logger.error("[guard] 待校验条目已达上限 %d，跳过本次回读校验（src=%s）",
                              _PENDING_VERIFY_CAP, src)
            return {"checked": False, "reason": "pending_cap"}
        try:
            req = parser.OFPFlowStatsRequest(
                datapath, match=parser.OFPMatch(eth_type=0x0800, ipv4_src=src))
            datapath.send_msg(req)
        except Exception as exc:  # noqa: BLE001
            slot = {"src": src, "expect": expect, "datapath": datapath, "deadline": 0.0}
            res = self._resolve_verify(slot, None, reason=f"request_failed:{type(exc).__name__}")
            return {"checked": True, "state": "failed", "reason": res.get("reason")}
        xid = _msg_xid(req)
        if xid is None:
            # Ryu 的 Datapath.send_msg 会给消息分配 xid；取不到就无法把回复与请求配对，
            # 此时**不能**把任意一条流表回复当成校验结果（会把周期采集的回复误判成"没生效"）。
            slot = {"src": src, "expect": expect, "datapath": datapath, "deadline": 0.0}
            res = self._resolve_verify(slot, None, reason="no_xid")
            return {"checked": True, "state": "failed", "reason": res.get("reason")}
        self._pending_verify[xid] = {
            "src": src, "expect": expect, "datapath": datapath, "xid": xid,
            "deadline": time.time() + max(0.1, float(self.verify_timeout)),
        }
        return {"checked": True, "state": "pending", "xid": int(xid)}

    @staticmethod
    def _match_rule(body, src: str, expect: str) -> dict:
        """在回读到的流表项里找"我们刚下发的那条规则"，并判断它的动作是否符合预期。

        判据（与 `_drop` / `_meter_or_limit` 下发时的写法一一对应）：
          * 优先级 ≥100 且 `ipv4_src` 精确等于该源（这是本作品下发的处置规则的特征）；
          * `expect=block`  → 存在 APPLY_ACTIONS 且**动作列表为空**（丢弃）；
          * `expect=throttle` → 存在 METER 指令（限速）。
        只"找到了同源规则"还不够：动作不对同样是账实不符（例如本该限速却变成了丢弃）。
        """
        n = 0
        for st in body or []:
            n += 1
            m = getattr(st, "match", {}) or {}
            try:
                priority = int(getattr(st, "priority", 0) or 0)
            except (TypeError, ValueError):
                priority = 0
            if priority < 100 or str(m.get("ipv4_src", "")) != src:
                continue
            kinds = []
            for inst in getattr(st, "instructions", None) or []:
                itype = getattr(inst, "type", None)
                if itype == _OFPIT_METER:
                    kinds.append("meter")
                elif itype == _OFPIT_APPLY_ACTIONS:
                    kinds.append("drop" if not getattr(inst, "actions", None) else "forward")
            ok = ("meter" in kinds) if expect == "throttle" else ("drop" in kinds)
            return {"ok": bool(ok), "found": True, "kinds": kinds, "n_stats": n}
        return {"ok": False, "found": False, "kinds": [], "n_stats": n}

    def _resolve_verify(self, slot: dict, body, reason: str | None = None) -> dict:
        """对一次回读校验下结论；不一致或超时则回滚，并写一条 `event=verify` 审计记录。"""
        src, expect = str(slot.get("src") or ""), str(slot.get("expect") or "")
        verdict = ({"ok": False, "found": False, "kinds": [], "reason": reason}
                   if reason else self._match_rule(body, src, expect))
        record = {
            "event": "verify", "time": time.time(), "src_ip": src, "expect_action": expect,
            "ok": bool(verdict.get("ok")), "found": bool(verdict.get("found", False)),
            "rule_kinds": verdict.get("kinds", []), "n_stats": int(verdict.get("n_stats", 0)),
            "reason": reason or ("ok" if verdict.get("ok") else "mismatch"),
            "datapath_id": (slot.get("datapath").id if slot.get("datapath") else None),
        }
        if verdict.get("ok"):
            self._n_verified = int(getattr(self, "_n_verified", 0)) + 1
            self.logger.info("[guard] 回读校验通过：src=%s 动作 %s 已在交换机流表中生效", src, expect)
        else:
            rolled = False
            if self.rollback_on_fail and slot.get("datapath") is not None:
                rolled = self._rollback(slot["datapath"], src, expect)
            record["rolled_back"] = bool(rolled)
            if rolled:
                self._n_rollbacks = int(getattr(self, "_n_rollbacks", 0)) + 1
                # 回滚后必须解开幂等窗口：否则该源在 block_seconds 内会被一直当作"已处置"，
                # 永远不会重试，等于把一次失败的处置永久固化成"看起来在处理"。
                self.blocked.pop(src, None)
                self.blocked_action.pop(src, None)
            self.logger.error("[guard] 回读校验失败（%s）：src=%s 期望动作 %s%s",
                              record["reason"], src, expect,
                              "，已回滚" if rolled else "（未回滚，仅记账）")
        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # noqa: BLE001
            self.logger.error("[guard] 校验结论写台账失败：%s", exc)
        return record

    def _rollback(self, datapath, src: str, expect: str) -> bool:
        """回滚一次"没被确认"的下发：删掉该源的处置规则（限速时连 Meter 一起删）。"""
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        try:
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src)
            datapath.send_msg(parser.OFPFlowMod(
                datapath=datapath, priority=100, match=match,
                command=ofproto.OFPFC_DELETE_STRICT,
                out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY))
            if expect == "throttle":
                datapath.send_msg(parser.OFPMeterMod(datapath=datapath, command=ofproto.OFPMC_DELETE,
                                                    meter_id=1))
            self.logger.warning("[guard] 已回滚 src=%s 的处置规则（回读未确认）", src)
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("[guard] 回滚 src=%s 失败：%s（交换机上可能残留规则）", src, exc)
            return False

    # ------------------------------------------------------------------ 检测 + 处置
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        # ⚠️ 先按 xid 分流：本模块会发**两类** OFPFlowStatsRequest——周期采集（喂检测器）与
        #    下发后回读校验。靠 xid 才能分清（OF1.3 规定回复必须沿用请求的 xid）。
        #    若不分流，校验回复会被当成新一轮流量再喂给检测器（重复告警），
        #    而采集回复又可能被误判成"规则没生效"（把正确的规则回滚掉）。
        xid = _msg_xid(ev.msg)
        slot = self._pending_verify.pop(xid, None) if xid is not None else None
        if slot is not None:
            self._resolve_verify(slot, ev.msg.body)
            return
        body = ev.msg.body
        rows = []
        for stat in sorted([f for f in body if f.priority == 1], key=lambda f: (
                f.match.get("eth_type", 0), f.match.get("ipv4_src", ""), f.match.get("ipv4_dst", ""),
                f.match.get("ip_proto", 0))):
            m = stat.match
            proto = m.get("ip_proto", 0)
            icmp_code, icmp_type, tp_src, tp_dst = -1, -1, 0, 0
            if proto == 1:
                icmp_code = m.get("icmpv4_code", -1)
                icmp_type = m.get("icmpv4_type", -1)
            elif proto == 6:
                tp_src, tp_dst = m.get("tcp_src", 0), m.get("tcp_dst", 0)
            elif proto == 17:
                tp_src, tp_dst = m.get("udp_src", 0), m.get("udp_dst", 0)
            dur_sec, dur_nsec = stat.duration_sec, stat.duration_nsec
            row = {
                "datapath_id": ev.msg.datapath.id,
                "tp_src": tp_src, "tp_dst": tp_dst, "ip_proto": proto,
                "icmp_code": icmp_code, "icmp_type": icmp_type,
                "flow_duration_sec": dur_sec, "flow_duration_nsec": dur_nsec,
                "idle_timeout": stat.idle_timeout, "hard_timeout": stat.hard_timeout,
                "flags": stat.flags,
                "packet_count": stat.packet_count, "byte_count": stat.byte_count,
                "packet_count_per_second": stat.packet_count / dur_sec if dur_sec else 0,
                "packet_count_per_nsecond": stat.packet_count / dur_nsec if dur_nsec else 0,
                "byte_count_per_second": stat.byte_count / dur_sec if dur_sec else 0,
                "byte_count_per_nsecond": stat.byte_count / dur_nsec if dur_nsec else 0,
                # 供审计与处置使用的上下文（不作为模型特征）
                "Src IP": m.get("ipv4_src", ""), "Dst IP": m.get("ipv4_dst", ""),
                "Dst Port": tp_dst,
            }
            rows.append(row)
        if not rows:
            return
        resp = self._call_detect(rows, datapath=ev.msg.datapath)
        if resp:
            for alert in resp.get("alerts", []):
                self._handle_alert(alert, ev.msg.datapath)

    def _call_detect(self, rows, datapath=None):
        payload = json.dumps({
            "rows": rows,
            "meta": {"src_ip": rows[0].get("Src IP"), "dst_ip": rows[0].get("Dst IP"),
                     "dst_port": rows[0].get("Dst Port")},
        }).encode()
        req = urllib.request.Request(self.detect_url + "/api/detect", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode())
        except urllib.error.URLError as exc:
            self.logger.error("[guard] 检测服务不可达（%s）：%s", self.detect_url, exc)
            return None

    def _prune_blocked(self, now: float) -> None:
        """清理已过期的封禁记录，保证 `self.blocked` 有界。

        为什么必须有：`self.blocked` 过去**只写、只读、从不清理**。攻击者用大量伪造源 IP
        就足以让它无限增长——和检测服务里那个"无上界 list"是同一类隐患，
        只是这里更容易被触发（伪造源 IP 本来就不花钱）。
        """
        if len(self.blocked) <= _BLOCKED_PRUNE_THRESHOLD:
            return
        for k in [k for k, exp in list(self.blocked.items()) if exp <= now]:
            self.blocked.pop(k, None)
            self.blocked_action.pop(k, None)
        # 仍有大量未到期记录时，按到期时间丢掉最旧的一半，确保不无界增长
        if len(self.blocked) > _BLOCKED_HARD_CAP:
            for k in sorted(self.blocked, key=lambda x: self.blocked[x])[:len(self.blocked) // 2]:
                self.blocked.pop(k, None)
                self.blocked_action.pop(k, None)

    def _handle_alert(self, alert: dict, datapath):
        """单条告警的处理入口：**异常隔离**。

        滑窗一次会返回多条 alert。如果某一条触发异常（例如非法 src_ip 让 OFPMatch 抛错），
        异常会冒泡出去、把同批剩下的告警**全部跳过**——处置系统不能因为一条坏数据就集体失灵。
        """
        try:
            self._handle_alert_inner(alert, datapath)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("[guard] 处理告警失败，已跳过该条并继续处理同批其它告警：%s | %s",
                              exc, alert)

    def _handle_alert_inner(self, alert: dict, datapath):
        src = alert.get("src_ip") or ""
        action = alert.get("action", "observe")
        executed = False
        suppressed = False
        verify: dict = {"checked": False}
        if action in ("block", "throttle") and src and src not in self.whitelist:
            now = time.time()
            # 幂等：同一条攻击流的连续告警（在线检测是滑动窗口，一次攻击会连续报好几个窗口）
            # 在封禁窗口内**不重复下发**规则。否则 60 s 的攻击会往交换机塞进几十条内容完全
            # 相同的流表项，既造成流表抖动，也让"处置动作数"这个指标失去意义。
            # 早期版本的 `self.blocked` 只写不读，等于没有这个保护（真机联调时暴露出来）。
            self._prune_blocked(now)
            in_window = self.blocked.get(src, 0.0) > now
            cur_action = self.blocked_action.get(src, "")
            # ⚠️ 幂等只能抑制「同级或更弱」的动作，**更强的动作必须放行**。
            # 否则服务端「同源反复告警自动升级」（throttle→block）会被这里的幂等吃掉：
            # 交换机上永远只挂着第一条弱规则，而升级能力形同虚设——
            # 账面上一切正常，只有交换机的实际状态和预期不一致。
            stronger = _ACTION_RANK.get(action, 0) > _ACTION_RANK.get(cur_action, -1)
            if in_window and not stronger:
                suppressed = True
                self.logger.info("[guard] src=%s 仍在封禁窗口内（剩余 %.0fs，当前动作 %s），本次告警不重复下发规则",
                                 src, self.blocked[src] - now, cur_action or "?")
            elif self.execute:
                # ⚠️ 这里必须真的把规则下发出去：早期实现写成 `executed = True if action == "block" ...`，
                # 结果是 block 动作只把 executed 记成 True、却**从未调用 `_drop()`**——
                # 审计台账会显示"已下发"，而交换机上什么都没变。对一套声称能处置的系统来说，
                # 这种"账实不符"比不处置更危险，故修正为两条动作各自走真实下发路径。
                executed = (self._meter_or_limit(datapath, src) if action == "throttle"
                            else self._drop(datapath, src))
                if executed:
                    self.blocked[src] = now + self.block_seconds
                    self.blocked_action[src] = action
                    # 回读校验（可选，默认关）：注意 `executed=True` 只表示"消息已发出"，
                    # 最终是否真的生效由 `event=verify` 那条记录给出——两者不可混为一谈。
                    if self.verify_readback:
                        verify = self._schedule_verify(datapath, src, action)
                else:
                    self.logger.error("[guard] 动作 %s 下发失败，src=%s —— 已记入台账但未生效", action, src)
            else:
                self.logger.warning("[guard][演练] 将下发动作 action=%s src=%s（到期自动解除 %ds）",
                                    action, src, self.block_seconds)
        record = {
            "time": time.time(), "src_ip": src, "dst_ip": alert.get("dst_ip"),
            "dst_port": alert.get("dst_port"), "pred_class": alert.get("pred_class"),
            "p_attack": alert.get("p_attack"), "action": action,
            "executed": executed, "dry_run": not self.execute,
            "suppressed": suppressed,          # True = 该源 IP 已在封禁窗口内，未重复下发
            "verify": verify,                  # 回读校验的**即时**状态（pending/failed/未开启）
            "block_seconds": self.block_seconds,
            "datapath_id": datapath.id if datapath else None,
        }
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.logger.warning("[guard] 告警：%s p=%.3f src=%s → %s%s",
                            record["pred_class"], float(record["p_attack"] or 0), src, action,
                            "（已下发）" if executed else ("（封禁窗口内，幂等抑制）" if suppressed
                                                          else "（仅记录）"))
        self._n_handled = int(getattr(self, "_n_handled", 0)) + 1

    def _meter_or_limit(self, datapath, src):
        """限速：下发 Meter **并把流表项挂到这个 Meter 上**，失败则退化为丢弃。

        ⚠️ 这里有一个很容易写错的坑：**只下发 `OFPMeterMod` 是不会限速的**——
        Meter 必须被某条流表项通过 `OFPInstructionMeter` 引用才会对该流生效。
        早期实现只创建了 Meter 就直接 `return True`，交换机上其实什么都没变，
        属于和 `block` 同一类的「账实不符」。因此这里第二步必须再下发一条
        带 `OFPInstructionMeter(meter_id)` 的流表项。
        """
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        meter_id = 1
        try:
            bands = [parser.OFPMeterBandDrop(rate=1000, burst_size=100)]
            datapath.send_msg(parser.OFPMeterMod(datapath=datapath, command=ofproto.OFPMC_ADD,
                                                 flags=ofproto.OFPMF_KBPS, meter_id=meter_id,
                                                 bands=bands))
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src)
            inst = [parser.OFPInstructionMeter(meter_id)]
            datapath.send_msg(parser.OFPFlowMod(datapath=datapath, priority=100, match=match,
                                                instructions=inst, hard_timeout=self.block_seconds,
                                                command=ofproto.OFPFC_ADD))
            self.logger.info("[guard] 已下发限速规则（Meter %d，1 Mbps）并挂到 src=%s 的流表项上",
                             meter_id, src)
            return True
        except Exception as exc:  # 交换机不支持 meter / OF1.3 未启用时退化为丢弃
            self.logger.warning("[guard] 限速不可用（%s），改为丢弃", exc)
            return self._drop(datapath, src)

    def _drop(self, datapath, src):
        """下发丢弃规则（带超时，到期自动失效）。"""
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src)
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, [])]
        mod = parser.OFPFlowMod(datapath=datapath, priority=100, match=match, instructions=inst,
                                hard_timeout=self.block_seconds, command=ofproto.OFPFC_ADD)
        datapath.send_msg(mod)
        self.logger.warning("[guard] 已下发丢弃规则：src=%s（%ds 后自动解除）", src, self.block_seconds)
        return True


# ======================================================================================
# 参数注册：**两条路都铺**（Ryu 4.34 用 oslo.config user-flags，旧版才调 add_arguments）
# --------------------------------------------------------------------------------------
# 真机实测结论：ryu-manager 4.34 只会注册自己那几个选项（app / app-lists / pid-file …），
# 不再遍历各 app 的 add_arguments()，于是 `--detect-url` 这类参数会被判为
# "unrecognized arguments" 直接启动失败。这里用 oslo.config 注册同名的选项，
# 既支持 `--user-flags guard.conf`，也支持命令行直接传（若启动器允许）。
# ======================================================================================
try:  # pragma: no cover - 只有在装了 ryu 的环境里才会走到
    from oslo_config import cfg as _oslo_cfg

    RYU_GUARD_OPTS = [
        _oslo_cfg.StrOpt("detect_url", default="http://127.0.0.1:5000",
                         help="检测服务地址（如 http://127.0.0.1:5000）"),
        _oslo_cfg.BoolOpt("execute", default=False,
                          help="开启真实处置（默认仅演练，不改流表）"),
        _oslo_cfg.IntOpt("block_seconds", default=60, help="处置有效期（秒）"),
        _oslo_cfg.ListOpt("whitelist", default=[], help="永不处置的源 IP（逗号分隔）"),
        _oslo_cfg.StrOpt("audit", default=os.path.join("results", "audit_alerts_ryu.jsonl"),
                         help="审计台账路径（JSONL）"),
        _oslo_cfg.StrOpt("schema", default="", help="模型特征口径（留空自动判断）"),
        _oslo_cfg.BoolOpt("verify_readback", default=False,
                          help="下发后回读校验（默认关）：发 OFPFlowStatsRequest 确认规则真的生效"),
        _oslo_cfg.FloatOpt("verify_timeout", default=2.0, help="回读校验等待回复的超时（秒）"),
        _oslo_cfg.BoolOpt("rollback_on_fail", default=True,
                          help="回读校验失败/超时是否回滚（默认回滚；仅在 verify_readback 打开时生效）"),
    ]
    _oslo_cfg.CONF.register_opts(RYU_GUARD_OPTS)
except Exception:  # noqa: BLE001 - 没装 ryu 时（例如单元测试打桩）静默跳过
    RYU_GUARD_OPTS = []


def _msg_xid(msg):
    """取一条 OpenFlow 消息的 xid（回读校验靠它与回复配对）。

    为什么要兼容两种形态：真机里 Ryu 的消息是**对象**（`.xid` 属性），
    而 `tests/test_ryu_guard.py` 的桩件用 dict 表示消息（便于断言构造参数）。
    只写 `getattr` 会让桩件里"取不到 xid"→ 每次校验都被判成"无法配对"→ 把正确的规则回滚掉。
    """
    if isinstance(msg, dict):
        return msg.get("xid")
    return getattr(msg, "xid", None)


def _conf_get(name: str, default=None):
    """从 oslo.config 的 DEFAULT 组读参数（读不到就返回默认值 —— 不抛异常）。"""
    try:
        from oslo_config import cfg as _cfg

        return getattr(_cfg.CONF, name, default)
    except Exception:  # noqa: BLE001
        return default


def add_arguments(parser):
    """（兼容旧版 Ryu）向启动器注册命令行参数。

    ⚠️ Ryu 4.34 已不再调用本函数；请改用 `--user-flags <conf>`（见模块 docstring）。
    保留它只是为了兼容仍会调用的旧版 Ryu 与第三方启动器。
    """
    parser.add_argument("--detect-url", default="http://127.0.0.1:5000", help="检测服务地址")
    parser.add_argument("--execute", action="store_true", help="开启真实处置（默认仅演练，不改流表）")
    parser.add_argument("--block-seconds", type=int, default=60, help="处置有效期（秒），到期自动解除")
    parser.add_argument("--whitelist", nargs="*", default=[], help="永不处置的源 IP（控制器/管理网段/网关）")
    parser.add_argument("--audit", default=os.path.join("results", "audit_alerts_ryu.jsonl"), help="审计台账路径")
    parser.add_argument("--verify-readback", action="store_true",
                        help="下发后回读校验（默认关；开启后超时/不一致即回滚）")
    parser.add_argument("--verify-timeout", type=float, default=2.0, help="回读校验等待回复的超时（秒）")
    parser.add_argument("--no-rollback-on-fail", dest="rollback_on_fail", action="store_false",
                        help="回读校验失败时只记账、不回滚")
    return parser
