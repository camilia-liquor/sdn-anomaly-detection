"""Ryu 控制器对接脚本：把「检测服务」真正接进 SDN 控制平面（采集 → 检测 → 处置）。

与仓库里原有的 `collect_normal_traffic.py`（只采集）相比，本脚本多做三件事：
1. **在线检测**：每轮流表统计不再只写 CSV，而是直接 POST 给检测服务（`/api/detect`）；
2. **自动处置**：检测服务返回告警后，按策略下发 OpenFlow 流表规则（丢弃 / 限速）；
3. **审计台账**：每条告警与处置动作写入 JSONL，可事后追溯（谁、何时、依据什么、做了什么）。

⚠️ 运行前提与安全说明（**务必先读**）
------------------------------------
* 本脚本**需要真实 Ryu + OpenFlow 交换机环境**（如 Mininet），本机（无 Ryu）**未做运行验证**；
  与检测服务之间的接口、判定与处置逻辑，已由 `tools/controller_simulator.py` 在无 Ryu 环境下
  端到端验证（见 `results/e2e_controller_report.md`）。
* **默认演练模式（dry_run=True）**：只打印将要下发的规则，不真正改流表。
  确认无误后再用 `--execute` 打开真实处置。
* **白名单**：控制器自身、管理网段、网关必须在 `--whitelist` 里，否则可能把管理链路断掉。
* 处置动作有超时（`--block-seconds`），到期自动解除，避免误封长时间影响业务。

用法（在 Mininet 环境中）::

    # 1) 先起检测服务（可跑在控制器同一台机器上）
    #    python app/app.py --port 5000 --processed-dir data/processed      # CICFlowMeter 字段模型
    #    python app/app.py --port 5000 --processed-dir data/processed_sdn  # 控制器原生字段模型
    # 2) 起 Ryu 控制器
    ryu-manager integrations/ryu_guard.py --observe-links \\
        --detect-url http://127.0.0.1:5000 --whitelist 10.0.0.1 --block-seconds 60
    # 3) 打开真实处置（确认演练输出无误后）
    ryu-manager integrations/ryu_guard.py --execute

说明：`ryu-manager` 会把 `--xxx` 参数透传给本模块的 `add_arguments` 所定义的选项。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

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


class GuardSwitch(simple_switch_13.SimpleSwitch13):
    """带异常检测与自动处置的 SDN 控制器应用。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.detect_url = kwargs.get("detect_url", "http://127.0.0.1:5000")
        self.execute = bool(kwargs.get("execute", False))
        self.block_seconds = int(kwargs.get("block_seconds", 60))
        self.whitelist = set(kwargs.get("whitelist", []) or [])
        self.audit_path = kwargs.get("audit", os.path.join("results", "audit_alerts_ryu.jsonl"))
        self.blocked: dict[str, float] = {}
        os.makedirs(os.path.dirname(self.audit_path) or ".", exist_ok=True)
        self.logger.info("[guard] 检测服务=%s 真实处置=%s 白名单=%s",
                         self.detect_url, self.execute, sorted(self.whitelist) or "（空，建议至少加入控制器与管理网段）")
        if not self.execute:
            self.logger.warning("[guard] 当前为演练模式：只输出将要下发的规则，不改动流表（加 --execute 开启真实处置）")
        self.monitor_thread = hub.spawn(self._monitor)

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
        """周期性拉取流表统计（周期与控制器采集策略一致）。"""
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            hub.sleep(10)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPFlowStatsRequest(datapath))

    # ------------------------------------------------------------------ 检测 + 处置
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
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

    def _handle_alert(self, alert: dict, datapath):
        src = alert.get("src_ip") or ""
        action = alert.get("action", "observe")
        executed = False
        if action in ("block", "throttle") and src and src not in self.whitelist:
            if self.execute:
                executed = True if action == "block" else self._meter_or_limit(datapath, src)
                if executed:
                    self.blocked[src] = time.time() + self.block_seconds
            else:
                self.logger.warning("[guard][演练] 将下发动作 action=%s src=%s（到期自动解除 %ds）",
                                    action, src, self.block_seconds)
        record = {
            "time": time.time(), "src_ip": src, "dst_ip": alert.get("dst_ip"),
            "dst_port": alert.get("dst_port"), "pred_class": alert.get("pred_class"),
            "p_attack": alert.get("p_attack"), "action": action,
            "executed": executed, "dry_run": not self.execute,
            "block_seconds": self.block_seconds,
            "datapath_id": datapath.id if datapath else None,
        }
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.logger.warning("[guard] 告警：%s p=%.3f src=%s → %s%s",
                            record["pred_class"], float(record["p_attack"] or 0), src, action,
                            "（已下发）" if executed else "（仅记录）")

    def _meter_or_limit(self, datapath, src):
        """限速：优先使用 OpenFlow Meter（需要交换机支持），失败则退化为丢弃。"""
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        try:
            bands = [parser.OFPMeterBandDrop(rate=1000, burst_size=100)]
            datapath.send_msg(parser.OFPMeterMod(datapath=datapath, command=ofproto.OFPMC_ADD,
                                                 flags=ofproto.OFPMF_KBPS, meter_id=1, bands=bands))
            self.logger.info("[guard] 已下发限速 Meter（1 Mbps）src=%s", src)
            return True
        except Exception as exc:  # 交换机不支持 meter 时退化为丢弃
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


def add_arguments(parser):
    """向 ryu-manager 注册命令行参数。"""
    parser.add_argument("--detect-url", default="http://127.0.0.1:5000", help="检测服务地址")
    parser.add_argument("--execute", action="store_true", help="开启真实处置（默认仅演练，不改流表）")
    parser.add_argument("--block-seconds", type=int, default=60, help="处置有效期（秒），到期自动解除")
    parser.add_argument("--whitelist", nargs="*", default=[], help="永不处置的源 IP（控制器/管理网段/网关）")
    parser.add_argument("--audit", default=os.path.join("results", "audit_alerts_ryu.jsonl"), help="审计台账路径")
    return parser
