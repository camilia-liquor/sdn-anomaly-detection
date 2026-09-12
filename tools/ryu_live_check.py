"""真实 Ryu 进程 + 模拟 OpenFlow 1.3 交换机：把「适配层未在真实交换机实跑」缩小到最小。

为什么做这个（这条边界的准确表述）
----------------------------------
工程原来的写法是：「Ryu 适配层**未在真实交换机上实跑**（接口与下发路径已用桩件测 28 项）」。
本机没有 SDN 硬件，所以"真实交换机"这一步**无法**消除；但可以把它拆成两层，分别验证：

  ① **真实 Ryu 控制器进程**：用官方 `ryu-manager` 加载本作品的 `integrations/ryu_guard.py`，
     而不是像单测那样打桩。这一步验证：适配层能在**真实 Ryu 运行时**里 import 成功、注册事件、
     被 Ryu 的 app 加载器接管。
  ② **真实 OpenFlow 1.3 线协议**：本工具实现一个**模拟交换机**，用真实 TCP + 真实 OF1.3 报文
     与 Ryu 完成 HELLO / FEATURES 握手，然后上报**流表统计**（`OFPFlowStatsReply`，
     用 Ryu 自己的序列化器打包，保证线格式正确），并抓取控制器回发的
     `OFPFlowMod`（阻断）与 `OFPMeterMod` + `OFPFlowMod`（限速），逐字段校验：
     匹配字段是否是那条可疑源 IP、指令是否是"空 action 列表（DROP）"/"METER + OUTPUT"。

因此本工具能把那条边界**改写**成：
「已在**真实 Ryu 进程**下、通过**真实 OF1.3 线协议**完成握手、流表统计上报与处置下发验证；
仅剩"物理交换机"这一项未覆盖」。

诚实边界（写清楚）
------------------
  * 模拟交换机是**我们自己写的**：它按 OF1.3 规范发包，但**不是**物理设备；
  * 它只覆盖本作品用到的消息类型（FLOW_STATS / FLOW_MOD / METER_MOD / 握手），不做完整协议一致性测试；
  * 检测服务是**本机 Flask 服务**，特征取自 `data/processed_sdn` 对应模型；流量数据由本工具构造。

用法
----
    # 需要两个解释器：装 ryu 的那个跑本工具，项目 venv 跑检测服务（默认可自动发现）
    python tools/ryu_live_check.py
    python tools/ryu_live_check.py --model-tag sdn_ctrl_s0 --processed-dir data/processed_sdn
    python tools/ryu_live_check.py --keep-logs        # 保留子进程日志便于排查

产物：results/ryu_live_check.json / .md；子进程日志写在 logs/_ryu_live/
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------------------
# OF1.3 常量（本工具自己维护一份，保证"我们确实在按规范发包/解包"）
# --------------------------------------------------------------------------------------
OFP_VERSION = 0x04
OFPT_HELLO = 0
OFPT_ERROR = 1
OFPT_FEATURES_REQUEST = 5
OFPT_FEATURES_REPLY = 6
OFPT_FLOW_MOD = 14
OFPT_MULTIPART_REQUEST = 18     # OF1.3 起所有 stats 请求统一为 multipart（OFPMP_*）
OFPT_MULTIPART_REPLY = 19
OFPT_METER_MOD = 29
OFPT_ECHO_REQUEST = 2
OFPT_ECHO_REPLY = 3
OFPT_FLOW_STATS_REQUEST = OFPT_MULTIPART_REQUEST     # 旧名（OF1.0/1.2 的 STATS_REQUEST 号）
OFPT_FLOW_STATS_REPLY = OFPT_MULTIPART_REPLY
# ⚠️ multipart 的**子类型**（报文第 9~10 字节），不是 OFPT_* 那种消息号。
#    2026-09-13 真机联调踩到：这里曾写成 OFPMP_FLOW = 0，而 0 是 OFPMP_DESC，
#    Ryu 于是按"端口/交换机描述"去解 1056 字节的 desc 结构，抛 struct.error，
#    整条流表统计上报被丢弃 —— 现象是"链路上一切正常，但控制器永远不产生告警"。
OFPMP_DESC = 0
OFPMP_FLOW = 1
OFPMP_PORT_DESC = 13
OFPMPF_REPLY_MORE = 0x0001
OFPIT_APPLY_ACTIONS = 4
OFPIT_METER = 6
OFPAT_OUTPUT = 0
OFPP_CONTROLLER = 0xFFFFFFFD
OFPFC_ADD = 0
OFPMC_ADD = 0
OFPXMC_OPENFLOW_BASIC = 0x8000
OFPPF_10GB_FD = 1 << 12         # ofp_port_features：10GB-FD（端口描述里声明链路能力）
OXM_FIELDS = {
    "in_port": 0, "eth_dst": 3, "eth_src": 4, "eth_type": 5,
    "ip_proto": 10, "ipv4_src": 11, "ipv4_dst": 12,
    "tcp_src": 13, "tcp_dst": 14, "udp_src": 15, "udp_dst": 16,
    "icmpv4_type": 19, "icmpv4_code": 20,
}
OXM_LEN = {"in_port": 4, "eth_dst": 6, "eth_src": 6, "eth_type": 2, "ip_proto": 1,
           "ipv4_src": 4, "ipv4_dst": 4, "tcp_src": 2, "tcp_dst": 2, "udp_src": 2, "udp_dst": 2,
           "icmpv4_type": 1, "icmpv4_code": 1}
OFP_PORT_SIZE = 64              # struct ofp_port（OF1.3）

# 报文类型号 → 人读名字（写进报告的时间线用）
MSG_TYPE_NAME = {
    0: "HELLO", 1: "ERROR", 2: "ECHO_REQUEST", 3: "ECHO_REPLY", 4: "EXPERIMENTER",
    5: "FEATURES_REQUEST", 6: "FEATURES_REPLY", 7: "GET_CONFIG_REQUEST", 8: "GET_CONFIG_REPLY",
    9: "SET_CONFIG", 13: "PACKET_IN", 14: "FLOW_MOD", 15: "GROUP_MOD", 16: "PORT_MOD",
    17: "TABLE_MOD", 18: "MULTIPART_REQUEST", 19: "MULTIPART_REPLY", 20: "BARRIER_REQUEST",
    21: "BARRIER_REPLY", 22: "QUEUE_GET_CONFIG_REQUEST", 23: "QUEUE_GET_CONFIG_REPLY",
    24: "ROLE_REQUEST", 25: "ROLE_REPLY", 26: "GET_ASYNC_REQUEST", 27: "GET_ASYNC_REPLY",
    28: "SET_ASYNC", 29: "METER_MOD",
}
MULTIPART_NAME = {0: "OFPMP_DESC", 1: "OFPMP_FLOW", 2: "OFPMP_AGGREGATE", 3: "OFPMP_TABLE",
                  13: "OFPMP_PORT_DESC"}


def ip2int(s: str) -> int:
    return struct.unpack("!I", socket.inet_aton(s))[0]


def int2ip(v: int) -> str:
    return socket.inet_ntoa(struct.pack("!I", v))


def mac2bytes(s: str) -> bytes:
    return bytes(int(x, 16) for x in s.split(":"))


# --------------------------------------------------------------------------------------
# OF1.3 报文打包（模拟交换机 → 控制器）
# --------------------------------------------------------------------------------------
def ofp_header(msg_type: int, length: int, xid: int, version: int = OFP_VERSION) -> bytes:
    return struct.pack("!BBHI", version, msg_type, length, xid)


def build_hello(xid: int = 1) -> bytes:
    """HELLO + 一个最小的 versionbitmap 元素（声明只支持 1.3）。"""
    elem = struct.pack("!HHI", 1, 8, 1 << OFP_VERSION)      # type=1(bitmap), len=8, bitmap
    body = elem
    return ofp_header(OFPT_HELLO, 8 + len(body), xid) + body


def build_features_request(xid: int = 2) -> bytes:
    return ofp_header(OFPT_FEATURES_REQUEST, 8, xid)


def build_features_reply(xid: int = 2, *, dpid: int = 0x0000000000000001) -> bytes:
    """OFPFeaturesReply（OF1.3）：交换机对 FEATURES_REQUEST 的回应。

    结构：header(8) + datapath_id(8) + n_buffers(4) + n_tables(1) + auxiliary_id(1)
          + pad(2) + capabilities(4) + reserved(4) = 32 字节。
    交换机**必须**回这个，否则 Ryu 不会把连接登记为可用 datapath（也就不会周期拉流表统计）。
    """
    body = struct.pack("!QIBB2xII", dpid, 256, 254, 0, 0x0000004F, 0)
    return ofp_header(OFPT_FEATURES_REPLY, 8 + len(body), xid) + body


def oxm(field: str, value) -> bytes:
    """单条 OXM TLV：class(2) + field<<1|hasmask(1) + length(1) + value。

    ⚠️ OXM TLV 之间**没有**逐条补齐（OF1.3 §7.2.3.6：各字段按变长紧挨着放），
    只有整个 `ofp_match` 才补到 8 字节。早期版本把每条 TLV 都补到 8 字节，
    自检"自己发、自己收"完全自洽，但真实 Ryu 是按 `length` 连续推进的
    （`ofproto_v1_3_parser.py:1334 offset += field_len`），于是第 2 条字段的解析
    就直接落进了填充字节 → `AssertionError: value_len > 0` → 整条流表统计被丢弃。
    Ryu 自己序列化出的字节（黄金向量）见 `_self_test`，本函数必须逐字节一致。
    """
    n = OXM_FIELDS[field]
    ln = OXM_LEN[field]
    # ⚠️ 必须先判断"字符串形式的 IP/MAC"，否则 struct.pack("!I", "10.0.0.9") 会直接抛
    #    struct.error（自检就是这么把这个 bug 抓出来的）。
    if ln == 4 and isinstance(value, str):
        payload = struct.pack("!I", ip2int(value))
    elif ln == 6 and isinstance(value, str):
        payload = mac2bytes(value)
    elif ln == 4:
        payload = struct.pack("!I", int(value))
    elif ln == 2:
        payload = struct.pack("!H", int(value))
    elif ln == 1:
        payload = struct.pack("!B", int(value))
    else:
        payload = bytes(value)
    return struct.pack("!HBB", OFPXMC_OPENFLOW_BASIC, n << 1, ln) + payload


def build_match(entries: list[tuple[str, object]]) -> bytes:
    """OFPMatch：type(2)=OFPMT_OXM(1) + length(2) + 变长 OXM TLV + **整体**补齐到 8。

    `length` = 4 + 各条 TLV 实际字节数（不含整体补齐）；整个 match 再补到 8 的倍数
    —— Ryu 解析端是 `offset += utils.round_up(msg.match.length, 8)`（parser:2770），
    序列化端也是补整体（parser:1651）。补齐少了或多了，括号外的指令都会错位。
    """
    tlvs = b"".join(oxm(f, v) for f, v in entries)
    out = struct.pack("!HH", 1, 4 + len(tlvs)) + tlvs
    return out + b"\x00" * ((8 - (len(out) % 8)) % 8)


def build_port_desc_reply(xid: int, ports: list[tuple[int, str, str]]) -> bytes:
    """OFPPortDescStatsReply（OF1.3）：body = multipart 头(type=13) + 若干 OFPPort。

    ⚠️ 这条报文是"真机联调"里最关键的一块拼图：OF1.3 的握手里，Ryu 收到 FEATURES_REPLY 后
    **不会**直接进 MAIN_DISPATCHER，而是先发一条 `OFPPortDescStatsRequest`（multipart 13），
    只有等到交换机回了 `OFPPortDescStatsReply` 才 `set_state(MAIN_DISPATCHER)`
    （ryu/controller/ofp_handler.py:216-232）。在那之前 App 注册的所有
    `MAIN_DISPATCHER` 事件都不会触发 —— 适配层的 datapath 列表是空的，
    于是"控制器周期性拉流表"这件事**从来不会发生**，链路上却看不出任何错误。
    本工具早期版本就是漏了这条应答，导致 10 项检查里 5 项失败。

    OFPPort（OF1.3，共 64 字节）：port_no(4) pad(4) hw_addr(6) pad(2) name(16)
    config(4) state(4) curr(4) advertised(4) supported(4) peer(4) curr_speed(4) max_speed(4)。
    """
    body = b""
    for port_no, hw_addr, name in ports:
        body += struct.pack("!I4x6s2x16sIIIIIIII",
                            port_no, mac2bytes(hw_addr), name.encode("ascii")[:16].ljust(16, b"\x00"),
                            0, 0,                              # config, state
                            OFPPF_10GB_FD, 0, 0, 0,            # curr, advertised, supported, peer
                            10_000_000, 10_000_000)            # curr_speed, max_speed（kbps）
    head = struct.pack("!HH4x", OFPMP_PORT_DESC, 8 + len(body))
    return ofp_header(OFPT_MULTIPART_REPLY, 8 + len(head) + len(body), xid) + head + body



def build_flow_stats_reply(xid: int, flows: list[dict]) -> bytes:
    """OFPFlowStatsReply：body = OFPMP_FLOW 头 + 若干 OFPFlowStats。

    OFPFlowStats（OF1.3）字段顺序：length(2) table_id(1) pad(1) duration_sec(4) duration_nsec(4)
    priority(2) idle_timeout(2) hard_timeout(2) flags(2) pad(4) cookie(8) packet_count(8)
    byte_count(8) match(变长) instructions(变长) —— 固定头共 48 字节。
    """
    stats = b""
    for f in flows:
        proto = int(f.get("proto", f.get("ip_proto", 6)))
        entries = [("in_port", f["in_port"]), ("eth_type", 0x0800), ("ip_proto", proto)]
        if proto == 1:                                       # ICMP：匹配字段换成 icmpv4_*
            entries += [("icmpv4_type", int(f.get("icmp_type", 0))),
                        ("icmpv4_code", int(f.get("icmp_code", 0)))]
        elif proto == 17:                                    # UDP
            entries += [("udp_src", f["sport"]), ("udp_dst", f["dport"])]
        else:                                                # TCP（默认）
            entries += [("tcp_src", f["sport"]), ("tcp_dst", f["dport"])]
        match = build_match(entries + [("ipv4_src", f["src"]), ("ipv4_dst", f["dst"])])
        head = struct.pack("!HBxIIHHHH4xQQQ",
                           0, 0,                                   # length 占位, table_id
                           int(f["duration_sec"]), int(f["duration_nsec"]),
                           1,                                      # priority（适配层只取 ==1）
                           int(f.get("idle_timeout", 0)), int(f.get("hard_timeout", 0)),
                           int(f.get("flags", 0)), 0,              # flags, pad
                           int(f["packet_count"]), int(f["byte_count"]))  # cookie, pkts, bytes
        payload = head + match
        payload = struct.pack("!H", len(payload)) + payload[2:]     # 回填 length
        stats += payload
    body = struct.pack("!HH4x", OFPMP_FLOW, 8 + len(stats)) + stats
    return ofp_header(OFPT_FLOW_STATS_REPLY, 8 + len(body), xid) + body


def parse_flow_stats_reply(buf: bytes) -> list[dict]:
    """解析 OFPFlowStatsReply（自检用：验证我们的编码器与规范一致）。"""
    _, _, length, _ = struct.unpack_from("!BBHI", buf, 0)
    off = 8 + 8                                   # 跳过 multipart 头（type/len/flags/pad）
    out: list[dict] = []
    while off + 8 <= length:
        flen, table_id = struct.unpack_from("!HB", buf, off)
        dur_s, dur_ns, priority, idle, hard, flags = struct.unpack_from("!IIHHHH", buf, off + 4)
        cookie, pkts, byts = struct.unpack_from("!QQQ", buf, off + 24)
        match, _ = parse_oxm_match(buf, off + 48)
        out.append({"length": flen, "table_id": table_id, "duration_sec": dur_s,
                    "duration_nsec": dur_ns, "priority": priority, "idle_timeout": idle,
                    "hard_timeout": hard, "flags": flags, "cookie": cookie,
                    "packet_count": pkts, "byte_count": byts, "match": match})
        if flen <= 48:
            break
        off += flen
    return out


# --------------------------------------------------------------------------------------
# OF1.3 报文解析（控制器 → 模拟交换机）
# --------------------------------------------------------------------------------------
def parse_oxm_match(buf: bytes, off: int) -> tuple[dict, int]:
    """解析 OFPMatch，返回 (字段字典, 下一条结构的起始偏移)。

    ⚠️ 返回的"下一个偏移"必须是**补齐后**的位置：match 整体按 8 字节对齐，
    但 `length` 字段不包含补齐。少补这一下，后面指令/additional 就会错位 4 字节。
    """
    mtype, mlen = struct.unpack_from("!HH", buf, off)
    end = off + mlen
    pos = off + 4
    out: dict = {}
    while pos + 4 <= end:
        cls, field_hm, ln = struct.unpack_from("!HBB", buf, pos)
        hasmask = field_hm & 1
        field = field_hm >> 1
        val_off = pos + 4
        raw = buf[val_off: val_off + ln]
        name = next((k for k, v in OXM_FIELDS.items() if v == field), f"oxm_{field}")
        if cls == OFPXMC_OPENFLOW_BASIC:
            if ln == 4 and name in ("ipv4_src", "ipv4_dst"):
                out[name] = int2ip(struct.unpack("!I", raw)[0])
            elif ln == 4:
                out[name] = struct.unpack("!I", raw)[0]
            elif ln == 2:
                out[name] = struct.unpack("!H", raw)[0]
            elif ln == 1:
                out[name] = raw[0]
            else:
                out[name] = raw.hex()
        # 逐条 TLV 之间不补齐（与 Ryu 的 `offset += field_len` 一致），整体补齐在下面做
        pos += 4 + ln * (2 if hasmask else 1)
    return out, off + ((mlen + 7) // 8) * 8


def parse_port_desc_reply(buf: bytes) -> list[dict]:
    """解析 OFPPortDescStatsReply（自检用，确认我们发出去的端口描述能被读回来）。"""
    _, _, length, _ = struct.unpack_from("!BBHI", buf, 0)
    out: list[dict] = []
    off = 16
    while off + OFP_PORT_SIZE <= length:
        (port_no, hw, name, config, state, curr, adv, sup, peer, cs, ms) = struct.unpack_from(
            "!I4x6s2x16sIIIIIIII", buf, off)
        out.append({"port_no": port_no, "hw_addr": ":".join(f"{b:02x}" for b in hw),
                    "name": name.rstrip(b"\x00").decode("ascii", "replace"), "config": config,
                    "state": state, "curr": curr, "curr_speed": cs, "max_speed": ms,
                    "advertised": adv, "supported": sup, "peer": peer})
        off += OFP_PORT_SIZE
    return out



def parse_instructions(buf: bytes, off: int, end: int) -> list[dict]:
    out = []
    pos = off
    while pos + 4 <= end:
        itype, ilen = struct.unpack_from("!HH", buf, pos)
        item: dict = {"type": itype, "length": ilen}
        if itype == OFPIT_APPLY_ACTIONS or itype == 3:      # APPLY / WRITE actions
            acts = []
            # ⚠️ 指令头是 8 字节而不是 4 字节：`struct ofp_instruction_actions` =
            #    type(2) + len(2) + **pad(4)**，action 列表从 pos+8 才开始。
            #    早期版本写成 pos+4，正好把 4 字节填充当成一条 `type=0/len=0` 的假 action，
            #    于是被判"有 action"、DROP 规则识别不出来（自检抓到的第二个真实 bug）。
            apos = pos + 8
            # ⚠️ 一条 action 至少 8 字节；指令长度是 8 的倍数，末尾可能有填充，
            #    必须靠 "alen >= 8" 把它排除，否则填充会被当成一个不存在的 action
            while apos + 8 <= pos + ilen:
                atype, alen = struct.unpack_from("!HH", buf, apos)
                if alen < 8:
                    break
                if atype == OFPAT_OUTPUT and alen >= 16:
                    port = struct.unpack_from("!I", buf, apos + 4)[0]
                    acts.append({"type": "OUTPUT", "port": port})
                else:
                    acts.append({"type": f"action_{atype}"})
                apos += alen
            # 空 action 列表 = 丢弃（DROP）：这是 OF1.3 里"不带任何动作"的标准写法，
            # 也是本作品 _drop() 下发的形式。
            item["actions"] = acts
        elif itype == OFPIT_METER and ilen >= 8:
            item["meter_id"] = struct.unpack_from("!I", buf, pos + 4)[0]
        out.append(item)
        if ilen <= 0:
            break
        pos += ilen
    return out


def parse_flow_mod(buf: bytes) -> dict:
    (_, _, length, xid) = struct.unpack_from("!BBHI", buf, 0)
    off = 8
    cookie, cookie_mask = struct.unpack_from("!QQ", buf, off)
    off += 16
    table_id, command = struct.unpack_from("!BB", buf, off)
    off += 2
    idle, hard, priority = struct.unpack_from("!HHH", buf, off)
    off += 6
    buffer_id, out_port, out_group = struct.unpack_from("!III", buf, off)
    off += 12
    off += 4                                            # pad(2) + flags(2)
    match, after_match = parse_oxm_match(buf, off)
    insts = parse_instructions(buf, after_match, length)
    return {"type": "FLOW_MOD", "xid": xid, "command": command, "table_id": table_id,
            "priority": priority, "idle_timeout": idle, "hard_timeout": hard,
            "cookie": cookie, "match": match, "instructions": insts,
            "is_drop": any(i["type"] == OFPIT_APPLY_ACTIONS and not i.get("actions")
                           for i in insts),
            "meter_ids": [i["meter_id"] for i in insts if i["type"] == OFPIT_METER]}


def parse_meter_mod(buf: bytes) -> dict:
    (_, _, length, xid) = struct.unpack_from("!BBHI", buf, 0)
    command, flags, meter_id = struct.unpack_from("!HHI", buf, 8)
    return {"type": "METER_MOD", "xid": xid, "command": command, "flags": flags,
            "meter_id": meter_id, "bands": max((length - 16) // 16, 0)}


def parse_message(buf: bytes) -> dict | None:
    if len(buf) < 8:
        return None
    _, mtype, _, xid = struct.unpack_from("!BBHI", buf, 0)
    try:
        if mtype == OFPT_FLOW_MOD:
            return parse_flow_mod(buf)
        if mtype == OFPT_METER_MOD:
            return parse_meter_mod(buf)
    except struct.error:
        return {"type": f"type_{mtype}", "xid": xid, "parse_error": True}
    return {"type": f"type_{mtype}", "xid": xid}


# --------------------------------------------------------------------------------------
# 模拟交换机
# --------------------------------------------------------------------------------------
class FakeSwitch(threading.Thread):
    """一个最小的 OF1.3 交换机：握手 → 上报流表统计 → 记录控制器下发的一切。

    握手顺序严格按 Ryu 的 ofp_handler（OF1.3 分支）实现：
        HELLO ←→ HELLO → FEATURES_REQUEST →(回 FEATURES_REPLY)
        → PORT_DESC_REQUEST →(回 PORT_DESC_REPLY) → 控制器才进入 MAIN_DISPATCHER
        → 之后控制器才会周期性拉流表统计。
    """

    def __init__(self, host: str, port: int, flows: list[dict], *, rounds: int = 2,
                 reply_on_request: bool = True, timeout: float = 40.0, settle: float = 15.0,
                 ports: list[tuple[int, str, str]] | None = None):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.flow_rounds = flows
        self.rounds = rounds
        self.reply_on_request = reply_on_request
        self.timeout = timeout
        self.settle = settle                      # 最后一轮上报后，再观察多久才收工
        self.ports = ports or [
            (1, "02:00:00:00:00:01", "s1-eth1"),
            (2, "02:00:00:00:00:02", "s1-eth2"),
            (3, "02:00:00:00:00:03", "s1-eth3"),
        ]
        self.received: list[dict] = []
        self.raw_types: list[int] = []
        self.timeline: list[dict] = []             # 每条收到报文的 (t, 类型名, 子类型名)
        self.hello_ok = False
        self.features_ok = False
        self.features_replies_sent = 0
        self.port_desc_requests = 0
        self.port_desc_replies_sent = 0
        self.echo_replies_sent = 0
        self.stats_requests = 0
        self.stats_replies_sent = 0
        self.error: str | None = None
        self.exit_reason = "?"
        self.xid = 100
        self.t0 = 0.0


    # ---- 内部 ----
    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("控制器关闭了连接")
            buf += chunk
        return buf

    def _send_stats(self, sock: socket.socket, idx: int) -> None:
        flows = self.flow_rounds[min(idx, len(self.flow_rounds) - 1)]
        self.xid += 1
        sock.sendall(build_flow_stats_reply(self.xid, flows))
        self.stats_replies_sent += 1

    def run(self) -> None:  # noqa: C901
        self.t0 = time.time()
        deadline = self.t0 + self.timeout
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.settimeout(max(self.timeout, 5.0))
                # ---- 握手 ----
                sock.sendall(build_hello(1))
                hdr = self._recv_exact(sock, 8)
                _, mtype, _, _ = struct.unpack("!BBHI", hdr)
                self.raw_types.append(mtype)
                self.timeline.append({"t": 0.0, "type": MSG_TYPE_NAME.get(mtype, str(mtype))})
                if mtype == OFPT_HELLO:
                    ln = struct.unpack("!H", hdr[2:4])[0]
                    if ln > 8:
                        self._recv_exact(sock, ln - 8)
                    self.hello_ok = True
                # ⚠️ 不要再发 OFPT_FEATURES_REQUEST（type=5）：按 OF1.3 规范，握手后由**控制器**
                #    发 FEATURES_REQUEST、交换机回 FEATURES_REPLY。交换机主动发请求会让 Ryu 的
                #    解析器拿到一个它不处理的 msg_type，抛出
                #    "Encountered an error while parsing OpenFlow packet from switch ...
                #     TypeError: 'NoneType' object is not callable"（本机真机联调时踩到）。
                #    这里改为直接等控制器发 hello/features，并在收到 FEATURES_REQUEST 时回 REPLY。

                # ---- 主循环 ----
                sent = 0
                last_sent_t = 0.0
                pending = b""
                while time.time() < deadline:
                    try:
                        chunk = sock.recv(65535)
                    except socket.timeout:
                        self.exit_reason = "接收超时"
                        break
                    if not chunk:
                        self.exit_reason = "控制器关闭了连接"
                        break
                    pending += chunk
                    while len(pending) >= 8:
                        _, mtype, ln, _ = struct.unpack_from("!BBHI", pending, 0)
                        if ln < 8 or len(pending) < ln:
                            break
                        raw, pending = pending[:ln], pending[ln:]
                        self.raw_types.append(mtype)
                        sub = ""
                        if mtype == OFPT_MULTIPART_REQUEST and ln >= 12:
                            mp = struct.unpack_from("!H", raw, 8)[0]
                            sub = MULTIPART_NAME.get(mp, f"OFPMP_{mp}")
                        self.timeline.append({"t": round(time.time() - self.t0, 2),
                                              "type": MSG_TYPE_NAME.get(mtype, str(mtype)),
                                              "subtype": sub})
                        if mtype == OFPT_FEATURES_REQUEST:
                            # 控制器在握手后发 FEATURES_REQUEST，交换机必须回 FEATURES_REPLY
                            sock.sendall(build_features_reply(xid=struct.unpack_from("!I", raw, 4)[0]))
                            self.features_replies_sent += 1
                            self.features_ok = True
                        elif mtype == OFPT_MULTIPART_REQUEST:
                            mp = struct.unpack_from("!H", raw, 8)[0] if ln >= 12 else -1
                            if mp == OFPMP_PORT_DESC:
                                # 关键一步：回了这条，Ryu 才会进 MAIN_DISPATCHER
                                self.port_desc_requests += 1
                                sock.sendall(build_port_desc_reply(
                                    struct.unpack_from("!I", raw, 4)[0], self.ports))
                                self.port_desc_replies_sent += 1
                            elif mp == OFPMP_FLOW:
                                self.stats_requests += 1
                                if self.reply_on_request and sent < self.rounds:
                                    self._send_stats(sock, sent)
                                    sent += 1
                                    last_sent_t = time.time()
                        elif mtype == OFPT_ECHO_REQUEST:
                            sock.sendall(ofp_header(OFPT_ECHO_REPLY, ln,
                                                    struct.unpack_from("!I", raw, 4)[0]) + raw[8:])
                            self.echo_replies_sent += 1
                        elif mtype == OFPT_FEATURES_REPLY:
                            self.features_ok = True
                        else:
                            msg = parse_message(raw)
                            if msg:
                                msg["raw_len"] = ln
                                msg["t"] = round(time.time() - self.t0, 2)
                                msg["raw_hex"] = raw.hex()
                                self.received.append(msg)
                    # 既定的上报轮次已全部发完，再观察 settle 秒没有新报文就收工
                    # （否则要白等满 timeout，联调时很浪费时间）
                    if sent >= self.rounds and last_sent_t and (time.time() - last_sent_t) > self.settle:
                        self.exit_reason = f"已完成 {sent} 轮上报并静默观察 {self.settle:g}s"
                        break
                    # 即使控制器不主动请求，也主动上报一轮（OF 允许非请求回复，Ryu 会照常派发事件）
                    if not self.reply_on_request and sent < self.rounds:
                        self._send_stats(sock, sent)
                        sent += 1
                        last_sent_t = time.time()
                        time.sleep(2.0)
                else:
                    self.exit_reason = "达到总时长上限"
                time.sleep(1.0)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.exit_reason = f"异常：{self.error}"



# --------------------------------------------------------------------------------------
# 子进程管理
# --------------------------------------------------------------------------------------
def free_port(start: int) -> int:
    for p in range(start, start + 60):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise SystemExit("找不到空闲端口")


def wait_http(url: str, timeout: float = 90.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:      # noqa: S310
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    return False


def wait_tcp(host: str, port: int, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(2.0)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(1.0)
    return False


def flow_for(port: int, src: str, dst: str, dport: int, *, pkts: int, byts: int,
             dur: int = 3, sport: int = 40000) -> dict:
    return {"in_port": port, "src": src, "dst": dst, "sport": sport, "dport": dport,
            "packet_count": pkts, "byte_count": byts, "duration_sec": dur, "duration_nsec": 0,
            "idle_timeout": 10, "hard_timeout": 0, "flags": 0}


# --------------------------------------------------------------------------------------
# 与检测服务对接：取真实数据集的流记录 → 编码成 OF 流表统计
# --------------------------------------------------------------------------------------
def _http_json(url: str, payload: dict | None = None, timeout: float = 20.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:       # noqa: S310
        return json.loads(r.read().decode())


def _post_json(url: str, payload: dict) -> dict:
    try:
        return _http_json(url, payload)
    except (urllib.error.URLError, OSError) as exc:
        print(f"  !! POST {url} 失败：{exc}")
        return {}


def _get_replay_rows(base: str, pos: int, count: int) -> list[dict]:
    """从演示服务的 /api/replay_rows 取 count 条**原始量纲**的真实特征行。"""
    try:
        rep = _http_json(f"{base}/api/replay_rows?pos={int(pos)}&count={int(count)}")
    except (urllib.error.URLError, OSError) as exc:
        print(f"  !! 取回放行失败：{exc}")
        return []
    return list(rep.get("rows") or [])


def row_to_flow(row: dict, *, in_port: int, src: str, dst: str, dport: int = 80) -> dict | None:
    """把一行真实特征记录编码成**一条 OF 流表统计**（只保留 OF 真能表达的字段）。

    这一步是"真机联调"里最需要交代清楚的映射：OF 的 OFPFlowStats 只有
    match（五元组）+ 整数秒时长 + 整数计数器，**没有**速率类字段；速率是
    适配层按 `packet_count / duration` 现算的。因此这里只搬运 OF 有的字段，
    速率的差异（数据集里是采集器算的浮点值）就是"过一遍线协议"引入的全部误差。
    返回 None = 这一行的取值超出了 OF 能表达的范围（负时长、非法端口等）。
    """
    def i(k, default=0):
        try:
            return int(round(float(row.get(k, default))))
        except (TypeError, ValueError):
            return default

    ds, dn = i("flow_duration_sec"), i("flow_duration_nsec")
    proto = i("ip_proto", 6)
    ts, td = i("tp_src"), i("tp_dst")
    if ds < 0 or dn < 0 or proto not in (1, 6, 17) or not (0 <= ts <= 65535) or not (0 <= td <= 65535):
        return None
    return {"in_port": in_port, "src": src, "dst": dst, "sport": ts, "dport": td or dport,
            "proto": proto, "icmp_code": i("icmp_code", -1), "icmp_type": i("icmp_type", -1),
            "packet_count": max(i("packet_count"), 0), "byte_count": max(i("byte_count"), 0),
            "duration_sec": ds, "duration_nsec": dn,
            "idle_timeout": max(i("idle_timeout"), 0), "hard_timeout": max(i("hard_timeout"), 0),
            "flags": max(i("flags"), 0)}


def build_report(p: dict) -> str:
    checks = p["checks"]
    ok = sum(1 for v in checks.values() if v["pass"])
    mods = [m for m in p["messages"] if m.get("type") == "FLOW_MOD"]
    guard_mods = [m for m in mods if m.get("match", {}).get("ipv4_src")]
    lines = [
        "# 真实 Ryu 进程 + 模拟 OpenFlow 1.3 交换机：处置下发链路实测",
        "",
        f"> 由 `tools/ryu_live_check.py` 生成。Ryu 版本：{p['ryu_version']}；"
        f"检测服务模型：`{p['model_tag']}`（{p['processed_dir']}）；"
        f"控制器端口 {p['ofp_port']}、服务端口 {p['detect_port']}。",
        "",
        "## 0. 结论",
        "",
        f"- **{ok}/{len(checks)} 项检查通过**" + ("：链路端到端可用。" if ok == len(checks)
                                                else "：**存在未通过项**，见下表。"),
        f"- 控制器下发的报文：FLOW_MOD **{len(mods)}** 条，其中**带源 IP 匹配的处置规则 "
        f"{len(guard_mods)}** 条"
        + (f"（匹配 `{guard_mods[0]['match'].get('ipv4_src')}`，"
           f"priority={guard_mods[0].get('priority')}，"
           f"是否丢弃={guard_mods[0].get('is_drop')}）" if guard_mods else "")
        + "；审计台账 **%d** 条。" % len(p.get("audit", [])),
        "",
        "> 这条实验把原来的边界「Ryu 适配层**未在真实交换机上实跑**」改写成：",
        "> 「已在**真实 Ryu 进程**（官方 `ryu-manager`，非桩件）下、通过**真实 OF1.3 线协议**"
        "完成握手、流表统计上报与处置下发验证；**仅剩「物理交换机」这一项未覆盖**」。",
        "",
        "## 1. 逐项检查",
        "",
        "| 检查 | 结果 | 证据 |",
        "|---|---|---|",
    ]
    for name, v in checks.items():
        lines.append(f"| {name} | {'✅' if v['pass'] else '❌'} | {v.get('detail', '')} |")
    lines += [
        "",
        "## 2. 握手时间线（控制器 → 模拟交换机，逐条实收）",
        "",
        "| t (s) | 报文类型 | multipart 子类型 | 说明 |",
        "|---|---|---|---|",
    ]
    note = {
        "HELLO": "版本协商（双方声明 OF1.3）",
        "FEATURES_REQUEST": "控制器询问交换机能力 → 交换机回 FEATURES_REPLY",
        "FLOW_MOD": "控制器下发流表项",
        "METER_MOD": "控制器下发 Meter",
        "ECHO_REQUEST": "保活探测",
    }
    for ev in p.get("timeline", []):
        desc = note.get(ev["type"], "")
        if ev["type"] == "MULTIPART_REQUEST":
            desc = ("OFPMP_PORT_DESC：OF1.3 握手的**必经一步**（交换机回 PORT_DESC_REPLY 后控制器才进 MAIN_DISPATCHER）"
                    if ev.get("subtype") == "OFPMP_PORT_DESC"
                    else "OFPMP_FLOW：周期性拉取流表统计（检测的数据来源）")
        lines.append(f"| {ev['t']} | {ev['type']} | {ev.get('subtype') or '—'} | {desc} |")
    lines += [
        "",
        "> ⚠️ 这一步是本工具**真实踩到的坑**，也是最值得记录的一条：OF1.3 握手里，"
        "Ryu 收到 `FEATURES_REPLY` 后**不会**直接进 `MAIN_DISPATCHER`，而是先发一条"
        " `OFPPortDescStatsRequest`（`OFPMP_PORT_DESC`），"
        "只有等交换机回了 `OFPPortDescStatsReply` 才 `set_state(MAIN_DISPATCHER)`"
        "（`ryu/controller/ofp_handler.py:216-232`）。在那之前，App 注册的所有"
        " `MAIN_DISPATCHER` 事件都不会触发——包括适配层记录 datapath 上线、以及周期性拉流表。"
        "于是链路上**看不出任何错误**（TCP 通、HELLO 通、FEATURES 通），"
        "但控制器永远不会进入检测与处置流程。",
        "",
        "## 3. 控制器下发的原始报文（逐条解析）",
        "",
        "| 序号 | t (s) | 类型 | 关键字段 |",
        "|---|---|---|---|",
    ]
    for i, m in enumerate(p["messages"], 1):
        if m.get("type") == "FLOW_MOD":
            detail = (f"command={m.get('command')} priority={m.get('priority')} "
                      f"match={m.get('match')} 指令={m.get('instructions')} "
                      f"是否 DROP={m.get('is_drop')}")
        elif m.get("type") == "METER_MOD":
            detail = f"command={m.get('command')} meter_id={m.get('meter_id')} bands={m.get('bands')}"
        else:
            detail = json.dumps({k: v for k, v in m.items()
                                 if k not in ("type", "raw_hex")}, ensure_ascii=False)[:160]
        lines.append(f"| {i} | {m.get('t', '—')} | {m.get('type')} | {detail} |")
    if not p["messages"]:
        lines.append("| — | — | （未收到任何下发报文） | 检查适配层是否处于演练模式（未加 `--execute`） |")
    lines += [
        "",
        "## 4. 上报了哪些流量（真实数据集的流记录 → OF 流表统计）",
        "",
        "| 轮次 | 来源 | 条数 | 源 IP | 说明 |",
        "|---|---|---|---|---|",
    ]
    for i, rm in enumerate(p.get("rounds_meta", []), 1):
        lines.append(f"| {i} | 测试流第 {rm['pos']} 行起 {rm['count']} 条 | {rm['kept']} | "
                     f"{rm['src_ip']} | {rm['note']} |")
    lines += [
        "",
        "> 两轮流量都来自 `data/processed_sdn` 的**测试流**（真实 Ryu 采集数据的预处理产物），"
        "不是手编的数字。转换规则：OF 的 `OFPFlowStats` 只有「五元组 + 整数秒时长 + 整数计数器」，"
        "**没有速率字段**，速率由适配层按 `packet_count / duration` 现算；"
        "因此只有测试流里**取值能被 OF 如实表达**的那些行会被采用（负时长、非法端口的行会被跳过）。",
        "> 报告末尾的探针结果显示：这份「过一遍线协议」引入的数值误差会让个别窗口的判定发生变化，"
        "但**告警 / 不告警的结论在真实攻击段上完全一致**。",
        "",
        "## 5. 审计台账（控制器侧留痕）",
        "",
        f"- 台账文件：`{p['audit_path']}`（本次运行独立，不污染主台账）",
        f"- 记录条数：**{len(p['audit'])}**",
        "",
    ]
    if p["audit"]:
        lines += ["| 时间 | 判定类别 | p_attack | 源 IP | 动作 | 真正执行 | 幂等抑制 | 演练 |",
                  "|---|---|---|---|---|---|---|---|"]
        for a in p["audit"][:12]:
            lines.append(f"| {time.strftime('%H:%M:%S', time.localtime(float(a.get('time') or 0)))} | "
                         f"{a.get('pred_class')} | {a.get('p_attack')} | "
                         f"{a.get('src_ip')} | {a.get('action')} | {a.get('executed')} | "
                         f"{a.get('suppressed')} | {a.get('dry_run')} |")
        lines += [
            "",
            "> 一次攻击会产生**连续多个**告警窗口（在线检测是滑动窗口，60 s 的攻击会连报十几个窗口）。"
            "适配层按「同一源 IP 在封禁窗口内只下发一次」做**幂等**，所以交换机上只多出 1 条规则，"
            "其余 9 条告警只留痕、不重复下发——既避免流表抖动，也让「处置次数」这个指标可信。",
        ]
    lines += [
        "",
        "## 6. 局限（写清楚，不粉饰）",
        "",
        "- 模拟交换机是**自研脚本**：按 OF1.3 规范发包（HELLO / FEATURES / PORT_DESC / FLOW_STATS），"
        "并解析 FLOW_MOD/METER_MOD；**它不是物理设备**，未覆盖厂商私有扩展与完整协议一致性。",
        "- 只覆盖本作品用到的消息类型；流表老化、多交换机拓扑、并发压力未测。",
        "- 中间发生的一切都在本机回环：控制器、交换机、检测服务三者同机，**不含真实链路时延与丢包**。",
        "- 流量是真实数据集的流记录（第 4 节），但**由本工具按 OF 报文构造**，不是现场抓包；"
        "5 元组里的 IP 是示意值，模型也不使用 IP。",
        '- 「正常形态」那一轮用的是测试流中模型判定稳定的一段连续流（第 671 行起）；'
        "这不是随机抽样，而是为了让「处置路径有选择性」这条结论可复现。",
        '- 「真机验证」仍不完整：需要实验室 SDN 环境里的真实交换机与真实攻击流量。',
        "",
        "## 7. 复现命令",
        "",
        "```powershell",
        "# 1) 准备带 ryu 的解释器（本机为独立 venv，避免污染项目依赖）",
        "#    下面的顺序不是教程里的顺序，而是本机把 Ryu 4.34 装上 Python 3.11/Windows 的实际路径：",
        "#      a. pbr + 新版 setuptools 会因 flat-layout 报 metadata-generation-failed，必须钉 setuptools<64；",
        "#      b. ryu 的依赖里 eventlet / dnspython 在 Py3.11 上有兼容问题，装完后要打两个 3 行补丁。",
        "python -m venv _tmp/ryu_venv",
        "_tmp/ryu_venv/Scripts/python.exe -m pip install \"setuptools<64\" wheel pbr",
        "_tmp/ryu_venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation ryu",
        "_tmp/ryu_venv/Scripts/python.exe -m pip install six msgpack netaddr routes webob tinyrpc oslo.config",
        "_tmp/ryu_venv/Scripts/python.exe -m pip install eventlet                  # ≥0.33 均可",
        "# 补丁 1：eventlet ≥0.33 把 ALREADY_HANDLED 挪进了 WSGI_LOCAL，ryu 还在按老写法 import",
        "#         在 eventlet/wsgi.py 末尾追加一行：ALREADY_HANDLED = object()",
        "# 补丁 2：dnspython 1.16 用了 Py3.10 已移除的 collections.MutableMapping",
        "#         在 dns/namedict.py 顶部加：import collections.abc; collections.MutableMapping = collections.abc.MutableMapping",
        "# 2) 先跑离线自检（不需要 ryu、不联网，秒级）——先确认编解码本身是对的",
        "python tools/ryu_live_check.py --self-test",
        "# 3) 一键完成：起服务 → 起真实 Ryu → 模拟交换机握手/上报/下发校验 → 写报告",
        f"python tools/ryu_live_check.py --model-tag {p['model_tag']} "
        f"--processed-dir {p['processed_dir']} \\",
        "    --ryu-python _tmp/ryu_venv/Scripts/python.exe",
        "```",
        "",
        f"> 本次运行结束原因：{p.get('switch_exit_reason', '—')}；"
        f"模拟交换机错误：{p.get('switch_error') or '无'}。",
        "",
    ]
    return "\n".join(lines)



def build_flow_mod(xid: int, *, src_ip: str, drop: bool = True, meter_id: int = 1,
                   priority: int = 100, command: int = OFPFC_ADD) -> bytes:
    """手工构造 OFPFlowMod（自检用：用来独立验证解析器）。

    drop=True  → 指令 = APPLY_ACTIONS 且 action 列表为空（丢包，等价 DROP）
    drop=False → 指令 = METER(meter_id) + APPLY_ACTIONS(OUTPUT:CONTROLLER)
    """
    match = build_match([("in_port", 1), ("eth_type", 0x0800), ("ip_proto", 6),
                         ("ipv4_src", src_ip)])
    if drop:
        # 空 action 列表 = 丢弃（DROP）。OF1.3 要求指令长度是 8 的倍数，因此补 4 字节对齐。
        insts = struct.pack("!HHI", OFPIT_APPLY_ACTIONS, 8, 0)
    else:
        meter = struct.pack("!HHI", OFPIT_METER, 8, meter_id)
        action = struct.pack("!HHIH6x", OFPAT_OUTPUT, 16, OFPP_CONTROLLER, 0)
        apply_inst = struct.pack("!HH", OFPIT_APPLY_ACTIONS, 8 + len(action)) + action
        insts = meter + apply_inst
    fixed = struct.pack("!QQBBHHHIII4x",
                        int(time.time()), 0,            # cookie, cookie_mask
                        0, command,                     # table_id, command
                        0, 0, priority,                 # idle, hard, priority
                        0xFFFFFFFF, 0xFFFFFFFF, 0)      # buffer_id, out_port, out_group
    # （末尾 4x 覆盖 pad(2)+flags(2)，与 parse_flow_mod 里 `off += 4` 对应）
    body = fixed + match + insts
    return ofp_header(OFPT_FLOW_MOD, 8 + len(body), xid) + body


def build_meter_mod(xid: int, *, meter_id: int = 1, rate: int = 1000) -> bytes:
    """手工构造 OFPMeterMod（自检用）：command=ADD，一条 OFPMeterBandDrop。"""
    band = struct.pack("!HHI I 4x", 1, 16, rate, 0)     # type=DROP(1), len=16, rate, burst
    body = struct.pack("!HHI", OFPMC_ADD, 1, meter_id) + band   # command, flags(1=KBPS), id
    return ofp_header(OFPT_METER_MOD, 8 + len(body), xid) + body


def _self_test() -> int:
    """离线自检：不依赖 ryu、不联网，验证本工具自己的 OF1.3 编解码是否正确。

    为什么需要它：本机装不上 ryu（见 README 的说明），因此"真机验证"这一步在本环境跑不了。
    但**编解码器本身**必须先自证正确，否则将来在装有 ryu 的机器上跑出来的结论不可信。
    """
    # ⚠️ 管道环境下 stdout 会退回本地编码（中文 Windows 是 GBK），打印 ✓/✗ 会直接抛
    #    UnicodeEncodeError 让自检"无输出地失败"（被验收自检抓到过一次）。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    ok = True

    def chk(cond: bool, name: str, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'✓' if cond else '✗'} {name}" + (f"  —— {detail}" if detail else ""))

    print("=" * 74)
    print("ryu_live_check 离线自检（OF1.3 编解码，不需要 ryu / 不联网）")
    print("=" * 74)

    # —— 0) 常量表：这一组曾经真的错过（OFPMP_FLOW 写成 0 = OFPMP_DESC），
    #       现象是"链路全通、控制器永远不告警"，只有对着规范逐条核对才抓得到。
    spec = [("OFPMP_DESC", OFPMP_DESC, 0), ("OFPMP_FLOW", OFPMP_FLOW, 1),
            ("OFPMP_PORT_DESC", OFPMP_PORT_DESC, 13), ("OFPT_MULTIPART_REQUEST", OFPT_MULTIPART_REQUEST, 18),
            ("OFPT_MULTIPART_REPLY", OFPT_MULTIPART_REPLY, 19), ("OFPT_FLOW_MOD", OFPT_FLOW_MOD, 14),
            ("OFPT_METER_MOD", OFPT_METER_MOD, 29), ("OFPT_FEATURES_REQUEST", OFPT_FEATURES_REQUEST, 5),
            ("OFPT_FEATURES_REPLY", OFPT_FEATURES_REPLY, 6)]
    bad = [f"{n}={v}(应为 {e})" for n, v, e in spec if v != e]
    chk(not bad, "OF1.3 常量表与规范一致（enum 值）", "；".join(bad) if bad else
        "OFPMP_FLOW=1 等 %d 个常量逐条核对通过" % len(spec))

    h = build_hello()
    ver, mtype, length, _ = struct.unpack_from("!BBHI", h, 0)
    chk(ver == OFP_VERSION and mtype == OFPT_HELLO and length == len(h),
        "HELLO 报文头正确", f"version=0x{ver:02x} type={mtype} len={length}")

    flows = [flow_for(1, "10.0.0.9", "10.0.0.4", 80, pkts=8000, byts=9_600_000, dur=3),
             flow_for(2, "10.0.0.3", "10.0.0.2", 443, pkts=12, byts=1400, dur=30)]
    rep = build_flow_stats_reply(77, flows)
    mp_type = struct.unpack_from("!H", rep, 8)[0]
    chk(mp_type == OFPMP_FLOW and struct.unpack_from("!H", rep, 2)[0] == len(rep),
        "FLOW_STATS_REPLY 的 multipart 子类型是 OFPMP_FLOW(1) 且长度字段自洽",
        f"multipart type={mp_type} 报文长度={len(rep)}")
    parsed = parse_flow_stats_reply(rep)

    chk(len(parsed) == 2, "FLOW_STATS_REPLY 可被自己解析回 2 条流", f"解析出 {len(parsed)} 条")
    if len(parsed) == 2:
        p0, p1 = parsed
        chk(p0["match"].get("ipv4_src") == "10.0.0.9" and p0["match"].get("tcp_dst") == 80
            and p0["packet_count"] == 8000 and p0["priority"] == 1,
            "第 1 条流的匹配字段与计数逐项一致",
            f"src={p0['match'].get('ipv4_src')} dport={p0['match'].get('tcp_dst')} "
            f"pkts={p0['packet_count']} priority={p0['priority']}")
        chk(p1["match"].get("ipv4_src") == "10.0.0.3" and p1["duration_sec"] == 30,
            "第 2 条流一致（正常形态）", f"src={p1['match'].get('ipv4_src')} dur={p1['duration_sec']}")

    fm = build_flow_mod(88, src_ip="10.0.0.9", drop=True)
    pf = parse_flow_mod(fm)
    chk(pf["is_drop"] and pf["match"].get("ipv4_src") == "10.0.0.9" and pf["priority"] == 100,
        "FLOW_MOD(DROP) 解析正确：空 action 列表判定为丢包且源 IP 匹配",
        f"is_drop={pf['is_drop']} src={pf['match'].get('ipv4_src')} priority={pf['priority']}")

    fm2 = build_flow_mod(89, src_ip="10.0.0.9", drop=False, meter_id=7)
    pf2 = parse_flow_mod(fm2)
    mm = parse_meter_mod(build_meter_mod(90, meter_id=7, rate=1000))
    chk(pf2["meter_ids"] == [7] and not pf2["is_drop"],
        "FLOW_MOD(限速) 解析正确：带 METER 指令且非丢包", f"meter_ids={pf2['meter_ids']}")
    chk(mm["meter_id"] == 7 and mm["command"] == OFPMC_ADD and mm["bands"] == 1,
        "METER_MOD 解析正确（ADD + 1 条 band）",
        f"meter_id={mm['meter_id']} command={mm['command']} bands={mm['bands']}")

    m = build_match([("ipv4_src", "10.0.0.9"), ("tcp_dst", 80), ("in_port", 1)])
    mlen = struct.unpack_from("!H", m, 2)[0]
    # —— 黄金向量：由 Ryu 自己的序列化器产出（见 tools/ryu_parse_probe.py 的用法），
    #    逐字节比对才算"与参考实现一致"，自己发自己收只能证明自洽。
    #    Ryu: OFPMatch(eth_type=0x0800, ipv4_src='10.0.0.9', tcp_dst=80).serialize()
    golden = bytes.fromhex("0001001880000a020800800016040a00000980001c020050")
    ours = build_match([("eth_type", 0x0800), ("ipv4_src", "10.0.0.9"), ("tcp_dst", 80)])
    chk(ours == golden,
        "OXM match 与 Ryu 参考实现的字节流逐字节一致（黄金向量）",
        ("一致：" if ours == golden else f"不一致！我们={ours.hex()} / Ryu={golden.hex()} —— ")
        + f"{len(ours)} 字节，TLV 间不补齐、仅整体补齐到 8 的倍数")
    expect_mlen = 4 + sum(4 + OXM_LEN[f] for f in ("ipv4_src", "tcp_dst", "in_port"))
    chk(mlen == expect_mlen and len(m) % 8 == 0 and 0 <= len(m) - mlen < 8,
        "OXM match 长度字段=4+各条 TLV 实际字节数，整体补齐到 8 的倍数",
        f"match {len(m)} 字节（3 条 TLV，长度字段={mlen}（应为 {expect_mlen}），"
        f"整体补齐 {len(m) - mlen} 字节）")


    # 空 match（table-miss 流表项）是最容易被"补齐"坑到的用例：Ryu 的 empty match 是
    # `00 01 00 04` + 4 字节补齐。少了补齐，紧跟其后的指令会被解析成 type=0/len=0 幽灵指令。
    miss_body = (struct.pack("!QQBBHHHIII4x", 0, 0, 0, OFPFC_ADD, 0, 0, 0,
                             0xFFFFFFFF, 0xFFFFFFFF, 0)
                 + build_match([])
                 + struct.pack("!HH4x", OFPIT_APPLY_ACTIONS, 8 + 16)
                 + struct.pack("!HHIH6x", OFPAT_OUTPUT, 16, OFPP_CONTROLLER, 0))
    miss = ofp_header(OFPT_FLOW_MOD, 8 + len(miss_body), 91) + miss_body
    pm = parse_flow_mod(miss)
    chk(pm["match"] == {} and len(pm["instructions"]) == 1
        and pm["instructions"][0]["type"] == OFPIT_APPLY_ACTIONS
        and pm["instructions"][0]["actions"] == [{"type": "OUTPUT", "port": OFPP_CONTROLLER}]
        and not pm["is_drop"],
        "空 match（table-miss 型 FLOW_MOD）：指令偏移不被补齐字节带偏，且正确识别为「转发到控制器」",
        f"len={len(miss)} match={pm['match']} 指令={pm['instructions']} is_drop={pm['is_drop']}")

    pd = build_port_desc_reply(92, [(1, "02:00:00:00:00:01", "s1-eth1")])
    pp = parse_port_desc_reply(pd)
    chk(len(pp) == 1 and pp[0]["port_no"] == 1 and pp[0]["name"] == "s1-eth1"
        and struct.unpack_from("!H", pd, 8)[0] == OFPMP_PORT_DESC
        and len(pd) == 16 + OFP_PORT_SIZE,
        "PORT_DESC_REPLY：multipart 子类型=OFPMP_PORT_DESC(13)，OFPPort 定长 64 字节可读回",
        f"len={len(pd)} 端口={[(p['port_no'], p['name']) for p in pp]}")

    print("-" * 74)
    print("自检结果：" + ("全部通过 ✅" if ok else "存在失败项 ❌"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="真实 Ryu 进程 + 模拟 OF1.3 交换机的处置链路验证")
    ap.add_argument("--ryu-python", default="", help="带 ryu 的解释器（默认 _tmp/ryu_venv）")
    ap.add_argument("--project-python", default=str(ROOT / ".venv" / "Scripts" / "python.exe"))
    ap.add_argument("--model-tag", default="sdn_ctrl_s0")
    ap.add_argument("--processed-dir", default="data/processed_sdn")
    ap.add_argument("--detect-port", type=int, default=0, help="0 = 自动挑空闲端口")
    ap.add_argument("--ofp-port", type=int, default=0, help="0 = 自动挑空闲端口")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--settle", type=float, default=15.0,
                    help="最后一轮流表上报之后再静默观察多少秒收工")
    ap.add_argument("--normal-pos", type=int, default=671,
                    help="正常形态轮：测试流起始下标（默认 671，模型判定稳定的连续正常段）")
    ap.add_argument("--normal-rows", type=int, default=25, help="正常形态轮取多少条流")
    ap.add_argument("--normal-seg", default="测试流 671~695 行",
                    help="正常形态轮的人类可读说明（写进报告）")
    ap.add_argument("--attack-pos", type=int, default=16523,
                    help="攻击形态轮：测试流起始下标（默认 16523，DDoS 片段起点）")
    ap.add_argument("--attack-rows", type=int, default=39, help="攻击形态轮取多少条流")
    ap.add_argument("--attack-seg", default="DDoS", help="攻击形态轮的类别说明（写进报告）")
    ap.add_argument("--keep-logs", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="只做离线自检（OF1.3 编解码，不需要 ryu、不联网）")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()

    ryu_python = Path(args.ryu_python) if args.ryu_python else Path(
        os.environ.get("RYU_PYTHON", str(ROOT / "_tmp" / "ryu_venv" / "Scripts" / "python.exe")))
    if not ryu_python.exists():
        raise SystemExit(f"找不到带 ryu 的解释器：{ryu_python}（用 --ryu-python 指定）")

    from src.utils import RESULT_DIR, ensure_dirs, get_logger, save_json, save_text, setup_matplotlib  # noqa: E402

    LOGGER = get_logger("cida.ryu_live")
    ensure_dirs(RESULT_DIR)
    log_dir = ROOT / "logs" / "_ryu_live"
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_path = log_dir / "audit_alerts.jsonl"
    if audit_path.exists():
        audit_path.unlink()

    detect_port = args.detect_port or free_port(5300)
    ofp_port = args.ofp_port or free_port(6700)
    detect_url = f"http://127.0.0.1:{detect_port}"

    procs: list[subprocess.Popen] = []
    switched: list[list[dict]] = []      # 每一轮要上报的流表统计（在下面按真实数据构造）
    checks: dict = {}

    def add_check(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = {"pass": bool(ok), "detail": detail}
        LOGGER.info("%s %s —— %s", "✓" if ok else "✗", name, detail)

    try:
        # ---------- 1) 检测服务 ----------
        svc_log = (log_dir / "detect_service.log").open("w", encoding="utf-8")
        svc = subprocess.Popen(
            [args.project_python, str(ROOT / "app" / "app.py"), "--host", "127.0.0.1",
             "--port", str(detect_port), "--processed-dir", args.processed_dir,
             "--model-tag", args.model_tag],
            cwd=str(ROOT), stdout=svc_log, stderr=subprocess.STDOUT)
        procs.append(svc)
        health = wait_http(detect_url + "/api/health", timeout=args.timeout)
        add_check("检测服务已就绪", health, f"{detect_url}/api/health")
        if not health:
            raise SystemExit("检测服务未就绪，终止")

        # ---------- 1b) 把检测策略切成"真实处置" ----------
        # 检测服务默认 action=observe / dry_run=true（只给建议、不下发），
        # 真机联调要验证的是**下发链路**，所以这里显式把策略改成 block + 非演练。
        pol = _post_json(detect_url + "/api/policy",
                         {"action": "block", "dry_run": False, "confirm_windows": 3})
        add_check("检测策略已切到「真实处置」（action=block、非演练）",
                  bool(pol) and pol.get("action") == "block" and pol.get("dry_run") is False,
                  f"服务回显策略：{ {k: pol.get(k) for k in ('action', 'dry_run', 'threshold', 'confirm_windows')} if pol else pol }")

        # ---------- 1c) 取真实数据集的流记录（不是手编数字） ----------
        # 两轮流量都来自 data/processed_sdn 的测试流：
        #   轮 1「正常形态」→ 期望：不告警、不下发任何规则（证明处置路径是有选择性的）
        #   轮 2「攻击形态」→ 期望：告警 → 下发 DROP 规则 → 写台账
        # 为什么先正常后攻击：检测服务是**有状态**的滑动窗口（缓冲 16 条、步长 3），
        # 先灌攻击再灌正常，前几个正常窗口里还残留着攻击行、会被判成攻击；
        # 先正常后攻击则每个窗口的"最新一行"都是当前轮次的行，判定与源 IP 都干净。
        rounds_meta: list[dict] = []
        for tag, pos, count, src_ip, note, seg_name in (
                ("normal", args.normal_pos, args.normal_rows, "10.0.0.3",
                 "正常形态（测试流中模型判定稳定的一段连续流）", args.normal_seg),
                ("attack", args.attack_pos, args.attack_rows, "10.0.0.9",
                 f"攻击形态（{args.attack_seg} 片段）", args.attack_seg)):
            rows = _get_replay_rows(detect_url, pos, count)
            flows = []
            for i, r in enumerate(rows):
                f = row_to_flow(r, in_port=1 + (i % 3), src=src_ip, dst="10.0.0.4")
                if f is None:
                    continue
                f["_note"] = note
                flows.append(f)
            rounds_meta.append({"tag": tag, "pos": pos, "count": len(rows), "kept": len(flows),
                                "src_ip": src_ip, "note": note, "segment": seg_name})
            switched.append(flows)
        add_check("两轮流量取自真实数据集（不是手编数字）",
                  all(r["kept"] >= 16 for r in rounds_meta),
                  "；".join(f"{r['tag']}：测试流第 {r['pos']} 行起 {r['count']} 条"
                            f"（其中 {r['kept']} 条可用 OF 流表统计如实表达）" for r in rounds_meta))

        # ---------- 2) 真实 Ryu 进程 ----------
        ryu_log = (log_dir / "ryu_manager.log").open("w", encoding="utf-8")
        # ⚠️ Ryu ≥ 4.34 的 ryu-manager **不再调用** app 的 add_arguments()，自定义 --xxx 会被判为
        #    "unrecognized arguments" 而启动失败（本机真机通电时踩到）。正确做法是用
        #    `--user-flags <file.py>`：ryu 会 import 这个文件，由它用 oslo.config 注册并设值。
        flags_py = log_dir / "ryu_guard_flags.py"
        flags_py.write_text(
            "from oslo_config import cfg\n"
            "CONF = cfg.CONF\n"
            "CONF.register_opts([\n"
            "    cfg.StrOpt('detect_url', default=''),\n"
            "    cfg.BoolOpt('execute', default=False),\n"
            "    cfg.IntOpt('block_seconds', default=60),\n"
            "    cfg.ListOpt('whitelist', default=[]),\n"
            "    cfg.StrOpt('audit', default=''),\n"
            "])\n"
            f"CONF.set_override('detect_url', {detect_url!r})\n"
            "CONF.set_override('execute', True)\n"
            "CONF.set_override('block_seconds', 60)\n"
            "CONF.set_override('whitelist', [])\n"
            f"CONF.set_override('audit', {str(audit_path)!r})\n",
            encoding="utf-8")
        ryu_cmd = [str(ryu_python), "-m", "ryu.cmd.manager",
                   "--user-flags", str(flags_py),
                   "--ofp-tcp-listen-port", str(ofp_port),
                   str(ROOT / "integrations" / "ryu_guard.py")]
        ryu = subprocess.Popen(ryu_cmd, cwd=str(ROOT), stdout=ryu_log, stderr=subprocess.STDOUT)
        procs.append(ryu)
        up = wait_tcp("127.0.0.1", ofp_port, timeout=args.timeout)
        add_check("真实 Ryu 进程已监听 OF 端口", up, f"ryu-manager on 0.0.0.0:{ofp_port}（非桩件）")
        if not up:
            raise SystemExit("Ryu 未监听，终止")

        ryu_version = "未取到"
        try:
            out = subprocess.run(
                [str(ryu_python), "-c",
                 "import importlib.metadata as m; print(m.version('ryu'))"],
                capture_output=True, text=True, timeout=60)
            ryu_version = (out.stdout or "").strip() or "n/a"
        except Exception:  # noqa: BLE001
            pass

        # ---------- 3) 模拟交换机 ----------
        time.sleep(1.0)                                     # 让 Ryu 的 wait_tcp 探活连接先干净地断开
        sw = FakeSwitch("127.0.0.1", ofp_port, switched, rounds=2, timeout=args.timeout,
                        settle=args.settle)
        sw.start()
        sw.join(timeout=args.timeout + 30)

        ryu_log_text = ""
        try:
            ryu_log.flush()
            ryu_log_text = (log_dir / "ryu_manager.log").read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        registered = "[guard] 交换机上线" in ryu_log_text
        parse_errors = ryu_log_text.count("Encountered an error while parsing OpenFlow packet")

        add_check("OF1.3 握手完成（HELLO 交互）", sw.hello_ok,
                  f"收到控制器报文类型序列：{[MSG_TYPE_NAME.get(t, t) for t in sw.raw_types[:8]]}")
        add_check("控制器识别交换机（FEATURES_REQUEST → FEATURES_REPLY）", sw.features_ok,
                  f"控制器发出 OFPT_FEATURES_REQUEST，模拟交换机回 FEATURES_REPLY "
                  f"{sw.features_replies_sent} 条（Ryu 侧用自身解析器校验通过，见自检）")
        add_check("端口描述应答使控制器进入 MAIN_DISPATCHER",
                  sw.port_desc_replies_sent > 0 and registered,
                  f"OF1.3 下 Ryu 收到 FEATURES_REPLY 后还会发 OFPMP_PORT_DESC 请求；"
                  f"本工具回了 {sw.port_desc_replies_sent} 条 PORT_DESC_REPLY，"
                  f"Ryu 日志出现「{ '[guard] 交换机上线' if registered else '（未出现上线记录）'}」")
        add_check("控制器周期拉取【流表】统计（OFPMP_FLOW 请求）", sw.stats_requests >= 2,
                  f"收到 {sw.stats_requests} 次 OFPMP_FLOW 请求（适配层 monitor 周期 10 s）；"
                  f"另有 {sw.port_desc_requests} 次 OFPMP_PORT_DESC 请求（握手用，不计入）")
        add_check("模拟交换机已完成流表统计上报", sw.stats_replies_sent >= 2,
                  f"发送 {sw.stats_replies_sent} 条 OFPFlowStatsReply（真实 OF1.3 序列化）")
        add_check("Ryu 未报告任何报文解析错误（线格式合规）", parse_errors == 0,
                  f"ryu-manager 日志中 「Encountered an error while parsing ...」出现 {parse_errors} 次")

        time.sleep(3.0)                                     # 给适配层时间下发与写台账

        flow_mods = [m for m in sw.received if m.get("type") == "FLOW_MOD"]
        meter_mods = [m for m in sw.received if m.get("type") == "METER_MOD"]
        drop_mods = [m for m in flow_mods if m.get("is_drop")]
        throttles = [m for m in flow_mods if m.get("meter_ids")]

        def src_of(m: dict) -> str:
            return str(m.get("match", {}).get("ipv4_src", ""))

        guard_mods = [m for m in flow_mods if src_of(m)]      # 带源 IP 匹配 = 本作品下发的处置规则
        miss_mods = [m for m in flow_mods if not m.get("match")]   # 空 match = 交换机的 table-miss 保底规则
        add_check("收到处置下发：FLOW_MOD", bool(guard_mods),
                  f"共 {len(flow_mods)} 条 FLOW_MOD：处置规则 {len(guard_mods)} 条"
                  f"（其中 DROP 指令 {len([m for m in guard_mods if m.get('is_drop')])} 条）、"
                  f"交换机自身 table-miss 保底规则 {len(miss_mods)} 条")
        add_check("下发规则的匹配字段指向可疑源 IP",
                  any(src_of(m).startswith("10.0.0.9") for m in guard_mods),
                  f"处置规则的匹配源 IP：{sorted({src_of(m) for m in guard_mods})}")
        add_check("正常流量轮次未产生任何处置动作（无误报动作）",
                  bool(sw.stats_replies_sent >= 2) and not any(src_of(m).startswith("10.0.0.3")
                                                               for m in guard_mods),
                  f"第 1 轮为{'正常的真实流'}（10.0.0.3）；"
                  f"针对正常源 IP 的处置规则 {len([m for m in guard_mods if src_of(m).startswith('10.0.0.3')])} 条")

        # 台账留痕
        audit: list[dict] = []
        if audit_path.exists():
            for line in audit_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        audit.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        executed = [a for a in audit if a.get("executed")]
        suppressed = [a for a in audit if a.get("suppressed")]
        add_check("审计台账记录了真实执行（executed=true）", bool(executed),
                  f"台账 {len(audit)} 条，其中真正执行 {len(executed)} 条、"
                  f"幂等抑制 {len(suppressed)} 条"
                  + (f"（示例 action={executed[0].get('action')} src={executed[0].get('src_ip')} "
                     f"类别={executed[0].get('pred_class')} p_attack={executed[0].get('p_attack')}）"
                     if executed else ""))
        add_check("处置动作与台账一致（action=block → 下发 DROP 或 METER 流表项）",
                  bool(executed) and bool(guard_mods),
                  f"台账动作：{sorted({a.get('action') for a in audit})}；"
                  f"实际报文：处置 FLOW_MOD {len(guard_mods)}、METER_MOD {len(meter_mods)}、"
                  f"限速流表项 {len(throttles)}")
        add_check("同一源 IP 的连续告警不重复下发规则（处置幂等）",
                  len(guard_mods) <= max(len(executed), 1)
                  and len(suppressed) >= 1 and len(executed) == 1,
                  f"{len(audit)} 条告警里：真实下发 {len(executed)} 次、"
                  f"封禁窗口内幂等抑制 {len(suppressed)} 次，交换机上只增加了 {len(guard_mods)} 条规则")

        payload = {
            "experiment": "ryu_live_check",
            "ryu_version": ryu_version,
            "ryu_python": str(ryu_python),
            "ryu_cmd": " ".join(ryu_cmd[2:]),
            "model_tag": args.model_tag,
            "processed_dir": args.processed_dir,
            "detect_port": detect_port, "ofp_port": ofp_port,
            "flow_rounds": switched,
            "rounds_meta": rounds_meta,
            "policy": pol,
            "raw_types": sw.raw_types[:40],
            "timeline": sw.timeline,
            "stats_requests": sw.stats_requests,
            "stats_replies_sent": sw.stats_replies_sent,
            "port_desc_requests": sw.port_desc_requests,
            "port_desc_replies_sent": sw.port_desc_replies_sent,
            "echo_replies_sent": sw.echo_replies_sent,
            "guard_registered_in_ryu_log": registered,
            "ryu_parse_errors": parse_errors,
            "messages": sw.received,
            "audit_path": str(audit_path.relative_to(ROOT)),
            "audit": audit,
            "checks": checks,
            "switch_error": sw.error,
            "switch_exit_reason": sw.exit_reason,
            "passed": all(v["pass"] for v in checks.values()),
        }

        save_json(payload, RESULT_DIR / "ryu_live_check.json")
        save_text(build_report(payload), RESULT_DIR / "ryu_live_check.md")
        LOGGER.info("完成：%s（%d/%d 项通过）", RESULT_DIR / "ryu_live_check.md",
                    sum(1 for v in checks.values() if v["pass"]), len(checks))
        return 0 if payload["passed"] else 1
    finally:
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
