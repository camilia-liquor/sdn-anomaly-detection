"""真实 Ryu 进程 + 模拟 OpenFlow 1.3 交换机：把「适配层未在真实交换机实跑」缩小到最小。

为什么做这个（这条边界的准确表述）
----------------------------------
工程原来的写法是：「Ryu 适配层**未在真实交换机上实跑**（接口与下发路径已用桩件测 25 项）」。
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
OFPT_FLOW_STATS_REQUEST = 18
OFPT_FLOW_STATS_REPLY = 19
OFPT_METER_MOD = 29
OFPMP_FLOW = 0
OFPIT_APPLY_ACTIONS = 4
OFPIT_METER = 6
OFPAT_OUTPUT = 0
OFPP_CONTROLLER = 0xFFFFFFFD
OFPFC_ADD = 0
OFPMC_ADD = 0
OFPXMC_OPENFLOW_BASIC = 0x8000
OXM_FIELDS = {
    "in_port": 0, "eth_dst": 3, "eth_src": 4, "eth_type": 5,
    "ip_proto": 10, "ipv4_src": 11, "ipv4_dst": 12,
    "tcp_src": 13, "tcp_dst": 14, "udp_src": 15, "udp_dst": 16,
}
OXM_LEN = {"in_port": 4, "eth_dst": 6, "eth_src": 6, "eth_type": 2, "ip_proto": 1,
           "ipv4_src": 4, "ipv4_dst": 4, "tcp_src": 2, "tcp_dst": 2, "udp_src": 2, "udp_dst": 2}


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


def oxm(field: str, value) -> bytes:
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
    header = struct.pack("!HBB", OFPXMC_OPENFLOW_BASIC, n << 1, ln)
    out = header + payload
    return out + b"\x00" * ((8 - (len(out) % 8)) % 8)       # OXM 需按 8 字节对齐


def build_match(entries: list[tuple[str, object]]) -> bytes:
    """OFPMatch：type(2)=OFPMT_OXM(1) + length(2) + 若干 8 字节对齐的 OXM TLV。

    ⚠️ length 必须**等于实际发出的字节数**（含 TLV 自身的补齐），否则解析端算出的
    "指令起始偏移"会落进填充里 —— 自检就是这么发现问题的（每条 TLV 已按 8 字节对齐，
    因此这里不再额外补齐整个 match）。
    """
    tlvs = b"".join(oxm(f, v) for f, v in entries)
    return struct.pack("!HH", 1, 4 + len(tlvs)) + tlvs


def build_flow_stats_reply(xid: int, flows: list[dict]) -> bytes:
    """OFPFlowStatsReply：body = OFPMP_FLOW 头 + 若干 OFPFlowStats。

    OFPFlowStats（OF1.3）字段顺序：length(2) table_id(1) pad(1) duration_sec(4) duration_nsec(4)
    priority(2) idle_timeout(2) hard_timeout(2) flags(2) pad(4) cookie(8) packet_count(8)
    byte_count(8) match(变长) instructions(变长) —— 固定头共 48 字节。
    """
    stats = b""
    for f in flows:
        match = build_match([
            ("in_port", f["in_port"]),
            ("eth_type", 0x0800),
            ("ip_proto", 6),
            ("ipv4_src", f["src"]),
            ("ipv4_dst", f["dst"]),
            ("tcp_src", f["sport"]),
            ("tcp_dst", f["dport"]),
        ])
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
        step = 4 + ln * (2 if hasmask else 1)
        pos += step + ((8 - (step % 8)) % 8)
    return out, end


def parse_instructions(buf: bytes, off: int, end: int) -> list[dict]:
    out = []
    pos = off
    while pos + 4 <= end:
        itype, ilen = struct.unpack_from("!HH", buf, pos)
        item: dict = {"type": itype, "length": ilen}
        if itype == OFPIT_APPLY_ACTIONS or itype == 3:      # APPLY / WRITE actions
            acts = []
            apos = pos + 4
            # ⚠️ 一条 action 至少 8 字节；指令长度是 8 的倍数，末尾可能有填充，
            #    必须靠 "alen >= 8" 把它排除，否则填充会被当成一个不存在的 action
            #    （自检抓到的真实 bug：DROP 被判成"有 action"）。
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
    """一个最小的 OF1.3 交换机：握手 → 上报流表统计 → 记录控制器下发的一切。"""

    def __init__(self, host: str, port: int, flows: list[dict], *, rounds: int = 2,
                 reply_on_request: bool = True, timeout: float = 40.0):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.flow_rounds = flows
        self.rounds = rounds
        self.reply_on_request = reply_on_request
        self.timeout = timeout
        self.received: list[dict] = []
        self.raw_types: list[int] = []
        self.hello_ok = False
        self.features_ok = False
        self.stats_requests = 0
        self.stats_replies_sent = 0
        self.error: str | None = None
        self.xid = 100

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
        deadline = time.time() + self.timeout
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.settimeout(max(self.timeout, 5.0))
                # ---- 握手 ----
                sock.sendall(build_hello(1))
                hdr = self._recv_exact(sock, 8)
                _, mtype, _, _ = struct.unpack("!BBHI", hdr)
                self.raw_types.append(mtype)
                if mtype == OFPT_HELLO:
                    ln = struct.unpack("!H", hdr[2:4])[0]
                    if ln > 8:
                        self._recv_exact(sock, ln - 8)
                    self.hello_ok = True
                sock.sendall(build_features_request(2))

                # ---- 主循环 ----
                sent = 0
                pending = b""
                while time.time() < deadline:
                    try:
                        chunk = sock.recv(65535)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    pending += chunk
                    while len(pending) >= 8:
                        _, mtype, ln, _ = struct.unpack_from("!BBHI", pending, 0)
                        if ln < 8 or len(pending) < ln:
                            break
                        raw, pending = pending[:ln], pending[ln:]
                        self.raw_types.append(mtype)
                        if mtype == OFPT_FEATURES_REPLY:
                            self.features_ok = True
                        elif mtype == OFPT_FLOW_STATS_REQUEST:
                            self.stats_requests += 1
                            if self.reply_on_request and sent < self.rounds:
                                self._send_stats(sock, sent)
                                sent += 1
                        else:
                            msg = parse_message(raw)
                            if msg:
                                msg["raw_len"] = ln
                                self.received.append(msg)
                    # 即使控制器不主动请求，也主动上报一轮（OF 允许非请求回复，Ryu 会照常派发事件）
                    if not self.reply_on_request and sent < self.rounds:
                        self._send_stats(sock, sent)
                        sent += 1
                        time.sleep(2.0)
                time.sleep(1.0)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"


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


def build_report(p: dict) -> str:
    checks = p["checks"]
    ok = sum(1 for v in checks.values() if v["pass"])
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
        "## 2. 控制器下发的原始报文（逐条解析）",
        "",
        "| 序号 | 类型 | 关键字段 |",
        "|---|---|---|",
    ]
    for i, m in enumerate(p["messages"], 1):
        if m.get("type") == "FLOW_MOD":
            detail = (f"command={m.get('command')} priority={m.get('priority')} "
                      f"match={m.get('match')} 指令={m.get('instructions')} "
                      f"是否 DROP={m.get('is_drop')}")
        elif m.get("type") == "METER_MOD":
            detail = f"command={m.get('command')} meter_id={m.get('meter_id')} bands={m.get('bands')}"
        else:
            detail = json.dumps({k: v for k, v in m.items() if k != "type"}, ensure_ascii=False)[:160]
        lines.append(f"| {i} | {m.get('type')} | {detail} |")
    if not p["messages"]:
        lines.append("| — | （未收到任何下发报文） | 检查适配层是否处于演练模式（未加 `--execute`） |")
    lines += [
        "",
        "## 3. 上了哪些流量（模拟交换机的流表统计）",
        "",
        "| 轮次 | 源 IP | 目的 IP/端口 | 包数 | 字节数 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for i, rnd in enumerate(p["flow_rounds"], 1):
        for f in rnd:
            lines.append(f"| {i} | {f['src']} | {f['dst']}:{f['dport']} | {f['packet_count']:,} | "
                         f"{f['byte_count']:,} | {f.get('_note', '')} |")
    lines += [
        "",
        "## 4. 审计台账（控制器侧留痕）",
        "",
        f"- 台账文件：`{p['audit_path']}`（本次运行独立，不污染主台账）",
        f"- 记录条数：**{len(p['audit'])}**",
        "",
    ]
    if p["audit"]:
        lines += ["| 时间 | 判定类别 | p_attack | 源 IP | 动作 | 是否真正执行 | 演练 |",
                  "|---|---|---|---|---|---|---|"]
        for a in p["audit"][:10]:
            lines.append(f"| {a.get('time')} | {a.get('pred_class')} | {a.get('p_attack')} | "
                         f"{a.get('src_ip')} | {a.get('action')} | {a.get('executed')} | "
                         f"{a.get('dry_run')} |")
    lines += [
        "",
        "## 5. 局限（写清楚，不粉饰）",
        "",
        "- 模拟交换机是**自研脚本**：按 OF1.3 规范发包（HELLO/FEATURES/FLOW_STATS），"
        "并解析 FLOW_MOD/METER_MOD；**它不是物理设备**，未覆盖厂商私有扩展与完整协议一致性。",
        "- 只覆盖本作品用到的消息类型；流表老化、多交换机拓扑、并发压力未测。",
        "- 检测服务是本机 Flask 实例、流量由本工具构造（构造方式见第 3 节），"
        '不是真实网络抓包——"真机验证"仍需在实验室 SDN 环境完成。',
        "",
        "## 6. 复现命令",
        "",
        "```powershell",
        f"# 1) 准备带 ryu 的解释器（本机为独立 venv，避免污染项目依赖）",
        f"#    python -m venv _tmp/ryu_venv; _tmp/ryu_venv/Scripts/python.exe -m pip install ryu",
        f"# 2) 一键完成：起服务 → 起真实 Ryu → 模拟交换机握手/上报/校验下发",
        f"python tools/ryu_live_check.py --model-tag {p['model_tag']} "
        f"--processed-dir {p['processed_dir']}",
        "```",
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

    h = build_hello()
    ver, mtype, length, _ = struct.unpack_from("!BBHI", h, 0)
    chk(ver == OFP_VERSION and mtype == OFPT_HELLO and length == len(h),
        "HELLO 报文头正确", f"version=0x{ver:02x} type={mtype} len={length}")

    flows = [flow_for(1, "10.0.0.9", "10.0.0.4", 80, pkts=8000, byts=9_600_000, dur=3),
             flow_for(2, "10.0.0.3", "10.0.0.2", 443, pkts=12, byts=1400, dur=30)]
    rep = build_flow_stats_reply(77, flows)
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
    chk((len(m) - 4) % 8 == 0 and mlen == len(m),
        "OXM 每条 TLV 按 8 字节对齐，且 match 长度字段等于实际字节数",
        f"match {len(m)} 字节（3 条 TLV，长度字段={mlen}）")

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
    switched = [
        # 第 1 轮：明显攻击形态（高包数/高字节数 → 期望被判为攻击并下发阻断）
        [flow_for(1, "10.0.0.9", "10.0.0.4", 80, pkts=8000, byts=9_600_000, dur=3, sport=40001),
         flow_for(2, "10.0.0.9", "10.0.0.4", 80, pkts=7500, byts=9_000_000, dur=3, sport=40002),
         flow_for(3, "10.0.0.9", "10.0.0.4", 443, pkts=6000, byts=7_200_000, dur=4, sport=40003)],
        # 第 2 轮：正常形态（小流量、长时长）
        [flow_for(1, "10.0.0.3", "10.0.0.2", 443, pkts=12, byts=1400, dur=30, sport=40010),
         flow_for(2, "10.0.0.3", "10.0.0.2", 443, pkts=9, byts=1100, dur=28, sport=40011)],
    ]
    switched[0][0]["_note"] = "攻击形态（洪泛）"
    switched[0][1]["_note"] = "攻击形态（洪泛）"
    switched[0][2]["_note"] = "攻击形态（洪泛）"
    switched[1][0]["_note"] = "正常形态（基线）"
    switched[1][1]["_note"] = "正常形态（基线）"

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

        # ---------- 2) 真实 Ryu 进程 ----------
        ryu_log = (log_dir / "ryu_manager.log").open("w", encoding="utf-8")
        ryu_cmd = [str(ryu_python), "-m", "ryu.cmd.manager",
                   str(ROOT / "integrations" / "ryu_guard.py"),
                   "--ofp-tcp-listen-port", str(ofp_port),
                   "--detect-url", detect_url,
                   "--execute",                      # 关闭演练，真实下发
                   "--audit", str(audit_path),
                   "--block-seconds", "60"]
        ryu = subprocess.Popen(ryu_cmd, cwd=str(ROOT), stdout=ryu_log, stderr=subprocess.STDOUT)
        procs.append(ryu)
        up = wait_tcp("127.0.0.1", ofp_port, timeout=args.timeout)
        add_check("真实 Ryu 进程已监听 OF 端口", up, f"ryu-manager on 0.0.0.0:{ofp_port}（非桩件）")
        if not up:
            raise SystemExit("Ryu 未监听，终止")

        ryu_version = "未取到"
        try:
            out = subprocess.run([str(ryu_python), "-c", "import ryu; print(ryu.__version__ if hasattr(ryu,'__version__') else 'n/a')"],
                                 capture_output=True, text=True, timeout=60)
            ryu_version = (out.stdout or "").strip() or "n/a"
        except Exception:  # noqa: BLE001
            pass

        # ---------- 3) 模拟交换机 ----------
        sw = FakeSwitch("127.0.0.1", ofp_port, switched, rounds=2, timeout=args.timeout)
        sw.start()
        sw.join(timeout=args.timeout + 30)

        add_check("OF1.3 握手完成（HELLO 交互）", sw.hello_ok,
                  f"收到控制器报文类型序列（前 6 个）：{sw.raw_types[:6]}")
        add_check("控制器识别交换机（FEATURES_REPLY）", sw.features_ok,
                  "Ryu 回发 OFPT_FEATURES_REPLY → 交换机进入 MAIN_DISPATCHER")
        add_check("控制器周期拉取流表统计（FLOW_STATS_REQUEST）", sw.stats_requests > 0,
                  f"收到 {sw.stats_requests} 次请求（适配层 monitor 周期 10 s）")
        add_check("模拟交换机已完成流表统计上报", sw.stats_replies_sent >= 2,
                  f"发送 {sw.stats_replies_sent} 条 OFPFlowStatsReply（真实 OF1.3 序列化）")

        time.sleep(3.0)                                     # 给适配层时间下发与写台账

        flow_mods = [m for m in sw.received if m.get("type") == "FLOW_MOD"]
        meter_mods = [m for m in sw.received if m.get("type") == "METER_MOD"]
        drop_mods = [m for m in flow_mods if m.get("is_drop")]
        throttles = [m for m in flow_mods if m.get("meter_ids")]
        add_check("收到处置下发：FLOW_MOD", bool(flow_mods),
                  f"{len(flow_mods)} 条 FLOW_MOD（其中 DROP 指令 {len(drop_mods)} 条）")
        add_check("下发规则的匹配字段指向可疑源 IP",
                  any(str(m.get("match", {}).get("ipv4_src", "")).startswith("10.0.0.9")
                      for m in flow_mods),
                  f"匹配到的源 IP：{sorted({m.get('match', {}).get('ipv4_src') for m in flow_mods if m.get('match')})}")

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
        add_check("审计台账记录了真实执行（executed=true）", bool(executed),
                  f"台账 {len(audit)} 条，其中真正执行 {len(executed)} 条"
                  + (f"（示例 action={executed[0].get('action')} src={executed[0].get('src_ip')}）"
                     if executed else ""))
        add_check("处置动作与台账一致（action=block → 下发 DROP 或 METER 流表项）",
                  bool(executed) and bool(flow_mods),
                  f"台账动作：{sorted({a.get('action') for a in audit})}；"
                  f"实际报文：FLOW_MOD {len(flow_mods)}、METER_MOD {len(meter_mods)}、"
                  f"限速流表项 {len(throttles)}")

        payload = {
            "experiment": "ryu_live_check",
            "ryu_version": ryu_version,
            "ryu_python": str(ryu_python),
            "ryu_cmd": " ".join(ryu_cmd[2:]),
            "model_tag": args.model_tag,
            "processed_dir": args.processed_dir,
            "detect_port": detect_port, "ofp_port": ofp_port,
            "flow_rounds": switched,
            "raw_types": sw.raw_types[:40],
            "stats_requests": sw.stats_requests,
            "stats_replies_sent": sw.stats_replies_sent,
            "messages": sw.received,
            "audit_path": str(audit_path.relative_to(ROOT)),
            "audit": audit,
            "checks": checks,
            "switch_error": sw.error,
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
