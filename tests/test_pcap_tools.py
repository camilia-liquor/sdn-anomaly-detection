"""`tools/pcap_to_flows.py` 的**往返一致性测试**（不依赖 pytest，退出码 0 = 全通过）。

为什么要有这个文件
------------------
"pcap → 流特征"这条链路最容易出的错，是**生成侧与解析侧各写一套口径**：
报文间隔算错一点、字节数差几字节、超时切分判反了，特征就会悄悄偏，
而模型照样会输出一个"看起来正常"的判定。这个文件用**往返（合成 → 解析 → 比对）**
把这个链路口径锁死：

1. **逐条比对 17 个特征**：`--write-sample` 生成 pcap 时把"打算编码进去的期望值"
   写进清单（manifest），解析回来后与清单**逐位**比对；
2. **与仿真源记录比对**：再拿回来跟 `data/raw/sdn_stream.csv` 的原始行比，
   看物理上能复现到什么程度（帧长是整数，仿真的 `byte_count` 是浮点，必然有量化）；
3. **边界与异常**：空 pcap、截断 pcap 不崩溃且给出清晰错误；大小端都能读；pcapng 能读；
   VLAN 能剥标签；非首片分片按策略跳过；空闲/硬超时切分行为正确；
4. **报文真的合法**：独立复算 IPv4 头校验和（不是靠生成侧自己说合法）。

容差怎么定的（**这是本文件最需要解释的一处**）
----------------------------------------------
* `packet_count`、`byte_count`、`ip_proto`、`tp_src`、`tp_dst`、`icmp_*`、4 个速率列：
  **必须逐位完全一致**（整数帧长 + 整数计数，没有浮点空间）；
* 时长：解析出来的时长 = `(packet_count-1) × 报文间隔`。生成时把报文间隔对齐到
  pcap 的时间戳网格（微秒精度 → 1000 ns 网格；纳秒精度 → 1 ns 网格），
  因此**解析值与清单里的期望值逐位一致**；
  与"仿真源记录里的 duration"相比会有量化误差，上界为
  `(packet_count-1) × 网格 / 2`（本测试按这个上界判，不拍脑袋给常数）。
  实测：微秒精度下最大 7.4 µs，纳秒精度下 ≤ 数 ns；
* `byte_count` 与仿真源记录相比：帧长只能是整数（14+20+20+载荷），
  而仿真的 `byte_count` 是浮点（例如 838.9411128275215），
  物理上不可能逐位复现 → 判据是 `|解析值 - 仿真值| ≤ 0.5 字节`，
  且 `round(解析值) == round(仿真值)` 必须成立。

用法：``python tests/test_pcap_tools.py``（可选 ``--flows 200``、``--json out.json``）
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOL_PATH = ROOT / "tools" / "pcap_to_flows.py"
META_PATH = ROOT / "data" / "processed_sdn" / "meta.json"

PASS, FAIL = "✓", "✗"
_results: list[tuple[bool, str, str]] = []
STATS: dict = {}


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  —— {detail}" if detail else ""))
    return bool(ok)


# ======================================================================================
#  工具加载与调用
# ======================================================================================
def _load_tool():
    """按文件路径加载被测工具。

    注意：**必须先注册进 sys.modules 再 exec_module**，否则 `@dataclass` 在解析
    类型注解时会去 `sys.modules[cls.__module__]` 找模块并拿到 None 而报
    `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    """
    spec = importlib.util.spec_from_file_location("pcap_to_flows_under_test", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _capture():
    """吞掉被测工具的打印（本文件只关心返回值与产物），同时保留文本供断言。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out


def run_tool(tool, argv: list[str]) -> tuple[int, str]:
    with _capture() as out:
        code = tool.main(argv)
    return code, out.getvalue()


def parse_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ======================================================================================
#  测试用最小报文/抓包构造器（**故意不复用被测工具的构造函数**，保证独立）
# ======================================================================================
def mk_frame(src: str = "10.1.1.1", dst: str = "10.1.1.2", sport: int = 12345,
             dport: int = 80, proto: int = 6, payload: int = 0, vlan: int | None = None,
             frag: int = 0x4000) -> bytes:
    """造一帧 Ethernet/(可选 VLAN)/IPv4/TCP(UDP) 报文（校验和故意留 0：解析器不校验）。"""
    s = bytes(int(x) for x in src.split("."))
    d = bytes(int(x) for x in dst.split("."))
    if proto == 6:
        l4 = struct.pack(">HHIIBBHHH", sport, dport, 1, 1, 5 << 4, 0x02, 8192, 0, 0)
    elif proto == 17:
        l4 = struct.pack(">HHHH", sport, dport, 8 + payload, 0)
    else:
        l4 = struct.pack(">BBHHH", 8, 0, 0, 1, 1)
    ip = struct.pack(">BBHHHBBH", 0x45, 0, 20 + len(l4) + payload, 1, frag, 64, proto, 0) + s + d
    eth = b"\x02\x00\x00\x00\x00\x02" + b"\x02\x00\x00\x00\x00\x01"
    if vlan is None:
        return eth + struct.pack(">H", 0x0800) + ip + l4 + b"\x00" * payload
    return (eth + struct.pack(">HHH", 0x8100, vlan, 0x0800) + ip + l4 + b"\x00" * payload)


def write_pcap(path: Path, packets: list[tuple[int, bytes]], *, order: str = "<",
               magic: int | None = None) -> Path:
    """写经典 pcap；magic 可指定（用于测微秒/纳秒，大端/小端由 order 决定）。

    注意 magic 的取值本身不随字节序变：微秒永远是 0xA1B2C3D4。
    小端文件里它的字节序是 d4 c3 b2 a1，大端文件里是 a1 b2 c3 d4 —— 这正是
    "两种 magic 都要认"的由来。
    """
    if magic is None:
        magic = 0xA1B2C3D4
    res = 1 if magic in (0xA1B23C4D, 0x4D3CB2A1) else 1000
    if magic == 0x4D3CB2A1:      # 允许调用方按"读回来的整数"写法传大端纳秒
        magic = 0xA1B23C4D
    buf = bytearray(struct.pack(order + "IHHiIII", magic, 2, 4, 0, 0, 65535, 1))
    for ts_ns, frame in packets:
        buf += struct.pack(order + "IIII", ts_ns // 1_000_000_000,
                           (ts_ns % 1_000_000_000) // res, len(frame), len(frame))
        buf += frame
    path.write_bytes(bytes(buf))
    return path


def write_pcapng(path: Path, packets: list[tuple[int, bytes]], *, order: str = "<",
                 tsresol: int = 6, snap: int = 65535) -> Path:
    """手写 pcapng（SHB + IDB + 若干 EPB），**与被测工具的读取实现完全独立**。"""
    def pad(b: bytes) -> bytes:
        return b + b"\x00" * ((4 - len(b) % 4) % 4)

    bom = 0x1A2B3C4D if order == "<" else 0x4D3C2B1A

    def block(btype: int, body: bytes) -> bytes:
        total = 12 + len(body)
        return (struct.pack(order + "II", btype, total) + body
                + struct.pack(order + "I", total))

    # SHB：块类型恒为 0A 0D 0D 0A（与字节序无关），正文 = 字节序魔数(4)+版本(2+2)+段长度(8)
    out = bytearray(block(0x0A0D0D0A, struct.pack(order + "IHHq", bom, 1, 0, -1)))
    # IDB：linktype(2) + reserved(2) + snaplen(4) + 选项 if_tsresol(code=9, len=1, value)
    out += block(0x00000001, struct.pack(order + "HHI", 1, 0, snap)
                 + pad(struct.pack(order + "HHB", 9, 1, tsresol)))
    for ts_ns, frame in packets:
        # 时间戳是"以 if_tsresol 为单位的整数刻度"，64 位；微秒精度下要先从 ns 换成 µs
        ticks = ts_ns // (10 ** (9 - tsresol)) if tsresol <= 9 else ts_ns * (10 ** (tsresol - 9))
        epb = (struct.pack(order + "IIIII", 0, (ticks >> 32) & 0xFFFFFFFF, ticks & 0xFFFFFFFF,
                           len(frame), len(frame)) + pad(frame))
        out += block(0x00000006, epb)
    path.write_bytes(bytes(out))
    return path


# ======================================================================================
#  ① 往返一致性：合成 → 解析 → 与清单逐位比对
# ======================================================================================
def test_roundtrip(tool, tmp: Path, n_flows: int) -> dict:
    pcap = tmp / "sample.pcap"
    man_path = tmp / "manifest.json"
    csv_path = tmp / "parsed.csv"
    code, _ = run_tool(tool, ["--write-sample", str(pcap), "--flows", str(n_flows),
                              "--manifest", str(man_path)])
    check(code == 0 and pcap.exists(), "--write-sample 生成合法 pcap",
          f"返回码 {code}，文件 {pcap.stat().st_size if pcap.exists() else 0} 字节")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    check(pcap.stat().st_size == man["pcap_bytes"], "pcap 实际字节数与清单记录一致",
          f"{pcap.stat().st_size} vs {man['pcap_bytes']}")
    code, _ = run_tool(tool, ["--input", str(pcap), "--output", str(csv_path)])
    check(code == 0 and csv_path.exists(), "解析器读回自己生成的 pcap（退出码 0）", f"返回码 {code}")
    rows = parse_csv(csv_path)
    exp = man["flows"]
    check(len(rows) == len(exp), "流的条数一致（没有额外合并/拆分）",
          f"生成 {len(exp)} 条 → 解析回 {len(rows)} 条")
    if len(rows) != len(exp):
        return {"aborted": True}
    # ---- 逐特征比对 ----
    feats = man["feature_names"]
    exact = {f: 0 for f in feats}
    max_rel = {f: 0.0 for f in feats}
    for got, want in zip(rows, exp):
        for f in feats:
            g, w = float(got[f]), float(want["features"][f])
            if g == w:
                exact[f] += 1
            else:
                max_rel[f] = max(max_rel[f], abs(g - w) / max(1e-300, abs(w)))
    n = len(rows)
    must_exact = [f for f in feats]
    bad = [f for f in must_exact if exact[f] != n]
    check(not bad, "17 个特征全部逐位一致（与生成清单比对）",
          f"完全一致 {min(exact.values())}/{n} 条" if not bad else f"不一致字段：{bad}")
    # flow_id / 上下文列
    fid_ok = sum(1 for g, w in zip(rows, exp) if g["flow_id"] == w["flow_id"])
    check(fid_ok == n, "flow_id 拼接方式与仿真采集脚本一致（str(ip_src)+str(port)+…）",
          f"{fid_ok}/{n}")
    check(all(g["label"] == "Unknown" for g in rows),
          "pcap 无真值 → label 列固定为 Unknown")
    # ---- 与仿真源记录比对 ----
    grid_ns = 1000 if man["time_resolution"] == "us" else 1
    pkt_bad = proto_bad = port_bad = 0
    bc_dev_max = 0.0
    bc_round_bad = 0
    dur_dev_max = 0
    dur_tol_max = 0
    for got, want in zip(rows, exp):
        s = want["source_row"]
        p_got, p_src = int(float(got["packet_count"])), int(float(s["packet_count"]))
        if p_got != p_src:
            pkt_bad += 1
        if int(float(got["ip_proto"])) != int(float(s["ip_proto"])):
            proto_bad += 1
        # ICMP 在仿真数据里的端口是仿真器随手填的（报文无端口）→ 单独说明，不计入比对
        if int(float(s["ip_proto"])) in (6, 17):
            if (int(float(got["tp_src"])) != int(float(s["tp_src"]))
                    or int(float(got["tp_dst"])) != int(float(s["tp_dst"]))):
                port_bad += 1
        bc_got, bc_src = float(got["byte_count"]), float(s["byte_count"])
        bc_dev_max = max(bc_dev_max, abs(bc_got - bc_src))
        if round(bc_got) != round(bc_src):
            bc_round_bad += 1
        d_got = float(got["flow_duration_sec"]) * 1e9 + float(got["flow_duration_nsec"])
        d_src = float(s["flow_duration_sec"]) * 1e9 + float(s["flow_duration_nsec"])
        dur_dev_max = max(dur_dev_max, abs(d_got - d_src))
        dur_tol_max = max(dur_tol_max, (p_src - 1) * grid_ns / 2 + grid_ns)
    check(pkt_bad == 0, "packet_count 与仿真源记录完全一致", f"{n - pkt_bad}/{n}")
    check(proto_bad == 0 and port_bad == 0,
          "ip_proto / tp_src / tp_dst 与仿真源记录完全一致（ICMP 端口除外，见下）",
          f"协议不一致 {proto_bad}，TCP/UDP 端口不一致 {port_bad}")
    check(bc_round_bad == 0 and bc_dev_max <= 0.5 + 1e-9,
          "byte_count 与仿真源记录一致（帧长只能取整，容差 ≤ 0.5 字节）",
          f"最大偏差 {bc_dev_max:.3f} 字节，取整后一致 {n - bc_round_bad}/{n}")
    check(dur_dev_max <= dur_tol_max,
          "时长与仿真源记录在网格量化容差内一致（容差 =(packet_count-1)×网格/2）",
          f"最大偏差 {dur_dev_max:.0f} ns，容差上界 {dur_tol_max:.0f} ns")
    icmp_rows = [w for w in exp if int(float(w["source_row"]["ip_proto"])) == 1]
    if icmp_rows:
        check(all(int(float(r["tp_src"])) == 0 and int(float(r["tp_dst"])) == 0
                  for r, w in zip(rows, exp)
                  if int(float(w["source_row"]["ip_proto"])) == 1),
              f"ICMP 流按采集口径记 tp_src=tp_dst=0（{len(icmp_rows)} 条；"
              f"仿真数据里的 ICMP 端口是仿真器随手填的，报文里不存在）")
    protos = sorted({int(float(r["ip_proto"])) for r in rows})
    check(len(protos) >= 2, f"样本覆盖多种协议：{protos}")
    return {
        "n_flows": n, "n_packets": man["n_packets"], "pcap_bytes": man["pcap_bytes"],
        "time_resolution": man["time_resolution"], "byte_order": man["byte_order"],
        "class_counts": man["class_counts"],
        "features_exact": {f: exact[f] for f in feats},
        "features_exact_all": int(min(exact.values())), "features_total": n,
        "byte_count_max_abs_dev_vs_sim": bc_dev_max,
        "duration_max_abs_dev_ns_vs_sim": dur_dev_max,
        "duration_tolerance_ns": dur_tol_max,
        "flow_id_match": fid_ok,
        "protos": protos,
    }


# ======================================================================================
#  ② 超时切分 / VLAN / 分片策略
# ======================================================================================
def _flows_from(tool, path: Path, extra: list[str] | None = None) -> list[dict]:
    out = path.with_suffix(".csv")
    code, _ = run_tool(tool, ["--input", str(path), "--output", str(out)] + (extra or []))
    assert code == 0, f"解析失败，返回码 {code}"
    return parse_csv(out)


def test_timeouts(tool, tmp: Path) -> None:
    t0 = 1_700_000_000 * 10 ** 9
    # 同一五元组：0s、1s 两条（间隔 1s < 15s）→ 再在 30s 来一条（间隔 > 15s）→ 应切成 2 条流
    p1 = tmp / "idle.pcap"
    write_pcap(p1, [(t0, mk_frame()), (t0 + 1_000_000_000, mk_frame()),
                    (t0 + 30_000_000_000, mk_frame())])
    rows = _flows_from(tool, p1)
    ok = len(rows) == 2 and int(float(rows[0]["packet_count"])) == 2 and int(float(rows[1]["packet_count"])) == 1
    check(ok, "空闲超时：间隔 1s 归同一流、间隔 30s(>15s) 切成新流",
          f"切出 {len(rows)} 条流，报文数 {[r['packet_count'] for r in rows]}")
    if len(rows) == 2:
        check(int(float(rows[0]["flow_duration_sec"])) == 1
              and int(float(rows[0]["flow_duration_nsec"])) == 0,
              "切出的第一条流时长为 1.000000000s（首末报文跨度）",
              f"sec={rows[0]['flow_duration_sec']} nsec={rows[0]['flow_duration_nsec']}")
        check(int(float(rows[1]["packet_count"])) == 1
              and float(rows[1]["flow_duration_sec"]) == 0,
              "只有 1 个报文的流时长为 0（报文跨度为零）",
              f"sec={rows[1]['flow_duration_sec']} nsec={rows[1]['flow_duration_nsec']}")
    # 间隔 14s < 15s → 不应切分
    p2 = tmp / "no_idle.pcap"
    write_pcap(p2, [(t0, mk_frame()), (t0 + 14_000_000_000, mk_frame())])
    rows2 = _flows_from(tool, p2)
    check(len(rows2) == 1 and int(float(rows2[0]["packet_count"])) == 2,
          "空闲超时边界：间隔 14s(<15s) 不切分", f"切出 {len(rows2)} 条流")
    # 硬超时：每 1s 一个报文，持续 130s（无空闲间隔）→ 120s 处强制切分 → 2 条
    p3 = tmp / "hard.pcap"
    write_pcap(p3, [(t0 + i * 1_000_000_000, mk_frame()) for i in range(130)])
    rows3 = _flows_from(tool, p3, ["--strict"])
    counts = [int(float(r["packet_count"])) for r in rows3]
    check(len(rows3) == 2 and counts[0] == 120 and sum(counts) == 130,
          "硬超时：报文连续但跨过 120s 仍强制切分（等价流表项到期）",
          f"切出 {len(rows3)} 条流，报文数 {counts}")
    # 自定义硬超时：--hard-timeout 10 时，同样 130 个报文应切成 13 条
    rows4 = _flows_from(tool, p3, ["--hard-timeout", "10"])
    check(len(rows4) == 13, "--hard-timeout 可调：改成 10s 后 130 个报文切成 13 条流",
          f"实际 {len(rows4)} 条")


def test_vlan_and_fragment(tool, tmp: Path) -> None:
    t0 = 1_700_000_000 * 10 ** 9
    p = tmp / "vlan.pcap"
    write_pcap(p, [(t0, mk_frame(vlan=100)), (t0 + 1_000_000_000, mk_frame(vlan=100))])
    rows = _flows_from(tool, p)
    ok = (len(rows) == 1 and int(float(rows[0]["packet_count"])) == 2
          and rows[0]["ip_src"] == "10.1.1.1" and rows[0]["ip_dst"] == "10.1.1.2"
          and int(float(rows[0]["tp_dst"])) == 80)
    check(ok, "802.1Q VLAN 标签被正确剥离（ethertype 0x8100）",
          "" if ok else f"解析结果 {rows[:1]}")
    # QinQ：两层标签
    eth = b"\x02\x00\x00\x00\x00\x02" + b"\x02\x00\x00\x00\x00\x01"
    ip = struct.pack(">BBHHHBBH", 0x45, 0, 20 + 4 + 20, 1, 0x4000, 64, 6, 0) + b"\x0a\x01\x01\x01" + b"\x0a\x01\x01\x02"
    l4 = struct.pack(">HHIIBBHHH", 9, 443, 1, 1, 5 << 4, 0x02, 8192, 0, 0)
    qinq = eth + struct.pack(">HHHH", 0x88A8, 10, 0x8100, 20) + struct.pack(">H", 0x0800) + ip + l4
    p2 = tmp / "qinq.pcap"
    write_pcap(p2, [(t0, qinq)])
    rows2 = _flows_from(tool, p2)
    ok2 = len(rows2) == 1 and int(float(rows2[0]["tp_src"])) == 9 and int(float(rows2[0]["tp_dst"])) == 443
    check(ok2, "QinQ 双层 VLAN 标签（0x88A8 + 0x8100）也能解析")
    # 分片：首片（offset=0）算正常流；非首片（offset>0）跳过并计数
    p3 = tmp / "frag.pcap"
    write_pcap(p3, [(t0, mk_frame(frag=0x2000)),                 # MF=1, offset=0 → 首片
                    (t0 + 1_000_000_000, mk_frame(frag=0x0001)),  # offset=8 → 非首片，跳过
                    (t0 + 2_000_000_000, mk_frame(frag=0x4000))])
    out = tmp / "frag.csv"
    code, text = run_tool(tool, ["--input", str(p3), "--output", str(out), "--json", str(tmp / "frag.json")])
    rows3 = parse_csv(out)
    j = json.loads((tmp / "frag.json").read_text(encoding="utf-8"))
    check(code == 0 and len(rows3) == 1 and int(float(rows3[0]["packet_count"])) == 2,
          "分片策略：首片计入本流，非首片跳过（它没有 L4 头，不匹配按端口建的流表项）",
          f"流内报文数 {rows3[0]['packet_count'] if rows3 else '无'}")
    check(j["parse_stats"].get("fragment_nonfirst_skipped", 0) == 1,
          "跳过的非首片被计数（不静默丢包）",
          f"fragment_nonfirst_skipped={j['parse_stats'].get('fragment_nonfirst_skipped')}")
    check(j["parse_stats"].get("packets_read") == 3 and j["parse_stats"].get("packets_parsed") == 2,
          "统计口径自洽：读 3 个报文、解析 2 个（1 个是分片非首片）",
          f"read={j['parse_stats'].get('packets_read')} parsed={j['parse_stats'].get('packets_parsed')}")


# ======================================================================================
#  ③ pcap / pcapng / 大小端 / 异常输入
# ======================================================================================
def test_formats(tool, tmp: Path) -> None:
    t0 = 1_700_000_000 * 10 ** 9
    pkts = [(t0, mk_frame()), (t0 + 1_000_000_000, mk_frame()), (t0 + 2_000_000_000, mk_frame())]
    base = _flows_from(tool, write_pcap(tmp / "a.pcap", pkts))
    key = lambda rows: [(r["packet_count"], r["byte_count"], r["tp_src"], r["tp_dst"],
                         r["flow_duration_sec"], r["flow_duration_nsec"]) for r in rows]  # noqa: E731
    # 小端微秒 vs 大端微秒
    be = _flows_from(tool, write_pcap(tmp / "be.pcap", pkts, order=">"))
    check(key(base) == key(be), "大端（magic 0xa1b2c3d4 字节序 a1b2c3d4）与小端解析结果完全相同",
          f"{key(base)} vs {key(be)}")
    # 小端纳秒 vs 小端微秒
    ns = _flows_from(tool, write_pcap(tmp / "ns.pcap", pkts, magic=0xA1B23C4D))
    check(key(base) == key(ns), "纳秒精度 pcap（magic 0xa1b23c4d）解析结果与微秒一致",
          f"{key(ns)}")
    # 大端纳秒
    be_ns = _flows_from(tool, write_pcap(tmp / "bens.pcap", pkts, order=">", magic=0xA1B23C4D))
    check(key(base) == key(be_ns), "大端纳秒 pcap 也能读")
    # pcapng
    pg = write_pcapng(tmp / "c.pcapng", pkts)
    rows_pg = _flows_from(tool, pg)
    check(key(base) == key(rows_pg), "pcapng（SHB+IDB+EPB，if_tsresol=微秒）解析结果与 pcap 一致",
          f"{key(rows_pg)}")
    pg_ns = write_pcapng(tmp / "d.pcapng", pkts, tsresol=9)
    rows_ns = _flows_from(tool, pg_ns)
    check(key(base) == key(rows_ns), "pcapng 的 if_tsresol=10^-9（纳秒）也被正确换算", f"{key(rows_ns)}")
    # 空 pcap：只有 24 字节全局头
    empty = write_pcap(tmp / "empty.pcap", [])
    raised = None
    try:
        info, gen = tool.iter_capture(empty)
        list(gen)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    code, text = run_tool(tool, ["--input", str(empty), "--output", str(tmp / "e.csv")])
    check(code == 3 and "没有任何报文" in text, "空 pcap（只有文件头）：返回码 3 且给出清晰错误",
          f"返回码 {code}；{text.strip().splitlines()[-1][:70] if text.strip() else ''}")
    # 截断：完整 pcap 砍掉尾部 10 字节
    good = write_pcap(tmp / "g.pcap", pkts)
    trunc = tmp / "trunc.pcap"
    trunc.write_bytes(good.read_bytes()[:-10])
    code2, text2 = run_tool(tool, ["--input", str(trunc), "--output", str(tmp / "f.csv")])
    ok2 = code2 == 3 and ("截断" in text2 or "损坏" in text2)
    check(ok2, "被截断的 pcap：不崩溃、返回码 3、错误信息里点明截断",
          f"返回码 {code2}；{text2.strip().splitlines()[-1][:80] if text2.strip() else ''}")
    # 截断到只有 8 字节
    tiny = tmp / "tiny.pcap"
    tiny.write_bytes(b"\xd4\xc3\xb2\xa1\x02\x00\x04\x00")
    code3, text3 = run_tool(tool, ["--input", str(tiny), "--output", str(tmp / "h.csv")])
    check(code3 == 3, "文件头都不完整的 pcap：返回码 3（不抛未捕获异常）", f"返回码 {code3}")
    # 不是抓包文件（随便一段字节）
    junk = tmp / "junk.pcap"
    junk.write_bytes(b"hello world, not a pcap at all" * 3)
    code4, text4 = run_tool(tool, ["--input", str(junk), "--output", str(tmp / "i.csv")])
    check(code4 == 3 and "pcap" in text4, "非法文件：返回码 3 且说明不是 pcap/pcapng",
          f"返回码 {code4}")
    check(raised is not None, "空 pcap 在 API 层也有明确异常（iter_capture 不静默返回空）",
          f"{type(raised).__name__ if raised else '未抛异常'}")


def test_max_flows_and_selfcheck(tool, tmp: Path, sample_csv: Path, sample_pcap: Path) -> None:
    out = tmp / "capped.csv"
    code, _ = run_tool(tool, ["--input", str(sample_pcap), "--output", str(out), "--max-flows", "50"])
    rows = parse_csv(out)
    check(code == 0 and len(rows) == 50, "--max-flows 精确截断（大抓包快速验证用）",
          f"实际 {len(rows)} 条")
    check(all(int(float(r["packet_count"])) >= 1 for r in rows), "截断后的行依然是完整流（不是半个流）")
    # 自检能抓出人为破坏
    rows_full = parse_csv(sample_csv)
    feats = json.loads(META_PATH.read_text(encoding="utf-8"))["feature_names"]
    _, jtext = run_tool(tool, ["--input", str(sample_pcap), "--output", str(tmp / "rc.csv"),
                               "--json", str(tmp / "rc.json")])
    pstats = json.loads((tmp / "rc.json").read_text(encoding="utf-8"))["parse_stats"]
    chk_ok = tool.self_check(rows_full, feats, parse_stats=pstats)
    n_hard = sum(c["n_violations"] for c in chk_ok["hard"])
    check(n_hard == 0, "自检器对正常输出判定全部通过（含全局能量守恒）",
          f"硬性违例 {n_hard}")
    chk_nostats = tool.self_check(rows_full, feats)
    check(sum(c["n_violations"] for c in chk_nostats["hard"]) == 0
          and any("跳过" in c["detail"] for c in chk_nostats["hard"]),
          "没给解析统计时，守恒检查会明确写「跳过」而不是假装通过")
    tampered = [dict(r) for r in rows_full]
    tampered[0]["packet_count_per_second"] = str(float(tampered[0]["packet_count_per_second"]) + 1.0)
    chk_bad = tool.self_check(tampered, feats, parse_stats=pstats)
    rate_check = [c for c in chk_bad["hard"] if "速率恒等式" in c["name"]][0]
    check(rate_check["n_violations"] == 1, "自检器能抓出人为破坏的速率列（不是永远说 OK）",
          f"违例 {rate_check['n_violations']} 处")
    tampered2 = [dict(r) for r in rows_full]
    tampered2[0]["byte_count"] = str(float(tampered2[0]["packet_count"]) * 10)   # 每包 10 字节 → 不物理
    chk_bad2 = tool.self_check(tampered2, feats, parse_stats=pstats)
    check(sum(c["n_violations"] for c in chk_bad2["soft"]) >= 1,
          "自检器能抓出 byte_count < packet_count×20 的不物理数据（软性告警）")


def test_schema(tool, tmp: Path, sample_csv: Path, sample_pcap: Path) -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    with sample_csv.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    expect = list(meta["feature_names"])
    check(header[:len(expect)] == expect,
          "CSV 前 17 列的名字与顺序与 data/processed_sdn/meta.json 的 feature_names 完全一致",
          f"{header[:len(expect)]}")
    check(header[len(expect):] == ["timestamp", "ip_src", "ip_dst", "flow_id", "label"],
          "上下文列与标签列紧随其后", f"{header[len(expect):]}")
    check(len(expect) == 17 and "datapath_id" in expect,
          "特征数 17 且包含 datapath_id（任务描述里漏了它，必须以 meta.json 为准）")
    # 列名集合能被 DetectionService 直接接受（不需要任何改名/补齐）
    rows = parse_csv(sample_csv)
    missing = [f for f in expect if f not in rows[0]]
    check(not missing, "解析输出可以直接喂给控制器原生口径的检测服务（无缺列）",
          f"缺列 {missing}" if missing else "17/17 列齐全")
    # 校验和：独立复算生成报文的 IPv4 头校验和
    info, gen = tool.iter_capture(sample_pcap)
    bad = 0
    n = 0
    for _ts, _orig, frame in gen:
        n += 1
        ip = frame[14:]
        total = struct.unpack(">H", ip[2:4])[0]
        if total != len(ip):
            bad += 1
        if _ip_csum(ip[:20]) != 0xFFFF:
            bad += 1
        if n >= 200:
            break
    check(bad == 0, f"生成的报文真的合法：IPv4 总长字段与头校验和都正确（抽检 {n} 个）",
          f"异常 {bad} 处")


def _ip_csum(hdr: bytes) -> int:
    """独立实现的反码求和（与工具里的实现无关）。"""
    s = 0
    for i in range(0, len(hdr), 2):
        s += (hdr[i] << 8) | hdr[i + 1]
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return s & 0xFFFF


# ======================================================================================
#  入口
# ======================================================================================
def _run_all(flows: int, json_out: str) -> int:
    print("=" * 78)
    print("pcap → 流特征 往返一致性测试（合成抓包 → 解析 → 逐位比对）")
    print("=" * 78)
    tool = _load_tool()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("[1] 往返一致性（微秒精度 / 小端）")
        st = test_roundtrip(tool, tmp, flows)
        STATS.update(st)
        sample_pcap = tmp / "sample.pcap"
        sample_csv = tmp / "parsed.csv"
        STATS["_sample_pcap"] = str(sample_pcap)
        print("[2] 纳秒精度往返（时长应精确到纳秒网格）")
        ns_tmp = tmp / "ns_dir"
        ns_tmp.mkdir()
        STATS["ns"] = test_roundtrip(tool, ns_tmp, max(24, flows // 5))
        print("[3] 超时切分（空闲 / 硬超时，含边界值）")
        test_timeouts(tool, tmp)
        print("[4] VLAN / QinQ / 分片策略")
        test_vlan_and_fragment(tool, tmp)
        print("[5] 抓包格式：大小端、微秒/纳秒、pcapng、异常输入")
        test_formats(tool, tmp)
        print("[6] --max-flows 截断 与 自检器有效性")
        test_max_flows_and_selfcheck(tool, tmp, sample_csv, sample_pcap)
        print("[7] 输出 schema 与生成报文的合法性")
        test_schema(tool, tmp, sample_csv, sample_pcap)
    n_fail = sum(1 for ok, _, _ in _results if not ok)
    print("-" * 78)
    print(f"{len(_results) - n_fail}/{len(_results)} 项通过"
          + ("" if n_fail == 0 else f"，{n_fail} 项失败 ❌"))
    if json_out:
        payload = {k: v for k, v in STATS.items() if not k.startswith("_")}
        payload["test_total"] = len(_results)
        payload["test_failed"] = n_fail
        payload["failures"] = [name for ok, name, _ in _results if not ok]
        Path(json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"测试统计已写出：{json_out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="pcap → 流特征 往返一致性测试")
    ap.add_argument("--flows", type=int, default=200, help="往返测试合成多少条流（默认 200）")
    ap.add_argument("--json", default="", help="把测试统计写成 JSON 到该路径")
    a = ap.parse_args()
    raise SystemExit(_run_all(a.flows, a.json))
