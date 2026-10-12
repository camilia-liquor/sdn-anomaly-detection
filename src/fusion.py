"""多源证据融合：控制面（OpenFlow 流表统计）+ 数据面（pcap 抓包）双路。

为什么需要这一层（以及**不**是为了什么）
----------------------------------------
两路证据各有看不到的东西，而且**漏的地方不一样**：

| 视角 | 看得到 | 看不到（因此会漏） |
|---|---|---|
| 控制面（流表统计） | 交换机上所有经过的流、不受加密影响、字段是交换机原生的 | 包级细节（包长分布/IAT/TCP 标志位）；表项过期前的短流可能只出现一次统计 |
| 数据面（pcap 抓包） | 包级细节、可复原完整 CICFlowMeter 特征 | **采集点之外**的流量（镜像口丢包、采样、分片、加密隧道）、丢包/乱序 |

所以融合的目的**不是"取个平均分让指标好看一点"**，而是两件具体的事：
1. **互补**：一路漏掉的样本，另一路可能抓到（评估时必须能把这部分**单独数出来**）；
2. **对冲单点**：任何单一视角都可能被绕过/失效，两路同时看才谈得上"单点不可绕过"。

⚠️ 由此推出本模块最重要的一条护栏（也是它与"简单投票/平均"最大的区别）：
**冲突必须显式暴露，绝不允许一路静默否掉另一路。**
如果控制面说"这是攻击"而数据面说"这是正常"，取平均可能得到一个"看起来正常"的分数，
于是两个视角的对立被融合层悄悄抹平——单点风险不但没消除，反而被藏了起来。
这里的做法是：冲突时取**更保守**的一侧（更高风险）并标记 `conflict=True` / `review_required=True`，
把"该由人看一眼"这件事明确交出去；只有一路可用时也**不假装**两路都在（标 `degraded=True`）。

诚实边界：本模块只做**证据层/决策层**融合（可解释、可审计、能对回台账），
不做"把两种不同口径的特征拼成一个向量再训练"的特征层融合——后者需要同一批流上**真实同时存在**
的两路原始特征，本工程手上只有仿真的双视角数据（见 results/multisource_fusion.md 的口径说明）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# 两路证据的规范名：写进台账与报告，避免各处自己拼字符串
SOURCE_CONTROL = "control"     # 控制面：OpenFlow 流表统计（控制器原生字段）
SOURCE_DATA = "data"           # 数据面：pcap 抓包还原出的流特征
SOURCES = (SOURCE_CONTROL, SOURCE_DATA)

# 从两种 schema 里取四元组的字段名（控制面用 tp_src/tp_dst/ip_proto，数据面用 CICFlowMeter 列名）
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    # ⚠️ 别名表必须覆盖**磁盘上真实存在的列名**：`data/raw/sdn_stream.csv` 用的是
    #    `ip_src/ip_dst/tp_src/tp_dst/ip_proto`，服务接口用的是 `Src IP/Dst IP`，
    #    CICFlowMeter 口径用 `Source IP/Destination Port`。少一个别名，那一侧的流就全都
    #    "取不到键"→ 对齐结果变成"两路零匹配"，而**不会报错**（本工具第一版就踩了这个）。
    "src_ip": ("Src IP", "Source IP", "src_ip", "ip_src", "ipv4_src"),
    "dst_ip": ("Dst IP", "Destination IP", "dst_ip", "ip_dst", "ipv4_dst"),
    "src_port": ("tp_src", "Source Port", "src_port", "tcp_src", "udp_src"),
    "dst_port": ("tp_dst", "Destination Port", "dst_port", "Dst Port", "tcp_dst", "udp_dst"),
    "ip_proto": ("ip_proto", "Protocol", "proto"),
}


def flow_key(row: dict, *, use_ports: bool = True) -> tuple | None:
    """把一个流记录归一成对齐用的键（源/目的 IP + 端口 + 协议）。

    为什么需要它：两路证据要"拼在一起看"，前提是能确认**它们说的是同一条流**。
    控制面与数据面的字段名完全不同（`tp_dst` vs `Destination Port`），
    所以这里用别名表把两边归一，而不是在调用处各写一份映射（那必然会漏改一处）。

    源/目的 IP 任一缺失就返回 None：宁可不融合，也不要把两条不同的流硬配成一对
    （错配的融合证据比没有证据更危险——它会给出一个看起来有依据的错误结论）。
    """
    src = _pick(row, "src_ip")
    dst = _pick(row, "dst_ip")
    if not src or not dst:
        return None
    key: list[Any] = [str(src), str(dst)]
    if use_ports:
        for name in ("src_port", "dst_port"):
            v = _pick(row, name)
            try:
                key.append(int(v) if v not in (None, "") else -1)
            except (TypeError, ValueError):
                key.append(-1)
    proto = _pick(row, "ip_proto")
    try:
        key.append(int(proto) if proto not in (None, "") else -1)
    except (TypeError, ValueError):
        key.append(-1)
    return tuple(key)


def _pick(row: dict, name: str) -> Any:
    for alias in _KEY_ALIASES[name]:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def align_by_key(control_rows: Iterable[dict], data_rows: Iterable[dict]) -> dict:
    """按流键把两路证据对齐，返回 matched / 各自独有的清单。

    返回 ``{"matched": [(control_row, data_row), ...], "control_only": [...], "data_only": [...],
    "unkeyed": {"control": n, "data": n}}``。

    **不丢证据**：只在一路出现的流也如实列出来（`control_only` / `data_only`），
    它们是"这一路看到了、另一路没看到"的直接证据——正是融合要补的那部分。
    缺少五元组的记录（如抓包里的非 IP 报文）计入 `unkeyed`，不算作任何一路的漏报。
    """
    c_map: dict[tuple, list[dict]] = {}
    d_map: dict[tuple, list[dict]] = {}
    unkeyed = {"control": 0, "data": 0}
    for row in control_rows:
        k = flow_key(row)
        if k is None:
            unkeyed["control"] += 1
            continue
        c_map.setdefault(k, []).append(row)
    for row in data_rows:
        k = flow_key(row)
        if k is None:
            unkeyed["data"] += 1
            continue
        d_map.setdefault(k, []).append(row)

    matched: list[tuple[dict, dict]] = []
    for k, crows in c_map.items():
        drows = d_map.get(k) or []
        for i, crow in enumerate(crows):
            if i < len(drows):
                matched.append((crow, drows[i]))
    control_only = [r for k, rows in c_map.items() for r in rows[len(d_map.get(k) or []):]]
    data_only = [r for k, rows in d_map.items() for r in rows[len(c_map.get(k) or []):]]
    return {"matched": matched, "control_only": control_only, "data_only": data_only,
            "unkeyed": unkeyed}


@dataclass
class SourceEvidence:
    """一路证据：分数 + 它自己的判定 + 可用性（不可用时**不许**编造分数）。"""

    source: str
    p_attack: float | None = None
    pred_class: str = ""
    available: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {"source": self.source,
                "p_attack": None if self.p_attack is None else round(float(self.p_attack), 4),
                "pred_class": self.pred_class, "available": bool(self.available),
                "note": self.note}


@dataclass
class FusedVerdict:
    """融合结论（进台账/报告的结构）。"""

    p_attack: float | None
    alarm: bool
    conflict: bool
    review_required: bool
    degraded: bool
    sources: list[dict] = field(default_factory=list)
    reason: str = ""
    weak_agreement: bool = False     # True = 两路都没到阈值、但两路一致地"差一点"（融合独有的一种报警）

    def to_dict(self) -> dict:
        return {"p_attack": None if self.p_attack is None else round(float(self.p_attack), 4),
                "alarm": bool(self.alarm), "conflict": bool(self.conflict),
                "review_required": bool(self.review_required), "degraded": bool(self.degraded),
                "weak_agreement": bool(self.weak_agreement),
                "reason": self.reason, "sources": list(self.sources)}


# 融合规则只有一条（写死为常量，避免"换个函数名就换了语义"）：
#   * 两路都可用：取**更保守**（更高风险）的分数——漏报比误报更贵；
#   * 冲突（一路报警、另一路连"接近报警"都不到）：必须标记 conflict/review_required；
#   * 只有一路可用：用那一路，但标 degraded（如实说明"这次只有一只眼睛在看"）；
#   * 两路都不可用：不给分数（None）——**不编造**。
RULE = "conservative_max_with_explicit_conflict"


def fuse_evidence(control: SourceEvidence, data: SourceEvidence, *,
                  threshold: float = 0.5, conflict_gap: float = 0.2,
                  weak_agree_ratio: float = 0.5) -> FusedVerdict:
    """把两路证据融合成一个判定。

    ``conflict_gap``：判定"两路真的对立"的裕度——一路 ≥ 阈值、另一路 < 阈值 − gap 才算冲突；
    仅仅在阈值附近的分差（例如 0.51 vs 0.49）属于"两路都不确定"，不标冲突，避免冲突标记被噪声淹没。

    ``weak_agree_ratio``：**融合唯一能超过单路的场景**——两路各自都没到阈值，但都达到
    ``阈值 × ratio``（默认 0.5）时也报警（"两个视角各自都差一点，但方向一致"）。
    设 0 即关闭，退化为"两路取并集"。为什么这个口子值得开：单路弱证据容易被噪声淹没，
    两路同时弱报警的信息量显著更高；代价是误报会上升，所以它默认 0.5 而不是更激进的值，
    且报警时明确标 `weak_agreement=True`（运维能看出"这条不是单路硬证据"）。
    """
    thr = float(threshold)
    gap = max(0.0, float(conflict_gap))
    avail = [e for e in (control, data) if e.available and e.p_attack is not None]
    sources = [control.to_dict(), data.to_dict()]
    if not avail:
        # 两路都没有证据：不编造分数，只如实说"无证据"
        return FusedVerdict(p_attack=None, alarm=False, conflict=False, review_required=False,
                            degraded=True, sources=sources,
                            reason="两路证据都不可用（无分数）→ 不给判定，交上游按缺数据策略处理")

    p = max(float(e.p_attack) for e in avail)
    alarm = p >= thr
    degraded = len(avail) < 2

    weak = False
    if not alarm and len(avail) == 2 and float(weak_agree_ratio or 0) > 0:
        floor = thr * float(weak_agree_ratio)
        if all(float(e.p_attack) >= floor for e in avail):
            alarm = True
            weak = True

    conflict = False
    if len(avail) == 2:
        hi = max(float(e.p_attack) for e in avail)
        lo = min(float(e.p_attack) for e in avail)
        conflict = (hi >= thr) and (lo < thr - gap)
    # 冲突与降级都必须交人看：冲突=两个视角互相矛盾；降级=这次只有一路在说话
    review = bool(conflict or degraded)
    if conflict:
        reason = (f"两路结论对立（{avail[0].p_attack:.3f} vs {avail[1].p_attack:.3f}，阈值 {thr:.2f}）："
                  f"取更保守的一侧并要求人工研判，**不允许**用平均把对立抹平")
    elif weak:
        reason = (f"两路弱证据一致（{avail[0].p_attack:.3f} / {avail[1].p_attack:.3f}，"
                  f"均 ≥ {thr * float(weak_agree_ratio):.2f} 但均未达阈值 {thr:.2f}）→ 报警并标注弱证据")
    elif degraded:
        live = avail[0]
        reason = f"只有 {live.source} 一路可用（degraded）：用它的分数，但如实标注覆盖度不足"
    else:
        reason = f"两路一致（{avail[0].p_attack:.3f} / {avail[1].p_attack:.3f}，阈值 {thr:.2f}）"
    return FusedVerdict(p_attack=p, alarm=alarm, conflict=conflict, review_required=review,
                        degraded=degraded, sources=sources, reason=reason, weak_agreement=weak)


def fuse_rows(aligned: dict, scores: dict, *, threshold: float = 0.5,
              conflict_gap: float = 0.2, weak_agree_ratio: float = 0.5) -> list[dict]:
    """对一个对齐结果做逐样本融合，返回可直接写报告的记录列表。

    ``scores``：``{"control": {id(row): p_attack}, "data": {id(row): p_attack}}``——
    按 `id()` 取分数，避免依赖行对象的可哈希性（它们可能是 dict/DataFrame 记录）。
    """
    out: list[dict] = []
    for kind, rows in (("matched", aligned.get("matched") or []),
                       ("control_only", [(r, None) for r in aligned.get("control_only") or []]),
                       ("data_only", [(None, r) for r in aligned.get("data_only") or []])):
        for crow, drow in rows:
            ev_c = SourceEvidence(SOURCE_CONTROL,
                                 p_attack=(scores.get(SOURCE_CONTROL) or {}).get(id(crow)),
                                 available=crow is not None,
                                 note="" if crow is not None else "该流只在数据面出现")
            ev_d = SourceEvidence(SOURCE_DATA,
                                 p_attack=(scores.get(SOURCE_DATA) or {}).get(id(drow)),
                                 available=drow is not None,
                                 note="" if drow is not None else "该流只在控制面出现")
            v = fuse_evidence(ev_c, ev_d, threshold=threshold, conflict_gap=conflict_gap,
                              weak_agree_ratio=weak_agree_ratio)
            out.append({"kind": kind, "verdict": v.to_dict(),
                        "key": flow_key(crow if crow is not None else drow)})
    return out


def fusion_metrics(records: list[dict], y_true: list[int], *, threshold: float = 0.5) -> dict:
    """对比"只用控制面 / 只用数据面 / 融合"三者的召回与误报，并**单独数出互补部分**。

    ``records`` 与 ``y_true`` 一一对应（同序）。三个必须单列的数：
      * `rescued_by_pair`：一路漏、另一路抓到（融合把这部分保住了 = 互补覆盖的价值）；
      * `recovered_by_fusion`：**两路单独都漏**、融合抓到（只有"弱证据一致"规则才可能 >0）；
      * `lost_by_fusion`：某一路抓到、融合反而漏——**必须为 0**，否则融合规则有 bug
        （这条是给未来改规则的人留的哨兵：任何"取平均/投票"式的改动都会让它 >0）。
    """
    thr = float(threshold)
    n = len(records)
    assert len(y_true) == n, "records 与 y_true 长度必须一致"
    stat = {"n": n, "pos": 0, "control_hit": 0, "data_hit": 0, "fused_hit": 0,
            "rescued_by_pair": 0, "recovered_by_fusion": 0, "lost_by_fusion": 0,
            "conflicts": 0, "degraded": 0, "weak_agreement": 0,
            "fused_fp": 0, "control_fp": 0, "data_fp": 0}
    for rec, y in zip(records, y_true):
        v = rec["verdict"]
        srcs = {s["source"]: s for s in v["sources"]}
        c_ok = bool(srcs[SOURCE_CONTROL]["available"] and
                    (srcs[SOURCE_CONTROL]["p_attack"] or 0.0) >= thr)
        d_ok = bool(srcs[SOURCE_DATA]["available"] and
                    (srcs[SOURCE_DATA]["p_attack"] or 0.0) >= thr)
        f_ok = bool(v["alarm"])
        if int(y) == 1:
            stat["pos"] += 1
            stat["control_hit"] += int(c_ok)
            stat["data_hit"] += int(d_ok)
            stat["fused_hit"] += int(f_ok)
            stat["rescued_by_pair"] += int(c_ok != d_ok)
            stat["recovered_by_fusion"] += int(f_ok and not c_ok and not d_ok)
            stat["lost_by_fusion"] += int((c_ok or d_ok) and not f_ok)
        else:
            stat["control_fp"] += int(c_ok)
            stat["data_fp"] += int(d_ok)
            stat["fused_fp"] += int(f_ok)
        stat["conflicts"] += int(v["conflict"])
        stat["degraded"] += int(v["degraded"])
        stat["weak_agreement"] += int(v.get("weak_agreement") or False)
    pos = max(stat["pos"], 1)
    neg = max(n - stat["pos"], 1)
    stat["recall_control"] = round(stat["control_hit"] / pos, 4)
    stat["recall_data"] = round(stat["data_hit"] / pos, 4)
    stat["recall_fused"] = round(stat["fused_hit"] / pos, 4)
    stat["fnr_control"] = round(1 - stat["recall_control"], 4)
    stat["fnr_data"] = round(1 - stat["recall_data"], 4)
    stat["fnr_fused"] = round(1 - stat["recall_fused"], 4)
    stat["fpr_control"] = round(stat["control_fp"] / neg, 4)
    stat["fpr_data"] = round(stat["data_fp"] / neg, 4)
    stat["fpr_fused"] = round(stat["fused_fp"] / neg, 4)
    stat["threshold"] = thr
    stat["rule"] = RULE
    return stat
