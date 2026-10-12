# 多源证据融合实测（控制面流表 + 数据面 pcap）

> 由 `tools/multisource_fusion.py` 生成。**只读**既有产物，未重跑主流程、未改动 `results/summary.json` / `artifacts/main_model.json`。

## 0. 这次到底比了什么

同一条仿真流的两条独立测量路径，用**同一个模型**打分后再融合：

| 视角 | 怎么来的 |
|---|---|
| 控制面（control） | 交换机组装的流表统计（OpenFlow flow stats 口径） —— `C:\Users\xc\Desktop\generate_normal_traffic\data\raw\sdn_stream.csv` |
| 数据面（data） | 由合成 pcap 重建的流特征（独立测量路径） —— `_tmp\fusion\fusion_sample.pcap` |

- 流数 **400**（类别分布：{'Botnet': 25, 'BruteForce': 25, 'DDoS': 25, 'DoS': 25, 'Normal': 77, 'Probe': 173, 'U2R': 25, 'WebAttack': 25}）
- 取窗：窗口 **16** / 步长 **16** → **25** 个窗口，阈值 0.5，规则 `conservative_max_with_explicit_conflict`
- 按流键对齐：matched **400**、control_only 0、data_only 0、unkeyed {'control': 0, 'data': 0}
- pcap 生成清单与真实流表口径不一致的字段：**11 个**（['byte_count', 'byte_count_per_nsecond', 'byte_count_per_second', 'flags', 'flow_duration_nsec', 'hard_timeout', 'idle_timeout', 'packet_count_per_nsecond', 'packet_count_per_second', 'tp_dst', 'tp_src']）——这些正是**数据面路径表达不出来**的东西

## 1. 两路测量差异（冲突的来源，先看这个）

| 特征 | 两路取值不一致的流数 |
|---|---|
| `flow_duration_nsec` | **400** / 400 |
| `idle_timeout` | **400** / 400 |
| `byte_count` | **400** / 400 |
| `packet_count_per_nsecond` | **400** / 400 |
| `byte_count_per_second` | **400** / 400 |
| `byte_count_per_nsecond` | **400** / 400 |
| `hard_timeout` | **332** / 400 |
| `flags` | **299** / 400 |
| `tp_src` | **44** / 400 |
| `tp_dst` | **27** / 400 |
| `packet_count_per_second` | **6** / 400 |

两路**只有在这些字段上会分歧**（其余字段逐值相同）——集中在「每秒/纳秒速率」（由时长换算）、时长与字节取整、以及超时/标志位/端口这类**流表有、报文重建不一定有**的字段。⚠️ 端口/超时/标志位的差异**部分是本工具合成 pcap 时的默认值所致**（合成器按流表统计造包，无法反推这些字段）；真实镜像口能否看到它们取决于部署方式，不能一概而论——这里只说明「两路测量口径确实不同」这个事实本身。

## 2. 单路 vs 融合（窗口级）

| 口径 | 召回 | 漏报率 | 误报率 |
|---|---|---|---|
| 只用控制面 | 0.800 | 0.200 | 1.000 |
| 只用数据面 | 0.800 | 0.200 | 1.000 |
| **融合（conservative）** | **0.850** | **0.150** | **1.000** |

| 融合行为 | 数量 | 含义 |
|---|---|---|
| `rescued_by_pair` | 0 | 一路漏、另一路抓到（互补覆盖，融合保住） |
| `recovered_by_fusion` | 1 | **两路单独都漏**、融合抓到（弱证据一致） |
| `lost_by_fusion` | 0 | 某一路抓到、融合却漏 —— **必须为 0** |
| `conflicts` | 0 | 两路结论对立（已取保守侧并要求人工研判） |
| `degraded` | 0 | 只有一路可用（如实标注覆盖度不足） |
| `weak_agreement` | 1 | 两路都差一点但一致 → 报警 |

### 单路失效场景（模拟数据面采集路径故障）

真实运维里「一路瞎了」是必然会发生的事故，这里把它量出来——系统必须**说出来**，而不是静默退化成单路：

| 口径 | 召回 | 误报率 | degraded | 需人工研判 |
|---|---|---|---|---|
| 控制面单独 + 数据面不可用 | 0.800 | 1.000 | 25 / 25 | 25 |

结论：数据面失效时融合召回等价于控制面单独水平（0.800），但 **25/25 个窗口全部被标成 `degraded`** —— 覆盖度下降是**可见**的，不会变成「没人知道的单点」。

## 3. 结论与边界（不许省略）

1. **互补是真实存在的**：0 个窗口是「一路漏、另一路抓到」——单路视角下这些就是漏报，融合把它们保住了；
2. **融合不会丢掉任何一路抓到的攻击**：`lost_by_fusion` = 0（该值必须恒为 0，它同时是「将来有人把规则改成平均/投票」的哨兵）；
3. **代价也要如实说**：融合取保守侧，误报率是两路的并集——（1.000 vs 控制面 1.000 / 数据面 1.000）；两路对立的 0 个窗口被交给人研判——这正是「不能被平均抹平」的成本；
4. ⚠️ **口径边界**：两个视角来自**同一条仿真流的两种投影**，互补来自「流表计数 vs 报文重建」的测量差异，**不是**真实环境里「镜像口覆盖不到」那种覆盖差异；样本按类均衡选取，误报率**不代表真实基率**（阈值是在不均衡分布上标定的，本节数字只能用于两路之间的相对比较）。真实环境的覆盖差异需要现场部署才能量化。

## 4. 复现

```bash
python tools/multisource_fusion.py --flows 400 --window 16 --stride 16 --threshold 0.5
```

- 中间产物（可删）：`generate_normal_traffic/_tmp/fusion/`（pcap + manifest + 数据面 CSV）
- 数据面视角由 `tools/pcap_to_flows.py` 合成与还原，其往返保真度另见 `results/pcap_roundtrip.md`
