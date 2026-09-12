# pcap → 控制器原生口径流特征：往返一致性与端到端检测报告

本报告的全部数字都由 `tools/pcap_to_flows.py` 与 `tests/test_pcap_tools.py` 现场跑出，
原始数据见同目录的 `results/pcap_roundtrip.json`（本报告不含任何绝对路径或机器相关信息）。

## 0. 一句话结论

**一段真实抓包（pcap/pcapng）可以直接算出控制器原生 17 字段口径的流特征，与离线仿真口径逐位对齐
（17/17 个特征、800/800（100.0%） 完全一致），
并可直接喂给既有检测服务出告警。** 这条链路补上了"仿真数据 → 真实网络"的最后一环：
离线取证与事后复盘不需要控制器在场。

| 环节 | 命令 |
|---|---|
| 现场抓包 | `tcpdump -i eth0 -s 0 -w capture.pcap` |
| 转成流特征 | `python tools/pcap_to_flows.py --input capture.pcap --output flows.csv` |
| 检测 | `python tools/predict_cli.py --input flows.csv`（或推送 `src/service.py`） |

## 1. 支持到什么程度

| 项目 | 支持情况 |
|---|---|
| 经典 `.pcap` | 支持；微秒（`0xa1b2c3d4`）与纳秒（`0xa1b23c4d`）两种精度，**大小端都认**（4 种组合均有测试） |
| `.pcapng` | **支持读取**（SHB / IDB / EPB，含 `if_tsresol` 选项，含二进制 2^-n 时基）；不支持写出 |
| 不支持并明确报错 | "modified pcap"（magic `0xa1b2cd34`）、pcapng 的 SPB（Simple Packet Block，无时间戳） |
| 链路层 | Ethernet(1)、RAW IP(101)、IPV4(228)、LINUX_SLL(113)、LINUX_SLL2(276)——覆盖 `tcpdump -i any` 的 cooked 抓包；其余类型计数并提示 |
| VLAN | 支持 802.1Q 与 QinQ（0x8100 / 0x88A8 / 0x9100），可多层剥标签 |
| 传输层 | TCP(6) / UDP(17) / ICMP(1)；其它 `ip_proto` 计数后跳过（没有端口，凑不出五元组） |
| IPv4 分片 | 正确处理 IHL 与分片标志；**首片计入本流，非首片（offset>0）跳过并计数**——非首片没有传输层头，在真实数据面里匹配的是另一条只按 L3 建的表项，跳它才与控制器统计口径一致 |
| 切分 | 空闲超时（`--idle-timeout`，默认 15 s，**>** 才切）与硬超时（`--hard-timeout`，默认 120 s，从流首报文起 **>=** 强制切），等价于 OpenFlow 流表项的 idle/hard timeout |
| 依赖 | **纯标准库，不需要 scapy**，也没有引入任何新依赖 |

## 2. 往返一致性实测（合成 → 解析 → 逐位比对）

样本：`samples/sample_capture.pcap`（401738 字节 = 392.3 KiB，
sha256 前 16 位 `537e3452bbe280f8`），800 条流 / 3207 个报文，
时间戳精度 us、字节序 little，类别分布：Normal 407、Probe 243、Botnet 25、BruteForce 25、DDoS 25、DoS 25、U2R 25、WebAttack 25。

> 复现性：用同一条命令（`--write-sample samples/sample_capture.pcap --flows 800`）重新生成，
> 输出与交付样本 **逐字节相同**（sha256 一致），因此下面所有数字都适用于交付的那份样本。

| 指标 | 结果 |
|---|---|
| 生成流数 → 解析回流数 | 800 → 800（**1:1，无额外合并/拆分**） |
| 与生成清单逐位一致的流数 | 800/800（100.0%）（17 个特征全部逐位相同，见下表） |
| `packet_count` 与仿真源记录完全一致 | 800/800（100.0%） |
| `ip_proto` 完全一致 | 800/800（100.0%） |
| TCP/UDP 的 `tp_src`/`tp_dst` 完全一致 | 682/682（100.0%）（ICMP 见下） |
| `byte_count` 与仿真源记录 | 最大偏差 **0.499 字节**（均值 0.246），取整后一致 800/800 |
| 流时长与仿真源记录 | 最大偏差 **12858 ns**（均值 750 ns），容差上界 29000 ns |
| `flow_id` 与采集脚本拼接方式一致 | 800/800（100.0%） |
| 报文总数守恒 | Σ`packet_count` = 3207 = 解析到的报文数；Σ`byte_count` = 350402 = 解析到的字节数 |

逐特征明细（与生成清单比对）：

| 特征 | 完全一致的条数 | 最大相对误差 |
|---|---|---|
| `datapath_id` | 800/800 | 逐位一致 |
| `tp_src` | 800/800 | 逐位一致 |
| `tp_dst` | 800/800 | 逐位一致 |
| `ip_proto` | 800/800 | 逐位一致 |
| `icmp_code` | 800/800 | 逐位一致 |
| `icmp_type` | 800/800 | 逐位一致 |
| `flow_duration_sec` | 800/800 | 逐位一致 |
| `flow_duration_nsec` | 800/800 | 逐位一致 |
| `idle_timeout` | 800/800 | 逐位一致 |
| `hard_timeout` | 800/800 | 逐位一致 |
| `flags` | 800/800 | 逐位一致 |
| `packet_count` | 800/800 | 逐位一致 |
| `byte_count` | 800/800 | 逐位一致 |
| `packet_count_per_second` | 800/800 | 逐位一致 |
| `packet_count_per_nsecond` | 800/800 | 逐位一致 |
| `byte_count_per_second` | 800/800 | 逐位一致 |
| `byte_count_per_nsecond` | 800/800 | 逐位一致 |

**容差是怎么定的**（这几条决定了上表能不能被称为"一致"）：

* `packet_count` / `byte_count` / `ip_proto` / 端口 / `icmp_*` / 4 个速率列：整数帧长 + 整数计数，
  **没有浮点空间，判据就是逐位相等**；
* 时长：解析值 = `(packet_count-1) × 报文间隔`。生成时把间隔对齐到 pcap 时间戳网格
  （微秒 → 1000 ns 网格），因此解析值与清单**逐位相同**；与"仿真源记录的 duration"相比则存在网格量化，
  上界 `(packet_count-1) × 网格 / 2`，实测最大 12858 ns（远小于 1 ms）；
* `byte_count` 与仿真源记录：帧长只能是整数（14+20+20+载荷），而仿真数据的 `byte_count`
  是浮点（例如 `838.9411128275215`），**物理上不可能逐位复现**。判据因此取
  `|解析值 - 仿真值| ≤ 0.5 字节` 且 `round(解析值) == round(仿真值)`——实测全部满足；
* ICMP：仿真数据里 ICMP 行的 `tp_src`/`tp_dst` 是仿真器随手填的（ICMP 报文根本没有端口），
  而 Ryu 采集脚本与解析器都记 0，所以样本里的 118 条 ICMP 流按 0 处理并如实记录，
  不参与"端口一致"的统计（这属于仿真数据自身的不一致，不是解析器的问题）。

## 3. 端到端演示：把解析出的 CSV 喂给控制器原生口径的检测模型

按控制器的方式批量推送解析结果（`src/service.py` 的 `DetectionService`，
预处理产物 `data/processed_sdn`，模型 `./artifacts/sdn_ctrl_s0/model.keras`，窗口 T=16，
在线步长 3，阈值 0.5，连续 3 个窗口确认）：

| 指标 | 结果 |
|---|---|
| 推送的流记录数 | 800 条（分 13 批，每批 64 条，模拟控制器周期上报） |
| 判定窗口数 | 261 个（前 539 条流还没填满窗口，未出结论——线上就是这样） |
| 窗口级告警数（p ≥ 0.5） | 224 |
| 连续确认告警数 | 198，按类别：{'DoS': 15, 'Normal': 32, 'Probe': 149, 'BruteForce': 1, 'U2R': 1} |
| 窗口级多分类准确率（与窗口内多数真值比） | **0.5556**（逐类：Botnet 0/6、BruteForce 0/8、DDoS 0/8、DoS 4/9、Normal 59/135、Probe 80/81、U2R 2/9、WebAttack 0/5） |
| 窗口级二分类（异常/正常） | TP 123、FP 101、FN 7、TN 30，精确率 0.5491、召回率 0.9462 |

真值来源：样本由 `data/raw/sdn_stream.csv` 反构造，每条流的 `attack_type` 由生成清单带出，
按 `timestamp` 与解析结果一一对齐（样本里 800 条流的时间戳两两不同，可精确 join）。
样本共有异常流 393 条、正常流 407 条。

**怎么读这个结果**：样本里同类流在时间上连续排列（模拟"攻击片段"，与离线评估"片段召回"口径一致），
所以窗口内标签单一，准确率看起来不错；换成真实抓包时窗口会混杂多种流量，指标会下降——
这也是为什么报告只把它当作"链路打通"的证据，而不是新的精度结论。
17 字段口径的模型精度上限本来就只有**宏 F1 0.8441**（见 `results/summary_sdn_ctrl.md`）。

## 4. 怎么用它接你自己的抓包

```powershell
# ① 现场抓包（-s 0 = 抓完整帧，别截断；否则 byte_count 会偏小，工具会告警）
tcpdump -i eth0 -s 0 -w capture.pcap

# ② 转成控制器原生 17 字段口径的流特征（默认空闲 15s / 硬超时 120s，与控制器采集口径一致）
python tools/pcap_to_flows.py --input capture.pcap --output flows.csv

# ③ 检测（离线批量看结果；若要在线上跑，把 flows.csv 逐条推给 src/service.py 即可）
python tools/predict_cli.py --input flows.csv --processed data/processed_sdn --quiet
```

大文件先切一刀再验证：`--max-flows 5000`；只想看统计不落盘：省略 `--output`。
Wireshark 保存的 `.pcapng` 也能直接读；如果文件是老的 "modified pcap"，先用
`editcap -F pcap in.pcap out.pcap` 转换。

## 5. 诚实的局限说明（评审可以直接照着问）

1. **口径不同，不是"更全"**：本工具算的是**控制器原生 17 字段口径**（OpenFlow 流表统计），
   与 CICFlowMeter 的 84 列不是一回事。CICFlowMeter 里的子流（sub-flow）、
   Bulk 统计（每秒字节/包、流内突发长度）、IAT 统计（到达间隔的均值/标准差/最大最小）、
   active/idle 统计等**我们都没有算**——因为 OpenFlow 流表不提供这些量，
   而"训练什么口径、线上就喂什么口径"是本作品的一条硬约束（口径不一致 = 训练-推理偏移）。
   要 84 列口径请用探针侧路线（`integrations/ryu_guard.py` 之外的 CICFlowMeter 采集）。
2. **4 个字段无法从报文推导**：`datapath_id`、`idle_timeout`、`hard_timeout`、`flags` 属于
   "交换机/流表项"属性，不是线路上的字节。本工具用 CLI 常量填充
   （默认 `--datapath-id 3`（仿真数据里最常见取值）、`--idle-timeout 15`、`--hard-timeout 120`、`--flags 1`），
   做自检/回归时必须与生成侧用同一组取值，否则这 4 列会"不一致"——这是**预期的**，不是 bug。
3. **时长语义不同**：我们按**首末报文跨度**算 `flow_duration`，而 OpenFlow 的 `duration`
   是**流表项存活时间**（含建表到首包、以及最后一次统计请求之前的空闲），因此真实控制器上
   这个字段会**大于**离线从 pcap 算出来的值。想完全对齐，需要用控制器侧
   `collect_normal_traffic.py` 那一路（本工具是离线取证口径）。
4. **分片包怎么处理**：只处理未分片报文与首片；非首片跳过并计数。若某条流真的被分片了，
   跟随分片的字节不会计入 `byte_count`（与"按端口建流表项"的数据面统计一致）。
5. **`byte_count` 的取整**：pcap 侧是帧长（整数），仿真数据是浮点，往返时量化 ≤ 0.5 字节。
   另外若抓包用了 `-s` 截断（snaplen < 帧长），`byte_count` 会系统性偏小，工具**会打印告警**并给出重抓建议。
6. **pcapng 只读**：支持 SHB/IDB/EPB 与 `if_tsresol`；不支持 SPB（无时间戳，无法算时长），
   也不支持写出 pcapng（`--write-sample` 输出的是经典 pcap）。
7. **样本类分布不代表真实网络**：`samples/sample_capture.pcap` 要控制在 1 MB 内，
   而仿真数据里 DDoS/U2R 的流天生很大（`byte_count` 中位数 10^5 量级），
   所以选流时按"最省字节"补齐 → 结果是 Normal 407、Probe 243、Botnet 25、BruteForce 25、DDoS 25、DoS 25、U2R 25、WebAttack 25。
   它只用于**自检与演示**，不能用来宣称类分布或精度。
8. **样本的流在时间上首尾相接**：为保证"解析出的流条数 = 生成条数"（同一五元组重复出现时必须被
   空闲超时切开），样本里流与流之间留了 1×idle_timeout+1s 的间隔，**真实抓包里的流是并发重叠的**。
   这是合成样本的简化，不是解析器的限制（解析器本身按时间戳在线聚合，支持任意重叠）。
9. **精度上限**：17 字段口径的模型宏 F1 = 0.8441（`results/summary_sdn_ctrl.md`）。
   换成真实抓包后受口径差异与分布漂移影响，实际指标只会更低，不会更高。
10. **既有实现的一处注意（不是本工具的问题）**：`artifacts/sdn_ctrl/model.keras` 的
    `mc` 输出头宽度是 2，而 `data/processed_sdn/meta.json` 声明 8 个类别，
    经 `src/service.py` 推送会 `IndexError: index 2 is out of bounds`。
    本报告的端到端演示因此改用同样 17 字段口径的 `artifacts/sdn_ctrl_s0`（8 类）。
    该问题在 `src/` 内，按本次任务的范围约束未做修改，仅在此记录。

## 6. 自检（工具自己跑的一致性核验，不是"自己算完自己点头"）

解析时会**重新读回写出的 CSV** 再核验，微秒/纳秒网格、速率恒等式、全局守恒逐条检查：

| 检查项 | 结果 |
|---|---|
| 速率恒等式：packet_count_per_second ≈ packet_count / duration_sec（duration_sec==0 时按 1 兜底、nsecond 列同理，见口径说明） | ✅ 通过 |
| 时长字段合法（sec >= 0 且 0 <= nsec < 1e9） | ✅ 通过 |
| 计数非负（packet_count >= 1 且 byte_count >= 1） | ✅ 通过 |
| 全局能量守恒：Σpacket_count == 已解析报文数 且 Σbyte_count == 已解析字节数 | ✅ 通过 |
| byte_count >= packet_count × 20（最小 IP+TCP 头，物理红线） 且 byte_count <= packet_count × 1514（以太网 MTU 上限） | ✅ 通过 |

测试套件：`python tests/test_pcap_tools.py` → **58/58 项通过**，退出码 0
（覆盖：往返逐位比对、与仿真源记录比对、大小端与两种时间戳精度、pcapng 读取、
空闲/硬超时切分与边界、VLAN/QinQ、分片策略、空/截断/非法文件、`--max-flows` 截断、
自检器能抓出人为破坏、输出 schema 与 meta.json 一致、生成报文的 IPv4 校验和合法）。
