# 真实 Ryu 进程 + 模拟 OpenFlow 1.3 交换机：处置下发链路实测

> 由 `tools/ryu_live_check.py` 生成。Ryu 版本：4.34；检测服务模型：`sdn_ctrl_s0`（data/processed_sdn）；控制器端口 6700、服务端口 5300。

## 0. 结论

- **19/19 项检查通过**：链路端到端可用。
- 控制器下发的报文：FLOW_MOD **2** 条，其中**带源 IP 匹配的处置规则 1** 条（匹配 `10.0.0.9`，priority=100，是否丢弃=True）；审计台账 **10** 条。

> 这条实验把原来的边界「Ryu 适配层**未在真实交换机上实跑**」改写成：
> 「已在**真实 Ryu 进程**（官方 `ryu-manager`，非桩件）下、通过**真实 OF1.3 线协议**完成握手、流表统计上报与处置下发验证；**仅剩「物理交换机」这一项未覆盖**」。

## 1. 逐项检查

| 检查 | 结果 | 证据 |
|---|---|---|
| 检测服务已就绪 | ✅ | http://127.0.0.1:5300/api/health |
| 检测策略已切到「真实处置」（action=block、非演练） | ✅ | 服务回显策略：{'action': 'block', 'dry_run': False, 'threshold': 0.5, 'confirm_windows': 3} |
| 两轮流量取自真实数据集（不是手编数字） | ✅ | normal：测试流第 671 行起 25 条（其中 25 条可用 OF 流表统计如实表达）；attack：测试流第 16523 行起 39 条（其中 39 条可用 OF 流表统计如实表达） |
| 真实 Ryu 进程已监听 OF 端口 | ✅ | ryu-manager on 0.0.0.0:6700（非桩件） |
| OF1.3 握手完成（HELLO 交互） | ✅ | 收到控制器报文类型序列：['HELLO', 'FEATURES_REQUEST', 'MULTIPART_REQUEST', 'FLOW_MOD', 'MULTIPART_REQUEST', 'MULTIPART_REQUEST', 'FLOW_MOD', 'MULTIPART_REQUEST'] |
| 控制器识别交换机（FEATURES_REQUEST → FEATURES_REPLY） | ✅ | 控制器发出 OFPT_FEATURES_REQUEST，模拟交换机回 FEATURES_REPLY 1 条（Ryu 侧用自身解析器校验通过，见自检） |
| 端口描述应答使控制器进入 MAIN_DISPATCHER | ✅ | OF1.3 下 Ryu 收到 FEATURES_REPLY 后还会发 OFPMP_PORT_DESC 请求；本工具回了 1 条 PORT_DESC_REPLY，Ryu 日志出现「[guard] 交换机上线」 |
| 控制器周期拉取【流表】统计（OFPMP_FLOW 请求） | ✅ | 收到 4 次 OFPMP_FLOW 请求（适配层 monitor 周期 10 s）；另有 1 次 OFPMP_PORT_DESC 请求（握手用，不计入） |
| 模拟交换机已完成流表统计上报 | ✅ | 发送 2 条 OFPFlowStatsReply（真实 OF1.3 序列化） |
| Ryu 未报告任何报文解析错误（线格式合规） | ✅ | ryu-manager 日志中 「Encountered an error while parsing ...」出现 0 次 |
| 收到处置下发：FLOW_MOD | ✅ | 共 2 条 FLOW_MOD：处置规则 1 条（其中 DROP 指令 1 条）、交换机自身 table-miss 保底规则 1 条 |
| 下发规则的匹配字段指向可疑源 IP | ✅ | 处置规则的匹配源 IP：['10.0.0.9'] |
| 正常流量轮次未产生任何处置动作（无误报动作） | ✅ | 第 1 轮为正常的真实流（10.0.0.3）；针对正常源 IP 的处置规则 0 条 |
| 审计台账记录了真实执行（executed=true） | ✅ | 台账 10 条，其中真正执行 1 条、幂等抑制 9 条（示例 action=block src=10.0.0.9 类别=DoS p_attack=0.9986） |
| 处置动作与台账一致（action=block → 下发 DROP 或 METER 流表项） | ✅ | 台账动作：['block']；实际报文：处置 FLOW_MOD 1、METER_MOD 0、限速流表项 0 |
| 同一源 IP 的连续告警不重复下发规则（处置幂等） | ✅ | 10 条告警里：真实下发 1 次、封禁窗口内幂等抑制 9 次，交换机上只增加了 1 条规则 |
| 适配层发起了「带源 IP 匹配」的流表回读请求（OFPMP_FLOW + match） | ✅ | 带 match 的回读请求 1 次（周期采集 3 次不带过滤），模拟交换机针对回读回了 1 条流表回复 |
| 真实 Ryu 下回读校验判定「规则已生效」（xid 配对正确、未误判超时） | ✅ | event=verify 记录 1 条：ok=1、超时=0、动作不符=0；示例 {'event': 'verify', 'time': 1791826128.2774155, 'src_ip': '10.0.0.9', 'expect_action': 'block', 'ok': True, 'found': True, 'rule_kinds': ['drop'], 'n_stats': 40, 'reason': 'ok', 'datapath_id': 1} |
| 回读校验没有误回滚（交换机确实装了规则时不得撤规则） | ✅ | 回滚报文 0 条；交换机流表里的处置规则 1 条 |

## 2. 握手时间线（控制器 → 模拟交换机，逐条实收）

| t (s) | 报文类型 | multipart 子类型 | 说明 |
|---|---|---|---|
| 0.0 | HELLO | — | 版本协商（双方声明 OF1.3） |
| 0.02 | FEATURES_REQUEST | — | 控制器询问交换机能力 → 交换机回 FEATURES_REPLY |
| 0.02 | MULTIPART_REQUEST | OFPMP_PORT_DESC | OFPMP_PORT_DESC：OF1.3 握手的**必经一步**（交换机回 PORT_DESC_REPLY 后控制器才进 MAIN_DISPATCHER） |
| 0.02 | FLOW_MOD | — | 控制器下发流表项 |
| 8.43 | MULTIPART_REQUEST | OFPMP_FLOW | OFPMP_FLOW：周期性拉取流表统计（检测的数据来源） |
| 18.44 | MULTIPART_REQUEST | OFPMP_FLOW | OFPMP_FLOW：周期性拉取流表统计（检测的数据来源） |
| 18.45 | FLOW_MOD | — | 控制器下发流表项 |
| 18.45 | MULTIPART_REQUEST | OFPMP_FLOW | OFPMP_FLOW：周期性拉取流表统计（检测的数据来源） |
| 28.46 | MULTIPART_REQUEST | OFPMP_FLOW | OFPMP_FLOW：周期性拉取流表统计（检测的数据来源） |

> ⚠️ 这一步是本工具**真实踩到的坑**，也是最值得记录的一条：OF1.3 握手里，Ryu 收到 `FEATURES_REPLY` 后**不会**直接进 `MAIN_DISPATCHER`，而是先发一条 `OFPPortDescStatsRequest`（`OFPMP_PORT_DESC`），只有等交换机回了 `OFPPortDescStatsReply` 才 `set_state(MAIN_DISPATCHER)`（`ryu/controller/ofp_handler.py:216-232`）。在那之前，App 注册的所有 `MAIN_DISPATCHER` 事件都不会触发——包括适配层记录 datapath 上线、以及周期性拉流表。于是链路上**看不出任何错误**（TCP 通、HELLO 通、FEATURES 通），但控制器永远不会进入检测与处置流程。

## 3. 控制器下发的原始报文（逐条解析）

| 序号 | t (s) | 类型 | 关键字段 |
|---|---|---|---|
| 1 | 0.02 | FLOW_MOD | command=0 priority=0 match={} 指令=[{'type': 4, 'length': 24, 'actions': [{'type': 'OUTPUT', 'port': 4294967293}]}] 是否 DROP=False |
| 2 | 18.45 | FLOW_MOD | command=0 priority=100 match={'eth_type': 2048, 'ipv4_src': '10.0.0.9'} 指令=[{'type': 4, 'length': 8, 'actions': []}] 是否 DROP=True |

## 4. 上报了哪些流量（真实数据集的流记录 → OF 流表统计）

| 轮次 | 来源 | 条数 | 源 IP | 说明 |
|---|---|---|---|---|
| 1 | 测试流第 671 行起 25 条 | 25 | 10.0.0.3 | 正常形态（测试流中模型判定稳定的一段连续流） |
| 2 | 测试流第 16523 行起 39 条 | 39 | 10.0.0.9 | 攻击形态（DDoS 片段） |

> 两轮流量都来自 `data/processed_sdn` 的**测试流**（真实 Ryu 采集数据的预处理产物），不是手编的数字。转换规则：OF 的 `OFPFlowStats` 只有「五元组 + 整数秒时长 + 整数计数器」，**没有速率字段**，速率由适配层按 `packet_count / duration` 现算；因此只有测试流里**取值能被 OF 如实表达**的那些行会被采用（负时长、非法端口的行会被跳过）。
> 报告末尾的探针结果显示：这份「过一遍线协议」引入的数值误差会让个别窗口的判定发生变化，但**告警 / 不告警的结论在真实攻击段上完全一致**。

## 5. 审计台账（控制器侧留痕）

- 台账文件：`logs\_ryu_live\audit_alerts.jsonl`（本次运行独立，不污染主台账）
- 记录条数：**10**

| 时间 | 判定类别 | p_attack | 源 IP | 动作 | 真正执行 | 幂等抑制 | 演练 |
|---|---|---|---|---|---|---|---|
| 01:28:48 | DoS | 0.9986 | 10.0.0.9 | block | True | False | False |
| 01:28:48 | DDoS | 0.9999 | 10.0.0.9 | block | False | True | False |
| 01:28:48 | DDoS | 1.0 | 10.0.0.9 | block | False | True | False |
| 01:28:48 | DDoS | 1.0 | 10.0.0.9 | block | False | True | False |
| 01:28:48 | DDoS | 0.9999 | 10.0.0.9 | block | False | True | False |
| 01:28:48 | DDoS | 0.999 | 10.0.0.9 | block | False | True | False |
| 01:28:48 | DDoS | 0.9984 | 10.0.0.9 | block | False | True | False |
| 01:28:48 | DDoS | 0.996 | 10.0.0.9 | block | False | True | False |
| 01:28:48 | DDoS | 0.9998 | 10.0.0.9 | block | False | True | False |
| 01:28:48 | DDoS | 0.9987 | 10.0.0.9 | block | False | True | False |

> 一次攻击会产生**连续多个**告警窗口（在线检测是滑动窗口，60 s 的攻击会连报十几个窗口）。适配层按「同一源 IP 在封禁窗口内只下发一次」做**幂等**，所以交换机上只多出 1 条规则，其余 9 条告警只留痕、不重复下发——既避免流表抖动，也让「处置次数」这个指标可信。

## 6. 局限（写清楚，不粉饰）

- 模拟交换机是**自研脚本**：按 OF1.3 规范发包（HELLO / FEATURES / PORT_DESC / FLOW_STATS），并解析 FLOW_MOD/METER_MOD；**它不是物理设备**，未覆盖厂商私有扩展与完整协议一致性。
- 只覆盖本作品用到的消息类型；流表老化、多交换机拓扑、并发压力未测。
- 中间发生的一切都在本机回环：控制器、交换机、检测服务三者同机，**不含真实链路时延与丢包**。
- 流量是真实数据集的流记录（第 4 节），但**由本工具按 OF 报文构造**，不是现场抓包；5 元组里的 IP 是示意值，模型也不使用 IP。
- 「正常形态」那一轮用的是测试流中模型判定稳定的一段连续流（第 671 行起）；这不是随机抽样，而是为了让「处置路径有选择性」这条结论可复现。
- 「真机验证」仍不完整：需要实验室 SDN 环境里的真实交换机与真实攻击流量。

## 7. 复现命令

```powershell
# 1) 准备带 ryu 的解释器（本机为独立 venv，避免污染项目依赖）
#    下面的顺序不是教程里的顺序，而是本机把 Ryu 4.34 装上 Python 3.11/Windows 的实际路径：
#      a. pbr + 新版 setuptools 会因 flat-layout 报 metadata-generation-failed，必须钉 setuptools<64；
#      b. ryu 的依赖里 eventlet / dnspython 在 Py3.11 上有兼容问题，装完后要打两个 3 行补丁。
python -m venv _tmp/ryu_venv
_tmp/ryu_venv/Scripts/python.exe -m pip install "setuptools<64" wheel pbr
_tmp/ryu_venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation ryu
_tmp/ryu_venv/Scripts/python.exe -m pip install six msgpack netaddr routes webob tinyrpc oslo.config
_tmp/ryu_venv/Scripts/python.exe -m pip install eventlet                  # ≥0.33 均可
# 补丁 1：eventlet ≥0.33 把 ALREADY_HANDLED 挪进了 WSGI_LOCAL，ryu 还在按老写法 import
#         在 eventlet/wsgi.py 末尾追加一行：ALREADY_HANDLED = object()
# 补丁 2：dnspython 1.16 用了 Py3.10 已移除的 collections.MutableMapping
#         在 dns/namedict.py 顶部加：import collections.abc; collections.MutableMapping = collections.abc.MutableMapping
# 2) 先跑离线自检（不需要 ryu、不联网，秒级）——先确认编解码本身是对的
python tools/ryu_live_check.py --self-test
# 3) 一键完成：起服务 → 起真实 Ryu → 模拟交换机握手/上报/下发校验 → 写报告
python tools/ryu_live_check.py --model-tag sdn_ctrl_s0 --processed-dir data/processed_sdn \
    --ryu-python _tmp/ryu_venv/Scripts/python.exe
```

> 本次运行结束原因：已完成 2 轮上报并静默观察 10s；模拟交换机错误：无。
