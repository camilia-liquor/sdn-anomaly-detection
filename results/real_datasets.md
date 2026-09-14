# 真实数据集清单与体检（能支撑哪种评测）

> 由 `python tools/fetch_datasets.py --profile` 生成。数据集本身**不入版本库**（`data/real/` 已忽略），本文件是它们的来源、许可与「能/不能支撑什么评测」的记录。

特征基准：`data/processed/meta.json` 的 71 维（CICFlowMeter 全称命名）。特征映射沿用 `tools/cross_dataset.py` 已验证的三级策略（显式映射表 → 同名 → 词元集合）。


## 1. 总览

| 数据集 | 行数 | 列数 | 标签数 | 时间列 | 精确重复率 | 特征映射覆盖 |
|---|---|---|---|---|---|---|
| `cic_ids2017` | 5,950,088 | 85 | 19 | Timestamp | 0.0836 | 100.0% |
| `cic_ids2018` | 3,145,725 | 80 | 8 | Timestamp | 0.3251 | 100.0% |
| `unsw_nb15` | 257,673 | 45 | 10 | — | 0.3885 | 0.0% |
| `ton_iot` | 211,043 | 44 | 10 | — | 0.5726 | 0.0% |
| `cic_iot2023` | 2,058,836 | 39 | 9 | — | 0.2415 | 0.0% |

## 2. 逐数据集

### cic_ids2017　CIC-IDS2017（原始 MachineLearningCSV，逐日 CICFlowMeter 结果）

- 来源：Hugging Face `bencorn/CICIDS2017`（镜像自原始发布方）
- 许可：CIC / University of New Brunswick，学术研究用途；引用 Sharafaldin et al., 2018
- 选它的理由：逐日抓包（周一~周五）、BENIGN + 7 类攻击；列名与本工程 71 维特征族一致，是唯一同时支持「多类」与「按时序/在线」评测的公开集。MachineLearningCSV 是 78 特征压缩版（无时间列），GeneratedLabelledFlows 带 Timestamp，在线告警口径（检测时延/误报每小时）必须用后者
- 规模：5,950,088 行 / 85 列，标签来源：数据列 `Label`（19 类），时间列 `Timestamp`
- 精确重复率（数值特征列）：**0.0836**
- 特征映射：**71/71（100.0%）**，方法分布 {'exact': 43, 'explicit': 28}
- **能支撑的评测**：
  - ✅ 二分类（正常/攻击） —— 数据列 `Label`
  - ✅ 多分类 —— 标签取值 19 个
  - ✅ 可直接用本工程 71 维模型 —— 特征映射命中 71/71
  - ✅ 时序 / 在线评测（告警级） —— 时间列可解析且（多数文件）行序即时间序 → 可直接做在线告警口径评测
  - ✅ 可做无泄漏切分 —— 精确重复率 0.084（重复率低）
- ⚠️ 镜像质量备注（用前必读）：
  - MachineLearningCSV（78 特征版）与 GeneratedLabelledFlows（含 Timestamp 与流标识版）是**同一批流量的两种格式**，上表行数是两者之和，真实规模约为其一半；使用时只应二选一，不可合并计数。
  - 周四下午（Infiltration）当日存在缺失标签（`nan`），归类前必须剔除或单独处理，否则「19 类」里会多出一个假类别。
  - 标签字符串存在编码变体（如 Web Attack 的连字符在不同文件里不同），聚合成 8 大类前必须先做标签归一化。
- 标签 `Label`（19 类）：BENIGN=4,546,194、DoS Hulk=462,146、PortScan=317,860、nan=288,602、DDoS=256,054、DoS GoldenEye=20,586、FTP-Patator=15,876、SSH-Patator=11,794、DoS slowloris=11,592、DoS Slowhttptest=10,998、Bot=3,932、Web Attack  Brute Force=1,507…

### cic_ids2018　CSE-CIC-IDS2018（逐日 CICFlowMeter CSV，先取 3 天）

- 来源：Hugging Face `c01dsnap/CIC-IDS2018`（镜像自原始发布方）
- 许可：CIC / UNB + AWS，学术研究用途；引用 Sharafaldin et al., 2019
- 选它的理由：与 2017 同 schema 但实验床、攻击工具、年份都不同 → 检验「同族特征跨实验床」的迁移性
- 规模：3,145,725 行 / 80 列，标签来源：数据列 `Label`（8 类），时间列 `Timestamp`
- 精确重复率（数值特征列）：**0.3251**
- 特征映射：**71/71（100.0%）**，方法分布 {'explicit': 71}
- **能支撑的评测**：
  - ✅ 二分类（正常/攻击） —— 数据列 `Label`
  - ✅ 多分类 —— 标签取值 8 个
  - ✅ 可直接用本工程 71 维模型 —— 特征映射命中 71/71
  - ✅ 时序 / 在线评测（告警级） —— 时间列可解析但**当前行序不是时间序** → 必须先按时间列排序再切分，否则时序评测无效
  - ❌ 可做无泄漏切分 —— 精确重复率 0.325（必须先按特征去重，否则同文件内切分必然泄漏）
- ⚠️ 镜像质量备注（用前必读）：
  - 镜像里三个文件的样本数完全相同（都是 1,048,575 = 2^20−1，即 Excel 行数上限），疑似经表格工具导出时被截断；使用前应与原始发布方的行数核对，并按截断风险下调结论强度。
- 标签 `Label`（8 类）：Benign=2,110,475、DoS attacks-Hulk=461,912、FTP-BruteForce=193,360、SSH-Bruteforce=187,589、DoS attacks-SlowHTTPTest=139,890、DoS attacks-GoldenEye=41,508、DoS attacks-Slowloris=10,990、Label=1

### unsw_nb15　UNSW-NB15（官方训练/测试集 + 特征说明）

- 来源：Hugging Face `Mireu-Lab/UNSW-NB15`（镜像自原始发布方）
- 许可：UNSW Canberra，学术研究用途；引用 Moustafa & Slay, 2015
- 选它的理由：特征体系完全不同（Zeek/Argus 系，49 维）→ 检验「换一套特征族还能不能用」，且官方自带 train/test 划分，避免自己切分带来的口径争议
- 规模：257,673 行 / 45 列，标签来源：数据列 `attack_cat`（10 类），时间列 `无`
- 精确重复率（数值特征列）：**0.3885**
- 特征映射：**0/71（0.0%）**
  - 未映射（前 15）：Protocol、Flow Duration、Total Fwd Packets、Total Backward Packets、Total Length of Fwd Packets、Total Length of Bwd Packets、Fwd Packet Length Max、Fwd Packet Length Min、Fwd Packet Length Mean、Fwd Packet Length Std、Bwd Packet Length Max、Bwd Packet Length Min、Bwd Packet Length Mean、Bwd Packet Length Std、Flow Bytes/s
- **能支撑的评测**：
  - ✅ 二分类（正常/攻击） —— 数据列 `attack_cat`
  - ✅ 多分类 —— 标签取值 10 个
  - ❌ 可直接用本工程 71 维模型 —— 特征映射命中 0/71 → 特征体系不同，需另建特征映射与模型，不能沿用现有权重
  - ❌ 时序 / 在线评测（告警级） —— 无可用时间列 → 只能做静态窗口分类（不得报告检测时延/误报每小时）
  - ❌ 可做无泄漏切分 —— 精确重复率 0.389（必须先按特征去重，否则同文件内切分必然泄漏）
- ⚠️ 镜像质量备注（用前必读）：
  - 该镜像的 train.csv = 82,332 行、test.csv = 175,341 行，与官方发布（训练 175,341 / 测试 82,332）相反，疑为文件名对调；引用具体数字前必须核对官方口径。
  - 特征为 Zeek/Argus 短名体系（45 列），与本工程 CICFlowMeter 71 维不通用（映射 0/71）。
- 标签 `attack_cat`（10 类）：Normal=93,000、Generic=58,871、Exploits=44,525、Fuzzers=24,246、DoS=16,353、Reconnaissance=13,987、Analysis=2,677、Backdoor=2,329、Shellcode=1,511、Worms=174
- 标签 `label`（2 类）：1=164,673、0=93,000

### ton_iot　TON_IoT（网络流量合并表）

- 来源：Hugging Face `codymlewis/TON_IoT_network`（镜像自原始发布方）
- 许可：UNSW Canberra，学术研究用途；引用 Moustafa et al., 2020
- 选它的理由：IoT/IIoT 场景、含 label 与 type 两级标签、有时间列；与本工程的「校园网/企业网」场景形成对照
- 规模：211,043 行 / 44 列，标签来源：数据列 `type`（10 类），时间列 `无`
- 精确重复率（数值特征列）：**0.5726**
- 特征映射：**0/71（0.0%）**
  - 未映射（前 15）：Protocol、Flow Duration、Total Fwd Packets、Total Backward Packets、Total Length of Fwd Packets、Total Length of Bwd Packets、Fwd Packet Length Max、Fwd Packet Length Min、Fwd Packet Length Mean、Fwd Packet Length Std、Bwd Packet Length Max、Bwd Packet Length Min、Bwd Packet Length Mean、Bwd Packet Length Std、Flow Bytes/s
- **能支撑的评测**：
  - ✅ 二分类（正常/攻击） —— 数据列 `type`
  - ✅ 多分类 —— 标签取值 10 个
  - ❌ 可直接用本工程 71 维模型 —— 特征映射命中 0/71 → 特征体系不同，需另建特征映射与模型，不能沿用现有权重
  - ❌ 时序 / 在线评测（告警级） —— 无可用时间列 → 只能做静态窗口分类（不得报告检测时延/误报每小时）
  - ❌ 可做无泄漏切分 —— 精确重复率 0.573（必须先按特征去重，否则同文件内切分必然泄漏）
- ⚠️ 镜像质量备注（用前必读）：
  - 特征为 Argus/Zeek 体系（44 列），无时间列；与本工程 71 维不通用（映射 0/71），只能用于独立建模或标签级对照。
- 标签 `label`（2 类）：1=161,043、0=50,000
- 标签 `type`（10 类）：normal=50,000、backdoor=20,000、ddos=20,000、dos=20,000、injection=20,000、password=20,000、ransomware=20,000、scanning=20,000、xss=20,000、mitm=1,043

### cic_iot2023　CIC-IoT2023（CICFlowMeter 逐攻击族 CSV，取 Benign + 7 族 DDoS）

- 来源：Hugging Face `baalajimaestro/DDoS-CICIoT2023`（镜像自原始发布方）
- 许可：CIC / UNB（CIC-IoT2023），学术研究用途；引用 Neto et al., 2023
- 选它的理由：同为 CICFlowMeter 特征族（能被本工程 71 维模型直接消费），但攻击族与 2017 完全不同，且含 SlowLoris / 分片洪泛这类低速隐蔽攻击 → 可在真实数据上做「训练未见过的攻击族」检验
- 规模：2,058,836 行 / 39 列，标签来源：文件名推断（数据内无标签列）（9 类），时间列 `无`
- 精确重复率（数值特征列）：**0.2415**
- 特征映射：**0/71（0.0%）**
  - 未映射（前 15）：Protocol、Flow Duration、Total Fwd Packets、Total Backward Packets、Total Length of Fwd Packets、Total Length of Bwd Packets、Fwd Packet Length Max、Fwd Packet Length Min、Fwd Packet Length Mean、Fwd Packet Length Std、Bwd Packet Length Max、Bwd Packet Length Min、Bwd Packet Length Mean、Bwd Packet Length Std、Flow Bytes/s
- **能支撑的评测**：
  - ✅ 二分类（正常/攻击） —— 文件名推断（数据内无标签列）
  - ✅ 多分类 —— 标签取值 9 个
  - ❌ 可直接用本工程 71 维模型 —— 特征映射命中 0/71 → 特征体系不同，需另建特征映射与模型，不能沿用现有权重
  - ❌ 时序 / 在线评测（告警级） —— 无可用时间列 → 只能做静态窗口分类（不得报告检测时延/误报每小时）
  - ❌ 可做无泄漏切分 —— 精确重复率 0.241（必须先按特征去重，否则同文件内切分必然泄漏）
- ⚠️ 镜像质量备注（用前必读）：
  - 数据里没有标签列，攻击族由文件名给出（工程按文件名推断标签并明确标注来源）。
  - 特征为 CICIoT2023 的 39 维（Header_Length / Tot sum / flag 计数 …），与本工程 71 维不通用（映射 0/71）。
- 标签 `__from_filename__`（9 类）：BenignTraffic=657,946、DDoS-ICMP_Flood=268,010、DDoS-PSHACK_Flood=267,811、DDoS-UDP_Flood=266,605、DDoS-SYN_Flood=266,024、DDoS-TCP_Flood=265,776、DDoS-SlowLoris=23,426、DDoS-UDP_Fragmentation=22,542、DDoS-ICMP_Fragmentation=20,696


## 3. 使用纪律

- 同文件内切分只有在**显式去重**后才可信；重复率高的数据集（见上表）必须先去重再切分；
- 没有单调时间列的数据集**不得**用于「在线告警/检测时延」口径的评测，只能做静态窗口分类；
- 跨数据集结论必须写明训练域与评估域，不能用同域数字冒充跨域能力；
- 数据集仅用于研究评测，不随仓库分发；引用信息见上表。
