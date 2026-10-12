# 部署与对接指南（DEPLOYMENT）

> 目标读者：拿到本仓库后想**真的跑起来、接到自己交换机上**的工程同学。
> 本文只讲「怎么部署、怎么对接、怎么保证不出事」，算法原理与实验数据见 `README.md` 与 `competition/02_技术报告.md`。

---

## 0. 三分钟看清这套系统由什么组成

```
   ┌──────────────────────┐   OpenFlow 流表统计    ┌────────────────────────────┐
   │  SDN 交换机 / Mininet │ ─────────────────────► │  Ryu 控制器                │
   │  (OpenFlow 1.3)      │ ◄───────────────────── │  integrations/ryu_guard.py │
   └──────────────────────┘   流表规则(丢弃/限速)   └────────────┬───────────────┘
                                                               │ HTTP POST /api/detect
                                                               │ （一批流记录 → 逐条判定 + 告警 + 处置建议）
                                                               ▼
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │  检测服务  app/app.py  +  src/service.py                                       │
   │  · 滑窗缓冲（T=16）× 阈值 × 连续确认  → 告警                                      │
   │  · 处置策略：observe / throttle / block（含白名单、演练模式、有效期）               │
   │  · 审计台账 JSONL（每条告警可追溯）    · Prometheus /metrics（可接现有监控）        │
   └───────────────────────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │  可视化大屏  http://<host>:<port>/   （离线可用，不依赖任何外网资源）             │
   └───────────────────────────────────────────────────────────────────────────────┘
```

三件产物、三条命令：

| 步骤 | 命令 | 说明 |
|---|---|---|
| ① 训练产出模型 | `python run_all.py` | 生成数据 → 预处理 → 训练 → 评估 → 出图；产物在 `data/processed`、`artifacts/`、`results/` |
| ② 起检测服务 + 大屏 | `python app/app.py --port 5000` | 同时提供大屏、`/api/*` 接口与 `/metrics` |
| ③ 接控制器 | `ryu-manager integrations/ryu_guard.py` | 采集流表统计 → 调 ② → 按策略下发规则 |

没有 Ryu 环境也能验证整条链路：`python tools/controller_simulator.py` 会扮演控制器，
把流记录推给检测服务、读回告警、执行处置，并写出 `results/e2e_controller_report.md`。

---

## 1. 环境要求

| 项目 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Windows 10/11、Linux、macOS 均可 | 本项目在 Windows 11 + Python 3.11.9 上完成全部实验 |
| Python | **3.10 – 3.11**（推荐 3.11.x） | TensorFlow 2.15 不支持 3.12+ |
| 内存 | ≥ 4 GB（训练）/ ≥ 1.5 GB（仅推理） | 完整数据集训练峰值约 2 GB |
| CPU | 任意 x86-64；推荐 4 核以上 | 本项目全程 CPU 训练，无需 GPU；有 GPU 会自动加速 |
| 磁盘 | ≥ 2 GB | 原始数据 90 MB、处理产物约 100 MB、模型约 1 MB |
| GPU（可选） | 任意支持 CUDA 11.8 的卡 | 不装也能跑，只是训练慢一些 |
| 网络 | **仅安装依赖时需要**；运行与演示全程离线 | 演示大屏不引用任何 CDN / 外部字体 |

### 1.1 安装（推荐用虚拟环境，避免污染系统 Python）

```bash
# 1) 创建并激活虚拟环境
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 2) 安装依赖（版本已锁定为实测通过的组合）
pip install -r requirements.txt

# 3) 自检：Python 版本、依赖、目录、数据、模型是否齐全
python tools/check_env.py
```

> 如果 `pip` 拉 TensorFlow 太慢或中断，用仓库自带的断点续传下载器：
> `python tools/fetch_wheel.py --package tensorflow==2.15.1`，
> 它会写入 `wheels/` 目录并在**已下载的字节数**上继续，下载完再 `pip install wheels/*.whl`。

### 1.2 三条快速验证（装完必跑）

```bash
python tools/smoke_test.py        # 冒烟：小样本全链路跑通（约 1 分钟，产物写在 _smoke 隔离目录）
python tests/test_core.py         # 单元测试：20 个用例，覆盖数据切分/窗口/泄漏防护/复核闭环
python tests/test_ryu_guard.py    # Ryu 适配层行为测试：58 个用例，无需安装 Ryu（见 3.5）
python tests/test_pcap_tools.py   # pcap 解析往返一致性：58 个用例，无需真实抓包
python tools/acceptance_check.py  # 交付验收：文件齐全性 + 指标复算 + 在线接口真实调用（130 项）
```

---

## 2. 生产部署（检测服务）

### 2.1 启动

```bash
# 默认：监听 127.0.0.1:5000，加载 run_all.py 记录的「主模型」
python app/app.py --port 5000

# 对外提供服务（注意：先用防火墙/安全组限制来源，见 5. 安全清单）
python app/app.py --host 0.0.0.0 --port 5000
```

#### 用环境变量配置（服务化部署推荐）

命令行参数都有对应的环境变量，因此 systemd / 容器 / Windows 服务里不用改代码或拼长命令行：

| 环境变量 | 对应参数 | 默认 | 说明 |
|---|---|---|---|
| `CIDA_HOST` | `--host` | `127.0.0.1` | 监听地址 |
| `CIDA_PORT` | `--port` | `5000` | 监听端口 |
| `CIDA_PROCESSED_DIR` | `--processed-dir` | `data/processed` | 预处理产物目录（决定特征口径） |
| `CIDA_MODEL_TAG` | `--model-tag` | 空（用主模型） | 指定模型子目录，如 `sdn_ctrl` |
| `CIDA_POLICY_ACTION` | — | `observe` | 初始处置动作：`observe`/`throttle`/`block`（非法值会被拒绝并回退为 `observe`） |
| `CIDA_DRY_RUN` | — | `1` | `1`=演练模式（只记录不改流表） |
| `CIDA_THRESHOLD` | — | 0.5 | 初始判定阈值 |
| `CIDA_CONFIRM_WINDOWS` | — | 3 | 初始连续确认窗口数 |
| `CIDA_WHITELIST` | — | 空 | 永不处置的源 IP，空格分隔 |
| `CIDA_UNKNOWN_THRESHOLD` | — | `0`（关闭） | 「未知形态」标记阈值（多分类概率熵），见 2.4 最后一节 |
| `CIDA_UNKNOWN_ACTION` | — | `throttle` | 未知形态的保守动作。**硬护栏：不允许升级为 `block`**，填了也会被降级 |
| `CIDA_OBSERVE_MARGIN` | — | `0.5` | 观察池下界比例（漏报留痕的起点），见 5. 安全清单 |
| `CIDA_API_TOKEN` | — | 空（放行） | 设置后写接口（改策略/写研判结论）必须带令牌；读接口保持开放 |
| `CIDA_AUDIT_MAX_BYTES` | — | `20000000` | 审计台账/回流文件单文件上限，超过即轮转 |
| `CIDA_AUDIT_BACKUPS` | — | `3` | 轮转保留的历史文件数 |
| `CIDA_REVIEW_EVIDENCE` | — | `1`（开） | 是否把研判队列的窗口特征落盘（`review_evidence.jsonl`）。关掉则人工标注无法用于重训 |
| `CIDA_AUTOTIGHTEN` | — | `0`（**关**） | 自动放松（自动降级）：人工复核结论以 `normal` 为主时自动上调判定阈值。**默认关**，开启前先看 5. 安全清单 |
| `CIDA_AUTOTIGHTEN_MIN_LABELED` | — | `10` | 自动放松的样本门槛：人工已给结论至少多少条才允许调整 |
| `CIDA_AUTOTIGHTEN_NORMAL_RATIO` | — | `0.8` | 自动放松的占比门槛：结论里 `normal` 占比达到多少才动手 |
| `CIDA_AUTOTIGHTEN_STEP` | — | `0.05` | 每次上调的幅度（0.05 = 阈值 +0.05） |
| `CIDA_AUTOTIGHTEN_MAX` | — | `0.9` | 阈值上限（硬护栏：不允许推到 1.0，那等于关掉检测） |

> ⚠️ **两个"演练开关"互不相关，必须一起设置 —— 否则会出现"账实不符"。**
> 检测服务的 `CIDA_DRY_RUN` 与 Ryu 适配层的 `--execute` 是**两个独立开关**：
>
> | 开关 | 位置 | 作用 |
> |---|---|---|
> | `CIDA_DRY_RUN=0` | 检测服务 | 服务不再把自己标成"演练"；其审计台账的 `executed` 会为真 |
> | `--execute` | `integrations/ryu_guard.py` | **真正向交换机下发规则**；不加就是只打印 |
>
> 只关服务侧、忘了给适配层加 `--execute`，结果是**服务台账写着"已执行"，而交换机上什么都没变**——
> 这正是适配层自己踩过并修过的"账实不符"，只是这次发生在两个组件之间。
> **权威记录以适配层台账 `results/audit_alerts_ryu.jsonl` 的 `executed` 为准**（服务侧那个字段只表示
> "本服务已给出可下发建议且非演练"）。
>
> ⚠️ **白名单同样有地方要两处配**：服务的 `CIDA_WHITELIST` 与适配层的 `--whitelist`。
> 两者互为纵深防御（服务对白名单源只给 `observe`，适配层再拦一次），但**只配一处等于少一层保险**，
> 上线检查时请两处都确认（服务启动与适配层启动都会就空白名单告警）。

环境变量只决定**启动时的初始策略**，运行时仍可用 `POST /api/policy` 热更新。

#### 现成的部署样例（`deploy/`）

| 文件 | 用途 |
|---|---|
| `deploy/cida-detect.service` | systemd 单元（固定工作目录、崩溃自动重启、日志进 journald、最小权限加固） |
| `deploy/cida-detect.env` | 环境变量模板（两种特征口径二选一，注释里写清了何时选哪个） |
| `deploy/nginx-cida.conf` | 反向代理 + Basic 认证；`/metrics` 只对监控机开放 |
| `deploy/install-windows-service.ps1` | 用 NSSM 注册成 Windows 服务，含启动后的健康检查 |
| `deploy/README.md` | 四步上线顺序（**先 observe + dry_run，确认后再打开处置**） |

```bash
# Linux
sudo cp deploy/cida-detect.service /etc/systemd/system/ && sudo cp deploy/cida-detect.env /etc/default/cida-detect
sudo systemctl enable --now cida-detect && journalctl -u cida-detect -f

# Windows（需要 nssm.exe 在 PATH 中）
.\deploy\install-windows-service.ps1 -ProjectRoot "D:\sdn_demo" -Port 5000
```

### 2.2 三种数据来源，按你的环境选一个

这是**上线前最容易踩的坑**：模型的特征列必须和喂进来的字段完全对得上。

| 数据来源 | 需要什么 | 预处理产物 / 模型 | 适用场景 |
|---|---|---|---|
| **`insdn`（84 列）** | CICFlowMeter / CIC-IDS2017 风格完整流特征 | `data/processed` + 主模型 | 有独立流量探针/镜像口，能算出完整流统计量 |
| **控制器原生字段（17 列）** | **OpenFlow 流表统计**（`packet_count`/`byte_count`/`flow_duration`/`tp_src`…），控制器可直接取到 | `data/processed_sdn` + `sdn_ctrl` | 只有控制器、没有探针的纯 SDN 环境 |
| **真实抓包 pcap（17 列）** | 一段 `tcpdump` 抓包文件，**不需要控制器在场** | 先用 `tools/pcap_to_flows.py` 转成 17 列 CSV，再喂给上面的控制器口径模型 | 离线取证、事后复盘、没有 SDN 控制器的现场 |

```bash
# 场景 A：完整流特征（精度最高）
python app/app.py --port 5000

# 场景 B：控制器原生字段（控制器能自给自足）
python app/app.py --port 5000 --processed-dir data/processed_sdn --model-tag sdn_ctrl

# 场景 C：从一段抓包开始（离线取证）
tcpdump -i eth0 -s 0 -w capture.pcap                 # 抓包（或直接用历史 pcap）
python tools/pcap_to_flows.py --input capture.pcap --output flows.csv
python tools/predict_cli.py --input flows.csv        # 批量出判定
# 也可以把 flows.csv 逐条推送进 /api/detect，走与控制器完全相同的在线链路
```

> `pcap_to_flows.py` **纯标准库实现、不依赖 scapy**：支持经典 `.pcap`（微秒/纳秒、大小端）
> 与 `.pcapng`（SHB/IDB/EPB、`if_tsresol` 二进制时基），支持 VLAN/QinQ、RAW IP / LINUX_SLL / LINUX_SLL2
> （覆盖 `tcpdump -i any` 的 cooked 抓包），按空闲/硬超时切分流（与 OpenFlow 流表项的 idle/hard timeout 等价）。
> 往返一致性已实测：自造 800 条流的抓包解析回 800 条，**17 个特征 × 800 条流逐位一致**
> （见 `results/pcap_roundtrip_report.md`、测试 `tests/test_pcap_tools.py` 58 项）。
>
> ⚠️ 局限：我们只算这 17 个控制器口径字段，**与 CICFlowMeter 的完整口径（含子流/Bulk/IAT 统计量）不同**，
> 因此该路径的精度上限就是控制器原生口径的水平（宏 F1 0.8441，见 `results/summary_sdn_ctrl.md`）。
> 要更高精度仍需完整流特征口径。

> 两种口径的实测差异、以及「字段不够时性能会掉多少、怎么补回来」，
> 见 `results/summary_sdn_ctrl.md` 与 `results/operating_point_sdn_ctrl.md`；
> 大屏与 `/api/detect` 会**同时**切换到对应口径（`--processed-dir` 一改，两边一起变），
> 避免出现「页面显示的模型」和「实际判定的模型」不是同一个的低级事故。

### 2.3 接口一览（都是 JSON，无第三方 SDK 依赖）

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | 可视化大屏 |
| GET | `/api/overview` | 模型信息、指标、告警确认参数（大屏用） |
| POST | `/api/detect` | **核心接口**：推一批流记录 → 逐条判定 + 新告警 + 处置建议 |
| GET | `/api/alerts?limit=100` | 告警审计台账（倒序） |
| GET/POST | `/api/policy` | 查询 / 热更新策略（阈值、确认次数、动作、白名单、演练模式），**无需重启** |
| GET | `/api/stats` | 运行统计（吞吐、时延分位、各类告警计数） |
| GET | `/metrics` | Prometheus 文本格式，可直接被现有监控抓取 |

### 2.4 策略热更新（运维最常用的动作）

```bash
# 查看当前策略
curl http://127.0.0.1:5000/api/policy

# 切换到「只告警不处置」（演练）——上线第一步永远是这句
curl -X POST http://127.0.0.1:5000/api/policy \
     -H "Content-Type: application/json" \
     -d "{\"action\":\"observe\",\"dry_run\":true}"

# 确认无副作用后，打开阻断（含白名单，白名单内的源 IP 永不处置）
curl -X POST http://127.0.0.1:5000/api/policy \
     -H "Content-Type: application/json" \
     -d "{\"action\":\"block\",\"dry_run\":false,\"whitelist\":[\"10.0.0.1\",\"10.0.0.254\"]}"

# 提高门槛（更保守）：阈值 0.9 + 连续确认 3 个窗口
curl -X POST http://127.0.0.1:5000/api/policy \
     -H "Content-Type: application/json" \
     -d "{\"threshold\":0.9,\"confirm_windows\":3}"
```

**策略参数说明**

| 参数 | 含义 | 调大的后果 | 建议 |
|---|---|---|---|
| `threshold` | 判定为攻击的概率阈值 | 误报↓、漏报↑、时延↑ | 0.5 起步；误报超标时先升到 0.9 |
| `confirm_windows` | 需要连续多少个窗口都超阈值才报警 | 误报↓↓、时延↑ | **默认 3**；这是把误报压到可用区间的关键旋钮 |
| `action` | `observe` / `throttle` / `block` | — | 上线首周用 `observe` |
| `dry_run` | `true` 时只记录不真改流表 | — | 演练必开 |
| `unknown_threshold` | 多分类概率**熵**超过它就标记为「未知形态」（`Unknown`） | 越高越保守（越少被标未知） | **默认 0 = 关闭**；要用先按下面标定 |
| `whitelist` | 永不处置的源 IP 列表 | — | **必须**含控制器管理口、网关、监控机 |
| `autotighten` | 自动放松（自动降级）总开关：人工复核结论以 `normal` 为主时自动上调阈值 | 越长越容易自动变钝 | **默认 `false`**；要开先满足 5. 安全清单的前置条件 |
| `autotighten_min_labeled` / `_normal_ratio` | 触发门槛：已研判结论条数 / 其中 `normal` 占比 | 越大越难触发（更保守） | 默认 10 条 / 80% |
| `autotighten_step` / `_max_threshold` | 每次上调幅度 / 阈值上限 | 越大越快变钝 | 默认 +0.05 / 上限 0.9 |

#### 「未知形态」标记怎么开（可选能力）

模型只在 8 类上训练过。当流量出现"模型没见过这种形态"时，如果什么都不做，它会**自信地把它归到某个已知类**
（见 `results/openset_report.md`：留出整类攻击时 100% 被安上已知标签）。为把这条结论落成可用能力，
服务提供了 `unknown_threshold` 旋钮：

```bash
# ① 先在**正常窗口**上按误报预算标定阈值（不需要重训，约 5 秒）
python tools/openset_calibrate.py
#    → results/openset_thresholds.md / .json（含代价表：正常被判未知 = 新增误报；
#      已知攻击被判未知 = 细粒度标签丢失，但**告警仍然会发**）

# ② 打开（用标定出来的阈值，例如 0.3914）
curl -X POST http://127.0.0.1:5000/api/policy -H "Content-Type: application/json" \
     -d "{\"unknown_threshold\": 0.3914}"

# ③ 关闭：设回 0；负值会被自动夹到 0（不会误配成"把所有窗口标成未知"）
```

开启后：熵超阈值的窗口其判定类别显示为 `Unknown（未知形态）`，**照常告警与给出处置建议**，
只是不再声称具体类别——运维据此把它送进"待研判"队列。同时在 `/metrics` 里新增
`sdn_ids_unknown_windows_total` 与 `sdn_ids_unknown_threshold`：
**未知占比本身就是一条免费的漂移指标**，突然升高通常意味着新攻击出现或流量分布变化。

> ⚠️ 语义边界（不要误解成"未知攻击检测器"）：生产模型见过全部 8 类，所以这个标记的含义是
> 「**模型没见过这种形态**」（新颖性/低置信）。它的能力边界由 `tools/openset_eval.py` 的留出实验给出：
> 跨场景 AUROC ≈ 0.978，但**留出的攻击越像正常流量，低误报预算下越检不出**（留出 Botnet 时 1% 预算 44.3%、
> 0.1% 预算 0.9%）。因此**不要用它替代二分类告警**，它是叠加在告警之上的一层"要不要人工研判"的提示；
> 换网络/换采集口径后阈值必须重新标定。

#### 自动放松（自动降级）怎么开（可选能力，默认关）

系统默认**只会自动升级**（同源反复告警把动作从限速抬到阻断），长期高误报没有自愈路径。复核闭环建好后，
人工结论里"判为正常"的占比本身就是**工作点是否偏松**的直接证据——自动放松就是把这份证据接进阈值：

```bash
# ① 打开（默认关；门槛见 deploy/cida-detect.env 的 CIDA_AUTOTIGHTEN_*）
curl -X POST http://127.0.0.1:5000/api/policy -H "Content-Type: application/json" \
     -d "{\"autotighten\": true}"

# ② 看它有没有动、为什么没动（"最近一次判定"直接可查）
curl -s http://127.0.0.1:5000/api/stats | python -c "import json,sys;print(json.load(sys.stdin)['autotighten_last'])"

# ③ 关闭：设回 false（字符串 "0"/"false" 也认）
```

规则（写死在代码里，不接受"看着差不多"）：已研判结论 ≥ `autotighten_min_labeled` 条、其中 `normal` 占比
≥ `autotighten_normal_ratio` → 阈值上调 `autotighten_step`；**上限** `autotighten_max_threshold`（默认 0.9，
不允许推到 1.0——那等于把检测关掉）。它**只认人工结论**：未研判的告警不计入，模型自己说"我误报了"也不计入。

> ⚠️ 这是**系统自己改工作点**的动作，所以默认关、有门槛、有上限、每次调整都写审计台账
> （`results/audit_alerts.jsonl` 里 `event=autotighten` 的记录带 `n_labeled` / `n_normal` / 前后阈值）。
> 与之配套的观测指标：`sdn_ids_review_labeled_normal_total`（人工判为误报的累计条数）、
> `sdn_ids_autotighten_total`（自动上调次数）、`sdn_ids_threshold`（当前阈值）。
> 若这三个数一起往上走而漏报也在涨，说明该重新标定工作点，而不是继续让它放松。

### 2.5 持续运行建议

* **进程守护**：Windows 用 `nssm`/计划任务，Linux 用 `systemd`，关键字都是「工作目录 = 项目根目录」。
* **日志**：服务写 `logs/`；告警审计写 `results/audit_alerts.jsonl`（一行一条 JSON，可直接被 ELK 采集）。
* **监控**：把 `/metrics` 加进 Prometheus，重点看 `cida_*` 里的窗口数、告警数、时延分位。
* **模型更新**：重新训练后把新的 `model.keras` 与 `data/processed` 一起替换，然后 `POST /api/policy` 或重启服务；
  **不要只换模型不换 scaler**——两者的特征顺序必须来自同一份 `meta.json`。

---

## 3. 接入 Ryu 控制器（真实 SDN 环境）

`integrations/ryu_guard.py` 是一个 Ryu 应用：周期性拉取流表统计 → 组装成模型需要的字段 →
POST 给检测服务 → 按返回的处置建议下发 OpenFlow 规则。

### 3.1 前置条件

* Ryu（`pip install ryu`，或 `apt install python3-ryu`）与一台支持 OpenFlow 1.3 的交换机 / Mininet 拓扑。
* 检测服务已按 2.2 的**场景 B** 启动（控制器字段口径），并监听在控制器可达的地址上。

### 3.2 启动

```bash
# ① 先起检测服务（控制器原生字段口径）
python app/app.py --host 0.0.0.0 --port 5000 --processed-dir data/processed_sdn --model-tag sdn_ctrl

# ② 起控制器——默认是「演练模式」：只打印将要下发的规则，不动流表
ryu-manager integrations/ryu_guard.py \
    --detect-url http://127.0.0.1:5000 \
    --whitelist 10.0.0.1 10.0.0.254 \
    --block-seconds 60

# ③ 观察日志确认判定与规则都正确后，再打开真实处置
ryu-manager integrations/ryu_guard.py --execute --whitelist 10.0.0.1 10.0.0.254 --block-seconds 60
```

### 3.3 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--detect-url` | `http://127.0.0.1:5000` | 检测服务地址 |
| `--execute` | 关（演练） | 打开后才真正下发流表规则 |
| `--block-seconds` | `60` | 处置有效期，到期由交换机自动解除（避免误封长期影响业务） |
| `--whitelist` | 空 | 永不处置的源 IP。**上线前必须填**，至少包含控制器管理口与网关 |
| `--audit` | `results/audit_alerts_ryu.jsonl` | 控制器侧审计台账 |
| `--verify-readback` | 关 | **下发后回读校验**：再发 `OFPFlowStatsRequest` 把流表读回来，确认该源上确实挂着**动作正确**的规则 |
| `--verify-timeout` | `2.0` | 回读等待回复的超时（秒）。超时按"未确认"处理（**不确定 ≠ 生效**） |
| `--no-rollback-on-fail` | 回滚 | 缺省校验失败/超时即回滚；加上本开关则改为"只记账、不回滚"（由运维按自己的风险偏好选 fail-open / fail-closed） |

> ⚠️ **`--verify-readback` 为什么默认关**：`executed=true` 只代表"消息已发出"，**不代表交换机上真的生效**
> （权限、table 满、指令不支持、meter 超限、中间设备吞包都可能让它落空）。默认开启会让"不确定"变成"回滚"，
> 在响应慢或不支持带 match 的 stats 请求的交换机上，反而把已经生效的处置撤掉。所以它需要显式打开。开启后：
> 校验结论单独写一条 `event=verify` 台账（含 `ok` / `found` / `rule_kinds` / `rolled_back` / 原因），
> 心跳里的 `verified` / `rollbacks` / `verify_pending` 可在 `python tools/cida.py status` 看到。

### 3.4 处置动作

* `observe`：只记录告警，不动流表；
* `throttle`：优先下发 OpenFlow Meter 限速（1 Mbps），交换机不支持 Meter 时自动退化为丢弃；
* `block`：下发 `ipv4_src=<攻击源>` 的高优先级空动作流表项，`hard_timeout=block-seconds` 自动过期。

### 3.5 诚实声明（重要）

* `integrations/ryu_guard.py` 已在**真实 `ryu-manager`（Ryu 4.34）+ 真实 OF1.3 线协议**下实跑：
  16/16 项检查通过、Ryu 侧 0 条报文解析错误，攻击轮实收按源 IP 匹配的 DROP 规则、正常轮 0 条，
  审计台账 `executed=true` 与报文对账一致（`results/ryu_live_check.md`，复现命令见该文件末节）。
  **仍未覆盖**：物理交换机、真实攻击流量、厂商私有扩展。
* 另外，**控制器↔检测服务之间的完整闭环也已被实测验证**：`tools/controller_simulator.py`
  在无 Ryu 环境下扮演控制器，实测吞吐 **1,364~1,918 条流/秒**、判定时延
  **p50 29.8 ms / p95 44.7 ms / p99 45.7 ms**（剔除首批 TensorFlow 计算图构建的约 0.5 s 一次性开销）、
  端到端处置动作全部正确执行，报告见 `results/e2e_controller_report.md`。
* 另外，「没装 Ryu 就跑不了」**不等于「没法验证」**：`tests/test_ryu_guard.py` 用桩件替代 `ryu` 模块，
  **真的调用适配层的核心逻辑**，把 58 项接口契约与安全默认值锁死（见下表）。
  这个测试在编写过程中就抓出了两处**「账实不符」**的真缺陷并已修复——
  它们恰好是最危险的一类问题：台账显示"已处置"，交换机上其实什么都没发生。
  本轮又补上「下发后回读校验 + 失败回滚」的 22 项 check（默认关、回读通过、动作不符回滚、
  超时回滚、限速连 Meter 一起删、回滚后解开幂等窗口、待校验表有界）。

  | 已验证项 | 说明 |
  |---|---|
  | 接口契约 | 适配层组装的流记录**覆盖控制器原生口径模型要求的全部 17 个特征名**（直接读 `data/processed_sdn/meta.json` 比对），字段名对不上服务会直接拒绝服务 |
  | 安全默认 | 不加 `--execute` 时**不发出任何流表消息**；审计台账如实标注「演练 / 未执行」 |
  | 白名单 | 白名单内的源 IP 即使开启真实处置也不下发规则 |
  | **`block` 真下发** | 必须观察到一条 `OFPFlowMod`（`priority=100`、`hard_timeout=60`、按攻击源 IP 匹配）——修掉了「只记账不下发」的缺陷 |
  | **`throttle` 真生效** | 除了 `OFPMeterMod`，还必须下发**引用该 Meter 的流表项**（`OFPInstructionMeter`）——只建 Meter 是不限速的，这也是一个已修缺陷 |
  | `observe` 无副作用 | 只记录，不下发任何规则 |
  | 审计可追溯 | 每条动作落盘，含源 IP / 类别 / 置信度 / 是否执行 / 有效期 |
  | **回读校验与回滚**（22 项） | 默认关时**不发**任何额外流表请求；开启后：回读确认生效 → 计入 `verified` 且不回滚；**动作不符**（本该丢弃却是限速）或**超时** → `DELETE_STRICT` 回滚、限速连 Meter 一起删、**解开幂等窗口**以便重试、写 `event=verify` 台账；校验回复不得被当成新一轮流量喂给检测器（靠 xid 分流）；待校验表有硬上限 |

* 请按照 3.2 的「先演练、后执行」顺序在你的环境中验证，不要一上来就 `--execute`。

---

## 4. 无 Ryu 环境下的端到端验证（推荐每次都跑）

```bash
# ① 一条命令看一场攻防（最快建立信心：约 10 秒演完，自动出事件报告）
python tools/live_demo.py
python tools/live_demo.py --attack Probe --speed 50   # 换攻击类型 / 调演示节奏
python tools/live_demo_campaign.py --speed 0        # 7 类攻击连演 → 每类检出时延对比表
#   产物：results/live_demo_incident.md（时间线 + 关键指标 + 逐窗口与离线评估的一致性核对）

# ② 用真实样本流回放给运行中的检测服务，模拟控制器的完整闭环
python tools/controller_simulator.py --rows 3000 --batch 50 --policy block --no-dry-run
```

`live_demo.py` 的价值在于**它把"系统在做事"变成肉眼可见的 10 秒**：安静基线 → 攻击爆发 → 告警 →
处置下发 → 攻击结束 → 风险回落，全程用真实留出测试流、真实判定、真实时间戳。
它还有一个工程细节值得注意：剧本在"攻击后恢复"阶段有一次位置跳转（等价于上游数据中断），
此时它会**重置滑窗缓冲**——否则中断前后的行会被拼进同一个窗口，产生现实中不存在的窗口。
重置后 227 个窗口全部是真实连续窗口，与离线评估逐窗口一致（最大 |Δp| 4.94e-05、判定类别不一致 0）。

产物：

* `results/live_demo_incident.md` —— 事件时间线、检测时延、告警/误报数、处置动作分布；
* `results/e2e_controller_report.md` / `.json` —— 吞吐、时延分位、告警命中/误报数、处置动作统计；
* `results/audit_alerts.jsonl` —— 每条告警的完整台账（时间、源 IP、类别、置信度、动作）。

这一步的意义：**它把「算法」变成了「链路」**——评委/运维能看到的不再是 F1 分数，
而是「N 条流进来、M 条告警出去、每一条都能追到是谁在什么时候因为什么被拦了」。

---

## 5. 安全清单（上线前逐条打勾）

- [ ] **白名单已配置**：控制器管理口、网关、监控/跳板机、任何自动化运维源 IP 全部列进 `whitelist`。
- [ ] **演练先行**：`dry_run=true` 或 `--execute` 缺省状态下至少观察一个完整业务周期。
- [ ] **处置有超时**：`--block-seconds` 不为 0（否则误封不会自动解除）。
- [ ] **不对外裸奔**：`/api/detect`、`/api/policy` 只在管理网可达；需要跨网时套反向代理 + 认证。
- [ ] **写接口鉴权（纵深防御）**：设置 `CIDA_API_TOKEN`。nginx Basic 认证是第一层，应用级令牌是第二层——
      只靠反代时，一旦反代配置失误或内网可达，任何人都能关掉演练模式、把建议动作改成真实封禁。
      只读接口不校验，故大屏前端无需持令牌。
- [ ] **审计可追溯**：`results/audit_alerts.jsonl`（含轮转出的 `.1/.2/.3`）、`results/review_feedback.jsonl`
      与 `results/review_evidence.jsonl` 已纳入日志留存策略。
- [ ] **回流数据可重训**：确认 `CIDA_REVIEW_EVIDENCE=1`——只存结论不存特征，人工标注完也训不了模型。
- [ ] **磁盘不会写满**：确认 `CIDA_AUDIT_MAX_BYTES` / `CIDA_AUDIT_BACKUPS` 与本单位日志留存要求一致
      （台账与回流文件只追加不删除，**不轮转会把磁盘写满**）。
- [ ] **内存不会只涨不降**：确认长期运行下 RSS 稳定（告警/观察池/回看缓冲/研判队列均为有界容器，
      累计量走计数器 `alerts_total`，可在 `/metrics` 观察）。
- [ ] **自动放松有人盯着**：`CIDA_AUTOTIGHTEN` 保持 `0`（默认关），除非满足两个前提——① 复核结论确实有人
      认真在看（不是应付式点"正常"）；② 已接受"阈值会被系统自己上调"。开启后每次调整都写进审计台账
      （`event=autotighten`）并暴露为 `sdn_ids_autotighten_total` / `sdn_ids_threshold`，**值班室要能发现"检测在变钝"**。
- [ ] **降级预案**：检测服务不可达时，控制器按 3.2 的实现只会记日志、不改流表（**fail-open，不会因为算法挂了把网络搞断**）。
- [ ] **模型与 scaler 同源**：替换模型时 `meta.json`、`scaler.joblib`、`model.keras` 三者一起换。
- [ ] **数据合规**：流入检测服务的流记录若含真实 IP，请确认符合所在单位的日志留存与隐私要求。

---

## 6. 常见问题（troubleshooting）

| 现象 | 原因 | 处理 |
|---|---|---|
| 大屏打开是「数据未就绪」 | 还没跑 `run_all.py`，`data/processed` 为空 | 先 `python run_all.py --smoke` 快速验证，再跑完整流程 |
| `/api/detect` 返回 503 | 检测服务没加载到模型/TensorFlow | 看启动日志里的 `reason`；确认 `artifacts/main_model.keras`（或 `--model-tag` 指定的模型）存在 |
| 页面数值和接口对不上 | `--processed-dir` 与模型口径不一致 | 用 `--processed-dir` + `--model-tag` **成对**指定，服务与大屏会自动同步口径 |
| 误报太多 | 工作点太激进 | 先加 `confirm_windows`（3 → 5），再考虑升 `threshold`；对照 `results/ops_impact.md` 选点 |
| 漏报变多 | 阈值/确认次数调过头 | 按 `results/operating_point.md` 的误报预算列回退一档 |
| 控制器收到的 `src_ip` 为空 | 流表匹配项里没有 `ipv4_src`（如二层流） | 在 `ryu_guard.py` 里扩展匹配字段映射，或对这类流只做 `observe` |
| 特征列对不上、报「缺少特征」 | 上游采集脚本的列名与模型 `feature_names` 不一致 | 用 `--processed-dir`/`--schema` 切到匹配的口径，或按 `src/data_pipeline.py` 的别名机制补映射 |
| TensorFlow 启动刷一堆 warning | TF 的常规 INFO/WARNING | 已在代码里抑制；不影响结果。PowerShell 里可能显示 `exit code: 1`，以日志结论与产物文件为准 |
| `n_jobs=-1` 报 WinError 5 | Windows 沙箱/权限限制进程池 | 本项目已自动回退单进程；也可显式设 `baseline_n_jobs=1` |

---

## 7. 复现实验（评审/答辩用）

```bash
python run_all.py --seeds 0,1,2                    # 主实验：3 个随机种子
python run_all.py --ablation --skip-train          # 窗口/分支/注意力等消融
python run_all.py --baselines --skip-train         # 与 RF/ET/LR/MLP/CNN/BiLSTM 对比
python tools/make_report.py                        # 自动生成技术报告素材与汇总表
python tools/acceptance_check.py                   # 交付自检（文件齐全 + 指标复算）
```

全部随机种子、数据切分方式、标准化器拟合范围都在 `src/config.py` 与 `src/utils.set_seed()` 里固定，
同一环境重复运行结果可复现。
