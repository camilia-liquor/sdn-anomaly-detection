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
python tests/test_core.py         # 单元测试：10 个用例，覆盖数据切分/窗口/泄漏防护
python tests/test_ryu_guard.py    # Ryu 适配层行为测试：25 个用例，无需安装 Ryu（见 3.5）
python tools/acceptance_check.py  # 交付验收：文件齐全性 + 指标复算 + 在线接口真实调用（56 项）
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

### 2.2 两种特征口径，按你的数据源选一个

这是**上线前最容易踩的坑**：模型的特征列必须和喂进来的字段完全对得上。

| 口径 | 需要的数据 | 预处理产物 | 模型 | 适用场景 |
|---|---|---|---|---|
| **`insdn`（84 列）** | CICFlowMeter / CIC-IDS2017 风格完整流特征 | `data/processed` | 主模型 | 有独立流量探针/镜像口，能算出完整流统计量 |
| **`sdn`（17 列）** | **OpenFlow 流表统计字段**（`packet_count`/`byte_count`/`flow_duration`/`tp_src`…），控制器可直接取到 | `data/processed_sdn` | `sdn_ctrl` | 只有控制器、没有探针的纯 SDN 环境 |

```bash
# 场景 A：完整流特征（精度最高）
python app/app.py --port 5000

# 场景 B：控制器原生字段（控制器能自给自足）
python app/app.py --port 5000 --processed-dir data/processed_sdn --model-tag sdn_ctrl
```

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
| `whitelist` | 永不处置的源 IP 列表 | — | **必须**含控制器管理口、网关、监控机 |

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

### 3.4 处置动作

* `observe`：只记录告警，不动流表；
* `throttle`：优先下发 OpenFlow Meter 限速（1 Mbps），交换机不支持 Meter 时自动退化为丢弃；
* `block`：下发 `ipv4_src=<攻击源>` 的高优先级空动作流表项，`hard_timeout=block-seconds` 自动过期。

### 3.5 诚实声明（重要）

* 本仓库**在没有 Ryu / OpenFlow 交换机的机器上完成了全部开发与验证**，
  因此 `integrations/ryu_guard.py` **未在真实交换机上实跑**，它属于「按 Ryu 官方 API 写好的适配层」。
* 但**控制器↔检测服务之间的完整闭环已经被实测验证**：`tools/controller_simulator.py`
  在无 Ryu 环境下扮演控制器，实测吞吐 **1,364~1,918 条流/秒**、判定时延
  **p50 29.8 ms / p95 44.7 ms / p99 45.7 ms**（剔除首批 TensorFlow 计算图构建的约 0.5 s 一次性开销）、
  端到端处置动作全部正确执行，报告见 `results/e2e_controller_report.md`。
* 另外，「没装 Ryu 就跑不了」**不等于「没法验证」**：`tests/test_ryu_guard.py` 用桩件替代 `ryu` 模块，
  **真的调用适配层的核心逻辑**，把 25 项接口契约与安全默认值锁死（见下表）。
  这个测试在编写过程中就抓出了两处**「账实不符」**的真缺陷并已修复——
  它们恰好是最危险的一类问题：台账显示"已处置"，交换机上其实什么都没发生。

  | 已验证项 | 说明 |
  |---|---|
  | 接口契约 | 适配层组装的流记录**覆盖控制器原生口径模型要求的全部 17 个特征名**（直接读 `data/processed_sdn/meta.json` 比对），字段名对不上服务会直接拒绝服务 |
  | 安全默认 | 不加 `--execute` 时**不发出任何流表消息**；审计台账如实标注「演练 / 未执行」 |
  | 白名单 | 白名单内的源 IP 即使开启真实处置也不下发规则 |
  | **`block` 真下发** | 必须观察到一条 `OFPFlowMod`（`priority=100`、`hard_timeout=60`、按攻击源 IP 匹配）——修掉了「只记账不下发」的缺陷 |
  | **`throttle` 真生效** | 除了 `OFPMeterMod`，还必须下发**引用该 Meter 的流表项**（`OFPInstructionMeter`）——只建 Meter 是不限速的，这也是一个已修缺陷 |
  | `observe` 无副作用 | 只记录，不下发任何规则 |
  | 审计可追溯 | 每条动作落盘，含源 IP / 类别 / 置信度 / 是否执行 / 有效期 |

* 请按照 3.2 的「先演练、后执行」顺序在你的环境中验证，不要一上来就 `--execute`。

---

## 4. 无 Ryu 环境下的端到端验证（推荐每次都跑）

```bash
# 用真实样本流回放给运行中的检测服务，模拟控制器的完整闭环
python tools/controller_simulator.py --rows 3000 --batch 50 --policy block --no-dry-run
```

产物：

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
- [ ] **审计可追溯**：`results/audit_alerts*.jsonl` 已纳入日志留存策略。
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
