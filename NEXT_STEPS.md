# NEXT_STEPS · 续做指南

> **给下一个接手的人或 AI 会话**：这份文件是**权威起点**。它不依赖记忆系统
> （项目记忆需要主动读取，关键记忆轨需要人工确认），只看这一个文件就能接着干。
> 最后更新：2026-10-13 ｜ 当前分支 `master` ｜ 工作区干净 ｜ 门禁 **130 项通过 / 1 警告 / 0 失败**。

## 0. 先跑一遍再动手（1 分钟）

```bash
python tools/cida.py doctor      # 环境 + 依赖 + 产物 + 数据
python tools/cida.py status      # 全系统状态：服务/适配层心跳/产物新鲜度/验收/待研判积压/磁盘
python tools/cida.py verify      # 三套测试 + 130 项交付自检 —— 改完必须重跑这个
```

## 1. 项目当前状态（不要重新调研）

| 项 | 值 |
|---|---|
| **赛道** | 算法创新赛 · **AI+学科交叉**（https://www.aicomp.cn/tracks/tracks-2/3806.html） |
| **所属学科** | 网络空间安全（工学门类一级学科，**0839**） |
| **团队** | **个人参赛：沈轩昊**（材料中不得出现学校名/学院名/LOGO/指导教师） |
| **提交版技术方案** | `competition/09_学科交叉技术方案.md`（按赛道**八节**大纲） |
| 详细版/附录 | `competition/02_技术报告.md` 与 `_精简稿.md`（**已不再是提交主体**） |
| 统一入口 | `python tools/cida.py {doctor,status,demo,guard,verify,inventory,config,retrain}` |
| 工具数 | **40**（新增 `tools/*.py` 会触发约 6 处文档口径级联，优先在**已有文件内**改动） |
| 测试项数 | 单元 22 / Ryu 58 / pcap 58 / 验收自检 130 |

## 2. 体系缺口：①②④⑤ 已完成；只剩 ③（漏报代价换算）

> 来源：`competition/06_检测与处置体系.md` §8「仍在的缺口」。**已完成的不要重做**，直接看下一节。

### ✅ ② 对抗训练做成默认可选（本轮完成）
- `src/config.py`：`adv_train_eps`（默认 `0.0` = 关）、`adv_train_warmup_epochs`（默认 2）
- `src/train.py`：`fgsm_perturb()` + `adversarial_augment()`（自包含，不从 `src` 反向 import `tools`）；
  L∞ 有界 + 投影回训练集逐特征可行域 + 标签逐行配对；**开启时先暖机再生成对抗样本**
  （对随机初始化模型做 FGSM 等于加噪声）；总轮数预算不变（暖机计入）、`eps=0` 时原样返回
- 默认关 → 既有产物与 sha256 不变；回归测试：`test_adversarial_augment_default_off_and_bounded`
- 用法：`python run_all.py --set adv_train_eps=0.2`

### ✅ ⑤ 自动放松（自动降级）（本轮完成）
- `src/service.py`：`DetectorConfig.autotighten`（**默认 `False`**）+ `autotighten_min_labeled`(10) /
  `_normal_ratio`(0.8) / `_step`(0.05) / `_max_threshold`(0.9) / `_check_windows`(200)
- 判定只认**人工复核结论**（`review_feedback.jsonl` + 轮转备份 + 内存队列，按 `review_id` 去重，
  按文件 `mtime+size` 缓存）：结论 ≥N 条且 `normal` 占比 ≥X → 阈值上调一档
- 触发点：`label_review()` 内即时检查 + `push()` 里按窗口节流兜底；每次调整写审计台账
  （`event=autotighten`）、计数进 `stats`/`snapshot`、暴露为 Prometheus `sdn_ids_autotighten_total`
  与 `sdn_ids_threshold`；开关与门槛可用 `POST /api/policy` 热更新
- 回归测试：`test_autotighten_requires_switch_and_human_verdicts`
- 用法：`CIDA_AUTOTIGHTEN=1`（门槛见 `deploy/cida-detect.env`）

### ✅ ① 下发后回读校验 + 失败回滚（本轮完成）
- `integrations/ryu_guard.py`：`verify_readback`（**默认关**）/ `verify_timeout`（2.0s）/ `rollback_on_fail`（默认回滚）
- 下发成功后发 `OFPFlowStatsRequest`，**用 xid 分流**「校验回复」与「周期采集回复」；
  判定不只看"找到同源规则"，还要看**动作对不对**（该丢弃却限速同样是账实不符）
- 不一致 / 超时（=未确认）→ `DELETE_STRICT` 回滚 + 限速连 Meter 一起删 + **解开幂等窗口**以便重试 +
  写 `event=verify` 台账；心跳新增 `verified` / `rollbacks` / `verify_pending`（`cida status` 可见）
- ⚠️ 设计要点：**不能**在下发路径上同步等待回复——Ryu 一个 App 只有一个事件派发循环，
  同步等会等不到回复、把正确规则全判成"未生效"再全部回滚。故做成「异步登记 + 回复/超时判定」
- 回归测试：`tests/test_ryu_guard.py` 新增 22 项 check（34 → **58**，文档口径已同步 ~10 处）
- **线路级已实测**：`tools/ryu_live_check.py` 已改为「有状态假交换机（把收到的 FLOW_MOD 装进流表）+
  沿用请求 xid 回复 + 带 match 的回读请求计数」，并开启 `verify_readback` 重跑 → **19/19 项通过**
  （新增 3 项：发起带 match 的回读请求 / 判定 ok=true 且无超时 / 无误回滚）。
  ⚠️ 仍未覆盖：**回滚路径**在真机上没有演练（需要一个"收下 FLOW_MOD 却静默不装表"的场景）；
  想补的话，给 FakeSwitch 加一个"忽略 FLOW_MOD"开关 + 跑第二个短场景即可

### ✅ ④ 多源证据融合（控制面流表 + 数据面 pcap 双路）（本轮完成）
- 已完成（本轮）：`src/fusion.py` —— 两路证据的**归一/对齐 + 融合策略 + 护栏**，配
  `tests/test_core.py::test_multisource_fusion_policy`
  - `flow_key()` 把两种 schema（控制面 `tp_dst` vs 数据面 `Destination Port`）归一到同一个键；缺 IP **宁可不融合**也不硬配
  - `align_by_key()` 同时给出 `control_only` / `data_only`（只在一路出现的流 = 融合要补的那部分），**不丢证据**
  - `fuse_evidence()`：**冲突不容许被平均抹平**（一路报警 + 一路判正常 → 取保守侧 + `conflict`/`review_required`）；
    只有一路可用 → `degraded`（不假装两路都在）；两路都没有 → **不给分数**（不编造）；
    另有一条"弱证据一致"规则（两路均 ≥ 阈值×0.5）——**这是融合唯一能超过单路的场景**
  - `fusion_metrics()`：把 `rescued_by_pair`（互补覆盖）、`recovered_by_fusion`（两路都漏、融合抓到）、
    `lost_by_fusion`（**必须为 0**，将来改规则时的哨兵）分开数；冲突/降级/误报代价一并如实计入
- **还没做（下一步就做这个）**：把双视角**实测数据**跑出来，写成 `results/multisource_fusion.{md,json}`。可行路径（已勘察）：
  1. 控制面视角：`data/raw_ctrl/sdn_stream.csv` → 用 `data/processed_sdn` 的模型打分（产物已在）
  2. 数据面视角：`python tools/pcap_to_flows.py --write-sample <pcap>`（从同一条仿真流合成抓包 + 写 manifest）
     → `--input <pcap> --output flows.csv` → 用主模型打分（`results/pcap_roundtrip.json` 已有往返校验）
  3. 用 `src/fusion.py` 对齐 + 融合 + `fusion_metrics` 出数
  - 建议做成**已有入口**的子命令（`python tools/cida.py fuse`）而不是新增 `tools/*.py`：
    否则工具数 40→41，又要同步约 6 处文档口径
  - ⚠️ 结论口径：双视角来自**同一条仿真流的两种投影**，只能证明"机制可用、互补与冲突会被如实标出"，
    **不能当真实环境数字**；报告里必须写清这一点

### ✅ 原清单第 1 项：回流重训的**触发自动化**（本轮完成）
- `src/feedback_trigger.py`：判定策略（纯函数）+ 状态（`results/retrain_state.json`），
  判据 = 新增结论 ≥10 **且** 其中新确认攻击 ≥3 **且** 距上次 ≥24h；坏状态**拒绝自动跑**
- 入口：`python tools/cida.py retrain --auto`（未达标**安静退出**，退出码 0 → 定时任务可反复调用）
- 为什么看「新增」而不是累计：累计越过门槛后会每次都重训同一批数据
- 回归测试：`tests/test_core.py::test_feedback_retrain_trigger_policy`（单元测试 21 → **22**）
- 定时任务配方见 `DEPLOYMENT.md` §2.5（systemd timer / Windows 计划任务）
### ⏳ ③ 漏报代价换算（影响面）　【价值中 / 成本中 / **风险高**（若改主流程）】
- 改 `src/evaluate.py::online_alarm_evaluation`：记录**被漏掉片段的持续时长**并换算影响面
- ⚠️ **重跑会改动 `results/summary.json` 与 `artifacts/main_model.json` 的 sha256**
  （项目对这两个有守护，此前全程未变），必须同步更新多处引用与文档
- ✅ 实现路径**已全部勘察清楚（只读派生，不碰守护产物）**：
  * `results/predictions_cli.csv`：1498 行，列 = `row/pred_class/p_attack/alarm/true_class` + 8 类概率
    ——**逐窗口预测 + 真值都有**，`alarm` 直接可用
  * `data/processed/tensors.npz`：含 `test_stream_labels`(21478) 与 **`test_stream_time`(21478, float64)**
    ——**时间戳齐了**，所以"被漏掉的攻击片段持续了多少秒"可以直接算
  * 还缺的只有一件：**影响面换算系数**（每秒攻击流量 → 业务损失）必须**先定义并写死判据**
    （例如"用同一批数据的攻击片段平均包速率/字节速率换算成等效攻击流量"），
    不能拍一个系数——那是编数字，会被现场追问打死
- 💡 若只为**答辩材料**要做这一项：**不必**改主流程——`results/predictions_cli.csv` 已缓存逐窗口预测，
  完全可以写一个**只读**的派生分析（读缓存 → 数漏片段时长 × 放大倍数 → 写新产物），
  一个守卫产物都不用动。动手前先确认缓存里的窗口与测试流标签能对齐

## 3. 必须遵守的纪律（都是踩过坑换来的）

1. **改 docx 一律用 `office_edit` 原子批处理**（`{atomic:true, ops:[…]}`）；
   **绝不碰 `.univer` 容器**——它会被 GUI 复核面板加载并把界面卡死（已复现两次）。
   容器已移出到 `Desktop\_cida_univer_backup\`。
2. **提交信息含引号时，必须写文件 + `git commit -F`**。用 `git commit -m "…引号…"`
   会被 PowerShell 截断（本项目踩过三次）。
3. **改完必须跑 `python tools/cida.py verify`**，它带三套测试 + 130 项自检。
4. **不要重新生成 `data/raw`**；附属实验用独立目录。
5. **每个数字都要能对回磁盘产物**。不能复核的数字不要写进材料。
6. 演示服务固定用 **5070**（5061 等会被 Chromium 当不安全端口拒开）。

## 4. 提交材料还剩三件（只有作者能做）

1. **导出 PDF**：VS Code 装 `Markdown PDF` → 打开 `competition/09_学科交叉技术方案.md`
   → `Ctrl+Shift+P` → `Markdown PDF: Export (pdf)`。
   导出后必核：无 `C:\Users` 路径、**无学校/学院/LOGO/指导教师**、≤10MB、8 张表不变形。
   （中文乱码时设 `markdown-pdf.fontFamily = "Microsoft YaHei"`）
2. **录演示视频**：按 `competition/04_演示视频脚本.md` 分镜录 3–5 分钟。
3. **建公开仓库**：补进 `09` 附录 A 与根 `README.md` 的 CI 徽章；推送前做匿名化检查
   （`python tools/sanitize_paths.py --apply` + `competition/05_匿名化与合规自查清单.md`）。

## 5. 新会话开场白（可直接复制）

> 读 `NEXT_STEPS.md`，然后按第 2 节继续做 **③ 漏报代价换算**（建议走只读派生分析那条路，不破产物守护）：
> 改完跑 `python tools/cida.py verify`，并同步受影响的文档口径。
> 已完成的是 ①②④⑤ 与「回流触发自动化」，**不要重做**；③ 若动主流程会破产物 sha256 守护，只做派生分析则不用。
