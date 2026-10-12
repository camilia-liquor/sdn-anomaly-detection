# NEXT_STEPS · 续做指南

> **给下一个接手的人或 AI 会话**：这份文件是**权威起点**。它不依赖记忆系统
> （项目记忆需要主动读取，关键记忆轨需要人工确认），只看这一个文件就能接着干。
> 最后更新：2026-10-12 ｜ 当前分支 `master` ｜ 工作区干净 ｜ 门禁 **130 项通过 / 1 警告 / 0 失败**。

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
| 测试项数 | 单元 18 / Ryu 34 / pcap 58 / 验收自检 130 |

## 2. 待做的五项体系缺口（已排序：先低风险）

> 来源：`competition/06_检测与处置体系.md` §8「仍在的缺口」。

### ② 对抗训练做成默认可选　【价值中 / 成本中 / **风险低** ← 建议先做】
- `src/config.py` 加 `adv_train_eps: float = 0.0`（0 = 关）
- `src/train.py` 在 `model.fit` 前用 FGSM 增广训练集（**在 src 内自包含一个小的 FGSM 函数**，
  不要从 `src` 反向 import `tools`）
- 默认关 → **不动既有产物、不破 sha256 守护**
- 测试：只测增广函数输出形状/范围，不跑完整训练

### ⑤ 自动降级（自动放松）　【价值中 / 成本低-中 / **风险低** ← 建议同时做】
- `src/service.py` 加 `_maybe_autotighten()`，**复用已有的复核数据**：
  已研判结论 ≥ N 条且其中 `normal` 占比 ≥ X（如 80%）→ 阈值上调一档
- 必须有**显式开关**（默认关）、有上限、写审计台账、记 stats 与 Prometheus
- `tests/test_core.py` 加断言
- 这是唯一能直接复用「复核闭环」的缺口（`review_feedback.jsonl` 已就绪）

### ① 下发后回读校验 + 失败回滚　【价值**高** / 成本中高 / 风险中】
- `integrations/ryu_guard.py`：下发后发 `OFPFlowStatsRequest` 回读确认，失败则回滚
- 需同步扩 `tests/test_ryu_guard.py` 的 `_FakeDatapath` 桩件
- ⚠️ 该测试现有 **34 项 check**，改动后要同步文档口径（约 21 处）

### ③ 漏报代价换算（影响面）　【价值中 / 成本中 / **风险高**】
- 改 `src/evaluate.py::online_alarm_evaluation`：记录**被漏掉片段的持续时长**并换算影响面
- ⚠️ **重跑会改动 `results/summary.json` 与 `artifacts/main_model.json` 的 sha256**
  （项目对这两个有守护，此前全程未变），必须同步更新多处引用与文档

### ④ 多源证据融合（控制面流表 + 数据面 pcap 双路）　【价值**高** / 成本高 / 风险中】
- 需引入第二数据源与融合层，是五项里最大的一项

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

> 读 `NEXT_STEPS.md`，然后按第 2 节做**缺口 ② 和 ⑤**（都是低风险项）。
> 改完跑 `python tools/cida.py verify`，并同步受影响的文档口径。
