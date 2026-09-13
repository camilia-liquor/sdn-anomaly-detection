# Word 交付稿生成说明

本文件说明 `competition/` 下两份 Word 交付稿的来源、生成管线、以及导出环节已知的两个限制。

## 1. 交付物

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `competition/实验报告.docx` | 实验报告（摘要、数据、方法、9 组实验、运维换算、讨论与局限、复现命令），含 5 张图表 | `docs/02_实验报告.md`（由 `tools/make_report.py` 生成） |
| `competition/技术报告.docx` | 完整技术报告（一～九章 + 附录），约 1.5 万字、47 张表 | `competition/02_技术报告.md` |

两份 `.docx` 均已纳入版本库；同名 `.univer` 是生成它们的可编辑容器，属中间产物，已在 `.gitignore` 中排除。

## 2. 生成管线

```
Markdown 源（docs/02_实验报告.md、competition/02_技术报告.md）
   ↓ 解析为块（标题 / 段落 / 列表 / 引用 / 代码块 / 表格 / 图片），写入 Univer Doc 单元
<name>.univer（工作树 worktree，可编辑、可复核）
   ↓ univer_export
<name>.docx
```

- 标题层级用 Univer 的命名样式（`TITLE` / `SUBTITLE` / `HEADING_1..4`）表达；表格为真实表格对象（表头行、列宽）；图片为内嵌图片（PNG，宽 600 px 等比缩放）；表格数据与正文数字逐项取自 `results/` 与 `artifacts/`，与 Markdown 源一致。
- 校验方式：导出前用 `univer_inspect` 读回段落数、表格数、图片数；导出后再导入一份临时 `.univer` 复核往返一致性（实验报告：788 段 / 15 表 / 5 图）。

## 3. 两个已知导出限制与绕过办法

Univer 当前的 `.docx` 导出器会丢弃部分样式信息，实测确认并已绕过：

| 限制 | 表现 | 绕过办法 |
| --- | --- | --- |
| 命名样式不写入 | 导出的 `document.xml` 中无 `w:pStyle`，`styles.xml` 只有 `Normal`/`TableNormal` 等 5 个样式，标题在 Word 里看不出层级 | 给标题段落同时写「直接格式」（粗体 + 字号：TITLE 22pt / H1 18pt / H2 17pt / H3 14pt / H4 12pt），run 级直接格式能正常导出（`w:b`、`w:sz`） |
| 表级边框不写入 | `setTableBorder()` 对表格整体设置边框后，导出结果里没有 `w:tblBorders` | 改为逐单元格 `cell.setBorder({preset: Outer, color: '#BFBFBF', width: 1})`，导出为 `w:tcBorders`；并用 `cell.setBackgroundColor('#EFEFEF')` 给表头行加底色（导出为 `w:shd`） |

代价说明：命名样式丢失意味着 Word 的「导航窗格 / 自动目录」无法按标题层级生成；正文排版、表格格线、图片、代码等宽字体、引用斜体均正常。若需要带真实标题样式的版本，可在 DSH 重启后用 `office_*` 工具（zagens-office）重新导出。

## 4. 重新生成

Markdown 源变更后（例如重跑 `python tools/make_report.py`），需要重新导出对应交付稿：

1. `univer_status` 确认 `.univer` 与工作树状态（`ready` 需先 `reopen`）；
2. 用 `univer_execute` 重跑内容写入脚本，再跑一次样式补丁脚本（标题直接格式 + 单元格边框）；
3. `univer_inspect` 读回校验 → `univer_export` 覆盖对应 `.docx`；
4. `univer_worktree` 置回 `ready`。

注意：`.docx` 是从 `.md` 派生的产物，修改 Markdown 后不会自动同步，需按上述步骤重导。
