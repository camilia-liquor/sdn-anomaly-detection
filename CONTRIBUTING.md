# 贡献指南（Contributing）

感谢关注本作品。这是一个面向 **SDN 流量异常检测** 的开源工程，
欢迎以 Issue 或 Pull Request 的形式反馈问题、补充数据、改进文档。

## 一、开发环境

```bash
git clone <仓库地址> && cd <仓库目录>
python -m venv .venv && .venv\Scripts\activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python tools/check_env.py
```

> 若无法安装 TensorFlow，仍可运行**不依赖 TF** 的全部数据链路与演示：
> `python tests/test_core.py`、`python tools/smoke_test.py`、`python app/app.py`。

## 二、提交前必须通过的自检

```bash
python tests/test_core.py          # 10 项单元测试（生成器/管线/滑窗/评估口径）
python tests/test_ryu_guard.py     # 28 项 Ryu 适配层行为测试（桩件替代 Ryu，无需安装 Ryu）
python run_all.py --smoke          # 25 秒跑通全流程（使用独立目录，不污染正式结果）
python tools/acceptance_check.py   # 130 项交付验收自检（含上面两组测试）
```

四条都必须通过。若你改动了模型结构或评估口径，请在 PR 描述里附上新旧指标对比
（可用 `python tools/reevaluate.py --all` 重算，无需重训）。

## 三、代码约定

- **可复现优先**：任何涉及随机性的改动都必须走 `src/utils.py` 的 `set_seed()`，
  不得引入未固定种子的随机源。
- **不得引入数据泄漏**：新增切分/预处理逻辑时必须保证「标准化器只在训练集拟合」
  「滑窗不跨子集」「同会话不跨子集」，并在 PR 中说明。
- **评估口径变更要留痕**：`src/evaluate.py` 里的指标定义是作品的核心资产之一，
  改动需在 `docs/` 下补充说明，并同步更新 `README.md` 与实验报告。
- **中文注释**：本工程面向中文评审与教学场景，注释与文档以中文为主，
  代码标识符用英文。

## 四、新增数据集的支持方式

本工程通过「按列名自动识别」支持任意流统计量数据集，接入新数据集**通常不需要改代码**：

```bash
python run_all.py --source real --real-data-dir <目录> --set real_label_column=<标签列名>
```

若数据集是**分块发布**（正常块 + 攻击块，只有一次标签切换），请加：

```bash
--set split_mode=random --set split_group_column="Flow ID"
```

若数据集存在大量近似重复，请先跑一次泄漏诊断（`tools/drift_test.py` 里的诊断思路，
或 `_tmp/real_nn_diag.py` 的最近邻分析），确认指标是否被冗余抬高。

## 五、报告问题（Issue）建议包含

1. 运行环境（`python tools/check_env.py` 的输出）
2. 完整命令与报错堆栈
3. 若能复现，附上 `logs/` 下对应的日志文件

## 六、许可与合规

- 提交即表示你同意以本工程的 **Apache-2.0** 许可发布你的贡献；
- **不要**提交无许可证的第三方数据、代码或模型权重；
- 若引入新的第三方依赖，请同时在 `NOTICE` 与
  `competition/01_开源及第三方资源使用清单.md` 中补登其版本与许可证。
