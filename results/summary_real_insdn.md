# 实验汇总（自动生成）

- 运行标签：`real_insdn`
- 数据来源：`real`（schema=`insdn`，输入模式=`sequence`，窗口=10）
- 随机种子：`42`（固定种子 + 时序切分，结果可复现）

## 1. 数据规模

- 训练/验证/测试窗口数：11083 / 2770 / 4617
- 特征数：65，类别数：2
- 类别：Attack, Normal

## 2. 多分类指标

| 指标 | 数值 |
|---|---|
| 准确率 Accuracy | 0.9998 |
| 宏平均精确率 Precision(macro) | 0.9997 |
| 宏平均召回率 Recall(macro) | 0.9998 |
| 宏平均 F1 F1(macro) | 0.9998 |
| 加权 F1 F1(weighted) | 0.9998 |
| Top-2 准确率 | nan |
| Kappa 系数 | 0.9995 |
| 宏平均 AUC(OvR) | nan |

### 逐类表现

| 类别 | 精确率 | 召回率 | F1 | 样本数 |
|---|---|---|---|---|
| Attack | 1.0000 | 0.9997 | 0.9998 | 3021 |
| Normal | 0.9994 | 1.0000 | 0.9997 | 1596 |

## 3. 二分类异常检测（正常 vs 异常）

| 指标 | 数值 |
|---|---|
| 判定阈值 | 0.50 |
| 准确率 | 0.9998 |
| 平衡准确率 | 0.9998 |
| 精确率 | 1.0000 |
| 召回率 | 0.9997 |
| F1 | 0.9998 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |
| MCC | 0.9995 |
| TPR@FPR=1% | 1.0000 |
| TPR@FPR=0.1% | 1.0000 |
| 最优阈值 / 最优 F1 | 0.114 / 1.0000 |

## 4. 在线（告警级）指标 —— 贴近真实部署

| 指标 | 数值 |
|---|---|
| 测试流长度 | 46187 条流记录 / 51.32 小时 |
| 攻击片段数 | 1 |
| 成功检出片段 | 1 |
| 片段召回率 | 1.0000 |
| 平均检测时延 | 41.0 秒 |
| P95 检测时延 | 41.0 秒 |
| 误报次数 | 0 |
| 误报率 | 0.0 次/小时 |

## 5. 训练信息

| 项目 | 数值 |
|---|---|
| 模型变体 | cnn_bilstm |
| 参数量 | 146,820 |
| 实际训练轮数 | 25 |
| 最优 epoch | 25 |
| 最优验证损失 | 0.0201 |
| 训练耗时 | 27.4 秒 |

## 7. 图表清单

- `figures_real\01_class_distribution.png`
- `figures_real\02_feature_profiles.png`
- `figures_real\04_confusion_matrix_real_insdn.png`
- `figures_real\05_multiclass_roc_pr_real_insdn.png`
- `figures_real\06_binary_detection_real_insdn.png`
- `figures_real\03_training_curves_real_insdn.png`
- `figures_real\07_online_timeline_real_insdn.png`
- `figures_real\08_attention_real_insdn.png`
