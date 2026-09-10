# 实验汇总（自动生成）

- 运行标签：`real_insdn_dedup3`
- 数据来源：`real`（schema=`insdn`，输入模式=`sequence`，窗口=10）
- 随机种子：`42`（固定种子 + 时序切分，结果可复现）

## 1. 数据规模

- 训练/验证/测试窗口数：9825 / 2455 / 4093
- 特征数：65，类别数：2
- 类别：Attack, Normal

## 2. 多分类指标

| 指标 | 数值 |
|---|---|
| 准确率 Accuracy | 0.9976 |
| 宏平均精确率 Precision(macro) | 0.9979 |
| 宏平均召回率 Recall(macro) | 0.9969 |
| 宏平均 F1 F1(macro) | 0.9974 |
| 加权 F1 F1(weighted) | 0.9976 |
| Top-2 准确率 | nan |
| Kappa 系数 | 0.9948 |
| 宏平均 AUC(OvR) | nan |

### 逐类表现

| 类别 | 精确率 | 召回率 | F1 | 样本数 |
|---|---|---|---|---|
| Attack | 0.9964 | 0.9996 | 0.9980 | 2521 |
| Normal | 0.9994 | 0.9943 | 0.9968 | 1572 |

## 3. 二分类异常检测（正常 vs 异常）

| 指标 | 数值 |
|---|---|
| 判定阈值 | 0.50 |
| 准确率 | 0.9983 |
| 平衡准确率 | 0.9979 |
| 精确率 | 0.9976 |
| 召回率 | 0.9996 |
| F1 | 0.9986 |
| ROC-AUC | 0.9996 |
| PR-AUC | 0.9998 |
| MCC | 0.9964 |
| TPR@FPR=1% | 0.9996 |
| TPR@FPR=0.1% | 0.9992 |
| 最优阈值 / 最优 F1 | 0.998 / 0.9996 |

## 4. 在线（告警级）指标 —— 贴近真实部署

| 指标 | 数值 |
|---|---|
| 测试流长度 | 40941 条流记录 / 45.48 小时 |
| 攻击片段数 | 1 |
| 成功检出片段 | 1 |
| 片段召回率 | 1.0000 |
| 平均检测时延 | 22.0 秒 |
| P95 检测时延 | 22.0 秒 |
| 误报次数 | 23 |
| 误报率 | 0.5056797361670942 次/小时 |

## 5. 训练信息

| 项目 | 数值 |
|---|---|
| 模型变体 | cnn_bilstm |
| 参数量 | 146,820 |
| 实际训练轮数 | 25 |
| 最优 epoch | 25 |
| 最优验证损失 | 0.0216 |
| 训练耗时 | 39.2 秒 |

## 7. 图表清单

- `figures_real_dedup3\01_class_distribution.png`
- `figures_real_dedup3\02_feature_profiles.png`
- `figures_real_dedup3\04_confusion_matrix_real_insdn_dedup3.png`
- `figures_real_dedup3\05_multiclass_roc_pr_real_insdn_dedup3.png`
- `figures_real_dedup3\06_binary_detection_real_insdn_dedup3.png`
- `figures_real_dedup3\03_training_curves_real_insdn_dedup3.png`
- `figures_real_dedup3\07_online_timeline_real_insdn_dedup3.png`
- `figures_real_dedup3\08_attention_real_insdn_dedup3.png`
