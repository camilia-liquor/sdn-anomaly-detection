"""SDN 入侵检测：CNN + BiLSTM 工程化实现（参赛版）。

模块划分
--------
* ``config``            —— 全流程配置（可 JSON / 命令行覆盖）
* ``traffic_generator`` —— 仿真 SDN 流量生成（insdn / sdn 双 schema）
* ``data_pipeline``     —— 读取、清洗、编码、标准化、滑窗、时序切分
* ``model``             —— CNN+BiLSTM 及消融变体
* ``train``             —— 训练流程（类别加权、回调、产物落盘）
* ``evaluate``          —— 多分类 + 二分类 + 在线告警级评估
* ``visualize``         —— 论文/答辩级图表
"""

__version__ = "1.0.0"
