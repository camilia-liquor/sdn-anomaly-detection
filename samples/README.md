# 样例数据（用于快速上手，不需要生成完整数据集）

本目录提供一份**小规模样例流量数据**，方便在不等待完整数据生成的情况下，
先验证环境、跑通代码链路、或用于教学演示。

| 文件 | 说明 |
|---|---|
| `insdn_stream.csv` | 389 条流记录 × 84 列，8 类流量（与主工程同一 schema：列名与 inSDN / CICFlowMeter 逐字一致） |

> 该样例由本作品自研生成器产出（`src/traffic_generator.py`），**随作品以 Apache-2.0 开放**，
> 不含任何第三方数据。

## 怎么用

### 1) 只看数据结构

```python
import pandas as pd
df = pd.read_csv("samples/insdn_stream.csv")
print(df.shape, list(df.columns[:10]))
print(df["Label"].value_counts())
```

### 2) 用样例数据跑一遍完整预处理（几秒钟）

```bash
python run_all.py --source real --real-data-dir samples --set real_label_column=Label \
  --tag sample_demo --set processed_dir=data/processed_sample \
  --set figure_dir=results/figures_sample --set publish_summary=false \
  --set window_size=8 --set window_stride=4 --epochs 5
```

> 样例只有几百条记录，指标本身没有参考意义（唯一目的是验证**代码链路是否通**）。
> 要做真正的实验，请用默认配置从零生成完整数据集：`python run_all.py`。

### 3) 作为"自己的数据"接入的示例

样例数据与"你用 Ryu 采集到的真实 CSV"在使用方式上完全一致 —— 都是把一个目录交给
`--source real --real-data-dir <目录>`，数据管线会按**列名**自动识别标签列与特征列。
因此可以把 `samples/` 当作接入真实数据的模板：

```bash
python run_all.py --source real --real-data-dir D:\my_sdn_capture --set real_label_column=target
```
