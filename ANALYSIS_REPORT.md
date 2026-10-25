# ECoG 分析结果

本报告直接读取当前分析 JSON、CSV 和验证记录。分析包含两只猕猴、四种条件，共八组动物—条件配对。

## 总体结果

64／128 通道、完整时段／120 秒四种设置下，broadband wPLI 的聚合效应量均接近零。四套分析各自的 17 项检验均未通过 FDR 校正。这不等于证明所有状态、动物或空间位置都没有效应；这些结果应作为探索性分析解释。

| 设置 | broadband wPLI 配对 d | 原始 p | FDR p | JZS BF10 |
|---|---:|---:|---:|---:|
| 64 通道 / full | -0.137 | 0.7109 | 0.8882 | 0.358 |
| 64 通道 / 120s | +0.052 | 0.8906 | 0.8906 | 0.339 |
| 128 通道 / full | -0.086 | 0.8594 | 0.9380 | 0.345 |
| 128 通道 / 120s | +0.138 | 0.8281 | 0.8799 | 0.359 |

## 分析方法

- wPLI 在每个频率上跨 STFT 窗计算，再等权平均频率和电极对；零分母返回零。
- 同步性使用 Kuramoto 序参量的时间均值，亚稳态使用其时间方差。
- FCD 使用 5 秒窗、3 秒步长；联盟熵使用 2 秒窗、1 秒步长，并统一特征向量符号。
- 两套通道配置共用有效连续时段选择规则，按状态联合排除坏通道。
- 配对 Bayes factor 使用双侧 JZS 积分，r=0.707；FDR 在每套分析的 17 项检验内计算。

## 运行与验证

| 通道数 | 运行内容 | 墙钟时间 | 峰值 RSS |
|---|---|---:|---:|
| 64 | 八个 dataset、两种状态、完整时段与 120 秒，含统计 | 202.0 秒 | 1.31 GiB |
| 128 | 八个 dataset、两种状态、完整时段与 120 秒，含统计 | 436.8 秒 | 2.02 GiB |

预处理性能验证：逐频段 SciPy 参考流程中位数 1.526 秒，共享流程 0.431 秒，约 3.54 倍。各频带输出最大绝对误差为 0。
该基准只包含固定随机输入下的降采样、公共平均参考与带通滤波，不代表整个分析流程的加速倍数。
真实数据 wPLI 验证：George_KT 两种状态各 120 秒，检查四个保留电极的六个电极对、五个频带，相对于独立逐频率参考计算的最大绝对误差为 2.78e-17。

## 解释范围

- 只有两只动物，八组配对不等于八只独立动物。配对层面的 p、CI、BF 依赖相应交换性与独立性假设。
- 两个动物均值的双侧精确符号翻转检验，最小 p 为 0.5，不能支持群体层面的确认性推断。
- 部分联盟熵接近饱和；退化 bootstrap 抽样被警告并排除，区间以有效抽样计算。
- 未额外改变滤波边缘裁剪、参考方案或伪迹剔除设计；这些可作为独立敏感性分析。

## 文件与复现

- 完整精度结果：[64 通道](results/analysis_64.json)、[128 通道](results/analysis_128.json)。
- 统计汇总：[64 通道](results/statistics_64.csv)、[128 通道](results/statistics_128.csv)。
- 验证记录：[预处理](results/validation/preprocessing_benchmark.json)、[wPLI](results/validation/wpli_validation.json)。
- 主图：[指标效应](figures/metric_effects.png)、[连接分布](figures/connectivity.png)、[稳健性](figures/robustness.png)。
- 补充图：[电极布局](figures/supplementary/electrode_layouts.png)、[条件差异](figures/supplementary/condition_shifts.png)。
- [运行说明](README.md)。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_pipeline.py --resume
.venv/bin/python run_empirical_128_channels.py --resume
.venv/bin/python scripts/generate_paper_figures.py
.venv/bin/python scripts/summarize_results.py
```
