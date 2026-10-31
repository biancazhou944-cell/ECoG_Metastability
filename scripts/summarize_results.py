"""Write the analysis report from the current result and validation files."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_paper_figures import load_analysis, validate_comparable
from src.pipeline import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--output', type=Path, default=ROOT/'ANALYSIS_REPORT.md')
    args = parser.parse_args()
    results = {n: load_analysis(args.results_dir/f'analysis_{n}.json') for n in [64, 128]}
    validate_comparable(results[64], results[128])
    current_hashes = source_hashes()
    for n, result in results.items():
        recorded = {**result.get('layout_source_sha256', result['source_sha256']),
                    **result.get('documentation_source_sha256', {})}
        for module in ['config', 'data_loading', 'preprocessing', 'metrics', 'statistics', 'pipeline']:
            path = f'src/{module}.py'
            if current_hashes[path] != recorded[path]:
                raise ValueError(f'{n}: source does not match the result metadata: {path}')
    benchmark = json.loads((args.results_dir/'validation/preprocessing_benchmark.json').read_text())
    audit = json.loads((args.results_dir/'validation/wpli_validation.json').read_text())
    def metric(n, duration, name, band):
        return next(row for row in results[n]['statistics'][duration]
                    if (row['metric'], row['band']) == (name, band))
    variants = [(n, duration) for n in [64, 128] for duration in ['full', '120s']]
    metrics = [('Theta 同步性', 'Kur_Sync', 'theta'),
               ('Delta 亚稳态', 'Kur_Meta', 'delta'),
               ('Broadband wPLI', 'wPLI', 'broadband')]
    descriptions = []
    for label, name, band in metrics:
        effects = [metric(n, duration, name, band)['d'] for n, duration in variants]
        direction = ('均为负值' if max(effects) < 0 else
                     '均为正值' if min(effects) > 0 else '方向随设置变化')
        descriptions.append(f'{label}：配对效应量{direction}（d = {min(effects):+.3f} 至 {max(effects):+.3f}）')
    counts = [(sum(row['fdr_significant'] for row in results[n]['statistics'][duration]),
               len(results[n]['statistics'][duration])) for n, duration in variants]
    if all(count == 0 for count, _ in counts):
        fdr_summary = '四种设置下，每套分析的 17 项检验均未达到 FDR 校正后的显著性阈值。'
    else:
        fdr_summary = 'FDR 校正后各设置的显著检验数依次为：' + '、'.join(
            f'{n} 通道 / {duration}：{count}/{total}'
            for (n, duration), (count, total) in zip(variants, counts)) + '。'
    report = [
        '# ECoG 分析结果',
        '',
        '本项目比较清醒与麻醉或自然睡眠时的皮层相位同步、亚稳态和功能连接。'
        '数据来自 NeuroTycho 的两只猕猴 George 和 Chibi；每只动物包含三种麻醉条件和自然睡眠，共八组动物—条件配对。',
        '',
        '## 主要结果',
        '',
        '比较包括 64／128 通道和完整时段／120 秒四种设置。'
        '配对差异定义为麻醉或睡眠减去清醒，因此负值表示相应指标的聚合水平较低。',
        '',
        '；'.join(descriptions) + '。',
        '',
        fdr_summary + '由于同一动物贡献多个条件，这些结果用于描述当前样本中的差异，群体推断的限制见下文。',
        '',
        '| 设置 | 指标 | 配对 d | FDR p | JZS BF10 |',
        '|---|---|---:|---:|---:|',
    ]
    for n, duration in variants:
        duration_label = '完整时段' if duration == 'full' else '120 秒'
        for label, name, band in metrics:
            row = metric(n, duration, name, band)
            report.append(f"| {n} 通道 / {duration_label} | {label} | {row['d']:+.3f} | "
                          f"{row['p_fdr']:.4f} | {row['bf']:.3f} |")
    report += [
        '',
        '表中列出三项指标作比较；全部 17 项指标／频带组合及其原始 p 值和置信区间见统计汇总文件。',
        '',
        '## 分析方法',
        '',
        '- 每个状态选择最长的有效连续时段，并从其起点截取 120 秒用于时长匹配。两套通道配置使用相同的时段。',
        '- 在每个通道配置内分别检测两个状态的坏通道，再将其并集用于两个状态和两种时长。保留通道降采样至 250 Hz，并进行公共平均参考。',
        '- 对所有保留电极的相位计算 Kuramoto 序参量的模 R(t)。同步性取其时间均值，亚稳态取时间方差。',
        '- 对每个电极对和频率，跨 STFT 时间窗计算 wPLI，再依次对频带内的频率和电极对等权平均。零分母时记为零。',
        '- 上述三项指标分别在 delta、theta、alpha、beta 和 broadband 计算。FCD 方差和归一化联盟熵仅在 delta 计算，共 17 项指标／频带组合。',
        '- FCD 用 5 秒窗、3 秒步长计算 PLV 连接矩阵，再比较时间窗间的连接模式。联盟熵用 2 秒窗、1 秒步长计算通道分区的多样性，并统一特征向量符号。',
        '- 统计分析采用配对符号翻转检验、效应量的 bootstrap 95% 置信区间和双侧 JZS Bayes factor（r=0.707）；每套分析分别进行 FDR 校正。',
        '',
        '## 结果的解释范围',
        '',
        '- 八组配对来自两只动物。同一动物内的条件存在依赖，配对层面的 p 值、置信区间和 Bayes factor 依赖配对差异的交换性／独立性假设，应作为探索性结果解释。',
        '- 将每只动物的条件差异取平均后，只有两个观测值；双侧精确符号翻转检验的最小 p 值为 0.5。群体层面的推断需要更多动物。',
        '- 部分联盟熵接近饱和，其区分不同状态的能力可能受限。效应量无法定义的 bootstrap 重采样会被记录并排除，区间根据有效重采样计算。',
        '- 当前采用公共平均参考，未裁剪滤波边缘，也未做时间片段伪迹剔除。参考方式、边缘裁剪和伪迹筛选可作为后续敏感性分析。',
        '',
        '## 运行与验证',
        '',
        '| 通道数 | 运行内容 | 运行时间 | 峰值内存（RSS） |',
        '|---|---|---:|---:|',
    ]
    for n in [64, 128]:
        runtime = results[n]['runtime']
        report.append(f"| {n} | 八个数据集、两种状态、完整时段与 120 秒，含统计 | "
                      f"{runtime['wall_seconds']:.1f} 秒 | {runtime['peak_rss_bytes']/1024**3:.2f} GiB |")
    report += [
        '',
        f"在固定随机输入的预处理基准中，逐频段 SciPy 参考流程耗时中位数为 {benchmark['reference']['median_seconds']:.3f} 秒，"
        f"复用降采样和参考步骤后为 {benchmark['shared']['median_seconds']:.3f} 秒，加速比为 {benchmark['speedup']:.2f}。"
        f"各频带输出最大绝对误差为 {max(benchmark['max_abs_errors'].values()):.3g}。",
        '该基准覆盖降采样、公共平均参考与带通滤波，不包含数据读取、Hilbert 变换、指标计算和绘图。',
        f"真实数据 wPLI 验证：George_KT 两种状态各 120 秒，检查四个保留电极的六个电极对、五个频带，"
        f"相对于独立逐频率参考计算的最大绝对误差为 {audit['max_absolute_error']:.3g}。",
        '',
        '## 文件与复现',
        '',
        '- 完整精度结果：[64 通道](results/analysis_64.json)、[128 通道](results/analysis_128.json)。',
        '- 统计汇总：[64 通道](results/statistics_64.csv)、[128 通道](results/statistics_128.csv)。',
        '- 验证记录：[预处理](results/validation/preprocessing_benchmark.json)、[wPLI](results/validation/wpli_validation.json)。',
        '- 主图：[指标效应](figures/metric_effects.png)、[连接分布](figures/connectivity.png)、[稳健性](figures/robustness.png)。',
        '- 补充图：[电极布局](figures/supplementary/electrode_layouts.png)、[条件差异](figures/supplementary/condition_shifts.png)。',
        '- [运行说明](README.md)。',
        '',
        '```bash',
        '.venv/bin/python -m unittest discover -s tests -v',
        '.venv/bin/python run_pipeline.py --resume',
        '.venv/bin/python run_empirical_128_channels.py --resume',
        '.venv/bin/python scripts/generate_paper_figures.py',
        '.venv/bin/python scripts/summarize_results.py',
        '```',
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
