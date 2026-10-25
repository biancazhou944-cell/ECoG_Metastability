"""Generate figures from full-precision analysis results, including source provenance."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from src.config import ANALYSIS_BANDS, COND_TO_DRUG, DATASETS
from src.pipeline import save_json
from src import visualization as viz

METRIC_NAMES = {'Kur_Sync': 'Synchrony', 'Kur_Meta': 'Metastability', 'wPLI': 'wPLI',
                'FCD_Var': 'FCD variance', 'H_norm': 'Coalition entropy'}
KEYS = [('Kur_Sync', 'theta'), ('Kur_Meta', 'delta'), ('wPLI', 'broadband')]
KEY_LABELS = ['Theta synchrony', 'Delta metastability', 'Broadband wPLI']


def load_analysis(path):
    result = json.loads(Path(path).read_text())
    if result.get('schema_version') != 2 or result.get('status') != 'complete':
        raise ValueError(f'{path}: expected a complete analysis result')
    if set(result['datasets']) != set(DATASETS):
        raise ValueError('Paper figures require all eight dataset pairs')
    for duration in ['full', '120s']:
        rows = result['statistics'].get(duration, [])
        if len(rows) != 17 or any(row['n_pairs'] != 8 for row in rows):
            raise ValueError(f'{path}: expected 17 complete eight-pair tests for {duration}')
        for row in rows:
            if any(row[key] is None or not np.isfinite(row[key]) for key in ['d', 'ci_lo', 'ci_hi', 'p_fdr', 'bf']):
                raise ValueError(f'{path}: cannot plot undefined statistics')
    return result


def validate_comparable(a, b):
    if len(a['channels']) != 64 or len(b['channels']) != 128:
        raise ValueError('Expected 64- and 128-channel results respectively')
    if a['epoch_manifest'] != b['epoch_manifest']:
        raise ValueError('Epoch manifests differ: channel-count robustness would be confounded')
    if a['parameters'] != b['parameters'] or a['durations'] != b['durations']:
        raise ValueError('Analysis parameters/durations differ between channel counts')
    for module in ['config', 'data_loading', 'preprocessing', 'metrics', 'statistics', 'pipeline']:
        key = f'src/{module}.py'
        if a['source_sha256'].get(key) != b['source_sha256'].get(key):
            raise ValueError(f'Analysis source differs between channel counts: {key}')


def _save(fig, directory, name):
    for extension in ['png', 'pdf']:
        fig.savefig(directory/f'{name}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def figure_1(result, directory):
    rows = sorted(result['statistics']['full'], key=lambda row: row['d'])
    fig, ax = plt.subplots(figsize=(9, 7.2), layout='constrained')
    ax.axvline(0, color='#87929d', linewidth=.9)
    for index, row in enumerate(rows):
        color = '#dc8423' if row['metric'] == 'wPLI' else '#3f7ea6'
        ax.plot([row['ci_lo'], row['ci_hi']], [index, index], color='#a0aab3', linewidth=1.5)
        ax.scatter(row['d'], index, color=color, s=32, zorder=3,
                   marker='s' if row['fdr_significant'] else 'o')
    ax.set_yticks(range(len(rows)), [f"{row['band'].capitalize()} {METRIC_NAMES[row['metric']]}" for row in rows])
    ax.set_xlabel("Paired Cohen's d (anesthesia/sleep − awake)")
    ax.set_title('64-channel analysis: full epochs\n8 animal–condition pairs from 2 macaques', pad=14)
    ax.grid(axis='x', alpha=.2)
    fig.supxlabel('Lines: paired bootstrap 95% CI. Squares: FDR < .05. Inference is exploratory.', fontsize=9)
    _save(fig, directory, 'metric_effects')


def figure_2(result, directory):
    conditions = list(COND_TO_DRUG)
    changes = []
    nodes = {}
    for name, record in result['datasets'].items():
        nodes[name] = {state: {band: np.asarray(values, dtype=float)
                              for band, values in metrics['node_strength'].items()}
                       for state, metrics in record['metrics']['full'].items()}
    for condition in conditions:
        records = [result['datasets'][name]['metrics']['full'] for name, ds in DATASETS.items()
                   if ds['condition'] == condition]
        changes.append([np.mean([r['Unconscious']['wpli'][band]-r['Awake']['wpli'][band] for r in records])
                        for band in ANALYSIS_BANDS])
    changes = np.asarray(changes)
    fig = plt.figure(figsize=(14, 6.0), layout='constrained')
    grid = fig.add_gridspec(2, 5, width_ratios=[2.0, 1, 1, 1, 1])
    heat = fig.add_subplot(grid[:, 0])
    vmax = max(float(np.max(np.abs(changes))), 1e-9)
    im = heat.imshow(changes, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
    heat.set_yticks(range(4), list(COND_TO_DRUG.values()))
    heat.set_xticks(range(5), [b.capitalize() for b in ANALYSIS_BANDS], rotation=40, ha='right')
    heat.set_title('Mean wPLI change', loc='left')
    heat.grid(False)
    for i in range(4):
        for j in range(5):
            heat.text(j, i, f'{changes[i, j]:+.3f}', ha='center', va='center', fontsize=8,
                      color='white' if abs(changes[i, j]) > .6*vmax else '#26323d')
    fig.colorbar(im, ax=heat, shrink=.6, label='wPLI change')
    axes = np.array([[fig.add_subplot(grid[row, col+1]) for col in range(4)] for row in range(2)])
    mappable = viz._plot_wpli_topography_axes(axes, nodes)
    fig.colorbar(mappable, ax=list(axes.flat), shrink=.65, extend='both', label='Broadband node-strength change')
    fig.suptitle('Phase-lagged connectivity: anesthesia/sleep − awake', fontsize=14)
    _save(fig, directory, 'connectivity')
    return {'node_vmin': mappable.norm.vmin, 'node_vmax': mappable.norm.vmax,
            'heatmap_vmin': -vmax, 'heatmap_vmax': vmax}


def figure_3(a, b, directory):
    variants = [a['statistics']['full'], a['statistics']['120s'],
                b['statistics']['full'], b['statistics']['120s']]
    maps = [{(row['metric'], row['band']): row for row in rows} for rows in variants]
    matrix = np.array([[mapping[key]['d'] for mapping in maps] for key in KEYS])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    ax = axes[0]
    vmax = max(float(np.max(np.abs(matrix))), 1e-9)
    im = ax.imshow(matrix, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
    ax.set_yticks(range(3), KEY_LABELS)
    ax.set_xticks(range(4), ['64 full', '64 120 s', '128 full', '128 120 s'], rotation=35, ha='right')
    ax.grid(False)
    ax.set_title('A  Effect size across variants', loc='left')
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f'{matrix[i,j]:+.2f}', ha='center', va='center',
                    color='white' if abs(matrix[i,j]) > .6*vmax else '#26323d')
    fig.colorbar(im, ax=ax, shrink=.65, label="Paired d")
    ax = axes[1]
    for index, key in enumerate(KEYS):
        diag = maps[0][key]['diagnostics']
        ax.barh(index, -diag['n_negative'], color='#3f7ea6', label='Decrease' if index == 0 else None)
        ax.barh(index, diag['n_positive'], color='#dc8423', label='Increase' if index == 0 else None)
    ax.set_yticks(range(3), KEY_LABELS)
    ax.axvline(0, color='gray', linewidth=.8)
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-.5, 3.0)
    ax.set_xticks([-8, -4, 0, 4, 8], ['8', '4', '0', '4', '8'])
    ax.set_xlabel('Number of paired comparisons')
    ax.set_title('B  Direction (64 full)', loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax = axes[2]
    bf_keys = KEYS+[('wPLI', 'delta')]
    bf = [maps[0][key]['bf'] for key in bf_keys]
    ax.barh(range(4), bf, color='#2a8a70')
    ax.set_yticks(range(4), KEY_LABELS+['Delta wPLI'])
    ax.set_xscale('log')
    ax.set_xlim(min(.1, min(bf)/2), max(20, max(bf)*2))
    for threshold in [1/3, 1, 3, 10]:
        ax.axvline(threshold, color='gray', linewidth=.7, linestyle='--')
    for index, value in enumerate(bf):
        ax.text(value*1.1, index, f'{value:.2f}', va='center', fontsize=9)
    ax.set_xlabel('JZS BF10 (log scale)')
    ax.set_title('C  Bayes factors (64 full)', loc='left')
    rows = a['statistics']['full']
    count = sum(row['fdr_significant'] for row in rows)
    minimum = min(row['p_fdr'] for row in rows)
    fig.supxlabel(f'64 full: {count}/17 tests pass FDR; minimum adjusted p = {minimum:.4f}. '
                  'Same epochs in both channel settings; exploratory inference.', fontsize=9)
    _save(fig, directory, 'robustness')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--figures-dir', type=Path, default=ROOT/'figures')
    args = parser.parse_args()
    directory = args.figures_dir.resolve()
    paths = [args.results_dir/f'analysis_{n}.json' for n in [64, 128]]
    a, b = [load_analysis(path) for path in paths]
    validate_comparable(a, b)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.grid': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    figure_1(a, directory)
    scales = figure_2(a, directory)
    figure_3(a, b, directory)
    viz.SUPPLEMENTARY_DIR = directory/'supplementary'
    kuramoto, wpli, bad = {}, {}, {}
    for name, record in a['datasets'].items():
        kuramoto[name] = {state: metrics['kuramoto'] for state, metrics in record['metrics']['full'].items()}
        wpli[name] = {state: metrics['wpli'] for state, metrics in record['metrics']['full'].items()}
        bad[name] = sorted(set(range(64))-set(record['good_channel_indices']))
    viz.plot_condition_dumbbells(kuramoto, wpli)
    viz.plot_electrode_layouts(bad)
    save_json(directory/'provenance.json', {
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'visualization_sha256': hashlib.sha256((ROOT/'src/visualization.py').read_bytes()).hexdigest(),
        'scales': scales, 'schema_version': 2,
    })
    print(f'Generated three main figures (PNG/PDF) and two supplementary figures: {directory}')


if __name__ == '__main__':
    main()
