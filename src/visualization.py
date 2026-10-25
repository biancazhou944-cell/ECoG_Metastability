"""Figures produced directly from the empirical ECoG analysis."""

import os
import warnings
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from .config import (
    CHANNELS_TO_LOAD,
    COND_COLORS,
    COND_TO_DRUG,
    DATASETS,
    MONKEY_MARKERS,
    PROJECT_ROOT,
)


FIGURES_DIR = Path(os.environ.get('ECOG_FIGURES_DIR', PROJECT_ROOT / 'figures'))
SUPPLEMENTARY_DIR = FIGURES_DIR / 'supplementary'
INTERMEDIATE_DIR = FIGURES_DIR / 'intermediate'
LAYOUT_DIR = PROJECT_ROOT / 'assets' / 'electrode_layouts'
FIGURE_DPI = 300
NON_AWAKE_LABEL = 'Anesthesia/sleep'

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.7,
    'legend.frameon': False,
})

NEUROTYCHO_LAYOUTS = {
    'Chibi': {
        'url': 'https://neurotycho.brain.riken.jp/download/2014/20110621KTMD_Spatial+Map+ECoG+array_Chibi_Toru+Yanagawa_mat_2Dimg.zip',
        'mat': 'ChibiMap.mat',
    },
    'George': {
        'url': 'https://neurotycho.brain.riken.jp/download/2014/20110112KTMD_Spatial+Map+ECoG+array_George_Toru+Yanagawa_mat_2Dimg.zip',
        'mat': 'GeorgeMap.mat',
    },
}


def _savefig(path, fig=None, tight_rect=None):
    if fig is None:
        fig = plt.gcf()
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout(rect=tight_rect)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def _download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response, destination.open('wb') as output:
        output.write(response.read())


def _load_neurotycho_layout(monkey):
    info = NEUROTYCHO_LAYOUTS[monkey]
    mat_path = LAYOUT_DIR / info['mat']
    if not mat_path.exists():
        archive_path = LAYOUT_DIR / Path(info['url']).name
        print(f'Downloading NeuroTycho electrode layout for {monkey} ...')
        _download_file(info['url'], archive_path)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [name for name in archive.namelist() if name.endswith(info['mat'])]
                if not members:
                    raise FileNotFoundError(f"{info['mat']} not found in {archive_path}")
                with archive.open(members[0]) as source, mat_path.open('wb') as target:
                    target.write(source.read())
        finally:
            archive_path.unlink(missing_ok=True)

    mat = sio.loadmat(mat_path)
    return {
        'image': mat['I'],
        'x': mat['X'].ravel(),
        'y': mat['Y'].ravel(),
    }


def _plot_wpli_topography_axes(axes, wpli_node_strengths, band='broadband'):
    selected_channels = np.asarray(CHANNELS_TO_LOAD, dtype=int)
    conditions = list(COND_TO_DRUG)
    monkeys = ['George', 'Chibi']

    differences = []
    for ds_name in DATASETS:
        awake = wpli_node_strengths.get(ds_name, {}).get('Awake', {}).get(band)
        non_awake = wpli_node_strengths.get(ds_name, {}).get('Unconscious', {}).get(band)
        if awake is not None and non_awake is not None:
            differences.append(non_awake - awake)
    finite = [values[np.isfinite(values)] for values in differences]
    pooled = np.concatenate(finite) if finite else np.array([], dtype=float)
    vmax = float(np.nanpercentile(np.abs(pooled), 97.5)) if pooled.size else 1.0
    norm = TwoSlopeNorm(vmin=-max(vmax, 1e-6), vcenter=0.0, vmax=max(vmax, 1e-6))

    last_scatter = None
    for row, monkey in enumerate(monkeys):
        layout = _load_neurotycho_layout(monkey)
        channel_x = layout['x'][selected_channels - 1]
        channel_y = layout['y'][selected_channels - 1]

        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            ds_name = next(
                name for name, ds in DATASETS.items()
                if ds['monkey'] == monkey and ds['condition'] == condition
            )
            awake = wpli_node_strengths[ds_name]['Awake'][band]
            non_awake = wpli_node_strengths[ds_name]['Unconscious'][band]
            difference = non_awake - awake
            valid = np.isfinite(difference)

            ax.imshow(layout['image'], alpha=0.25)
            ax.scatter(
                channel_x,
                channel_y,
                s=18,
                color='#CFD8DC',
                edgecolors='white',
                linewidths=0.25,
                alpha=0.75,
            )
            last_scatter = ax.scatter(
                channel_x[valid],
                channel_y[valid],
                c=difference[valid],
                s=48,
                cmap='RdBu_r',
                norm=norm,
                edgecolors='#263238',
                linewidths=0.35,
            )
            if row == 0:
                ax.set_title(f'{condition}\n{COND_TO_DRUG[condition]}', fontsize=10.5)
            if col == 0:
                ax.text(
                    -0.06,
                    0.5,
                    monkey,
                    transform=ax.transAxes,
                    rotation=90,
                    ha='right',
                    va='center',
                    fontsize=11,
                    fontweight='bold',
                )
            ax.set_axis_off()

    return last_scatter


def plot_condition_dumbbells(kuramoto_results, wpli_results):
    """Plot awake and non-awake values for each monkey-condition pair."""
    metric_specs = [
        ('theta', 'sync', 'Kuramoto synchrony <R(t)>'),
        ('delta', 'meta', 'Metastability Var[R(t)]'),
        ('broadband', 'wpli', 'wPLI'),
    ]
    dataset_order = []
    for condition in COND_TO_DRUG:
        for monkey in MONKEY_MARKERS:
            ds_name = next(
                name for name, ds in DATASETS.items()
                if ds['condition'] == condition and ds['monkey'] == monkey
            )
            dataset_order.append((condition, monkey, ds_name, f'{condition} | {monkey}'))

    y_positions = np.arange(len(dataset_order))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.8), sharey=True)

    for panel_idx, (ax, (band, metric, xlabel)) in enumerate(zip(axes, metric_specs)):
        for condition_index, condition in enumerate(COND_TO_DRUG):
            group_top = len(dataset_order) - condition_index * 2 - 0.5
            group_bottom = len(dataset_order) - (condition_index + 1) * 2 - 0.5
            shade = '#F7FAFB' if condition_index % 2 == 0 else '#FFFFFF'
            ax.axhspan(group_bottom, group_top, color=shade, zorder=0)

        for y_pos, (condition, _, ds_name, _) in zip(y_positions, dataset_order):
            color = COND_COLORS[condition]
            if metric == 'wpli':
                awake = wpli_results[ds_name]['Awake'][band]
                non_awake = wpli_results[ds_name]['Unconscious'][band]
            else:
                awake = kuramoto_results[ds_name]['Awake'][band][metric]
                non_awake = kuramoto_results[ds_name]['Unconscious'][band][metric]
            ax.plot([awake, non_awake], [y_pos, y_pos], color=color, lw=2.4, alpha=0.9)
            ax.scatter(
                [awake], [y_pos], s=60, facecolors='white', edgecolors=color,
                linewidths=1.6, zorder=3,
            )
            ax.scatter(
                [non_awake], [y_pos], s=60, facecolors=color, edgecolors='white',
                linewidths=0.8, zorder=4,
            )

        for condition_index in range(1, len(COND_TO_DRUG)):
            ax.axhline(
                len(dataset_order) - condition_index * 2 - 0.5,
                color='#D7E0E4',
                lw=0.9,
            )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(xlabel, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.2)
        ax.text(
            -0.12, 1.03, f'({chr(65 + panel_idx)})', transform=ax.transAxes,
            fontsize=13, fontweight='bold',
        )

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([label for _, _, _, label in dataset_order], fontsize=10)
    axes[0].set_ylabel('Dataset pair', fontsize=11)

    state_handles = [
        Line2D(
            [0], [0], marker='o', color='none', markerfacecolor='white',
            markeredgecolor='#37474F', markeredgewidth=1.4, markersize=7, label='Awake',
        ),
        Line2D(
            [0], [0], marker='o', color='none', markerfacecolor='#37474F',
            markeredgecolor='white', markersize=7, label=NON_AWAKE_LABEL,
        ),
    ]
    condition_handles = [
        Line2D([0], [0], color=color, lw=2.4, label=f'{cond} ({COND_TO_DRUG[cond]})')
        for cond, color in COND_COLORS.items()
    ]
    fig.legend(
        handles=state_handles + condition_handles,
        loc='upper center',
        ncol=3,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 1.01),
    )
    _savefig(SUPPLEMENTARY_DIR / 'condition_shifts.png', fig, tight_rect=(0, 0, 1, .90))


def plot_electrode_layouts(bad_channels):
    """Plot the fixed 64-channel subset and channels flagged in any dataset."""
    selected_channels = np.asarray(CHANNELS_TO_LOAD, dtype=int)
    monkey_bad_channels = {}
    for monkey in MONKEY_MARKERS:
        bad_actual = set()
        for ds_name, ds in DATASETS.items():
            if ds['monkey'] != monkey:
                continue
            for index in bad_channels.get(ds_name, set()):
                if 0 <= index < len(selected_channels):
                    bad_actual.add(int(selected_channels[index]))
        monkey_bad_channels[monkey] = sorted(bad_actual)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, monkey in zip(axes, ['George', 'Chibi']):
        layout = _load_neurotycho_layout(monkey)
        channel_ids = np.arange(1, len(layout['x']) + 1)
        subset_mask = np.isin(channel_ids, selected_channels)
        bad_mask = np.isin(channel_ids, monkey_bad_channels[monkey])

        ax.imshow(layout['image'])
        ax.scatter(
            layout['x'], layout['y'], s=16, facecolors='none', edgecolors='#B0BEC5',
            linewidths=0.8, alpha=0.8,
        )
        ax.scatter(
            layout['x'][subset_mask], layout['y'][subset_mask], s=24,
            color='#1565C0', alpha=0.85,
        )
        if np.any(bad_mask):
            ax.scatter(
                layout['x'][bad_mask], layout['y'][bad_mask], s=54,
                color='#D32F2F', marker='x', linewidths=1.6,
            )
        ax.set_title(monkey, fontsize=13, fontweight='bold')
        ax.axis('off')
        ax.text(
            0.02,
            0.03,
            f'Selected subset: {subset_mask.sum()}/128\n'
            f'Flagged bad in >=1 dataset: {bad_mask.sum()}',
            transform=ax.transAxes,
            fontsize=9,
            va='bottom',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
        )

    legend_handles = [
        Line2D(
            [0], [0], marker='o', color='none', markerfacecolor='none',
            markeredgecolor='#B0BEC5', label='All 128 electrodes', markersize=6,
        ),
        Line2D(
            [0], [0], marker='o', color='none', markerfacecolor='#1565C0',
            markeredgecolor='#1565C0', label='64-channel analysis subset', markersize=6,
        ),
        Line2D(
            [0], [0], marker='x', color='#D32F2F',
            label='Flagged bad in >=1 dataset', markersize=7,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle('NeuroTycho ECoG Layout and Current Analysis Subset', fontsize=14, y=1.06)
    _savefig(SUPPLEMENTARY_DIR / 'electrode_layouts.png', fig)


def plot_wpli_topography(wpli_node_strengths, band='broadband'):
    """Create the topographic source image used in main Figure 2."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4))
    last_scatter = _plot_wpli_topography_axes(axes, wpli_node_strengths, band=band)
    if last_scatter is not None:
        colorbar = fig.colorbar(last_scatter, ax=axes.ravel().tolist(), shrink=0.78, pad=0.015)
        colorbar.set_label('Node-strength change (non-awake - awake)', fontsize=10)
    fig.suptitle('Broadband wPLI node-strength topography', fontsize=14, y=1.02)
    _savefig(INTERMEDIATE_DIR / 'fig12_wpli_topography.png', fig)
