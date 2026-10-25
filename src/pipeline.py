"""One streaming runner for the 64- and 128-channel analyses."""

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from .config import ANALYSIS_BANDS, CHANNELS_TO_LOAD, DATASETS, FS, FS_DOWN, FREQ_BANDS, PROJECT_ROOT
from .data_loading import build_epoch_manifest, load_epoch
from .metrics import compute_wpli_bands, compute_coalition_entropy, fcd_variance, kuramoto_summary
from .preprocessing import channel_quality, downsample_car, phase_from_referenced
from .statistics import statistical_rows

SCHEMA_VERSION = 2
DEFAULT_OUTPUT = PROJECT_ROOT / 'results'
STATES = ('Awake', 'Unconscious')


def json_ready(value):
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _hash(value):
    return hashlib.sha256(json.dumps(json_ready(value), sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT_ROOT/'src').glob('*.py'))}


def _fingerprint(dataset, selection, channels, durations, hashes):
    files = {}
    for record in selection['selected'].values():
        session = Path(dataset['path'])/f"Session{record['session']}"
        for filename in ['ECoGTime.mat', *[f'ECoG_ch{ch}.mat' for ch in channels]]:
            path = session/filename
            stat = path.stat()
            files[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return _hash({'selection': selection, 'channels': channels, 'durations': durations,
                  'files': files, 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'sources': {key: value for key, value in hashes.items()
                              if Path(key).name in {'config.py', 'metrics.py', 'preprocessing.py', 'data_loading.py', 'pipeline.py'}}})


def compute_dataset(dataset, selection, channels, durations):
    raw, quality, bad = {}, {}, set()
    timing = {}
    start = time.perf_counter()
    for state in STATES:
        record = selection['selected'][state]
        raw[state] = load_epoch(dataset['path'], record, channels)
        state_bad, quality[state] = channel_quality(raw[state])
        bad |= state_bad
        print(f"  {state}: session {record['session']}, {record['duration_sec']:.3f}s", flush=True)
    timing['load_and_qc_sec'] = time.perf_counter()-start
    good = sorted(set(range(len(channels)))-bad)
    if len(good) < 3:
        raise ValueError('Fewer than three retained channels after paired-state QC')
    print(f'  Retained {len(good)}/{len(channels)} channels', flush=True)
    metrics = {label: {} for label in durations}
    for state in STATES:
        data = raw.pop(state)
        for label, duration in durations.items():
            start = time.perf_counter()
            n_samples = data.shape[1] if duration is None else int(duration*FS)
            if n_samples > data.shape[1]:
                raise ValueError(f'{state} is shorter than requested {duration}s')
            referenced = downsample_car(data[:, :n_samples], good_channels=good)
            wpli, retained_nodes = compute_wpli_bands(referenced, FS_DOWN)
            nodes = {}
            for band, values in retained_nodes.items():
                full = np.full(len(channels), np.nan)
                full[good] = values
                nodes[band] = full
            record = {'wpli': wpli, 'node_strength': nodes, 'kuramoto': {},
                      'samples_raw': n_samples, 'samples_downsampled': referenced.shape[1]}
            for band in ANALYSIS_BANDS:
                phases = phase_from_referenced(referenced, band)
                record['kuramoto'][band] = kuramoto_summary(phases)
                if band == 'delta':
                    record['fcd_var'] = fcd_variance(phases)
                    record['coalition_entropy'] = compute_coalition_entropy(phases)
                del phases
            metrics[label][state] = record
            del referenced
            timing[f'{state}_{label}_sec'] = time.perf_counter()-start
            print(f"  {state}/{label}: {timing[f'{state}_{label}_sec']:.1f}s", flush=True)
        del data
        gc.collect()
    return {'good_channel_indices': good, 'good_channel_ids': [channels[i] for i in good],
            'bad_channel_ids': [channels[i] for i in sorted(bad)], 'quality': quality,
            'metrics': metrics, 'timing': timing}


def collect_result_rows(records, datasets, duration):
    sources = [{} for _ in range(4)]
    for name in datasets:
        metrics = records[name]['metrics'][duration]
        for source, key in zip(sources, ['kuramoto', 'wpli', 'fcd_var', 'coalition_entropy']):
            source[name] = {state: metrics[state][key] for state in STATES}
    return statistical_rows(*sources, datasets, ANALYSIS_BANDS)


def export_statistics(output, n_channels, statistics):
    fields = ['duration', 'metric', 'band', 'n_pairs', 'mean_difference', 'd', 'ci_lo',
              'ci_hi', 'p_raw', 'p_fdr', 'fdr_significant', 'bf', 'monkey_mean_signflip_p']
    with (output/f'statistics_{n_channels}.csv').open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for duration, rows in statistics.items():
            print(f'\n{n_channels} channels / {duration}: exploratory paired statistics')
            print('metric       band          d       p_raw    p_FDR    BF10   animal-p')
            for row in rows:
                flat = {key: row[key] for key in fields if key in row}
                flat.update(duration=duration, monkey_mean_signflip_p=row['diagnostics']['monkey_mean_signflip_p'])
                writer.writerow(flat)
                print(f"{row['metric']:12} {row['band']:10} {row['d']:+8.3f} "
                      f"{row['p_raw']:8.4f} {row['p_fdr']:8.4f} {row['bf']:7.3f} "
                      f"{flat['monkey_mean_signflip_p']:8.4f}")
            print(f"FDR significant: {sum(r['fdr_significant'] for r in rows)}/{len(rows)}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_analysis(n_channels=64, output_dir=DEFAULT_OUTPUT, dataset_names=None,
                 durations=None, resume=False):
    output = Path(output_dir).resolve()
    if n_channels not in (64, 128):
        raise ValueError('Channel count must be 64 or 128')
    channels = CHANNELS_TO_LOAD if n_channels == 64 else list(range(1, 129))
    datasets = DATASETS if dataset_names is None else {name: DATASETS[name] for name in dataset_names}
    durations = {'full': None, '120s': 120.0} if durations is None else durations
    if not datasets or not durations or any(v is not None and (not np.isfinite(v) or v < 8) for v in durations.values()):
        raise ValueError('Datasets and durations are required; finite durations must be >=8 seconds')
    output.mkdir(parents=True, exist_ok=True)
    cache = output/'.cache'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/f'run_{n_channels}.log').open('w', encoding='utf-8') as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            started = time.perf_counter()
            hashes = source_hashes()
            manifest = build_epoch_manifest(datasets)
            records, reused = {}, []
            for name, dataset in datasets.items():
                print(f'\n{name}: {n_channels} candidate channels', flush=True)
                selection = manifest['datasets'][name]
                fingerprint = _fingerprint(dataset, selection, channels, durations, hashes)
                checkpoint = cache/f'checkpoints_{n_channels}'/f'{name}.json'
                cached = json.loads(checkpoint.read_text()) if resume and checkpoint.exists() else None
                if cached is not None and cached.get('fingerprint') == fingerprint:
                    records[name] = cached['record']
                    reused.append(name)
                    print('  Reused validated checkpoint', flush=True)
                else:
                    records[name] = compute_dataset(dataset, selection, channels, durations)
                    save_json(checkpoint, {'fingerprint': fingerprint, 'record': records[name]})
            statistics = ({duration: collect_result_rows(records, datasets, duration) for duration in durations}
                          if len(datasets) >= 2 else {})
            export_statistics(output, n_channels, statistics)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_bytes = peak if sys.platform == 'darwin' else peak*1024
            result = {
                'schema_version': SCHEMA_VERSION, 'status': 'complete' if statistics else 'metrics_only',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                                'platform': platform.platform()},
                'source_sha256': hashes, 'channels': channels, 'durations': durations,
                'parameters': {'fs': FS, 'fs_down': FS_DOWN, 'bands': FREQ_BANDS,
                               'stft_nperseg': 512, 'stft_noverlap': 256, 'bootstrap_draws': 5000,
                               'random_seed': 42, 'jzs_r': .707,
                               'wpli': 'per_frequency_across_windows_then_equal_frequency_mean',
                               'coalition': 'largest_loading_positive_complement_equivalence',
                               'fcd_window_sec': 5, 'fcd_step_sec': 3,
                               'coalition_window_sec': 2, 'coalition_step_sec': 1,
                               'phase_edge_trim_sec': 0},
                'inference': {'unit': 'animal_condition_pair', 'n_pairs': len(datasets),
                              'n_animals': len({v['monkey'] for v in datasets.values()}),
                              'scope': 'Exploratory. Pairwise p, CI and BF assume exchangeable/independent pair differences. '
                                       'Animal-mean sign-flip p is reported separately; two animals do not support population confirmation.'},
                'dataset_metadata': {name: {k: v for k, v in ds.items() if k != 'path'} for name, ds in datasets.items()},
                'epoch_manifest': manifest, 'datasets': records, 'statistics': statistics,
                'runtime': {'wall_seconds': time.perf_counter()-started, 'peak_rss_bytes': peak_bytes,
                            'reused_datasets': reused},
            }
            save_json(output/f'analysis_{n_channels}.json', result)
            print(f"\nComplete: {result['runtime']['wall_seconds']:.1f}s; process peak RSS {peak_bytes/1024**3:.2f} GiB", flush=True)
            return result


def run_cli(default_channels=64):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', type=int, choices=[64, 128], default=default_channels)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS))
    parser.add_argument('--durations', nargs='+', default=['full', '120'], help='full and/or durations in seconds (>=8)')
    parser.add_argument('--resume', action='store_true', help='Reuse checkpoints only if inputs, settings and source hashes match')
    args = parser.parse_args()
    durations = {}
    for item in args.durations:
        if item == 'full':
            durations['full'] = None
        else:
            seconds = float(item)
            durations[f'{seconds:g}s'] = seconds
    run_analysis(args.channels, args.output_dir, args.datasets, durations, args.resume)
