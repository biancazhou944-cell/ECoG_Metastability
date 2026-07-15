"""Run the primary 64-channel macaque ECoG analysis.

The script writes the statistical output used by the paper figures to
``results/stats_64_channels.txt``. Set ``ECOG_DATA_DIR`` to use a data folder
outside the project directory.
"""

import csv
import gc
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

from src.config import ANALYSIS_BANDS, COND_TO_DRUG, DATASETS, FS, PROJECT_ROOT, STATE_MAP
from src.data_loading import (
    extract_epochs,
    list_session_numbers,
    load_ecog_session,
    parse_info_file,
)
from src.metrics import compute_fcd_and_coalition, compute_kuramoto, compute_wpli_all
from src.preprocessing import compute_all_phases, detect_bad_channels
from src.statistics import collect_tests, print_group_diagnostics, print_stats_table
from src.visualization import (
    INTERMEDIATE_DIR,
    SUPPLEMENTARY_DIR,
    plot_condition_dumbbells,
    plot_electrode_layouts,
    plot_wpli_topography,
)


MATCHED_DURATION_SEC = 120
RESULTS_DIR = PROJECT_ROOT / 'results'
LOG_PATH = Path(os.environ.get('ECOG_64_LOG', RESULTS_DIR / 'stats_64_channels.txt'))


class Tee:
    """Write console output to both the terminal and a results file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = LOG_PATH.open('w', encoding='utf-8')
    stdout_original = sys.stdout
    stderr_original = sys.stderr
    sys.stdout = Tee(stdout_original, log_file)
    sys.stderr = Tee(stderr_original, log_file)
    return log_file, stdout_original, stderr_original


def load_all_epochs():
    """Load the epochs used for the reported 64-channel results.

    Sessions are processed in numerical order. When the same label occurs in
    more than one session, the later matching session replaces the earlier one.
    This preserves the selection behavior of the original 64-channel analysis
    and therefore reproduces the values reported in the manuscript.
    """
    all_raw = {}
    for ds_name, dataset in DATASETS.items():
        print(f'Loading {ds_name} ...')
        events = parse_info_file(dataset['path'])
        dataset_epochs = {}
        for session_number in list_session_numbers(dataset['path']):
            session_dir = dataset['path'] / f'Session{session_number}'
            if not session_dir.exists():
                continue
            session_data, session_time = load_ecog_session(dataset['path'], session_number)
            session_events = events.get(session_number, [])
            dataset_epochs.update(
                extract_epochs(session_data, session_time, session_events)
            )
            del session_data, session_time
            gc.collect()

        state_map = STATE_MAP[ds_name]
        raw = {}
        for state, label in (
            ('Awake', state_map['Awake']),
            ('Unconscious', state_map['Unconscious']),
        ):
            if label in dataset_epochs:
                raw[state] = dataset_epochs[label]
        all_raw[ds_name] = raw
        for state, (data, _) in raw.items():
            print(
                f'  {state}: {data.shape[0]} channels x {data.shape[1]} samples '
                f'({data.shape[1] / FS:.1f} s)'
            )
    return all_raw


def print_duration_key_results(full_tests, matched_tests):
    full_map = {(name, band): (effect, p_value) for name, band, _, _, effect, p_value in full_tests}
    keys = {
        ('Kur_Sync', 'theta'),
        ('Kur_Meta', 'delta'),
        ('wPLI', 'broadband'),
    }
    print('\nMatched-duration key effects')
    print('-' * 72)
    for name, band, _, _, matched_effect, matched_p in matched_tests:
        key = (name, band)
        if key not in keys:
            continue
        full_effect, full_p = full_map[key]
        print(
            f'{name} {band}: full d={full_effect:+.3f}, p={full_p:.4f} | '
            f'120 s d={matched_effect:+.3f}, p={matched_p:.4f}'
        )


def save_condition_mean_wpli(wpli_results):
    """Save the condition-by-band changes plotted in main Figure 2A."""
    output_path = RESULTS_DIR / 'wpli_condition_mean_changes.csv'
    with output_path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['condition', *ANALYSIS_BANDS])
        for condition, full_name in COND_TO_DRUG.items():
            changes = []
            for band in ANALYSIS_BANDS:
                values = [
                    wpli_results[ds_name]['Unconscious'][band]
                    - wpli_results[ds_name]['Awake'][band]
                    for ds_name, dataset in DATASETS.items()
                    if dataset['condition'] == condition
                ]
                changes.append(sum(values) / len(values))
            writer.writerow([full_name, *changes])
    print(f'Condition-level wPLI summary: {output_path}')


def main():
    log_file, stdout_original, stderr_original = setup_logging()
    started = time.time()
    try:
        print('=' * 72)
        print('Primary 64-channel empirical ECoG analysis')
        print('=' * 72)
        print(f'Log file: {LOG_PATH}')

        print('\nSTEP 1: Load labeled ECoG epochs')
        print('-' * 72)
        all_raw = load_all_epochs()
        loaded_mb = sum(
            data.nbytes for epochs in all_raw.values() for data, _ in epochs.values()
        ) / 1024**2
        print(f'Loaded approximately {loaded_mb:.0f} MB.')

        print('\nSTEP 2: Detect bad channels and extract phase')
        print('-' * 72)
        bad_channels, good_channels = detect_bad_channels(all_raw, DATASETS)
        all_phases = compute_all_phases(all_raw, DATASETS, good_channels)
        gc.collect()

        print('\nSTEP 3: Compute full-epoch metrics')
        print('-' * 72)
        kuramoto = compute_kuramoto(all_phases, DATASETS)
        wpli, node_strength = compute_wpli_all(
            all_raw,
            DATASETS,
            good_channels,
            duration_sec=None,
            return_node_strength=True,
        )
        fcd, coalition = compute_fcd_and_coalition(all_phases, DATASETS)

        print('\nSTEP 4: Full-epoch statistical analysis')
        print('-' * 72)
        full_tests, p_fdr, significant_fdr, bayes_factors = collect_tests(
            kuramoto,
            wpli,
            fcd,
            coalition,
            DATASETS,
            ANALYSIS_BANDS,
        )
        print_stats_table(full_tests, p_fdr, significant_fdr, bayes_factors)
        print_group_diagnostics(full_tests, DATASETS, bayes_factors)
        save_condition_mean_wpli(wpli)

        print(f'\nSTEP 5: Duration robustness (first {MATCHED_DURATION_SEC} s)')
        print('-' * 72)
        phases_120 = compute_all_phases(
            all_raw,
            DATASETS,
            good_channels,
            duration_sec=MATCHED_DURATION_SEC,
        )
        kuramoto_120 = compute_kuramoto(phases_120, DATASETS)
        wpli_120 = compute_wpli_all(
            all_raw,
            DATASETS,
            good_channels,
            duration_sec=MATCHED_DURATION_SEC,
        )
        fcd_120, coalition_120 = compute_fcd_and_coalition(phases_120, DATASETS)
        matched_tests, _, _, _ = collect_tests(
            kuramoto_120,
            wpli_120,
            fcd_120,
            coalition_120,
            DATASETS,
            ANALYSIS_BANDS,
        )
        print_duration_key_results(full_tests, matched_tests)

        print('\nSTEP 6: Generate supplementary and intermediate figures')
        print('-' * 72)
        plot_condition_dumbbells(kuramoto, wpli)
        plot_electrode_layouts(bad_channels)
        plot_wpli_topography(node_strength)

        elapsed_minutes = (time.time() - started) / 60
        print(f'\nAnalysis complete in {elapsed_minutes:.1f} minutes.')
        print(f'Supplementary figures: {SUPPLEMENTARY_DIR}')
        print(f'Intermediate figure source: {INTERMEDIATE_DIR}')
        print('Run scripts/generate_paper_figures.py after the 128-channel analysis.')
    finally:
        sys.stdout = stdout_original
        sys.stderr = stderr_original
        log_file.close()


if __name__ == '__main__':
    main()
