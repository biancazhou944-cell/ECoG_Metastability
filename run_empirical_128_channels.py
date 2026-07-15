"""
128-channel empirical robustness run.

This script mirrors the empirical analysis while loading all 128 ECoG channels
before dataset-specific bad-channel exclusion. Statistical output is written to
``results/stats_128_channels.txt``.
"""
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.signal import hilbert

from src.config import ANALYSIS_BANDS, DATASETS, DTYPE, FS, FS_DOWN, PROJECT_ROOT, STATE_MAP
from src.data_loading import (
    parse_info_file,
    load_ecog_session,
    extract_epochs,
    merge_epochs_by_longest,
    list_session_numbers,
)
from src.preprocessing import preprocess_car
from src.metrics import compute_wpli_bands, compute_fcd, compute_coalition_entropy
from src.statistics import collect_tests, print_stats_table, print_group_diagnostics


CHANNELS_128 = list(range(1, 129))
MATCHED_DURATION_SEC = 120
LOG_PATH = Path(
    os.environ.get(
        "ECOG_128_LOG",
        PROJECT_ROOT / "results" / "stats_128_channels.txt",
    )
)


class Tee:
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
    log_file = LOG_PATH.open("w", encoding="utf-8")
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = Tee(stdout_orig, log_file)
    sys.stderr = Tee(stderr_orig, log_file)
    return log_file, stdout_orig, stderr_orig


def load_dataset_epochs(ds_name, ds):
    print(f"Loading {ds_name} with all 128 channels ...")
    events = parse_info_file(ds["path"])
    ds_epochs = {}
    for sess_num in list_session_numbers(ds["path"]):
        sess_dir = ds["path"] / f"Session{sess_num}"
        if not sess_dir.exists():
            continue
        data_sess, time_sess = load_ecog_session(
            ds["path"], sess_num, channels=CHANNELS_128
        )
        sess_events = events.get(sess_num, [])
        merge_epochs_by_longest(
            ds_epochs, extract_epochs(data_sess, time_sess, sess_events)
        )
        del data_sess, time_sess
        gc.collect()

    state_map = STATE_MAP[ds_name]
    raw = {}
    for state_label, cond_key in [
        ("Awake", state_map["Awake"]),
        ("Unconscious", state_map["Unconscious"]),
    ]:
        if cond_key in ds_epochs:
            raw[state_label] = ds_epochs[cond_key]

    for state, (data, _) in raw.items():
        print(f"  {state}: {data.shape[0]}ch x {data.shape[1]} ({data.shape[1] / FS:.1f}s)")
    return raw


def detect_bad_channels_128(raw):
    """Same criteria as the main pipeline, applied within the 128-channel set."""
    n_ch = next(iter(raw.values()))[0].shape[0]
    bad_union = set()
    for state in ["Awake", "Unconscious"]:
        if state not in raw:
            continue
        data, _ = raw[state]
        ch_var = np.var(data, axis=1)
        median_var = np.median(ch_var)
        bad_var = set(
            np.where((ch_var > 10 * median_var) | (ch_var < 0.01 * median_var))[0]
        )

        n_samp = min(data.shape[1], 10 * FS)
        mean_sig = data[:, :n_samp].mean(axis=0)
        corrs = np.empty(n_ch, dtype=np.float64)
        for ch in range(n_ch):
            with np.errstate(invalid="ignore", divide="ignore"):
                corrs[ch] = np.corrcoef(data[ch, :n_samp], mean_sig)[0, 1]
        bad_corr = set(np.where((np.abs(corrs) < 0.05) | ~np.isfinite(corrs))[0])
        bad_union |= bad_var | bad_corr

    good = sorted(set(range(n_ch)) - bad_union)
    return bad_union, good


def compute_phase_metrics(raw, good_channels, duration_sec=None):
    label = "full epoch" if duration_sec is None else f"first {duration_sec}s"
    kuramoto = {}
    fcd = {}
    coalition = {}

    for state in ["Awake", "Unconscious"]:
        if state not in raw:
            continue
        data, _ = raw[state]
        if duration_sec is not None:
            n_samp = min(data.shape[1], int(duration_sec * FS))
            data = data[:, :n_samp]

        kuramoto[state] = {}
        for band in ANALYSIS_BANDS:
            filtered = preprocess_car(data, band, good_channels=good_channels)
            phases = np.angle(hilbert(filtered, axis=1)).astype(DTYPE)
            order_parameter = np.abs(np.mean(np.exp(1j * phases), axis=0))
            kuramoto[state][band] = {
                "sync": float(np.mean(order_parameter)),
                "meta": float(np.var(order_parameter)),
            }

            if band == "delta":
                fcd_matrix = compute_fcd(phases)
                fcd_values = fcd_matrix[np.triu_indices_from(fcd_matrix, k=1)]
                fcd[state] = float(np.var(fcd_values))
                coalition[state] = compute_coalition_entropy(phases)

            del filtered, phases
            gc.collect()

    print(f"  Phase metrics done ({label})")
    return kuramoto, fcd, coalition


def compute_wpli_metrics(raw, good_channels, duration_sec=None):
    label = "full epoch" if duration_sec is None else f"first {duration_sec}s"
    wpli = {}

    for state in ["Awake", "Unconscious"]:
        if state not in raw:
            continue
        data, _ = raw[state]
        n_samp = data.shape[1] if duration_sec is None else min(
            data.shape[1], int(duration_sec * FS)
        )
        downsampled = signal.decimate(
            data[:, :n_samp], int(FS / FS_DOWN), axis=1
        ).astype(DTYPE)
        car_mean = downsampled[good_channels].mean(axis=0, keepdims=True)
        retained = (downsampled - car_mean)[good_channels]
        band_results, _ = compute_wpli_bands(retained, FS_DOWN, ANALYSIS_BANDS)
        wpli[state] = band_results
        del downsampled, retained
        gc.collect()

    print(f"  wPLI done ({label})")
    return wpli


def collect_and_print(label, kuramoto, wpli, fcd, coalition):
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    all_tests, p_fdr, sig_fdr, bf_cache = collect_tests(
        kuramoto, wpli, fcd, coalition, DATASETS, ANALYSIS_BANDS
    )
    print_stats_table(all_tests, p_fdr, sig_fdr, bf_cache)
    print_group_diagnostics(all_tests, DATASETS, bf_cache)
    return all_tests, bf_cache


def print_duration_key_results(full_tests, matched_tests):
    print("\nMatched-duration key effects")
    print("-" * 72)
    full_map = {(name, band): (d, p) for name, band, _, _, d, p in full_tests}
    for name, band, _, _, d_matched, p_matched in matched_tests:
        key = (name, band)
        if key not in {
            ("Kur_Sync", "theta"),
            ("Kur_Meta", "delta"),
            ("wPLI", "broadband"),
        }:
            continue
        d_full, p_full = full_map[key]
        print(
            f"{name} {band}: full d={d_full:+.3f}, p={p_full:.4f} | "
            f"120 s d={d_matched:+.3f}, p={p_matched:.4f}"
        )


def main():
    log_file, stdout_orig, stderr_orig = setup_logging()
    started = time.time()
    print("=" * 72)
    print("128-channel empirical robustness run")
    print("=" * 72)
    print(f"Candidate channels: {CHANNELS_128[0]}-{CHANNELS_128[-1]} ({len(CHANNELS_128)} total)")
    print(f"Log file: {LOG_PATH}")

    kuramoto_full = {}
    wpli_full = {}
    fcd_full = {}
    coalition_full = {}
    kuramoto_120 = {}
    wpli_120 = {}
    fcd_120 = {}
    coalition_120 = {}
    good_counts = {}

    try:
        for ds_name, ds in DATASETS.items():
            print("\n" + "-" * 72)
            raw = load_dataset_epochs(ds_name, ds)
            bad, good = detect_bad_channels_128(raw)
            good_counts[ds_name] = len(good)
            bad_channel_ids = [idx + 1 for idx in sorted(bad)]
            print(
                f"  Bad channels: {len(bad)} -> {len(good)}/128 retained "
                f"(bad channel IDs: {bad_channel_ids})"
            )

            kuramoto_full[ds_name], fcd_full[ds_name], coalition_full[ds_name] = (
                compute_phase_metrics(raw, good, duration_sec=None)
            )
            wpli_full[ds_name] = compute_wpli_metrics(raw, good, duration_sec=None)

            kuramoto_120[ds_name], fcd_120[ds_name], coalition_120[ds_name] = (
                compute_phase_metrics(raw, good, duration_sec=MATCHED_DURATION_SEC)
            )
            wpli_120[ds_name] = compute_wpli_metrics(
                raw, good, duration_sec=MATCHED_DURATION_SEC
            )

            del raw
            gc.collect()

        print("\nRetained good-channel counts")
        print("-" * 72)
        for ds_name, count in good_counts.items():
            print(f"{ds_name:12s}: {count}/128")

        full_tests, _ = collect_and_print(
            "Full-epoch 128-channel statistical analysis",
            kuramoto_full,
            wpli_full,
            fcd_full,
            coalition_full,
        )
        matched_tests, _ = collect_and_print(
            "Duration-matched 128-channel statistical analysis (first 120 s)",
            kuramoto_120,
            wpli_120,
            fcd_120,
            coalition_120,
        )
        print_duration_key_results(full_tests, matched_tests)

        elapsed_min = (time.time() - started) / 60
        print(f"\nDone in {elapsed_min:.1f} minutes.")
    finally:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file.close()


if __name__ == "__main__":
    main()
