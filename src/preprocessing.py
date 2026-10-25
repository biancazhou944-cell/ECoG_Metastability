"""Shared channel QC and reusable downsampling/reference stage."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate, hilbert

from .config import FS, FS_DOWN, DTYPE, FREQ_BANDS, ANALYSIS_BANDS


def channel_quality(data, fs=FS):
    """Inspect one epoch; exclude non-finite channels before forming the reference."""
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError('Expected at least two channels and samples')
    finite = np.isfinite(data).all(axis=1)
    variances = np.full(data.shape[0], np.nan)
    # Per-channel accumulation avoids a full float64 copy of a long recording.
    for ch in np.flatnonzero(finite):
        variances[ch] = np.var(data[ch], dtype=np.float64)
    usable = finite & (variances > 0)
    if not usable.any():
        raise ValueError('No finite non-flat channels')
    median_var = np.median(variances[usable])
    bad_var = (~usable) | (variances > 10*median_var) | (variances < .01*median_var)
    n_samp = min(data.shape[1], 10*fs)
    reference = data[usable, :n_samp].mean(axis=0, dtype=np.float64)
    corrs = np.full(data.shape[0], np.nan)
    for ch in np.flatnonzero(usable):
        with np.errstate(invalid='ignore', divide='ignore'):
            corrs[ch] = np.corrcoef(data[ch, :n_samp], reference)[0, 1]
    bad = bad_var | ~np.isfinite(corrs) | (np.abs(corrs) < .05)
    reasons = {}
    for ch in np.flatnonzero(bad):
        reason = []
        if not finite[ch]: reason.append('nonfinite')
        elif variances[ch] == 0: reason.append('flat')
        elif bad_var[ch]: reason.append('variance')
        if not np.isfinite(corrs[ch]): reason.append('undefined_correlation')
        elif abs(corrs[ch]) < .05: reason.append('low_correlation')
        reasons[int(ch)] = reason
    return set(np.flatnonzero(bad).tolist()), {
        'variance': variances.tolist(), 'correlation': corrs.tolist(), 'reasons': reasons,
    }


def detect_bad_channels(all_raw, datasets):
    bad_all, good_all = {}, {}
    for name in datasets:
        raw = all_raw[name]
        if set(raw) != {'Awake', 'Unconscious'}:
            raise ValueError(f'{name}: both paired states are required')
        bad = set()
        n_ch = raw['Awake'][0].shape[0]
        for data, _ in raw.values():
            if data.shape[0] != n_ch:
                raise ValueError(f'{name}: channel counts differ between states')
            state_bad, _ = channel_quality(data)
            bad |= state_bad
        good = sorted(set(range(n_ch))-bad)
        if len(good) < 3:
            raise ValueError(f'{name}: fewer than three good channels')
        bad_all[name], good_all[name] = bad, good
    return bad_all, good_all


def downsample_car(data, fs_orig=FS, fs_down=FS_DOWN, good_channels=None):
    """Downsample once and reference only retained finite channels."""
    if fs_down <= 0 or fs_orig <= 0:
        raise ValueError('Sampling rates must be positive')
    ratio = fs_orig/fs_down
    if ratio < 1 or not float(ratio).is_integer():
        raise ValueError('Downsampling requires an integer fs_orig / fs_down >= 1')
    if data.ndim != 2:
        raise ValueError('Expected a channels-by-samples array')
    good = np.arange(data.shape[0]) if good_channels is None else np.asarray(good_channels, dtype=int)
    if len(good) < 2 or len(set(good)) != len(good):
        raise ValueError('At least two unique good channels are required')
    if np.any(good < 0) or np.any(good >= data.shape[0]):
        raise ValueError('Good-channel index out of range')
    n_out = (data.shape[1]+int(ratio)-1)//int(ratio)
    result = np.empty((len(good), n_out), dtype=DTYPE)
    for index, ch in enumerate(good):
        if not np.isfinite(data[ch]).all():
            raise ValueError(f'Non-finite samples in retained channel {ch}')
        result[index] = decimate(data[ch], int(ratio)) if ratio > 1 else data[ch]
    result -= result.mean(axis=0, keepdims=True)
    return result


def bandpass_referenced(data, band, fs=FS_DOWN):
    lo, hi = FREQ_BANDS[band]
    if not 0 < lo < hi < fs/2:
        raise ValueError(f'{band} is incompatible with sampling rate {fs}')
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=1).astype(DTYPE)


def phase_from_referenced(data, band, fs=FS_DOWN):
    filtered = bandpass_referenced(data, band, fs)
    # Channel-wise FFT bounds peak memory for long sleep epochs.
    for ch in range(filtered.shape[0]):
        filtered[ch] = np.angle(hilbert(filtered[ch]))
    return filtered


def preprocess_car(data, band, fs_orig=FS, fs_down=FS_DOWN, good_channels=None):
    return bandpass_referenced(downsample_car(data, fs_orig, fs_down, good_channels), band, fs_down)


def compute_all_phases(all_raw, datasets, good_ch_all, duration_sec=None):
    """Compatibility API; the streaming runner computes and releases one band at a time."""
    output = {}
    for name in datasets:
        output[name] = {}
        for state, (data, _) in all_raw[name].items():
            if duration_sec is not None:
                n = int(duration_sec*FS)
                if data.shape[1] < n:
                    raise ValueError(f'{name}/{state}: shorter than {duration_sec}s')
                data = data[:, :n]
            referenced = downsample_car(data, good_channels=good_ch_all[name])
            output[name][state] = {band: phase_from_referenced(referenced, band) for band in ANALYSIS_BANDS}
    return output
