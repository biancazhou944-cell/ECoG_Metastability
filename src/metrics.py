"""Phase and connectivity metrics with bounded intermediate memory."""

import numpy as np
from scipy import signal
from scipy.linalg import eigh

from .config import FS, FS_DOWN, DTYPE, FREQ_BANDS, ANALYSIS_BANDS


def kuramoto_summary(phases):
    # Accumulate over channels without allocating channels x time complex arrays.
    order = np.zeros(phases.shape[1], dtype=np.complex128)
    for phase in phases:
        order += np.exp(1j * phase)
    order = np.abs(order / phases.shape[0])
    return {'sync': float(order.mean()), 'meta': float(order.var())}


def compute_kuramoto(all_phases, datasets):
    return {name: {state: {band: kuramoto_summary(phases)
                          for band, phases in bands.items()}
                   for state, bands in all_phases[name].items()} for name in datasets}


def _pairwise_wpli_spectrum(z, pair_block=16):
    """wPLI at each frequency: |sum_window Im(Sxy)| / sum_window |Im(Sxy)|.

    Zero imaginary power is assigned zero (no evidence of lagged coupling).
    The ratio uses no absolute amplitude threshold and is scale invariant.
    """
    if z.ndim != 3 or z.shape[0] < 2 or z.shape[1] == 0 or z.shape[2] < 2:
        raise ValueError('wPLI requires >=2 channels, >=1 frequency and >=2 windows')
    if not np.isfinite(z).all():
        raise ValueError('Non-finite Fourier coefficients')
    pairs = np.triu_indices(z.shape[0], 1)
    spectrum = np.empty((len(pairs[0]), z.shape[1]), dtype=np.float64)
    offset = 0
    for i in range(z.shape[0]-1):
        for start in range(i+1, z.shape[0], pair_block):
            stop = min(start+pair_block, z.shape[0])
            imag = (z[i][None] * z[start:stop].conj()).imag
            numerator = np.abs(imag.sum(axis=-1, dtype=np.float64))
            denominator = np.abs(imag).sum(axis=-1, dtype=np.float64)
            count = stop-start
            spectrum[offset:offset+count] = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
            offset += count
    return spectrum, pairs


def _summarize_pairs(values, pairs, n_ch):
    node_sum = np.bincount(pairs[0], weights=values, minlength=n_ch)
    node_sum += np.bincount(pairs[1], weights=values, minlength=n_ch)
    return float(values.mean()), (node_sum/(n_ch-1)).astype(DTYPE)


def _wpli_from_stft_band(z_band):
    spectrum, pairs = _pairwise_wpli_spectrum(z_band)
    return _summarize_pairs(spectrum.mean(axis=1), pairs, z_band.shape[0])


def compute_wpli_bands(data, fs, bands=ANALYSIS_BANDS, nperseg=512):
    """Compute per-frequency wPLI once, then average frequencies and channel pairs.

    STFT windows are 512 samples with 50% overlap, constant detrending, and
    no boundary extension/padding. Only requested frequencies are retained.
    """
    if data.ndim != 2 or data.shape[0] < 2 or not np.isfinite(data).all():
        raise ValueError('wPLI expects finite data with at least two channels')
    if fs <= 0 or nperseg < 2 or data.shape[1] < nperseg+nperseg-nperseg//2:
        raise ValueError('At least two complete STFT windows are required')
    bands = list(bands)
    if not bands:
        raise ValueError('At least one band is required')
    frequencies = np.fft.rfftfreq(nperseg, 1/fs)
    requested = np.zeros(len(frequencies), dtype=bool)
    for band in bands:
        lo, hi = FREQ_BANDS[band]
        mask = (frequencies >= lo) & (frequencies <= hi)
        if hi >= fs/2 or not mask.any():
            raise ValueError(f'No valid STFT frequencies for {band}')
        requested |= mask
    frequencies = frequencies[requested]
    n_windows = 1+(data.shape[1]-nperseg)//(nperseg-nperseg//2)
    z = np.empty((data.shape[0], len(frequencies), n_windows), dtype=np.complex64)
    for ch in range(data.shape[0]):
        _, _, channel_z = signal.stft(data[ch], fs=fs, nperseg=nperseg,
                                      noverlap=nperseg//2, detrend='constant',
                                      boundary=None, padded=False)
        z[ch] = channel_z[requested]
    spectrum, pairs = _pairwise_wpli_spectrum(z)
    del z
    results, nodes = {}, {}
    for band in bands:
        lo, hi = FREQ_BANDS[band]
        values = spectrum[:, (frequencies >= lo) & (frequencies <= hi)].mean(axis=1)
        results[band], nodes[band] = _summarize_pairs(values, pairs, data.shape[0])
    return results, nodes


def compute_wpli_mean(data, fs, band, nperseg=512):
    return compute_wpli_bands(data, fs, [band], nperseg)[0][band]


def compute_wpli_all(all_raw, datasets, good_ch_all, duration_sec=None,
                     return_node_strength=False, topography_bands=('broadband',)):
    from .preprocessing import downsample_car
    results, nodes = {}, {}
    for name in datasets:
        results[name], nodes[name] = {}, {}
        for state, (data, _) in all_raw[name].items():
            if duration_sec is not None:
                if data.shape[1] < int(duration_sec*FS):
                    raise ValueError(f'{name}/{state}: epoch shorter than requested duration')
                data = data[:, :int(duration_sec*FS)]
            d = downsample_car(data, good_channels=good_ch_all[name])
            results[name][state], strength = compute_wpli_bands(d, FS_DOWN)
            if return_node_strength:
                nodes[name][state] = {}
                for band in topography_bands:
                    full = np.full(data.shape[0], np.nan, dtype=DTYPE)
                    full[good_ch_all[name]] = strength[band]
                    nodes[name][state][band] = full
    return (results, nodes) if return_node_strength else results


def _window_starts(phases, window_sec, step_sec, fs, min_channels=2):
    if phases.ndim != 2 or phases.shape[0] < min_channels or not np.isfinite(phases).all():
        raise ValueError(f'Expected finite phases with >= {min_channels} channels')
    win, step = int(window_sec*fs), int(step_sec*fs)
    if win < 1 or step < 1 or phases.shape[1] < win:
        raise ValueError('Invalid window size/step or epoch shorter than one window')
    return win, range(0, phases.shape[1]-win+1, step)


def compute_fcd(phases, window_sec=5, step_sec=3, fs=FS_DOWN):
    win, starts = _window_starts(phases, window_sec, step_sec, fs, min_channels=3)
    indices = np.triu_indices(phases.shape[0], 1)
    vectors = np.empty((len(starts), len(indices[0])), dtype=DTYPE)
    for row, start in enumerate(starts):
        window = np.exp(1j*phases[:, start:start+win])
        plv = np.abs(window @ window.conj().T / win)
        vectors[row] = plv[indices]
    centered = vectors-vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('FCD correlation is undefined for constant connectivity vectors')
    normalized = centered/norms
    return (normalized @ normalized.T).astype(DTYPE)


def fcd_variance(phases):
    matrix = compute_fcd(phases)
    if len(matrix) < 2:
        raise ValueError('FCD variance requires at least two complete windows')
    return float(np.var(matrix[np.triu_indices_from(matrix, 1)], dtype=np.float64))


def _canonical_coalition(vector):
    """v and -v represent the same partition; orient the largest loading positive."""
    vector = vector.copy()
    if vector[np.argmax(np.abs(vector))] < 0:
        vector *= -1
    tolerance = np.finfo(vector.dtype).eps * 8 * np.max(np.abs(vector))
    return (vector > tolerance).astype(np.uint8)


def compute_coalition_entropy(phases, window_sec=2, step_sec=1, fs=FS_DOWN):
    win, starts = _window_starts(phases, window_sec, step_sec, fs)
    n_ch = phases.shape[0]
    coalitions = np.empty((len(starts), n_ch), dtype=np.uint8)
    for row, start in enumerate(starts):
        window = phases[:, start:start+win]
        cosine, sine = np.cos(window), np.sin(window)
        coherence = (cosine @ cosine.T + sine @ sine.T)/win
        _, vectors = eigh(coherence, subset_by_index=[n_ch-1, n_ch-1], check_finite=False)
        coalitions[row] = _canonical_coalition(vectors[:, 0])
    _, counts = np.unique(np.packbits(coalitions, axis=1), axis=0, return_counts=True)
    probs = counts/counts.sum()
    entropy = -float(np.sum(probs*np.log2(probs)))
    max_entropy = np.log2(len(coalitions))
    return entropy/max_entropy if max_entropy > 0 else 0.0


def compute_fcd_and_coalition(all_phases, datasets, primary_band='delta'):
    fcd, coalition = {}, {}
    for name in datasets:
        fcd[name], coalition[name] = {}, {}
        for state, bands in all_phases[name].items():
            fcd[name][state] = fcd_variance(bands[primary_band])
            coalition[name][state] = compute_coalition_entropy(bands[primary_band])
    return fcd, coalition
