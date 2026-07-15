import numpy as np
from scipy import signal

from .config import FS, FS_DOWN, DTYPE, FREQ_BANDS, ANALYSIS_BANDS


def compute_kuramoto(all_phases, datasets):
    """Compute Kuramoto order parameter: synchrony and metastability."""
    kuramoto_results = {}
    for ds_name in datasets:
        kuramoto_results[ds_name] = {}
        for state in ['Awake', 'Unconscious']:
            if state not in all_phases[ds_name]:
                continue
            kuramoto_results[ds_name][state] = {}
            for band in ANALYSIS_BANDS:
                ph = all_phases[ds_name][state][band]
                R = np.abs(np.mean(np.exp(1j * ph), axis=0))
                kuramoto_results[ds_name][state][band] = {
                    'sync': float(np.mean(R)),
                    'meta': float(np.var(R)),
                }
    return kuramoto_results


def _wpli_from_stft_band(z_band):
    """Return mean pairwise wPLI and node strengths for a preselected STFT band."""
    n_ch = z_band.shape[0]
    node_sum = np.zeros(n_ch, dtype=np.float64)
    node_count = np.zeros(n_ch, dtype=np.int32)
    wpli_sum = 0.0
    wpli_count = 0

    for i in range(n_ch - 1):
        zi = z_band[i]
        zj = z_band[i + 1:]
        # Match the previous CSD-based definition, but compute all pairs for node i at once.
        cross_spectrum = np.mean(zi[None, :, :] * np.conj(zj), axis=-1)
        imag_csd = np.imag(cross_spectrum)
        den = np.mean(np.abs(imag_csd), axis=1)
        vals = np.zeros(len(den), dtype=np.float64)
        valid = den > 1e-12
        vals[valid] = np.abs(np.mean(imag_csd[valid], axis=1)) / den[valid]

        js = np.arange(i + 1, n_ch)
        node_sum[i] += np.sum(vals)
        node_sum[js] += vals
        node_count[i] += len(vals)
        node_count[js] += 1
        wpli_sum += float(np.sum(vals))
        wpli_count += len(vals)

    node_count[node_count == 0] = 1
    node_strength = (node_sum / node_count).astype(DTYPE)
    return float(wpli_sum / max(wpli_count, 1)), node_strength


def compute_wpli_bands(data, fs, bands=ANALYSIS_BANDS, nperseg=512):
    """Compute mean wPLI and node-wise wPLI strength for all requested bands."""
    f, _, z = signal.stft(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend='constant',
        boundary=None,
        padded=False,
        axis=-1,
    )
    z = z.astype(np.complex64, copy=False)
    band_results = {}
    node_results = {}
    for band in bands:
        lo, hi = FREQ_BANDS[band]
        mask = (f >= lo) & (f <= hi)
        mean_wpli, node_strength = _wpli_from_stft_band(z[:, mask, :])
        band_results[band] = mean_wpli
        node_results[band] = node_strength
    return band_results, node_results


def compute_wpli_mean(data, fs, band, nperseg=512):
    band_results, _ = compute_wpli_bands(data, fs, bands=[band], nperseg=nperseg)
    return band_results[band]


def compute_wpli_all(
    all_raw,
    datasets,
    good_ch_all,
    duration_sec=None,
    return_node_strength=False,
    topography_bands=('broadband',),
):
    """Compute wPLI for all datasets."""
    wpli_results = {}
    node_results_all = {}
    label = 'full epoch' if duration_sec is None else f'first {duration_sec}s'
    for ds_name in datasets:
        good = good_ch_all[ds_name]
        wpli_results[ds_name] = {}
        node_results_all[ds_name] = {}
        for state in ['Awake', 'Unconscious']:
            if state not in all_raw[ds_name]:
                continue
            data, _ = all_raw[ds_name][state]
            n_samp = data.shape[1] if duration_sec is None else min(data.shape[1], int(duration_sec * FS))
            d = signal.decimate(data[:, :n_samp], int(FS / FS_DOWN), axis=1).astype(DTYPE)
            car_mean = d[good].mean(axis=0, keepdims=True)
            d = (d - car_mean)[good]
            band_results, node_results = compute_wpli_bands(d, FS_DOWN, ANALYSIS_BANDS)
            wpli_results[ds_name][state] = band_results
            if return_node_strength:
                node_results_all[ds_name][state] = {}
                for band in topography_bands:
                    full_node_strength = np.full(data.shape[0], np.nan, dtype=DTYPE)
                    full_node_strength[good] = node_results[band]
                    node_results_all[ds_name][state][band] = full_node_strength
            del d
        print(f'{ds_name}: wPLI done ({label})')
    if return_node_strength:
        return wpli_results, node_results_all
    return wpli_results


def compute_fcd(phases, window_sec=5, step_sec=3, fs=FS_DOWN):
    n_ch, n_t = phases.shape
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    starts = list(range(0, n_t - win, step))
    triu_idx = np.triu_indices(n_ch, k=1)
    fc_vecs = np.empty((len(starts), len(triu_idx[0])), dtype=DTYPE)
    exp_cache = np.exp(1j * phases.astype(np.float32))
    for w, s in enumerate(starts):
        win_data = exp_cache[:, s:s + win]
        plv = np.abs(win_data @ win_data.conj().T / win)
        fc_vecs[w] = plv[triu_idx].real
    del exp_cache
    fc_z = fc_vecs - fc_vecs.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(fc_z, axis=1, keepdims=True)
    norms[norms == 0] = 1
    fcd = (fc_z / norms @ (fc_z / norms).T).astype(DTYPE)
    return fcd


def compute_coalition_entropy(phases, window_sec=2, step_sec=1, fs=FS_DOWN):
    n_ch, n_t = phases.shape
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    starts = list(range(0, n_t - win, step))
    coalitions = np.empty((len(starts), n_ch), dtype=np.uint8)
    for w, s in enumerate(starts):
        pw = phases[:, s:s + win].astype(DTYPE)
        cos_ph = np.cos(pw)
        sin_ph = np.sin(pw)
        coh = (cos_ph @ cos_ph.T + sin_ph @ sin_ph.T) / pw.shape[1]
        _, eigvecs = np.linalg.eigh(coh)
        coalitions[w] = (eigvecs[:, -1] > 0).astype(np.uint8)
    packed = np.packbits(coalitions, axis=1)
    patterns = [row.tobytes() for row in packed]
    unique, counts = np.unique(patterns, return_counts=True)
    probs = counts / counts.sum()
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = np.log2(len(coalitions))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_fcd_and_coalition(all_phases, datasets, primary_band='delta'):
    """Compute FCD variance and coalition entropy for all datasets."""
    fcd_results = {}
    coal_results = {}
    for ds_name in datasets:
        fcd_results[ds_name] = {}
        coal_results[ds_name] = {}
        for state in ['Awake', 'Unconscious']:
            if state not in all_phases[ds_name]:
                continue
            ph = all_phases[ds_name][state][primary_band]
            fcd_mat = compute_fcd(ph)
            triu = fcd_mat[np.triu_indices_from(fcd_mat, k=1)]
            fcd_results[ds_name][state] = float(np.var(triu))
            coal_results[ds_name][state] = compute_coalition_entropy(ph)
            print(f'{ds_name}/{state}: FCD_Var={fcd_results[ds_name][state]:.6f}, '
                  f'H_norm={coal_results[ds_name][state]:.4f}')
    return fcd_results, coal_results
