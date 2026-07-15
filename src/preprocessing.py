import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate, hilbert

from .config import FS, FS_DOWN, N_CH, DTYPE, FREQ_BANDS, ANALYSIS_BANDS


def detect_bad_channels(all_raw, datasets):
    """Detect bad channels per dataset using variance and correlation criteria."""
    bad_ch_all = {}
    good_ch_all = {}
    for ds_name in datasets:
        bad_union = set()
        for state in ['Awake', 'Unconscious']:
            if state not in all_raw[ds_name]:
                continue
            data, _ = all_raw[ds_name][state]
            ch_var = np.var(data, axis=1)
            median_var = np.median(ch_var)
            bad_var = set(np.where((ch_var > 10 * median_var) | (ch_var < 0.01 * median_var))[0])
            n_samp = min(data.shape[1], 10 * FS)
            mean_sig = data[:, :n_samp].mean(axis=0)
            corrs = np.array([np.corrcoef(data[ch, :n_samp], mean_sig)[0, 1]
                              for ch in range(data.shape[0])])
            bad_corr = set(np.where(np.abs(corrs) < 0.05)[0])
            bad_union |= bad_var | bad_corr
        bad_ch_all[ds_name] = bad_union
        good_ch_all[ds_name] = sorted(set(range(N_CH)) - bad_union)
        print(f'{ds_name:15s}: {len(bad_union):2d} bad -> {len(good_ch_all[ds_name])}/{N_CH} good')
    return bad_ch_all, good_ch_all


def preprocess_car(data, band, fs_orig=FS, fs_down=FS_DOWN, good_channels=None):
    """Downsample -> CAR (over good channels) -> bandpass -> return ONLY good channels."""
    factor = int(fs_orig / fs_down)
    d = decimate(data, factor, axis=1).astype(DTYPE)
    if good_channels is not None:
        car = d[good_channels].mean(axis=0, keepdims=True)
    else:
        car = d.mean(axis=0, keepdims=True)
    d = d - car
    if good_channels is not None:
        d = d[good_channels]
    lo, hi = FREQ_BANDS[band]
    nyq = fs_down / 2
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
    d = sosfiltfilt(sos, d, axis=1).astype(DTYPE)
    return d


def compute_all_phases(all_raw, datasets, good_ch_all, duration_sec=None):
    """Preprocess all data: optional truncation -> CAR -> Multi-Band -> Hilbert phase."""
    all_phases = {}
    for ds_name in datasets:
        good = good_ch_all[ds_name]
        all_phases[ds_name] = {}
        for state in ['Awake', 'Unconscious']:
            if state not in all_raw[ds_name]:
                continue
            data, _ = all_raw[ds_name][state]
            if duration_sec is not None:
                max_samp = min(data.shape[1], int(duration_sec * FS))
                data = data[:, :max_samp]
            all_phases[ds_name][state] = {}
            for band in ANALYSIS_BANDS:
                d = preprocess_car(data, band, good_channels=good)
                ph = np.empty_like(d)
                for ch in range(d.shape[0]):
                    ph[ch] = np.angle(hilbert(d[ch])).astype(DTYPE)
                all_phases[ds_name][state][band] = ph
                del d
        print(f'{ds_name}: {len(good)} good channels, done')
    return all_phases
