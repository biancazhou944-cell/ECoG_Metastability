"""Check optimized wPLI against independent scalar calculations on real ECoG."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import stft

from src.config import DATASETS, FREQ_BANDS, FS, FS_DOWN
from src.data_loading import load_epoch
from src.metrics import compute_wpli_bands
from src.preprocessing import downsample_car
from src.pipeline import save_json


def main():
    result = json.loads((ROOT/'results/analysis_64.json').read_text())
    name = 'George_KT'
    good = result['datasets'][name]['good_channel_indices']
    audit = {'dataset': name, 'duration_sec': 120,
             'note': 'Four retained channels after the full 64-candidate QC/CAR reference; six pairs.',
             'states': {}}
    for state in ['Awake', 'Unconscious']:
        record = dict(result['epoch_manifest']['datasets'][name]['selected'][state])
        record['samples'] = 120*FS
        record['stop_sample'] = record['start_sample']+record['samples']
        data = load_epoch(DATASETS[name]['path'], record, result['channels'])
        referenced = downsample_car(data, good_channels=good)[:4]
        fast, nodes = compute_wpli_bands(referenced, FS_DOWN)
        frequencies, _, z = stft(referenced, fs=FS_DOWN, nperseg=512, noverlap=256,
                                 detrend='constant', boundary=None, padded=False, axis=-1)
        state_rows = {}
        for band, (lo, hi) in FREQ_BANDS.items():
            mask = (frequencies >= lo) & (frequencies <= hi)
            scalar_pairs = []
            for i in range(4):
                for j in range(i+1, 4):
                    imaginary = (z[i, mask]*z[j, mask].conj()).imag.astype(float)
                    spectrum = []
                    for observations in imaginary:
                        denominator = sum(abs(observations))
                        spectrum.append(abs(sum(observations))/denominator if denominator else 0)
                    scalar_pairs.append(np.mean(spectrum))
            expected = float(np.mean(scalar_pairs))
            error = abs(fast[band]-expected)
            if error > 1e-7:
                raise AssertionError((state, band, fast[band], expected))
            np.testing.assert_allclose(np.mean(nodes[band]), expected, atol=1e-7)
            state_rows[band] = {'optimized': fast[band], 'scalar_reference': expected,
                                'absolute_error': error}
        audit['states'][state] = state_rows
    audit['max_absolute_error'] = max(row['absolute_error'] for bands in audit['states'].values() for row in bands.values())
    save_json(ROOT/'results/validation/wpli_validation.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
