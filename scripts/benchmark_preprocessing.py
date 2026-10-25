"""Benchmark repeated SciPy preprocessing against the shared preprocessing stage."""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(mode, directory):
    import numpy as np
    import scipy
    from scipy.signal import butter, decimate, sosfiltfilt
    from src.config import ANALYSIS_BANDS, FREQ_BANDS, FS, FS_DOWN
    from src.preprocessing import downsample_car, bandpass_referenced
    data = np.random.default_rng(42).standard_normal((64, 120000), dtype=np.float32)
    good = list(range(54))
    def calculate():
        if mode == 'reference':
            # Independently calculate the repeated operations with SciPy.
            result = {}
            for band in ANALYSIS_BANDS:
                d = decimate(data, FS//FS_DOWN, axis=1).astype(np.float32)
                referenced = (d-d[good].mean(axis=0, keepdims=True))[good]
                lo, hi = FREQ_BANDS[band]
                sos = butter(4, [lo/(FS_DOWN/2), hi/(FS_DOWN/2)],
                             btype='band', output='sos')
                result[band] = sosfiltfilt(sos, referenced, axis=1).astype(np.float32)
            d = decimate(data, FS//FS_DOWN, axis=1).astype(np.float32)
            result['wpli_input'] = (d-d[good].mean(axis=0, keepdims=True))[good]
        else:
            referenced = downsample_car(data, good_channels=good)
            result = {band: bandpass_referenced(referenced, band) for band in ANALYSIS_BANDS}
            result['wpli_input'] = referenced
        return result
    result = calculate()
    del result
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        result = calculate()
        timings.append(time.perf_counter()-started)
        del result
    result = calculate()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    np.savez(directory/f'{mode}.npz', **result)
    metadata = {'mode': mode, 'seconds': timings, 'median_seconds': statistics.median(timings),
                'peak_rss_bytes': peak if sys.platform == 'darwin' else peak*1024,
                'numpy': np.__version__, 'scipy': scipy.__version__, 'python': sys.version.split()[0]}
    (directory/f'{mode}.json').write_text(json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['reference', 'shared'])
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results/validation/preprocessing_benchmark.json')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.directory)
        return
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='ecog-benchmark-') as directory:
        directory = Path(directory)
        for mode in ['reference', 'shared']:
            subprocess.run([sys.executable, __file__, '--worker', mode, '--directory', str(directory)], env=environment, check=True)
        import numpy as np
        from src.pipeline import save_json
        reference = json.loads((directory/'reference.json').read_text())
        shared = json.loads((directory/'shared.json').read_text())
        errors = {}
        with np.load(directory/'reference.npz') as before, np.load(directory/'shared.npz') as after:
            for key in before.files:
                np.testing.assert_allclose(before[key], after[key], atol=1e-6, rtol=1e-5)
                errors[key] = float(np.max(np.abs(before[key]-after[key])))
        report = {'scope': 'Synthetic 64-channel, 120-second, 1000-Hz input; 54 retained channels; five bandpass outputs plus wPLI preprocessing. Excludes I/O, Hilbert, metric calculation and plotting.',
                  'method': 'Separate single-thread processes, one warm-up, median of five repeats; same input seed and filters. The reference repeats preprocessing per band using SciPy.',
                  'reference': reference, 'shared': shared, 'max_abs_errors': errors,
                  'speedup': reference['median_seconds']/shared['median_seconds'],
                  'rss_reduction_fraction': 1-shared['peak_rss_bytes']/reference['peak_rss_bytes']}
        save_json(args.output, report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
