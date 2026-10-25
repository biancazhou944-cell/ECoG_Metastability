"""Numerical regressions for scientific definitions and data-integrity failures."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import signal
from scipy.io import savemat
from scipy.stats import false_discovery_control
from scipy.special import erfcx

from src.config import FREQ_BANDS
from src.data_loading import extract_epochs, build_epoch_manifest, load_epoch
from src.metrics import (_wpli_from_stft_band, compute_wpli_bands, compute_coalition_entropy,
                         compute_fcd, fcd_variance, _canonical_coalition)
from src.preprocessing import channel_quality, downsample_car, preprocess_car
from src.statistics import (bayes_factor_paired, benjamini_hochberg, permutation_test,
                            collect_tests, compute_group_diagnostics)


class ConnectivityTests(unittest.TestCase):
    def coefficients(self, imaginary):
        imaginary = np.asarray(imaginary, dtype=np.float32)
        z = np.ones((2, *imaginary.shape), dtype=np.complex64)
        z[1] = -1j*imaginary
        return z

    def test_time_sign_cancellation(self):
        value, nodes = _wpli_from_stft_band(self.coefficients([[1, -1, 1, -.9]]*2))
        self.assertAlmostEqual(value, .1/3.9, places=7)
        np.testing.assert_allclose(nodes, value)

    def test_opposite_frequency_directions_do_not_cancel(self):
        value, _ = _wpli_from_stft_band(self.coefficients([[1]*4, [-1]*4]))
        self.assertEqual(value, 1)

    def test_amplitude_scale_invariance(self):
        z = self.coefficients([[1, -1, 1, -.9]]*2)
        self.assertAlmostEqual(_wpli_from_stft_band(z)[0], _wpli_from_stft_band(z*1e-8)[0], places=6)

    def test_zero_imaginary_power(self):
        self.assertEqual(_wpli_from_stft_band(np.ones((3, 2, 4), complex))[0], 0)

    def test_stft_against_independent_scalar_reference(self):
        data = np.random.default_rng(8).normal(size=(5, 5000)).astype(np.float32)
        results, nodes = compute_wpli_bands(data, 250)
        f, _, z = signal.stft(data, fs=250, nperseg=512, noverlap=256,
                             detrend='constant', boundary=None, padded=False, axis=-1)
        for band, (lo, hi) in FREQ_BANDS.items():
            mask = (f >= lo) & (f <= hi)
            matrix = np.zeros((5, 5))
            for i in range(5):
                for j in range(i+1, 5):
                    per_frequency = []
                    for frequency in np.flatnonzero(mask):
                        im = (z[i, frequency]*z[j, frequency].conj()).imag.astype(float)
                        per_frequency.append(abs(sum(im))/sum(abs(im)) if sum(abs(im)) else 0)
                    matrix[i, j] = matrix[j, i] = np.mean(per_frequency)
            self.assertAlmostEqual(results[band], matrix[np.triu_indices(5, 1)].mean(), places=7)
            np.testing.assert_allclose(nodes[band], matrix.sum(axis=1)/4, rtol=1e-6)

    def test_short_signal_rejected(self):
        with self.assertRaises(ValueError):
            compute_wpli_bands(np.ones((3, 600)), 250)


class DynamicsTests(unittest.TestCase):
    def test_window_endpoint_included(self):
        phases = np.random.default_rng(4).uniform(-np.pi, np.pi, (4, 2000)).astype(np.float32)
        matrix = compute_fcd(phases)
        self.assertEqual(matrix.shape, (2, 2))
        np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-6)

    def test_insufficient_fcd_windows_rejected(self):
        phases = np.random.default_rng(4).uniform(-np.pi, np.pi, (4, 1500)).astype(np.float32)
        with self.assertRaises(ValueError):
            fcd_variance(phases)

    def test_coalitions_ignore_global_sign_and_zero_loadings(self):
        vector = np.array([.5, -.4, 0, .1], dtype=np.float32)
        np.testing.assert_array_equal(_canonical_coalition(vector), _canonical_coalition(-vector))

    def test_entropy_is_invariant_to_solver_signs(self):
        import src.metrics as metrics
        phases = np.tile(np.array([0, 0, np.pi, np.pi], dtype=np.float32)[:, None], (1, 1500))
        original = metrics.eigh
        calls = [0]
        def alternating(*args, **kwargs):
            values, vectors = original(*args, **kwargs)
            calls[0] += 1
            vectors *= -1 if calls[0] % 2 else 1
            return values, vectors
        baseline = compute_coalition_entropy(phases)
        with patch('src.metrics.eigh', side_effect=alternating):
            self.assertEqual(compute_coalition_entropy(phases), baseline)
        self.assertEqual(baseline, 0)


class StatisticalTests(unittest.TestCase):
    def test_jzs_published_paired_example(self):
        # Pingouin documentation, t=3.5, n=20, paired=True, r=.707: BF10=17.185.
        d = np.arange(20, dtype=float)
        d = (d-d.mean())/d.std(ddof=1)+3.5/np.sqrt(20)
        self.assertAlmostEqual(bayes_factor_paired(np.zeros(20), d), 17.185, places=3)

    def test_jzs_zero_t_reference(self):
        d = np.arange(8, dtype=float)-3.5
        # For t=0 the JZS integral reduces analytically to erfcx(sqrt(n*r^2/2)).
        self.assertAlmostEqual(bayes_factor_paired(np.zeros(8), d), erfcx(np.sqrt(8*.707**2/2)), places=10)

    def test_exact_signflip_resolution(self):
        self.assertEqual(permutation_test(np.zeros(8), np.arange(1, 9))[0], 2/256)
        self.assertEqual(permutation_test(np.zeros(4), [-2, -1, 1, 2])[0], 1)

    def test_bh_matches_scipy(self):
        p = np.r_[0, 1, np.random.default_rng(42).uniform(size=30)]
        np.testing.assert_allclose(benjamini_hochberg(p)[0], false_discovery_control(p))
        self.assertEqual(benjamini_hochberg([]), ([], []))

    def test_invalid_pairs_rejected(self):
        for left, right in [([], []), ([1], [1, 2]), ([1, np.nan], [1, 2])]:
            with self.assertRaises(ValueError):
                permutation_test(left, right)

    def test_missing_state_is_not_zero(self):
        datasets = {'A': {}, 'B': {}}
        kur = {name: {state: {'delta': {'sync': .1, 'meta': .2}} for state in ['Awake', 'Unconscious']}
               for name in datasets}
        wpli = {'A': {'Awake': {'delta': .1}, 'Unconscious': {'delta': .2}},
                'B': {'Awake': {'delta': .1}}}
        with self.assertRaisesRegex(ValueError, 'B/Unconscious/wPLI'):
            collect_tests(kur, wpli, {}, {}, datasets, ['delta'])

    def test_group_alignment_checked(self):
        datasets = {'A': {'monkey': 'G', 'condition': 'KT'}, 'B': {'monkey': 'C', 'condition': 'KT'}}
        with self.assertRaisesRegex(ValueError, 'complete pairs'):
            compute_group_diagnostics([('x', 'delta', [.1], [.2], 1., 1.)], datasets)


class LoadingTests(unittest.TestCase):
    def test_repeated_labels_choose_longest_valid_pair(self):
        data = np.arange(6000)[None, :]
        times = np.arange(6000)/1000
        epochs = extract_epochs(data, times, [(1, 'X-Start'), (2, 'X-End'), (3, 'X-Start'), (5, 'X-End')])
        np.testing.assert_array_equal(epochs['X'][0], data[:, 3000:5000])

    def test_actual_time_offset_used(self):
        data = np.arange(4000)[None, :]
        time = 10+np.arange(4000)/1000
        epochs = extract_epochs(data, time, [(11, 'X-Start'), (12, 'X-End')])
        np.testing.assert_array_equal(epochs['X'][0], data[:, 1000:2000])

    def test_out_of_range_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError, 'exceeds recording'):
            extract_epochs(np.zeros((3, 1000)), np.arange(1000)/1000, [(0, 'X-Start'), (2, 'X-End')])

    def test_manifest_selects_valid_longest_and_records_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for session in [1, 2, 3]:
                (root/f'Session{session}').mkdir()
                savemat(root/f'Session{session}'/'ECoGTime.mat', {'time': np.arange(10000)/1000})
            (root/'Info-test.txt').write_text('Session : 1\ntime: 1 [s] : Sleeping-Start\ntime: 3 [s] : Sleeping-End\n'
                'Session : 2\ntime: 0 [s] : Sleeping-Start\ntime: 4 [s] : Sleeping-End\n'
                'time: 5 [s] : AwakeEyesOpened-Start\ntime: 8 [s] : AwakeEyesOpened-End\n'
                'Session : 3\ntime: 0 [s] : Sleeping-Start\ntime: 100 [s] : Sleeping-End\n')
            with self.assertWarns(UserWarning):
                manifest = build_epoch_manifest({'George_SLP': {'path': root}})
            entry = manifest['datasets']['George_SLP']
            self.assertEqual(entry['selected']['Unconscious']['session'], 2)
            self.assertEqual(len(entry['rejected']), 1)
            savemat(root/'Session2'/'ECoG_ch1.mat', {'x': np.arange(10000)})
            loaded = load_epoch(root, entry['selected']['Unconscious'], [1])
            np.testing.assert_array_equal(loaded[0], np.arange(4000))


class PreprocessingTests(unittest.TestCase):
    def test_shared_preprocessing_matches_original_operations(self):
        data = np.random.default_rng(2).normal(size=(7, 12000)).astype(np.float32)
        good = [0, 2, 3, 6]
        d = signal.decimate(data, 4, axis=1).astype(np.float32)
        expected = (d-d[good].mean(axis=0, keepdims=True))[good]
        np.testing.assert_allclose(downsample_car(data, good_channels=good), expected, atol=1e-7)
        for band, (lo, hi) in FREQ_BANDS.items():
            sos = signal.butter(4, [lo/125, hi/125], btype='band', output='sos')
            filtered = signal.sosfiltfilt(sos, expected, axis=1).astype(np.float32)
            np.testing.assert_allclose(preprocess_car(data, band, good_channels=good), filtered, atol=1e-6)

    def test_nonfinite_channel_does_not_poison_all_channels(self):
        rng = np.random.default_rng(4)
        common = rng.normal(size=4000)
        data = common+rng.normal(scale=.05, size=(6, 4000))
        data[0] = np.nan
        bad, report = channel_quality(data)
        self.assertEqual(bad, {0})
        self.assertIn('nonfinite', report['reasons'][0])

    def test_noninteger_downsampling_rejected(self):
        with self.assertRaises(ValueError):
            downsample_car(np.ones((4, 1000)), fs_down=300)


if __name__ == '__main__':
    unittest.main()
