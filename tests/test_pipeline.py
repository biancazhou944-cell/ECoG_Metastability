"""End-to-end MAT loading, structured output and checkpoint invalidation."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from src.config import CHANNELS_TO_LOAD
from src.pipeline import run_analysis, compute_dataset, save_json


class PipelineTests(unittest.TestCase):
    def test_mat_run_resume_and_changed_input_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root/'data'
            session = data_dir/'Session1'
            session.mkdir(parents=True)
            time = np.arange(16000)/1000
            savemat(session/'ECoGTime.mat', {'time': time})
            (data_dir/'Info-test.txt').write_text('Session : 1\n'
                'time: 0 [s] : AwakeEyesOpened-Start\ntime: 8 [s] : AwakeEyesOpened-End\n'
                'time: 8 [s] : Anesthetized-Start\ntime: 16 [s] : Anesthetized-End\n')
            rng = np.random.default_rng(123)
            common = rng.normal(size=len(time))
            for channel in CHANNELS_TO_LOAD:
                signal = common+rng.normal(size=len(time))*.5
                savemat(session/f'ECoG_ch{channel}.mat', {'signal': signal.astype(np.float32)})
            datasets = {'George_KT': {'path': data_dir, 'monkey': 'George', 'condition': 'KT'}}
            kwargs = {'output_dir': root/'results', 'dataset_names': ['George_KT'], 'durations': {'8s': 8.0}}
            with patch('src.pipeline.DATASETS', datasets), contextlib.redirect_stdout(io.StringIO()):
                first = run_analysis(**kwargs)
                self.assertEqual(first['status'], 'metrics_only')
                self.assertEqual(len(first['datasets']['George_KT']['metrics']['8s']['Awake']['wpli']), 5)
                with patch('src.pipeline.compute_dataset', side_effect=AssertionError('Should reuse checkpoint')):
                    second = run_analysis(**kwargs, resume=True)
                self.assertEqual(second['runtime']['reused_datasets'], ['George_KT'])
                changed = session/f'ECoG_ch{CHANNELS_TO_LOAD[0]}.mat'
                stat = changed.stat()
                os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000_000))
                with patch('src.pipeline.compute_dataset', wraps=compute_dataset) as recompute:
                    third = run_analysis(**kwargs, resume=True)
                self.assertEqual(recompute.call_count, 1)
                self.assertEqual(third['runtime']['reused_datasets'], [])
            parsed = json.loads((root/'results'/'analysis_64.json').read_text())
            self.assertEqual(parsed['epoch_manifest']['datasets']['George_KT']['selected']['Awake']['samples'], 8000)

    def test_json_uses_standard_null_for_undefined_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.json'
            save_json(path, {'values': np.array([1, np.nan, np.inf]), 'ok': np.bool_(True)})
            self.assertEqual(json.loads(path.read_text()), {'values': [1.0, None, None], 'ok': True})
            self.assertFalse(path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
