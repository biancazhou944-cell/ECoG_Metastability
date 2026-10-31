"""Validated event selection and loading of only the selected ECoG segments."""

import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

from .config import FS, DTYPE, CHANNELS_TO_LOAD, STATE_MAP


def list_session_numbers(dataset_path):
    return sorted(int(p.name[7:]) for p in Path(dataset_path).iterdir()
                  if p.is_dir() and re.fullmatch(r'Session\d+', p.name))


def parse_info_file(dataset_path):
    files = sorted(Path(dataset_path).glob('Info-*.txt'))
    if len(files) != 1:
        raise ValueError(f'Expected exactly one Info file in {dataset_path}, found {len(files)}')
    events = {}
    session = None
    with files[0].open(encoding='utf-8-sig') as source:
        for line in source:
            line = line.strip()
            if line.startswith('Session :'):
                session = int(line.split(':', 1)[1])
            elif line.startswith('time:'):
                match = re.fullmatch(r'time:\s*([+-]?[\d.]+)\s*\[s\]\s*:\s*(.+)', line)
                if match is None or session is None:
                    raise ValueError(f'Malformed event in {files[0]}: {line}')
                events.setdefault(session, []).append((float(match[1]), match[2].strip()))
    return events


def event_intervals(events, labels=None):
    """Pair each start with its following end; keep repeated intervals separate."""
    pending = {}
    intervals = []
    for timestamp, tag in sorted(events, key=lambda item: item[0]):
        if not np.isfinite(timestamp):
            raise ValueError('Non-finite event time')
        if not tag.endswith(('-Start', '-End')):
            continue
        label, kind = tag.rsplit('-', 1)
        if labels is not None and label not in labels:
            continue
        if kind == 'Start':
            if label in pending:
                raise ValueError(f'Overlapping starts for {label}')
            pending[label] = timestamp
        else:
            if label not in pending:
                raise ValueError(f'End without start for {label}')
            start = pending.pop(label)
            if timestamp <= start:
                raise ValueError(f'Non-positive interval for {label}')
            intervals.append((label, start, timestamp))
    if pending:
        raise ValueError(f'Unclosed event intervals: {sorted(pending)}')
    return intervals


def _load_vector(path):
    mat = sio.loadmat(path)
    values = [v for k, v in mat.items() if not k.startswith('_')]
    if len(values) != 1 or values[0].ndim != 2 or min(values[0].shape) != 1:
        raise ValueError(f'Expected one vector in {path}')
    return values[0].reshape(-1)


def validate_time_axis(time, fs=FS):
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError('Time axis must be a finite vector with at least two samples')
    if not np.allclose(np.diff(time), 1 / fs, rtol=1e-5, atol=1e-8):
        raise ValueError(f'Time axis is not uniformly sampled at {fs} Hz')


def interval_bounds(time, start, end, fs=FS):
    """Use a half-open interval on the actual time axis; reject out-of-range labels."""
    tolerance = 1e-7
    if start < time[0] - tolerance or end > time[-1] + 1 / fs + tolerance:
        raise ValueError(f'Annotation [{start}, {end}) exceeds recording '
                         f'[{time[0]}, {time[-1] + 1 / fs})')
    i0 = int(np.searchsorted(time, start - tolerance))
    i1 = int(np.searchsorted(time, end - tolerance))
    if i1 <= i0:
        raise ValueError('Empty epoch')
    return i0, i1


def build_epoch_manifest(datasets):
    """Choose longest valid continuous epoch, independently of channel count.

    Out-of-range intervals are logged as rejected candidates with a warning.
    Ties are resolved by session number and start time (earliest first).
    """
    manifest = {'policy': 'longest_valid_continuous', 'fs': FS, 'datasets': {}}
    for name, dataset in datasets.items():
        path = Path(dataset['path'])
        events = parse_info_file(path)
        info = next(path.glob('Info-*.txt'))
        states = STATE_MAP[name]
        candidates = {state: [] for state in states}
        rejected = []
        for session in list_session_numbers(path):
            intervals = event_intervals(events.get(session, []), set(states.values()))
            if not intervals:
                continue
            time = _load_vector(path / f'Session{session}' / 'ECoGTime.mat').astype(np.float64)
            validate_time_axis(time)
            for label, start, end in intervals:
                state = next(key for key, value in states.items() if value == label)
                record = {'session': session, 'label': label, 'start_sec': start,
                          'end_sec': end, 'session_samples': len(time)}
                try:
                    i0, i1 = interval_bounds(time, start, end)
                except ValueError as exc:
                    rejected.append({**record, 'reason': str(exc)})
                    warnings.warn(f'{name}/Session{session}: {exc}', stacklevel=2)
                    continue
                record.update(start_sample=i0, stop_sample=i1, samples=i1-i0,
                              duration_sec=(i1-i0)/FS, actual_start_sec=float(time[i0]))
                candidates[state].append(record)
        selected = {}
        for state, records in candidates.items():
            if not records:
                raise ValueError(f'{name}: no valid {state} epoch')
            selected[state] = sorted(records, key=lambda r: (-r['samples'], r['session'], r['start_sec']))[0]
        manifest['datasets'][name] = {
            'selected': selected, 'candidates': candidates, 'rejected': rejected,
            'info_sha256': hashlib.sha256(info.read_bytes()).hexdigest(),
        }
    return manifest


def load_epoch(dataset_path, record, channels=None):
    """Read channels one at a time, retaining only the selected sample interval."""
    channels = CHANNELS_TO_LOAD if channels is None else list(channels)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError('Channel IDs must be non-empty and unique')
    data = np.empty((len(channels), record['samples']), dtype=DTYPE)
    session_dir = Path(dataset_path) / f"Session{record['session']}"
    for index, channel in enumerate(channels):
        values = _load_vector(session_dir / f'ECoG_ch{channel}.mat')
        if len(values) != record['session_samples']:
            raise ValueError(f'{session_dir}/ECoG_ch{channel}: signal/time length mismatch')
        data[index] = values[record['start_sample']:record['stop_sample']]
    return data


def load_ecog_session(dataset_path, session_num, channels=None):
    """Compatibility API for callers needing an entire session."""
    time = _load_vector(Path(dataset_path) / f'Session{session_num}' / 'ECoGTime.mat').astype(np.float64)
    validate_time_axis(time)
    record = dict(session=session_num, samples=len(time), session_samples=len(time),
                  start_sample=0, stop_sample=len(time))
    return load_epoch(dataset_path, record, channels), time


def extract_epochs(data, time, events):
    """Keep the longest valid interval for each label, including within a session."""
    validate_time_axis(time)
    if data.ndim != 2 or data.shape[1] != len(time):
        raise ValueError('Signal/time shape mismatch')
    epochs = {}
    for label, start, end in event_intervals(events):
        i0, i1 = interval_bounds(time, start, end)
        if label not in epochs or i1-i0 > epochs[label][0].shape[1]:
            epochs[label] = data[:, i0:i1].copy(), time[i0:i1].copy()
    return epochs


def merge_epochs_by_longest(dst_epochs, src_epochs):
    for label, epoch in src_epochs.items():
        if label not in dst_epochs or epoch[0].shape[1] > dst_epochs[label][0].shape[1]:
            dst_epochs[label] = epoch
    return dst_epochs
