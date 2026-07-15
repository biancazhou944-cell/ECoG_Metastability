import re
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .config import FS, DTYPE, CHANNELS_TO_LOAD


def list_session_numbers(dataset_path):
    """Return available session numbers discovered from the dataset folder."""
    session_nums = []
    for path in dataset_path.iterdir():
        if path.is_dir() and path.name.startswith('Session'):
            suffix = path.name.replace('Session', '', 1)
            if suffix.isdigit():
                session_nums.append(int(suffix))
    return sorted(session_nums)


def parse_info_file(dataset_path):
    info_files = list(dataset_path.glob('Info-*.txt'))
    if not info_files:
        raise FileNotFoundError(f'No Info file in {dataset_path}')
    events = {}
    current_session = None
    with open(info_files[0], 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Session :'):
                current_session = int(line.split(':')[1].strip())
            elif line.startswith('time:') and current_session is not None:
                time_str = line.split('[s]')[0].replace('time:', '').strip()
                time_sec = float(time_str)
                label = line.split(': ')[-1].strip()
                events.setdefault(current_session, []).append((time_sec, label))
    return events


def load_ecog_session(dataset_path, session_num, channels=None):
    session_dir = dataset_path / f'Session{session_num}'
    if channels is None:
        channels = CHANNELS_TO_LOAD
    time_mat = sio.loadmat(str(session_dir / 'ECoGTime.mat'))
    time_key = [k for k in time_mat if not k.startswith('_')][0]
    time = time_mat[time_key].flatten().astype(DTYPE)
    data = np.zeros((len(channels), len(time)), dtype=DTYPE)
    for i, ch in enumerate(tqdm(channels, desc=f'S{session_num}', leave=False)):
        ch_mat = sio.loadmat(str(session_dir / f'ECoG_ch{ch}.mat'))
        ch_key = [k for k in ch_mat if not k.startswith('_')][0]
        data[i] = ch_mat[ch_key].flatten()[:len(time)].astype(DTYPE)
    return data, time


def extract_epochs(data, time, events):
    epochs = {}
    for t_start, label_start in events:
        if label_start.endswith('-Start'):
            cond_name = label_start.replace('-Start', '')
            for t_end, label_end in events:
                if label_end == cond_name + '-End':
                    i0 = max(0, min(int(t_start * FS), data.shape[1] - 1))
                    i1 = max(0, min(int(t_end * FS), data.shape[1]))
                    epochs[cond_name] = (data[:, i0:i1].copy(), time[i0:i1].copy())
                    break
    return epochs


def merge_epochs_by_longest(dst_epochs, src_epochs):
    """Merge epoch maps, keeping the longest continuous segment per condition."""
    for cond_name, epoch in src_epochs.items():
        if cond_name not in dst_epochs or epoch[0].shape[1] > dst_epochs[cond_name][0].shape[1]:
            dst_epochs[cond_name] = epoch
    return dst_epochs
