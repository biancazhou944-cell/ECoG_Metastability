import os
from pathlib import Path

import numpy as np

DTYPE = np.float32
FS = 1000
FS_DOWN = 250
N_CH = 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / 'ECoG_Metastability_data'
DATA_DIR = Path(os.environ.get('ECOG_DATA_DIR', DEFAULT_DATA_DIR)).expanduser()
CHANNELS_TO_LOAD = np.linspace(1, 128, N_CH, dtype=int).tolist()

FREQ_BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'broadband': (1, 50),
}
ANALYSIS_BANDS = ['delta', 'theta', 'alpha', 'beta', 'broadband']

COND_COLORS = {'KT': '#FF9800', 'MD': '#9C27B0', 'PF': '#009688', 'SLP': '#607D8B'}
MONKEY_MARKERS = {'George': 'o', 'Chibi': 's'}
COND_TO_DRUG = {'KT': 'Ketamine', 'MD': 'Medetomidine', 'PF': 'Propofol', 'SLP': 'Sleep'}

DATASETS = {
    'George_KT':  {'path': DATA_DIR / '20120810KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', 'monkey': 'George', 'condition': 'KT'},
    'George_MD':  {'path': DATA_DIR / '20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', 'monkey': 'George', 'condition': 'MD'},
    'George_PF':  {'path': DATA_DIR / '20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', 'monkey': 'George', 'condition': 'PF'},
    'George_SLP': {'path': DATA_DIR / '20120712SLP_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', 'monkey': 'George', 'condition': 'SLP'},
    'Chibi_KT':   {'path': DATA_DIR / '20120813KT_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128', 'monkey': 'Chibi', 'condition': 'KT'},
    'Chibi_MD':   {'path': DATA_DIR / '20120809MD_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128', 'monkey': 'Chibi', 'condition': 'MD'},
    'Chibi_PF':   {'path': DATA_DIR / '20120802PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128', 'monkey': 'Chibi', 'condition': 'PF'},
    'Chibi_SLP':  {'path': DATA_DIR / '20120717SLP_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128', 'monkey': 'Chibi', 'condition': 'SLP'},
}

STATE_MAP = {}
for name, ds in DATASETS.items():
    if ds['condition'] == 'SLP':
        STATE_MAP[name] = {'Awake': 'AwakeEyesOpened', 'Unconscious': 'Sleeping'}
    else:
        STATE_MAP[name] = {'Awake': 'AwakeEyesOpened', 'Unconscious': 'Anesthetized'}
