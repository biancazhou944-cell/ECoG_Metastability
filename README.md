# Large-scale cortical dynamics across anesthesia and sleep

This repository contains the analysis code for a secondary analysis of public
macaque electrocorticography (ECoG) recordings from NeuroTycho. Wakeful epochs
are compared with ketamine, medetomidine, propofol, and natural sleep epochs in
two macaques.

The primary analysis uses a fixed 64-channel subset. A separate robustness
analysis repeats the workflow using all 128 channels.

## Repository contents

```text
run_pipeline.py                    Primary 64-channel analysis
run_empirical_128_channels.py      128-channel robustness analysis
scripts/generate_paper_figures.py  Final main-figure assembly
src/                               Loading, preprocessing, metrics, statistics, plots
assets/electrode_layouts/          NeuroTycho two-dimensional electrode maps
results/                           Statistical output used by the figure script
figures/main_figures/              Three main figures
figures/supplementary/              Supplementary figures
paper/                             Final manuscript and supplementary PDFs
```

## Data

The raw recordings are not included in this repository. By default, the code
looks for them in a folder beside the repository named
`ECoG_Metastability_data/`. Download the anesthesia and sleep ECoG datasets
from [NeuroTycho](http://neurotycho.org/) and use this structure:

```text
ECoG_Metastability_data/
  20120712SLP_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
  20120717SLP_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/
  20120802PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/
  20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
  20120809MD_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/
  20120810KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
  20120813KT_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/
  20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
```

Alternatively, set `ECOG_DATA_DIR` to any directory containing the eight
dataset folders.

## Environment

The analysis was tested with Python 3.14.2. Create an isolated environment and
install the recorded package versions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the analysis

Run the scripts from the repository root in this order:

```bash
python run_pipeline.py
python run_empirical_128_channels.py
python scripts/generate_paper_figures.py
```

The first script writes the primary statistical report to
`results/stats_64_channels.txt`, creates the supplementary figures, and saves
the topographic source image used in main Figure 2. The second script writes
`results/stats_128_channels.txt`. The final script assembles the three main
figures in `figures/main_figures/`.

The saved result files and figures retain the values reported in the manuscript
and supplementary material. The primary script also retains the session-order
epoch-selection behavior used for the reported 64-channel analysis.

The full-epoch and 120-s analyses use the same dataset-specific bad-channel
masks. The 128-channel script detects bad channels again within the full
128-channel set.

## Code availability

The repository contains the code used for preprocessing, metric computation,
statistical analysis, robustness checks, and figure generation. Raw ECoG
recordings are not redistributed; they remain available from NeuroTycho. See
`CODE_AVAILABILITY.md` for a manuscript-ready statement.

## Reference dataset

Yanagawa T, Chao ZC, Hasegawa N, Fujii N. Large-scale information flow in
conscious and unconscious states: an ECoG study in monkeys. *PLoS ONE*.
2013;8:e80845. https://doi.org/10.1371/journal.pone.0080845
