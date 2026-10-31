# Cortical network dynamics across wakefulness, anesthesia, and sleep

This project examines how cortical phase synchrony, metastability, and functional
connectivity differ between wakefulness and anesthesia or natural sleep. It uses
public NeuroTycho ECoG recordings from two macaques, George and Chibi, with three
anesthetic conditions (ketamine, medetomidine, and propofol) and a sleep recording
for each animal.

The analysis combines measures of whole-array phase alignment with pairwise
phase-lagged connectivity. Synchrony describes the average alignment of electrode
phases; metastability describes how that alignment varies over time. wPLI measures
consistent phase lags between electrode pairs. Sliding-window connectivity and
coalition entropy describe changes in connectivity patterns and the diversity of
channel partitions. Estimates are compared across 64/128-channel configurations
and full-duration/120-second epochs.

In the saved results, pooled theta synchrony and delta metastability differences
are negative across all four analysis settings, while broadband wPLI differences
are small. None of the 17 tests in any setting passes FDR correction. These are
exploratory comparisons from two animals; see [the analysis report](ANALYSIS_REPORT.md)
for effect sizes and limitations.

## Run

Python and package versions are recorded in each result JSON. Install the pinned
packages in an isolated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Then run both channel settings and generate figures:

```bash
.venv/bin/python run_pipeline.py --resume
.venv/bin/python run_empirical_128_channels.py --resume
.venv/bin/python scripts/generate_paper_figures.py
.venv/bin/python scripts/summarize_results.py
```

Both channel settings use the same analysis code. Results are written to
`results/` and figures to `figures/`. `--resume` reuses matching dataset checkpoints
from `results/.cache/`; see the technical notes below for the matching criteria.

For a short check on real data:

```bash
.venv/bin/python run_pipeline.py --datasets George_KT --durations 120 --output-dir output/smoke
```

A one-dataset run produces metrics without group statistics. Use `--help` for
channel count, datasets, durations, and output location. Requested durations must
be at least 8 seconds and fit entirely within each selected epoch.

## Data

Raw recordings are not redistributed. Download the anesthesia and sleep ECoG
MAT datasets from [NeuroTycho](http://neurotycho.org/). By default the loader uses
`../ECoG_Metastability_data/`. Alternatively, set `ECOG_DATA_DIR`.
The directory must contain these eight folders:

```text
20120712SLP_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
20120717SLP_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/
20120802PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/
20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
20120809MD_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/
20120810KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
20120813KT_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128/
20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/
```

For each state, the loader selects the longest valid continuous interval from the
session annotations. The 64- and 128-channel analyses use the same intervals.
Repeated start/end labels are paired chronologically within sessions; intervals
outside the recorded time axis are logged as rejected candidates. In the local
George sleep recording, Session 2's sleep end annotation exceeds the time axis,
so Session 1 supplies the selected sleep epoch. Each 120-second epoch is taken
from the start of its selected full interval.

## Methods

- **Preprocessing:** channel QC uses variance and correlation with the channel
  average over the first 10 seconds. Nonfinite and flat channels are excluded
  before forming that average. Channels with variance outside 0.01–10 times the
  median, an undefined correlation, or an absolute correlation below 0.05 are
  flagged. The union of the two states' bad-channel masks is applied to both states
  and both epoch durations. QC is performed separately for each channel setting.
  Data are downsampled from 1000 to 250 Hz and common-average referenced using the
  retained channels. Phase extraction uses fourth-order Butterworth bandpass
  filters applied forward and backward, followed by the Hilbert transform. Full
  and 120-second epochs are filtered separately. No post-filter edge trimming or
  temporal artifact rejection is applied.
- **Synchrony and metastability:** at each sample, compute the Kuramoto order
  parameter magnitude, `R(t) = abs(mean_channels exp(1j * phase(t)))`. Synchrony is
  its temporal mean and metastability its temporal variance. These summarize the
  phase concentration across all retained channels and its variation over time.
  Both are calculated in delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta
  (13–30 Hz), and broadband (1–50 Hz).
- **wPLI:** for each electrode pair and frequency, compute
  `abs(sum_window Im(Sxy)) / sum_window abs(Im(Sxy))`; average equally across
  frequencies within a band, then across unique channel pairs. A zero denominator
  produces zero. wPLI is calculated in the same five bands. STFT uses
  512 samples, 256 overlap, constant detrending, no extension/padding, at 250 Hz.
  See [Vinck et al. (2011)](https://doi.org/10.1016/j.neuroimage.2011.01.055) and
  [MNE's definition](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html).
- **Functional connectivity dynamics (FCD):** construct PLV matrices in 5-second
  windows with 3-second steps, then correlate their upper-triangular entries
  between windows. The reported metric is the variance of the off-diagonal FCD
  entries. It is calculated in the delta band.
- **Coalition entropy:** in 2-second windows with 1-second steps, partition channels
  by the signs of the leading eigenvector of the phase-coherence matrix. Orient
  its largest absolute loading positive so that `v` and `-v` give the same
  partition; near-zero loadings are assigned zero using a floating-point
  tolerance. The Shannon entropy of partition frequencies is divided by
  `log2(number of windows)`. This measure is also calculated in the delta band.

Both sliding-window measures include the last complete window. FCD requires at
least two windows with nonconstant connectivity vectors.

## Statistical interpretation

Each animal–condition recording contributes a paired difference, defined as
anesthesia/sleep minus awake. The analysis reports paired Cohen's d, exact
sign-flip tests, bootstrap 95% confidence intervals, and two-sided paired JZS Bayes
factors (`BF10`, Cauchy effect-size scale `r=0.707`; Rouder et al., 2009, equation 1).
FDR correction is applied across the 17 metric/band combinations separately for
each channel-count and duration setting. Numerical checks for the Bayes factor
use a [Pingouin reference example](https://github.com/raphaelvallat/pingouin/blob/master/src/pingouin/bayesian.py)
and an analytical zero-t case.

The dataset contains **two animals contributing eight animal–condition pairs**.
Repeated conditions within animals limit the independence of those pairs, so
pair-level p values, intervals, and Bayes factors are exploratory. Condition
summaries, leave-one-condition/animal effects, and a sign-flip test of animal-mean
differences provide additional checks. With only two animal means, the smallest
possible two-sided sign-flip p value is 0.5. Population-level inference would
require more animals.

Statistical calculations require complete, finite paired observations in dataset
order. Undefined standardized effects are stored as JSON `null`. Bootstrap
resamples with undefined effect sizes are counted, warned about, and excluded
from percentile estimation. Figures require finite effect sizes and intervals.

## Outputs and traceability

```text
src/                         Analysis, preprocessing, statistics and plotting
scripts/                     Figure generation, reporting and validation
tests/                       Scientific and end-to-end regression tests
assets/electrode_layouts/     Electrode coordinates
results/
  analysis_64.json            Full results, QC and epoch provenance
  analysis_128.json
  statistics_64.csv           Statistical summaries
  statistics_128.csv
  validation/                Preprocessing and wPLI validation records
figures/
  metric_effects.png/.pdf     Metric effect sizes and intervals
  connectivity.png/.pdf      Connectivity and spatial distributions
  robustness.png/.pdf        Channel-count and duration comparisons
  supplementary/             Electrode layouts and condition shifts
  provenance.json            Figure inputs and plotting source hashes
ANALYSIS_REPORT.md            Current result summary
```

JSON files contain per-state and per-dataset measurements, channel IDs, QC records,
selected epochs, analysis parameters, software versions, source hashes, and timing.
CSV files summarize the statistical results. Figures are drawn from the JSON
files; input hashes and color limits are recorded in `figures/provenance.json`.

## Technical notes

Checkpoints contain metrics and QC records. Reuse requires matching input file
sizes and modification times, annotations, channel IDs, durations, package
versions, and computation-source hashes. Cache files and run logs are stored in
`results/.cache/` and excluded from Git.

`source_sha256` records the code used for computation. Hashes from later file-layout
and docstring revisions are stored separately as `layout_source_sha256` and
`documentation_source_sha256`, with notes describing those revisions.

The runner holds one dataset at a time and one frequency band's phases at a time.
STFT stores only requested frequencies and processes channel pairs in small blocks.
Loading avoids irrelevant sessions and retains only selected samples from each
MAT channel file (MAT v5 files still require decompressing that channel).
For reproducible single-thread timing, set `OPENBLAS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `OMP_NUM_THREADS=1` before launching Python.

## Dataset reference

Yanagawa T, Chao ZC, Hasegawa N, Fujii N (2013).
*Large-scale information flow in conscious and unconscious states: an ECoG study
in monkeys*. PLoS ONE 8:e80845. https://doi.org/10.1371/journal.pone.0080845
