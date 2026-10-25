# Large-scale cortical dynamics across anesthesia and sleep

Analysis of public NeuroTycho macaque ECoG recordings: awake versus ketamine,
medetomidine, propofol, and natural sleep in George and Chibi.

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

The scripts use the same runner, epoch selection, preprocessing, and definitions.
The default outputs are `results/` and `figures/`.
`--resume` reuses a dataset checkpoint only when selected inputs (file size and
mtime), annotation metadata, channel IDs, durations, package versions, and analysis
source hashes match. Checkpoints and run logs live in `results/.cache/`, which is
ignored by Git. Checkpoints hold metrics and QC, not raw ECoG or phase arrays.

For a short check on real data:

```bash
.venv/bin/python run_pipeline.py --datasets George_KT --durations 120 --output-dir output/smoke
```

A one-dataset run produces metrics only; it does not perform group inference.
Use `--help` to see options for channel count, datasets, durations, and output
location. A duration must be at least 8 seconds and fit entirely inside each
selected epoch; short data are not silently treated as duration matched.

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

The loader checks time axes and channel/time lengths. It selects the longest
**valid continuous** interval for each state, using the same selection for 64 and
128 channels. Repeated labels are paired chronologically within sessions. Invalid
out-of-range candidates are recorded and warned about; if either state has no
valid candidate, the analysis stops. For the local George sleep recording,
Session 2's sleep end annotation exceeds the available time axis and is rejected.
Session 1 supplies the valid longest sleep epoch in both channel settings.

## Methods

- **wPLI:** at each frequency, compute
  `abs(sum_window Im(Sxy)) / sum_window abs(Im(Sxy))`; average equally across
  frequencies within a band, then across unique channel pairs. A zero denominator
  produces zero. There is no amplitude-dependent absolute threshold. STFT uses
  512 samples, 256 overlap, constant detrending, no extension/padding, at 250 Hz.
  All bands share one set of per-frequency pair estimates. See
  [MNE's definition](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html).
- **Bayes factor:** two-sided paired JZS BF10, Cauchy effect-size scale `r=0.707`,
  with the inverse-gamma mixture from Rouder et al. (2009), equation 1. Numerical
  tests use the [Pingouin reference example](https://github.com/raphaelvallat/pingouin/blob/master/src/pingouin/bayesian.py)
  and an analytical zero-t case.
- **Coalition entropy:** the leading eigenvector's largest absolute loading is
  oriented positive; `v` and `-v` therefore encode the same partition. Near-zero
  loadings are assigned zero using a floating-point tolerance. Shannon entropy
  is divided by `log2(number of windows)`.
- **Windowing:** include the last complete window. FCD uses 5-s windows, 3-s steps;
  coalition entropy uses 2-s windows, 1-s steps. Undefined FCD correlations or
  insufficient windows raise errors instead of generating silent NaNs.
- **Preprocessing:** variance and first-10-s correlation QC are shared across
  channel settings; nonfinite/flat channels are excluded before the QC reference.
  The union of state-specific bad-channel masks is used for both states and both
  duration settings within a dataset. QC is repeated within the 128-channel set.
  Each epoch/duration is downsampled and common-average referenced once, then
  reused by wPLI and five phase bands. Full and 120-s data are filtered separately.
  No additional edge trimming or artifact-removal method was introduced.

## Statistical interpretation

There are **two animals and eight animal–condition pairs**, not eight independent
animals. The paired effect sizes, sign-flip tests, bootstrap intervals, and Bayes
factors are exploratory and retain an exchangeability/independence assumption at
the pair level. FDR is applied across 17 metrics/bands separately for each analysis
variant. Condition summaries, leave-one-group effects, and a sign-flip test of the
two animal-mean differences are saved alongside the paired results. The latter has
very coarse resolution (minimum two-sided p=0.5) and does not establish population
significance. Additional animals are needed for such inference.

All metrics require complete, finite, correctly ordered paired observations.
Missing measurements are never filled with zero. Undefined standardized effects
are serialized as JSON `null`; figure generation rejects undefined statistics.
Degenerate bootstrap resamples are warned about and excluded from percentile
estimation rather than assigned artificially large effect sizes.

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

JSON files retain full precision, per-state/per-dataset values, physical channel
IDs, QC reasons, epoch provenance, package/source versions, parameters, and timing.
Epoch selections are embedded in the analysis JSON, without a duplicate manifest
file. CSV files are convenient statistical summaries. Figures read JSON directly, reject mismatched epoch manifests,
compute FDR annotations from the current results, and use the same mappable for
topographic points and their colorbar. Source hashes and color limits are saved in
`figures/provenance.json`.

Saved results retain the source hashes recorded during computation. A layout-only
reorganization is recorded separately in `layout_source_sha256`; numerical results
are retained without being presented as a fresh analysis run.

The runner holds one dataset at a time and one frequency band's phases at a time.
STFT stores only requested frequencies and processes channel pairs in small blocks.
Loading avoids irrelevant sessions and retains only selected samples from each
MAT channel file (MAT v5 files still require decompressing that channel).
For reproducible single-thread timing, set `OPENBLAS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `OMP_NUM_THREADS=1` before launching Python.

Reference dataset: Yanagawa T, Chao ZC, Hasegawa N, Fujii N (2013).
*Large-scale information flow in conscious and unconscious states: an ECoG study
in monkeys*. PLoS ONE 8:e80845. https://doi.org/10.1371/journal.pone.0080845
