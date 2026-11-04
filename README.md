# TDC-Detect

Code for **Detecting Adversarial Attacks on Video Action Recognition via Temporal-Semantic Consistency**.

TDC-Detect combines temporal responses from a frozen action recognizer with semantic evidence from CLIP. Five global probes and four local windows produce a 21-dimensional representation (TDC-v2), followed by a calibrated logistic readout. The implementation supports CSN-R101, MViT-B, SlowFast-R50 and X3D-M.

This repository includes the method implementation, experiment scripts, fitted readout states, compact result tables and a CPU scoring example. The original Python module name `TSC_Detect` is retained to preserve experiment imports; the project name is **TDC-Detect**.

## Quick start

Use Python 3.12. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m tdc_detect --state states/k400_csn_r101.json --input examples/features/k400_csn_r101.csv --output outputs/example_scores.csv
```

The example scores **precomputed features** and requires no GPU or model download. It writes the detection score, frozen threshold and decision for each row. Output files are not overwritten; choose a new output name when repeating the command.

The eight example CSVs contain 80 real archived test rows. The regression tests compare their predictions with archived scores. All 59,805 archived main-test rows were also checked with this scoring interface: maximum absolute score error was below `2e-13`, with no decision mismatches. See [validation/release_checks.json](validation/release_checks.json).

For your own feature CSV, provide the 21 named columns listed in [tdc_detect/readout.py](tdc_detect/readout.py). Select a state matching the dataset and action backbone. Missing or non-finite features are rejected.

## Method and calibration

1. Construct five ordered global probes: original, early crop, late crop, sparse sampling and motion-aware sampling; add four chronological local windows.
2. Evaluate the frozen action recognizer and CLIP ViT-B/32. CLIP uses the 400 Kinetics labels, three prompts and up to eight frames per probe, averaging logits before softmax.
3. Compute 21 features describing action variation, action–semantic disagreement, local evidence and variation across the ordered global probes.
4. Apply the fitted standardizer and logistic readout. A sample is flagged when `score > threshold`, with the threshold set to the clean-calibration 99th percentile using the `higher` quantile rule.

Calibration and test sources are disjoint. The logistic readout uses balanced class weights, L2 regularization (`C=1`), `lbfgs`, `max_iter=5000` and seed `20260714`. DFT features describe variation across the fixed probe sequence. Transfer experiments retain the source-fitted readout and threshold.

The JSON files in `states/` contain reconstructed calibration states that were checked against the archived results. They contain the standardizer and readout parameters; action-recognition and CLIP weights are obtained separately.

## Repository layout

```text
tdc_detect/               Portable calibrated feature-scoring interface
TSC_Detect/               Original feature, sampling, model and extraction modules
scripts/                  Historical main-evaluation and experiment scripts
states/                   Eight condition-specific 21-feature readout states
assets/                   Kinetics-400 label list
examples/                 Archived feature examples and expected scores
experiments/revision/     Revised CPU/GPU experiments, states and protocols
experiments/hmdb51/       Transfer experiment code, protocol and sample manifest
experiments/historical_*  Additional source snapshot, protocols and legacy tests
environments/             Recorded experiment environments
results/                  Selected final main, runtime and HMDB51 result tables
tests/                    Portable readout regression tests
tools/                    File-integrity verification
provenance/                Original source-to-release mapping and file hashes
validation/               Recorded numerical and packaging checks
```

## Experiments

The original scientific source files are preserved byte-for-byte; duplicate snapshot files are mapped to one copy in [provenance/source_files.json](provenance/source_files.json). The portable scorer is a new interface to the same saved readout.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the final experiment entry points, environment records and path configuration. Full GPU reproduction requires video datasets, pretrained weights and the additional inputs specified there. Several original workers retain server-specific paths. A clean-machine end-to-end GPU run has not been performed for this release.

For the final calibrated method, use the 21-feature states and revised full-detector implementation. The older `TSC_Detect/run_tsc_detect.py` CLI defaults to a fixed score with zero local windows, and the historical E8 attack differs from the revised full-detector G02 evaluation.

## Verification

```bash
python tools/verify_release.py
```

This standard-library check verifies the release file hashes and parses all Python sources. The regression tests above additionally verify numerical predictions. The main eight-condition mean ROC-AUC recorded in the accompanying results is **0.8148**; this is an aggregate benchmark result, not the score of the small quick-start example.

The repository contains code, compact evidence and readout states. Video datasets, backbone weights, large response caches and full per-source experiment outputs are maintained separately.
