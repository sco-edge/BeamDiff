# BeamDiff

**BeamDiff: Sensing-Aware Diffusion for Wi-Fi Beamforming Feedback Augmentation**

BeamDiff generates synthetic Beamforming Feedback Angle (BFA) windows to supplement measured training data for Wi-Fi human activity recognition (HAR). It preserves a measured first-frame anchor and generates the nine subsequent transitions, conditioned on the anchor and activity label. Noise prediction, clean-transition reconstruction, temporal-fidelity objectives, and supervision through a frozen BeamSense classifier support sensing-aware generation.

This repository contains the BeamDiff pipeline: BFI preprocessing, diffusion training and generation, downstream sensing evaluation, fidelity auditing, and result visualization. Exploratory models and competing generators are excluded.

## Overview

```text
BFI captures -> BFA windows (10 x 234 x 4)
                         |
               anchor + activity label + noisy transitions
                         |
                conditional diffusion denoiser
                         |
                  generated transitions
                         |
              anchor + transition accumulation
                         |
                  synthetic BFA windows
                         |
           measured + synthetic training windows
                         |
                 downstream BeamSense HAR
```

During generator training, the frozen classifier supplies an activity loss through reconstructed windows; its weights are not updated.

## Repository structure

| File | Purpose |
| --- | --- |
| `scripts/har1_bfi_to_bfa.py` | Convert HAR-1 PCAPNG captures into BFA windows |
| `scripts/train_bfa_delta_diffusion.py` | Train BeamDiff and generate synthetic windows |
| `scripts/train_beamsense_har1.py` | Train the teacher or downstream classifier |
| `scripts/evaluate_beamsense_npz.py` | Evaluate saved sensing models |
| `scripts/audit_bfa_generation_fidelity.py` | Audit anchors and transition distributions |
| `scripts/summarize_beamdiff_ratio_sweep.py` | Summarize paired seeds and augmentation ratios |
| `scripts/plot_augmentation_ratio_comparison.py` | Plot BeamDiff accuracy across ratios |
| `tfdiff/` | BFI parsing and window construction |
| `tests/` | Preprocessing and augmentation-subset checks |
| `results/beamdiff_ratio_sweep.csv` | Recorded BeamDiff accuracy summary |

## Setup

Use a Python virtual environment on Linux. A CUDA-capable GPU is recommended for training; install a PyTorch build compatible with your CUDA environment. TensorFlow is required for BeamSense and loading the frozen Keras teacher.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The dependency list is not a frozen experimental environment. GPU training has not been rerun during repository packaging.

## Dataset

Obtain CSI-BFI-HAR separately under its own terms:

- [IEEE DataPort dataset](https://doi.org/10.21227/nbp8-7r47)
- [Dataset files and extraction documentation](https://huggingface.co/datasets/foysalhaque/CSI-BFI-HAR-Dataset)

Raw captures, measured/generated datasets, pretrained weights, and paper PDFs are not bundled.

```bash
python scripts/har1_bfi_to_bfa.py data/HAR-1/BFI/M1 data/har1_bfa.npz
```

The NPZ contains `x` shaped `(N,10,234,4)`, zero-based labels `y` in `[0,19]`, and metadata including `source`, `window_start`, and `participant`. Channel order is `phi_11, phi_21, psi_21, psi_31`; phi ranges from 0 to 511 and psi from 0 to 127. Keep input ordering and real-data partitions fixed throughout an experiment.

## Usage

### 1. Real-only sensing teacher

```bash
python scripts/train_beamsense_har1.py data/har1_bfa.npz runs/teacher \
  --split random-window --split-seed 111 --seed 111 --normalize none
```

The checkpoint is `runs/teacher/random_window_best.keras`. Only real training data may be used to fit the teacher or generator.

### 2. BeamDiff training and generation

Set auxiliary-loss weights from your experiment configuration. This command is a template, not an exact-results reproduction recipe:

```bash
# Define X0_WEIGHT, TEMPORAL_WEIGHT, and CLASSIFICATION_WEIGHT first.
python scripts/train_bfa_delta_diffusion.py data/har1_bfa.npz runs/beamdiff \
  --split-seed 111 --seed 42 \
  --teacher-model runs/teacher/random_window_best.keras \
  --teacher-normalization none \
  --x0-weight "${X0_WEIGHT:?set X0_WEIGHT}" \
  --temporal-weight "${TEMPORAL_WEIGHT:?set TEMPORAL_WEIGHT}" \
  --classification-weight "${CLASSIFICATION_WEIGHT:?set CLASSIFICATION_WEIGHT}"
```

The script trains and then generates one synthetic window per real training anchor by default. Outputs include `checkpoint_latest.pt`, `generated_bfa.npz`, and `protocol.json`. Auxiliary-loss weights default to zero; calling the trainer without these settings does not activate the full sensing-aware objective. Set epochs, diffusion steps, schedule, and other hyperparameters explicitly; see `--help` for options.

### 3. Augmented sensing training

```bash
python scripts/train_beamsense_har1.py data/har1_bfa.npz runs/ratio1_seed42 \
  --split random-window --split-seed 111 --seed 42 --normalize none \
  --augment-npz runs/beamdiff/generated_bfa.npz \
  --augment-ratio 1.0 --augment-seed 111
```

Compare with real-only training using the same classifier seed, split, and normalization. Repeat classifier seeds `42 111 2026 3407 7777` and ratios `0 0.25 0.5 0.75 1`. Validation and testing use real data only. Ratio 1.0 adds one synthetic window per real training window.

### 4. Fidelity audit and result summaries

```bash
python scripts/audit_bfa_generation_fidelity.py \
  data/har1_bfa.npz runs/beamdiff/generated_bfa.npz runs/beamdiff/fidelity.json
python scripts/summarize_beamdiff_ratio_sweep.py runs \
  --ratios 0 0.25 0.5 0.75 1 --seeds 42 111 2026 3407 7777
python scripts/plot_augmentation_ratio_comparison.py \
  results/beamdiff_ratio_sweep.csv results/beamdiff_accuracy
```

The summarizer expects `runs/ratio{ratio}_seed{seed}/random_window_metrics.json`, including matching real-only baseline runs at ratio 0.

## Recorded results

HAR-1 fixed random-window split, seed 111: 28,529 training, 6,064 validation, and 6,082 test windows. Accuracy is mean ± sample SD over five classifier initialization seeds, not five independent generator trainings.

| Added synthetic data | Accuracy (%) |
| --- | ---: |
| 0% (real-only) | 92.46 ± 2.63 |
| 25% | 93.15 ± 2.80 |
| 50% | 94.70 ± 1.78 |
| 75% | 95.49 ± 0.62 |
| 100% (1:1) | 96.08 ± 1.18 |

These are previously recorded results, not measurements newly produced by this release. The random-window protocol does not establish generalization to unseen participants, captures, or environments. Original datasets, checkpoints, and complete server-run configurations are not bundled, so this is not a turnkey exact-results reproduction package.

## Implementation notes

The research code is published without changing its numerical behavior. In this snapshot, `--delta-mode circular` applies modular differences to **all four channels**, with moduli `(512,512,128,128)`. Reconstruction wraps phi and clips psi. This differs from the later manuscript description of circular phi differences and ordinary psi differences. The fidelity auditor also uses modular differences for all four channels.

The default schedule is linear with 20 steps. A short linear schedule does not reach an approximately pure-noise terminal state; cosine is also supported. Match training and sampling configurations when comparing runs.

## Attribution and reuse

The dataset and BeamSense sensing architecture are prior work, not contributions introduced by BeamDiff. See [ATTRIBUTION.md](ATTRIBUTION.md). This release does not include a repository-wide license grant; GitHub publication alone is not permission to relicense or redistribute third-party material.
