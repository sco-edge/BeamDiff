# BeamDiff augmentation-ratio sweep

## Protocol

- Dataset: CSI-BFI-HAR, HAR-1 BFI/BFA windows
- Split: fixed random-window split, `split_seed=111`
- Real train / validation / test: 28,529 / 6,064 / 6,082
- Downstream classifier: BeamSense CNN
- Classifier seeds: 42, 111, 2026, 3407, 7777
- Augmentation seed: 111
- Synthetic selection: nested seeded prefix
- Ratios: 0%, 25%, 50%, 75%, and 100% of the real training count
- Reported uncertainty: sample standard deviation across the five paired seeds

The real validation and test folds are unchanged across all runs. Only the
synthetic subset size and classifier initialization seed vary. Each gain is
paired against the real-only result with the same classifier seed.

## Results

| Synthetic ratio | Synthetic windows | Accuracy (mean +/- sample SD) | Mean paired gain | Seeds improved |
|---:|---:|---:|---:|---:|
| 0% | 0 | 92.4564 +/- 2.6306% | baseline | - |
| 25% | 7,132 | 93.1503 +/- 2.7950% | +0.6939%p | 4/5 |
| 50% | 14,264 | 94.7024 +/- 1.7834% | +2.2460%p | 4/5 |
| 75% | 21,397 | 95.4850 +/- 0.6244% | +3.0286%p | 5/5 |
| **100% (1:1)** | **28,529** | **96.0835 +/- 1.1753%** | **+3.6271%p** | **5/5** |

The 100% row is the current canonical BeamDiff 1:1 result. It replaces the
earlier standalone five-seed summary (91.6245% real-only and 94.6268%
augmented), which was produced before the ratio sweep was consolidated under a
single evaluation pipeline.

## Interpretation

Within the evaluated range, mean accuracy rises monotonically as the synthetic
ratio increases. The 75% and 100% settings improve all five paired classifier
seeds. The largest measured mean gain is obtained at 100%, where adding one
synthetic window per real training window increases mean accuracy by 3.6271
percentage points. The experiment supports the use of BeamDiff augmentation up
to a 1:1 mixture; it does not justify extrapolation beyond 100% synthetic data.

## GPU result paths

- Generated BeamDiff BFA:
  `/home/leehan/new_Diffusion/generator/sensing_aware_v1_seed42/generated_bfa.npz`
- Ratio-sweep results:
  `/home/leehan/new_Diffusion/evaluation/ratio_sweep_v1/`
- Individual metrics files:
  `ratio{ratio}_seed{seed}/random_window_metrics.json`

Large generated NPZ files and model checkpoints remain on the GPU server. Git
stores the protocol, numerical summary, and the scripts required to summarize
and plot the experiment.
