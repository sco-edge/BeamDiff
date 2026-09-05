# Sensing-Aware BFA Delta Diffusion v1

## 1. Method overview

Sensing-Aware BFA Delta Diffusion v1 is a BFA-specific, anchor-conditioned
conditional DDPM. It is not a direct input-format conversion of RF-Diffusion.

The method receives a quantized BFA window `X` with shape `(10,234,4)`:

- 10 consecutive BFI feedback frames;
- 234 subcarriers;
- four quantized BFA angle channels;
- channel ranges `(512,512,128,128)`.

Instead of generating all ten BFA frames independently, the method retains the
first real frame as an anchor and generates the nine transitions that follow it.

```text
real first-frame anchor + activity label + Gaussian noise
    -> conditional delta denoiser
    -> nine generated BFA deltas
    -> cumulative BFA reconstruction
    -> synthetic uint16 BFA window (10,234,4)
```

The implementation is in `scripts/train_bfa_delta_diffusion.py`.

## 2. Circular BFA delta representation

For a ten-frame BFA window, the frame-to-frame changes are

```text
Delta X[t] = X[t+1] - X[t],  t = 0,...,8.
```

Because the BFA channels are quantized angular variables, ordinary subtraction
can produce an artificially large change near a quantization boundary. For each
channel with range `M`, v1 calculates the shortest signed change:

```text
Delta X[t] = ((X[t+1] - X[t] + M/2) mod M) - M/2.
```

The channel ranges are `M=(512,512,128,128)`. The resulting tensor has shape
`(9,234,4)`. Its mean and standard deviation are calculated using only the fixed
real training fold, then used to normalize the diffusion target.

## 3. First-frame anchor conditioning

The first real BFA frame is retained as the generation anchor:

```text
A = X[0],  A.shape = (234,4).
```

The two 512-level channels are encoded as sine and cosine values to avoid a
discontinuity at the circular boundary. The two 128-level channels are scaled to
`[-1,1]`. This produces six anchor feature channels:

```text
anchor features = [cos(phi_1), cos(phi_2),
                   sin(phi_1), sin(phi_2), psi_1, psi_2].
```

The anchor features are repeated across the nine transition positions and
concatenated with the four noisy-delta channels. The denoiser therefore receives
ten input channels in total.

## 4. Conditional denoising network

The denoising network is a lightweight residual 2-D CNN:

```text
four noisy-delta channels + six anchor channels
    -> 3x3 convolution (10 -> 64 channels)
    -> six residual CNN blocks
    -> GroupNorm + SiLU
    -> 3x3 convolution (64 -> 4 channels)
    -> predicted Gaussian noise
```

Each residual block contains GroupNorm, SiLU activations, two 3x3 convolutions,
and a residual connection. Two additional conditions are injected into every
block:

- a sinusoidal embedding of the diffusion timestep;
- a learned embedding of the requested activity among 20 classes.

The combined condition is

```text
c = TimeEmbedding(t) + ActivityEmbedding(y).
```

The network predicts the Gaussian noise added to the clean normalized delta.

## 5. Forward diffusion and noise-prediction objective

At training time, a diffusion timestep `t` is sampled and Gaussian noise is added
to the clean normalized delta `x_0`:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon.
```

The conditional denoiser predicts that noise:

```text
epsilon_hat = epsilon_theta(x_t, t, y, A).
```

The basic DDPM objective is

```text
L_diffusion = MSE(epsilon_hat, epsilon).
```

The schedule supports both linear and cosine beta schedules. The default number
of diffusion steps is 20.

## 6. Sensing-aware training objectives

Noise prediction alone did not reliably preserve the activity semantics and
temporal variation needed by BeamSense. V1 therefore optimizes the following
combined objective:

```text
L_total = L_diffusion
        + lambda_x0       * L_clean_delta
        + lambda_temporal * L_temporal_fidelity
        + lambda_cls      * L_BeamSense.
```

### 6.1 Clean-delta reconstruction loss

The predicted clean delta is calculated from the noisy input and predicted noise:

```text
x0_hat = (x_t - sqrt(1-alpha_bar_t)*epsilon_hat)
         / sqrt(alpha_bar_t).
```

A Smooth L1 loss between `x0_hat` and the real clean delta directly constrains
the reconstructed BFA transition:

```text
L_clean_delta = SmoothL1(x0_hat, x_0).
```

### 6.2 Temporal-fidelity loss

The temporal-fidelity objective compares real and predicted absolute deltas using
three channel-wise statistics:

- mean absolute movement;
- standard deviation of movement;
- 95th-percentile movement.

```text
L_temporal_fidelity = L_mean
                    + lambda_std  * L_std
                    + lambda_tail * L_p95.
```

This objective attempts to preserve typical movement, variability, and relatively
large temporal changes, rather than allowing the generator to collapse toward
nearly static BFA sequences.

### 6.3 Frozen BeamSense classification loss

The predicted delta is differentiably accumulated from the real anchor to form a
ten-frame BFA window. This window is passed through a pretrained BeamSense CNN:

```text
X_hat[0]   = A
X_hat[t+1] = X_hat[t] + Delta X_hat[t]
L_BeamSense = CrossEntropy(BeamSense(X_hat), y).
```

The public Keras BeamSense CNN is converted to an equivalent PyTorch model. Its
weights are frozen, and a Keras/PyTorch prediction-parity check is performed before
training. BeamSense parameters are not updated. The classification gradient passes
through the frozen network only to update the diffusion denoiser, encouraging the
generated trajectory to retain the requested activity semantics.

## 7. Reverse diffusion and BFA reconstruction

Generation begins with one Gaussian-noise delta tensor for each real training
anchor. Reverse diffusion is conditioned on the anchor and its activity label:

```text
x_T -> x_(T-1) -> ... -> x_0.
```

The generated delta is de-normalized and rounded. Starting from the real anchor,
the nine generated transitions are accumulated sequentially. Values are wrapped
or clipped to their corresponding quantization ranges and stored as a
`uint16 (10,234,4)` BFA window compatible with BeamSense.

Random Gaussian initialization permits different trajectories to be generated
from the same anchor and activity condition. The completed v1 evaluation generates
one synthetic window per real training anchor.

## 8. Training and evaluation protocol

The completed HAR-1 evaluation uses a fixed random-window split:

- split seed: 111;
- real training windows: 28,529;
- validation windows: 6,064;
- test windows: 6,082;
- synthetic windows: 28,529;
- final training ratio: real 1 : synthetic 1.

Only real training windows are used as diffusion anchors. Validation and test
windows are not used for diffusion training, synthetic generation, or downstream
classifier training. BeamSense evaluation is performed only on the held-out real
test fold.

Across five BeamSense model initialization seeds, real-only accuracy increased
from 91.6245% on average to 94.6268% with v1 augmentation, a mean paired gain of
3.0023 percentage points. Four of five model seeds improved.

## 9. Scope and current limitations

V1 generates a new nine-transition BFA trajectory from a real first-frame anchor.
It must not be described as fully unconditional generation of all ten frames.

Current fidelity measurements also show temporal over-smoothing. Generated mean
movement was approximately 55.2%--68.8% of real movement across the four BFA
channels, although global and class-conditional delta-distribution divergences
were relatively low. In addition, model seed 7777 degraded after augmentation.
The supported claim is therefore an average improvement over five seeds, not a
guarantee of improvement for every classifier initialization.

Further validation should include:

- HAR-3 replication under an independently fixed split;
- source-trace or cross-environment evaluation;
- ablation of clean-delta, temporal, and BeamSense losses;
- comparison with RF-Diffusion, TimeVAE, and direct augmentation baselines;
- generation-time and memory-cost comparisons.

## 10. Sensing-aware v1 experiment results

### Sensing-aware one-shot BFA diffusion — implementation

The BFA Delta Diffusion trainer applies the following training-time quality
objectives:

- clean normalized-delta reconstruction loss on the predicted diffusion `x0`;
- per-angle temporal-magnitude matching loss;
- activity cross-entropy from a frozen BeamSense teacher;
- differentiable reconstruction from generated deltas to BFA frames, including
  circular handling of the two 512-level angle channels.

The Keras BeamSense checkpoint is converted to an equivalent frozen PyTorch
network so its classification gradient can reach the diffusion model. The
trainer checks Keras/PyTorch prediction parity before training. These losses are
opt-in, preserving the behavior of all earlier commands. Evaluation must use a
non-teacher BeamSense seed and the fixed `split_seed=111` test fold to distinguish
generator improvement from teacher-specific optimization.

### Sensing-aware one-shot BFA diffusion — completed five-seed evaluation

The sensing-aware generator produced exactly 28,529 synthetic BFA windows, allowing
the real training fold and synthetic pool to be combined at a 1:1 ratio. All runs
used the same random-window split (`split_seed=111`), augmentation ordering
(`augmentation_seed=111`), and held-out real validation/test folds. Only the
BeamSense model initialization seed was changed.

| Model seed | Real-only accuracy | Sensing-aware 1:1 accuracy | Paired gain |
|---:|---:|---:|---:|
| 42 | 93.5054% | 96.1361% | +2.6307%p |
| 111 | 93.7849% | 94.8866% | +1.1016%p |
| 2026 | 91.3515% | 94.3440% | +2.9924%p |
| 3407 | 85.1529% | 96.3170% | +11.1641%p |
| 7777 | 94.3275% | 91.4502% | -2.8773%p |
| **Mean** | **91.6245%** | **94.6268%** | **+3.0023%p** |

- Mean Macro F1: 91.7092% -> 94.8584% (+3.1492%p).
- Mean Macro recall: 92.0131% -> 94.8427% (+2.8296%p).
- Accuracy improved for four of five model seeds.
- Excluding teacher seed 111, mean accuracy changed from 91.0843% to 94.5618%
  (+3.4775%p), with three of four independent model seeds improving.
The result supports the claim that training-time sensing supervision can provide
useful 1:1 BFA augmentation with one synthetic window per real training anchor.
Because model seed 7777 degraded, the current claim is an average improvement with
four-of-five seed consistency, not universal improvement for every initialization.

### Sensing-aware generated-data fidelity

The generated temporal-delta distributions remained close in distributional shape
(mean global JS divergence 0.0353; mean class-conditional JS divergence 0.0451),
and the mean temporal Z-score was 0.9231. However, generated mean temporal movement
was only 55.2--68.8% of the corresponding real-data movement across the four BFA
channels, and generated 95th-percentile deltas were also smaller. The generator is
therefore still temporally over-smoothed despite improving downstream sensing on
average. This limitation and the seed-7777 regression must be retained in the final
analysis rather than reporting accuracy alone.

## 11. GPU-server data and result paths

The following paths are the GPU-server locations used in the completed experiments.
They are recorded separately from local macOS paths and temporary `/tmp` scripts.

### 11.1 HAR-1 BFI source and BeamSense input

| Item | GPU-server path |
|---|---|
| HAR-1 raw BFI dataset directory | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/` |
| HAR-1 real BFA MAT windows | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/MAT/M1_w10/` |
| HAR-1 real BFA NPZ used by BeamSense and Delta Diffusion | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/har1_m1_bfa_w10.npz` |
| HAR-1 extracted complex V matrices | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/V/M1/` |

The principal real-data NPZ contains BFA windows with shape `(N,10,234,4)` and
is the common input for the fixed BeamSense baseline and all BFA Delta Diffusion
experiments.

### 11.2 BeamSense teacher and evaluation code

| Item | GPU-server path |
|---|---|
| Frozen BeamSense teacher checkpoint (model seed 111) | `/home/leehan/results/beamsense_fixedsplit111_baseline_modelseed111_final/random_window_best.keras` |
| BFA Delta Diffusion trainer | `/home/leehan/RF-Diffusion/scripts/train_bfa_delta_diffusion.py` |
| BeamSense training/evaluation script | `/home/leehan/RF-Diffusion/scripts/train_beamsense_har1.py` |

The teacher checkpoint is used only for the sensing-aware classification objective.
Downstream evaluation must also include independently initialized BeamSense models
and must not report only the teacher seed.

### 11.3 Sensing-aware v1 generated dataset

| Item | GPU-server path |
|---|---|
| Sensing-aware v1 result directory | `/home/leehan/results/bfa_sensing_aware_full_seed42/` |
| Sensing-aware v1 generated BFA used in the five-seed 1:1 evaluation | `/home/leehan/results/bfa_sensing_aware_full_seed42/generated_bfa.npz` |
| Sensing-aware v1 checkpoint | `/home/leehan/results/bfa_sensing_aware_full_seed42/checkpoint_latest.pt` |
| Sensing-aware v1 protocol | `/home/leehan/results/bfa_sensing_aware_full_seed42/protocol.json` |

The generated NPZ is the only Delta Diffusion synthetic dataset retained for the
completed v1 evaluation.

### 11.4 HAR-3 paths for the next validation

| Item | GPU-server path |
|---|---|
| HAR-3 BFI dataset directory | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/` |
| HAR-3 real BFA MAT windows | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/MAT/M1_w10/` |
| HAR-3 real BFA NPZ | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/har3_m1_bfa_w10.npz` |
| HAR-3 fixed random-window split | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/splits/random_window_seed111/` |
| HAR-3 train indices | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/splits/random_window_seed111/train_indices.npy` |
| HAR-3 validation indices | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/splits/random_window_seed111/validation_indices.npy` |
| HAR-3 test indices | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/splits/random_window_seed111/test_indices.npy` |
| HAR-3 split protocol | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/splits/random_window_seed111/protocol.json` |

HAR-3 Delta Diffusion training must use only the 29,163 samples referenced by
`train_indices.npy`. Its 6,249 validation and 6,252 test samples must remain excluded
from generator training, anchor selection, and synthetic generation.

### 11.5 Path-handling notes

- GPU result directories under `/home/leehan/results/` are not stored in Git because
  they contain generated datasets and model checkpoints.
- Git records the code, protocol, paths, and numerical summaries; the large NPZ,
  MAT, and checkpoint files remain on the GPU server.
- Files created under `/tmp` are temporary and should not be treated as the canonical
  implementation. Reusable scripts must be copied into
  `/home/leehan/RF-Diffusion/scripts/` and committed before final evaluation.
- Before a new run, verify every recorded path with `test -e <path>` or `ls -lh
  <path>` because result directories can be renamed during repeated experiments.
