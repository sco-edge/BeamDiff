# Sensing-Aware BFA Delta Diffusion v1

## 1. Method overview

Sensing-Aware BFA Delta Diffusion v1 is a BFA-specific, anchor-conditioned
conditional DDPM. It is not a direct input-format conversion of RF-Diffusion
and does not use the earlier best-of-five post-generation selection procedure.

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
from the same anchor and activity condition. In v1 evaluation, however, only one
candidate is generated per anchor.

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

## 9. Difference from best-of-five selection

The earlier best-of-five experiment generated five candidates per anchor and used
a frozen BeamSense teacher plus a temporal-realism score to select one candidate.
V1 does not perform this inference-time candidate search. The BeamSense and
temporal objectives are applied during generator training, after which one
synthetic trajectory is generated per real anchor.

The best-of-five method reached 95.1398% mean downstream accuracy, whereas v1
reached 94.6268%. The difference was 0.5130 percentage points, while v1 required
one candidate rather than five at generation time.

## 10. Scope and current limitations

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
- comparison with the original one-shot delta DDPM, best-of-five selection,
  TimeVAE, and direct augmentation baselines;
- generation-time and memory-cost comparisons.

## 11. Delta Diffusion experiment history

### BFA-specific generator

An anchor-conditioned diffusion model was trained directly on the circular
frame-to-frame deltas of quantized BFA windows. Each input window has shape
`(10,234,4)`. The first real BFA frame is retained as the anchor, and the model
generates the remaining nine frame deltas conditioned on the activity label.
The four BFA channels use their respective circular ranges (512, 512, 128,
128). The generator uses only the fixed real training fold (`split_seed=111`).

The first unfiltered one-candidate-per-anchor pool contained 28,529 synthetic
windows. Its frozen-Teacher label agreement was 78.047%, Macro F1 was 77.204%,
and Macro recall was 77.252%. Adding this entire pool at 1:1 for model seed 111
reduced BeamSense accuracy from 93.7849% to 92.5518% (-1.2331%p). This showed
that the initial one-shot generator was not sufficiently reliable for 1:1
augmentation.

### Best-of-five sensing-aware selection

Five independent diffusion candidates were generated for every real training
anchor (142,645 candidates total). One candidate per anchor was selected using
the assigned-activity confidence of the frozen real-only BeamSense teacher and
a temporal-realism penalty based on the real per-class delta distribution.
Incorrect teacher predictions received an additional penalty. Validation and
test windows were not used as generation anchors or synthetic training data.

- Candidate teacher label agreement: 78.1773%.
- Selected teacher label agreement: 93.0211%.
- Selected mean target confidence: 91.6981%.
- Selected mean temporal Z-score: 0.9083.
- Selected synthetic windows: 28,529 (exactly one per real training anchor).
- Selected class counts exactly match the real training-fold class counts.
- Selection parameters: realism weight 0.15; incorrect-prediction penalty 5.0.

The selected pool was added to the 28,529 real training windows at exactly 1:1.
All downstream runs use the same split (`split_seed=111`), generated pool, and
augmentation ordering (`augmentation_seed=111`). Only BeamSense model seed
changes.

| Model seed | Real-only accuracy | Selected 1:1 accuracy | Paired gain |
|---:|---:|---:|---:|
| 42 | 93.5054% | 96.4979% | +2.9924%p |
| 111 | 93.7849% | 91.4831% | -2.3019%p |
| 2026 | 91.3515% | 95.3305% | +3.9790%p |
| 3407 | 85.1529% | 95.9060% | +10.7530%p |
| 7777 | 94.3275% | 96.4814% | +2.1539%p |
| **Mean ± sample SD** | **91.6245 ± 3.7904%** | **95.1398 ± 2.1000%** | **+3.5153 ± 4.7055%p** |

Mean Macro F1 increased from 91.7092% to 95.3031% (+3.5939%p), and mean Macro
recall increased from 92.0131% to 95.3238% (+3.3106%p). Accuracy improved for
four of five model seeds, and the median paired accuracy gain was +2.9924%p.
Excluding teacher seed 111, all four independent model seeds improved; their
mean accuracy changed from 91.0843% to 96.0539% (+4.9696%p). Teacher seed 111
is reported rather than discarded, but the independent-seed result is also
reported because that teacher checkpoint participated in synthetic selection.

This experiment establishes that sensing-aware candidate selection can turn
the initially harmful 1:1 synthetic pool into a beneficial one on average. It
does not yet establish that five candidates are necessary or that the gain
generalizes beyond this inspected HAR-1 random-window test split. Required next
checks are candidate-count ablation (K=2/3/5), an untouched HAR-3 evaluation,
and comparison against TimeVAE under the same protocol.

### One-shot selection distillation — implementation completed, evaluation pending

To reduce the five-candidate generation cost, the selected one-per-anchor pool
will be used as teacher targets to fine-tune the existing Delta Diffusion model.
Fine-tuning mixes 28,529 real deltas and 28,529 selected teacher deltas, then
generates only one candidate for each anchor. This produces a 100% synthetic
pool directly instead of generating a 500% candidate pool at inference.

- Implementation commit: `6b6ebf7`.
- Initialization: original full Delta Diffusion checkpoint, seed 42.
- Planned fine-tuning: 20 epochs, learning rate `5e-5`.
- Planned output: `bfa_delta_ddpm_har1_distilled_seed42/generated_bfa.npz`.
- Evaluation target: approach the best-of-five mean accuracy while retaining
  exactly 28,529 one-shot synthetic windows.

The initial best-of-five teacher construction cost must remain disclosed. The
one-shot variant is considered successful only after standalone generated-label
diagnostics and paired five-seed 1:1 downstream evaluation are completed.

### One-shot distillation and K=2 candidate ablation

The distilled one-shot pool achieved 75.0745% frozen-teacher label agreement,
below both the original one-shot pool (78.0470%) and the selected K=5 pool
(93.0211%). Nevertheless, its seed-42 1:1 downstream accuracy was 94.2289%, a
+0.7234%p gain over the paired 93.5054% real-only baseline. This is weaker than
the K=5 gain and therefore does not yet replace candidate selection.

Generating two candidates per anchor and selecting one increased frozen-teacher
label agreement from 78.0276% over all candidates to 86.8099% after selection.
The selected K=2 pool retained exactly 28,529 windows and the real training-fold
class distribution. Its seed-42 1:1 downstream accuracy was 94.7221%, a
+1.2167%p paired gain. The remaining model seeds are pending.

### Sensing-aware one-shot BFA diffusion — implementation

The BFA Delta Diffusion trainer now optionally applies training-time quality
objectives instead of relying only on post-generation candidate selection:

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
- The best-of-five selected pool reached 95.1398% mean accuracy, only 0.5130%p
  above the one-shot sensing-aware result, while requiring five candidates per
  anchor at generation time.

The result supports the claim that training-time sensing supervision can provide
useful 1:1 BFA augmentation without best-of-five inference-time candidate selection.
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

## 12. GPU-server data and result paths

The following paths are the GPU-server locations used in the completed experiments.
They are recorded separately from local macOS paths and temporary `/tmp` scripts.

### 12.1 HAR-1 BFI source and BeamSense input

| Item | GPU-server path |
|---|---|
| HAR-1 raw BFI dataset directory | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/` |
| HAR-1 real BFA MAT windows | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/MAT/M1_w10/` |
| HAR-1 real BFA NPZ used by BeamSense and Delta Diffusion | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/har1_m1_bfa_w10.npz` |
| HAR-1 extracted complex V matrices | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/V/M1/` |

The principal real-data NPZ contains BFA windows with shape `(N,10,234,4)` and
is the common input for the fixed BeamSense baseline and all BFA Delta Diffusion
experiments.

### 12.2 BeamSense teacher and evaluation code

| Item | GPU-server path |
|---|---|
| Frozen BeamSense teacher checkpoint (model seed 111) | `/home/leehan/results/beamsense_fixedsplit111_baseline_modelseed111_final/random_window_best.keras` |
| BFA Delta Diffusion trainer | `/home/leehan/RF-Diffusion/scripts/train_bfa_delta_diffusion.py` |
| BeamSense training/evaluation script | `/home/leehan/RF-Diffusion/scripts/train_beamsense_har1.py` |

The teacher checkpoint is used only for the sensing-aware classification objective.
Downstream evaluation must also include independently initialized BeamSense models
and must not report only the teacher seed.

### 12.3 Delta Diffusion generated datasets

| Experiment | GPU-server path |
|---|---|
| Original one-shot Delta Diffusion directory | `/home/leehan/results/bfa_delta_ddpm_har1_full_seed42/` |
| Original one-shot generated BFA | `/home/leehan/results/bfa_delta_ddpm_har1_full_seed42/generated_bfa.npz` |
| Best-of-five selected 1:1 BFA | `/home/leehan/results/bfa_delta_ddpm_har1_full_seed42/generated_bfa_selected_1to1.npz` |
| K=2 candidate experiment directory | `/home/leehan/results/bfa_delta_ddpm_har1_candidates2_seed42/` |
| K=2 selected 1:1 BFA with fixed metadata | `/home/leehan/results/bfa_delta_ddpm_har1_candidates2_seed42/generated_bfa_selected_1to1_fixed.npz` |
| One-shot distillation directory | `/home/leehan/results/bfa_delta_ddpm_har1_distilled_seed42/` |
| Distilled one-shot generated BFA | `/home/leehan/results/bfa_delta_ddpm_har1_distilled_seed42/generated_bfa.npz` |
| Sensing-aware v1 directory | `/home/leehan/results/bfa_sensing_aware_full_seed42/` |
| Sensing-aware v1 generated BFA used in the five-seed 1:1 evaluation | `/home/leehan/results/bfa_sensing_aware_full_seed42/generated_bfa.npz` |

Each experiment directory may additionally contain `checkpoint_latest.pt`,
`protocol.json`, generation chunks, and evaluation subdirectories. The NPZ files
listed above are the synthetic datasets passed to the downstream BeamSense training
script.

### 12.4 HAR-3 paths for the next validation

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

### 12.5 Path-handling notes

- GPU result directories under `/home/leehan/results/` are not stored in Git because
  they contain generated datasets and model checkpoints.
- Git records the code, protocol, paths, and numerical summaries; the large NPZ,
  MAT, and checkpoint files remain on the GPU server.
- Files created under `/tmp` are temporary and should not be treated as the canonical
  implementation. Reusable scripts must be copied into
  `/home/leehan/RF-Diffusion/scripts/` and committed before final evaluation.
- Before a new run, verify every recorded path with `test -e <path>` or `ls -lh
  <path>` because result directories can be renamed during repeated experiments.
