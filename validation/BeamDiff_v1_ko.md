# BeamDiff v1 한글 기술 및 실험 기록

## 1. 명칭

최종 제안 모델의 이름은 **BeamDiff**로 통일한다. BeamDiff는 BFA 변화량을
생성하는 기본 Delta Diffusion 구조에 센싱 목적의 학습 손실을 결합한 모델이다.

- **BeamDiff**: 최종 제안 모델
- **Base Delta Diffusion**: sensing-aware 손실이 없는 초기 모델이며 ablation에서만 사용
- **sensing-aware**: 별도 모델명이 아니라 BeamDiff의 학습 특성을 설명하는 표현

GPU의 기존 `sensing_aware_v1_seed42` 디렉터리는 결과 파일 경로의 호환성을
유지하기 위해 이름을 변경하지 않는다.

## 2. 모델 개요

BeamDiff v1은 BFA에 맞게 설계된 anchor-conditioned conditional DDPM이다.
RF-Diffusion에 BFA 입력 형식만 적용한 모델이 아니라, BFA의 양자화된 각도 구조와
시간 변화 특성을 직접 반영한다.

입력 BFA window의 형태는 `(10,234,4)`이다.

- 연속된 BFI feedback frame 10개
- subcarrier 234개
- 양자화된 BFA angle channel 4개
- 각 channel의 범위: `(512,512,128,128)`

BeamDiff는 10개 frame 전체를 독립적으로 생성하지 않는다. 원본 window의 첫
frame은 anchor로 유지하고, 이후 9개 frame을 구성하는 frame 간 변화량을 생성한다.

```text
원본 첫 frame anchor + activity label + Gaussian noise
    -> conditional delta denoiser
    -> 9개의 BFA 변화량 생성
    -> anchor부터 변화량을 누적하여 BFA 복원
    -> uint16 BFA 합성 window (10,234,4)
```

구현 파일은 `scripts/train_bfa_delta_diffusion.py`이다.

## 3. Circular BFA delta 표현

10-frame BFA window `X`의 시간 변화량은 다음과 같다.

```text
Delta X[t] = X[t+1] - X[t],  t = 0,...,8
```

BFA는 양자화된 각도이므로 경계에서 일반적인 뺄셈을 사용하면 실제로는 작은
변화가 매우 큰 변화로 계산될 수 있다. 따라서 channel 범위가 `M`일 때 가장 짧은
방향의 signed circular delta를 사용한다.

```text
Delta X[t] = ((X[t+1] - X[t] + M/2) mod M) - M/2
```

네 channel의 `M`은 `(512,512,128,128)`이며, 생성 대상 delta tensor의 형태는
`(9,234,4)`이다. 정규화에 사용하는 평균과 표준편차는 고정된 실제 train fold에서만
계산한다.

## 4. 첫 frame anchor 조건

원본 BFA의 첫 frame을 생성 기준점으로 사용한다.

```text
A = X[0],  A.shape = (234,4)
```

512-level channel 두 개는 circular boundary의 불연속성을 방지하기 위해 sine과
cosine으로 표현한다. 128-level channel 두 개는 `[-1,1]` 범위로 변환한다.

```text
anchor feature = [cos(phi_1), cos(phi_2),
                  sin(phi_1), sin(phi_2), psi_1, psi_2]
```

6개의 anchor feature channel과 4개의 noisy-delta channel을 결합하므로 denoiser의
입력 channel은 총 10개이다.

## 5. 조건부 denoising network

denoiser는 경량 residual 2-D CNN으로 구성된다.

```text
4 noisy-delta channels + 6 anchor channels
    -> 3x3 convolution (10 -> 64 channels)
    -> residual CNN block 6개
    -> GroupNorm + SiLU
    -> 3x3 convolution (64 -> 4 channels)
    -> Gaussian noise 예측
```

모든 residual block에는 diffusion timestep의 sinusoidal embedding과 20개 activity
class의 학습 가능한 embedding이 함께 주입된다.

```text
c = TimeEmbedding(t) + ActivityEmbedding(y)
```

## 6. 학습 목적함수

기본 noise-prediction loss만으로는 BeamSense 분류에 필요한 activity 의미와 시간
변화를 안정적으로 보존하지 못했다. BeamDiff는 다음 결합 목적함수를 사용한다.

```text
L_total = L_diffusion
        + lambda_x0       * L_clean_delta
        + lambda_temporal * L_temporal_fidelity
        + lambda_cls      * L_BeamSense
```

### 6.1 Diffusion loss

실제 normalized delta `x_0`에 timestep `t`에 해당하는 Gaussian noise를 추가하고,
denoiser가 추가된 noise를 예측하도록 학습한다.

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1-alpha_bar_t) * epsilon

L_diffusion = MSE(epsilon_hat, epsilon)
```

기본 diffusion step은 20이다.

### 6.2 Clean-delta reconstruction loss

예측 noise로부터 복원한 clean delta와 실제 delta 사이의 Smooth L1 loss를 계산한다.
이를 통해 최종 BFA 변화량 자체를 직접 제한한다.

```text
x0_hat = (x_t - sqrt(1-alpha_bar_t)*epsilon_hat)
         / sqrt(alpha_bar_t)

L_clean_delta = SmoothL1(x0_hat, x_0)
```

### 6.3 Temporal-fidelity loss

실제 delta와 예측 delta의 channel별 절대 변화량 평균, 표준편차 및 95 percentile을
비교한다. 이는 합성 BFA가 거의 정적인 sequence로 수렴하는 현상을 줄이기 위한 것이다.

```text
L_temporal_fidelity = L_mean
                    + lambda_std  * L_std
                    + lambda_tail * L_p95
```

### 6.4 Frozen BeamSense classification loss

예측 delta를 실제 anchor부터 미분 가능한 방식으로 누적하여 10-frame BFA를 만든 뒤,
미리 학습된 BeamSense CNN에 입력한다.

```text
X_hat[0]   = A
X_hat[t+1] = X_hat[t] + Delta X_hat[t]
L_BeamSense = CrossEntropy(BeamSense(X_hat), y)
```

Keras BeamSense checkpoint는 동일한 PyTorch network로 변환하며, 학습 전에 두
모델의 prediction parity를 검사한다. BeamSense parameter는 고정하고 diffusion
denoiser만 갱신한다. 따라서 분류 gradient가 생성 신호가 지정한 activity 특징을
유지하도록 BeamDiff를 지도한다.

## 7. 생성 과정

각 실제 train window의 첫 frame을 anchor로 사용하고 Gaussian-noise delta에서
reverse diffusion을 시작한다.

```text
x_T -> x_(T-1) -> ... -> x_0
```

생성된 delta를 역정규화한 후 anchor부터 순서대로 누적한다. 각 값은 해당 channel의
양자화 범위에 맞게 circular wrapping 또는 clipping하고, BeamSense와 호환되는
`uint16 (10,234,4)` BFA window로 저장한다.

완성된 v1 실험에서는 실제 train anchor 하나당 합성 window 하나를 생성하여
real과 synthetic을 1:1로 결합했다. 합성 window의 첫 frame은 실제 anchor이지만,
이후 9개 변화량은 새로 생성된다. 원본과 완전히 동일한 합성 window의 비율은 0%다.

## 8. HAR-1 실험 결과

### 8.1 프로토콜

- split: fixed random-window
- split seed: 111
- 실제 train: 28,529개
- validation: 6,064개
- test: 6,082개
- synthetic: 28,529개
- 최종 train 구성: real 1 : synthetic 1

validation과 test window는 BeamDiff 학습, anchor 선택, 합성 생성 및 downstream
classifier 학습에 사용하지 않았다. 최종 평가는 고정된 실제 test fold에서 수행했다.

### 8.2 5-seed paired comparison

| Model seed | Real-only accuracy | BeamDiff 1:1 accuracy | Paired gain |
|---:|---:|---:|---:|
| 42 | 93.5054% | 96.1361% | +2.6307%p |
| 111 | 93.7849% | 94.8866% | +1.1016%p |
| 2026 | 91.3515% | 94.3440% | +2.9924%p |
| 3407 | 85.1529% | 96.3170% | +11.1641%p |
| 7777 | 94.3275% | 91.4502% | -2.8773%p |
| **평균** | **91.6245%** | **94.6268%** | **+3.0023%p** |

- 평균 Macro-F1: 91.7092% -> 94.8584% (`+3.1492%p`)
- 평균 Macro-recall: 92.0131% -> 94.8427% (`+2.8296%p`)
- 5개 seed 중 4개에서 accuracy 향상
- teacher seed 111을 제외한 독립 seed 평균: 91.0843% -> 94.5618%
  (`+3.4775%p`)

따라서 HAR-1에서는 모든 초기화에서 반드시 향상된다고 주장하는 대신, 5-seed 평균
성능이 향상됐으며 5개 중 4개 seed에서 개선됐다고 기술한다.

## 9. HAR-3 실험 결과

### 9.1 Fixed random-window 결과

HAR-3에서도 동일 split과 model seed 42를 사용해 real-only와 1:1 증강을 paired
comparison했다.

| 지표 | Real-only | BeamDiff 1:1 | 변화 |
|---|---:|---:|---:|
| Accuracy | 98.7044% | 99.6481% | **+0.9437%p** |
| Macro-F1 | 98.6310% | 99.6851% | **+1.0541%p** |
| Macro-recall | 98.6942% | 99.6211% | **+0.9269%p** |
| 오분류 수 | 81/6,252 | 22/6,252 | **59개 감소** |

오류율은 1.2956%에서 0.3519%로 감소했으며, 상대 오류 감소율은 약 72.8%다.
현재 HAR-3 random-window 결과는 model seed 42 한 번의 결과이므로 추가 seed 검증이
필요하다. 합성 BFA를 frozen BeamSense teacher로 평가한 label accuracy는
96.6636%였고 전체 20개 class가 유지되어 class collapse는 관찰되지 않았다.

### 9.2 Cross-participant/source-trace baseline

P3는 전체 435개 중 S가 393개이고 나머지 활동은 각각 1~7개뿐인 불완전 수집
데이터이므로 평가에서 제외했다. P1과 P2를 교차하여 측정한 baseline은 다음과 같다.

| 방향 | Accuracy | Macro-F1 | Macro-recall |
|---|---:|---:|---:|
| P1 -> P2 | 8.9350% | 4.8421% | 7.9014% |
| P2 -> P1 | 6.8036% | 2.6723% | 5.2642% |
| **평균** | **7.8693%** | **3.7572%** | **6.5828%** |

양방향 성능이 20-class 무작위 추측 정확도 5%에 가까워 BFI의 participant domain
shift가 매우 큰 것으로 나타났다. 이 결과는 low-data in-domain 결과가 아니라
cross-participant 결과로 구분해야 한다. Cross-participant 조건에서 BeamDiff의
증강 이득은 아직 측정하지 않았다.

## 10. 생성 데이터 fidelity 및 한계

HAR-1 합성 데이터의 평균 global delta JS divergence는 0.0353, 평균 class-conditional
delta JS divergence는 0.0451, 평균 temporal Z-score는 0.9231이었다. 그러나 생성된
평균 시간 변화량은 실제 신호의 약 55.2~68.8% 수준으로 temporal over-smoothing이
남아 있다.

HAR-3 합성 데이터도 원본과 완전히 같은 window는 없었고 전체 class를 유지했지만,
평균 시간 변화량이 실제 신호의 약 54.4~60.1% 수준이었다. 평균 temporal Z-score는
2.3065로 HAR-1보다 시간 통계 차이가 컸다.

현재 주장 가능한 범위는 다음과 같다.

- BeamDiff는 HAR-1 5-seed 평균 random-window 성능을 향상시켰다.
- BeamDiff는 HAR-3 seed 42 random-window 성능을 향상시켰다.
- 첫 실제 frame을 anchor로 사용하므로 완전한 unconditional 10-frame 생성은 아니다.
- random-window 결과만으로 새로운 participant에 대한 일반화를 주장할 수 없다.
- low-data, 추가 HAR-3 seed, ablation 및 cross-participant 증강 검증이 필요하다.

## 11. GPU 데이터 및 결과 경로

### 11.1 HAR-1

| 항목 | GPU 경로 |
|---|---|
| 실제 BFA NPZ | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-1/BFI/har1_m1_bfa_w10.npz` |
| BeamDiff 생성 결과 | `/home/leehan/new_Diffusion/generator/sensing_aware_v1_seed42/generated_bfa.npz` |
| BeamDiff checkpoint | `/home/leehan/new_Diffusion/generator/sensing_aware_v1_seed42/checkpoint_latest.pt` |
| BeamDiff protocol | `/home/leehan/new_Diffusion/generator/sensing_aware_v1_seed42/protocol.json` |
| frozen BeamSense teacher | `/home/leehan/new_Diffusion/teacher/beamsense_model_seed111.keras` |
| 5-seed 평가 디렉터리 | `/home/leehan/new_Diffusion/evaluation/` |

### 11.2 HAR-3

| 항목 | GPU 경로 |
|---|---|
| 실제 BFA NPZ | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/har3_m1_bfa_w10.npz` |
| fixed random-window split | `/home/leehan/RF-Diffusion/dataset/hug_CLI/HAR-3/BFI/splits/random_window_seed111/` |
| BeamDiff 생성 결과 | `/home/leehan/new_Diffusion/HAR-3/generator/sensing_aware_v1_seed42/generated_bfa.npz` |
| P1 train/P2 test split | `/home/leehan/new_Diffusion/HAR-3/splits/train_p1_test_p2/` |
| P2 train/P1 test split | `/home/leehan/new_Diffusion/HAR-3/splits/train_p2_test_p1/` |

### 11.3 구현 코드

| 항목 | 경로 |
|---|---|
| BeamDiff 학습 및 생성 | `scripts/train_bfa_delta_diffusion.py` |
| BeamSense 학습 및 평가 | `scripts/train_beamsense_har1.py` |
| 생성 데이터 fidelity 평가 | `scripts/audit_bfa_generation_fidelity.py` |
| source-disjoint split 생성 | `scripts/create_bfa_source_split.py` |
| cross-participant split 생성 | `scripts/create_bfa_cross_participant_split.py` |

대용량 NPZ, MAT 및 checkpoint 파일은 Git에 포함하지 않고 GPU 서버에 보관한다.
Git에는 재현 코드, 실험 프로토콜, 경로와 수치 결과를 기록한다.
