# 모델 실험 정리

2026-10-06 기준. KoMaP 상 분할(Al / Al₃Ni / Eutectic Si / Primary Si)에서 시험한 모델 구조, 사전학습,
입력 해상도 전략, loss, 학습 설정을 정리합니다. 데이터 증강은 `AUGMENTATION_EXPERIMENTS.md`,
추론·앙상블은 `ENSEMBLE_EXPERIMENTS.md`에 따로 있습니다.

모든 수치는 validation 20장의 대회 지표(이미지별 4-class mIoU의 평균)입니다. 특별히 적지 않으면
학습 중 검증 점수(tile 512, TTA 없음)의 best checkpoint 기준입니다.

## 결론

- **모델:** NASA 현미경 사전학습(MicroNet) SE-ResNeXt-50 + U-Net++가 3-seed 앙상블 80.11~80.15%로,
  EfficientNet-B3 200 에폭 6개 앙상블(80.14%)과 동등한 최상위 모델입니다.
  ViT 계열(SegFormer, DINOv2, DINOv3)은 모두 뒤처졌습니다.
- **입력:** 이미지를 축소하지 않고 원본 해상도 crop으로 학습하는 것이 결정적이었습니다(축소하면 상한 66.9%).
- **구조 개선:** 디코더에 원본 해상도 branch를 더한 것(`--fullres_skip`)만 확실한 이득이었습니다(+0.70pp).
- **loss:** 경계를 겨냥한 loss 8종은 모두 효과가 없거나 해로웠습니다. 기본 CE + Dice가 최선입니다.
- **학습 길이:** 60 에폭은 부족했습니다. 120~200 에폭이 실제 이득입니다.
- **현재 최고:** EfficientNet-B3 6개 + SE-ResNeXt-50 6개(60/200 에폭) 혼합 앙상블, val **80.41%**.

## 1. 학습 설정 (공통)

| 항목 | 값 |
|---|---|
| 입력 | 원본 해상도 512×512 crop, 스캔당 8개 → 에폭당 560개, 배치 8 → 70 step |
| optimizer | AdamW, lr 3e-4, weight decay 1e-4, cosine 스케줄 |
| 인코더 lr | 기본은 lr과 같음(3e-4). 초기 run(s2, s4)만 1e-4 |
| loss | 0.5 CE + 0.5 Dice(클래스 평균) |
| 기타 | AMP, 2 에폭마다 val 전체 이미지 검증, 가장 높은 checkpoint 저장 |
| 추론 | 겹치는 타일(50%), 타일별 확률을 cosine 가중치로 합친 뒤 argmax |

인코더 lr 1e-4(77.00%)와 3e-4(77.02 / 77.15%)는 차이가 없었습니다.

## 2. 입력 해상도 전략

| 방식 | 결과 |
|---|---|
| 이미지 전체를 512로 축소 (초기 `train.py`) | 정답 마스크를 축소·복원하는 것만으로 상한 **66.9%**, Eutectic Si 상한 44.2% |
| 원본 해상도 512 crop 학습 + 겹치는 타일 추론 | 77.00% (B3 U-Net++) |

Eutectic Si는 두께가 1.7~3.5px라서 축소하면 사라집니다. 이 전환이 참고 기준(약 75%)을 넘은 핵심이었습니다.
이후 실험에서도 해상도를 높이는 쪽(원본 해상도 branch, 입력 2배 확대)만 꾸준히 효과가 있었습니다.

## 3. 모델 구조와 사전학습

| 모델 | 사전학습 | val mIoU | 비고 |
|---|---|---|---|
| SegFormer-B2 | ImageNet (ADE20K) | 72.97% | 1/4 해상도 출력 |
| **EfficientNet-B3 + U-Net++** | **MicroNet** | **77.00%** | Al 93.3, Al₃Ni 71.9, Eutectic Si 55.8, Primary Si 88.8 |
| EfficientNet-B4 + U-Net++ | MicroNet | 76.91% | B3보다 나아지지 않음 |
| DINOv2 ViT-B/14 + conv head | DINOv2 | 56.06% | |
| DINOv3 ViT-B/16 + conv head | DINOv3 | 63.29% | |
| DINOv3 ViT-L/16 + conv head | DINOv3 | 64.00% | Eutectic Si 34.1 |

- **MicroNet 사전학습이 가장 크게 기여했습니다.** SegFormer-B2보다 +4.0pp입니다. 이 데이터에서는 모델이 얼마나
  최신인지보다 사전학습 도메인이 얼마나 가까운지가 중요했습니다. 가중치 URL은 `microscopynet`으로 끝납니다.
- **ViT 계열은 feature 해상도가 병목입니다.** patch 16px의 거친 특징을 업샘플하는 구조라 3px 상에 불리합니다.
  DINOv3를 B에서 L로 4배 키워도 +0.7pp뿐이었습니다.
- DINO 계열 구현은 ViT + 가벼운 conv head이고, EoMT 본체가 아닙니다.
- B3 → B4로 인코더를 키운 효과는 없었습니다.
- 초기 비교용 run(`unet_r34`, `unet_r34_native`, `unetpp_r50_native`)은 점수 기록이 남아 있지 않습니다.

### 3.1 후속 인코더 비교 (2026-10-05~06)

모델만 비교하기 위해 U-Net++, 원본 해상도 branch, `cs2_aug3`, 입력 2배, 3 seed를 고정했습니다.
HRNet-W18은 전용 segmentation head와 ImageNet 사전학습을 사용했고, 나머지는 MicroNet 사전학습입니다.

| 모델 | 에폭 | 학습 중 best | 최종 추론 3-seed 앙상블 | 기준 B3 앙상블 대비 |
|---|---:|---:|---:|---:|
| EfficientNet-B3 + U-Net++ | 60 | 78.81% | 79.67% | – |
| HRNet-W18 | 60 | 78.43% | 79.19% | -0.49pp |
| **SE-ResNeXt-50 32×4d + U-Net++** | 60 | **78.93%** | **80.11%** | **+0.43pp** |
| SE-ResNeXt-101 32×4d + U-Net++ | 60 | 78.57% | 79.67% | -0.01pp |
| Inception-ResNet-v2 + U-Net++ | 60 | 78.30% | 79.10% | -0.57pp |
| DPN-92 + U-Net++ | 60 | 76.22% | 77.15% | -2.53pp |
| **SE-ResNeXt-50 32×4d + U-Net++** | 200 | 78.85% | **80.15%** | B3 200 에폭 ×3보다 +0.30pp |

- SE-ResNeXt-50 60 에폭은 최종 추론 단일 모델 평균에서 B3보다 +0.34pp(Welch p=0.008),
  checkpoint 선택 효과를 줄인 held-out 비교에서도 +0.24pp였습니다.
- SE-ResNeXt-50의 60 → 200 에폭 연장은 단일 모델 평균을 올리지 못했지만, 두 그룹의 오류 차이 덕분에
  기존 B3 6개와 모두 혼합했을 때 **80.41%**로 올랐습니다.
- SE-ResNeXt-101은 독립 성능은 50보다 낮지만, B3 6개 + SE-ResNeXt-50 3개와 혼합하면 80.35%였습니다.

## 4. 원본 해상도 branch — `--fullres_skip`

### 문제

smp의 U-Net++ 디코더는 인코더의 stride-1 특징(이미지 자체)을 버립니다. 마지막 블록은 stride-2 특징을 nearest로
2배 키운 뒤 16채널 conv 두 개를 거치는 것이 전부입니다. 즉 3px짜리 상의 경계를 원본 해상도를 보지 않고 추정합니다.

### 방법 (`src/models.py`의 `FullResSkip`)

입력 이미지에 3×3 conv 두 개(32채널)를 걸어 원본 해상도 특징을 만들고, 디코더 출력과 이어 붙인 뒤
3×3 conv 하나로 합쳐 segmentation head에 넣습니다. 파라미터는 13.63M에서 13.64M으로 거의 늘지 않습니다.
근거는 얇은 혈관 분할에서 원본 해상도를 유지한 FR-UNet(JBHI 2022)입니다.

### 결과 (3 seed, 같은 조건의 baseline 77.02 ± 0.13 대비)

| 지표 | 결과 |
|---|---|
| 단일 모델 | 77.72 ± 0.12, +0.70 (p<0.01) |
| 마지막 5회 검증 평균 | +1.00 (p=0.03) |
| 이미지별 | 20장 중 17장 개선 |
| 675×480 스캔 / 나머지 | +1.12 (Al₃Ni +2.05) / +0.35 |
| 3-seed 앙상블 (최종 추론 설정) | 78.04 → 78.35% |

줄어든 것은 1px 이내의 경계 위치 오류였고(전부 고칠 때 상한 +9.78 → +9.37pp), 상 오분류는 그대로였습니다.

### 변형 (3 seed, fullres 77.72 대비)

| 변형 | val mIoU | 차이 | 판정 |
|---|---|---|---|
| branch 채널 64 (`--fullres_ch 64`) | 77.52 | −0.19 (p=0.08) | 이득 없음 |
| fullres + Lovász loss | 77.43 | −0.29 (p=0.03) | 조금 나빠짐 |
| fullres B4 (인코더 lr 1e-4) | 77.28 | −0.44 (p=0.26) | seed 편차 큼 (76.71 / 77.50 / 77.63) |

채널을 늘려도 소용없었습니다. 이득은 원본 해상도를 본다는 것 자체에서 오고, branch 용량은 병목이 아닙니다.

## 5. Loss 실험

### 5.1 경계 loss (fullres 없이, 3 seed, baseline 77.02 ± 0.13 대비)

| loss | 플래그 | val mIoU | 차이 | 판정 |
|---|---|---|---|---|
| Jaccard Metric Loss | `--loss ce_jml` | 77.30 | +0.28 (p=0.29) | 효과 없음 |
| Active Boundary Loss | `--loss ce_lovasz --aux_loss abl` | 77.13 | +0.11 (p=0.45) | 효과 없음 |
| Lovász (ABL 대조군) | `--loss ce_lovasz` | 77.10 | +0.08 (p=0.47) | 효과 없음 |
| Al₃Ni–Si 계면 가중 CE | `--ce_mode interface` | 77.07 | +0.05 (p=0.82) | 효과 없음 |
| Boundary DoU | `--loss ce_bdou` | 76.92 | −0.10 (p=0.60) | 효과 없음 |
| Kervadec boundary loss | `--aux_loss kervadec` | 76.25 | −0.77 (p=0.08) | 해로운 경향 |
| JML + 경계 label smoothing | `--loss ce_jml --bls_eps 0.5` | 76.02 | −0.99 (p<0.01) | 해로움 |
| 경계 label relaxation | `--ce_mode relax` | 75.05 | −1.96 (p<0.01) | 해로움 |

- B-DoU, ABL, Kervadec 거리맵은 저자 코드와 수치가 일치하는 것을 확인한 구현입니다(`src/losses.py`).
- 이전에 따로 돌린 CE + Dice + Lovász도 3 seed에서 +0.11pp(p=0.23)로 효과가 없었습니다.

### 5.2 왜 안 됐나

- **경계를 느슨하게 하는 loss가 해로운 이유:** 3px짜리 상에서는 경계 픽셀이 곧 상 전체입니다. 경계 라벨을 풀면
  얇은 상의 감독이 통째로 사라집니다. relaxation 모델은 Al₃Ni 예측 면적이 GT 대비 0.96에서 0.91로 줄고 Al이 늘었습니다.
- **이 결과의 의미:** 남은 1px 경계 오류는 주석 노이즈보다 모델 해상도의 한계에 가깝습니다. 실제로 해상도를 높인
  fullres와 입력 2배 확대는 효과가 있었습니다.
- 클래스별 판정 기준(logit bias)을 조정하는 후처리도 leave-one-out에서 −0.04pp로 효과가 없었습니다.
  예측 면적이 GT의 0.97~1.05배라, 특정 상을 체계적으로 적게·많이 잡는 편향이 없습니다.

## 6. 학습 길이

aug_color 설정(배율 1, 3 seed씩 두 번 = 6 run)으로 잰 결과입니다.

| 에폭 | 단일 모델 | 60 에폭 대비 | 고른 효과를 뺀 차이 | 3-seed 앙상블 |
|---|---|---|---|---|
| 60 | 78.15 | – | – | 78.87% |
| 120 | 78.46 | +0.31 (p<0.01) | +0.58 | 79.34% |
| 200 | 78.61 | +0.47 (p<0.001) | +0.57 | 79.49% |

- "고른 효과를 뺀 차이"는 val을 반으로 나눠 한쪽으로 checkpoint를 고르고 다른 쪽으로 채점한 값입니다(`split_half_check.py`).
  이 값에서도 이득이 남아, 에폭 연장의 효과는 진짜입니다.
- 120과 200 에폭의 차이(+0.16, p=0.12)는 확정되지 않았습니다.
- 첫 200 에폭 실험에서는 후반에 점수가 내려갔지만 두 번째에서는 그렇지 않았습니다. best checkpoint를 쓰므로 큰 문제는 아닙니다.

## 7. 최종 모델

| 항목 | 내용 |
|---|---|
| 구조 | MicroNet EfficientNet-B3 / SE-ResNeXt-50 + U-Net++ + 원본 해상도 branch |
| 입력 | 2배 확대(`--input_scale 2`), crop 1024 |
| 증강 | 밝기 약하게, Primary Si 입자 이동, Fourier 혼합, 조명 필드 (+ 한 구성은 작은 스캔 4배) |
| 학습 | CE + Dice; B3 200 에폭 6개, SE-ResNeXt-50 60/200 에폭 각 3개 |
| 앙상블 | 12개 동일 가중, 8방향 TTA + tile 768 + Si 입자 투표 |
| val mIoU | **80.41%** (Al 94.5, Al₃Ni 75.8, Eutectic Si 61.6, Primary Si 91.6) |

- B3 학습은 `run_final_ep200.sh`, SE-ResNeXt 학습은 `run_seresnext.sh`/`run_encoders.sh`로 실행했습니다.
- val 20장으로 checkpoint를 고르고 같은 20장으로 평가한 점수라, 단일 모델 기준 0.5~0.8pp 낙관적입니다.
- 80.41%는 이전 9모델 앙상블(80.30%)대비 +0.11pp이며, paired bootstrap 95% 구간은 -0.04~+0.31pp입니다.
- test 예측은 아직 이 구성으로 만들지 않았습니다.

## 8. 남은 오류

현재 최고 모델에서 mIoU가 100에서 모자란 몫의 절반 가까이는 Eutectic Si(IoU 61.6), 그다음은 Al₃Ni(75.8)에서 나옵니다.

- Eutectic Si는 두께가 2~3px라 경계가 1px만 어긋나도 IoU가 크게 떨어집니다.
- Al₃Ni ↔ Eutectic Si 혼동이 전체 오류의 약 3분의 1입니다. 그중 상당수는 이미지 밝기와 정답 라벨이 어긋나는 곳이라,
  라벨 노이즈이거나 외관상 모호한 영역일 가능성이 있습니다.
- 675×480 스캔은 평균 76%로, 다른 크기(80~82%)보다 낮습니다.

## 9. 파일 위치

| 파일 | 내용 |
|---|---|
| `src/models.py` | SegFormer, MicroNet U-Net++, DINO ViT, `FullResSkip` |
| `src/losses.py` | 경계 loss 8종 |
| `train_v2.py` | 학습 (`--arch`, `--encoder`, `--fullres_skip`, `--loss`, `--ce_mode`, `--aux_loss` 등) |
| `run_boundary_suite.sh` | 경계 loss, fullres 변형 실험 일괄 실행 |
| `run_final_ep200.sh` | B3 200 에폭 학습 |
| `run_hrnet.sh`, `run_seresnext.sh`, `run_encoders.sh` | 후속 백본 비교 |
| `eval_with_best.py` | 신규 백본과 기존 최고 앙상블 조합 비교 |
| `summarize_runs.py`, `split_half_check.py` | seed 통계, 고른 효과 제거 |
