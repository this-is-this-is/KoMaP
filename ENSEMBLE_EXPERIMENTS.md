# 앙상블 추론 실험 정리 (8방향 TTA + tile 768 + Si 투표)

2026-10-01 기준. 모든 수치는 validation 20장의 대회 지표(이미지별 4-class mIoU의 평균)입니다.

## 결론

- **추론 설정:** 8방향 TTA + tile 768 + Si 입자 투표. 재학습 없이 77.61% → 78.09% (+0.48pp).
- **현재 최고 구성:** color_scale2 ×3 seed = **79.61%** (7절, 2026-10-01 저녁 갱신).
  color_scale2는 fullres에 aug_color와 배율 2를 함께 적용한 모델입니다.
- 그 전 최고는 fullres B3 ×3 seed + B4 = 78.47%였고, fullres 계열끼리는 78.43–78.47%에서 포화했습니다.

## 추론 설정의 세 요소

| 요소 | 옵션 | 내용 |
|---|---|---|
| 8방향 TTA | `--tta --d4` | 타일마다 좌우·상하 반전 4가지 × transpose 2가지를 추론해 확률을 평균 |
| tile 768 | `--tile 768` | 겹치는 타일의 크기를 512에서 768로 키움(겹침 50%). 타일 경계 영향이 줄어듦 |
| Si 입자 투표 | `--si_vote` | 예측된 Eutectic Si와 Primary Si를 하나의 연결 성분으로 묶고, 성분마다 두 클래스의 확률 합이 큰 쪽 하나로 통일 |

Si 입자 투표를 넣은 이유: Eutectic Si ↔ Primary Si 혼동의 99%가 경계가 아니라 입자를 통째로
잘못 분류한 경우였습니다. 세 요소 모두 조정할 파라미터가 없습니다.

## 1. 추론 설정 비교 (모델 고정: B3 + B4)

모델은 `runs/s2_micronet_unetpp` + `runs/s4_micronet_b4`로 고정하고 추론 방법만 바꿨습니다.

| 타일 | TTA | Si 투표 없음 | Si 투표 있음 |
|---|---|---|---|
| 512 | 반전 4방향 (기존 채택) | 77.61% | 77.78% |
| 512 | 8방향 | 77.65% | 78.08% |
| 768 | 반전 4방향 | 77.71% | 78.05% |
| **768** | **8방향** | 77.76% | **78.09%** |
| 이미지 전체 | 반전 4방향 | 77.76% | 78.05% |
| 이미지 전체 | 8방향 | 77.78% | 78.07% |
| 1024 | 반전 4방향 | 77.79% | – |

- 최종 설정(768 + 8방향 + 투표)은 기존 대비 20장 중 15장이 좋아지고 4장이 나빠졌습니다.
- 타일 768, 이미지 전체, 1024는 차이가 0.03pp 이내라 사실상 같습니다.
- 클래스별 IoU(최종 설정): Al 93.6, Al₃Ni 72.7, Eutectic Si 57.9, Primary Si 90.0.
  기존 대비 Eutectic Si +1.1, Primary Si +0.8이고 나머지는 같습니다.

효과가 없었던 것:

- 클래스별 logit bias 보정: leave-one-out으로 −0.04pp.
- 다른 계열 모델 추가(tile 512, 반전 TTA 기준): SegFormer를 넣으면 77.52%, DINOv3-L/16을 넣으면
  77.31%로 B3 + B4(77.61%)보다 낮아집니다.

## 2. 앙상블 구성 비교 (추론 설정 고정)

추론은 최종 설정(8방향 TTA + tile 768 + Si 투표)으로 고정하고 모델 구성만 바꿨습니다.

| 구성 | 모델 수 | val mIoU | Al | Al₃Ni | Eutectic Si | Primary Si |
|---|---|---|---|---|---|---|
| B3 + B4 (기존 채택) | 2 | 78.09% | 93.6 | 72.7 | 57.9 | 90.0 |
| B3 ×3 seed (baseline) | 3 | 78.04% | 93.6 | 72.1 | 58.2 | 90.1 |
| fullres B3 ×3 seed | 3 | 78.35% | 93.9 | 72.9 | 58.3 | 90.1 |
| fullres B4 ×3 seed | 3 | 78.31% | 93.9 | 73.2 | 57.9 | 90.0 |
| **fullres B3 ×3 + B4** | 4 | **78.47%** | 93.9 | 73.1 | 58.5 | 90.3 |
| fullres B3 ×3 + fullres B4 (seed 2) | 4 | 78.47% | 93.9 | 73.1 | 58.4 | 90.3 |
| fullres B3 ×3 + fullres B4 ×3 | 6 | 78.45% | 93.9 | 73.2 | 58.3 | 90.2 |
| fullres(Lovász) ×3 + fullres B4 ×3 | 6 | 78.44% | 93.9 | 72.9 | 58.5 | 90.3 |
| fullres(채널 64) ×3 + fullres B4 ×3 | 6 | 78.43% | 93.9 | 72.9 | 58.6 | 90.1 |
| fullres 계열 B3 ×9 + fullres B4 ×3 | 12 | 78.46% | 93.9 | 72.9 | 58.5 | 90.4 |

용어:

- **fullres**: U-Net++에 원본 해상도(stride 1) branch를 추가한 모델(`--fullres_skip`).
  단일 모델로는 3 seed 평균 77.72%이고, 같은 조건 baseline은 77.02%입니다.
- **B4**: `runs/s4_micronet_b4` (EfficientNet-B4, fullres 없음).
- B3 ×3 seed (baseline): `runs/seed_base_{0,1,2}`. fullres B3 ×3 seed: `runs/b_fullres_s{0,1,2}`.

읽는 법:

- **fullres의 이득은 앙상블에서도 유지되지만 줄어듭니다.** 단일 모델에서는 +0.70pp인데, 3-seed
  앙상블끼리는 78.04 → 78.35%로 +0.31pp입니다. 앙상블과 TTA도 경계 흔들림을 줄이기 때문입니다.
- **같은 계열을 더 넣어도 오르지 않습니다.** 4개(78.47%)와 12개(78.46%)가 같습니다. 남은 오류는
  이 모델들이 공통으로 틀리는 것입니다.
- **B4는 무엇을 넣든 같습니다.** 기존 B4와 fullres B4의 차이가 0.02pp 이내입니다.

## 3. 오류 유형별 분해 (최종 추론 설정)

오류 픽셀마다 "예측한 클래스의 GT 영역까지의 거리"로 나눈 것입니다(`--breakdown`).
괄호 안은 그 유형을 전부 고쳤을 때 오르는 mIoU 상한입니다.

| 구성 | ≤1px 경계 위치 오류 | 2–5px | >5px 상 오분류 |
|---|---|---|---|
| B3 + B4 (기존 채택) | 57.1% (+9.88pp) | 15.8% (+3.34pp) | 27.2% (+6.25pp) |
| B3 ×3 seed (baseline) | 56.0% (+9.78pp) | 15.5% (+3.32pp) | 28.5% (+6.36pp) |
| fullres B3 ×3 seed | 54.8% (+9.37pp) | 16.2% (+3.41pp) | 29.0% (+6.42pp) |
| fullres B3 ×3 + B4 | 55.4% (+9.44pp) | 16.0% (+3.38pp) | 28.6% (+6.28pp) |

fullres는 ≤1px 경계 오류를 줄였고(+9.78 → +9.37pp), 상 오분류는 그대로입니다.

## 4. seed 앙상블만 늘렸을 때 (참고)

tile 512, 반전 TTA, 투표 없음 기준으로 baseline 모델 수만 늘린 결과입니다.

| 구성 | val mIoU |
|---|---|
| B3 + B4 | 77.61% |
| B3 ×3 seed | 77.70% |
| B3 ×3 seed + B4 | 77.79% |
| B3 ×6 | 77.78% |
| B3 ×6 + B4 (7개) | 77.84% |
| 7개 + 최종 추론 설정 | 78.14% |

7개 앙상블에 최종 추론 설정을 써도 78.14%로, 2개(78.09%)와 0.05pp 차이입니다.
모델 수를 늘리는 것보다 추론 설정과 모델 구조(fullres)를 바꾼 효과가 컸습니다.

## 5. 실행 명령

```bash
cd ~/Desktop/komap

# val 점수와 오류 분해 (현재 최고 구성)
./.conda/bin/python predict.py \
    --ckpt runs/b_fullres_s0/best.pth runs/b_fullres_s1/best.pth runs/b_fullres_s2/best.pth \
           runs/s4_micronet_b4/best.pth \
    --split valid --tta --d4 --tile 768 --si_vote --breakdown

# test 10장 예측 저장
./.conda/bin/python predict.py \
    --ckpt runs/b_fullres_s0/best.pth runs/b_fullres_s1/best.pth runs/b_fullres_s2/best.pth \
           runs/s4_micronet_b4/best.pth \
    --split test --tta --d4 --tile 768 --si_vote --save_dir preds/test_fullres3_b4
```

## 6. 주의할 점

- val 20장으로 checkpoint(best epoch)를 고르고 같은 20장으로 평가했습니다. 절대 수치는 낙관적이고,
  구성 간 비교만 유효합니다.
- 구성 간 차이가 0.1pp 이하인 것은 구분할 수 없습니다. 단일 모델의 seed 노이즈가 ±0.15pp입니다.
- `preds/test_ensemble`의 test 예측은 아직 기존 구성(B3 + B4, 반전 TTA, tile 512)입니다.
- 2절의 표는 3라운드 이전 모델만 다룹니다. aug_color, scale2, color_scale2는 7절에 있습니다.

## 7. 증강 모델을 넣은 앙상블 (2026-10-01 저녁)

`./.conda/bin/python eval_ensembles.py`의 결과입니다(약 3분 30초). 추론 설정은 같고, 배율 2 모델은
이미지를 2배로 확대해 추론합니다. "개선/하락"은 fullres ×3 + B4 대비 20장 중 이미지 수입니다.

| 구성 | 모델 수 | val mIoU | 차이 | 개선/하락 | Al | Al₃Ni | Eutectic Si | Primary Si |
|---|---|---|---|---|---|---|---|---|
| fullres ×3 + B4 (이전 최고) | 4 | 78.47% | – | – | 93.9 | 73.1 | 58.5 | 90.3 |
| aug_color ×3 | 3 | 78.91% | +0.44 | 15/5 | 94.0 | 74.1 | 58.8 | 90.5 |
| scale2 ×3 | 3 | 78.99% | +0.51 | 13/7 | 93.8 | 73.6 | 59.6 | 90.9 |
| aug_color ×3 + scale2 ×3 | 6 | 79.07% | +0.59 | 15/4 | 94.0 | 74.1 | 59.3 | 90.6 |
| aug_color ×3 + scale2 ×3 + B4 | 7 | 79.08% | +0.60 | 17/2 | 94.0 | 74.1 | 59.4 | 90.7 |
| **color_scale2 ×3** | 3 | **79.61%** | +1.14 | 17/3 | 94.1 | 74.8 | 60.3 | 91.0 |
| color_scale2 ×3 + B4 | 4 | 79.47% | +1.00 | 18/2 | 94.1 | 74.6 | 60.1 | 90.9 |
| color_scale2 ×3 + aug_color ×3 + scale2 ×3 | 9 | 79.37% | +0.89 | 17/3 | 94.1 | 74.4 | 59.9 | 90.9 |
| 위 9개 + B4 | 10 | 79.32% | +0.85 | 18/2 | 94.1 | 74.4 | 59.9 | 90.8 |
| fullres ×3 + aug_color ×3 + scale2 ×3 + B4 | 10 | 78.83% | +0.35 | 19/1 | 94.0 | 73.8 | 58.9 | 90.4 |

- 두 방법을 한 모델에 합친 color_scale2가, 따로 학습한 모델을 앙상블로 섞은 것(79.07%)보다 높습니다.
- color_scale2에 더 약한 모델(B4, aug_color, scale2, fullres)을 섞으면 점수가 내려갑니다.
- 단일 모델(학습 중 검증, 3 seed 평균): fullres 77.72, aug_color 78.21, scale2 78.21, color_scale2 78.74.

```bash
./.conda/bin/python predict.py \
    --ckpt runs/b_color_scale2_s0/best.pth runs/b_color_scale2_s1/best.pth runs/b_color_scale2_s2/best.pth \
    --split test --tta --d4 --tile 768 --si_vote --save_dir preds/test_color_scale2
```
