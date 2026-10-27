# 마스크 조건부 디퓨전 증강 실험 — D2

2026-10-04. **구현·시험 학습 완료, 본 실험의 분할 성능은 아직 미측정.**
목표는 train 70장의 이미지·마스크로 생성기를 학습하고, 새 이미지·마스크 쌍을 기존 분할 모델 학습에 섞는 것이다.
현재 최고 기준은 `b_cs2_aug3_ep200_s{0,1,2}` + `b_cs2_plus_ep200_s{0,1,2}` 6모델,
8방향 TTA + native tile 768 + Si 투표의 **validation mIoU 80.14%**다.

## 기존 D1에서 확인한 것

기존 `train_diffusion.py` / `sample_diffusion.py`는 회색 이미지 1채널과 4상 one-hot 마스크를
함께 확산시키는 joint EDM이다. 상 면적비를 조건으로 받고, 256px로 학습한 뒤 512px로 생성했다.
`runs/compare_b_cs2_aug3_vs_b_cs2_aug3_diff.json`의 같은 최종 추론 설정 비교는 다음과 같다.

| 60 epoch, 배율 2, 3 seed 앙상블 | mIoU |
|---|---:|
| 기존 `b_cs2_aug3` | 79.67% |
| D1 합성 crop 30% | 79.33% |
| 차이(반올림 전 점수에서 계산) | −0.35pp |

20장 중 개선 7장, 하락 13장. 고정된 두 앙상블의 이미지 단위 paired bootstrap 95% CI는
**[−0.84, +0.12]pp**다(`runs/md2_pilot/d1_paired.json`). 개선 근거가 없지만, 이것만으로
디퓨전 일반의 실패나 과적합을 단정할 수는 없다. 이 CI는 checkpoint 선택과 여러 실험 중 선택한 효과를 보정하지 않는다.

`synth/d1/quality.json`에서 실제/합성의 상별 평균 회색값은
Al 211.5/209.8, Al₃Ni 143.1/135.8, Eutectic Si 102.0/89.1, Primary Si 63.6/49.6이었다.
실데이터로 학습한 분할기의 합성 pair agreement는 79.30%, 실제 valid에서는 79.62%였다.
따라서 **전체 pair가 틀렸다고 볼 근거는 없으며**, 다음을 분리해서 점검한다.

1. Si가 더 어두운 명암 분포 차이. KoMaP에서는 명암이 상 구분 단서이므로 style 조건으로 다룬다.
2. 생성된 마스크 경계의 정확도. 공동 생성도 이미지와 마스크의 픽셀 일치를 보장하지 않는다.
3. 256→512 생성 크기 변화. 합성곱 모델이 실행된다는 사실과 같은 품질을 유지한다는 것은 다르다.
4. 합성 crop 30%는 같은 epoch 내 실제 crop 30%를 대체한다. 생성 품질과 실제 데이터 노출 감소가 섞인다.
5. D1의 3시간 학습만으로 충분한 수렴 여부는 판단할 수 없다.

## 논문에서 무엇을 가져왔나

2026-10-04 기준 공개 원문을 확인했다. 생명과학에서의 성능 향상 수치를 금속조직의 예상 향상으로 환산하지 않는다.

| 연구 | 핵심 방법 | 이번 선택과의 관계 |
|---|---|---|
| [NeuroDiff, ISBI 2026](https://arxiv.org/abs/2601.15779) | 3D EM 뉴런 마스크를 여러 해상도에서 조건으로 주입하고, 생물학적 제약으로 마스크를 변형한 뒤 새 영상을 생성 | **주요 설계 근거.** 얇은 구조를 조건으로 유지하는 2D 생성기로 적용. 원 논문의 3D Mamba·해상도 위치 인코딩·미토콘드리아 배치는 구현하지 않음 |
| [MSDM, 2025 preprint](https://arxiv.org/abs/2510.09121) | 세포/핵 형태의 HV map, RGB 색 통계, assay/indication 조건을 결합 | **주요 설계 근거.** 4상 one-hot + signed distance와 상별 회색 평균·표준편차로 단순화. HV instance map, BERT, 원 논문의 cross-attention 구현과는 다름 |
| [PF-DiffSeg, 2025 preprint](https://arxiv.org/abs/2508.00896) | 다상 철강에서 상 면적비 조건의 이미지·마스크 공동 생성 | 기존 D1과 가까운 축. 같은 방식을 더 크게 반복하기보다 마스크 조건부 방식을 먼저 비교 |
| [NuDiff, MICCAI 2023](https://arxiv.org/abs/2310.14197) | 구조 마스크/거리맵 생성 후 조건부 핵 이미지 생성 | 구조와 외관을 분리하는 기초 근거. 이번에는 구조 생성기를 추가 학습하지 않고 실제 GT 마스크를 재사용 |
| [EDM, NeurIPS 2022](https://arxiv.org/abs/2206.00364) | denoiser preconditioning, noise별 loss 가중, 수치적 sampling 설계 | 실제 학습·sampling의 기반. 최신 논문으로 소개하는 것은 아니며, 현재 환경에서 검증 가능한 pixel-space 구현으로 선택 |

**선정 모델: 2D pixel-space mask-conditional EDM U-Net, 약 10.09M parameters.**
NeuroDiff/MSDM의 아이디어를 KoMaP에 맞춰 구현한 실험이며 논문 모델의 재현이나 사전학습 가중치 사용은 아니다.
자연영상 latent diffusion + adapter보다 이 구성을 먼저 택한 이유는, 이 데이터의 Eutectic Si가
약 1.7~3.5px로 얇고 4상 명암 자체가 중요하기 때문이다. VAE 압축에 따른 세부 손실 가능성과
추가 도메인 적응 변수를 줄이려는 선택이지, latent 모델이 반드시 열등하다는 결론은 아니다.

## 학습·생성 설계

```text
train/train 70장
    ├─ 내부 60장 → 생성기 학습
    └─ 내부 10장 → 고정 denoising loss로 학습 step 선택
                  ↓
train/train 전체 70장으로 생성기를 처음부터 선택 step까지 다시 학습
                  ↓
실제 train 마스크 + train에서 고른 상별 명암 통계 + Gaussian noise
                  ↓
새 이미지 / 조건으로 사용한 마스크 1,000쌍
                  ↓
실제 crop 90% + 합성 crop 10% (별도 실험: 70% + 30%)
                  ↓
기존 분할 학습 recipe → 8방향 TTA + tile 768 + Si 투표
```

| 항목 | 기본값 |
|---|---|
| 생성기 입력/출력 | noisy grayscale 1채널 → clean grayscale 1채널; 마스크는 확산시키지 않음 |
| 공간 조건 | 4채널 one-hot + 4채널 signed distance, 거리 ±8 native px에서 clipping |
| 조건 주입 | U-Net encoder/decoder 여러 해상도, 얇은 상의 점유율을 보존하는 area pooling |
| 외관 조건 | 원본 train 스캔의 상별 평균·표준편차·존재 여부, 12차원 |
| 해상도 | 학습 512×512 / 생성 512×512, 원본 축소 없음; 작은 스캔은 reflect padding |
| train crop 증강 | 이미지·마스크에 같은 D4 변환, 추가 명암 jitter 없음 |
| optimizer | AdamW, lr 2e-4, warmup 500 step, gradient clip 2 |
| batch | 8 × accumulation 2 = 16 |
| noise | log(σ) ~ Normal(−0.4, 1.4), σ_data = 0.5 |
| loss | EDM MSE × (1 + 0.5×경계 + 0.5×Eutectic Si); 공간 가중치를 patch 평균 1로 정규화 |
| EMA | 최대 0.9999, 초기에는 더 작은 decay |
| selection 예산 | 최대 12,000 optimizer step 또는 3시간; 내부 평가 1,000 step마다 |
| checkpoint 기준 | 내부 10장 각각의 고정 crop과 고정 noise, σ={0.05,0.2,1,5,20}의 **가중치 없는** EDM loss 평균 |
| refit | 선택한 설정·step으로 전체 train 70장 재학습, select 가중치에서 이어 학습하지 않음 |
| sampling | EDM Heun 40 step, σ=80→0.002→0, ρ=7 |
| 합성량 | 기본 1,000쌍; 수량보다 downstream 효과를 우선 평가 |

경계/Si loss 가중치는 이번에 제안하는 ablation 대상이다. 인용 논문의 확정된 최적값이 아니다.
생성기 train/selection/refit/sampling은 공식 valid/test를 읽지 않는다. 내부 분할은 `A<number>` 파일명
접두사를 묶는다(seed 17, 실제 60/10). **이 접두사가 시편 ID라는 보장은 없으므로**, 시편 정보가 확보되면
시편별 그룹으로 바꾸어야 한다. 작은 내부 holdout의 denoising loss도 downstream 성능을 보장하지는 않는다.

기본 `appearance`는 실제 GT crop을 그대로 조건으로 쓴다. 외관 donor는 같은 스캔 높이 구간
(≤800 / >800)에 속하고 필요한 상을 모두 가진 다른 train 스캔에서 고른다. 생성기에는 donor의
원본 이미지 대신 상별 명암 통계만 들어간다. 추론 시 source 외관 고정은 `sample_mask_diffusion.py --style_mode source`다.
실제 마스크를 재사용하므로 새로운 형태의 다양성이 제한된다는 점도 대조군 실험에서 확인해야 한다.

선택 실험 `remodel`은 **마스크만** 약한 smooth displacement(축별 최대 0.75px)로 바꾼 뒤
그 조건에 맞는 **새 이미지**를 생성한다. 각 상과 그 여집합의 연결 성분 수가 바뀌거나
상 면적 변화가 max(2px, 5%)보다 크면 원래 마스크로 되돌린다. 이 검사는 완전한 위상 동일성이나
금속학적 타당성을 증명하지 않는다. `diagnostics.json`의 `remodeled_count`가 낮으면
실질적으로 appearance 실험과 비슷해질 수 있다. 원래 이미지에 변형된 마스크를 정답으로 붙이지 않는다.

## 비교 실험과 채택 기준

우선 기존 `b_cs2_aug3` 60 epoch를 복제한다. 원래 seed별 `history.json`에서 optimizer,
augmentation, crop, batch, scale, 검증 주기 등 설정을 그대로 읽는다. 변경점은 합성/대조 데이터 경로와 비율이다.

| 실험군 | 기존 실제 scan crop | 추가 데이터에서 뽑는 crop | 확인할 것 |
|---|---:|---:|---|
| R0: 기존 baseline | 100% | 0% | 기존 성능 |
| R10 / R30: real-repeat control | 90% / 70% | 같은 source·위치·D4의 실제 패치 10% / 30% | 반복 패치와 실제 crop 대체의 효과 |
| G10 / G30: D2 appearance | 90% / 70% | 생성 패치 10% / 30% | R0와 R10/R30 양쪽 대비 생성 이미지의 효과 |
| 선택 실험: D2 remodel | 90% / 70% | 변형 마스크에서 생성한 패치 10% / 30% | 형태 변형이 appearance보다 나은지 |

각 군 seed 0/1/2, 같은 optimizer step 수를 사용한다. 기본 appearance screen은 새 분할 학습
**12회 = 2비율 × (실제 반복/생성) × 3 seed**다. 실제 반복 대조군도 1,000쌍을 내보내며
같은 loader로 처리한다. remodel의 실제 대조군에는 변형 전 이미지와 원래 마스크가 들어간다.
이 경우 차이는 외관 생성과 구조 변형의 결합 효과이므로, appearance와의 비교도 필요하다.
합성 비율은 crop 선택 확률이며, 기존 Fourier/입자 이동 등 augmentation도 계속 적용된다.

기능별 확인은 `--variant`로 별도 생성기를 학습한다. 이름에 variant가 자동으로 붙는다.

| variant | 조건/손실 | 비교 목적 |
|---|---|---|
| `full` | one-hot + 거리맵 + style + 경계/Si 가중 | 제안 구성 |
| `no_geometry` | full에서 거리맵 제외 | 거리 정보 기여 |
| `no_style` | full에서 명암 조건 제외 | 외관 조건 기여 |
| `unweighted` | full에서 경계/Si 추가 가중 제외 | loss 가중 기여 |
| `mask_only` | one-hot 조건, style/거리맵/추가 가중 없음 | 단순한 조건부 diffusion 기준선; 여러 요소가 동시에 다름 |

모든 ablation을 한 번에 돌리기보다 full G10/R10, G30/R30부터 확인한다. 그 후 선택한 비율에서
ablation을 비교한다. 시간 제한과 선택 step이 달라질 수 있으므로 실제 step·학습 시간을 함께 기록한다.
strict compute ablation은 각 generator를 같은 step, 충분히 큰 동일 시간 상한으로 다시 실행한다.

판정은 `compare_groups.py`가 같은 seed와 최종 추론 조건으로 계산한다.

- 주 지표: 이미지별 mIoU의 평균, seed별 변화, 3모델 앙상블 변화.
- 보조 지표: 4상 IoU, ≤1px / 2–5px / >5px 오류, Al₃Ni→Eutectic / Eutectic→Al₃Ni 혼동.
- `summarize_mask_diffusion.py`: 10,000회 이미지 paired bootstrap과 파일명 그룹 cluster bootstrap CI.
  검증 데이터 재사용·여러 후보 선택·학습 seed 전체 불확실성을 포함한 유의성 검정으로 해석하지 않는다.
- 기존 실험 중 checkpoint 선택 편향은 프로젝트의 `split_half_check.py`도 함께 참고한다.
- 합성 이미지의 선명함이나 teacher agreement만으로 채택하지 않는다. teacher가 잘 맞히는 합성만
  고르면 기존 모델의 편향을 반복할 수 있어, 기본 파이프라인에 teacher 필터를 넣지 않았다.

**권장 진입 조건:** R0와 matched real-repeat 양쪽보다 좋아지고, Eutectic Si가 일관되게 악화되지
않는 후보 하나만 200 epoch로 확대한다. 이 조건은 실험 전 판단 원칙이지 자동 채택 규칙이 아니다.
CI가 넓거나 seed별 부호가 엇갈리면 미확정으로 기록한다. 최종은 기존 6모델과 새 6모델을
같은 추론 설정에서 비교한다. 60 epoch baseline 79.67%와 최종 baseline 80.14%를 직접 비교하지 않는다.
공식 test는 후보 선택에 사용하지 않는다. 최종 6모델 단계에서는 기존 최고 대비 실용적 개선을 보며,
생성 자체의 인과 효과를 엄격히 확인하려면 선택 비율의 real-repeat 200 epoch군도 추가해야 한다.

## 실행

```bash
cd /home/user/Desktop/komap

# 명령·설정만 확인 (학습하지 않음)
./.conda/bin/python run_mask_diffusion.py --stage plan

# 내부 selection → 전체 train refit
./.conda/bin/python run_mask_diffusion.py --stage generator

# 1,000개 합성 pair와 같은 source crop의 실제 대조군
./.conda/bin/python run_mask_diffusion.py --stage sample

# 60 epoch, 10%/30%, 3 seed, matched control과 비교
./.conda/bin/python run_mask_diffusion.py --stage screen

# 위 세 단계를 순차 실행하려면
./.conda/bin/python run_mask_diffusion.py --stage all

# 선택한 후보만 200 epoch × 6모델로 평가: 10%가 선택되었다는 예시
./.conda/bin/python run_mask_diffusion.py --stage final --ratios 0.1 --modes appearance

# 선택적 형태 변형: 기존 refit 생성기를 재사용
./.conda/bin/python run_mask_diffusion.py --stage sample --modes remodel
./.conda/bin/python run_mask_diffusion.py --stage screen --modes remodel --ratios 0.1

# 예: style 조건의 효과를 같은 비율에서 별도 실험
./.conda/bin/python run_mask_diffusion.py --stage all --variant no_style --ratios 0.1
```

`all`은 generator/sample/screen까지만 실행한다. final은 비교 결과를 보고 선택한 mode/ratio를
명시해서 실행한다. `--seeds 0`은 빠른 탐색용으로 가능하며 baseline도 같은 seed로 제한된다.
최종 6모델은 반드시 `0 1 2`를 사용한다. 기존 최고 checkpoint를 수정하거나 자동 대체하지 않는다.

기본 로그는 `runs/md2/{select,refit,sample_appearance,...}.log`, 생성기 checkpoint는
`runs/md2/{select,refit}/best.pt`, 데이터는 `synth/md2_appearance`와
`synth/md2_control_appearance`, 분할 모델은 `runs/md2_*_r10_s0` 등의 경로에 저장한다.
screen 비교 JSON은 `runs/compare_*_seeds0_1_2.json`, CI는 같은 이름의 `.paired.json`이다.
최종 점수/예측과 paired 비교는 `runs/md2/final_appearance_r10_*`에 저장한다.

기존 완료 단계는 건너뛴다. 중단된 generator는 `last.pt`에서 optimizer/EMA를 복구한다.
data worker의 정확한 crop 순서까지 복원하지는 않는다. generator 예산/데이터 수량 등 설정을 바꿀 때는
`--name md2_newrecipe`처럼 새 이름을 사용한다. 샘플 생성은 불완전한 기존 폴더에 덮어쓰지 않으며,
중단된 sampling은 새 출력 경로나 새 실험 이름으로 다시 시작한다. 분할 학습 자체에는 resume 기능을 추가하지 않았다.

## 구현 파일과 재현 기록

| 파일 | 역할 |
|---|---|
| `src/mask_diffusion.py` | 공간/명암 조건부 U-Net, EDM loss, Heun sampler |
| `src/mask_diffusion_data.py` | train 전용 reader, 내부 그룹 분할, D4 crop, 제한된 mask remodeling |
| `train_mask_diffusion.py` | selection/refit, EMA, checkpoint/resume, 고정 내부 평가 |
| `sample_mask_diffusion.py` | 생성 pair + 실제 반복 대조군, manifest, preview, 명암 진단 |
| `run_mask_diffusion.py` | 기존 recipe를 상속하는 단계별 실험 실행 |
| `summarize_mask_diffusion.py` | 같은 이미지·seed의 차이, bootstrap CI |
| `tests/test_mask_diffusion.py` | 얇은 상 보존, 조건 gradient, sampler, 분할/재학습/내보내기/loader 통합 검사 |

생성기 checkpoint와 `split.json`에 원본 파일 이름·SHA256·학습 분할을 기록한다.
합성/실제 대조군의 `manifest.jsonl`에는 원본 스캔 높이, crop 좌표, D4, style donor,
마스크 hash가 기록된다. **조건 마스크를 저장했다는 사실이 생성 이미지의 경계 일치를 증명하지는 않는다.**
`preview.png`와 `diagnostics.json`에서 구조 누락·명암 편차를 확인하고 최종 판단은 분할 실험으로 한다.

`train_v2.py`는 이 manifest의 `source_height`를 읽도록 보완했다. 예를 들어 1024px 높이 스캔에서
만든 512px 합성 패치를 `cs2_plus`가 작은 스캔으로 착각해 4배 확대하지 않고, 원본 기준 2배로 처리한다.
작은 스캔에서 온 패치는 4배를 유지한다. manifest가 없는 기존 D1 데이터의 동작은 그대로다.

2026-10-04 실제 검증:

- CPU 통합 테스트: select → refit → 생성 → 실제 대조군 → segmentation loader까지 통과.
- RTX PRO 6000 GPU, 실제 train 내부 60장, 512px, 200 optimizer step 시험 학습 완료.
  내부 denoising loss는 step 100의 0.3543에서 step 200의 0.2106으로 감소.
- `runs/md2_pilot/samples`에 실제 512px 생성 pair 4개, `real_control`에 대응 실제 pair 4개를 내보냄.
  200-step 모델은 미수렴 상태이며 분할 학습용으로 채택하지 않음.
- 시험 이미지: `runs/md2_pilot/preview_000200.png`, `runs/md2_pilot/samples/preview.png`.
  세로 순서는 실제 이미지 / 조건 마스크 / 생성 이미지. 거친 질감이 남아 있어 품질 개선 여부를 더 학습해서 확인해야 함.
- pilot의 안정 구간은 optimizer step당 약 1.3초였다. 12,000 step은 학습 부분만 약 4.3시간이므로
  기본 3시간 제한이 먼저 작동할 수 있다. refit은 선택된 step만큼 추가 시간이 들며, sampling/분할 학습 시간은 별도다.

본 학습·1,000쌍 생성·12회 분할 screen·최종 6모델 재학습은 아직 실행하지 않았다.
