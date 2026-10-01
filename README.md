# KoMaP — 미세조직 상 분할 (phase segmentation)

고탄성 주조용 알루미늄 합금 현미경 이미지(OM image)를 4개 상으로 분할합니다.
**"원본을 축소하지 않고 512×512 crop으로 학습 → 원본 해상도에서 겹치는 타일로 추론"**
전략을 기본으로 합니다.

## 대회 기준

$$\mathrm{Score}=\frac{100}{10}\sum_{i=1}^{10}\Big[\frac14\sum_{c=1}^{4}\frac{TP_{i,c}}{TP_{i,c}+FP_{i,c}+FN_{i,c}}\Big]$$

이미지별로 4개 클래스 IoU를 평균한 뒤, 이미지 10장에 대해 다시 평균합니다.
전체 confusion matrix를 합산하는 방식과 값이 다르므로, 모델 선택·early stopping·
ensemble 채택은 **모두 이 이미지별 평균 기준**으로 했습니다. 참고 기준 성능치는 **약 75%**.

## 결과 (validation 20장)

| 단계 | 모델 | val mIoU | Al | Al₃Ni | Eutectic Si | Primary Si |
|---|---|---|---|---|---|---|
| 1 | SegFormer-B2 | 72.97% | – | – | – | – |
| 2 | **MicroNet EfficientNet-B3 + U-Net++** | **77.00%** | 93.3 | 71.9 | 55.8 | 88.8 |
| 3 | DINOv2 ViT-B/14 + mask head | 56.06% | – | – | – | – |
| 4 | MicroNet EfficientNet-B4 + U-Net++ | 76.91% | – | – | – | – |
| 5 | **DINOv3 ViT-B/16** + mask head | 63.29% | – | – | – | – |
| 5 | **DINOv3 ViT-L/16** + mask head | 64.00% | 82.1 | 57.0 | 34.1 | 85.5 |
| 6 | B3 + TTA | 77.24% | 93.4 | 72.2 | 56.3 | 88.9 |
| 6 | **B3 + B4 ensemble + TTA** | **77.61%** | 93.6 | 72.7 | 56.8 | 89.2 |
| 6 | B3 + B4 + SegFormer ensemble + TTA | 77.52% | 93.2 | 72.3 | 56.8 | 89.5 |
| 6 | B3 + B4 + DINOv3-L/16 ensemble + TTA | 77.31% | 93.1 | 72.1 | 56.4 | 89.4 |
| 7 | B3 + B4, 8방향 TTA + tile 768 + Si 입자 투표 | 78.09% | 93.6 | 72.7 | 57.9 | 90.0 |
| 8 | **B3 + 원본 해상도 skip** (3 seed 평균; 같은 조건 baseline 77.02%) | **77.72%** | 93.7 | 72.5 | 57.0 | 89.4 |
| 8 | **fullres B3 ×3 + B4**, 8방향 TTA + tile 768 + Si 입자 투표 | **78.47%** | 93.9 | 73.1 | 58.5 | 90.3 |

**채택: B3 + B4 ensemble + TTA (77.61%)** — SegFormer(77.52%)나 DINOv3-L/16(77.31%)을
추가하면 오히려 낮아져 제외했습니다. 성능 차가 큰 모델을 넣으면 앙상블을 끌어내립니다.
테스트 10장은 이 선택에 사용하지 않았습니다.

**2026-10-01 갱신:** 경계 실험 결과 단계 8 구성(78.47%)이 현재 최고입니다(아래
*경계 정밀도 실험* 참고). `preds/test_ensemble`의 테스트 예측은 아직 이전 구성 그대로입니다.

### 관찰된 점

**현미경 사전학습이 이 데이터에서 가장 크게 기여했습니다.** MicroNet(NASA) 사전학습
EfficientNet + U-Net++가 SegFormer-B2보다 +4.0%p 높았습니다. 즉 이 데이터에서는
"사전학습 도메인이 얼마나 가까운가"가 "모델이 얼마나 최신인가"보다 중요했습니다.

**DINO 계열은 크게 뒤처졌습니다.** DINOv2 56.1% → DINOv3-B/16 63.3% → DINOv3-L/16 64.0%로
DINOv3가 DINOv2보다 +7.9pp 좋지만, MicroNet U-Net++(77.0%)에는 13pp 못 미칩니다.
클래스별로 보면 원인이 분명합니다 — DINOv3-L/16의 Eutectic Si IoU가 **34.1%** 로
MicroNet(55.8%)의 60% 수준입니다. patch 16×16의 coarse feature를 upsample하는 구성이
평균 두께 3.1px인 얇은 상에 불리하다는 예상과 일치합니다. B/16 → L/16으로 4배 키워도
+0.7pp에 그쳐, **용량이 아니라 feature 해상도가 병목**임을 보여줍니다.

단, 이 수치를 **"DINOv3+EoMT의 성능"으로 해석하면 안 됩니다.** 여기 구현은 EoMT가 아니라
ViT + 경량 conv head입니다 (아래 *미구현* 참고).

**해상도가 모델 크기보다 결정적이었습니다.** 전체 이미지를 512로 축소하면 정답 마스크를
축소·복원하는 것만으로도 상한이 **66.9%**로 제한됩니다(`Eutectic Si` 상한 44.2%).
crop+tile로 바꾼 것이 75% 돌파의 핵심이었고, 인코더를 B3→B4로 키운 효과는 미미했습니다.

## 클래스 / 색상

마스크 PNG에서 직접 확인한 팔레트입니다.

| idx | 상 | 색 | RGB | 픽셀 비율 |
|---|---|---|---|---|
| 0 | Al | 갈색 | (153, 127, 76) | 57.98% |
| 1 | Al₃Ni | 파란색 | (25, 76, 153) | 25.35% |
| 2 | Eutectic Si (Si_E) | 흰색 | (204, 204, 204) | 7.10% |
| 3 | Primary Si (Si_P) | 초록색 | (76, 178, 76) | 9.57% |

Al도 평가 대상 클래스이므로 background로 빼지 않았고, 5번째 클래스도 추가하지
않았습니다. SegFormer의 `do_reduce_labels`(label 0 무시)는 **사용하지 않았습니다** —
켜면 Al이 사라집니다.

**정답·예측 모두에 없는 클래스**는 IoU 분모가 0이라 평균에서 제외합니다. 제공된 마스크
2장에 Primary Si가 아예 없어서, 0점 처리하면 해당 이미지가 약 25점 낮아집니다.
공식 평가 코드가 1로 처리한다면 `src/komap.py`의 `per_image_miou`를 그에 맞게 바꾸세요.

## 환경

이 폴더의 `./.conda` 프리픽스에 독립 설치되어 있습니다(WeatherCity/Omnire 환경과 분리).

```
torch 2.11.0+cu128 (sm_120 Blackwell), smp 0.5.0, transformers 5.17.0,
albumentations 2.0.8, timm 1.0.29
```

## 사용법

```bash
cd ~/Desktop/komap

# 1) 기준 모델 — 파이프라인 검증용
./.conda/bin/python train_v2.py --arch segformer_b2 --out runs/s1_segformer_b2

# 2) 메인 추천 — 현미경 사전학습
./.conda/bin/python train_v2.py --arch unetpp --encoder efficientnet-b3 \
    --pretrain micronet --out runs/s2_micronet_unetpp

# 3) DINO 계열 (encoder는 낮은 lr)
./.conda/bin/python train_v2.py --arch dinov2 --lr 1e-4 --enc_lr 1e-5 \
    --out runs/s3_dinov2
./.conda/bin/python train_v2.py --arch dinov2 --freeze_encoder --out runs/s3_frozen

# DINOv3 (가중치는 gated — weights/dinov3/ 에 수동 배치, 저자 레포 필요)
./.conda/bin/python train_v2.py --arch dinov3 --lr 1e-4 --enc_lr 1e-5 --out runs/s5_dinov3_b16
./.conda/bin/python train_v2.py --arch dinov3_vitl16 --lr 1e-4 --enc_lr 1e-5 --out runs/s6_dinov3_l16

# 검증 점수 (대회 지표와 동일)
./.conda/bin/python predict.py --ckpt runs/s2_micronet_unetpp/best.pth --split valid --tta

# 평가용 10장 — 최종 ensemble
./.conda/bin/python predict.py \
    --ckpt runs/s2_micronet_unetpp/best.pth runs/s4_micronet_b4/best.pth \
    --split test --tta --save_dir preds/test_ensemble
```

주요 옵션: `--tile 768`, `--overlap 0.25`, `--class_ratio`, `--samples_per_image`,
`--pretrain {micronet,image-micronet,imagenet,none}`.

## 경계 정밀도 실험 (2026-09-30)

### 오류가 실제로 어디서 나오나

`predict.py --breakdown`은 오류 픽셀마다 "예측한 클래스의 GT 영역까지의 거리"를 잽니다.
"GT 경계 ±2px 안"으로 세면, 3px짜리 Eutectic Si lamella를 통째로 잘못 분류한 것도
경계 오류로 잡히기 때문입니다. B3+B4 + flip TTA 기준:

| 오류 유형 | 오류 픽셀 비중 | 전부 고칠 때 상한 |
|---|---|---|
| ≤1px: 경계 위치 오류 | 56% | +9.9pp |
| 2–5px | 14% | +3.4pp |
| >5px: 상 자체를 잘못 분류 | 30% | +6.6pp |

클래스별 bias(threshold) 보정은 leave-one-out에서 −0.04pp로 효과가 없었습니다.
예측/GT 면적비가 0.97–1.05라 두께 편향은 없고, 경계 오류는 대칭적인 ±1px 흔들림입니다.

### 재학습 없는 추론 옵션 (val 20장, B3+B4)

| 옵션 | val mIoU |
|---|---|
| `--tta` (기존 채택) | 77.61% |
| `--tta --d4 --tile 768` | 77.76% |
| `--tta --d4 --tile 768 --si_vote` | **78.09%** (20장 중 15장 개선, 4장 하락) |

`--d4`는 transpose까지 더한 8방향 TTA입니다. `--si_vote`는 연결된 Si 입자마다
Eutectic/Primary 중 확률 합이 큰 쪽 하나로 통일하며, 파라미터가 없습니다.

```bash
./.conda/bin/python predict.py \
    --ckpt runs/s2_micronet_unetpp/best.pth runs/s4_micronet_b4/best.pth \
    --split test --tta --d4 --tile 768 --si_vote --save_dir preds/test_ensemble_d4_vote
```

### 학습 실험 (`run_boundary_suite.sh`)

| 이름 | 플래그 | 근거 |
|---|---|---|
| fullres | `--fullres_skip` | smp U-Net++는 stride-1 특징을 버림 → 원본 해상도 branch 추가 (FR-UNet 착안) |
| relax | `--ce_mode relax` | 경계 label relaxation (Zhu+ CVPR 2019) |
| jml_bls | `--loss ce_jml --bls_eps 0.5` | JML + 경계 label smoothing, k=3, ε=0.5, CE:JML=0.25:0.75 (Wang+ NeurIPS 2023) |
| jml | `--loss ce_jml` | jml_bls의 대조군 (같은 loss, hard label) |
| interface | `--ce_mode interface` | Al₃Ni–Si 계면 1.5, Si 1.25 가중 (Strohmann+ Sci. Rep. 2019) |
| bdou | `--loss ce_bdou` | Dice 대신 Boundary DoU (Sun+ MICCAI 2023) |
| kervadec | `--aux_loss kervadec` | boundary loss, 가중치가 epoch마다 +0.01 (Kervadec+ MIDL 2019) |
| abl | `--loss ce_lovasz --aux_loss abl` | Active Boundary Loss, 논문의 w=1.0에 해당 (Wang+ AAAI 2022) |
| lovasz | `--loss ce_lovasz` | abl의 대조군 (ABL 없이 같은 base loss) |

B-DoU, ABL, Kervadec 거리맵은 저자 코드와 같은 값을 내는 것을 확인했습니다.
seed 노이즈가 ±0.15pp라 방법마다 3 seed를 돌립니다. run 하나는 약 10분이고,
3개를 동시에 돌리면 처리량이 약 1.4배가 됩니다.

```bash
JOBS=3 ./run_boundary_suite.sh             # 9종 × 3 seed (약 3시간 20분)
JOBS=3 SEEDS=0 ./run_boundary_suite.sh     # seed 0만 먼저 훑기 (약 1시간 10분)
./run_boundary_suite.sh fullres relax      # 일부 방법만
./.conda/bin/python summarize_runs.py      # seed 평균 ± sd, baseline 대비 차이, Welch p
```

### 결과 (2026-10-01, 각 3 seed)

baseline은 같은 조건의 ce_dice 3 seed(`runs/seed_base_{0,1,2}`, 77.02 ± 0.13)입니다.
`runs/s2_micronet_unetpp`는 `--enc_lr 1e-4`로 학습되어 기준에서 뺐습니다.
"마지막 5회"는 epoch 52–60 검증의 평균으로, best epoch 선택의 운을 뺀 비교입니다.

| 그룹 | best | 차이 | p | 마지막 5회 차이 | p |
|---|---|---|---|---|---|
| **fullres** | **77.72** | **+0.70** | <0.01 | **+1.00** | 0.03 |
| jml | 77.30 | +0.28 | 0.29 | +0.40 | 0.33 |
| abl | 77.13 | +0.11 | 0.45 | +0.09 | 0.79 |
| lovasz (abl 대조군) | 77.10 | +0.08 | 0.47 | +0.45 | 0.14 |
| interface | 77.07 | +0.05 | 0.82 | +0.19 | 0.45 |
| bdou | 76.92 | −0.10 | 0.60 | +0.05 | 0.89 |
| kervadec | 76.25 | −0.77 | 0.08 | −0.76 | 0.07 |
| jml_bls | 76.02 | −0.99 | <0.01 | −0.64 | 0.09 |
| relax | 75.05 | −1.96 | <0.01 | −1.82 | <0.01 |

- **원본 해상도 skip만 확실히 효과가 있습니다.** 20장 중 17장이 좋아졌고, 경계가 조밀한
  675×480 이미지에서 +1.12pp(Al₃Ni +2.05), 큰 이미지에서 +0.35pp입니다. 줄어든 건
  ≤1px 경계 오류(전부 고칠 때 상한 +9.78 → +9.37pp)이고, >5px 상 오분류는 그대로입니다.
  최종 추론 설정의 3-seed 앙상블끼리는 78.04 → 78.35%로 이득이 줄지만 유지됩니다.
- **경계 라벨을 느슨하게 하는 방법은 해롭습니다.** 3px짜리 상에서는 "경계 픽셀"이 곧 상
  전체라서, label relaxation(3×3 창)은 lamella 전체의 감독을 없앱니다. relax 모델은 Al₃Ni
  예측 면적이 GT 대비 0.96 → 0.91로 줄고 Al이 늘었습니다. BLS와 Kervadec도 같은 방향으로
  얇은 상을 줄였습니다. 즉 ±1px 경계 오류는 주석 노이즈보다 모델 해상도의 한계에 가깝습니다.
- ABL, 계면 가중, B-DoU, JML, Lovász는 seed 노이즈 안이라 효과가 없습니다.

### 2라운드 결과: 해상도 축 (2026-10-01, 각 3 seed, fullres 대비)

| 그룹 | best | fullres 대비 | p | 마지막 5회 차이 | p |
|---|---|---|---|---|---|
| fullres (B3, 채널 32) | 77.72 | – | – | – | – |
| fullres_ch64 | 77.52 | −0.19 | 0.08 | −0.17 | 0.23 |
| fullres_lovasz | 77.43 | −0.29 | 0.03 | −0.26 | 0.04 |
| fullres_b4 (enc_lr 1e-4) | 77.28 | −0.44 | 0.26 | −0.41 | 0.27 |

- stride-1 branch를 넓혀도(32 → 64) 나아지지 않습니다. 이득은 원본 해상도를 본다는 사실
  자체에서 오고, 그 branch의 용량은 병목이 아닙니다.
- Lovász를 fullres 위에 얹으면 오히려 조금 떨어집니다.
- fullres B4는 seed별로 76.71 / 77.50 / 77.63%로 편차가 큽니다. 같은 조건의 B4(s4, seed 1)는
  76.91%여서 B4에서 fullres 효과는 판단하기 어렵습니다.
- 최종 추론 설정(8방향 TTA + tile 768 + Si 투표)에서 앙상블은 78.43–78.47%에서 더 오르지
  않습니다. fullres×3 + s4가 78.47%, s4 대신 fullres_b4를 넣어도 78.45–78.47%, 12개 전부
  넣어도 78.46%입니다. 같은 계열 모델을 더 넣어서는 남은 오류가 줄지 않습니다.

## 증강·샘플링 옵션 (3라운드, fullres 위에서 비교)

| 증강·샘플링 | 플래그 | 구현 |
|---|---|---|
| 90도 회전 + 반전 | `--aug_d4` | D4 8가지 중 균등 선택(`A.D4`). 기존 flip + rot90 조합은 8가지가 1:3 정도로 치우쳐 있었음 |
| 원본 해상도 random crop | `--tile 512` (기본) | 기존 동작 그대로. `--class_ratio 0`이면 전부 무작위 위치 |
| 밝기·대비·감마 | `--color_aug oneof` (`--color_p 0.5`) | 밝기 ±10%, 대비 0.9–1.1, 감마 0.9–1.1 중 하나만 적용 |
| 배율 변화 | `--input_scale 1.5` / `2` | 학습·추론 모두 스캔을 s배 확대하고 점수는 원본 해상도에서 매김. `--tile`은 원본 픽셀 기준이라 시야는 그대로(crop 768 / 1024) |
| 얇은 조직 중심 샘플링 | `--thin_ratio 0.3` | crop의 30%는 얇은 Eutectic Si 비율에 비례해 스캔을 고르고, 가장 조밀한 곳을 중심으로 자름 |
| CutMix | `--cutmix_p 0.2` | 다른 crop의 사각형 영역을 이미지·마스크째 붙임 (λ ~ U(0, 1)) |

- 옵션을 끄면 학습은 이전과 같은 crop 순서와 같은 증강을 씁니다. 원본 샘플러와 crop이
  일치하는 것을 확인했습니다.
- `predict.py`는 체크포인트의 `input_scale`을 읽어 모델마다 자동으로 맞춥니다. 배율이 다른
  모델끼리도 앙상블할 수 있습니다.
- 얇은 조직 샘플링은 효과 폭이 원래 작습니다. 512 crop이 스캔 대부분을 덮어서, 한 스캔
  안에서 가장 좋은 창을 골라도 얇은 Eutectic Si 비율이 5.7% → 6.8%입니다. 그래서 스캔
  자체를 그 비율(1–17%)에 비례해 고르며, crop당 비율은 5.8% → 6.2%(30% 설정)입니다.
- 계산량은 run당 배율 1.5가 약 2.25배, 2가 약 4배입니다. 학습 메모리는 crop 512 / 768 /
  1024에서 6.6 / 14.6 / 25.9 GiB라, `JOBS=3`이면 배율 2가 셋 겹쳐도 96GB GPU에 들어갑니다.

```bash
JOBS=3 ./run_boundary_suite.sh aug_d4 aug_color thin30 cutmix20 scale15 scale2   # 약 3시간 40분
./.conda/bin/python summarize_runs.py --baseline runs/b_fullres_s{0,1,2}           # fullres 대비
```

## 추가 촬영·구조 증강 실험 (2026-10-01)

`train_v2.py`에 다음 이미지·마스크 증강을 각각 켤 수 있게 추가했습니다. 모든 확률의
기본값은 0이므로 기존 체크포인트와 학습 설정은 그대로 유지됩니다. 강도는 KoMaP용
**실험 시작값**이며 논문 설정이나 검증된 최적값이 아닙니다.

| 실험 | 플래그 | 처리 |
|---|---|---|
| 조명 불균일 | `--illumination_p 0.2` | 약 96 원본 픽셀 간격의 매끄러운 ±5% 밝기장; 마스크 유지 |
| 약한 초점 흐림 | `--blur_p 0.15` | 원본 픽셀 기준 Gaussian σ 0.3–0.65; 마스크 유지 |
| Fourier 스타일 | `--fourier_p 0.1` | 다른 **train** crop의 저주파 amplitude 10–20% 혼합; 원영상 phase와 마스크 유지 |
| 문맥 Copy-Paste | `--context_paste_p 0.1` | Si가 든 작은 영역을 영상·4클래스 마스크째 경계 문맥이 비슷한 곳에 복사 |
| 연결 조직 대비 약화 | `--si_attenuate_p 0.05` | GT상 연결된 Eutectic Si의 작은 부분 영상 대비를 약화; 마스크 유지 |

Copy-Paste는 NucleiMix의 문맥 선택에서 착안한 경량 실험이며 diffusion inpainting은
사용하지 않습니다. 마지막 옵션은 CoLeTra에서 착안한 보수적인 변형입니다. 실제로
분리된 Si를 잇는 오검출이 늘 수 있어 후순위로 평가합니다. 영상 열화 증강은 모두
현미경 원본의 회색 강도를 3채널에 동일하게 적용합니다.

GPU가 있는 환경에서 아래 스크립트를 실행하면 각각 B3 fullres 모델 3 seed를 학습하고,
기존 B4와 앙상블해 **8방향 TTA + 원본 tile 768 + Si 입자 투표**로 valid 20장을
평가합니다. `runs/a_ensemble_baseline_valid`에는 기존 fullres B3×3 + B4의
같은 방식 재평가를 저장합니다. 비교 표에는 이미지별 mIoU, Eutectic Si IoU,
개선된 이미지 수, 여러 GT Si 입자를 잘못 연결한 예측 입자 수를 출력합니다.

```bash
./run_augmentation_suite.sh                         # 다섯 단독 증강 × 3 seed + 앙상블 비교
SEEDS=0 MODE=train ./run_augmentation_suite.sh blur  # 빠른 1-seed 확인
MODE=eval ./run_augmentation_suite.sh blur           # 3-seed 학습 후 앙상블 평가
./run_augmentation_suite.sh imaging_combo            # 단독 실험 후 선택적 조합
./.conda/bin/python summarize_augmentation_ensembles.py
```

최종 test 예측은 검증에서 선택한 방법의 세 체크포인트와 기존 B4를 사용합니다.
아래 예시는 `illumination`이 개선된 경우입니다. 검증 전에 test로 증강을 고르지 않습니다.

```bash
./.conda/bin/python predict.py --ckpt \
  runs/a_illumination_s0/best.pth runs/a_illumination_s1/best.pth \
  runs/a_illumination_s2/best.pth runs/s4_micronet_b4/best.pth \
  --split test --tta --d4 --tile 768 --si_vote --save_dir preds/test_illumination
```

CPU만 연결된 환경에서는 전체 학습 실행을 막습니다. 작은 기능 시험에 한해
`ALLOW_CPU=1`로 허용할 수 있습니다.

## 구성

| 파일 | 역할 |
|---|---|
| `src/komap.py` | 팔레트 ↔ class ID, 이미지별 mIoU 지표 |
| `src/tiling.py` | 겹치는 타일 추론. 타일별 argmax가 아니라 **클래스 확률을 결합 후 argmax** |
| `src/sampling.py` | class-aware crop (일반 random crop과 혼합) |
| `src/models.py` | SegFormer / MicroNet U-Net++ / DINO ViT, 원본 해상도 skip(`FullResSkip`) |
| `src/losses.py` | 경계 loss: label relaxation, BLS+JML, 계면 가중 CE, Boundary DoU, Kervadec, ABL |
| `src/postprocess.py` | Si 입자 투표 (Eutectic/Primary Si를 입자 단위로 통일) |
| `src/diagnostics.py` | 오류 분해: 경계 위치 오류(≤1px) vs 상 오분류(>5px) |
| `train_v2.py` | crop 학습 + 원본 이미지 단위 검증 |
| `predict.py` | 채점, 예측 마스크 저장, 다중 체크포인트 ensemble |
| `run_boundary_suite.sh` | 경계 실험 8종 × 3 seed 일괄 실행 |
| `summarize_runs.py` | seed 평균 ± sd, baseline 대비 차이와 Welch p |
| `train.py`, `evaluate.py` | 초기 단순 베이스라인(전체 리사이즈). 비교용으로 유지 |

검증 결과는 `runs/*/val_per_image.json`에 **20장 × 4클래스 IoU 표**로 저장되어,
어떤 변경이 큰 입자만 개선했는지 얇은 상을 개선했는지 확인할 수 있습니다.

## 구현 시 주의한 점

**타일 결합** — 타일마다 argmax해서 붙이면 겹친 영역의 불일치를 해소할 정보가 사라지므로,
클래스별 확률을 raised-cosine 가중치로 누적한 뒤 마지막에 argmax합니다. 이음매가
보이지 않도록 타일 가장자리 기여를 낮췄습니다.

**마스크 보간** — 기하 변환에서 마스크는 항상 nearest-neighbor입니다. bilinear로
보간하면 존재하지 않는 중간 라벨이 생깁니다.

**learning rate 분리** — 사전학습 encoder와 새로 초기화된 head의 lr을 나눴습니다
(`--enc_lr` / `--lr`). ViT backbone은 1e-5 수준을 권장합니다.

**검증은 crop이 아니라 원본 단위** — 학습만 crop이고, 검증·추론은 항상 원본 전체를
복원해 점수를 계산합니다.

**crop 분할 금지** — 공식 70/20/10 분할을 그대로 유지했습니다. crop을 만든 뒤 나누면
같은 원본이 train과 validation에 동시에 들어갑니다.

## 미구현 / 확인 필요

**EoMT 본체** — DINOv3 가중치는 받아서 연결을 마쳤지만(`--arch dinov3`,
`--arch dinov3_vitl16`), 현재 `DinoSegmenter`는 **EoMT가 아니라** ViT + 경량 conv head
입니다. 실제 EoMT(학습 가능한 segmentation query를 ViT 후반 블록에 통과시키는 구조)와
그 loss(query CE + mask BCE/Dice)는 `tue-mps/eomt` 저자 코드를 통합해야 합니다.
위 DINOv3 수치는 그 구조를 평가한 것이 아닙니다.

**class-aware crop의 효과는 이 데이터에서 작습니다.** 512 crop에서는 이미 거의 모든
crop에 4개 상이 모두 포함되어(Primary Si 기준 86.8% → 88.8%), 기본값 0.5로 두었지만
큰 이득은 확인되지 않았습니다. crop을 256 이하로 줄이면 의미가 커질 수 있습니다.

**외부 사전학습 가중치 사용 가능 여부는 대회 규정 확인이 필요합니다.** 현재 결과는
ImageNet + MicroNet 사전학습에 의존합니다. 금지라면 `--pretrain none`으로 재학습해야
하고 점수는 크게 떨어집니다.

**추가 시도 후보** — Lovász-Softmax loss, `--tile 768`, 90장(train+valid) 기반
GroupKFold 교차검증 후 fold ensemble.
