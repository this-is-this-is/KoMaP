# 실험 보고서: 이건

## 1. 환경과 기준
| 항목 | 내용 |
|---|---|
| 작성자 | 이건 |
| 작성일 | 2026-10-03 (후속 실험 2026-10-06 갱신) |
| 조사한 경로 | /home/user/Desktop/komap (runs/, logs_*.log, runs_train*.log, suite_*.log, train.py, train_v2.py, predict.py, eval_ensembles.py, src/, preds/, README.md, EXPERIMENTS.md, ENSEMBLE_EXPERIMENTS.md, AUGMENTATION_EXPERIMENTS.md, MODEL_EXPERIMENTS.md); 작업 메모 ~/.claude/projects/-home-user-Desktop/memory/komap-*.md; git 기록 (커밋 1개: c6ea911, 2026-10-01 17:25, 코드 스냅샷이라 실험 날짜 근거로는 쓰지 않음) |
| 하드웨어 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition 1장, 97,887 MiB (nvidia-smi) |
| 소프트웨어 | Python 3.10.20; PyTorch 2.11.0+cu128 (CUDA 12.8); segmentation_models_pytorch 0.5.0; albumentations 2.0.8; timm 1.0.29; transformers 5.17.0; numpy 2.2.6; OpenCV 5.0.0 (komap/.conda 환경) |
| 데이터 분할 | 공식 70/20/10 그대로 (train/train 70장으로 학습, valid/valid 20장으로 검증, test/test 10장은 예측만) |
| mIoU 계산 방식 | 이미지별 평균 (이미지마다 클래스 IoU를 평균한 뒤 20장 평균); 정답·예측 모두 없는 클래스는 그 이미지 평균에서 제외 (src/komap.py:97; src/komap.py:109) |
| 점수 측정 기본 조건 | 학습 실험: valid 20장, 학습 중 val mIoU 최고 checkpoint, 겹치는 타일(학습 tile 크기, 겹침 0.5), TTA 없음 (train_v2.py:302). 추론·앙상블 실험: 같은 checkpoint에 각 행의 추론 설정. 모든 val 점수는 같은 20장으로 checkpoint를 고른 값이라 낙관적 |
| 실험 수 | 본문 100개(10/3 스냅샷) + 10/5~6 후속 백본·앙상블 실험(아래 최신 갱신) |

### 1.1 2026-10-06 최신 갱신

본문 LG-E01~E100 이후에 다른 모델 계열을 같은 `cs2_aug3`, 입력 2배, 3 seed 조건으로 비교했다.
점수는 validation 20장, 8방향 TTA + tile 768 + Si 입자 투표 기준이다.

| 모델 | 에폭 | 학습 중 best (3-seed 평균) | 3-seed 앙상블 |
|---|---:|---:|---:|
| EfficientNet-B3 + U-Net++ (기준) | 60 | 78.81% | 79.67% |
| HRNet-W18 | 60 | 78.43% | 79.19% |
| **MicroNet SE-ResNeXt-50 32×4d + U-Net++** | 60 | **78.93%** | **80.11%** |
| MicroNet SE-ResNeXt-101 32×4d + U-Net++ | 60 | 78.57% | 79.67% |
| MicroNet Inception-ResNet-v2 + U-Net++ | 60 | 78.30% | 79.10% |
| MicroNet DPN-92 + U-Net++ | 60 | 76.22% | 77.15% |
| **MicroNet SE-ResNeXt-50 32×4d + U-Net++** | 200 | 78.85% | **80.15%** |

최신 상위 5개 앙상블 구성은 다음과 같다.

| 순위 | 구성 | Val mIoU |
|---:|---|---:|
| 1 | B3 200 에폭 6개 + SE-ResNeXt-50 60 에폭 3개 + 200 에폭 3개(동일 가중) | **80.41%** |
| 2 | B3 6개 + SE-ResNeXt-50 3개 + SE-ResNeXt-101 3개(신규 모델 0.5 가중) | 80.36% |
| 3 | 위 구성의 동일 가중 | 80.35% |
| 4 | 1위 구성의 SE-ResNeXt-50 200 에폭 모델만 0.5 가중 | 80.34% |
| 5 | B3 6개 + SE-ResNeXt-50 60 에폭 3개 | 80.30% |

1위의 클래스별 IoU는 Al 94.5 / Al₃Ni 75.8 / Eutectic Si 61.6 / Primary Si 91.6이다.
이전 9모델 기준(80.30%)보다 +0.11pp이지만 paired bootstrap 95% 구간이 -0.04~+0.31pp로
0을 포함하므로, 현재 validation에서 관찰된 최고치이지 우열이 확정된 것은 아니다.
근거는 `runs/with_best_b_seresnext50_cs2_aug3_ep200.json`, `runs/with_best_b_seresnext101_cs2_aug3.json`,
`runs/compare_b_cs2_aug3_vs_b_*cs2_aug3.json`에 남아 있다.

## 2. 실험 요약표
| ID | 실험 이름 | 모델 (백본) | 사전학습 | 입력 처리 | 주요 증강 | Loss | Optimizer (lr) | Epoch | 추론·TTA | 앙상블 | Val mIoU | Test·LB | 상태 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LG-E01 | ResNet-34 U-Net, 전체 리사이즈 | U-Net (ResNet-34) | ImageNet-1k | 전체 리사이즈 512 | 수평·수직 반전, 90도 회전, 밝기·대비, 가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 120 | 전체 리사이즈; TTA 없음 | 없음 | 65.81% | 확인 불가 | 완료 |
| LG-E02 | ResNet-34 U-Net, 원본 해상도 crop | U-Net (ResNet-34) | ImageNet-1k | 원본 해상도 crop 512 | 수평·수직 반전, 90도 회전, 밝기·대비, 가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 120 | 전체 이미지; TTA 없음 | 없음 | 73.12% | 확인 불가 | 완료 |
| LG-E03 | ResNet-50 U-Net++, 원본 해상도 crop | U-Net++ (ResNet-50) | ImageNet-1k | 원본 해상도 crop 512 | 수평·수직 반전, 90도 회전, 밝기·대비, 가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 160 | 전체 이미지; TTA 없음 | 없음 | 75.69% | 확인 불가 | 완료 |
| LG-E04 | SegFormer-B2 | SegFormer (MiT-B2) | ImageNet-1k | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4, 인코더 1e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 72.97% | 확인 불가 | 완료 |
| LG-E05 | MicroNet EfficientNet-B3 U-Net++ | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4, 인코더 1e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.00% | 확인 불가 | 완료 |
| LG-E06 | DINOv2 ViT-B/14 + conv head | ViT + conv head (ViT-B/14) | DINOv2 | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 1e-4, 인코더 1e-5) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 56.06% | 확인 불가 | 완료 |
| LG-E07 | MicroNet EfficientNet-B4 U-Net++ | U-Net++ (EfficientNet-B4) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4, 인코더 1e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 76.91% | 확인 불가 | 완료 |
| LG-E08 | B3 + 반전 TTA | 사용한 모델과 같음 (LG-E05) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05) | 학습 없음 (사용한 모델: LG-E05) | 학습 없음 (사용한 모델: LG-E05) | 학습 없음 (사용한 모델: LG-E05) | 타일 512; TTA 반전 4가지; 후처리 없음 | 없음 | 77.24% | 확인 불가 | 완료 |
| LG-E09 | B3 + B4 앙상블 + 반전 TTA | 사용한 모델과 같음 (LG-E05, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 타일 512; TTA 반전 4가지; 후처리 없음 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | 77.61% | test 예측만 있음 (preds/test_ensemble), 점수 기록 없음 | 완료 |
| LG-E10 | B3 + B4 + SegFormer 앙상블 + 반전 TTA | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E04) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | 타일 512; TTA 반전 4가지; 후처리 없음 | 확률 평균 (동일 가중); LG-E05 + LG-E07 + LG-E04 | 77.52% | 확인 불가 | 완료 |
| LG-E11 | DINOv3 ViT-B/16 + conv head | ViT + conv head (ViT-B/16) | DINOv3 | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 1e-4, 인코더 1e-5) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 63.29% | 확인 불가 | 완료 |
| LG-E12 | DINOv3 ViT-L/16 + conv head | ViT + conv head (ViT-L/16) | DINOv3 | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 1e-4, 인코더 1e-5) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 64.00% | 확인 불가 | 완료 |
| LG-E13 | B3 + B4 + DINOv3-L 앙상블 + 반전 TTA | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E12) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | 타일 512; TTA 반전 4가지; 후처리 없음 | 확률 평균 (동일 가중); LG-E05 + LG-E07 + LG-E12 | 77.31% | 확인 불가 | 완료 |
| LG-E14 | 무작위 배율 증강 0.7–1.4배 | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈, 무작위 배율 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 76.66% | 확인 불가 | 완료 |
| LG-E15 | 무작위 배율 증강 1.0–1.6배 (확대만) | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈, 무작위 배율 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 75.19% | 확인 불가 | 완료 |
| LG-E16 | 추론 타일 크기 512 / 384 / 320 | 사용한 모델과 같음 (LG-E05) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05) | 학습 없음 (사용한 모델: LG-E05) | 학습 없음 (사용한 모델: LG-E05) | 학습 없음 (사용한 모델: LG-E05) | 타일 512 / 384 / 320; TTA 없음; 후처리 없음 | 없음 | 77.00 / 76.75 / 76.51% | 확인 불가 | 완료 |
| LG-E17 | CE + Lovász (단일 run) | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Lovász | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.17% | 확인 불가 | 완료 |
| LG-E18 | CE + Dice + Lovász (3 seed) | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice + Lovász | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.17% ± 0.11 (3 seed) | 확인 불가 | 완료 |
| LG-E19 | CE + Dice 기준 (인코더 lr 3e-4, 3 seed) | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.02% ± 0.13 (3 seed) | 확인 불가 | 완료 |
| LG-E20 | 클래스별 logit bias 보정 | 사용한 모델과 같음 (LG-E05, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 타일 512; TTA 반전 4가지; 후처리 클래스별 logit bias (Al3Ni, Eutectic Si, Primary Si; 격자 탐색, leave-one-image-out) | 확률 평균 (동일 가중); LG-E05 + LG-E07 | 77.57% (leave-one-out; 기준 77.61 대비 −0.04pp) | 확인 불가 | 완료 |
| LG-E21 | Si 입자 투표 | 사용한 모델과 같음 (LG-E05, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 타일 512; TTA 반전 4가지; 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | 77.78% | 확인 불가 | 완료 |
| LG-E22 | 8방향 TTA | 사용한 모델과 같음 (LG-E05, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 타일 512; TTA 8방향(D4); 후처리 없음 / 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | 77.65% (투표 없음) / 78.08% (Si 입자 투표) | 확인 불가 | 완료 |
| LG-E23 | 타일 768 | 사용한 모델과 같음 (LG-E05, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 타일 768; TTA 반전 4가지; 후처리 없음 / 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | 77.71% (투표 없음) / 78.05% (Si 입자 투표) | 확인 불가 | 완료 |
| LG-E24 | 최종 추론 설정 (tile 768 + 8방향 TTA + Si 입자 투표) | 사용한 모델과 같음 (LG-E05, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | 78.09% (투표 없으면 77.76%) | test 예측만 있음 (preds/test_ensemble_d4_vote), 점수 기록 없음 | 완료 |
| LG-E25 | 이미지 전체 추론 | 사용한 모델과 같음 (LG-E05, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 전체 이미지; TTA 반전 4가지 / 8방향(D4); 후처리 없음 / 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | 77.76 / 77.78% (투표 없음, 반전 / 8방향); 78.05 / 78.07% (Si 입자 투표) | 확인 불가 | 완료 |
| LG-E26 | 타일 1024 | 사용한 모델과 같음 (LG-E05, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 학습 없음 (사용한 모델: LG-E05, LG-E07) | 타일 1024; TTA 반전 4가지; 후처리 없음 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | 77.79% | 확인 불가 | 완료 |
| LG-E27 | 이미 학습된 7개 모델 앙상블 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E18, LG-E19) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | 타일 512 / 최종 추론 설정; TTA 반전 4가지 / 8방향(D4); 후처리 없음 / 연결 성분 단위 투표 | 확률 평균 (동일 가중); B3 6개(LG-E05, LG-E18 3개, LG-E19 seed 1·2) + B4(LG-E07) | 77.84% (tile 512, 반전 TTA) / 78.14% (최종 추론 설정) | 확인 불가 | 완료 |
| LG-E28 | 원본 해상도 branch (fullres) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.72% ± 0.12 (3 seed) | 확인 불가 | 완료 |
| LG-E29 | 경계 label relaxation | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 75.05% ± 0.04 (3 seed) | 확인 불가 | 완료 |
| LG-E30 | JML + 경계 label smoothing | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Jaccard Metric Loss | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 76.02% ± 0.08 (3 seed) | 확인 불가 | 완료 |
| LG-E31 | Jaccard Metric Loss (대조군) | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Jaccard Metric Loss | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.30% ± 0.34 (3 seed) | 확인 불가 | 완료 |
| LG-E32 | Al3Ni–Si 계면 가중 CE | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | 가중 CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.07% ± 0.30 (3 seed) | 확인 불가 | 완료 |
| LG-E33 | Boundary DoU | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Boundary loss | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 76.92% ± 0.26 (3 seed) | 확인 불가 | 완료 |
| LG-E34 | Kervadec boundary loss | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice + Boundary loss | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 76.25% ± 0.42 (3 seed) | 확인 불가 | 완료 |
| LG-E35 | Active Boundary Loss | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Lovász + Boundary loss | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.13% ± 0.18 (3 seed) | 확인 불가 | 완료 |
| LG-E36 | CE + Lovász (ABL 대조군) | U-Net++ (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Lovász | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.10% ± 0.12 (3 seed) | 확인 불가 | 완료 |
| LG-E37 | 기준 B3 3-seed 앙상블 | 사용한 모델과 같음 (LG-E19) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E19) | 학습 없음 (사용한 모델: LG-E19) | 학습 없음 (사용한 모델: LG-E19) | 학습 없음 (사용한 모델: LG-E19) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E19 3개 | 78.04% | 확인 불가 | 완료 |
| LG-E38 | fullres 3-seed 앙상블 | 사용한 모델과 같음 (LG-E28) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E28) | 학습 없음 (사용한 모델: LG-E28) | 학습 없음 (사용한 모델: LG-E28) | 학습 없음 (사용한 모델: LG-E28) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E28 3개 | 78.35% | 확인 불가 | 완료 |
| LG-E39 | fullres 3-seed + B4 앙상블 | 사용한 모델과 같음 (LG-E28, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E28, LG-E07) | 학습 없음 (사용한 모델: LG-E28, LG-E07) | 학습 없음 (사용한 모델: LG-E28, LG-E07) | 학습 없음 (사용한 모델: LG-E28, LG-E07) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E28 3개 + LG-E07 | 78.47% | 확인 불가 | 완료 |
| LG-E40 | fullres + EfficientNet-B4 | U-Net++ + 원본 해상도 branch (EfficientNet-B4) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4, 인코더 1e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.28% ± 0.50 (3 seed) | 확인 불가 | 완료 |
| LG-E41 | fullres branch 채널 64 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.52% ± 0.06 (3 seed) | 확인 불가 | 완료 |
| LG-E42 | fullres + CE + Lovász | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Lovász | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.43% ± 0.07 (3 seed) | 확인 불가 | 완료 |
| LG-E43 | fullres B4 3-seed 앙상블 | 사용한 모델과 같음 (LG-E40) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E40) | 학습 없음 (사용한 모델: LG-E40) | 학습 없음 (사용한 모델: LG-E40) | 학습 없음 (사용한 모델: LG-E40) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E40 3개 | 78.31% | 확인 불가 | 완료 |
| LG-E44 | fullres B3 ×3 + fullres B4 ×3 앙상블 | 사용한 모델과 같음 (LG-E28, LG-E40) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E40) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E28 3개 + LG-E40 3개 | 78.45% | 확인 불가 | 완료 |
| LG-E45 | fullres B3 ×3 + fullres B4 (seed 2) 앙상블 | 사용한 모델과 같음 (LG-E28, LG-E40) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E40) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E28 3개 + LG-E40 seed 2 | 78.47% | 확인 불가 | 완료 |
| LG-E46 | fullres(Lovász) ×3 + fullres B4 ×3 앙상블 | 사용한 모델과 같음 (LG-E42, LG-E40) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E42, LG-E40) | 학습 없음 (사용한 모델: LG-E42, LG-E40) | 학습 없음 (사용한 모델: LG-E42, LG-E40) | 학습 없음 (사용한 모델: LG-E42, LG-E40) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E42 3개 + LG-E40 3개 | 78.44% | 확인 불가 | 완료 |
| LG-E47 | fullres(채널 64) ×3 + fullres B4 ×3 앙상블 | 사용한 모델과 같음 (LG-E41, LG-E40) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E41, LG-E40) | 학습 없음 (사용한 모델: LG-E41, LG-E40) | 학습 없음 (사용한 모델: LG-E41, LG-E40) | 학습 없음 (사용한 모델: LG-E41, LG-E40) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E41 3개 + LG-E40 3개 | 78.43% | 확인 불가 | 완료 |
| LG-E48 | fullres 계열 12개 앙상블 | 사용한 모델과 같음 (LG-E28, LG-E41, LG-E42, LG-E40) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); B3 9개(LG-E28, LG-E41, LG-E42) + LG-E40 3개 | 78.46% | 확인 불가 | 완료 |
| LG-E49 | D4 균등 선택 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전 (D4 균등), 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.21% ± 0.20 (3 seed) | 확인 불가 | 완료 |
| LG-E50 | 밝기 증강 약하게 (aug_color) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.21% ± 0.03 (3 seed) | 확인 불가 | 완료 |
| LG-E51 | 입력 1.5배 확대 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 배율 확대 ×1.5 (입력 768px) | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.06% ± 0.12 (3 seed) | 확인 불가 | 완료 |
| LG-E52 | 입력 2배 확대 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 배율 확대 ×2 (입력 1024px) | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.21% ± 0.11 (3 seed) | 확인 불가 | 완료 |
| LG-E53 | 얇은 조직 중심 샘플링 30% | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈, 오버샘플링 (얇은 조직) | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.32% ± 0.27 (3 seed) | 확인 불가 | 완료 |
| LG-E54 | CutMix 20% | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마·가우시안 노이즈, CutMix | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.44% ± 0.51 (3 seed) | 확인 불가 | 완료 |
| LG-E55 | aug_color 3-seed 앙상블 | 사용한 모델과 같음 (LG-E50) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E50) | 학습 없음 (사용한 모델: LG-E50) | 학습 없음 (사용한 모델: LG-E50) | 학습 없음 (사용한 모델: LG-E50) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E50 3개 | 78.91% | 확인 불가 | 완료 |
| LG-E56 | aug_color 3-seed + B4 앙상블 | 사용한 모델과 같음 (LG-E50, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E50, LG-E07) | 학습 없음 (사용한 모델: LG-E50, LG-E07) | 학습 없음 (사용한 모델: LG-E50, LG-E07) | 학습 없음 (사용한 모델: LG-E50, LG-E07) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E50 3개 + LG-E07 | 78.84% | 확인 불가 | 완료 |
| LG-E57 | scale2 3-seed 앙상블 | 사용한 모델과 같음 (LG-E52) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E52) | 학습 없음 (사용한 모델: LG-E52) | 학습 없음 (사용한 모델: LG-E52) | 학습 없음 (사용한 모델: LG-E52) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E52 3개 | 78.99% | 확인 불가 | 완료 |
| LG-E58 | scale2 3-seed + B4 앙상블 | 사용한 모델과 같음 (LG-E52, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E52, LG-E07) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E52 3개 + LG-E07 | 78.91% | 확인 불가 | 완료 |
| LG-E59 | aug_color ×3 + scale2 ×3 앙상블 | 사용한 모델과 같음 (LG-E50, LG-E52) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E50, LG-E52) | 학습 없음 (사용한 모델: LG-E50, LG-E52) | 학습 없음 (사용한 모델: LG-E50, LG-E52) | 학습 없음 (사용한 모델: LG-E50, LG-E52) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E50 3개 + LG-E52 3개 | 79.07% | 확인 불가 | 완료 |
| LG-E60 | aug_color ×3 + scale2 ×3 + B4 앙상블 | 사용한 모델과 같음 (LG-E50, LG-E52, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E50 3개 + LG-E52 3개 + LG-E07 | 79.08% | 확인 불가 | 완료 |
| LG-E61 | fullres ×3 + aug_color ×3 + scale2 ×3 + B4 앙상블 | 사용한 모델과 같음 (LG-E28, LG-E50, LG-E52, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); 10개 | 78.83% | 확인 불가 | 완료 |
| LG-E62 | aug_color + 입력 2배 (color_scale2) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 배율 확대 ×2 (입력 1024px) | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.74% ± 0.20 (3 seed) | 확인 불가 | 완료 |
| LG-E63 | color_scale2 3-seed 앙상블 | 사용한 모델과 같음 (LG-E62) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E62) | 학습 없음 (사용한 모델: LG-E62) | 학습 없음 (사용한 모델: LG-E62) | 학습 없음 (사용한 모델: LG-E62) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E62 3개 | **79.61%** | 확인 불가 | 완료 |
| LG-E64 | color_scale2 3-seed + B4 앙상블 | 사용한 모델과 같음 (LG-E62, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E62, LG-E07) | 학습 없음 (사용한 모델: LG-E62, LG-E07) | 학습 없음 (사용한 모델: LG-E62, LG-E07) | 학습 없음 (사용한 모델: LG-E62, LG-E07) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E62 3개 + LG-E07 | 79.47% | 확인 불가 | 완료 |
| LG-E65 | color_scale2 ×3 + aug_color ×3 + scale2 ×3 앙상블 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); 9개 | 79.37% | 확인 불가 | 완료 |
| LG-E66 | color_scale2 ×3 + aug_color ×3 + scale2 ×3 + B4 앙상블 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52, LG-E07) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); 10개 | 79.32% | 확인 불가 | 완료 |
| LG-E67 | aug_color 120 에폭 (1차) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 120 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.40% ± 0.15 (3 seed) | 확인 불가 | 완료 |
| LG-E68 | aug_color 200 에폭 (1차) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 200 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.65% ± 0.07 (3 seed) | 확인 불가 | 완료 |
| LG-E69 | 1차 에폭 실험 앙상블 (120 / 200 에폭) | 사용한 모델과 같음 (LG-E67, LG-E68) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E67, LG-E68) | 학습 없음 (사용한 모델: LG-E67, LG-E68) | 학습 없음 (사용한 모델: LG-E67, LG-E68) | 학습 없음 (사용한 모델: LG-E67, LG-E68) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E67 3개 / LG-E68 3개 | 79.28% (120 에폭) / 79.52% (200 에폭) | 확인 불가 | 완료 |
| LG-E70 | aug_color 60 에폭 재실행 (2차, 이미지별 기록) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.09% ± 0.17 (3 seed) | 확인 불가 | 완료 |
| LG-E71 | aug_color 120 에폭 (2차) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 120 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.51% ± 0.23 (3 seed) | 확인 불가 | 완료 |
| LG-E72 | aug_color 200 에폭 (2차) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 200 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.58% ± 0.18 (3 seed) | 확인 불가 | 완료 |
| LG-E73 | 2차 에폭 실험 앙상블 (60 / 120 / 200 에폭) | 사용한 모델과 같음 (LG-E70, LG-E71, LG-E72) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); 에폭마다 3개 | 78.82% / 79.39% / 79.46% | 확인 불가 | 완료 |
| LG-E74 | 조명 필드 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10%, 밝기 (조명 필드) | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.21% ± 0.19 (3 seed) | 확인 불가 | 완료 |
| LG-E75 | 초점 흐림 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10%, 가우시안 블러 | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.93% ± 0.07 (3 seed) | 확인 불가 | 완료 |
| LG-E76 | Fourier 혼합 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10%, 색 변환 (Fourier) | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.22% ± 0.07 (3 seed) | 확인 불가 | 완료 |
| LG-E77 | 스캔별 명암 표준화 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 명암 표준화 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 77.91% ± 0.12 (3 seed) | 확인 불가 | 완료 |
| LG-E78 | 작은 스캔만 2배 확대 (6 seed) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 작은 스캔 ×2 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10% | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.25% ± 0.35 (6 seed) | 확인 불가 | 완료 |
| LG-E79 | Primary Si 입자 이동 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10%, Copy-Paste (Primary Si) | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.39% ± 0.06 (3 seed) | 확인 불가 | 완료 |
| LG-E80 | 조명 필드 3-seed 앙상블 | 사용한 모델과 같음 (LG-E74) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E74) | 학습 없음 (사용한 모델: LG-E74) | 학습 없음 (사용한 모델: LG-E74) | 학습 없음 (사용한 모델: LG-E74) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E74 seed 0–2 | 79.12% | 확인 불가 | 완료 |
| LG-E81 | 초점 흐림 3-seed 앙상블 | 사용한 모델과 같음 (LG-E75) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E75) | 학습 없음 (사용한 모델: LG-E75) | 학습 없음 (사용한 모델: LG-E75) | 학습 없음 (사용한 모델: LG-E75) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E75 seed 0–2 | 78.94% | 확인 불가 | 완료 |
| LG-E82 | Fourier 혼합 3-seed 앙상블 | 사용한 모델과 같음 (LG-E76) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E76) | 학습 없음 (사용한 모델: LG-E76) | 학습 없음 (사용한 모델: LG-E76) | 학습 없음 (사용한 모델: LG-E76) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E76 seed 0–2 | 79.10% | 확인 불가 | 완료 |
| LG-E83 | 명암 표준화 3-seed 앙상블 | 사용한 모델과 같음 (LG-E77) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E77) | 학습 없음 (사용한 모델: LG-E77) | 학습 없음 (사용한 모델: LG-E77) | 학습 없음 (사용한 모델: LG-E77) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E77 seed 0–2 | 78.81% | 확인 불가 | 완료 |
| LG-E84 | 작은 스캔 2배 (seed 0–2) 3-seed 앙상블 | 사용한 모델과 같음 (LG-E78) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E78) | 학습 없음 (사용한 모델: LG-E78) | 학습 없음 (사용한 모델: LG-E78) | 학습 없음 (사용한 모델: LG-E78) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E78 seed 0–2 | 79.24% | 확인 불가 | 완료 |
| LG-E85 | Primary Si 입자 이동 3-seed 앙상블 | 사용한 모델과 같음 (LG-E79) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E79) | 학습 없음 (사용한 모델: LG-E79) | 학습 없음 (사용한 모델: LG-E79) | 학습 없음 (사용한 모델: LG-E79) | 타일 768; TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E79 seed 0–2 | 79.14% | 확인 불가 | 완료 |
| LG-E86 | 작은 스캔 2배 6-seed 앙상블 대 기준 6-run 앙상블 | 사용한 모델과 같음 (LG-E78, LG-E50, LG-E70) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E78 6개 / 기준 LG-E50 3개 + LG-E70 3개 | 79.42% 대 78.83% (seed 3–5만의 3-seed 앙상블은 79.34%) | 확인 불가 | 완료 |
| LG-E87 | color_scale2 + 증강 3종 + 작은 스캔 4배 (cs2_plus) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 배율 확대 ×2 (입력 1024px), 작은 스캔 ×4 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10%, Copy-Paste (Primary Si), 밝기 (조명 필드), 색 변환 (Fourier) | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.87% ± 0.12 (3 seed) | 확인 불가 | 완료 |
| LG-E88 | color_scale2 + 증강 3종 (cs2_aug3) | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 배율 확대 ×2 (입력 1024px) | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10%, Copy-Paste (Primary Si), 밝기 (조명 필드), 색 변환 (Fourier) | CE + Dice | AdamW (lr 3e-4) | 60 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.81% ± 0.05 (3 seed) | 확인 불가 | 완료 |
| LG-E89 | cs2_plus 3-seed 앙상블 | 사용한 모델과 같음 (LG-E87) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E87) | 학습 없음 (사용한 모델: LG-E87) | 학습 없음 (사용한 모델: LG-E87) | 학습 없음 (사용한 모델: LG-E87) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E87 3개 | 79.54% | 확인 불가 | 완료 |
| LG-E90 | cs2_plus ×3 + color_scale2 ×3 앙상블 | 사용한 모델과 같음 (LG-E87, LG-E62) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E87, LG-E62) | 학습 없음 (사용한 모델: LG-E87, LG-E62) | 학습 없음 (사용한 모델: LG-E87, LG-E62) | 학습 없음 (사용한 모델: LG-E87, LG-E62) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E87 3개 + LG-E62 3개 | 79.86% | 확인 불가 | 완료 |
| LG-E91 | cs2_aug3 3-seed 앙상블 | 사용한 모델과 같음 (LG-E88) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E88) | 학습 없음 (사용한 모델: LG-E88) | 학습 없음 (사용한 모델: LG-E88) | 학습 없음 (사용한 모델: LG-E88) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E88 3개 | 79.67% | 확인 불가 | 완료 |
| LG-E92 | cs2_aug3 ×3 + color_scale2 ×3 앙상블 | 사용한 모델과 같음 (LG-E88, LG-E62) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E88, LG-E62) | 학습 없음 (사용한 모델: LG-E88, LG-E62) | 학습 없음 (사용한 모델: LG-E88, LG-E62) | 학습 없음 (사용한 모델: LG-E88, LG-E62) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E88 3개 + LG-E62 3개 | 79.70% | 확인 불가 | 완료 |
| LG-E93 | cs2_aug3 ×3 + cs2_plus ×3 앙상블 | 사용한 모델과 같음 (LG-E88, LG-E87) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E88, LG-E87) | 학습 없음 (사용한 모델: LG-E88, LG-E87) | 학습 없음 (사용한 모델: LG-E88, LG-E87) | 학습 없음 (사용한 모델: LG-E88, LG-E87) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E88 3개 + LG-E87 3개 | 79.91% | 확인 불가 | 완료 |
| LG-E94 | cs2_aug3 ×3 + cs2_plus ×3 + color_scale2 ×3 앙상블 | 사용한 모델과 같음 (LG-E88, LG-E87, LG-E62) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); 9개 | 79.90% | 확인 불가 | 완료 |
| LG-E95 | cs2_aug3 200 에폭 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 배율 확대 ×2 (입력 1024px) | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10%, Copy-Paste (Primary Si), 밝기 (조명 필드), 색 변환 (Fourier) | CE + Dice | AdamW (lr 3e-4) | 200 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 78.99% ± 0.11 (3 seed) | 확인 불가 | 완료 |
| LG-E96 | cs2_plus 200 에폭 | U-Net++ + 원본 해상도 branch (EfficientNet-B3) | MicroNet(현미경) | 원본 해상도 crop 512, 배율 확대 ×2 (입력 1024px), 작은 스캔 ×4 | 수평·수직 반전·90도 회전, 밝기·대비·감마 중 하나 ±10%, Copy-Paste (Primary Si), 밝기 (조명 필드), 색 변환 (Fourier) | CE + Dice | AdamW (lr 3e-4) | 200 | 겹치는 타일 512 (원본 기준), 겹침 0.5; TTA 없음 | 없음 | 79.19% ± 0.18 (3 seed) | 확인 불가 | 완료 |
| LG-E97 | cs2_aug3 200 에폭 3-seed 앙상블 | 사용한 모델과 같음 (LG-E95) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E95) | 학습 없음 (사용한 모델: LG-E95) | 학습 없음 (사용한 모델: LG-E95) | 학습 없음 (사용한 모델: LG-E95) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E95 3개 | 79.86% | 확인 불가 | 완료 |
| LG-E98 | cs2_plus 200 에폭 3-seed 앙상블 | 사용한 모델과 같음 (LG-E96) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E96) | 학습 없음 (사용한 모델: LG-E96) | 학습 없음 (사용한 모델: LG-E96) | 학습 없음 (사용한 모델: LG-E96) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E96 3개 | 79.95% | 확인 불가 | 완료 |
| LG-E99 | cs2_aug3 200 에폭 ×3 + cs2_plus 200 에폭 ×3 앙상블 | 사용한 모델과 같음 (LG-E95, LG-E96) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E95, LG-E96) | 학습 없음 (사용한 모델: LG-E95, LG-E96) | 학습 없음 (사용한 모델: LG-E95, LG-E96) | 학습 없음 (사용한 모델: LG-E95, LG-E96) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); LG-E95 3개 + LG-E96 3개 | 80.14% | 확인 불가 | 완료 |
| LG-E100 | 200 에폭 6개 + 60 에폭 6개 앙상블 | 사용한 모델과 같음 (LG-E95, LG-E96, LG-E88, LG-E87) | 사용한 모델과 같음 | 사용한 모델과 같음 | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | 타일 768 (학습 배율로 확대); TTA 8방향(D4); 후처리 연결 성분 단위 투표 | 확률 평균 (동일 가중); 12개 | 80.09% | 확인 불가 | 완료 |

## 3. 2026-10-03 당시 최고 성능 구성
| 항목 | 내용 |
|---|---|
| 실험 ID | LG-E99 (cs2_aug3 200 에폭 ×3 + cs2_plus 200 에폭 ×3 앙상블; 구성 모델 LG-E95 3개 + LG-E96 3개) |
| 점수 | Val mIoU 80.14% (valid 20장; 각 run의 학습 중 val 최고 checkpoint; tile 768 + 8방향 TTA + Si 입자 투표; 6개 확률 평균) (EXPERIMENTS.md:248); 클래스별 Al 94.5 / Al3Ni 75.5 / Eutectic Si 60.9 / Primary Si 91.4; Test·LB 확인 불가 |
| 구성 요약 | 모델: U-Net++ (EfficientNet-B3, MicroNet(현미경) 사전학습) + 원본 해상도 branch 32채널; 증강: 수평·수직 반전·90도 회전 (p=0.5씩), 밝기·대비·감마 중 하나 ±10% (p=0.5), Copy-Paste (Primary Si 입자 이동, p=0.25), 밝기 (조명 필드, p=0.2), 색 변환 (Fourier, p=0.1); 학습: 입력 2배 확대 (LG-E96만 높이 800px 이하 스캔 4배), 클래스 균형 crop, CE + Dice (0.5/0.5), AdamW lr 3e-4, wd 1e-4, Cosine, 200 에폭, 배치 8, AMP, seed 0·1·2; 추론: 겹치는 타일 768 (겹침 0.5) + 8방향(D4) TTA + 연결 성분 단위 투표; 앙상블: 6개 확률 평균 |
| 재현 명령 | 학습: `nohup ./run_final_ep200.sh > suite_final_ep200.log 2>&1 &` (run_final_ep200.sh; 인자는 runs/b_cs2_plus_ep200_s0/history.json → args); 검증 앙상블 점수: `./.conda/bin/python eval_ensembles.py` (eval_ensembles.py, 조합 'cs2_aug3 ep200 x3 + cs2_plus ep200 x3'); test 예측: `./.conda/bin/python predict.py --ckpt runs/b_cs2_aug3_ep200_s{0,1,2}/best.pth runs/b_cs2_plus_ep200_s{0,1,2}/best.pth --split test --tta --d4 --tile 768 --si_vote --save_dir preds/test_final_ep200` (EXPERIMENTS.md:286; 아직 실행 안 함) |

## 4. 실험 상세

### LG-E01 ResNet-34 U-Net, 전체 리사이즈
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs_train.log |
| 날짜 | 2026-09-17 | runs_train.log 파일 수정 시각 |
| 목적 | train.py 기본 파이프라인(이미지 전체를 512로 축소)의 기준 점수 확인 | - |
| 비교 기준 | 없음 | - |
| 모델 구조 | U-Net (smp.Unet) | runs/unet_r34/best.pth → args.arch; src/komap.py:115 |
| 백본 | ResNet-34 (resnet34) | runs/unet_r34/best.pth → args.encoder |
| 사전학습 가중치 | ImageNet-1k (smp encoder_weights='imagenet') | src/komap.py:115 |
| 구조 변경 | 없음 | - |
| 입력 처리 | 전체 리사이즈 + 무작위 crop (RandomResizedCrop 512, 면적 0.35–1.0, 비율 0.85–1.18) | train.py:55 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train.py:55 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.2, p=0.5); 가우시안 노이즈 (GaussNoise, p=0.2) | train.py:55 |
| 샘플링 | 무작위 crop (RandomResizedCrop, 스캔당 에폭마다 1회) | train.py:55 |
| Loss | CE + Dice (1:1 합) | train.py:191 |
| Optimizer | AdamW; lr 3e-4 (전체 동일); weight decay 1e-4; betas 기본값 | runs/unet_r34/best.pth → args.lr; train.py:173 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=120); warmup 없음 | train.py:174 |
| Batch size | 8 | runs/unet_r34/best.pth → args.batch_size |
| Epoch·Step | 120 에폭; 에폭당 8 step (70장 ÷ 배치 8, drop_last) | runs/unet_r34/best.pth → args.epochs |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train.py:175 |
| Seed | 0 (1회) | runs/unet_r34/best.pth → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train.py:157 |
| 검증·checkpoint 선택 | valid 20장; 매 에폭; val mIoU 최고 checkpoint 저장 | train.py:200 |
| 추론 방식 | 전체 리사이즈 (512×512) 후 원본 크기로 보간해 채점 | train.py:78 |
| TTA | 없음 | - |
| 후처리 | 없음 | - |
| 앙상블 | 없음 | - |
| Val mIoU | 65.81% (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 전체 리사이즈 (512×512) 후 원본 크기로 보간해 채점, TTA 없음 | runs_train.log:162 |
| 클래스별 IoU | 84.0 / 58.5 / 39.0 / 83.5 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs_train.log (마지막 '<- best' 에폭의 per-class IoU); train.py:130 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 9분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | runs_train.log |
| 결론 | 원본 해상도 정보를 잃음. 정답 마스크를 512로 축소·복원만 해도 상한이 66.9%라 이 방식은 한계가 있음 | - |

### LG-E02 ResNet-34 U-Net, 원본 해상도 crop
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs_train_native.log |
| 날짜 | 2026-09-17 | runs_train_native.log 파일 수정 시각 |
| 목적 | 축소 대신 원본 해상도 crop으로 학습·전체 이미지로 검증 | - |
| 비교 기준 | LG-E01 | - |
| 모델 구조 | U-Net (smp.Unet) | runs/unet_r34_native/best.pth → args.arch; src/komap.py:115 |
| 백본 | ResNet-34 (resnet34) | runs/unet_r34_native/best.pth → args.encoder |
| 사전학습 가중치 | ImageNet-1k (smp encoder_weights='imagenet') | src/komap.py:115 |
| 구조 변경 | 없음 | - |
| 입력 처리 | 원본 해상도 crop (RandomCrop 512; 작은 스캔은 반사 패딩) | train.py:45 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train.py:45 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.2, p=0.5); 가우시안 노이즈 (GaussNoise, p=0.2) | train.py:45 |
| 샘플링 | 무작위 crop (스캔당 에폭마다 1회) | train.py:45 |
| Loss | CE + Dice (1:1 합) | train.py:191 |
| Optimizer | AdamW; lr 3e-4 (전체 동일); weight decay 1e-4; betas 기본값 | runs/unet_r34_native/best.pth → args.lr; train.py:173 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=120); warmup 없음 | train.py:174 |
| Batch size | 8 | runs/unet_r34_native/best.pth → args.batch_size |
| Epoch·Step | 120 에폭; 에폭당 8 step (70장 ÷ 배치 8, drop_last) | runs/unet_r34_native/best.pth → args.epochs |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train.py:175 |
| Seed | 0 (1회) | runs/unet_r34_native/best.pth → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train.py:157 |
| 검증·checkpoint 선택 | valid 20장; 매 에폭; val mIoU 최고 checkpoint 저장 | train.py:200 |
| 추론 방식 | 전체 이미지 (원본 해상도, 32 배수 반사 패딩) | train.py:72 |
| TTA | 없음 | - |
| 후처리 | 없음 | - |
| 앙상블 | 없음 | - |
| Val mIoU | 73.12% (단일 run); LG-E01 대비 +7.31pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 전체 이미지 (원본 해상도, 32 배수 반사 패딩), TTA 없음 | runs_train_native.log:160 |
| 클래스별 IoU | 91.5 / 67.1 / 51.1 / 82.8 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs_train_native.log (마지막 '<- best' 에폭의 per-class IoU); train.py:130 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 9분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | runs_train_native.log |
| 결론 | 원본 해상도 학습이 결정적 | - |

### LG-E03 ResNet-50 U-Net++, 원본 해상도 crop
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs_train_unetpp.log |
| 날짜 | 2026-09-17 | runs_train_unetpp.log 파일 수정 시각 |
| 목적 | 디코더(U-Net++)와 백본(ResNet-50)을 키움 | - |
| 비교 기준 | LG-E02 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | runs/unetpp_r50_native/best.pth → args.arch; src/komap.py:115 |
| 백본 | ResNet-50 (resnet50) | runs/unetpp_r50_native/best.pth → args.encoder |
| 사전학습 가중치 | ImageNet-1k (smp encoder_weights='imagenet') | src/komap.py:115 |
| 구조 변경 | 없음 | - |
| 입력 처리 | 원본 해상도 crop (RandomCrop 512; 작은 스캔은 반사 패딩) | train.py:45 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train.py:45 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.2, p=0.5); 가우시안 노이즈 (GaussNoise, p=0.2) | train.py:45 |
| 샘플링 | 무작위 crop (스캔당 에폭마다 1회) | train.py:45 |
| Loss | CE + Dice (1:1 합) | train.py:191 |
| Optimizer | AdamW; lr 3e-4 (전체 동일); weight decay 1e-4; betas 기본값 | runs/unetpp_r50_native/best.pth → args.lr; train.py:173 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=160); warmup 없음 | train.py:174 |
| Batch size | 8 | runs/unetpp_r50_native/best.pth → args.batch_size |
| Epoch·Step | 160 에폭; 에폭당 8 step (70장 ÷ 배치 8, drop_last) | runs/unetpp_r50_native/best.pth → args.epochs |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train.py:175 |
| Seed | 0 (1회) | runs/unetpp_r50_native/best.pth → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train.py:157 |
| 검증·checkpoint 선택 | valid 20장; 매 에폭; val mIoU 최고 checkpoint 저장 | train.py:200 |
| 추론 방식 | 전체 이미지 (원본 해상도, 32 배수 반사 패딩) | train.py:72 |
| TTA | 없음 | - |
| 후처리 | 없음 | - |
| 앙상블 | 없음 | - |
| Val mIoU | 75.69% (단일 run); LG-E02 대비 +2.57pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 전체 이미지 (원본 해상도, 32 배수 반사 패딩), TTA 없음 | runs_train_unetpp.log:198 |
| 클래스별 IoU | 92.5 / 69.5 / 54.1 / 88.4 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs_train_unetpp.log (마지막 '<- best' 에폭의 per-class IoU); train.py:130 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 14분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | runs_train_unetpp.log |
| 결론 | 참고 기준(약 75%)을 처음 넘음 | - |

### LG-E04 SegFormer-B2
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s1_segformer_b2/history.json → history |
| 날짜 | 2026-09-17 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 최신 Transformer 분할 모델과 비교 | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | SegFormer (SegformerForSemanticSegmentation) | src/models.py:208 |
| 백본 | MiT-B2 (nvidia/mit-b2) | src/models.py:208 |
| 사전학습 가중치 | ImageNet-1k (nvidia/mit-b2 HF 체크포인트; args.pretrain=micronet은 SegFormer에 쓰이지 않음) | src/models.py:208 |
| 구조 변경 | 출력 1/4 해상도를 bilinear로 원본 크기에 맞춤 | src/models.py:76 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/s1_segformer_b2/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s1_segformer_b2/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | args에 loss 키 없음 → 코드 기본값 ce_dice (train_v2.py:333); train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 1e-4; weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s1_segformer_b2/history.json → args.lr; runs/s1_segformer_b2/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s1_segformer_b2/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s1_segformer_b2/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0 (1회) | runs/s1_segformer_b2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s1_segformer_b2/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s1_segformer_b2/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 72.97% (단일 run); LG-E05 대비 −4.03pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s1_segformer_b2/history.json → best_miou |
| 클래스별 IoU | 88.9 / 66.8 / 49.7 / 88.7 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s1_segformer_b2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 9분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s1_segformer.log |
| 결론 | MicroNet U-Net++보다 낮음. 1/4 해상도 출력이 얇은 상에 불리 | - |

### LG-E05 MicroNet EfficientNet-B3 U-Net++
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s2_micronet_unetpp/history.json → history |
| 날짜 | 2026-09-17 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 현미경 사전학습(MicroNet) 인코더 적용 | - |
| 비교 기준 | LG-E03 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/s2_micronet_unetpp/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/s2_micronet_unetpp/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | args에 fullres_skip 없음 (기능 도입 전) |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/s2_micronet_unetpp/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s2_micronet_unetpp/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | args에 loss 키 없음 → 코드 기본값 ce_dice (train_v2.py:333); train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 1e-4; weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s2_micronet_unetpp/history.json → args.lr; runs/s2_micronet_unetpp/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s2_micronet_unetpp/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s2_micronet_unetpp/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0 (1회) | runs/s2_micronet_unetpp/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s2_micronet_unetpp/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s2_micronet_unetpp/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.00% (단일 run); LG-E03 대비 +1.31pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s2_micronet_unetpp/history.json → best_miou |
| 클래스별 IoU | 93.3 / 71.9 / 55.8 / 88.8 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s2_micronet_unetpp/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 10분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s2_micronet.log |
| 결론 | 초기 최고. 이후 기준 모델. 인코더 lr이 1e-4라 LG-E19(3e-4)와 조건이 다름. 사전학습·백본·학습 스크립트가 함께 바뀐 비교 | - |

### LG-E06 DINOv2 ViT-B/14 + conv head
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s3_dinov2/history.json → history |
| 날짜 | 2026-09-17 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 자기지도 사전학습 ViT 백본 적용 | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | ViT + 경량 conv head (DinoSegmenter; EoMT 아님) | src/models.py:115 |
| 백본 | ViT-B/14 (facebook/dinov2-base) | src/models.py:212 |
| 사전학습 가중치 | DINOv2 | src/models.py:212 |
| 구조 변경 | patch token → conv head(2배 업샘플) → bilinear | src/models.py:115 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/s3_dinov2/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s3_dinov2/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | args에 loss 키 없음 → 코드 기본값 ce_dice (train_v2.py:333); train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 1e-4 / 인코더 1e-5; weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s3_dinov2/history.json → args.lr; runs/s3_dinov2/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s3_dinov2/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s3_dinov2/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0 (1회) | runs/s3_dinov2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s3_dinov2/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s3_dinov2/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 56.06% (단일 run); LG-E05 대비 −20.93pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s3_dinov2/history.json → best_miou |
| 클래스별 IoU | 76.4 / 46.3 / 22.5 / 81.4 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s3_dinov2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 11분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s3_dinov2.log |
| 결론 | patch 14의 거친 특징으로 얇은 상을 못 맞춤 | - |

### LG-E07 MicroNet EfficientNet-B4 U-Net++
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s4_micronet_b4/history.json → history |
| 날짜 | 2026-09-17 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 인코더 크기 확대(B3 → B4) | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B4 (efficientnet-b4) | runs/s4_micronet_b4/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/s4_micronet_b4/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | args에 fullres_skip 없음 (기능 도입 전) |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/s4_micronet_b4/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s4_micronet_b4/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | args에 loss 키 없음 → 코드 기본값 ce_dice (train_v2.py:333); train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 1e-4; weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s4_micronet_b4/history.json → args.lr; runs/s4_micronet_b4/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s4_micronet_b4/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s4_micronet_b4/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 1 (1회) | runs/s4_micronet_b4/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s4_micronet_b4/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s4_micronet_b4/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 76.91% (단일 run); LG-E05 대비 −0.09pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s4_micronet_b4/history.json → best_miou |
| 클래스별 IoU | 93.4 / 72.1 / 55.3 / 88.5 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s4_micronet_b4/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 13분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s4_b4.log |
| 결론 | B3와 차이 없음(단일 run) | - |

### LG-E08 B3 + 반전 TTA
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | README.md:25 |
| 날짜 | 2026-09-17 (추정: README 결과표 6단계, 날짜 기록 없음) | - |
| 목적 | TTA 효과 확인 | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05) | - |
| 백본 | 사용한 모델과 같음 (LG-E05) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05) | - |
| 추론 방식 | 겹치는 타일 (tile 512, 겹침 0.5, cosine 가중); 원본 해상도 | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) | src/tiling.py:56 |
| 후처리 | 없음 | - |
| 앙상블 | 없음 | - |
| Val mIoU | 77.24%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | README.md:25 |
| 클래스별 IoU | 93.4 / 72.2 / 56.3 / 88.9 (Al / Al3Ni / Eutectic Si / Primary Si) | README.md:25 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05) | - |
| 결론 | +0.24pp | - |

### LG-E09 B3 + B4 앙상블 + 반전 TTA
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | README.md:26 |
| 날짜 | 2026-09-17 (추정: preds/test_ensemble 폴더 수정 시각) | preds/test_ensemble |
| 목적 | 인코더가 다른 두 모델 앙상블 | - |
| 비교 기준 | LG-E08 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 512, 겹침 0.5, cosine 가중); 원본 해상도 | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) | src/tiling.py:56 |
| 후처리 | 없음 | - |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | - |
| Val mIoU | 77.61%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | README.md:26 |
| 클래스별 IoU | 93.6 / 72.7 / 56.8 / 89.2 (Al / Al3Ni / Eutectic Si / Primary Si) | README.md:26 |
| Test·LB 점수 | test 예측만 있음 (preds/test_ensemble), 점수 기록 없음 | README.md:115 |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 결론 | +0.37pp. 처음 채택한 제출 구성 | - |

### LG-E10 B3 + B4 + SegFormer 앙상블 + 반전 TTA
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | README.md:27 |
| 날짜 | 2026-09-17 (추정: README 결과표 6단계, 날짜 기록 없음) | - |
| 목적 | 다른 계열 모델 추가 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E04) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E04) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E04) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E04) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E04) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E04) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| 추론 방식 | 겹치는 타일 (tile 512, 겹침 0.5, cosine 가중); 원본 해상도 | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) | src/tiling.py:56 |
| 후처리 | 없음 | - |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 + LG-E04 | - |
| Val mIoU | 77.52%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | README.md:27 |
| 클래스별 IoU | 93.2 / 72.3 / 56.8 / 89.5 (Al / Al3Ni / Eutectic Si / Primary Si) | README.md:27 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E04) | - |
| 결론 | −0.09pp. 성능 차가 큰 모델은 앙상블을 끌어내림 | - |

### LG-E11 DINOv3 ViT-B/16 + conv head
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s5_dinov3_b16/history.json → history |
| 날짜 | 2026-09-19 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | DINOv2를 DINOv3로 교체 | - |
| 비교 기준 | LG-E06 | - |
| 모델 구조 | ViT + 경량 conv head (DinoSegmenter; EoMT 아님) | src/models.py:115 |
| 백본 | ViT-B/16 (dinov3_vitb16) | runs/s5_dinov3_b16/history.json → args.arch |
| 사전학습 가중치 | DINOv3 (LVD-1689M 가중치 파일) | src/models.py:25 |
| 구조 변경 | patch token → conv head(2배 업샘플) → bilinear | src/models.py:115 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/s5_dinov3_b16/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s5_dinov3_b16/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | args에 loss 키 없음 → 코드 기본값 ce_dice (train_v2.py:333); train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 1e-4 / 인코더 1e-5; weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s5_dinov3_b16/history.json → args.lr; runs/s5_dinov3_b16/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s5_dinov3_b16/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s5_dinov3_b16/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0 (1회) | runs/s5_dinov3_b16/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s5_dinov3_b16/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s5_dinov3_b16/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 63.29% (단일 run); LG-E06 대비 +7.22pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s5_dinov3_b16/history.json → best_miou |
| 클래스별 IoU | 81.7 / 55.8 / 33.0 / 84.8 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s5_dinov3_b16/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 11분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s5_dinov3_b16.log |
| 결론 | DINOv2보다 낫지만 MicroNet U-Net++에 13pp 이상 뒤짐 | - |

### LG-E12 DINOv3 ViT-L/16 + conv head
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s6_dinov3_l16/history.json → history |
| 날짜 | 2026-09-19 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | ViT 크기 확대(B → L) | - |
| 비교 기준 | LG-E11 | - |
| 모델 구조 | ViT + 경량 conv head (DinoSegmenter; EoMT 아님) | src/models.py:115 |
| 백본 | ViT-L/16 (dinov3_vitl16) | runs/s6_dinov3_l16/history.json → args.arch |
| 사전학습 가중치 | DINOv3 (LVD-1689M 가중치 파일) | src/models.py:25 |
| 구조 변경 | patch token → conv head(2배 업샘플) → bilinear | src/models.py:115 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/s6_dinov3_l16/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s6_dinov3_l16/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | args에 loss 키 없음 → 코드 기본값 ce_dice (train_v2.py:333); train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 1e-4 / 인코더 1e-5; weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s6_dinov3_l16/history.json → args.lr; runs/s6_dinov3_l16/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s6_dinov3_l16/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s6_dinov3_l16/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0 (1회) | runs/s6_dinov3_l16/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s6_dinov3_l16/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s6_dinov3_l16/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 64.00% (단일 run); LG-E11 대비 +0.72pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s6_dinov3_l16/history.json → best_miou |
| 클래스별 IoU | 82.0 / 56.8 / 34.1 / 85.3 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s6_dinov3_l16/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 28분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s6_dinov3_l16.log |
| 결론 | 크기를 키워도 거의 그대로. Eutectic Si가 특히 낮아 feature 해상도가 병목 | - |

### LG-E13 B3 + B4 + DINOv3-L 앙상블 + 반전 TTA
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | README.md:28 |
| 날짜 | (추정: LG-E12 학습 이후, 날짜 기록 없음) | runs/s6_dinov3_l16/history.json 수정 시각 |
| 목적 | DINOv3 모델 추가 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E12) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E12) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E12) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E12) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E12) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E12) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| 추론 방식 | 겹치는 타일 (tile 512, 겹침 0.5, cosine 가중); 원본 해상도 | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) | src/tiling.py:56 |
| 후처리 | 없음 | - |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 + LG-E12 | - |
| Val mIoU | 77.31%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | README.md:28 |
| 클래스별 IoU | 93.1 / 72.1 / 56.4 / 89.4 (Al / Al3Ni / Eutectic Si / Primary Si) | README.md:28 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E12) | - |
| 결론 | −0.30pp | - |

### LG-E14 무작위 배율 증강 0.7–1.4배
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s7_scale_aug/history.json → history |
| 날짜 | 2026-09-21 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 스캔마다 다른 배율을 학습 때 섞어 줌 | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/s7_scale_aug/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/s7_scale_aug/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | args에 fullres_skip 없음 (기능 도입 전) |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 무작위 배율 (scale_aug=[0.7, 1.4], 학습 때만) | runs/s7_scale_aug/history.json → args.tile; train_v2.py:131; src/sampling.py:60 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s7_scale_aug/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | args에 loss 키 없음 → 코드 기본값 ce_dice (train_v2.py:333); train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s7_scale_aug/history.json → args.lr; runs/s7_scale_aug/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s7_scale_aug/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s7_scale_aug/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0 (1회) | runs/s7_scale_aug/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s7_scale_aug/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s7_scale_aug/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 76.66% (단일 run); LG-E05 대비 −0.34pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s7_scale_aug/history.json → best_miou |
| 클래스별 IoU | 92.6 / 70.9 / 56.0 / 89.1 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s7_scale_aug/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 12분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s7_scale_aug.log |
| 결론 | 손해. 학습 때만 배율을 흔들어 추론 조건과 어긋남 | - |

### LG-E15 무작위 배율 증강 1.0–1.6배 (확대만)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s8_zoomin/history.json → history |
| 날짜 | 2026-09-21 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 얇은 상을 키워서 보여 줌 | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/s8_zoomin/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/s8_zoomin/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | args에 fullres_skip 없음 (기능 도입 전) |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 무작위 배율 (scale_aug=[1.0, 1.6], 학습 때만) | runs/s8_zoomin/history.json → args.tile; train_v2.py:131; src/sampling.py:60 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s8_zoomin/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | args에 loss 키 없음 → 코드 기본값 ce_dice (train_v2.py:333); train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s8_zoomin/history.json → args.lr; runs/s8_zoomin/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s8_zoomin/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s8_zoomin/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0 (1회) | runs/s8_zoomin/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s8_zoomin/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s8_zoomin/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 75.19% (단일 run); LG-E05 대비 −1.81pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s8_zoomin/history.json → best_miou |
| 클래스별 IoU | 91.7 / 68.7 / 53.9 / 88.5 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s8_zoomin/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 12분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s8_zoomin.log |
| 결론 | 더 큰 손해. 시야가 좁아짐 | - |

### LG-E16 추론 타일 크기 512 / 384 / 320
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:115 |
| 날짜 | 2026-09-21 (추정: 같은 내용의 메모 기록일) | /home/user/.claude/projects/-home-user-Desktop/memory/komap-scale-aug-does-not-help.md |
| 목적 | 작은 스캔의 낮은 점수가 타일 패딩 탓인지 확인 | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05) | - |
| 백본 | 사용한 모델과 같음 (LG-E05) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05) | - |
| 추론 방식 | 겹치는 타일 (tile 512 / 384 / 320, 겹침 0.5) | src/tiling.py:56 |
| TTA | 없음 | - |
| 후처리 | 없음 | - |
| 앙상블 | 없음 | - |
| Val mIoU | 77.00 / 76.75 / 76.51%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:115 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05) | - |
| 결론 | 타일이 작을수록 단조 감소. 패딩 탓이 아님 | - |

### LG-E17 CE + Lovász (단일 run)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s9_ce_lovasz/history.json → history |
| 날짜 | 2026-09-21 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | IoU를 직접 겨냥한 Lovász loss | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/s9_ce_lovasz/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/s9_ce_lovasz/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | args에 fullres_skip 없음 (기능 도입 전) |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/s9_ce_lovasz/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s9_ce_lovasz/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Lovász (0.5·CE + 0.5·Lovász) | runs/s9_ce_lovasz/history.json → args.loss; train_v2.py:484 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s9_ce_lovasz/history.json → args.lr; runs/s9_ce_lovasz/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s9_ce_lovasz/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s9_ce_lovasz/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0 (1회) | runs/s9_ce_lovasz/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s9_ce_lovasz/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s9_ce_lovasz/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.17% (단일 run); LG-E05 대비 +0.17pp (단일 run); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s9_ce_lovasz/history.json → best_miou |
| 클래스별 IoU | 93.2 / 72.3 / 55.9 / 89.0 (Al / Al3Ni / Eutectic Si / Primary Si; best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s9_ce_lovasz/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 10분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s9_ce_lovasz.log |
| 결론 | 단일 run으로는 좋아 보였으나 LG-E18·LG-E19의 3 seed 비교와 LG-E36에서 효과 없음으로 판명 | - |

### LG-E18 CE + Dice + Lovász (3 seed)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/s10_ce_dice_lovasz/history.json → history |
| 날짜 | 2026-09-21 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | Lovász 효과를 seed 3개로 재확인 | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/s10_ce_dice_lovasz/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/s10_ce_dice_lovasz/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | args에 fullres_skip 없음 (기능 도입 전) |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/s10_ce_dice_lovasz/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/s10_ce_dice_lovasz/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice + Lovász (0.34 / 0.33 / 0.33) | runs/s10_ce_dice_lovasz/history.json → args.loss; train_v2.py:486 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/s10_ce_dice_lovasz/history.json → args.lr; runs/s10_ce_dice_lovasz/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/s10_ce_dice_lovasz/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/s10_ce_dice_lovasz/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/s10_ce_dice_lovasz/history.json → args.seed; runs/seed_cdl_1/history.json → args.seed; runs/seed_cdl_2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/s10_ce_dice_lovasz/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/s10_ce_dice_lovasz/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.17% ± 0.11 (seed 0: 77.27; seed 1: 77.06; seed 2: 77.17); LG-E19 대비 +0.15pp (Welch p=0.203); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/s10_ce_dice_lovasz/history.json → best_miou; runs/seed_cdl_1/history.json → best_miou; runs/seed_cdl_2/history.json → best_miou |
| 클래스별 IoU | 93.3 / 71.8 / 56.3 / 89.1 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/s10_ce_dice_lovasz/val_per_image.json → per_image; runs/seed_cdl_1/val_per_image.json → per_image; runs/seed_cdl_2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 10분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_s10_ce_dice_lovasz.log, logs_seed_cdl_1.log, logs_seed_cdl_2.log |
| 결론 | 차이가 seed 노이즈(±0.15pp) 안. 이후 모든 비교를 3 seed로 하는 계기 | - |

### LG-E19 CE + Dice 기준 (인코더 lr 3e-4, 3 seed)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/seed_base_1/history.json → history |
| 날짜 | 2026-09-21 ~ 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | seed 노이즈 측정과 1라운드 비교 기준. seed 0은 2026-09-30에 같은 조건으로 추가 학습 | - |
| 비교 기준 | LG-E05 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/seed_base_1/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/seed_base_1/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | args에 fullres_skip 없음 (기능 도입 전) |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/seed_base_1/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/seed_base_1/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/seed_base_1/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/seed_base_1/history.json → args.lr; runs/seed_base_1/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/seed_base_1/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/seed_base_1/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 1, 2, 0 (3회) | runs/seed_base_1/history.json → args.seed; runs/seed_base_2/history.json → args.seed; runs/seed_base_0/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/seed_base_1/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/seed_base_1/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.02% ± 0.13 (seed 1: 77.02; seed 2: 77.15; seed 0: 76.89); LG-E05 대비 +0.02pp (비교 기준이 단일 run이라 검정 없음); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/seed_base_1/history.json → best_miou; runs/seed_base_2/history.json → best_miou; runs/seed_base_0/history.json → best_miou |
| 클래스별 IoU | 93.3 / 71.3 / 56.2 / 89.0 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/seed_base_1/val_per_image.json → per_image; runs/seed_base_2/val_per_image.json → per_image; runs/seed_base_0/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 14분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_seed_base_1.log, logs_seed_base_2.log, logs_seed_base_0.log |
| 결론 | 1라운드 실험의 비교 기준. LG-E05 대비 인코더 lr만 다름 | - |

### LG-E20 클래스별 logit bias 보정
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:44 |
| 날짜 | 2026-09-30 | /home/user/.claude/projects/-home-user-Desktop/memory/komap-inference-only-gains.md |
| 목적 | 얇은 상을 더 예측하도록 판정 기준 조정 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 512, 겹침 0.5, cosine 가중); 원본 해상도 | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) | src/tiling.py:56 |
| 후처리 | 클래스별 logit bias (Al3Ni, Eutectic Si, Primary Si; 격자 탐색, leave-one-image-out) | - |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | - |
| Val mIoU | 77.57% (leave-one-out; 기준 77.61 대비 −0.04pp); 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:44 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 결론 | 효과 없음. 예측/정답 면적비 0.97–1.05로 체계적 편향이 없음 | - |

### LG-E21 Si 입자 투표
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:29 |
| 날짜 | 2026-09-30 | /home/user/.claude/projects/-home-user-Desktop/memory/komap-inference-only-gains.md |
| 목적 | Eutectic/Primary Si 입자 단위 오분류 교정 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 512, 겹침 0.5, cosine 가중); 원본 해상도 | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) | src/tiling.py:56 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | - |
| Val mIoU | 77.78%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:29 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 결론 | +0.17pp. 파라미터 없는 후처리 | - |

### LG-E22 8방향 TTA
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:30 |
| 날짜 | 2026-09-30 | /home/user/.claude/projects/-home-user-Desktop/memory/komap-inference-only-gains.md |
| 목적 | transpose를 더한 8방향 TTA | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 512, 겹침 0.5, cosine 가중); 원본 해상도 | src/tiling.py:56 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 없음 / 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | - |
| Val mIoU | 77.65% (투표 없음) / 78.08% (Si 입자 투표); 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:30 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 결론 | 단독 +0.04pp, 투표와 함께 +0.47pp | - |

### LG-E23 타일 768
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:31 |
| 날짜 | 2026-09-30 | /home/user/.claude/projects/-home-user-Desktop/memory/komap-inference-only-gains.md |
| 목적 | 추론 타일 확대 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768, 겹침 0.5) | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) | src/tiling.py:56 |
| 후처리 | 없음 / 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | - |
| Val mIoU | 77.71% (투표 없음) / 78.05% (Si 입자 투표); 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:31 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 결론 | +0.10pp | - |

### LG-E24 최종 추론 설정 (tile 768 + 8방향 TTA + Si 입자 투표)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:32 |
| 날짜 | 2026-09-30 | /home/user/.claude/projects/-home-user-Desktop/memory/komap-inference-only-gains.md |
| 목적 | 효과 있던 추론 요소 결합 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768, 겹침 0.5) | src/tiling.py:56 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | - |
| Val mIoU | 78.09% (투표 없으면 77.76%); 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:32 |
| 클래스별 IoU | 93.6 / 72.7 / 57.9 / 90.0 (Al / Al3Ni / Eutectic Si / Primary Si) | README.md:29 |
| Test·LB 점수 | test 예측만 있음 (preds/test_ensemble_d4_vote), 점수 기록 없음 | preds/test_ensemble_d4_vote |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 결론 | +0.48pp. 20장 중 15장 개선. 이후 모든 앙상블 평가에 사용 | - |

### LG-E25 이미지 전체 추론
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:34 |
| 날짜 | 2026-09-30 | /home/user/.claude/projects/-home-user-Desktop/memory/komap-inference-only-gains.md |
| 목적 | 타일 없이 이미지 전체를 한 번에 추론 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 추론 방식 | 전체 이미지 (32 배수 반사 패딩) | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) / 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 없음 / 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | - |
| Val mIoU | 77.76 / 77.78% (투표 없음, 반전 / 8방향); 78.05 / 78.07% (Si 입자 투표); 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:34 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 결론 | tile 768과 차이 0.03pp 이내 | - |

### LG-E26 타일 1024
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:35 |
| 날짜 | 2026-09-30 | /home/user/.claude/projects/-home-user-Desktop/memory/komap-inference-only-gains.md |
| 목적 | 더 큰 추론 타일 (반사 패딩 버그 수정 후) | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 1024, 겹침 0.5) | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) | src/tiling.py:56 |
| 후처리 | 없음 | - |
| 앙상블 | 확률 평균 (동일 가중); LG-E05 + LG-E07 | - |
| Val mIoU | 77.79%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:35 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07) | - |
| 결론 | tile 768과 같은 수준 | - |

### LG-E27 이미 학습된 7개 모델 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:105 |
| 날짜 | 2026-09-30 | /home/user/.claude/projects/-home-user-Desktop/memory/komap-inference-only-gains.md |
| 목적 | seed가 다른 모델을 더 섞어 경계 흔들림 감소 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 백본 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| Loss | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| Seed | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 추론 방식 | 겹치는 타일 (tile 512, 겹침 0.5, cosine 가중); 원본 해상도 / 최종 추론 설정 | src/tiling.py:56 |
| TTA | 반전 4가지 (원본, 좌우, 상하, 좌우+상하) / 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 없음 / 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); B3 6개(LG-E05, LG-E18 3개, LG-E19 seed 1·2) + B4(LG-E07) | - |
| Val mIoU | 77.84% (tile 512, 반전 TTA) / 78.14% (최종 추론 설정); 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:105 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E05, LG-E07, LG-E18, LG-E19) | - |
| 결론 | LG-E24 대비 +0.05pp뿐. 추론 비용 3.5배라 채택 안 함 | - |

### LG-E28 원본 해상도 branch (fullres)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_fullres_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | smp 디코더가 버리는 stride-1 특징을 되살림 | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_fullres_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_fullres_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_fullres_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_fullres_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_fullres_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_fullres_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_fullres_s0/history.json → args.lr; runs/b_fullres_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_fullres_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_fullres_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_fullres_s0/history.json → args.seed; runs/b_fullres_s1/history.json → args.seed; runs/b_fullres_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_fullres_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_fullres_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.72% ± 0.12 (seed 0: 77.68; seed 1: 77.85; seed 2: 77.62); LG-E19 대비 +0.70pp (Welch p=0.002); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_fullres_s0/history.json → best_miou; runs/b_fullres_s1/history.json → best_miou; runs/b_fullres_s2/history.json → best_miou |
| 클래스별 IoU | 93.7 / 72.5 / 57.0 / 89.4 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_fullres_s0/val_per_image.json → per_image; runs/b_fullres_s1/val_per_image.json → per_image; runs/b_fullres_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 21분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_fullres_s0.log, logs_b_fullres_s1.log, logs_b_fullres_s2.log |
| 결론 | 효과 있음. 20장 중 17장 개선, 1px 경계 오류 감소. 이후 기본 구조 | - |

### LG-E29 경계 label relaxation
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_relax_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 경계 ±1px 주석 흔들림을 허용 | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_relax_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_relax_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | runs/b_relax_s0/history.json → args.fullres_skip |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_relax_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_relax_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE(경계 label relaxation, 3×3 창) + Dice (0.5 / 0.5) | runs/b_relax_s0/history.json → args.loss; train_v2.py:470 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_relax_s0/history.json → args.lr; runs/b_relax_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_relax_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_relax_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_relax_s0/history.json → args.seed; runs/b_relax_s1/history.json → args.seed; runs/b_relax_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_relax_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_relax_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 75.05% ± 0.04 (seed 0: 75.03; seed 1: 75.10; seed 2: 75.02); LG-E19 대비 −1.96pp (Welch p=0.001); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_relax_s0/history.json → best_miou; runs/b_relax_s1/history.json → best_miou; runs/b_relax_s2/history.json → best_miou |
| 클래스별 IoU | 91.3 / 68.1 / 54.3 / 88.4 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_relax_s0/val_per_image.json → per_image; runs/b_relax_s1/val_per_image.json → per_image; runs/b_relax_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 18분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_relax_s0.log, logs_b_relax_s1.log, logs_b_relax_s2.log |
| 결론 | 해로움. 3px 상은 경계 픽셀이 곧 상 전체라 감독이 사라짐. Al3Ni 예측 면적 0.96 → 0.91 | - |

### LG-E30 JML + 경계 label smoothing
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_jml_bls_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 경계에만 label smoothing (논문 설정 k=3, ε=0.5) | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_jml_bls_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_jml_bls_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | runs/b_jml_bls_s0/history.json → args.fullres_skip |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_jml_bls_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_jml_bls_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE(경계 label smoothing ε=0.5, k=3) + Jaccard Metric Loss (0.25 / 0.75) | runs/b_jml_bls_s0/history.json → args.loss; train_v2.py:489 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_jml_bls_s0/history.json → args.lr; runs/b_jml_bls_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_jml_bls_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_jml_bls_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_jml_bls_s0/history.json → args.seed; runs/b_jml_bls_s1/history.json → args.seed; runs/b_jml_bls_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_jml_bls_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_jml_bls_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 76.02% ± 0.08 (seed 0: 76.12; seed 1: 75.99; seed 2: 75.97); LG-E19 대비 −0.99pp (Welch p=0.001); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_jml_bls_s0/history.json → best_miou; runs/b_jml_bls_s1/history.json → best_miou; runs/b_jml_bls_s2/history.json → best_miou |
| 클래스별 IoU | 92.7 / 70.2 / 54.6 / 88.5 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_jml_bls_s0/val_per_image.json → per_image; runs/b_jml_bls_s1/val_per_image.json → per_image; runs/b_jml_bls_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 19분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_jml_bls_s0.log, logs_b_jml_bls_s1.log, logs_b_jml_bls_s2.log |
| 결론 | 해로움 | - |

### LG-E31 Jaccard Metric Loss (대조군)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_jml_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | LG-E30의 base loss만 적용 | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_jml_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_jml_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | runs/b_jml_s0/history.json → args.fullres_skip |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_jml_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_jml_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Jaccard Metric Loss (0.25·CE + 0.75·JML) | runs/b_jml_s0/history.json → args.loss; train_v2.py:489 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_jml_s0/history.json → args.lr; runs/b_jml_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_jml_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_jml_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_jml_s0/history.json → args.seed; runs/b_jml_s1/history.json → args.seed; runs/b_jml_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_jml_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_jml_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.30% ± 0.34 (seed 0: 77.36; seed 1: 76.93; seed 2: 77.61); LG-E19 대비 +0.28pp (Welch p=0.292); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_jml_s0/history.json → best_miou; runs/b_jml_s1/history.json → best_miou; runs/b_jml_s2/history.json → best_miou |
| 클래스별 IoU | 93.2 / 71.9 / 56.6 / 89.3 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_jml_s0/val_per_image.json → per_image; runs/b_jml_s1/val_per_image.json → per_image; runs/b_jml_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 17분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_jml_s0.log, logs_b_jml_s1.log, logs_b_jml_s2.log |
| 결론 | 효과 없음 (seed 노이즈 수준) | - |

### LG-E32 Al3Ni–Si 계면 가중 CE
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_interface_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 최대 혼동쌍(Al3Ni ↔ Eutectic Si) 계면 강조 (Strohmann 2019) | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_interface_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_interface_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | runs/b_interface_s0/history.json → args.fullres_skip |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_interface_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_interface_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | 가중 CE (Al3Ni–Si 계면 1.5, Si 1.25) + Dice (0.5 / 0.5) | runs/b_interface_s0/history.json → args.loss; train_v2.py:472 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_interface_s0/history.json → args.lr; runs/b_interface_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_interface_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_interface_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_interface_s0/history.json → args.seed; runs/b_interface_s1/history.json → args.seed; runs/b_interface_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_interface_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_interface_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.07% ± 0.30 (seed 0: 76.89; seed 1: 76.89; seed 2: 77.41); LG-E19 대비 +0.05pp (Welch p=0.817); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_interface_s0/history.json → best_miou; runs/b_interface_s1/history.json → best_miou; runs/b_interface_s2/history.json → best_miou |
| 클래스별 IoU | 93.2 / 71.8 / 56.1 / 89.0 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_interface_s0/val_per_image.json → per_image; runs/b_interface_s1/val_per_image.json → per_image; runs/b_interface_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 18분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_interface_s0.log, logs_b_interface_s1.log, logs_b_interface_s2.log |
| 결론 | 효과 없음 | - |

### LG-E33 Boundary DoU
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_bdou_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | Dice를 Boundary DoU로 교체 | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_bdou_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_bdou_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | runs/b_bdou_s0/history.json → args.fullres_skip |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_bdou_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_bdou_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Boundary loss (Boundary DoU; 0.5·CE + 0.5·B-DoU) | runs/b_bdou_s0/history.json → args.loss; train_v2.py:492 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_bdou_s0/history.json → args.lr; runs/b_bdou_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_bdou_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_bdou_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_bdou_s0/history.json → args.seed; runs/b_bdou_s1/history.json → args.seed; runs/b_bdou_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_bdou_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_bdou_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 76.92% ± 0.26 (seed 0: 76.91; seed 1: 77.19; seed 2: 76.66); LG-E19 대비 −0.10pp (Welch p=0.601); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_bdou_s0/history.json → best_miou; runs/b_bdou_s1/history.json → best_miou; runs/b_bdou_s2/history.json → best_miou |
| 클래스별 IoU | 93.2 / 71.3 / 55.8 / 89.1 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_bdou_s0/val_per_image.json → per_image; runs/b_bdou_s1/val_per_image.json → per_image; runs/b_bdou_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 17분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_bdou_s0.log, logs_b_bdou_s1.log, logs_b_bdou_s2.log |
| 결론 | 효과 없음 | - |

### LG-E34 Kervadec boundary loss
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_kervadec_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 거리맵 기반 경계 loss (저자 스케줄) | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_kervadec_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_kervadec_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | runs/b_kervadec_s0/history.json → args.fullres_skip |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_kervadec_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_kervadec_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) + Boundary loss (Kervadec surface loss; 가중치 에폭마다 +0.01, 기본 loss 가중치 하한 0.1) | runs/b_kervadec_s0/history.json → args.loss; train_v2.py:482; train_v2.py:498 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_kervadec_s0/history.json → args.lr; runs/b_kervadec_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_kervadec_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_kervadec_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_kervadec_s0/history.json → args.seed; runs/b_kervadec_s1/history.json → args.seed; runs/b_kervadec_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_kervadec_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_kervadec_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 76.25% ± 0.42 (seed 0: 76.52; seed 1: 75.76; seed 2: 76.46); LG-E19 대비 −0.77pp (Welch p=0.076); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_kervadec_s0/history.json → best_miou; runs/b_kervadec_s1/history.json → best_miou; runs/b_kervadec_s2/history.json → best_miou |
| 클래스별 IoU | 92.9 / 70.6 / 54.6 / 88.8 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_kervadec_s0/val_per_image.json → per_image; runs/b_kervadec_s1/val_per_image.json → per_image; runs/b_kervadec_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 16분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_kervadec_s0.log, logs_b_kervadec_s1.log, logs_b_kervadec_s2.log |
| 결론 | 해로운 경향 | - |

### LG-E35 Active Boundary Loss
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_abl_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 예측 경계를 정답 경계 쪽으로 이동 (CE + Lovász 위에 추가) | - |
| 비교 기준 | LG-E36 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_abl_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_abl_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | runs/b_abl_s0/history.json → args.fullres_skip |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_abl_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_abl_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Lovász (0.5·CE + 0.5·Lovász) + Boundary loss (Active Boundary Loss, 가중치 0.5) | runs/b_abl_s0/history.json → args.loss; train_v2.py:484; train_v2.py:503 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_abl_s0/history.json → args.lr; runs/b_abl_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_abl_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_abl_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_abl_s0/history.json → args.seed; runs/b_abl_s1/history.json → args.seed; runs/b_abl_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_abl_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_abl_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.13% ± 0.18 (seed 0: 77.14; seed 1: 77.31; seed 2: 76.94); LG-E36 대비 +0.03pp (Welch p=0.850); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_abl_s0/history.json → best_miou; runs/b_abl_s1/history.json → best_miou; runs/b_abl_s2/history.json → best_miou |
| 클래스별 IoU | 93.3 / 71.7 / 56.2 / 89.2 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_abl_s0/val_per_image.json → per_image; runs/b_abl_s1/val_per_image.json → per_image; runs/b_abl_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 24분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_abl_s0.log, logs_b_abl_s1.log, logs_b_abl_s2.log |
| 결론 | 대조군과 차이 없음 | - |

### LG-E36 CE + Lovász (ABL 대조군)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_lovasz_s0/history.json → history |
| 날짜 | 2026-09-30 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | LG-E35의 base loss만 적용 (설정이 LG-E17 그대로) | - |
| 비교 기준 | LG-E19 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_lovasz_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_lovasz_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 없음 | runs/b_lovasz_s0/history.json → args.fullres_skip |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_lovasz_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_lovasz_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Lovász (0.5·CE + 0.5·Lovász) | runs/b_lovasz_s0/history.json → args.loss; train_v2.py:484 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_lovasz_s0/history.json → args.lr; runs/b_lovasz_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_lovasz_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_lovasz_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_lovasz_s0/history.json → args.seed; runs/b_lovasz_s1/history.json → args.seed; runs/b_lovasz_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_lovasz_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_lovasz_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.10% ± 0.12 (seed 0: 77.18; seed 1: 76.96; seed 2: 77.17); LG-E19 대비 +0.08pp (Welch p=0.467); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_lovasz_s0/history.json → best_miou; runs/b_lovasz_s1/history.json → best_miou; runs/b_lovasz_s2/history.json → best_miou |
| 클래스별 IoU | 93.3 / 71.9 / 56.0 / 88.9 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_lovasz_s0/val_per_image.json → per_image; runs/b_lovasz_s1/val_per_image.json → per_image; runs/b_lovasz_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 15분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_lovasz_s0.log, logs_b_lovasz_s1.log, logs_b_lovasz_s2.log |
| 결론 | 효과 없음 | - |

### LG-E37 기준 B3 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:55 |
| 날짜 | 2026-09-30 (추정: LG-E19 seed 0 학습 이후) | runs/seed_base_0/history.json 수정 시각 |
| 목적 | 최종 추론 설정에서의 기준 앙상블 점수 | - |
| 비교 기준 | 없음 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E19) | - |
| 백본 | 사용한 모델과 같음 (LG-E19) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E19) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E19) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E19) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E19) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E19) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E19) | - |
| Loss | 학습 없음 (사용한 모델: LG-E19) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E19) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E19) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E19) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E19) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E19) | - |
| Seed | 학습 없음 (사용한 모델: LG-E19) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E19) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E19) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E19 3개 | - |
| Val mIoU | 78.04%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:55 |
| 클래스별 IoU | 93.6 / 72.1 / 58.2 / 90.1 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:55 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E19) | - |
| 결론 | fullres 앙상블 비교 기준 | - |

### LG-E38 fullres 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:56 |
| 날짜 | 2026-10-01 (추정: LG-E28 학습 이후) | runs/b_fullres_s2/history.json 수정 시각 |
| 목적 | fullres 이득이 앙상블에서도 남는지 | - |
| 비교 기준 | LG-E37 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E28) | - |
| 백본 | 사용한 모델과 같음 (LG-E28) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E28) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E28) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E28) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E28) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E28) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E28) | - |
| Loss | 학습 없음 (사용한 모델: LG-E28) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E28) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E28) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E28) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E28) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E28) | - |
| Seed | 학습 없음 (사용한 모델: LG-E28) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E28) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E28) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E28 3개 | - |
| Val mIoU | 78.35%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:56 |
| 클래스별 IoU | 93.9 / 72.9 / 58.3 / 90.1 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:56 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E28) | - |
| 결론 | +0.31pp. 단일 모델보다 이득이 줄었지만 유지 | - |

### LG-E39 fullres 3-seed + B4 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:58 |
| 날짜 | 2026-10-01 (추정: LG-E28 학습 이후) | - |
| 목적 | 새 기준 앙상블 구성 | - |
| 비교 기준 | LG-E09 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E28, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E28, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E28, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E28, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E28, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E28, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E28 3개 + LG-E07 | - |
| Val mIoU | 78.47%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:58 |
| 클래스별 IoU | 93.9 / 73.1 / 58.5 / 90.3 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:58 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E28, LG-E07) | - |
| 결론 | LG-E24 대비 +0.38pp. 3라운드 전까지 최고 | - |

### LG-E40 fullres + EfficientNet-B4
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_fullres_b4_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | B4에도 fullres 적용 (인코더 lr 1e-4, LG-E07 설정 그대로) | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B4 (efficientnet-b4) | runs/b_fullres_b4_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_fullres_b4_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_fullres_b4_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_fullres_b4_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_fullres_b4_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_fullres_b4_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 1e-4; weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_fullres_b4_s0/history.json → args.lr; runs/b_fullres_b4_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_fullres_b4_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_fullres_b4_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_fullres_b4_s0/history.json → args.seed; runs/b_fullres_b4_s1/history.json → args.seed; runs/b_fullres_b4_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_fullres_b4_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_fullres_b4_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.28% ± 0.50 (seed 0: 76.71; seed 1: 77.50; seed 2: 77.63); LG-E28 대비 −0.44pp (Welch p=0.264); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_fullres_b4_s0/history.json → best_miou; runs/b_fullres_b4_s1/history.json → best_miou; runs/b_fullres_b4_s2/history.json → best_miou |
| 클래스별 IoU | 93.7 / 72.5 / 56.5 / 87.6 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_fullres_b4_s0/val_per_image.json → per_image; runs/b_fullres_b4_s1/val_per_image.json → per_image; runs/b_fullres_b4_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 24분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_fullres_b4_s0.log, logs_b_fullres_b4_s1.log, logs_b_fullres_b4_s2.log |
| 결론 | seed 편차가 커서 판단 불가 | - |

### LG-E41 fullres branch 채널 64
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_fullres_ch64_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 원본 해상도 branch 용량 확대 | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_fullres_ch64_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_fullres_ch64_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 64채널 → 디코더 출력과 결합) | runs/b_fullres_ch64_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_fullres_ch64_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_fullres_ch64_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_fullres_ch64_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_fullres_ch64_s0/history.json → args.lr; runs/b_fullres_ch64_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_fullres_ch64_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_fullres_ch64_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_fullres_ch64_s0/history.json → args.seed; runs/b_fullres_ch64_s1/history.json → args.seed; runs/b_fullres_ch64_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_fullres_ch64_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_fullres_ch64_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.52% ± 0.06 (seed 0: 77.59; seed 1: 77.51; seed 2: 77.47); LG-E28 대비 −0.19pp (Welch p=0.081); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_fullres_ch64_s0/history.json → best_miou; runs/b_fullres_ch64_s1/history.json → best_miou; runs/b_fullres_ch64_s2/history.json → best_miou |
| 클래스별 IoU | 93.8 / 72.1 / 56.8 / 89.3 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_fullres_ch64_s0/val_per_image.json → per_image; runs/b_fullres_ch64_s1/val_per_image.json → per_image; runs/b_fullres_ch64_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 23분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_fullres_ch64_s0.log, logs_b_fullres_ch64_s1.log, logs_b_fullres_ch64_s2.log |
| 결론 | 효과 없음. 용량은 병목이 아님 | - |

### LG-E42 fullres + CE + Lovász
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_fullres_lovasz_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | fullres 위에서 Lovász 재확인 | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_fullres_lovasz_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_fullres_lovasz_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_fullres_lovasz_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_fullres_lovasz_s0/history.json → args.tile; train_v2.py:131 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_fullres_lovasz_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Lovász (0.5·CE + 0.5·Lovász) | runs/b_fullres_lovasz_s0/history.json → args.loss; train_v2.py:484 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_fullres_lovasz_s0/history.json → args.lr; runs/b_fullres_lovasz_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_fullres_lovasz_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_fullres_lovasz_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_fullres_lovasz_s0/history.json → args.seed; runs/b_fullres_lovasz_s1/history.json → args.seed; runs/b_fullres_lovasz_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_fullres_lovasz_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_fullres_lovasz_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.43% ± 0.07 (seed 0: 77.39; seed 1: 77.38; seed 2: 77.51); LG-E28 대비 −0.29pp (Welch p=0.029); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_fullres_lovasz_s0/history.json → best_miou; runs/b_fullres_lovasz_s1/history.json → best_miou; runs/b_fullres_lovasz_s2/history.json → best_miou |
| 클래스별 IoU | 93.7 / 72.0 / 56.7 / 89.1 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_fullres_lovasz_s0/val_per_image.json → per_image; runs/b_fullres_lovasz_s1/val_per_image.json → per_image; runs/b_fullres_lovasz_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 23분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_fullres_lovasz_s0.log, logs_b_fullres_lovasz_s1.log, logs_b_fullres_lovasz_s2.log |
| 결론 | 조금 나빠짐 | - |

### LG-E43 fullres B4 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:57 |
| 날짜 | 2026-10-01 (추정: 2라운드 학습 이후) | - |
| 목적 | 2라운드 모델로 앙상블 구성 비교 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E40) | - |
| 백본 | 사용한 모델과 같음 (LG-E40) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E40) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E40) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E40) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E40) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E40) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E40) | - |
| Loss | 학습 없음 (사용한 모델: LG-E40) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E40) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E40) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E40) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E40) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E40) | - |
| Seed | 학습 없음 (사용한 모델: LG-E40) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E40) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E40) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E40 3개 | - |
| Val mIoU | 78.31%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:57 |
| 클래스별 IoU | 93.9 / 73.2 / 57.9 / 90.0 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:57 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E40) | - |
| 결론 | LG-E39(78.47%)와 같은 수준. 같은 계열 앙상블은 78.4%대에서 포화 | - |

### LG-E44 fullres B3 ×3 + fullres B4 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:60 |
| 날짜 | 2026-10-01 (추정: 2라운드 학습 이후) | - |
| 목적 | 2라운드 모델로 앙상블 구성 비교 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 백본 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Loss | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Seed | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E28 3개 + LG-E40 3개 | - |
| Val mIoU | 78.45%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:60 |
| 클래스별 IoU | 93.9 / 73.2 / 58.3 / 90.2 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:60 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 결론 | LG-E39(78.47%)와 같은 수준. 같은 계열 앙상블은 78.4%대에서 포화 | - |

### LG-E45 fullres B3 ×3 + fullres B4 (seed 2) 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:59 |
| 날짜 | 2026-10-01 (추정: 2라운드 학습 이후) | - |
| 목적 | 2라운드 모델로 앙상블 구성 비교 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 백본 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E28, LG-E40) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Loss | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| Seed | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E28 3개 + LG-E40 seed 2 | - |
| Val mIoU | 78.47%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:59 |
| 클래스별 IoU | 93.9 / 73.1 / 58.4 / 90.3 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:59 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E28, LG-E40) | - |
| 결론 | LG-E39(78.47%)와 같은 수준. 같은 계열 앙상블은 78.4%대에서 포화 | - |

### LG-E46 fullres(Lovász) ×3 + fullres B4 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:61 |
| 날짜 | 2026-10-01 (추정: 2라운드 학습 이후) | - |
| 목적 | 2라운드 모델로 앙상블 구성 비교 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E42, LG-E40) | - |
| 백본 | 사용한 모델과 같음 (LG-E42, LG-E40) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E42, LG-E40) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E42, LG-E40) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E42, LG-E40) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E42, LG-E40) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| Loss | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| Seed | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E42 3개 + LG-E40 3개 | - |
| Val mIoU | 78.44%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:61 |
| 클래스별 IoU | 93.9 / 72.9 / 58.5 / 90.3 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:61 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E42, LG-E40) | - |
| 결론 | LG-E39(78.47%)와 같은 수준. 같은 계열 앙상블은 78.4%대에서 포화 | - |

### LG-E47 fullres(채널 64) ×3 + fullres B4 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:62 |
| 날짜 | 2026-10-01 (추정: 2라운드 학습 이후) | - |
| 목적 | 2라운드 모델로 앙상블 구성 비교 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E41, LG-E40) | - |
| 백본 | 사용한 모델과 같음 (LG-E41, LG-E40) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E41, LG-E40) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E41, LG-E40) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E41, LG-E40) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E41, LG-E40) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| Loss | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| Seed | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E41 3개 + LG-E40 3개 | - |
| Val mIoU | 78.43%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:62 |
| 클래스별 IoU | 93.9 / 72.9 / 58.6 / 90.1 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:62 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E41, LG-E40) | - |
| 결론 | LG-E39(78.47%)와 같은 수준. 같은 계열 앙상블은 78.4%대에서 포화 | - |

### LG-E48 fullres 계열 12개 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:63 |
| 날짜 | 2026-10-01 (추정: 2라운드 학습 이후) | - |
| 목적 | 2라운드 모델로 앙상블 구성 비교 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 백본 | 사용한 모델과 같음 (LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| Loss | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| Seed | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); B3 9개(LG-E28, LG-E41, LG-E42) + LG-E40 3개 | - |
| Val mIoU | 78.46%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:63 |
| 클래스별 IoU | 93.9 / 72.9 / 58.5 / 90.4 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:63 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E28, LG-E41, LG-E42, LG-E40) | - |
| 결론 | LG-E39(78.47%)와 같은 수준. 같은 계열 앙상블은 78.4%대에서 포화 | - |

### LG-E49 D4 균등 선택
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_aug_d4_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 회전·반전 8가지를 균등하게 | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_aug_d4_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_aug_d4_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_aug_d4_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_aug_d4_s0/history.json → args.tile; runs/b_aug_d4_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전·수직 반전·90도 회전 (A.D4, 8가지 균등, p=1.0); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:150; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_aug_d4_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_aug_d4_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_aug_d4_s0/history.json → args.lr; runs/b_aug_d4_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_aug_d4_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_aug_d4_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_aug_d4_s0/history.json → args.seed; runs/b_aug_d4_s1/history.json → args.seed; runs/b_aug_d4_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_aug_d4_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_aug_d4_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.21% ± 0.20 (seed 0: 76.98; seed 1: 77.28; seed 2: 77.37); LG-E28 대비 −0.51pp (Welch p=0.029); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_aug_d4_s0/history.json → best_miou; runs/b_aug_d4_s1/history.json → best_miou; runs/b_aug_d4_s2/history.json → best_miou |
| 클래스별 IoU | 93.6 / 72.1 / 56.7 / 87.7 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_aug_d4_s0/val_per_image.json → per_image; runs/b_aug_d4_s1/val_per_image.json → per_image; runs/b_aug_d4_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 23분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_aug_d4_s0.log, logs_b_aug_d4_s1.log, logs_b_aug_d4_s2.log |
| 결론 | 손해. Primary Si 하락. 원인 불명 | - |

### LG-E50 밝기 증강 약하게 (aug_color)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_aug_color_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 밝기 증강 폭 축소, 한 번에 하나만, 노이즈 제거 | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_aug_color_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_aug_color_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_aug_color_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_aug_color_s0/history.json → args.tile; runs/b_aug_color_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_aug_color_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_aug_color_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_aug_color_s0/history.json → args.lr; runs/b_aug_color_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_aug_color_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_aug_color_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_aug_color_s0/history.json → args.seed; runs/b_aug_color_s1/history.json → args.seed; runs/b_aug_color_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_aug_color_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_aug_color_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.21% ± 0.03 (seed 0: 78.22; seed 1: 78.22; seed 2: 78.17); LG-E28 대비 +0.49pp (Welch p=0.014); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_aug_color_s0/history.json → best_miou; runs/b_aug_color_s1/history.json → best_miou; runs/b_aug_color_s2/history.json → best_miou |
| 클래스별 IoU | 93.9 / 73.6 / 57.6 / 89.5 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_aug_color_s0/val_per_image.json → per_image; runs/b_aug_color_s1/val_per_image.json → per_image; runs/b_aug_color_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 23분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_aug_color_s0.log, logs_b_aug_color_s1.log, logs_b_aug_color_s2.log |
| 결론 | 효과 있음. 밝기가 상 구분 단서라 덜 흔드는 게 유리 | - |

### LG-E51 입력 1.5배 확대
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_scale15_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 학습·추론 모두 1.5배 | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_scale15_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_scale15_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_scale15_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 배율 확대·축소 (input_scale=1.5, 학습 crop 768px) | runs/b_scale15_s0/history.json → args.tile; runs/b_scale15_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_scale15_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_scale15_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_scale15_s0/history.json → args.lr; runs/b_scale15_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_scale15_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_scale15_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_scale15_s0/history.json → args.seed; runs/b_scale15_s1/history.json → args.seed; runs/b_scale15_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_scale15_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 입력 1.5배 확대 후 확률맵을 원본 크기로 축소; 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_scale15_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.06% ± 0.12 (seed 0: 77.11; seed 1: 77.15; seed 2: 76.92); LG-E28 대비 −0.66pp (Welch p=0.002); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_scale15_s0/history.json → best_miou; runs/b_scale15_s1/history.json → best_miou; runs/b_scale15_s2/history.json → best_miou |
| 클래스별 IoU | 92.8 / 71.7 / 56.2 / 89.3 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_scale15_s0/val_per_image.json → per_image; runs/b_scale15_s1/val_per_image.json → per_image; runs/b_scale15_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 51분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_scale15_s0.log, logs_b_scale15_s1.log, logs_b_scale15_s2.log |
| 결론 | 손해. 원인 미확인 (정수배가 아닌 마스크 보간을 의심) | - |

### LG-E52 입력 2배 확대
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_scale2_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 학습·추론 모두 2배, 시야 유지(crop 1024) | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_scale2_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_scale2_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_scale2_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 배율 확대·축소 (input_scale=2, 학습 crop 1024px) | runs/b_scale2_s0/history.json → args.tile; runs/b_scale2_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_scale2_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_scale2_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_scale2_s0/history.json → args.lr; runs/b_scale2_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_scale2_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_scale2_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_scale2_s0/history.json → args.seed; runs/b_scale2_s1/history.json → args.seed; runs/b_scale2_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_scale2_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 입력 2배 확대 후 확률맵을 원본 크기로 축소; 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_scale2_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.21% ± 0.11 (seed 0: 78.09; seed 1: 78.20; seed 2: 78.32); LG-E28 대비 +0.49pp (Welch p=0.006); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_scale2_s0/history.json → best_miou; runs/b_scale2_s1/history.json → best_miou; runs/b_scale2_s2/history.json → best_miou |
| 클래스별 IoU | 93.7 / 73.2 / 58.0 / 89.8 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_scale2_s0/val_per_image.json → per_image; runs/b_scale2_s1/val_per_image.json → per_image; runs/b_scale2_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 82분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_scale2_s0.log, logs_b_scale2_s1.log, logs_b_scale2_s2.log |
| 결론 | 효과 있음. 얇은 상을 크게 보여 줌 | - |

### LG-E53 얇은 조직 중심 샘플링 30%
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_thin30_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 얇고 조밀한 Eutectic Si 위주 crop | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_thin30_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_thin30_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_thin30_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_thin30_s0/history.json → args.tile; runs/b_thin30_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15) | train_v2.py:151; train_v2.py:165 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭; 오버샘플링 (thin_ratio=0.3: 얇은 Eutectic Si 비율에 비례해 스캔 선택 후 밀집 지점 중심 crop) | runs/b_thin30_s0/history.json → args.class_ratio; train_v2.py:131; train_v2.py:212 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_thin30_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_thin30_s0/history.json → args.lr; runs/b_thin30_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_thin30_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_thin30_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_thin30_s0/history.json → args.seed; runs/b_thin30_s1/history.json → args.seed; runs/b_thin30_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_thin30_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_thin30_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.32% ± 0.27 (seed 0: 77.62; seed 1: 77.09; seed 2: 77.24); LG-E28 대비 −0.40pp (Welch p=0.112); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_thin30_s0/history.json → best_miou; runs/b_thin30_s1/history.json → best_miou; runs/b_thin30_s2/history.json → best_miou |
| 클래스별 IoU | 93.6 / 71.8 / 56.5 / 89.2 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_thin30_s0/val_per_image.json → per_image; runs/b_thin30_s1/val_per_image.json → per_image; runs/b_thin30_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 22분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_thin30_s0.log, logs_b_thin30_s1.log, logs_b_thin30_s2.log |
| 결론 | 효과 없음. 512 crop에서는 구성 비율이 거의 안 바뀜 | - |

### LG-E54 CutMix 20%
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_cutmix20_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 다른 crop의 사각형 영역 붙이기 | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_cutmix20_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_cutmix20_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_cutmix20_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_cutmix20_s0/history.json → args.tile; runs/b_cutmix20_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비 (RandomBrightnessContrast ±0.15, p=0.5); 감마 (RandomGamma 90–110, p=0.3); 가우시안 노이즈 (GaussNoise, p=0.15); CutMix (cutmix_p=0.2, 상자 면적 λ~U(0,1)) | train_v2.py:151; train_v2.py:165; train_v2.py:189 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_cutmix20_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_cutmix20_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_cutmix20_s0/history.json → args.lr; runs/b_cutmix20_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_cutmix20_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_cutmix20_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_cutmix20_s0/history.json → args.seed; runs/b_cutmix20_s1/history.json → args.seed; runs/b_cutmix20_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_cutmix20_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_cutmix20_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.44% ± 0.51 (seed 0: 77.72; seed 1: 77.75; seed 2: 76.84); LG-E28 대비 −0.28pp (Welch p=0.446); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_cutmix20_s0/history.json → best_miou; runs/b_cutmix20_s1/history.json → best_miou; runs/b_cutmix20_s2/history.json → best_miou |
| 클래스별 IoU | 93.6 / 72.9 / 56.7 / 87.6 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_cutmix20_s0/val_per_image.json → per_image; runs/b_cutmix20_s1/val_per_image.json → per_image; runs/b_cutmix20_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 23분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_cutmix20_s0.log, logs_b_cutmix20_s1.log, logs_b_cutmix20_s2.log |
| 결론 | 효과 없음. Primary Si 하락(사각형이 입자를 자름) | - |

### LG-E55 aug_color 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | runs/a_ensemble_baseline_valid/scores.json → mean_miou; ENSEMBLE_EXPERIMENTS.md:144 |
| 날짜 | 2026-10-01 (추정: 3라운드 학습 이후) | - |
| 목적 | 3라운드 모델 앙상블 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E50) | - |
| 백본 | 사용한 모델과 같음 (LG-E50) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E50) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E50) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E50) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E50) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E50) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E50) | - |
| Loss | 학습 없음 (사용한 모델: LG-E50) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E50) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E50) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E50) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E50) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E50) | - |
| Seed | 학습 없음 (사용한 모델: LG-E50) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E50) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E50) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E50 3개 | - |
| Val mIoU | 78.91%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | runs/a_ensemble_baseline_valid/scores.json → mean_miou; ENSEMBLE_EXPERIMENTS.md:144 |
| 클래스별 IoU | 94.0 / 74.1 / 58.8 / 90.5 (Al / Al3Ni / Eutectic Si / Primary Si) | runs/a_ensemble_baseline_valid/scores.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E50) | - |
| 결론 | +0.44pp | - |

### LG-E56 aug_color 3-seed + B4 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | README.md:276 |
| 날짜 | 2026-10-01 (추정: 3라운드 학습 이후) | - |
| 목적 | 3라운드 모델 앙상블 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E50, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E50, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E50, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E50, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E50, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E50, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E50 3개 + LG-E07 | - |
| Val mIoU | 78.84%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | README.md:276 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E50, LG-E07) | - |
| 결론 | B4를 섞으면 LG-E55보다 −0.07pp | - |

### LG-E57 scale2 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:145 |
| 날짜 | 2026-10-01 (추정: 3라운드 학습 이후) | - |
| 목적 | 3라운드 모델 앙상블 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E52) | - |
| 백본 | 사용한 모델과 같음 (LG-E52) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E52) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E52) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E52) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E52) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E52) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E52) | - |
| Loss | 학습 없음 (사용한 모델: LG-E52) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E52) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E52) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E52) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E52) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E52) | - |
| Seed | 학습 없음 (사용한 모델: LG-E52) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E52) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E52) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E52)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E52 3개 | - |
| Val mIoU | 78.99%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:145 |
| 클래스별 IoU | 93.8 / 73.6 / 59.6 / 90.9 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:145 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E52) | - |
| 결론 | +0.52pp | - |

### LG-E58 scale2 3-seed + B4 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 날짜 | 2026-10-01 (추정: 3라운드 학습 이후) | - |
| 목적 | 3라운드 모델 앙상블 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E52, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E52, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E52, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E52, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E52, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E52, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E52)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E52 3개 + LG-E07 | - |
| Val mIoU | 78.91%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E52, LG-E07) | - |
| 결론 | B4를 섞으면 LG-E57보다 −0.08pp | - |

### LG-E59 aug_color ×3 + scale2 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:146 |
| 날짜 | 2026-10-01 (추정: 3라운드 학습 이후) | - |
| 목적 | 3라운드 모델 앙상블 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E50, LG-E52) | - |
| 백본 | 사용한 모델과 같음 (LG-E50, LG-E52) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E50, LG-E52) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E50, LG-E52) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E50, LG-E52) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E50, LG-E52) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| Loss | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| Seed | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E52)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E50 3개 + LG-E52 3개 | - |
| Val mIoU | 79.07%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:146 |
| 클래스별 IoU | 94.0 / 74.1 / 59.3 / 90.6 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:146 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E50, LG-E52) | - |
| 결론 | +0.60pp. 두 방법 모델 혼합 | - |

### LG-E60 aug_color ×3 + scale2 ×3 + B4 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:147 |
| 날짜 | 2026-10-01 (추정: 3라운드 학습 이후) | - |
| 목적 | 3라운드 모델 앙상블 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E50, LG-E52, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E50, LG-E52, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E50, LG-E52, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E50, LG-E52, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E50, LG-E52, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E50, LG-E52, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E52)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E50 3개 + LG-E52 3개 + LG-E07 | - |
| Val mIoU | 79.08%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:147 |
| 클래스별 IoU | 94.0 / 74.1 / 59.4 / 90.7 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:147 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E50, LG-E52, LG-E07) | - |
| 결론 | LG-E59 대비 +0.01pp | - |

### LG-E61 fullres ×3 + aug_color ×3 + scale2 ×3 + B4 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:152 |
| 날짜 | 2026-10-01 (추정: 3라운드 학습 이후) | - |
| 목적 | 3라운드 모델 앙상블 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E52)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); 10개 | - |
| Val mIoU | 78.83%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:152 |
| 클래스별 IoU | 94.0 / 73.8 / 58.9 / 90.4 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:152 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E28, LG-E50, LG-E52, LG-E07) | - |
| 결론 | 약한 모델이 섞여 하락 | - |

### LG-E62 aug_color + 입력 2배 (color_scale2)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_color_scale2_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 3라운드에서 효과 있던 두 방법 결합 | - |
| 비교 기준 | LG-E28 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_color_scale2_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_color_scale2_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_color_scale2_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 배율 확대·축소 (input_scale=2, 학습 crop 1024px) | runs/b_color_scale2_s0/history.json → args.tile; runs/b_color_scale2_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_color_scale2_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_color_scale2_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_color_scale2_s0/history.json → args.lr; runs/b_color_scale2_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_color_scale2_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_color_scale2_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_color_scale2_s0/history.json → args.seed; runs/b_color_scale2_s1/history.json → args.seed; runs/b_color_scale2_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_color_scale2_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 입력 2배 확대 후 확률맵을 원본 크기로 축소; 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_color_scale2_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.74% ± 0.20 (seed 0: 78.52; seed 1: 78.93; seed 2: 78.76); LG-E28 대비 +1.02pp (Welch p=0.004); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_color_scale2_s0/history.json → best_miou; runs/b_color_scale2_s1/history.json → best_miou; runs/b_color_scale2_s2/history.json → best_miou |
| 클래스별 IoU | 94.0 / 74.3 / 58.5 / 89.9 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_color_scale2_s0/val_per_image.json → per_image; runs/b_color_scale2_s1/val_per_image.json → per_image; runs/b_color_scale2_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 93분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_color_scale2_s0.log, logs_b_color_scale2_s1.log, logs_b_color_scale2_s2.log |
| 결론 | 효과 있음. 두 이득이 거의 그대로 더해짐. 새 기준 | - |

### LG-E63 color_scale2 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:148 |
| 날짜 | 2026-10-01 (추정: LG-E62 학습 이후) | - |
| 목적 | color_scale2 앙상블 구성 | - |
| 비교 기준 | LG-E39 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E62) | - |
| 백본 | 사용한 모델과 같음 (LG-E62) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E62) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E62) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E62) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E62) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E62) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E62) | - |
| Loss | 학습 없음 (사용한 모델: LG-E62) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E62) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E62) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E62) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E62) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E62) | - |
| Seed | 학습 없음 (사용한 모델: LG-E62) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E62) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E62) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E62)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E62 3개 | - |
| Val mIoU | **79.61%**; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:148 |
| 클래스별 IoU | 94.1 / 74.8 / 60.3 / 91.0 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:148 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E62) | - |
| 결론 | +1.14pp. 당시 최고 | - |

### LG-E64 color_scale2 3-seed + B4 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:149 |
| 날짜 | 2026-10-01 (추정: LG-E62 학습 이후) | - |
| 목적 | color_scale2 앙상블 구성 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E62, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E62, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E62, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E62, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E62, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E62, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E62)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E62 3개 + LG-E07 | - |
| Val mIoU | 79.47%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:149 |
| 클래스별 IoU | 94.1 / 74.6 / 60.1 / 90.9 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:149 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E62, LG-E07) | - |
| 결론 | −0.14pp | - |

### LG-E65 color_scale2 ×3 + aug_color ×3 + scale2 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:150 |
| 날짜 | 2026-10-01 (추정: LG-E62 학습 이후) | - |
| 목적 | color_scale2 앙상블 구성 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52) | - |
| 백본 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| Loss | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| Seed | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E62, LG-E52)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); 9개 | - |
| Val mIoU | 79.37%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:150 |
| 클래스별 IoU | 94.1 / 74.4 / 59.9 / 90.9 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:150 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52) | - |
| 결론 | −0.24pp | - |

### LG-E66 color_scale2 ×3 + aug_color ×3 + scale2 ×3 + B4 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | ENSEMBLE_EXPERIMENTS.md:151 |
| 날짜 | 2026-10-01 (추정: LG-E62 학습 이후) | - |
| 목적 | color_scale2 앙상블 구성 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 백본 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| Loss | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| Seed | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E62, LG-E52)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); 10개 | - |
| Val mIoU | 79.32%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | ENSEMBLE_EXPERIMENTS.md:151 |
| 클래스별 IoU | 94.1 / 74.4 / 59.9 / 90.8 (Al / Al3Ni / Eutectic Si / Primary Si) | ENSEMBLE_EXPERIMENTS.md:151 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E62, LG-E50, LG-E52, LG-E07) | - |
| 결론 | −0.29pp. 약한 모델을 섞을수록 하락 | - |

### LG-E67 aug_color 120 에폭 (1차)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_ep120_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 학습 길이 연장 | - |
| 비교 기준 | LG-E50 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_ep120_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_ep120_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_ep120_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_ep120_s0/history.json → args.tile; runs/b_ep120_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_ep120_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_ep120_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_ep120_s0/history.json → args.lr; runs/b_ep120_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=120, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_ep120_s0/history.json → args.batch_size |
| Epoch·Step | 120 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_ep120_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_ep120_s0/history.json → args.seed; runs/b_ep120_s1/history.json → args.seed; runs/b_ep120_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 4 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_ep120_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_ep120_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.40% ± 0.15 (seed 0: 78.47; seed 1: 78.23; seed 2: 78.50); LG-E50 대비 +0.20pp (Welch p=0.142); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_ep120_s0/history.json → best_miou; runs/b_ep120_s1/history.json → best_miou; runs/b_ep120_s2/history.json → best_miou |
| 클래스별 IoU | 94.1 / 73.9 / 57.7 / 89.7 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_ep120_s0/val_per_image.json → per_image; runs/b_ep120_s1/val_per_image.json → per_image; runs/b_ep120_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 42분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_ep120_s0.log, logs_b_ep120_s1.log, logs_b_ep120_s2.log |
| 결론 | best는 올랐지만 마지막 에폭 점수는 오르지 않음. 이 실험만으로는 판단 불가 → LG-E71에서 재확인 | - |

### LG-E68 aug_color 200 에폭 (1차)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_ep200_s0/history.json → history |
| 날짜 | 2026-10-01 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 학습 길이 연장 | - |
| 비교 기준 | LG-E50 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_ep200_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_ep200_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_ep200_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/b_ep200_s0/history.json → args.tile; runs/b_ep200_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_ep200_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_ep200_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_ep200_s0/history.json → args.lr; runs/b_ep200_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=200, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_ep200_s0/history.json → args.batch_size |
| Epoch·Step | 200 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_ep200_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_ep200_s0/history.json → args.seed; runs/b_ep200_s1/history.json → args.seed; runs/b_ep200_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 6 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_ep200_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_ep200_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.65% ± 0.07 (seed 0: 78.58; seed 1: 78.72; seed 2: 78.66); LG-E50 대비 +0.45pp (Welch p=0.004); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_ep200_s0/history.json → best_miou; runs/b_ep200_s1/history.json → best_miou; runs/b_ep200_s2/history.json → best_miou |
| 클래스별 IoU | 94.2 / 74.3 / 57.9 / 89.9 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_ep200_s0/val_per_image.json → per_image; runs/b_ep200_s1/val_per_image.json → per_image; runs/b_ep200_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 70분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_ep200_s0.log, logs_b_ep200_s1.log, logs_b_ep200_s2.log |
| 결론 | best는 올랐지만 마지막 에폭 점수는 내려감. checkpoint를 val로 고른 효과와 구분 불가 → LG-E72에서 재확인 | - |

### LG-E69 1차 에폭 실험 앙상블 (120 / 200 에폭)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 날짜 | 2026-10-01 (추정: LG-E68 학습 이후) | runs/b_ep200_s2/history.json 수정 시각 |
| 목적 | 에폭 연장이 앙상블에서도 유지되는지 | - |
| 비교 기준 | LG-E55 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E67, LG-E68) | - |
| 백본 | 사용한 모델과 같음 (LG-E67, LG-E68) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E67, LG-E68) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E67, LG-E68) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E67, LG-E68) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E67, LG-E68) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| Loss | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| Seed | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E67 3개 / LG-E68 3개 | - |
| Val mIoU | 79.28% (120 에폭) / 79.52% (200 에폭); 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E67, LG-E68) | - |
| 결론 | +0.37 / +0.61pp. checkpoint를 val로 고른 효과가 섞였을 수 있음 | - |

### LG-E70 aug_color 60 에폭 재실행 (2차, 이미지별 기록)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/c_ep60_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 같은 설정(LG-E50)을 이미지별 점수 기록과 함께 재실행 (에폭 비교 기준) | - |
| 비교 기준 | LG-E50 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/c_ep60_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/c_ep60_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/c_ep60_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/c_ep60_s0/history.json → args.tile; runs/c_ep60_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/c_ep60_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/c_ep60_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/c_ep60_s0/history.json → args.lr; runs/c_ep60_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/c_ep60_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/c_ep60_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/c_ep60_s0/history.json → args.seed; runs/c_ep60_s1/history.json → args.seed; runs/c_ep60_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/c_ep60_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/c_ep60_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.09% ± 0.17 (seed 0: 78.21; seed 1: 77.90; seed 2: 78.15); LG-E50 대비 −0.12pp (Welch p=0.338); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/c_ep60_s0/history.json → best_miou; runs/c_ep60_s1/history.json → best_miou; runs/c_ep60_s2/history.json → best_miou |
| 클래스별 IoU | 93.9 / 73.3 / 57.4 / 89.5 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/c_ep60_s0/val_per_image.json → per_image; runs/c_ep60_s1/val_per_image.json → per_image; runs/c_ep60_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 22분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_c_ep60_s0.log, logs_c_ep60_s1.log, logs_c_ep60_s2.log |
| 결론 | LG-E50 3개 + 이 실험 3개 = 6 run이 에폭·5라운드 비교 기준 | - |

### LG-E71 aug_color 120 에폭 (2차)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/c_ep120_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | val 반쪽으로 고르고 나머지로 채점해 고른 효과 제거 | - |
| 비교 기준 | LG-E50 + LG-E70 (seed run 합침) | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/c_ep120_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/c_ep120_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/c_ep120_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/c_ep120_s0/history.json → args.tile; runs/c_ep120_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/c_ep120_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/c_ep120_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/c_ep120_s0/history.json → args.lr; runs/c_ep120_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=120, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/c_ep120_s0/history.json → args.batch_size |
| Epoch·Step | 120 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/c_ep120_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/c_ep120_s0/history.json → args.seed; runs/c_ep120_s1/history.json → args.seed; runs/c_ep120_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 4 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/c_ep120_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/c_ep120_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.51% ± 0.23 (seed 0: 78.26; seed 1: 78.54; seed 2: 78.72); LG-E50 + LG-E70 (seed run 합침) 대비 +0.36pp (Welch p=0.098); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/c_ep120_s0/history.json → best_miou; runs/c_ep120_s1/history.json → best_miou; runs/c_ep120_s2/history.json → best_miou |
| 클래스별 IoU | 94.0 / 73.7 / 58.1 / 90.0 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/c_ep120_s0/val_per_image.json → per_image; runs/c_ep120_s1/val_per_image.json → per_image; runs/c_ep120_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 42분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_c_ep120_s0.log, logs_c_ep120_s1.log, logs_c_ep120_s2.log |
| 결론 | 고른 효과를 뺀 차이 +0.58pp (split_half_check.py). 에폭 연장 효과 확정 | - |

### LG-E72 aug_color 200 에폭 (2차)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/c_ep200_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 위와 같음 | - |
| 비교 기준 | LG-E50 + LG-E70 (seed run 합침) | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/c_ep200_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/c_ep200_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/c_ep200_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/c_ep200_s0/history.json → args.tile; runs/c_ep200_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/c_ep200_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/c_ep200_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/c_ep200_s0/history.json → args.lr; runs/c_ep200_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=200, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/c_ep200_s0/history.json → args.batch_size |
| Epoch·Step | 200 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/c_ep200_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/c_ep200_s0/history.json → args.seed; runs/c_ep200_s1/history.json → args.seed; runs/c_ep200_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 6 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/c_ep200_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/c_ep200_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.58% ± 0.18 (seed 0: 78.47; seed 1: 78.78; seed 2: 78.48); LG-E50 + LG-E70 (seed run 합침) 대비 +0.43pp (Welch p=0.033); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/c_ep200_s0/history.json → best_miou; runs/c_ep200_s1/history.json → best_miou; runs/c_ep200_s2/history.json → best_miou |
| 클래스별 IoU | 94.2 / 74.1 / 57.9 / 89.8 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/c_ep200_s0/val_per_image.json → per_image; runs/c_ep200_s1/val_per_image.json → per_image; runs/c_ep200_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 70분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_c_ep200_s0.log, logs_c_ep200_s1.log, logs_c_ep200_s2.log |
| 결론 | 고른 효과를 뺀 차이 +0.57pp. 120 에폭 대비 추가 이득은 미확정 | - |

### LG-E73 2차 에폭 실험 앙상블 (60 / 120 / 200 에폭)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 날짜 | 2026-10-02 (추정: LG-E72 학습 이후) | runs/c_ep200_s2/history.json 수정 시각 |
| 목적 | 에폭별 앙상블 비교 | - |
| 비교 기준 | 없음 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E70, LG-E71, LG-E72) | - |
| 백본 | 사용한 모델과 같음 (LG-E70, LG-E71, LG-E72) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E70, LG-E71, LG-E72) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E70, LG-E71, LG-E72) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E70, LG-E71, LG-E72) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E70, LG-E71, LG-E72) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| Loss | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| Seed | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); 에폭마다 3개 | - |
| Val mIoU | 78.82% / 79.39% / 79.46%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 클래스별 IoU | 60 에폭 94.1 / 73.7 / 58.9 / 90.4; 120 에폭 94.2 / 74.3 / 59.9 / 91.0; 200 에폭 94.3 / 74.8 / 59.7 / 90.9 (Al / Al3Ni / Eutectic Si / Primary Si) | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E70, LG-E71, LG-E72) | - |
| 결론 | 1차와 2차를 평균하면 78.87 / 79.34 / 79.49% (EXPERIMENTS.md 에폭 절) | - |

### LG-E74 조명 필드
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/a_illumination_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 위치별로 매끄럽게 변하는 밝기 (팀원 제안 증강) | - |
| 비교 기준 | LG-E50 + LG-E70 (seed run 합침) | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/a_illumination_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/a_illumination_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/a_illumination_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/a_illumination_s0/history.json → args.tile; runs/a_illumination_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5); 밝기 (조명 필드 illumination_field ±5%, 약 96px 간격, p=0.2) | train_v2.py:151; train_v2.py:155; train_v2.py:230 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/a_illumination_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/a_illumination_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/a_illumination_s0/history.json → args.lr; runs/a_illumination_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/a_illumination_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/a_illumination_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/a_illumination_s0/history.json → args.seed; runs/a_illumination_s1/history.json → args.seed; runs/a_illumination_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/a_illumination_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/a_illumination_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.21% ± 0.19 (seed 0: 78.26; seed 1: 78.38; seed 2: 78.01); LG-E50 + LG-E70 (seed run 합침) 대비 +0.07pp (Welch p=0.608); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/a_illumination_s0/history.json → best_miou; runs/a_illumination_s1/history.json → best_miou; runs/a_illumination_s2/history.json → best_miou |
| 클래스별 IoU | 93.9 / 73.5 / 57.7 / 89.5 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/a_illumination_s0/val_per_image.json → per_image; runs/a_illumination_s1/val_per_image.json → per_image; runs/a_illumination_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 22분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_a_illumination_s0.log, logs_a_illumination_s1.log, logs_a_illumination_s2.log |
| 결론 | 단일 모델 차이는 노이즈 수준, 고른 효과를 뺀 차이 +0.39pp. 약한 긍정 | - |

### LG-E75 초점 흐림
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/a_blur_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 약한 PSF 흐림 | - |
| 비교 기준 | LG-E50 + LG-E70 (seed run 합침) | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/a_blur_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/a_blur_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/a_blur_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/a_blur_s0/history.json → args.tile; runs/a_blur_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5); 가우시안 블러 (weak_blur σ 0.3–0.65px, p=0.15) | train_v2.py:151; train_v2.py:155; train_v2.py:232 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/a_blur_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/a_blur_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/a_blur_s0/history.json → args.lr; runs/a_blur_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/a_blur_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/a_blur_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/a_blur_s0/history.json → args.seed; runs/a_blur_s1/history.json → args.seed; runs/a_blur_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/a_blur_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/a_blur_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.93% ± 0.07 (seed 0: 77.85; seed 1: 77.96; seed 2: 77.98); LG-E50 + LG-E70 (seed run 합침) 대비 −0.22pp (Welch p=0.014); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/a_blur_s0/history.json → best_miou; runs/a_blur_s1/history.json → best_miou; runs/a_blur_s2/history.json → best_miou |
| 클래스별 IoU | 93.7 / 73.2 / 57.1 / 89.3 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/a_blur_s0/val_per_image.json → per_image; runs/a_blur_s1/val_per_image.json → per_image; runs/a_blur_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 22분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_a_blur_s0.log, logs_a_blur_s1.log, logs_a_blur_s2.log |
| 결론 | 손해. 해상도가 병목이라 흐림은 해로움 | - |

### LG-E76 Fourier 혼합
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/a_fourier_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 다른 스캔의 저주파 진폭 일부 혼합 | - |
| 비교 기준 | LG-E50 + LG-E70 (seed run 합침) | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/a_fourier_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/a_fourier_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/a_fourier_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/a_fourier_s0/history.json → args.tile; runs/a_fourier_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5); 색 변환 (Fourier 저주파 진폭 10–20% 혼합 fourier_style, p=0.1) | train_v2.py:151; train_v2.py:155; train_v2.py:237 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/a_fourier_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/a_fourier_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/a_fourier_s0/history.json → args.lr; runs/a_fourier_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/a_fourier_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/a_fourier_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/a_fourier_s0/history.json → args.seed; runs/a_fourier_s1/history.json → args.seed; runs/a_fourier_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/a_fourier_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/a_fourier_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.22% ± 0.07 (seed 0: 78.29; seed 1: 78.21; seed 2: 78.15); LG-E50 + LG-E70 (seed run 합침) 대비 +0.07pp (Welch p=0.324); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/a_fourier_s0/history.json → best_miou; runs/a_fourier_s1/history.json → best_miou; runs/a_fourier_s2/history.json → best_miou |
| 클래스별 IoU | 93.9 / 73.4 / 57.8 / 89.6 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/a_fourier_s0/val_per_image.json → per_image; runs/a_fourier_s1/val_per_image.json → per_image; runs/a_fourier_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 22분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_a_fourier_s0.log, logs_a_fourier_s1.log, logs_a_fourier_s2.log |
| 결론 | 단일 모델 차이는 노이즈 수준, 고른 효과를 뺀 차이 +0.53pp. 약한 긍정 | - |

### LG-E77 스캔별 명암 표준화
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/a_tone_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 스캔마다 두 백분위를 같은 밝기로 맞춤 | - |
| 비교 기준 | LG-E50 + LG-E70 (seed run 합침) | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/a_tone_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/a_tone_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/a_tone_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/a_tone_s0/history.json → args.tile; runs/a_tone_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize); 명암 표준화 (tone_norm 5/70 백분위 → 43/219) | train_v2.py:169; train_v2.py:114; runs/a_tone_s0/history.json → args.tone_target |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/a_tone_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/a_tone_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/a_tone_s0/history.json → args.lr; runs/a_tone_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/a_tone_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/a_tone_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/a_tone_s0/history.json → args.seed; runs/a_tone_s1/history.json → args.seed; runs/a_tone_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/a_tone_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/a_tone_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 77.91% ± 0.12 (seed 0: 77.87; seed 1: 78.04; seed 2: 77.80); LG-E50 + LG-E70 (seed run 합침) 대비 −0.24pp (Welch p=0.050); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/a_tone_s0/history.json → best_miou; runs/a_tone_s1/history.json → best_miou; runs/a_tone_s2/history.json → best_miou |
| 클래스별 IoU | 93.8 / 72.9 / 57.1 / 89.5 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/a_tone_s0/val_per_image.json → per_image; runs/a_tone_s1/val_per_image.json → per_image; runs/a_tone_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 22분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_a_tone_s0.log, logs_a_tone_s1.log, logs_a_tone_s2.log |
| 결론 | 스캔 간 밝기 차이는 줄였지만 점수는 안 오름 | - |

### LG-E78 작은 스캔만 2배 확대 (6 seed)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/a_scale_small_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 높이 800px 이하 스캔만 2배 (조직 크기 표준화) | - |
| 비교 기준 | LG-E50 + LG-E70 (seed run 합침) | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/a_scale_small_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/a_scale_small_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/a_scale_small_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 배율 확대·축소 (scale_by_height 800:2: 높이 800px 이하 스캔은 2배) | runs/a_scale_small_s0/history.json → args.tile; runs/a_scale_small_s0/history.json → args.input_scale; train_v2.py:116 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5) | train_v2.py:151; train_v2.py:155 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/a_scale_small_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/a_scale_small_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/a_scale_small_s0/history.json → args.lr; runs/a_scale_small_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/a_scale_small_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/a_scale_small_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2, 3, 4, 5 (6회) | runs/a_scale_small_s0/history.json → args.seed; runs/a_scale_small_s1/history.json → args.seed; runs/a_scale_small_s2/history.json → args.seed … |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/a_scale_small_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 높이 800px 이하 스캔은 2배; 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/a_scale_small_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.25% ± 0.35 (seed 0: 77.71; seed 1: 78.26; seed 2: 78.68; seed 3: 78.02; seed 4: 78.48; seed 5: 78.38); LG-E50 + LG-E70 (seed run 합침) 대비 +0.11pp (Welch p=0.503); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/a_scale_small_s0/history.json → best_miou; runs/a_scale_small_s1/history.json → best_miou; runs/a_scale_small_s2/history.json → best_miou 외 3개 |
| 클래스별 IoU | 93.9 / 73.2 / 58.4 / 89.2 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/a_scale_small_s0/val_per_image.json → per_image; runs/a_scale_small_s1/val_per_image.json → per_image; runs/a_scale_small_s2/val_per_image.json → per_image 외 3개 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 23분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_a_scale_small_s0.log, logs_a_scale_small_s1.log, logs_a_scale_small_s2.log 외 |
| 결론 | 학습 중 검증으로는 노이즈 수준이지만 최종 추론 설정 단일 모델 +0.46pp(p=0.03). 이득은 확대하지 않은 큰 스캔에서 나옴 | - |

### LG-E79 Primary Si 입자 이동
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/a_prisi_paste_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 입자 단위 오분류(Eutectic ↔ Primary Si) 겨냥 | - |
| 비교 기준 | LG-E50 + LG-E70 (seed run 합침) | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/a_prisi_paste_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/a_prisi_paste_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/a_prisi_paste_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop) | runs/a_prisi_paste_s0/history.json → args.tile; runs/a_prisi_paste_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5); Copy-Paste (같은 스캔 안 Primary Si 입자 이동 paste_particle, p=0.25) | train_v2.py:151; train_v2.py:155; train_v2.py:219 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/a_prisi_paste_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/a_prisi_paste_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/a_prisi_paste_s0/history.json → args.lr; runs/a_prisi_paste_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/a_prisi_paste_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/a_prisi_paste_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/a_prisi_paste_s0/history.json → args.seed; runs/a_prisi_paste_s1/history.json → args.seed; runs/a_prisi_paste_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/a_prisi_paste_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/a_prisi_paste_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.39% ± 0.06 (seed 0: 78.37; seed 1: 78.35; seed 2: 78.46); LG-E50 + LG-E70 (seed run 합침) 대비 +0.25pp (Welch p=0.005); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/a_prisi_paste_s0/history.json → best_miou; runs/a_prisi_paste_s1/history.json → best_miou; runs/a_prisi_paste_s2/history.json → best_miou |
| 클래스별 IoU | 93.9 / 73.5 / 58.1 / 89.9 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/a_prisi_paste_s0/val_per_image.json → per_image; runs/a_prisi_paste_s1/val_per_image.json → per_image; runs/a_prisi_paste_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 22분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_a_prisi_paste_s0.log, logs_a_prisi_paste_s1.log, logs_a_prisi_paste_s2.log |
| 결론 | 효과 있음. 고른 효과를 뺀 차이 +0.74pp | - |

### LG-E80 조명 필드 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | runs/a_ensemble_illumination_valid/scores.json → mean_miou |
| 날짜 | 2026-10-02 (추정: 마지막 seed 학습 이후) | runs/a_illumination_s2/history.json 수정 시각 |
| 목적 | 5라운드 모델의 최종 추론 설정 점수 | - |
| 비교 기준 | LG-E55 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E74) | - |
| 백본 | 사용한 모델과 같음 (LG-E74) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E74) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E74) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E74) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E74) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E74) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E74) | - |
| Loss | 학습 없음 (사용한 모델: LG-E74) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E74) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E74) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E74) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E74) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E74) | - |
| Seed | 학습 없음 (사용한 모델: LG-E74) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E74) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E74) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E74 seed 0–2 | - |
| Val mIoU | 79.12%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | runs/a_ensemble_illumination_valid/scores.json → mean_miou |
| 클래스별 IoU | 94.1 / 74.1 / 59.4 / 90.7 (Al / Al3Ni / Eutectic Si / Primary Si) | runs/a_ensemble_illumination_valid/scores.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E74) | - |
| 결론 | LG-E55(78.91%) 대비 +0.21pp | - |

### LG-E81 초점 흐림 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | runs/a_ensemble_blur_valid/scores.json → mean_miou |
| 날짜 | 2026-10-02 (추정: 마지막 seed 학습 이후) | runs/a_blur_s2/history.json 수정 시각 |
| 목적 | 5라운드 모델의 최종 추론 설정 점수 | - |
| 비교 기준 | LG-E55 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E75) | - |
| 백본 | 사용한 모델과 같음 (LG-E75) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E75) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E75) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E75) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E75) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E75) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E75) | - |
| Loss | 학습 없음 (사용한 모델: LG-E75) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E75) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E75) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E75) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E75) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E75) | - |
| Seed | 학습 없음 (사용한 모델: LG-E75) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E75) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E75) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E75 seed 0–2 | - |
| Val mIoU | 78.94%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | runs/a_ensemble_blur_valid/scores.json → mean_miou |
| 클래스별 IoU | 93.9 / 73.9 / 59.0 / 90.6 (Al / Al3Ni / Eutectic Si / Primary Si) | runs/a_ensemble_blur_valid/scores.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E75) | - |
| 결론 | LG-E55(78.91%) 대비 +0.03pp | - |

### LG-E82 Fourier 혼합 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | runs/a_ensemble_fourier_valid/scores.json → mean_miou |
| 날짜 | 2026-10-02 (추정: 마지막 seed 학습 이후) | runs/a_fourier_s2/history.json 수정 시각 |
| 목적 | 5라운드 모델의 최종 추론 설정 점수 | - |
| 비교 기준 | LG-E55 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E76) | - |
| 백본 | 사용한 모델과 같음 (LG-E76) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E76) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E76) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E76) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E76) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E76) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E76) | - |
| Loss | 학습 없음 (사용한 모델: LG-E76) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E76) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E76) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E76) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E76) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E76) | - |
| Seed | 학습 없음 (사용한 모델: LG-E76) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E76) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E76) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E76 seed 0–2 | - |
| Val mIoU | 79.10%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | runs/a_ensemble_fourier_valid/scores.json → mean_miou |
| 클래스별 IoU | 94.0 / 74.0 / 59.6 / 90.7 (Al / Al3Ni / Eutectic Si / Primary Si) | runs/a_ensemble_fourier_valid/scores.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E76) | - |
| 결론 | LG-E55(78.91%) 대비 +0.19pp | - |

### LG-E83 명암 표준화 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | runs/a_ensemble_tone_valid/scores.json → mean_miou |
| 날짜 | 2026-10-02 (추정: 마지막 seed 학습 이후) | runs/a_tone_s2/history.json 수정 시각 |
| 목적 | 5라운드 모델의 최종 추론 설정 점수 | - |
| 비교 기준 | LG-E55 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E77) | - |
| 백본 | 사용한 모델과 같음 (LG-E77) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E77) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E77) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E77) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E77) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E77) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E77) | - |
| Loss | 학습 없음 (사용한 모델: LG-E77) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E77) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E77) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E77) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E77) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E77) | - |
| Seed | 학습 없음 (사용한 모델: LG-E77) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E77) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E77) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E77 seed 0–2 | - |
| Val mIoU | 78.81%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | runs/a_ensemble_tone_valid/scores.json → mean_miou |
| 클래스별 IoU | 94.0 / 73.5 / 59.0 / 90.5 (Al / Al3Ni / Eutectic Si / Primary Si) | runs/a_ensemble_tone_valid/scores.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E77) | - |
| 결론 | LG-E55(78.91%) 대비 −0.10pp | - |

### LG-E84 작은 스캔 2배 (seed 0–2) 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | runs/a_ensemble_scale_small_valid/scores.json → mean_miou |
| 날짜 | 2026-10-02 (추정: 마지막 seed 학습 이후) | runs/a_scale_small_s2/history.json 수정 시각 |
| 목적 | 5라운드 모델의 최종 추론 설정 점수 | - |
| 비교 기준 | LG-E55 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E78) | - |
| 백본 | 사용한 모델과 같음 (LG-E78) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E78) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E78) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E78) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E78) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E78) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E78) | - |
| Loss | 학습 없음 (사용한 모델: LG-E78) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E78) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E78) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E78) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E78) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E78) | - |
| Seed | 학습 없음 (사용한 모델: LG-E78) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E78) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E78) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E78)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E78 seed 0–2 | - |
| Val mIoU | 79.24%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | runs/a_ensemble_scale_small_valid/scores.json → mean_miou |
| 클래스별 IoU | 94.0 / 74.1 / 60.0 / 90.7 (Al / Al3Ni / Eutectic Si / Primary Si) | runs/a_ensemble_scale_small_valid/scores.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E78) | - |
| 결론 | LG-E55(78.91%) 대비 +0.33pp | - |

### LG-E85 Primary Si 입자 이동 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | runs/a_ensemble_prisi_paste_valid/scores.json → mean_miou |
| 날짜 | 2026-10-02 (추정: 마지막 seed 학습 이후) | runs/a_prisi_paste_s2/history.json 수정 시각 |
| 목적 | 5라운드 모델의 최종 추론 설정 점수 | - |
| 비교 기준 | LG-E55 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E79) | - |
| 백본 | 사용한 모델과 같음 (LG-E79) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E79) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E79) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E79) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E79) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E79) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E79) | - |
| Loss | 학습 없음 (사용한 모델: LG-E79) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E79) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E79) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E79) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E79) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E79) | - |
| Seed | 학습 없음 (사용한 모델: LG-E79) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E79) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E79) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중) | src/tiling.py:56; eval_ensembles.py:26 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E79 seed 0–2 | - |
| Val mIoU | 79.14%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | runs/a_ensemble_prisi_paste_valid/scores.json → mean_miou |
| 클래스별 IoU | 94.1 / 74.0 / 59.6 / 90.8 (Al / Al3Ni / Eutectic Si / Primary Si) | runs/a_ensemble_prisi_paste_valid/scores.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E79) | - |
| 결론 | LG-E55(78.91%) 대비 +0.23pp | - |

### LG-E86 작은 스캔 2배 6-seed 앙상블 대 기준 6-run 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:224 |
| 날짜 | 2026-10-02 (추정: LG-E78 학습 이후) | runs/a_scale_small_s5/history.json 수정 시각 |
| 목적 | 작은 스캔 확대 효과 재확인 | - |
| 비교 기준 | LG-E55 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E78, LG-E50, LG-E70) | - |
| 백본 | 사용한 모델과 같음 (LG-E78, LG-E50, LG-E70) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E78, LG-E50, LG-E70) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E78, LG-E50, LG-E70) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E78, LG-E50, LG-E70) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E78, LG-E50, LG-E70) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| Loss | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| Seed | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E78)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E78 6개 / 기준 LG-E50 3개 + LG-E70 3개 | - |
| Val mIoU | 79.42% 대 78.83% (seed 3–5만의 3-seed 앙상블은 79.34%); 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:224 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E78, LG-E50, LG-E70) | - |
| 결론 | +0.59pp. 큰 스캔 +0.68pp(p<0.01), 작은 스캔 +0.23pp(n.s.) | - |

### LG-E87 color_scale2 + 증강 3종 + 작은 스캔 4배 (cs2_plus)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_cs2_plus_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 배율 1에서 효과 있던 것들을 배율 2 위에 결합 | - |
| 비교 기준 | LG-E62 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_cs2_plus_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_cs2_plus_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_cs2_plus_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 배율 확대·축소 (input_scale=2, 학습 crop 1024px); 배율 확대·축소 (scale_by_height 800:4: 높이 800px 이하 스캔은 4배) | runs/b_cs2_plus_s0/history.json → args.tile; runs/b_cs2_plus_s0/history.json → args.input_scale; train_v2.py:116 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5); Copy-Paste (같은 스캔 안 Primary Si 입자 이동 paste_particle, p=0.25); 밝기 (조명 필드 illumination_field ±5%, 약 96px 간격, p=0.2); 색 변환 (Fourier 저주파 진폭 10–20% 혼합 fourier_style, p=0.1) | train_v2.py:151; train_v2.py:155; train_v2.py:219; train_v2.py:230; train_v2.py:237 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_cs2_plus_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_cs2_plus_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_cs2_plus_s0/history.json → args.lr; runs/b_cs2_plus_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_cs2_plus_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_cs2_plus_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_cs2_plus_s0/history.json → args.seed; runs/b_cs2_plus_s1/history.json → args.seed; runs/b_cs2_plus_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_cs2_plus_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 입력 2배 확대 후 확률맵을 원본 크기로 축소; 높이 800px 이하 스캔은 4배; 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_cs2_plus_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.87% ± 0.12 (seed 0: 78.87; seed 1: 78.76; seed 2: 78.99); LG-E62 대비 +0.13pp (Welch p=0.391); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_cs2_plus_s0/history.json → best_miou; runs/b_cs2_plus_s1/history.json → best_miou; runs/b_cs2_plus_s2/history.json → best_miou |
| 클래스별 IoU | 94.0 / 73.3 / 59.5 / 90.6 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_cs2_plus_s0/val_per_image.json → per_image; runs/b_cs2_plus_s1/val_per_image.json → per_image; runs/b_cs2_plus_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 97분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_cs2_plus_s0.log, logs_b_cs2_plus_s1.log, logs_b_cs2_plus_s2.log |
| 결론 | 결합 이득 없음. Al3Ni 하락, Eutectic·Primary Si 상승 | - |

### LG-E88 color_scale2 + 증강 3종 (cs2_aug3)
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_cs2_aug3_s0/history.json → history |
| 날짜 | 2026-10-02 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | LG-E87에서 작은 스캔 4배만 제외 | - |
| 비교 기준 | LG-E62 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_cs2_aug3_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_cs2_aug3_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_cs2_aug3_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 배율 확대·축소 (input_scale=2, 학습 crop 1024px) | runs/b_cs2_aug3_s0/history.json → args.tile; runs/b_cs2_aug3_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5); Copy-Paste (같은 스캔 안 Primary Si 입자 이동 paste_particle, p=0.25); 밝기 (조명 필드 illumination_field ±5%, 약 96px 간격, p=0.2); 색 변환 (Fourier 저주파 진폭 10–20% 혼합 fourier_style, p=0.1) | train_v2.py:151; train_v2.py:155; train_v2.py:219; train_v2.py:230; train_v2.py:237 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_cs2_aug3_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_cs2_aug3_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_cs2_aug3_s0/history.json → args.lr; runs/b_cs2_aug3_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=60, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_cs2_aug3_s0/history.json → args.batch_size |
| Epoch·Step | 60 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_cs2_aug3_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_cs2_aug3_s0/history.json → args.seed; runs/b_cs2_aug3_s1/history.json → args.seed; runs/b_cs2_aug3_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 2 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_cs2_aug3_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 입력 2배 확대 후 확률맵을 원본 크기로 축소; 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_cs2_aug3_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.81% ± 0.05 (seed 0: 78.77; seed 1: 78.86; seed 2: 78.82); LG-E62 대비 +0.08pp (Welch p=0.585); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_cs2_aug3_s0/history.json → best_miou; runs/b_cs2_aug3_s1/history.json → best_miou; runs/b_cs2_aug3_s2/history.json → best_miou |
| 클래스별 IoU | 94.0 / 74.6 / 58.5 / 90.0 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_cs2_aug3_s0/val_per_image.json → per_image; runs/b_cs2_aug3_s1/val_per_image.json → per_image; runs/b_cs2_aug3_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 92분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_cs2_aug3_s0.log, logs_b_cs2_aug3_s1.log, logs_b_cs2_aug3_s2.log |
| 결론 | 증강 3종은 배율 2 위에서 이득 없음. LG-E87의 Al3Ni 하락 원인은 작은 스캔 4배 | - |

### LG-E89 cs2_plus 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:235 |
| 날짜 | 2026-10-02 (추정: LG-E87·LG-E88 학습 이후) | - |
| 목적 | 배율 2 위 조합의 앙상블 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E87) | - |
| 백본 | 사용한 모델과 같음 (LG-E87) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E87) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E87) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E87) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E87) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E87) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E87) | - |
| Loss | 학습 없음 (사용한 모델: LG-E87) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E87) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E87) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E87) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E87) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E87) | - |
| Seed | 학습 없음 (사용한 모델: LG-E87) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E87) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E87) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E87)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E87 3개 | - |
| Val mIoU | 79.54%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:235 |
| 클래스별 IoU | 94.2 / 73.9 / 60.7 / 91.3 (Al / Al3Ni / Eutectic Si / Primary Si) | Al3Ni만 EXPERIMENTS.md:239; 나머지 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E87) | - |
| 결론 | −0.07pp | - |

### LG-E90 cs2_plus ×3 + color_scale2 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 날짜 | 2026-10-02 (추정: LG-E87·LG-E88 학습 이후) | - |
| 목적 | 배율 2 위 조합의 앙상블 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E87, LG-E62) | - |
| 백본 | 사용한 모델과 같음 (LG-E87, LG-E62) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E87, LG-E62) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E87, LG-E62) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E87, LG-E62) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E87, LG-E62) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| Loss | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| Seed | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E87, LG-E62)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E87 3개 + LG-E62 3개 | - |
| Val mIoU | 79.86%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 클래스별 IoU | 94.2 / 74.6 / 61.1 / 91.5 (Al / Al3Ni / Eutectic Si / Primary Si) | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E87, LG-E62) | - |
| 결론 | +0.25pp | - |

### LG-E91 cs2_aug3 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:234 |
| 날짜 | 2026-10-02 (추정: LG-E87·LG-E88 학습 이후) | - |
| 목적 | 배율 2 위 조합의 앙상블 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E88) | - |
| 백본 | 사용한 모델과 같음 (LG-E88) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E88) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E88) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E88) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E88) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E88) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E88) | - |
| Loss | 학습 없음 (사용한 모델: LG-E88) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E88) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E88) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E88) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E88) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E88) | - |
| Seed | 학습 없음 (사용한 모델: LG-E88) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E88) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E88) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E88)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E88 3개 | - |
| Val mIoU | 79.67%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:234 |
| 클래스별 IoU | 94.2 / 75.3 / 60.2 / 90.9 (Al / Al3Ni / Eutectic Si / Primary Si) | Al3Ni만 EXPERIMENTS.md:239; 나머지 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E88) | - |
| 결론 | +0.06pp | - |

### LG-E92 cs2_aug3 ×3 + color_scale2 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 날짜 | 2026-10-02 (추정: LG-E87·LG-E88 학습 이후) | - |
| 목적 | 배율 2 위 조합의 앙상블 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E88, LG-E62) | - |
| 백본 | 사용한 모델과 같음 (LG-E88, LG-E62) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E88, LG-E62) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E88, LG-E62) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E88, LG-E62) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E88, LG-E62) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| Loss | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| Seed | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E88, LG-E62)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E88 3개 + LG-E62 3개 | - |
| Val mIoU | 79.70%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 클래스별 IoU | 94.2 / 75.1 / 60.4 / 91.0 (Al / Al3Ni / Eutectic Si / Primary Si) | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E88, LG-E62) | - |
| 결론 | +0.09pp. 성격이 비슷해 이득 작음 | - |

### LG-E93 cs2_aug3 ×3 + cs2_plus ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:236 |
| 날짜 | 2026-10-02 (추정: LG-E87·LG-E88 학습 이후) | - |
| 목적 | 배율 2 위 조합의 앙상블 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E88, LG-E87) | - |
| 백본 | 사용한 모델과 같음 (LG-E88, LG-E87) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E88, LG-E87) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E88, LG-E87) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E88, LG-E87) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E88, LG-E87) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| Loss | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| Seed | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E88, LG-E87)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E88 3개 + LG-E87 3개 | - |
| Val mIoU | 79.91%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:236 |
| 클래스별 IoU | 94.2 / 74.8 / 61.0 / 91.4 (Al / Al3Ni / Eutectic Si / Primary Si) | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E88, LG-E87) | - |
| 결론 | +0.30pp. 60 에폭 구성 중 최고. 성격이 다른 두 구성이라 앙상블이 잘 오름 | - |

### LG-E94 cs2_aug3 ×3 + cs2_plus ×3 + color_scale2 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 날짜 | 2026-10-02 (추정: LG-E87·LG-E88 학습 이후) | - |
| 목적 | 배율 2 위 조합의 앙상블 | - |
| 비교 기준 | LG-E63 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E88, LG-E87, LG-E62) | - |
| 백본 | 사용한 모델과 같음 (LG-E88, LG-E87, LG-E62) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E88, LG-E87, LG-E62) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E88, LG-E87, LG-E62) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E88, LG-E87, LG-E62) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E88, LG-E87, LG-E62) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| Loss | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| Seed | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E88, LG-E87, LG-E62)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); 9개 | - |
| Val mIoU | 79.90%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| 클래스별 IoU | 94.2 / 74.9 / 61.0 / 91.4 (Al / Al3Ni / Eutectic Si / Primary Si) | 세션 출력 (파일 기록 없음; 재현: 같은 체크포인트로 predict.py/eval_ensembles.py) |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E88, LG-E87, LG-E62) | - |
| 결론 | +0.29pp | - |

### LG-E95 cs2_aug3 200 에폭
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_cs2_aug3_ep200_s0/history.json → history |
| 날짜 | 2026-10-03 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 확인된 에폭 연장 효과 적용 | - |
| 비교 기준 | LG-E88 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_cs2_aug3_ep200_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_cs2_aug3_ep200_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_cs2_aug3_ep200_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 배율 확대·축소 (input_scale=2, 학습 crop 1024px) | runs/b_cs2_aug3_ep200_s0/history.json → args.tile; runs/b_cs2_aug3_ep200_s0/history.json → args.input_scale |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5); Copy-Paste (같은 스캔 안 Primary Si 입자 이동 paste_particle, p=0.25); 밝기 (조명 필드 illumination_field ±5%, 약 96px 간격, p=0.2); 색 변환 (Fourier 저주파 진폭 10–20% 혼합 fourier_style, p=0.1) | train_v2.py:151; train_v2.py:155; train_v2.py:219; train_v2.py:230; train_v2.py:237 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_cs2_aug3_ep200_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_cs2_aug3_ep200_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_cs2_aug3_ep200_s0/history.json → args.lr; runs/b_cs2_aug3_ep200_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=200, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_cs2_aug3_ep200_s0/history.json → args.batch_size |
| Epoch·Step | 200 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_cs2_aug3_ep200_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_cs2_aug3_ep200_s0/history.json → args.seed; runs/b_cs2_aug3_ep200_s1/history.json → args.seed; runs/b_cs2_aug3_ep200_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 6 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_cs2_aug3_ep200_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 입력 2배 확대 후 확률맵을 원본 크기로 축소; 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_cs2_aug3_ep200_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 78.99% ± 0.11 (seed 0: 79.11; seed 1: 78.89; seed 2: 78.97); LG-E88 대비 +0.18pp (Welch p=0.094); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_cs2_aug3_ep200_s0/history.json → best_miou; runs/b_cs2_aug3_ep200_s1/history.json → best_miou; runs/b_cs2_aug3_ep200_s2/history.json → best_miou |
| 클래스별 IoU | 94.3 / 74.7 / 58.4 / 90.2 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_cs2_aug3_ep200_s0/val_per_image.json → per_image; runs/b_cs2_aug3_ep200_s1/val_per_image.json → per_image; runs/b_cs2_aug3_ep200_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 300분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_cs2_aug3_ep200_s0.log, logs_b_cs2_aug3_ep200_s1.log, logs_b_cs2_aug3_ep200_s2.log |
| 결론 | LG-E62 대비 +0.26pp(p=0.15). 고른 효과를 뺀 점수 78.36% | - |

### LG-E96 cs2_plus 200 에폭
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 (모든 run이 설정한 에폭까지 학습) | runs/b_cs2_plus_ep200_s0/history.json → history |
| 날짜 | 2026-10-03 | runs/*/history.json 파일 수정 시각 (학습 종료 시각) |
| 목적 | 확인된 에폭 연장 효과 적용 | - |
| 비교 기준 | LG-E87 | - |
| 모델 구조 | U-Net++ (smp.UnetPlusPlus) | src/models.py:219 |
| 백본 | EfficientNet-B3 (efficientnet-b3) | runs/b_cs2_plus_ep200_s0/history.json → args.encoder |
| 사전학습 가중치 | MicroNet(현미경) (imagenet-microscopynet 가중치) | runs/b_cs2_plus_ep200_s0/history.json → args.pretrain; src/models.py:175 |
| 구조 변경 | 원본 해상도 branch (FullResSkip, 입력에 conv3×3 2개, 32채널 → 디코더 출력과 결합) | runs/b_cs2_plus_ep200_s0/history.json → args.fullres_skip; src/models.py:144 |
| 입력 처리 | 원본 해상도 crop (tile=512, 겹침 없는 학습 crop); 배율 확대·축소 (input_scale=2, 학습 crop 1024px); 배율 확대·축소 (scale_by_height 800:4: 높이 800px 이하 스캔은 4배) | runs/b_cs2_plus_ep200_s0/history.json → args.tile; runs/b_cs2_plus_ep200_s0/history.json → args.input_scale; train_v2.py:116 |
| 전처리·정규화 | ImageNet mean/std 정규화 (A.Normalize) | train_v2.py:169 |
| 데이터 증강 | 수평 반전 (HorizontalFlip, p=0.5); 수직 반전 (VerticalFlip, p=0.5); 90도 회전 (RandomRotate90, p=0.5); 밝기·대비·감마 중 하나 (OneOf: RandomBrightnessContrast 밝기 ±0.1 또는 대비 ±0.1, RandomGamma 90–110; p=0.5); Copy-Paste (같은 스캔 안 Primary Si 입자 이동 paste_particle, p=0.25); 밝기 (조명 필드 illumination_field ±5%, 약 96px 간격, p=0.2); 색 변환 (Fourier 저주파 진폭 10–20% 혼합 fourier_style, p=0.1) | train_v2.py:151; train_v2.py:155; train_v2.py:219; train_v2.py:230; train_v2.py:237 |
| 샘플링 | 클래스 균형 crop (ClassAwareCropper class_ratio=0.5); 스캔당 8 crop/에폭 | runs/b_cs2_plus_ep200_s0/history.json → args.class_ratio; train_v2.py:131 |
| Loss | CE + Dice (0.5·CE + 0.5·Dice) | runs/b_cs2_plus_ep200_s0/history.json → args.loss; train_v2.py:482 |
| Optimizer | AdamW; lr 디코더·헤드 3e-4 / 인코더 3e-4 (args.enc_lr=None → lr과 같음); weight decay 1e-4; betas 기본값 (0.9, 0.999) | runs/b_cs2_plus_ep200_s0/history.json → args.lr; runs/b_cs2_plus_ep200_s0/history.json → args.enc_lr; train_v2.py:446 |
| LR 스케줄 | Cosine (CosineAnnealingLR, T_max=200, 에폭마다 갱신); warmup 없음 | train_v2.py:455 |
| Batch size | 8 | runs/b_cs2_plus_ep200_s0/history.json → args.batch_size |
| Epoch·Step | 200 에폭; 에폭당 70 step (70장 × 8 crop ÷ 배치 8) | runs/b_cs2_plus_ep200_s0/history.json → args.epochs; train_v2.py:395 |
| 정규화·안정화 | Mixed precision(AMP); Weight decay 1e-4; EMA 없음; Gradient clipping 없음 | train_v2.py:456 |
| Seed | 0, 1, 2 (3회) | runs/b_cs2_plus_ep200_s0/history.json → args.seed; runs/b_cs2_plus_ep200_s1/history.json → args.seed; runs/b_cs2_plus_ep200_s2/history.json → args.seed |
| 학습 데이터 | train만 (train/train 70장) | train_v2.py:422 |
| 검증·checkpoint 선택 | valid 20장 전체 이미지; 6 에폭마다 + 마지막 에폭; val mIoU 최고 checkpoint 저장 (best.pth) | runs/b_cs2_plus_ep200_s0/history.json → args.val_every; train_v2.py:550 |
| 추론 방식 | 겹치는 타일 (tile 512px 원본 기준, 겹침 0.5, cosine 가중 합성); 입력 2배 확대 후 확률맵을 원본 크기로 축소; 높이 800px 이하 스캔은 4배; 원본 해상도에서 채점 | train_v2.py:302; src/tiling.py:56; runs/b_cs2_plus_ep200_s0/history.json → args.overlap |
| TTA | 없음 (학습 중 검증 기준) | train_v2.py:302 |
| 후처리 | 없음 | train_v2.py:302 |
| 앙상블 | 없음 | - |
| Val mIoU | 79.19% ± 0.18 (seed 0: 79.15; seed 1: 79.03; seed 2: 79.39); LG-E87 대비 +0.32pp (Welch p=0.076); 측정 조건: valid 20장, 학습 중 val 최고 checkpoint, 겹치는 타일, TTA 없음 | runs/b_cs2_plus_ep200_s0/history.json → best_miou; runs/b_cs2_plus_ep200_s1/history.json → best_miou; runs/b_cs2_plus_ep200_s2/history.json → best_miou |
| 클래스별 IoU | 94.3 / 74.5 / 59.2 / 90.5 (Al / Al3Ni / Eutectic Si / Primary Si; seed 평균, best checkpoint, 그 클래스가 있는 이미지만 평균) | runs/b_cs2_plus_ep200_s0/val_per_image.json → per_image; runs/b_cs2_plus_ep200_s1/val_per_image.json → per_image; runs/b_cs2_plus_ep200_s2/val_per_image.json → per_image |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | run당 약 303분 (로그의 에폭 시간 합, 검증 제외; 여러 run을 동시에 돌린 경우 포함); GPU 1장; GPU 메모리 확인 불가 | logs_b_cs2_plus_ep200_s0.log, logs_b_cs2_plus_ep200_s1.log, logs_b_cs2_plus_ep200_s2.log |
| 결론 | LG-E62 대비 +0.45pp(p=0.05). 고른 효과를 뺀 점수 78.69%. 200 에폭에서는 작은 스캔 4배가 Al3Ni를 깎지 않음 | - |

### LG-E97 cs2_aug3 200 에폭 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:246 |
| 날짜 | 2026-10-03 (추정: 200 에폭 학습 이후) | - |
| 목적 | 최종 구성 앙상블 | - |
| 비교 기준 | LG-E91 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E95) | - |
| 백본 | 사용한 모델과 같음 (LG-E95) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E95) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E95) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E95) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E95) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E95) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E95) | - |
| Loss | 학습 없음 (사용한 모델: LG-E95) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E95) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E95) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E95) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E95) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E95) | - |
| Seed | 학습 없음 (사용한 모델: LG-E95) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E95) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E95) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E95)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E95 3개 | - |
| Val mIoU | 79.86%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:246 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E95) | - |
| 결론 | +0.19pp | - |

### LG-E98 cs2_plus 200 에폭 3-seed 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:247 |
| 날짜 | 2026-10-03 (추정: 200 에폭 학습 이후) | - |
| 목적 | 최종 구성 앙상블 | - |
| 비교 기준 | LG-E89 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E96) | - |
| 백본 | 사용한 모델과 같음 (LG-E96) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E96) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E96) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E96) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E96) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E96) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E96) | - |
| Loss | 학습 없음 (사용한 모델: LG-E96) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E96) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E96) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E96) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E96) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E96) | - |
| Seed | 학습 없음 (사용한 모델: LG-E96) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E96) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E96) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E96)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E96 3개 | - |
| Val mIoU | 79.95%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:247 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E96) | - |
| 결론 | +0.41pp | - |

### LG-E99 cs2_aug3 200 에폭 ×3 + cs2_plus 200 에폭 ×3 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:248 |
| 날짜 | 2026-10-03 (추정: 200 에폭 학습 이후) | - |
| 목적 | 최종 구성 앙상블 | - |
| 비교 기준 | LG-E93 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E95, LG-E96) | - |
| 백본 | 사용한 모델과 같음 (LG-E95, LG-E96) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E95, LG-E96) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E95, LG-E96) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E95, LG-E96) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E95, LG-E96) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| Loss | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| Seed | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E95, LG-E96)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); LG-E95 3개 + LG-E96 3개 | - |
| Val mIoU | 80.14%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:248 |
| 클래스별 IoU | 94.5 / 75.5 / 60.9 / 91.4 (Al / Al3Ni / Eutectic Si / Primary Si) | EXPERIMENTS.md:251 |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E95, LG-E96) | - |
| 결론 | +0.23pp. 10/3 당시 최고. 처음 기준보다 20장 전부 개선 | - |

### LG-E100 200 에폭 6개 + 60 에폭 6개 앙상블
| 항목 | 내용 | 근거 |
|---|---|---|
| 상태 | 완료 | EXPERIMENTS.md:249 |
| 날짜 | 2026-10-03 (추정: 200 에폭 학습 이후) | - |
| 목적 | 최종 구성 앙상블 | - |
| 비교 기준 | LG-E99 | - |
| 모델 구조 | 사용한 모델과 같음 (LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 백본 | 사용한 모델과 같음 (LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 사전학습 가중치 | 사용한 모델과 같음 (LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 구조 변경 | 사용한 모델과 같음 (LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 입력 처리 | 사용한 모델과 같음 (LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 전처리·정규화 | 사용한 모델과 같음 (LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 데이터 증강 | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 샘플링 | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| Loss | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| Optimizer | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| LR 스케줄 | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| Batch size | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| Epoch·Step | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 정규화·안정화 | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| Seed | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 학습 데이터 | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 검증·checkpoint 선택 | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 추론 방식 | 겹치는 타일 (tile 768 원본 기준, 겹침 0.5, cosine 가중); 배율을 바꿔 학습한 모델(LG-E95, LG-E96, LG-E88, LG-E87)은 학습 때와 같은 배율로 입력을 확대해 추론하고 확률맵을 원본 크기로 축소 | src/tiling.py:56; eval_ensembles.py:26; predict.py:111 |
| TTA | 8방향(D4) (반전 4가지 × transpose) | predict.py:58 |
| 후처리 | 연결 성분 단위 투표 (Eutectic/Primary Si 입자마다 확률 합이 큰 쪽으로 통일) | src/postprocess.py:6 |
| 앙상블 | 확률 평균 (동일 가중); 12개 | - |
| Val mIoU | 80.09%; 측정 조건: valid 20장, 각 모델의 학습 중 val 최고 checkpoint, 위 추론 설정 | EXPERIMENTS.md:249 |
| 클래스별 IoU | 확인 불가 | - |
| Test·LB 점수 | 확인 불가 | - |
| 학습 시간·자원 | 학습 없음 (사용한 모델: LG-E95, LG-E96, LG-E88, LG-E87) | - |
| 결론 | −0.05pp. 60 에폭 모델은 도움 안 됨 | - |

## 5. 실험에서 얻은 결론
| 바꾼 것 | 비교한 실험 ID | mIoU 차이 | 판단 |
|---|---|---|---|
| 전체 리사이즈 → 원본 해상도 crop | LG-E02 vs LG-E01 | +7.31pp (단일 run) | 효과 있음 (가장 큰 차이) |
| ResNet-50 U-Net++ (ImageNet-1k, train.py) → EfficientNet-B3 U-Net++ (MicroNet, train_v2.py) | LG-E05 vs LG-E03 | +1.31pp (단일 run) | 효과 있음 (사전학습·백본·crop 샘플링이 함께 바뀌어 원인 분리 불가) |
| U-Net++ → SegFormer-B2 | LG-E04 vs LG-E05 | −4.03pp (단일 run) | 손해 |
| U-Net++ → DINOv2 / DINOv3-B / DINOv3-L ViT | LG-E06, LG-E11, LG-E12 vs LG-E05 | −20.93pp / −13.71pp / −12.99pp (단일 run) | 손해 (patch 단위 feature가 얇은 상에 너무 거침) |
| 인코더 B3 → B4 | LG-E07 vs LG-E05 | −0.09pp (단일 run) | 효과 없음 (단일 run) |
| 반전 TTA / B3+B4 앙상블 (tile 512) | LG-E08, LG-E09 vs LG-E05 | +0.24 / +0.61pp | 효과 있음 |
| SegFormer·DINOv3 모델을 앙상블에 추가 | LG-E10, LG-E13 vs LG-E09 | −0.09 / −0.30pp | 손해 |
| 무작위 배율 증강 0.7–1.4 / 1.0–1.6 | LG-E14, LG-E15 vs LG-E05 | −0.34pp / −1.81pp (단일 run) | 손해 |
| 추론 타일 512 → 384 / 320 | LG-E16 | −0.25 / −0.49pp | 손해 |
| Lovász 추가 (3 seed) | LG-E18 vs LG-E19 | +0.15pp (Welch p=0.203) | 효과 없음 |
| tile 768 + 8방향 TTA + Si 입자 투표 | LG-E24 vs LG-E09 | +0.48pp | 효과 있음 |
| 클래스별 logit bias 보정 | LG-E20 vs LG-E09 | −0.04pp (leave-one-out) | 효과 없음 |
| 이미 학습된 모델 7개 앙상블 | LG-E27 vs LG-E24 | +0.05pp | 효과 없음 (비용 대비) |
| 원본 해상도 branch (fullres) | LG-E28 vs LG-E19 | +0.70pp (Welch p=0.002) | 효과 있음 |
| 경계 label relaxation / JML+BLS / Kervadec | LG-E29, LG-E30, LG-E34 vs LG-E19 | −1.96pp / −0.99pp / −0.77pp | 손해 |
| JML / 계면 가중 CE / Boundary DoU / Lovász | LG-E31, LG-E32, LG-E33, LG-E36 vs LG-E19 | +0.28pp / +0.05pp / −0.10pp / +0.08pp | 효과 없음 |
| Active Boundary Loss 추가 | LG-E35 vs LG-E36 | +0.03pp (Welch p=0.850) | 효과 없음 |
| fullres 변형: B4 / 채널 64 / Lovász | LG-E40, LG-E41, LG-E42 vs LG-E28 | −0.44pp / −0.19pp / −0.29pp | 효과 없음 또는 손해 |
| 밝기 증강 약하게 (aug_color) | LG-E50 vs LG-E28 | +0.49pp (Welch p=0.014) | 효과 있음 |
| 입력 2배 확대 | LG-E52 vs LG-E28 | +0.49pp (Welch p=0.006) | 효과 있음 |
| 입력 1.5배 / D4 균등 / 얇은 조직 샘플링 / CutMix | LG-E51, LG-E49, LG-E53, LG-E54 vs LG-E28 | −0.66pp / −0.51pp / −0.40pp / −0.28pp | 손해 또는 효과 없음 |
| 밝기 약하게 + 2배 확대 (color_scale2) | LG-E62 vs LG-E28 | +1.02pp (Welch p=0.004) | 효과 있음 (두 이득이 거의 그대로 더해짐) |
| 앙상블에 약한 모델(B4 등) 섞기 | LG-E64, LG-E65, LG-E66 vs LG-E63 | −0.14 / −0.24 / −0.29pp | 손해 |
| 에폭 60 → 120 / 200 (aug_color) | LG-E71, LG-E72 vs LG-E50 + LG-E70 | +0.36pp / +0.43pp; val 반쪽으로 고르고 나머지로 채점하면 +0.58 / +0.57pp | 효과 있음 |
| Primary Si 입자 이동 | LG-E79 vs LG-E50 + LG-E70 | +0.25pp (Welch p=0.005) | 효과 있음 |
| 조명 필드 / Fourier 혼합 | LG-E74, LG-E76 vs LG-E50 + LG-E70 | +0.07pp / +0.07pp; 앙상블 +0.21 / +0.19pp | 약한 긍정 |
| 초점 흐림 / 명암 표준화 | LG-E75, LG-E77 vs LG-E50 + LG-E70 | −0.22pp / −0.24pp | 손해 또는 효과 없음 |
| 작은 스캔만 2배 확대 | LG-E78 vs LG-E50 + LG-E70; LG-E86 | +0.11pp (Welch p=0.503) (학습 중 검증); 최종 추론 설정 단일 모델 +0.46pp (p=0.03); 6개 앙상블 +0.59pp | 효과 있음 (최종 추론 설정에서만 보임) |
| 배율 2 위에 증강 3종 / + 작은 스캔 4배 | LG-E88, LG-E87 vs LG-E62 | +0.08pp / +0.13pp | 효과 없음 (앙상블 −0.07 ~ +0.06pp) |
| 성격이 다른 두 구성 앙상블 (60 에폭) | LG-E93 vs LG-E63 | +0.30pp | 효과 있음 |
| 에폭 60 → 200 (배율 2 조합) | LG-E95, LG-E96 vs LG-E88, LG-E87 | +0.18pp / +0.32pp; 3개 앙상블 +0.19 / +0.41pp | 효과 있음 |
| 200 에폭 두 구성 앙상블 | LG-E99 vs LG-E93 | +0.23pp | 효과 있음 (10/3 당시 최고 80.14%) |
| 최고 구성에 60 에폭 모델 6개 추가 | LG-E100 vs LG-E99 | −0.05pp | 효과 없음 |

## 6. 확인하지 못한 것
- Test·리더보드 점수: 어떤 실험도 기록이 없음. test 예측 PNG만 있음: preds/test_ensemble (LG-E09 구성), preds/test_ensemble_d4_vote (LG-E24 구성), preds/test_color_scale2 (LG-E62 3개, 최종 추론 설정). 최신 12모델 최고 구성의 test 예측은 아직 만들지 않음.
- GPU 메모리: 파일 기록 없음. 세션 중 nvidia-smi로 본 값은 배율 2 run 3개를 동시에 돌릴 때 합계 약 90GB.
- 학습 시간: 로그의 에폭 시간 합이라 검증 시간이 빠지고, 여러 run을 동시에 돌린 실험은 단독 실행보다 길게 잡힘. 총 경과 시간은 기록이 없어 확인 불가.
- 파일 기록 없이 세션 출력으로만 남은 점수 (근거 칸에 '세션 출력'이라고 표시; 같은 checkpoint로 eval_ensembles.py를 다시 돌리면 재현 가능): LG-E58 (78.91%), LG-E69 (79.28 / 79.52%), LG-E73 (78.82 / 79.39 / 79.46%와 클래스별 IoU), LG-E90 (79.86%), LG-E92 (79.70%), LG-E94 (79.90%), LG-E89·LG-E91·LG-E93의 클래스별 IoU 일부.
- 날짜를 '(추정)'으로 쓴 항목: 추론·앙상블 실험은 실행 시각 기록이 없어 사용한 모델의 학습 종료 시각이나 메모 날짜로 추정. README 결과표에만 있는 실험: LG-E08, LG-E10, LG-E13.
- 클래스별 IoU: 학습 실험은 val_per_image.json(train_v2.py)이나 로그(train.py)에서, 그 클래스가 있는 이미지만 평균한 값 (두 스크립트 모두 per_image_miou 사용). 앙상블 실험의 값은 문서 표나 세션 출력에서 옮긴 것이라, 같은 방식으로 계산됐는지는 eval_ensembles.py 기준으로 추정 (추정: 같은 per_image_miou를 씀).
- 기준 조건 불일치: LG-E05의 인코더 lr은 1e-4 (runs/s2_micronet_unetpp/history.json → args.enc_lr), LG-E19 및 이후 run은 3e-4(기본값 = lr). 1라운드 비교에는 같은 조건의 LG-E19를 씀. LG-E07, LG-E40도 인코더 lr 1e-4.
- 같은 설정의 재실행: (LG-E17, LG-E36), (LG-E50, LG-E70)은 각각 같은 설정을 다른 날 학습한 것. 뒤쪽 쌍의 3개 앙상블은 78.91% / 78.82%로 0.09pp 차이가 남(앙상블 노이즈).
- 같은 실험으로 묶은 run 중 history.json의 args 키 수가 다른 것이 있음(이후 추가된 인자가 기본값으로 저장됨). 값이 같아 하나로 묶음: LG-E19, LG-E40, LG-E41, LG-E42.
- LG-E04의 args.pretrain=micronet은 SegFormer에서 쓰이지 않음 (SegFormer는 nvidia/mit-b2 가중치를 직접 불러옴).
- p 값은 seed 3개(일부 6개) Welch t-test라 검정력이 낮음. 모든 val 점수는 checkpoint를 같은 val로 골라 단일 모델 기준 0.5~0.8pp 낙관적 (split_half_check.py로 측정).
- 실패·중단된 학습 run: 기록상 없음 (runs/ 아래 모든 history.json이 설정 에폭까지 기록됨). 추론 중 tile 1024 반사 패딩 오류가 한 번 있었고 코드 수정 후 다시 잼(LG-E26).
