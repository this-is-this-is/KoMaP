# OOF logit fusion

EfficientNet-B3와 SE-ResNeXt-50의 4-class OOF 확률을 입력으로 받아 작은 residual
fusion head를 학습한다. 입력은 두 모델의 log-probability 8채널과 entropy 2채널이며,
출력은 두 확률의 산술평균 logit에 더하는 4채널 보정값이다. 마지막 층을 0으로
초기화하므로 학습 시작점은 정확히 단순 확률 평균이다.

모든 base 확률은 A-ID GroupKFold의 held-out 예측이어야 한다. Fusion 성능도 같은
5개 A-ID fold에서 교차적합한 held-out 결과 90장을 합쳐 평가하며, B3, SE, 단순 평균,
학습 fusion의 mIoU를 함께 기록한다. 최종 배포용 head만 이 비교가 끝난 뒤 OOF 90장
전체로 학습한다.

## 1. Base OOF 확률 생성

기존 B3 fold 0은 완료되어 있으면 자동으로 건너뛴다. 두 계열의 모든 fold를 학습한 뒤
동일한 D4 + 3-scale 설정으로 OOF 확률을 저장한다.

~~~bash
cd /home/user/Desktop/komap

./.conda/bin/python run_group_cv.py \
  --stage train \
  --recipes b3_aug3 seresnext50_aug3 \
  --seeds 0

./.conda/bin/python run_group_cv.py \
  --stage oof \
  --recipes b3_aug3 seresnext50_aug3 \
  --seeds 0 \
  --tile 768 --overlap 0.5 \
  --scale_tta 0.875 1 1.125
~~~

추가 seed가 있으면 --seeds 0 1 2로 생성한다. Fusion 입력에서는 같은 계열의
seed 확률을 먼저 평균한다. 모든 scores.json의 tile, overlap, D4, scale 설정이
일치하지 않으면 학습기는 중단한다.

## 2. Fusion 교차검증 및 최종 head 학습

~~~bash
./.conda/bin/python train_oof_fusion.py \
  --stage all \
  --b3_oof runs/group_cv/oof/b3_aug3 \
  --se_oof runs/group_cv/oof/seresnext50_aug3 \
  --folds_json runs/group_cv/manifests/folds.json \
  --out runs/oof_logit_fusion
~~~

주요 결과물:

- cv_results.json: 90장 held-out 기준 네 방법의 mIoU, 이미지별 결과, paired
  image/A-ID bootstrap 신뢰구간
- fold_0.pth ... fold_4.pth: fusion 교차검증 head
- fusion_final.pth: OOF 90장 전체로 학습한 배포용 head

cv_results.json에서 fusion이 단순 평균보다 개선되고 A-ID bootstrap 구간도 확인된
경우에만 최종 test 제출에 쓰는 것이 안전하다.

## 3. 90장 base 최종 학습 및 test 추론

먼저 각 base 계열을 90장 전체로 재학습한다. epoch는 fold별 best epoch의 중앙값으로
자동 선택된다.

~~~bash
./.conda/bin/python run_group_cv.py \
  --stage final \
  --recipes b3_aug3 seresnext50_aug3 \
  --seeds 0

./.conda/bin/python predict_logit_fusion.py \
  --b3_ckpt runs/group_cv/final/b3_aug3/seed_0/final.pth \
  --se_ckpt runs/group_cv/final/seresnext50_aug3/seed_0/final.pth \
  --fusion_ckpt runs/oof_logit_fusion/fusion_final.pth \
  --split test \
  --save_dir preds/test_oof_logit_fusion
~~~

추론기는 fusion checkpoint에 기록된 OOF tile/overlap/D4/scale 설정을 자동 적용한다.
따라서 학습과 test의 추론 분포가 달라지는 실수를 막는다. 여러 seed의 최종 base
모델을 사용하려면 각 --b3_ckpt, --se_ckpt 뒤에 checkpoint를 모두 나열한다.
