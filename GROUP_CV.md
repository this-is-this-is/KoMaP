# GroupKFold · 멀티스케일 · OOF 라벨 감사

세 기능은 기존 최고 학습 레시피를 바꾸지 않고, 평가 누수와 최종 데이터 활용을 개선하는
별도 파이프라인으로 구현되어 있다.

## 1. A-ID 기준 5-fold와 90장 최종 학습

파일명의 첫 `A###`를 그룹으로 사용한다. 기존 train 70장과 valid 20장을 먼저 합친 뒤
동일 A-ID의 스캔은 항상 같은 fold에 둔다. 현재 데이터는 90장, 71개 A-ID이며 각 이미지는
정확히 한 번 OOF validation이 된다.

```bash
# manifest와 실행할 명령만 확인한다. GPU 학습은 시작하지 않는다.
./.conda/bin/python run_group_cv.py --stage plan

# 단계별 실행
./.conda/bin/python run_group_cv.py --stage prepare
./.conda/bin/python run_group_cv.py --stage train --seeds 0 1 2
./.conda/bin/python run_group_cv.py --stage oof --seeds 0 1 2
./.conda/bin/python run_group_cv.py --stage audit --seeds 0 1 2
./.conda/bin/python run_group_cv.py --stage final --seeds 0 1 2

# 위 단계를 순서대로 한 번에 실행
./.conda/bin/python run_group_cv.py --stage all --seeds 0 1 2
```

기본 recipe는 `b3_aug3`이다. 다음처럼 기존 최종 모델의 세 가지 상보 구성을 함께 만들 수
있다. 작업은 GPU 메모리 충돌을 피하려고 순차 실행하며, 완료된 결과는 자동으로 건너뛴다.

```bash
./.conda/bin/python run_group_cv.py --stage all \
  --recipes b3_aug3 b3_plus seresnext50_aug3 --seeds 0 1 2
```

fold마다 `best.pth`를 고르고, 각 fold의 최고 epoch 중앙값으로 90장 전체를 고정 budget
재학습한다. 이 마지막 학습에는 validation/early stopping이 없으며 `final.pth`가 생성된다.
epoch를 사전에 고정하려면 `--final_epochs 200`을 쓴다. 기본 출력은 `runs/group_cv/`이다.

## 2. 멀티스케일 확률 앙상블

`predict.py --scale_tta`는 체크포인트별로 여러 상대 배율에서 추론한 **확률**을 먼저
평균한다. 그 뒤 모델 간 확률 평균과 Si 입자 투표를 적용한다. `1`만 주면 기존 추론과 같다.

```bash
./.conda/bin/python predict.py --ckpt runs/model/best.pth \
  --split valid --tta --d4 --tile 768 --si_vote \
  --scale_tta 0.875 1 1.125 --save_dir runs/model_multiscale
```

작은 사전 선언 grid는 다음 명령으로 비교한다. 첫 설정 대비 이미지별 paired bootstrap
95% 구간도 `summary.json`에 기록한다.

```bash
./.conda/bin/python evaluate_multiscale.py --ckpt runs/model/best.pth \
  --scale_sets 1 0.875,1,1.125 --tiles 768 896
```

5-fold OOF 생성 때도 같은 배율을 쓰려면 `run_group_cv.py`에
`--scale_tta 0.875 1 1.125`를 넘긴다.

## 3. OOF 예측 기반 라벨 감사

`predict.py --save_prob`가 저장한 float16 확률을 `audit_oof_labels.py`가 읽는다. 기본적으로
정답 Al3Ni인데 멀리 떨어진 Eutectic Si로 확신하는 영역과 그 반대 방향을 연결요소 단위로
순위화한다. 여러 seed/recipe의 OOF가 있으면 평균 확률과 모델 간 동의율을 함께 사용한다.

```bash
./.conda/bin/python audit_oof_labels.py \
  --oof_dirs runs/group_cv/oof --out runs/group_cv/label_audit
```

결과는 `audit.csv`, `audit.json`, `panels/*.png`이다. 도구는 마스크를 절대 수정하지 않는다.
패널을 사람이 확인한 뒤 CSV의 `decision`, `review_note`만 채우도록 되어 있다. 같은 이미지로
학습한 예측을 쓰면 라벨 오류가 아니라 암기를 잡게 되므로 반드시 OOF 확률만 넣어야 한다.
