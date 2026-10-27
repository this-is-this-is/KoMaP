# 현미경 외부 데이터 사전학습 실험

KoMaP valid/test 영상은 외부 사전학습에 넣지 않는다. 모든 downstream 비교는 현재 최종
추론 설정인 **8방향 D4 TTA + 원본 tile 768 + Si 입자 투표**로 수행한다. 외부 데이터의
클래스 이름을 Al-Si 4개 상에 대응시키지 않고, 표현 또는 경계 decoder만 전이한다.

## 구현한 세 경로

### 1. MatSSL ResNet-50

[MatSSL (ICCVW 2025)](https://github.com/aimat-ust/MatSSL)이 Aachen, UHCS,
MetalDAM으로 학습해 공개한 all-three checkpoint를 그대로 사용한다. 공식 고정 revision의
SHA-256을 검사하고 `backbone.*` 318개 tensor를 SMP ResNet-50 encoder에 `strict=True`로
넣는다. SSL projection/gating head와 중복 등록된 `layer*` 키는 사용하지 않는다.

- checkpoint: `dhp510/MatSSL`, revision `40465a494399de7cc21a16c056a9009a9c73d2c8`
- SHA-256: `3bac102385d019172afc67a04c688447750102be4bcc09eec54bede77badebe2`
- 실행: `train_v2.py --encoder resnet50 --pretrain matssl ...`

논문에서 MatSSL은 MetalDAM에서 ImageNet 사전학습 U-Net++보다 높았고, NASA EBC의 작은
학습 분할에서도 MicroNet보다 일관되게 높았다. KoMaP에서는 B3와 다른 ResNet-50의 오류
다양성이 기존 6-model ensemble에 도움이 되는지도 함께 잰다. 공식 MatSSL 저장소와 model
hub에는 명시적 라이선스 파일이 보이지 않으므로, 대회 제출 또는 외부 배포 전에는 사용
조건을 별도로 확인한다.

### 2. 동일 ROI cross-microscope 자기지도 학습

[Cross-Microscope Optical Steel dataset (Scientific Reports 2026)](https://zenodo.org/records/19851246)은
20개 시편의 같은 ROI를 Leica, Olympus LEXT, Zeiss 현미경과 여러 촬영 조건으로 반복 촬영한
5,327개 패치다. Zenodo 원본 MD5를 검사하고 XLSX를 읽어 228개 ROI 그룹을 만든다.

- CC BY 4.0, 5,327 images, 228 registered ROI groups
- 검증은 이미지/ROI 무작위 분할이 아니라 시편 18/2 분할이다.
- positive는 `같은 ROI + 같은 배율 + 다른 현미경/촬영 조건`을 우선한다. 20x를 50x 크기로
  올려 만든 비물리적 보간 무늬를 학습하지 않도록 같은 배율 쌍이 있으면 cross-scale 쌍을
  쓰지 않는다.
- MicroNet EfficientNet-B3 encoder에 ROI identity 기반 multi-positive supervised contrastive
  loss를 적용한다. 외부 ROI 75%, KoMaP **train 영상만** 25%로 샘플한다.
- 두 view는 같은 crop/D4 기하 변환을 받고, 독립적인 약한 밝기·대비·감마·blur/noise를 받는다.
  RGB 색조가 현미경 shortcut이 되지 않도록 grayscale 3채널로 만든다.
- steel의 Bainite/Martensite patch label은 KoMaP 상으로 변환하지 않는다.

출력 `cross_ssl/best_encoder.pth`는 `komap_encoder_v1` 포맷이며 downstream에서 572개
encoder tensor를 strict load한다.

### 3. NASA Ni-superalloy 지도 decoder 사전학습

[NASA pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models)의
Super1–4는 MIT 라이선스다. 네 폴더를 합치면 같은 영상이 반복되므로 고유한 23개 쌍만
사용한다. 공식 RGBA overlay에서 RGB blue를 secondary, RGB red를 tertiary, 나머지를
matrix로 정확히 복원한다.

- 18개 source-domain 영상으로 epoch를 학습하고, 5개 `different_test` OOD 영상의
  이미지별 macro mIoU로 epoch를 고른다.
- 같은 MicroNet B3 + U-Net++ + FullResSkip 전체를 선택된 epoch 수만큼 23개 모두로 처음부터
  다시 학습한다. 0.5 CE + 0.5 macro Dice, D4, 약한 gray photometric 변환을 쓴다.
- KoMaP 전이 때 `encoder/decoder/stem/fuse`를 exact load하고 3-class
  `segmentation_head.*`는 무조건 버린 뒤 4-class head를 새로 초기화한다.

## 자동 실험과 승격 규칙

`run_external_pretraining.py --stage all`은 다음을 순서대로 수행한다.

1. 원본 다운로드, checksum/commit/파일 수 검증, manifest 생성
2. cross-microscope SSL과 NASA decoder 사전학습
3. 각 방법을 `b_cs2_aug3`의 나머지 설정과 같게 60 epoch × 3 seed 화면 실험
4. epoch 선택 편향을 확인하는 validation split-half 검사
5. 기존 최종 6개 모델과 각 3개 모델을 D4/tile768/Si vote로 평가
6. 고정 equal-member 혼합이 기존 6개보다 +0.02 pp 이상이거나, 단독 3-seed ensemble이
   같은 60-epoch MicroNet 기준의 0.15 pp 이내인 최상위 방법만 200 epoch로 승격
7. 승자를 `cs2_aug3`와 `cs2_plus(800:4)` 각각 3 seed로 학습하고, 기존 6개와 새 6개 및
   두 ensemble의 혼합 결과를 저장

주요 산출물은 `runs/external_pretraining_suite/` 아래의 `screen_ensemble.json`,
`split_half.log`, `promotion_decision.json`, `final_<method>_ensemble.json`이다. 개별 KoMaP
checkpoint에는 외부 초기화 파일의 절대 경로, SHA-256, 실제로 읽은 tensor 수와 reset한
head가 `initialization.json`으로 함께 저장된다.

```bash
./.conda/bin/python run_external_pretraining.py --stage plan
./.conda/bin/python run_external_pretraining.py --stage all
cat runs/external_pretraining_queue/status.json
```
