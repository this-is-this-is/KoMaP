# B3+SE 12-model teacher → HRNet-W18 student

## 구성

Teacher는 현재 최고 앙상블과 같은 12개 체크포인트다.

- EfficientNet-B3 `cs2_aug3` 200 epoch ×3
- EfficientNet-B3 `cs2_plus` 200 epoch ×3
- SE-ResNeXt-50 `cs2_aug3` 60 epoch ×3
- SE-ResNeXt-50 `cs2_aug3` 200 epoch ×3

12개 모델을 HRNet 학습 step마다 실행하지 않는다. 먼저 train 70장의 D4 확률을 원본 크기로
한 번만 계산해 `runs/kd_teacher_mixed12_train/*_prob.npz`에 저장한다. 이후 HRNet 데이터셋이
확률맵을 영상·마스크와 동일하게 crop/flip/rotate한다. 밝기·조명·Fourier 변환은 영상에만
적용된다. Primary-Si paste로 실제 픽셀이 바뀐 위치는 supervised loss에는 남기되 KD에서는
제외한다.

Student는 기존 HRNet 비교와 동일한 `HRNet-W18 + ImageNet + cs2_aug3`다. 기본 loss는:

```text
L = 0.5 CE + 0.5 Dice
  + 0.3 KL(teacher || student, T=2)
  + 0.1 KL_boundary(teacher || student, T=2)
```

teacher 최대 확률 0.5 미만인 픽셀은 KD에서 제외하고, 그 이상도 confidence에 비례해
가중한다. 캐시에는 float16 확률을 저장하지만 crop 직후 float32로 승격해 KL을 계산한다.

## 실행

```bash
cd /home/user/Desktop/komap

# 실제 작업 없이 teacher cache와 학습 명령 확인
./.conda/bin/python run_hrnet_kd.py --stage plan

# 12-model teacher 확률 70장 캐시
./.conda/bin/python run_hrnet_kd.py --stage cache

# 1-seed 화면 실험
./.conda/bin/python run_hrnet_kd.py --stage train --seeds 0

# 캐시 → 3 seeds → 기존 HRNet과 selection-free/최종 추론 비교
./.conda/bin/python run_hrnet_kd.py --stage all --seeds 0 1 2
```

기본 student 출력은 `runs/b_hrnet18_kd_mixed12_s{0,1,2}`다. 기존 HRNet 기준은
`runs/b_hrnet18_cs2_aug3_s{0,1,2}`이며, 3-seed 완료 후 `split_half_check.py`와
`compare_groups.py`가 자동 실행된다. 중단 후 같은 명령을 다시 실행하면 현재 teacher
체크포인트와 설정이 일치하는 완료 결과는 건너뛴다.

KD cache는 clean train scan에서 만든 soft target이다. `synth_dir`, CutMix,
context-paste처럼 캐시에 없는 내용을 삽입하는 증강과 함께 쓰면 학습 명령이 오류를 내도록
막았다. Primary-Si paste만 변경 픽셀을 KD mask에서 제외하는 방식으로 지원한다.
