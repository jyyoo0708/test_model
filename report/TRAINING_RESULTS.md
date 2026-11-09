# 사람 검출 모델 개선 학습 결과

문서의 1, 2, 3, 4, 5, 7번을 적용해 교사와 학생을 floating point로 학습하고 추론했다. 모든 신규 데이터·코드·모델·로그·결과는 `advance_network` 안에 있다. 기존 자료는 복사본을 사용했다.

기존 모델 대비 독립 테스트 AP50은 48.76%→53.97%, AP75는 12.16%→22.40%로 높아졌다. 존재 여부 정확도는 75.29%→73.91%로 낮아졌다. 박스 검출 지표와 존재 판정 지표를 구분해서 사용해야 한다.

## 기존 독립 테스트 결과

원래 CrowdHuman test 437장을 유지했다. 체크포인트와 score threshold는 원래 CrowdHuman validation에서만 선택했고 test로 조정하지 않았다. AP는 score floor 0.01에서 계산했다. 존재 정확도는 threshold를 넘는 예측 bbox가 하나 이상 있는지로 계산하며 정답 ignore 영역으로 예측을 지우지 않는다.

| 모델 | 존재 정확도 | AP50 | AP75 | 검출 precision | 검출 recall |
|---|---:|---:|---:|---:|---:|
| 기존 run02 | 75.29% | 48.76% | 12.16% | 61.31% | 49.62% |
| 개선 학생 증류 없음 | 71.62% | 52.53% | 20.47% | 60.00% | 55.19% |
| 개선 학생 증류 적용 | 73.91% | 53.97% | 22.40% | 62.04% | 56.89% |

기존 모델은 복사한 TorchScript를 다시 추론했다. 기존 추론 기준인 최소 예측 폭 6·높이 20px, threshold 0.329를 유지했다. 새 학생은 최소 예측 폭 8·높이 24px, threshold 0.325를 사용한다. 정답 최소 크기는 모든 평가에서 폭 8·높이 24px다. 구조·학습·예측 크기 필터 차이가 함께 반영된 비교다.

## 추가 영상 평가

| 모델 | MOT17 10 존재 정확도 | AP50 | AP75 | 검출 recall |
|---|---:|---:|---:|---:|
| 기존 run02 | 79.87% | 60.92% | 14.05% | 64.86% |
| 개선 학생 증류 없음 | 80.52% | 61.99% | 25.27% | 81.08% |
| 개선 학생 증류 적용 | 81.82% | 69.30% | 26.10% | 82.43% |

추가 데이터는 [MOT17 공식 원본](https://motchallenge.net/data/MOT17/)에서 받았다. MOT17 02·04는 train, 09는 validation, 10은 test로 영상 전체를 분리했다. FRCNN 버전의 원본 이미지와 수작업 정답만 사용했으며 FRCNN 검출 결과는 학습에 사용하지 않았다. 프레임은 30장 간격으로 뽑고, 전체 화면과 고정 격자 타일 6개를 만들었다. 추가 train 385개 뷰는 원본 55프레임에서 생성했으며, 추가 validation은 원본 18프레임의 126개 뷰, test는 원본 22프레임의 154개 뷰다. 이번에 새로 다운로드한 데이터셋은 MOT17 한 종류다. 같은 프레임·영상에서 나온 타일은 같은 split에만 둔다. MOT17 test는 한 영상의 파생 이미지이므로 다양한 카메라에 대한 일반화 지표로 확대 해석하지 않는다.

```json
{
  "mot17": {
    "MOT17-02-FRCNN": {
      "images": 140,
      "people": 85,
      "positive_images": 57
    },
    "MOT17-04-FRCNN": {
      "images": 245,
      "people": 432,
      "positive_images": 179
    },
    "MOT17-09-FRCNN": {
      "images": 126,
      "people": 86,
      "positive_images": 50
    },
    "MOT17-10-FRCNN": {
      "images": 154,
      "people": 74,
      "positive_images": 57
    }
  },
  "splits": {
    "train": {
      "images": 3881,
      "people": 8489
    },
    "val": {
      "images": 563,
      "people": 1129
    },
    "test": {
      "images": 591,
      "people": 1134
    }
  }
}
```

추가 데이터의 가시율은 `max(0, official_visibility - (1-retained))`를 써서 보수적으로 판정한다. 기존 CrowdHuman은 원래 bbox 기반 가시율 정책을 유지한다. 실제 사용자 CCTV 10장과 test1·test2는 정답이 없는 정성 추론용이며 학습과 성능 지표에 사용하지 않았다. 정성 결과에서는 test2에 전신에 가까운 박스 한 개가 남았지만, 회의실 의자 주변 오검출과 일부 CCTV 사람 미검출도 남아 있다. 정성 결과 모음에서 이 오류들을 직접 확인할 수 있다.

## 적용 내용

1. 데이터 확대: MOT17 실제 감시·거리 영상, 전체 화면·타일의 사람 및 배경을 추가했다. 원래 CrowdHuman train/val/test 그룹은 유지했다.
2. 증강: scale 0.85~1.18, 이동 ±10px, 좌우 반전, 밝기·대비 0.7~1.3, gamma 0.8~1.2, 약한 blur/noise. 변환 후 가시율·유지율 80% 및 최소 크기를 다시 판정한다. 잘린 가시율은 제거 면적 전체가 가시 영역이었다고 보는 하한으로 계산한다.
3. 채널 확대: 12→24→32→48 backbone, 32→24 decoder. 입력 96×96 grayscale, 출력 24×24를 유지한다.
4. Head: shared 분류 경로와 별도 24→24 회귀 tower, 1채널 IoU quality head. 품질 정답은 gradient를 끊은 예측/GT IoU이며, 배경 품질 BCE도 학습한다. 최종 점수는 `sigmoid(heat)*sqrt(sigmoid(quality))`다.
5. 박스 손실·할당: smooth L1 offset/size와 DIoU를 사용한다. 기본 중심 셀 외에 표현 가능한 인접 셀에 회귀 정답을 주며 중심 셀 우선으로 충돌을 해결한다. Offset은 `2*sigmoid-0.5`, decode도 같은 표현을 쓴다. 분류 heatmap의 정점은 원래 중심 셀 하나에 둔다.
7. 증류: 학생의 모든 채널을 2배로 늘린 교사를 동일한 적격 정답으로 학습했다. 적격 GT 주변 heatmap·feature와 교사 quality>0.35인 할당 셀의 bbox를 모방한다. 교사는 동결하며 feature adapter 24→48은 학습에만 사용한다. Annotation 없는 영역에 pseudo-positive를 추가하지 않는다.

증류 없는 학생과 증류 학생은 같은 초기 학생 가중치·seed·증강 난수·batch 순서·epoch 상한·모델 선택 규칙을 사용했다. 단일 seed 비교이므로 seed에 따른 편차는 이 실험에서 측정하지 않았다. 6번 출력 격자 확대와 8번 후처리 공동 탐색은 적용하지 않았다. Score threshold 한 개의 validation 선택은 기존 학습 절차를 따른다.

## 학습 및 검증

- `teacher`: best epoch 61, 81 epochs, 673.7s.
- `student_supervised`: best epoch 98, 118 epochs, 906.3s.
- `student_kd`: best epoch 98, 118 epochs, 1071.0s.

- CPU, float32. 학생 MACs: 20,321,280, threads8 배치1 TorchScript 시간: 1.041ms.
- 새 좌표 표현·인접 할당·증강 정답 재판정·빈 정답 batch·유한 gradient·교사 동결 검사를 통과했다. `logs/behavior_checks.log` 참조.
- BN folding 최대 차이 2.7894974e-05, TorchScript 저장·재로드 최대 차이 0.
- AP75, edge/center·단일/다중·크기별 recall, bbox 중심/폭/높이 오차, 같은 GT의 중복 예측 수는 각 모델의 `results.json`에 있다. 중복 수는 단순 진단 집계다.

## SRAM 산정

Activation/weight int8, bias int32, 배치1, output stationary PE 레지스터 누산을 가정한다. 실제 int8 추론이나 RTL을 구현·검증한 결과는 아니다.

| 항목 | Bytes |
|---|---:|
| Activation bank A | 32,256 |
| Activation bank B | 32,256 |
| Weight int8 | 89,820 |
| Bias int32 | 1,096 |
| 네트워크 합계 | 155,428 |
| 출력 채널별 int32 multiplier + int8 shift 제안 | 1,370 |
| 위 계수 저장 제안까지 포함 | 156,798 |

두 bank의 레이어별 생존 텐서와 복사·재사용 가정은 `models/student_kd/memory.json`에 기록했다. Upsample은 concat 목적지에 바로 기록하고 사용이 끝난 tensor는 회수한다. 별도 int32 누산 SRAM은 없다. 정렬·실제 재양자화 형식·후처리 저장 구조는 미확정이므로 해당 비용까지 검증한 하드웨어 전체 SRAM이라고 주장하지 않는다. 위 합계는 문서 예산 175,640B와 상위 제약 300,000B보다 작다. 교사와 adapter는 배포에서 제외한다.

## 파일과 재실행

- 배포 학생: `models/student_kd/model_float32.ts`, `inference/config.json`
- 학습 checkpoint: `models/teacher/best.pt`, `models/student_supervised/best.pt`, `models/student_kd/best.pt`
- 정량 결과: 각 모델 `results.json`, `history.json`, `*_predictions.jsonl`
- 정성 추론: `inference/results/`, `report/inference_contact_sheet.jpg`
- 학습 곡선: `report/learning_curves.png`
- 증강 전후 정답: `report/augmentation_preview.jpg` (초록: 적격, 노랑: ignore, 빨강: 가시율/유지율 미달).
- 실행 로그: `logs/`, 실행 환경: `.venv/`, 임시·캐시: `tmp/`, `.cache/`

```bash
cd advance_network
./run_env.sh infer.py --input inference_input --output inference/results
./run_env.sh scripts/check_advanced.py
```

학습 재현 시 `scripts/train_advanced.py`를 쓰고 `--output`을 새 디렉토리로 지정한다. `config.json`에 seed·손실·source hash·데이터 설정 hash·교사 hash가 기록되어 있다. 학습된 TorchScript는 floating point이며 int8 양자화 정확도는 별도 평가 대상이다.
