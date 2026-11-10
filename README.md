# Advanced person detector

96×96 grayscale 사람 검출 모델. 데이터·학습·추론 파일은 이 디렉토리 안에 둔다.

## 학습 데이터와 검증

- **학습 데이터**: CrowdHuman 공식 validation 4,370장을 자체 train/val/test로 분할하고, MOT17 영상 02·04를 학습에 추가했다. 최종 학습 데이터는 3,881개 이미지·뷰이며, 96×96 grayscale로 변환해 float32 교사·학생 증류 학습을 진행했다. 사람의 가시·유지 비율은 80% 이상, 입력 기준 최소 크기는 폭 8×높이 24px로 제한했다.
- **검증 방법**: 원본·유사 이미지 그룹과 MOT17 영상 단위로 분리해 데이터 누출을 방지했다. CrowdHuman 검증 437장에서 AP50으로 모델을 선택하고 검출 F1으로 threshold(0.325)를 정한 뒤, 별도 테스트인 CrowdHuman 437장과 MOT17 영상 10의 154개 뷰에서 평가했다(MOT17 영상 09는 별도 검증용).
- **결과**: Student KD의 사람 존재 정확도 / 박스 AP50은 CrowdHuman 테스트 **73.91% / 53.97%**, MOT17 테스트 **81.82% / 69.30%**다. 자세한 내용은 [학습 결과와 비교](report/TRAINING_RESULTS.md)를 참고한다.

## 네트워크 구조

현재 추론에 사용하는 Student KD 모델(`AdvancedDetector(width=1)`) 기준이다.
출력 크기는 `채널 × 높이 × 너비`이며 배치 크기는 1이다.
`ConvBlock`은 학습 시 Conv2D → BatchNorm → ReLU로 구성하며, 추론 시 BatchNorm을 Conv2D에 folding한다.

| 레이어 | 입력 | 연산 | 커널 / stride | 출력 크기 (C×H×W) |
| --- | --- | --- | --- | --- |
| input | grayscale 이미지 | 0~1 범위 float32 입력 | — | 1×96×96 |
| c1 | input | ConvBlock | 3×3 / 2 | 12×48×48 |
| c2 | c1 | ConvBlock | 3×3 / 2 | 24×24×24 |
| c3 | c2 | ConvBlock | 3×3 / 2 | 32×12×12 |
| c4 | c3 | ConvBlock | 3×3 / 2 | 48×6×6 |
| context | c4 | ConvBlock | 3×3 / 1 | 48×6×6 |
| up1 | context | Nearest upsample ×2 | — | 48×12×12 |
| concat1 | up1, c3 | 채널 방향 결합 | — | 80×12×12 |
| fuse1 | concat1 | ConvBlock | 3×3 / 1 | 32×12×12 |
| up2 | fuse1 | Nearest upsample ×2 | — | 32×24×24 |
| concat2 | up2, c2 | 채널 방향 결합 | — | 56×24×24 |
| fuse2 | concat2 | ConvBlock | 3×3 / 1 | 24×24×24 |
| head | fuse2 | ConvBlock | 3×3 / 1 | 24×24×24 |
| regression | head | ConvBlock | 3×3 / 1 | 24×24×24 |
| heatmap | head | Conv2D: 사람 중심 점수 logits | 1×1 / 1 | 1×24×24 |
| offset | regression | Conv2D: 중심 위치 보정 logits (x, y) | 1×1 / 1 | 2×24×24 |
| size | regression | Conv2D: 박스 크기 logits (w, h) | 1×1 / 1 | 2×24×24 |
| quality | regression | Conv2D: 박스 품질 logits | 1×1 / 1 | 1×24×24 |
| output | heatmap, offset, size, quality | 채널 방향 결합 | — | 6×24×24 |

모든 3×3 Conv2D는 padding=1, 1×1 Conv2D는 padding=0이다.
최종 출력 채널 순서는 `heatmap, offset_x, offset_y, width, height, quality`이며,
출력 stride는 4이다. Conv2D는 총 13개, 연산량은 이미지당 20,321,280 MACs이다.
구현은 [models/advanced_model.py](models/advanced_model.py),
구조 그림은 [network_structure.png](report/network_structure/network_structure.png)를 참고한다.

## Git 저장 범위와 실행 준비

저장소에는 코드, 설정, 문서, 성능 요약과 함께 테스트 입력(`image/`,
`inference_input/`), 리포트 그림(`report/`), 추론 결과(`result/`,
`inference/results/`), 기존 RTL 검증 결과(`rtl/results/`)를 포함한다.
학습·검증 데이터셋, 학습 체크포인트, 가상환경, 일반 로그, 백업,
빌드 및 합성 실행 디렉터리는 `.gitignore`로 제외하며 로컬 파일은 유지한다.
문서에 표시된 백업·합성 실행 산출물 링크 중 일부는 로컬 작업 경로다.
추론용 가중치를 포함한 `models/student_kd/model_float32.ts`는 Git 저장 대상이다.
학습용 `.pt` 체크포인트는 별도로 준비하거나 학습해야 한다.

다른 사람이 clone한 뒤 **이미지 추론만 실행**하려면 Python 3.10~3.12 환경에서
프로젝트 루트로 이동해 다음 순서로 실행한다. 아래 고정 의존성은 Linux x86_64 CPU 환경 기준이다.
학습 데이터셋 다운로드는 필요하지 않다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-inference.txt
python3 infer.py --input image --output result --threshold 0.325
```

자신의 이미지는 `image/`에 넣거나 `--input`으로 파일·폴더를 지정한다.
추론에는 `models/student_kd/model_float32.ts`, `inference/config.json`,
저장소의 추론 코드와 위 Python 의존성이 필요하다. TorchScript에 학습된
가중치와 네트워크가 포함되어 있어 별도 `best.pt` 파일은 필요하지 않다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt
bash scripts/download_data.sh
```

다운로드 스크립트의 고정된 CrowdHuman 소스와 저장된 SHA256SUMS로 원본을
준비한다. 데이터 사용 조건을 확인하고 학습·검증 데이터셋 원본·파생 이미지는 재배포하지 않는다.
데이터 준비는 `scripts/prepare_data.py`, `scripts/prepare_person_detector.py`,
추가 학습 데이터 준비는 `scripts/prepare_extra.py`를 참고한다.
각 스크립트의 `--help`로 입력·출력 경로를 확인한다.
학습 순서와 인자는 `scripts/pipeline.py`, 결과와 설정은
`report/TRAINING_RESULTS.md` 및 `models/*/config.json`에 기록되어 있다.
데이터 분할과 사람 가시 비율·최소 크기 기준은 준비 코드와
`data/` 하위 README를 함께 확인한다.

## 이미지 추론

`image/`에 입력 이미지를 넣고 실행한다.

```bash
cd advance_network
python3 infer.py --threshold 0.325
```

`python infer.py --threshold 0.4` 또는 `./run_env.sh infer.py --threshold 0.4`도 가능하다. 시스템 Python으로 실행하면 이 디렉토리의 `.venv`를 자동으로 사용한다. Threshold는 0~1이며 생략하면 `inference/config.json`의 기본값 0.325를 사용한다.

학습과 같은 전처리로 종횡비를 유지해 96×96으로 축소하고 남는 부분은 회색 114로 채운다. 단일 채널 grayscale 입력을 float32 `[1,1,96,96]`, 0~1 범위로 변환해 연산한다.

결과는 `result/`에 저장한다. 원본 확장자를 이름에 남겨 같은 이름의 JPG와 PNG도 구분한다.

- `원본파일명_input96.png`: 실제 추론 입력, 96×96 grayscale.
- `원본파일명_bbox96.png`: 위 입력 픽셀 위에 흑백 박스를 그린 결과, 96×96 단일 채널 grayscale.
- `원본파일명.json`: 박스 좌표·점수·threshold.
- `summary.json`: 전체 처리 결과와 오류.

입력 한 장 또는 별도 결과 경로도 지정할 수 있다. 결과 경로는 `advance_network` 내부로 제한한다.

```bash
python3 infer.py --input image/test2.jfif --output result_single --threshold 0.4
```

모델은 `models/student_kd/model_float32.ts`, 설정은 `inference/config.json`이다.

## RTL 구현 및 이미지 시뮬레이션

[rtl/README.md](rtl/README.md)에 DVP 입력 → INT8 NPU → bbox/NMS/빨간 박스 출력 구현과 preload/register 연결 방법을 정리했다. 현재 RTL은 **pixel27MHz / NPU·후처리100MHz**이며 프레임 및 박스 전달에 CDC를 적용한다. 전체 원본은 `orig/`에 백업했다.

프로젝트 루트에서 `python3 rtl/tools/run_tests.py --trace --all-images`로 검증한다. 이미지별 결과는 `rtl/results/<image_stem>/`, 모음은 `rtl/results/dualclock_contact_sheet.png`에 저장한다. 스펙·주소·블록 구성·FPS·Vivado 설정은 [VIVADO_HANDOFF.md](rtl/VIVADO_HANDOFF.md), 최신 SRAM 블랙박스 DC area/timing은 [합성 결과](design_compiler/blackbox_synthesis_result.md)에 있다(후보/NMS timing 분할 후 DC 최적화 합성 완료, NPU100MHz setup slack+4.81ns; SRAM 내부 제외). 원본 단일클록 timeout 기록은 [이전 결과](design_compiler/synthesis_result.md)에 보존했다.

## 학습 자료

- [학습 결과와 비교](report/TRAINING_RESULTS.md)
- [기존 학습 후 추론 모음](report/inference_contact_sheet.jpg)
- 학습 동작 검사: `./run_env.sh scripts/check_advanced.py`
- `baseline/`: 기존 모델·코드 복사본.
