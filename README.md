# Advanced person detector

96×96 grayscale 사람 검출 모델. 데이터·학습·추론 파일은 이 디렉토리 안에 둔다.

## Git 저장 범위와 실행 준비

저장소에는 코드, 설정, 문서, 성능 요약을 포함한다. 학습·검증 데이터,
원본 및 파생 이미지, 모델 가중치와 TorchScript, 가상환경, 로그, 백업,
시뮬레이션·합성 실행 산출물은 `.gitignore`로 제외하며 로컬 파일은 유지한다.
문서에 표시된 이미지·백업·실행 산출물 링크 중 일부는 로컬 작업 경로다.
새로 clone한 뒤에는 가중치를 별도로 준비하거나 학습해야 추론할 수 있다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt
bash scripts/download_data.sh
```

다운로드 스크립트의 고정된 CrowdHuman 소스와 저장된 SHA256SUMS로 원본을
준비한다. 데이터 사용 조건을 확인하고 원본·파생 이미지는 재배포하지 않는다.
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
