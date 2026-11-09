# PersonDetector: 전체 영상에서 사람 bbox 검출

전체 영상을 96×96 grayscale로 letterbox해 여러 사람의 중심·크기를 예측한다.
기존 `models/person96/`는 패치 분류 모델이며, 이 디렉토리의 모델은 bbox 검출 모델이다.

- `model.py`: Conv backbone + upsample/skip 연결 + 중심 heatmap/offset/size head.
- `utils.py`: 정답 생성, loss, bbox decode/NMS, 원본 좌표 변환, 평가.
- `run02/`: 음성 supervision·이동 증강·초기화 보완 모델.
- `run01/`: 기본 학습 비교 모델. `source/`에 당시 코드를 보관했다.
- `run02_interrupted/`: 증강 배치 루프 오류를 수정하기 전 중단된 실행. 사용하지 않는다.
- `selected.json`: 최종 선택 run과 score threshold.

입력은 float32 `[B,1,96,96]`, 값 범위 `[0,1]`이다. TorchScript 원시 출력은
float32 `[B,5,24,24]`이며 `[heatmap_logit,offset_x,offset_y,size_w,size_h]` 순서다.
최종 bbox는 후처리를 거쳐 원본 영상의 `[x1,y1,x2,y2]` 좌표로 변환한다.

```bash
env -u LD_LIBRARY_PATH /tmp/person96-train-venv/bin/python scripts/infer_person_detector.py <원본이미지.jpg> --output output/person_detector/result_bbox.jpg
```

위 명령은 bbox 이미지와 같은 이름의 JSON을 저장한다. JSON에는 `person_present`, `count`,
원본 bbox 좌표와 score가 들어 있다. 사용 환경/의존성은 루트 `requirements-train.txt`를 참고한다.

구조도·학습·정량 평가·bbox 예시·한계는 [검출 모델 보고서](../../report/person_detector_training.md)에 있다.
모델은 float32이며, 정수 추론·RTL은 구현하지 않았다. 252,408 bytes SRAM 수치는 미래 int8 구성의 추정치다.
