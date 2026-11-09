# 96×96 전체 영상 사람 검출 데이터

원본 CrowdHuman 4,370장을 종횡비 보존 resize + letterbox(114)로 변환했다.
이전 `person96` 패치 분류 데이터와 별도의 전체 화면 검출 데이터다.

- `train/val/test.npz`: uint8 `x` `[N,1,96,96]`, `source_id`.
- `train/val/test.jsonl`: 모든 적격 bbox, ignore/음성 대상, 원본 크기와 좌표 변환.
- `images/{train,val,test}/`: NPZ와 같은 grayscale PNG.
- `config.json`, `stats.json`: 기준과 실제 수량.

정답 bbox는 원본 80% 가시/유지 비율(사각형 면적 근사), 원본 최소 높이/너비 48/12px,
입력 최소 높이/너비 24/8px 기준이다. 좌표는 `[x1,y1,x2,y2]`, 우측·하단 exclusive다.
`ignore_boxes`에는 source ignore와 작은 사람이 포함되며, 최종 모델은 알려진 작은 사람에 대한
음성 heatmap supervision도 적용한다. 큰 사람의 가시성 미달은 `negative_people`이다.

분할은 기존 원본/dHash 그룹을 유지한다. 최종 수량은 train 3,496장/7,972명,
val 437장/1,043명, test 437장/1,060명이다. 원본 공식 validation subset에서 만든 자체 split이다.
CrowdHuman 이용 조건에 따라 교육·비상업 연구 목적의 로컬 사용만 하고 이미지를 재배포하지 않는다.

보고서: [`report/person_detector_training.md`](../../../report/person_detector_training.md).
