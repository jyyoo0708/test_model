# 96×96 grayscale 사람 분류 데이터

기준: 루트 `AGENTS.md`. 준비 방법·annotation 기준·한계·이용 조건은
[`report/data_preparation.md`](../report/data_preparation.md)를 참고한다.

- 원본: `raw/crowdhuman/CrowdHuman_val.zip`, `annotation_val.odgt`
- 학습 데이터: `processed/person96/{train,val,test}.npz`
- 입력 PNG: `processed/person96/images/{train,val,test}/{person,non_person}/`
- 샘플 annotation: `processed/person96/{train,val,test}.jsonl`
- 원본 annotation 및 제외 사유: `processed/person96/source_annotations.jsonl`
- 수량: `processed/person96/stats.json`
- 검토용 이미지: `processed/person96/preview/`
- 검증: `../report/data_validation.json`

NPZ의 `x`는 uint8 `[N,1,96,96]`, `y`는 int64 `[N]`이다.
학습할 배치만 `x.astype(np.float32) / 255.0`으로 변환한다.
`y=1`은 bbox 기반 가시/유지 비율 ≥80%의 적격 사람,
`y=0`은 배경 또는 전신의 50% 이하만 담긴 부분 신체다.
실제 신체 노출 비율은 bbox로 근사하며 segmentation으로 측정하지 않았다.

공식 validation 4,370장을 자체 train/val/test로 나눈 초기 학습 데이터다.
공식 CrowdHuman 벤치마크의 train/val/test 구성과 다르다.
교육·비상업 연구 용도로 사용하고 원본/파생 이미지를 재배포하지 않는다.

루트에서 재검사:

```bash
python3 scripts/validate_data.py
```

재생성은 기존 결과를 덮어쓰지 않도록 다른 `--output` 경로를 사용한다.

```bash
bash scripts/download_data.sh
python3 scripts/prepare_data.py --output data/processed/person96_rebuild
python3 scripts/validate_data.py --data data/processed/person96_rebuild
```
