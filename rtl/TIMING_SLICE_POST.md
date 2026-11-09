# bbox 후처리 timing 경로 분할

## 설계 계약

기존 `bbox_overlay`의 포트, config 주소, SRAM 크기와 read latency, 입력 LUT/INT8 결과 해석, NMS 산술, 출력 bbox 좌표 및 순서를 유지한다. pixel27MHz/NPU100MHz와 bbox bank request/ack CDC 계약도 유지한다. 처리 cycle 수가 늘어나며 `o_box_valid`/`o_boxes_done`의 발생 시점은 달라진다. Busy 동안 새 추론을 시작하지 않는 기존 계약을 유지한다. 언어는 기존 plain Verilog-2005다.

기존 DC 매핑/DRC 결과에서 NPU domain의 최악 경로는 candidate score register feedback의49.25ns였다. 한 clock 내에서50개 후보의 최솟값/최댓값을 누적 선택하는 조합 연쇄를 제거한다. 타이밍 목표를 충족했는지는 수정 후 실제 합성 report로 별도 확인한다.

## 구조

- Candidate extrema 선택은 `p_scan_read`에서 한 후보의 valid/score/raster-position을 register에 읽고, 다음 `p_scan_compare`에서 scalar best register와 비교한다. 후보당2cycles다.
- 별도 `p_scan_finish`와 commit state를 통해 마지막 후보의 비교 결과까지 반영한 뒤 candidate bank를 갱신한다. 배열 mux와 비교 결과를50단계 연속 연결하지 않는다.
- Full table의 최솟값은 동일 score에서 raster position이 가장 큰 후보를 고른다. 새 후보 score가 이 최솟값보다 **클 때만** 교체한다.
- 출력 최댓값은 동일 score에서 raster position이 가장 작은 후보를 고른다. 유효 candidate만 선택하고 선택된 candidate를 무효화한 뒤 기존 dimension/NMS 검사로 전달한다.
- NMS는 intersection 및 box 폭/높이 register → area register → union register → IoU threshold product register → 비교/분기의5단계 FSM이다. 기존 `intersection*65536 > union*run_iou`의 엄격한 부등호를 유지한다.
- NMS 폭/높이는 Q8 endpoint 범위0..24576에 맞는15bits다. area30bits, union31bits, 최종 비교47bits를 유지한다. `current` box 및 `compare_q`는 NMS5단계 동안 고정한다.
- SRAM access process와 pixel render process는 변경하지 않았다. 새로운 register는 NPU domain 내부 state이며 CDC 경로를 추가하지 않는다.

## 지연 변화

기본50개 후보에서 scan은100cycles다. `p_select`부터 candidate 추출 또는 empty 판정까지103cycles로, 기존1cycle보다102cycles 증가한다. Full-table 교체 검사의 추가 지연도 scan/finish/commit에서102cycles다. 각 cell 이후 `p_next_cell`이1cycle 추가된다(총576cycles). 각 NMS comparison은 기존1cycle에서5cycles로4cycles 추가된다.

따라서 추가 후처리 cycle 수는 입력의 유효 peak 수, full-table 검사 횟수, 유효 candidate 수 및 NMS comparison 수에 따라 달라진다. NPU convolution cycle 수와 SRAM 크기는 바뀌지 않는다. 실제 이미지별 전체 지연과 FPS는 수정 후 회귀 결과를 사용한다.

## Lint 및 기능 검증

```bash
verilator --lint-only --language 1364-2005 -Wall -Wno-UNUSED \
  --top-module bbox_overlay rtl/bbox_overlay.v

verilator --cc --exe --build --language 1364-2005 -Wall -Wno-UNUSED \
  --top-module bbox_overlay -Mdir rtl/build/post_slice \
  rtl/bbox_overlay.v "$PWD/rtl/tb/test_bbox_overlay.cpp" \
  -CFLAGS '-std=c++11 -O2' -o test
rtl/build/post_slice/test 12345
rtl/build/post_slice/test 0
rtl/build/post_slice/test 499999
```

Lint PASS. 세 async phase 모두 PASS이며 각 실행에서73728 RGB pixel을 정확 비교했다. 검증에는 아래가 포함된다.

- overlap NMS accept/reject 및 bbox 좌표/score
- 64개 isolated peak에서 score 상위50개 출력
- equal-score63개와 마지막 stronger peak에서 최신 raster 최소값 교체 및 동일 score의 earliest-raster 출력 순서
- threshold/config readback, pixel enable 동기화
- empty 결과의 bank publication, VSYNC에서만 bank 적용, ack 전 bank 재사용 금지
- pending publication 중 공통 reset 및 reset 후 phantom publication 부재

세 phase의 NPU tick 수는379756/379756/379757이며, pixel tick 수는 모두102535다. 이는 네 번의 synthetic 처리와 여러 DVP frame 및 reset을 합친 TB 전체 tick 수이고, 단일 이미지 inference 지연이 아니다.

원본 동작 RTL은 `design_compiler/backups/pre_timing_slice_20261106_124914/rtl/`에 별도 보존했다. 프로젝트 원본 전체 백업 `orig/`는 변경하지 않는다.
