# Timing 분할 및 재합성 결과

아래 표는 RTL 분할 직후 실행(`125118`)의 기록이다. 이후 두 clock을 명시적인 asynchronous group으로 선언해 재합성했다. **최신 수치는 [비동기 제약 결과](async_clock_result.md)**이며, 일반 CDC STA는 No paths다. 아래 bbox20ns 행은 최신 구현 setup 판정에 포함하지 않는다.

bbox 후보 비교 연쇄를 순차 scan으로 바꾸고 NMS 계산을5단계로 나눴다. 기존 pixel27MHz/NPU100MHz 조건에서 최적화 합성 후 **모든 setup 그룹 slack이 양수이며 setup/hold 위반은0개**다.

| 항목 | 수정 전 | 수정 후 |
|---|---:|---:|
| NPU100MHz setup slack |−39.34ns|+4.82ns|
| pixel27MHz setup slack |+31.39ns|+29.45ns|
| SRAM read 경계 |+4.24ns|+3.94ns|
| SRAM NPU 입력 경계 |+4.12ns|+3.48ns|
| SRAM pixel write 경계 |+35.77ns|+35.75ns|
| 고정 bbox CDC |+18.31ns|+18.23ns|
| 최악 보고 hold slack |−0.09ns|+0.00ns(반올림), 위반0|
| 표준셀 면적,SRAM 제외 |135795.582412µm²|85688.342029µm²|
| 표준셀 수 |131700|83647|
| 대표 frame 종료→boxes_done |40983406cycles|40984186cycles|

수정 전은 `compile -only_design_rule`, 수정 후는 기본 `compile` 최적화 및 `set_fix_hold` 결과다. 따라서 면적 개선 전체를 RTL 분할만의 효과로 분리해 해석하면 안 된다. Clock period, uncertainty, I/O, CDC 예외 및 SRAM 경계 예산은 동일하게 유지했고 hold 보정을 활성화했다.

## 코드와 동작

- [bbox_overlay.v](../rtl/bbox_overlay.v):50개 후보의 조합 최소/최대 연쇄를 array-read→scalar-compare 순차 scan으로 변경했다. 후보당2cycle,50개 scan 및 commit103cycle이다.
- NMS는 width/height→area→union→IoU product→compare의5cycle이다. 동점에서는 최소 교체 시 latest raster, 최대 출력 시 earliest raster를 유지한다.
- Top 포트·config 주소·SRAM 주소/용량·CDC·pixel bypass/render는 유지했다. NPU 연산 코어의40965537cycles와 parameter hex는 동일하다.
- [구조·검증 계약](../rtl/TIMING_SLICE_POST.md), [후처리 계약](../rtl/DESIGN_POST.md), [Vivado 인계](../rtl/VIVADO_HANDOFF.md).
- 수정 직전 소스: `backups/pre_timing_slice_20261106_124914/rtl/`. 프로젝트 전체 원본 `../orig/`는 수정하지 않았다.

## 검증

`python3 rtl/tools/run_tests.py --trace --all-images` PASS. strict Verilog2005 lint, capture3종, NPU 정수출력3456byte 및40965537cycle 검사, post3위상, 전체12개 실영상에서41472 logits와110592 RGB pixel이 일치했다. 추가 post 시험에서64개 peak의 상위50개 및 동점 교체/출력 순서, NMS accept/reject, empty/reset/config, bank publication/ack를 확인했다. 세 위상에서221184 RGB pixel을 검사했다.

SRAM 경계를 분리한 behavioral view도 대표 실영상3456 logits와9216 RGB pixel이 일치했다. 합성 최종 netlist에는8개 empty SRAM boundary가 유지됐고 표준셀83,647개를 독립 집계한 Liberty 면적이 `area.rpt`와 일치했다. Gate-level 등가성 검증은 실행하지 않았다.

대표 시간은409.84186ms,연산 구간 처리율 상한2.439966FPS다. 기존보다7.8µs 늘었다. 센서 캡처·padding 준비·다음VSYNC 대기는 추가된다.

## 리포트와 이미지

- [최신 area](area.rpt), [전체 setup](timing.rpt), [NPU setup](timing_npu.rpt), [pixel setup](timing_pixel.rpt), [hold](timing_hold.rpt), [constraints](constraints.rpt), [QoR](qor.rpt)
- [합성 조건·경고 상세](blackbox_synthesis_result.md), [감사 JSON](blackbox_synthesis_status.json)
- [12개 결과 이미지 모음](../rtl/results/dualclock_contact_sheet.png)
- [요청 이미지 frame_20261021_091348_677 결과](../rtl/results/frame_20261021_091348_677/rtl_overlay.png)
- [전체 이미지 검증 JSON](../rtl/results/dualclock_image_summary.json), [source hash](../rtl/results/dualclock_source_manifest.json)

이번 결과는 Samsung28LPP LVT SS0.95V−25°C의 표준셀 합성 결과이며 SRAM220332bytes의 내부 면적·timing arc는 제외한다. Macro 입력 부하가 없어 min-capacitance 경고103개가 남아 있다. 실제 SRAM Liberty와 배치·배선·CTS/Vivado BRAM을 연결한 물리 timing closure는 별도 확인해야 한다.
