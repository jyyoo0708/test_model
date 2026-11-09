# Design Compiler 합성 자료

최신 두 클록(pixel27MHz/NPU100MHz) RTL에서 SRAM8개,220332bytes를 블랙박스로 제외한 표준셀 최적화 합성을 완료했다. [결과·조건·재실행 방법](blackbox_synthesis_result.md)을 먼저 확인한다.

| 항목 | 결과 |
|---|---:|
| 표준셀 면적 (SRAM 제외) |86321.545900µm² (0.086322mm²) |
| 표준셀 수 |82828 |
| NPU100MHz setup slack |+4.81ns, 충족 |
| pixel27MHz setup slack |+29.45ns |
| 최악 hold slack |+0.00ns(반올림), 위반0 |

`clk_pixel`과 `clk_npu`를 명시적인 asynchronous group으로 선언해 재합성했다. [비동기 제약 결과](async_clock_result.md)와 [clock groups](clock_groups.rpt)를 참고한다. 양방향 cross-domain STA는 No paths이며 bbox20ns는 일반 구현 제약에서 제거했다. 별도 settling audit으로만 보존했다.

후보 선택을 순차 scan으로, NMS를5단계로 분할하고 `compile` 최적화와 `set_fix_hold`를 실행했다. Setup/hold 위반은0개다. SRAM 모델 부재로 min-capacitance 경고103개는 남아 있으며 SRAM 포함 전체 timing closure는 아니다. [수정·회귀·전후 비교](timing_slice_result.md)를 참고한다. 배치·배선 및 Vivado 합성은 수행하지 않았다.

- [area](area.rpt), [전체 setup](timing.rpt), [NPU setup](timing_npu.rpt), [pixel setup](timing_pixel.rpt), [hold](timing_hold.rpt), [constraints](constraints.rpt), [QoR](qor.rpt)
- [독립 netlist/면적 감사](blackbox_synthesis_status.json)
- [SRAM 추출 및 기능 검증 계약](BLACKBOX_CONTRACT.md)
- 최종 netlist/DDC/SDC/로그와 소스 snapshot은 `latest_blackbox_run.txt`의 실행 폴더에 있다.
- [이전 원본 단일27MHz 실행 기록](synthesis_result.md): 큰 SRAM을 포함한 elaboration timeout 기록으로 보존했다.

동작 RTL의 bbox 후처리를 timing 분할했다. 수정 직전 RTL은 `backups/pre_timing_slice_20261106_124914/rtl/`에 보존했고 `orig/` 전체 백업은 변경하지 않았다. `blackbox_rtl/`은 ASIC 합성용 별도 view이며 Vivado BRAM 구현용 소스를 대체하지 않는다.
