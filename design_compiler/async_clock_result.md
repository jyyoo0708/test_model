# 비동기 clock group 적용 및 재합성

`clk_pixel`27MHz와 `clk_npu`100MHz를 서로 다른 asynchronous group으로 선언한 뒤 일반 `compile` 및 hold fixing을 다시 실행했다.

```tcl
create_clock -name clk_pixel -period 37.037037037 [get_ports clk_pixel]
create_clock -name clk_npu -period 10.0 [get_ports clk_npu]
set_clock_groups -name pixel_npu_async -asynchronous \
    -group [get_clocks clk_pixel] -group [get_clocks clk_npu]
```

일반 구현 SDC/DDC/netlist와 setup/hold 리포트는 이 비동기 조건에서 생성했다. NPU↔pixel 경로는 setup/hold에서 제외되며 각 domain 내부 및 synchronizer stage1→stage2는 정상 timing 대상이다. 이전의 첫 synchronizer pin별 false-path 방식에서 명시적인 clock-group 방식으로 변경했다.

| 항목 | 결과 |
|---|---:|
| NPU domain100MHz setup slack |+4.81ns|
| Pixel domain27MHz setup slack |+29.45ns|
| SRAM read NPU 경계,9ns 가정 |+3.94ns|
| SRAM NPU 입력 경계,9ns 가정 |+3.48ns|
| SRAM pixel write,36ns 가정 |+35.75ns|
| NPU→pixel 일반 STA |No paths|
| Pixel→NPU 일반 STA |No paths|
| Setup / hold 위반 |0 / 0|
| 최악 보고 hold slack |+0.00ns(반올림)|
| 표준셀 면적,SRAM 제외 |86321.545900µm² (0.086322mm²)|
| 표준셀 수 |82828|

[Clock groups](clock_groups.rpt), [NPU→pixel](timing_async_npu_to_pixel.rpt), [pixel→NPU](timing_async_pixel_to_npu.rpt), [NPU setup](timing_npu.rpt), [pixel setup](timing_pixel.rpt), [hold](timing_hold.rpt), [area](area.rpt).

## Bbox20ns의 구분

20ns는 clock 주파수도, NPU 연산 주기도 아니다. 이전에는 고정 bbox bundle의 전달 지연 예산을 일반 STA에 포함했다. 이번에는 **최종 구현 SDC에서 bbox20ns를 제거**했다.

요청/ack로 유지되는 bbox bus의 물리 settling은 async 예외만으로 증명되지 않으므로 별도 진단으로 보존했다. 구현 파일과 일반 리포트를 먼저 저장한 뒤, 메모리상의 진단 view에서만 async group을 잠시 제거하고20ns max-delay를 적용했다. 이 view는 재합성하거나 구현 DDC/SDC로 저장하지 않았다. [별도 settling audit](bbox_settling_audit.rpt)의 slack은+18.23ns이며 일반 setup/hold 판정에 포함하지 않는다. 실제 SRAM/배선 후 settling 및 CDC는 별도 검증 대상이다.

## 소스와 검증

기능 RTL은 직전12개 이미지 회귀를 통과한 소스와 byte 단위로 동일하다. 따라서40984186cycles의 대표 NPU+post 지연과 출력 bbox/이미지 결과는 동일하다. 변경한 것은 DC와 Vivado 제약 및 문서다. Vivado 생성 clock.xdc도 synthesis 전에 asynchronous grouping을 읽는다. Tcl mock으로 순서를 점검했으며 실제 Vivado 실행은 하지 않았다.

DC 종료코드0,8개 SRAM boundary 유지, 표준셀별 수량×Liberty 면적 독립 집계와 area report 일치를 확인했다. 상세는 [합성 보고서](blackbox_synthesis_result.md), [감사 JSON](blackbox_synthesis_status.json) 및 최신 실행의 `validation/async_constraint_change.json`에 있다.

SRAM220332bytes의 내부 면적/timing arc는 제외했고 macro 입력 cap 부재에 따른 min-capacitance 경고103개는 남아 있다. 이는 전체 SRAM 포함 물리 signoff 결과가 아니다. 일반 timing closure와 별도 CDC/settling 검증을 구분한다.
