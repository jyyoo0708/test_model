# Bbox 후처리·영상 overlay 계약 — 27MHz / 100MHz

## 인터페이스와 clock domain

합성용 소스는 plain Verilog-2005다. `bbox_overlay`는 다음 두 domain으로 동작한다.

| 영역 | clock / reset | 입력·출력 |
|---|---|---|
| 후처리 연산 | `clk_npu` 100MHz, `rst_npu_n` | NPU `i_result_*`, LUT `i_lut_*`, 설정 `i_cfg_*`/`o_cfg_rdata`, debug bbox `o_box_*`, `o_boxes_done`, `o_busy` |
| 영상 렌더링 | `clk_pixel` 27MHz, `rst_pixel_n` | `i_vsync`, `i_hsync`, `i_rgb` → `o_vsync`, `o_hsync`, `o_rgb` |

reset은 active-low 비동기 assert다. 상위에서 공통 reset을 두 domain에 동시에 assert하고 각 domain에서 해제를 동기화한다. 한쪽 domain만 독립적으로 reset하는 운용은 지원하지 않는다. reset 중 데이터 SRAM/박스 배열 자체는 지우지 않고 valid·count·제어 상태로 무효화한다.

## 후처리 연산

NPU는 channel-major signed INT8 6×24×24 결과를 전달한다. preload한 1536×16 LUT의 주소는 `channel*256+uint8(logit_byte)`다. 채널0~4는 sigmoid Q0.16, 채널5는 sqrt(sigmoid) Q0.16이다. 동기 LUT read와 valid/index pipeline을 통해 3456×16 result SRAM에 확률을 저장한다. 마지막 valid와 done이 겹치는 경우도 두 클록 drain으로 처리한다. 모든 SRAM read는 한 클록 latency이며 array reset은 없다.

후처리 timing 분할은 [TIMING_SLICE_POST.md](TIMING_SLICE_POST.md)에 정리했다. 상위 후보의 최소/최대 선택은 array read와 scalar 비교를 분리한 순차 scan이며 후보당2cycle,50개 scan과 commit은103cycle이다. NMS는 폭/높이→면적→합집합→threshold product→비교의5cycle로 분할했다. 동점에서는 최소 후보 교체 시 latest raster, 출력 시 earliest raster를 유지한다. SRAM·레지스터 주소·top 포트·CDC 계약은 유지한다.

100MHz FSM은 576개 cell을 훑어 6개 확률과 3×3 heatmap 이웃을 읽는다. peak3, score threshold를 통과한 후보 중 상위50개를 유지하고 raster-order tie break로 정렬한다. score는 `(heat_prob*quality_sqrt_prob)>>16`이다. 좌표는 center=`(cell+2*sigmoid(offset)-.5)*4`, size=`96*sigmoid(size)`이며 [0,96]으로 clip한 뒤 Q8 endpoint로 줄인다. 최소 clipped 크기를 확인하고 greedy NMS를 수행한다. NMS는 Q8 좌표의 면적에 대해 `intersection*65536 > union*IoU_q16`을 비교한다. 경계 조건에서는 floating NMS와 차이가 있을 수 있다.

`o_box_x1/y1`은 left/top의 floor, `o_box_x2/y2`는 right/bottom의 ceil-1인 inclusive 96×96 픽셀 좌표다. `o_box_score`는 Q0.16이며 `o_box_valid`인 100MHz 클록에서 유효하다. 모든 결과 출력 뒤 `o_boxes_done`이 100MHz 한 클록 pulse로 발생한다. 이는 pixel domain에 새 박스가 이미 게시되었다는 의미가 아니다.

## 설정 register

주소와 기본값은 기존 모델과 동일하며 모든 read/write는 NPU domain이다.

| byte offset | 의미 | 기본값 |
|---|---|---|
| `0x00` | confidence Q0.16 | 21299≈0.325 |
| `0x04` | IoU Q0.16 | 32768=0.5 |
| `0x08` | low8 최소 width, 다음8 최소 height | width8,height24 |
| `0x0c` | bit0 overlay enable, bit1 NMS enable | 3 |
| `0x10` | `[7:2]` 게시 완료 박스 수, bit1 pending, bit0 후처리 FSM busy | 0 |

threshold/IoU/최소 크기/NMS enable은 결과 완료 시 snapshot한다. overlay enable은 2FF를 거쳐 pixel domain에 전달하므로 변경이 보이기까지 최소 두 pixel edge가 필요하다. 안정적인 영상 전환을 위해 blanking에서 변경한다. status의 게시 완료 박스 수는 pixel acknowledgment가 NPU domain으로 되돌아온 뒤 갱신된다.

## 박스 집합의 CDC와 bank 소유권

작은 sensor-coordinate 박스 배열은 두 bank의 register 저장소다. NPU는 inactive bank만 쓰고 pixel renderer는 active bank만 읽는다. 출발 domain의 데이터 버스를 비트별 2FF로 통과시키는 방식은 사용하지 않는다.

1. reset 뒤 pixel active bank=0, NPU write bank=1, active count=0이다.
2. NPU는 결과의 모든 박스 좌표를 write bank에 기록한다. 마지막 write 이후 select/complete 단계를 거쳐 `publish_bank_q`, `publish_count_q`를 고정하고 `publish_req_toggle_q`를 반전한다.
3. 요청은 `publish_req_meta_q` → `publish_req_sync_q`의 `ASYNC_REG` 2FF를 통과한다. 이후 pixel domain에서 다음 VSYNC rising을 기다린다.
4. 해당 VSYNC에서 `active_bank_q`, `active_count_pixel_q`를 원자적으로 교체하고 `publish_ack_toggle_q`를 요청 값으로 바꾼다. 요청이 첫 VSYNC에 늦으면 그 프레임 전체는 이전 bank를 유지한다. 프레임 중간 교체는 없다.
5. ack는 `publish_ack_meta_q` → `publish_ack_sync_q`의 NPU-domain 2FF를 통과한다. NPU는 이를 관측한 뒤에만 pending을 내리고 write bank를 반전해 이전 active bank를 재사용한다.

`o_busy`는 FSM 처리 또는 ack 대기 pending 동안1이다. VSYNC가 오지 않으면 요청과 payload를 계속 유지한다. 게시할 박스가0개여도 동일 handshake로 empty bank를 게시해 이전 박스를 제거한다. 소스의 publish bank/count와 완료된 bank 내용은 ack까지 안정적으로 유지된다. 새 결과를 받는 조건은 idle이며 pending이 없어야 한다.

pixel 영상은 `o_boxes_done`보다 뒤의 VSYNC부터 새 박스를 사용한다. RGB888/VSYNC/HSYNC는 한 pixel clock register를 거친다. 센서 원본 좌표는 capture와 같은 round-half-up resize 크기 및 중앙 padding으로 역변환한다. 빨간색 한 픽셀 테두리만 덮고 다른 픽셀은 보존한다.

## 물리 CDC 제약과 검증 범위

요청/ack/overlay-enable 2FF 첫 stage로 들어가는 비동기 경로는 동기기 경로로 처리하고 `ASYNC_REG` 배치 의도를 유지한다. stage1→stage2의 같은-domain 경로는 정상 timing 분석 대상이다.

`publish_bank_q`, `publish_count_q`, `box_sensor_x1/y1/x2/y2_q`는 소유권 handshake로 유지하는 bundled-data 경로다. 두 clock은 명시적인 asynchronous group이며 일반 STA에서 cross-domain setup/hold를 제외한다. 데이터 settling은 별도 CDC/물리 검증으로 확인한다. DC의20ns settling audit은 일반 구현 SDC와 분리된 진단 view에만 적용한다. 목적지는 pixel active bank/count 및 RGB register까지의 render cone이다. Async 예외 아래에서 무시되는 max-delay를 유효한 물리 제약으로 간주하지 않는다.

두 domain의 reset 해제를 동기화하고 공통 assert를 유지하는 것이 handshake 초기화 전제다. 디지털 시뮬레이션은 metastability의 아날로그 동작이나 MTBF를 증명하지 않는다.

## 단위 검증 기록

lint를 먼저 실행해 correctness 경고가 없음을 확인한 뒤 C++ TB를 수정했다.

```bash
verilator --lint-only --language 1364-2005 -Wall -Wno-UNUSED --top-module bbox_overlay rtl/bbox_overlay.v
verilator --cc --exe --build --language 1364-2005 -Wall -Wno-UNUSED --top-module bbox_overlay --Mdir rtl/build/post_dual rtl/bbox_overlay.v "$PWD/rtl/tb/test_bbox_overlay.cpp" -CFLAGS '-std=c++11 -O2' -o test
rtl/build/post_dual/test 12345
rtl/build/post_dual/test 0
rtl/build/post_dual/test 499999
```

세 위상 모두 PASS. scheduler 단위는1/27ps이며 100MHz half-period=135000,27MHz half-period=500000으로 주파수를 정확히 표현한다. 위상 인수는 첫 pixel edge의 scheduler offset이다. NPU 결과는 NPU rising에서만, 영상은 pixel rising에서만 검사한다.

검증 항목:

- 동기 LUT 및 마지막 valid와 done이 겹치는 pipeline drain.
- 정확한 synthetic bbox 좌표·score 순서, 겹친 후보 NMS.
- 64개 독립 peak 중 점수 상위50개를 정확한 내림차순으로 출력. 동일 score 후보의 latest-raster 교체와 earliest-raster 출력도 별도 검사.
- 요청이 VSYNC보다 늦게 도착할 때 프레임 전체의 이전 bank 유지.
- 다음 VSYNC에서의 원자적 게시, ack를 받을 때까지 busy 유지.
- overlay enable CDC, 설정 write/readback, empty 결과 게시 및 이전 박스 제거.
- pending 상태에서 공통 reset 후 phantom 요청/박스가 생기지 않음.
- 각 위상에서 **8프레임,73728개 RGB 픽셀** 전수 비교. 총221184개 출력 픽셀.

Timing 분할 및 동점 검사를 추가한 최종 단위시험은 위상12345/0에서 NPU379756 rising, 위상499999에서 NPU379757 rising이고 세 위상 모두 Pixel102535 rising에서 완료했다. 이 수치는 단위 시험 전체의 시간이며 NPU 한 이미지 추론 cycle 수가 아니다. 큰 SRAM은 LUT3072+result6912=9984bytes로 기존과 같다. 후보/kept/dual-bank 센서 박스는 약1.8KB register이며 handshake 제어 register가 소량 추가된다. 실제 FPGA BRAM 배치·배선 후 timing/CDC 분석은 별도다.
