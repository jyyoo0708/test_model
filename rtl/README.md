# 96×96 사람 검출 RTL — pixel27MHz / NPU100MHz

입력 영상은27MHz에서 받고 NPU와 bbox 후처리는100MHz에서 실행한다. 원본 단일클록 프로젝트 전체는 `../orig/`에 백업되어 있다. 최신 소스와 원본을 합성 source set에 섞지 않는다.

| top 직속 모듈 | 역할과 clock |
|---|---|
| `dvp_capture.v` / `u_capture` |27MHz DVP, gray/축소/양자화, 입력 frame SRAM write |
| `npu_core.v` / `u_npu` | dual-clock 입력 SRAM→100MHz activation 복사,13개 convolution·skip·upsample·INT8 재양자화 |
| `bbox_overlay.v` / `u_post` |100MHz decode·top50·NMS, request/ack CDC,27MHz 빨간 박스 render |

[**VIVADO_HANDOFF.md**](VIVADO_HANDOFF.md)에 NPU 스펙, 전달 파일, 모든 preload/register/레이어 주소, BRAM 연결, 블록 다이어그램, 합성·CDC 설정과 FPS 계산을 정리했다. 상세 계약은 [DESIGN.md](DESIGN.md), [DESIGN_DUALCLOCK.md](DESIGN_DUALCLOCK.md), [DESIGN_NPU.md](DESIGN_NPU.md), [DESIGN_POST.md](DESIGN_POST.md)를 참고한다.

SRAM을 제외한 ASIC 표준셀 매핑/DRC 결과는 [DC 보고서](../design_compiler/blackbox_synthesis_result.md)에 있다. 후보 순차 scan/NMS5단계 분할과 최적화 합성 후 표준셀 면적0.086322mm², NPU100MHz setup slack+4.81ns, pixel27MHz+29.45ns, setup/hold 위반0개다. SRAM 포함 물리 timing은 별도이며 [수정·검증 기록](../design_compiler/timing_slice_result.md)을 참고한다. Vivado/BRAM 구현 결과와 구분한다.

## 연결과 clock domain

- top 입력 `clk_pixel`=27MHz, `clk_npu`=100MHz. PLL/MMCM은 외부에서 제공한다.
- `rst_n`은 두 domain에 함께 비동기 assert하며 top이 각 clock에서2FF 동기 release한다. 한 domain만 독립 reset하지 않는다.
- DVP `i_vsync/i_hsync/i_data`, 출력 `o_vsync/o_hsync/o_rgb`, `o_ready/o_busy/o_frame_error`는 pixel domain이다.
- `i_load_*`, `i_cfg_*`, `o_cfg_rdata`, `o_result_*`, `o_npu_done`, `o_box_*`, `o_boxes_done`은 NPU domain이다. I2C/CPU 어댑터가100MHz에 맞춰야 한다.
- `i_vsync` high는 frame envelope, `i_hsync` high 한 clock은 한 pixel이다. VSYNC rising과 첫 pixel 사이, 줄 사이, 마지막 줄 이후에는 최소 한 pixel clock의 HSYNC low를 둔다.
- top parameter는 `p_sensor_width/p_sensor_height`(기본96), `p_input_type`(0=gray8,1=RGB888,2=병렬 RGB565)이다. byte 직렬 RGB565는 입력 어댑터에서 조립한다.

입력 buffer는27MHz write/100MHz read의9216×8 BRAM이다. frame 완료 toggle을 동기화한 뒤 NPU가 복사한다. capture는 post 완료가 CDC로 돌아오기 전까지 buffer를 덮어쓰지 않는다. bbox도 완료된 inactive bank를 고정하고 request/ack를 전달해 다음 VSYNC에서만 교체한다. 자세한 소유권·물리 제약은 handoff 문서에 있다.

## 모델과 SRAM preload

모델 파라미터는 weight89,820개 +bias274개=90,094개다. 입력은 signed INT8, bias/누산은 INT32, 재양자화 중간값은 INT64다. 입력 scale=1/127, q=(gray*127+127)/255이며 BN folding과 output-channel별 weight scale을 사용한다.

`generated/`의 다음 파일을 **100MHz에서** 첫 추론 전에 preload한다. 주소는 byte 주소가 아닌 element index다. design-side 초기화는 없다.

| space | 파일 | 개수 | 폭 |
|---:|---|---:|---:|
|0|`weights.hex`|89820|8|
|1|`bias.hex`|274|32|
|2|`descriptors.hex`|208|32|
|3|`requant.hex`|274|32|
|4|`post_lut.hex`|1536|16|

float scale과 tensor 주소는 `generated/manifest.json`, quantization 근거는 `quantization_report.json`에 있다. hardware scale은 정수 multiplier+고정shift24와 LUT로 표현한다. 두 클록 변경으로 이 다섯 hex의 내용은 바뀌지 않았다.

## I2C용 설정 register

I2C slave는 사용자가 추가한다. 아래 offset은 `i_cfg_addr` 값이며 preload space와 별도 주소 공간이다. write는100MHz 한 clock의 `i_cfg_we`, read는 `o_cfg_rdata` 조합 출력이다.

| offset | 의미 | reset |
|---|---|---|
|`0x00`|confidence Q0.16|21299≈0.325|
|`0x04`|NMS IoU Q0.16|32768=0.5|
|`0x08`|low8 최소 폭/다음8 최소 높이,96좌표|`0x1808`|
|`0x0c`|bit0 overlay enable/bit1 NMS enable|3|
|`0x10`|RO `[7:2]` active bbox 수/bit1 pending/bit0 post FSM busy|0|

실행 설정은 결과 완료 시 snapshot한다. overlay enable은2FF로 pixel domain에 전달한다. 원본 single-clock 버전의 즉시 적용과 달리 최소 두 pixel edge가 필요하다.

## 출력과 성능

`o_npu_done` 후100MHz에서 후처리를 진행한다. 유효 박스마다 `o_box_valid`와96×96 inclusive 좌표/score를 출력하고 모두 끝나면 `o_boxes_done` pulse가 나온다. 최대50개이며0개여도 done이 나온다. 새 박스는 pixel 쪽 요청 동기화 이후의 VSYNC에서 적용한다. 영상은 계속 bypass하므로 추론 프레임과 박스를 그리는 프레임에 시간 차이가 있다.

NPU start→done은40,965,537 NPU clocks(입력 복사18,432 포함),100MHz에서409.655ms, 코어 상한2.4411FPS다. 대표 통합 TB 후처리 포함 구간은 약409.834ms다. capture·padding9216 pixel clocks·센서 VSYNC 대기로 실효 FPS는 더 낮다. sensor 영상 FPS와 추론 갱신 FPS는 다르다.100MHz routed timing은 아직 검증하지 않았다.

SRAM은 **220,332bytes**(NPU210,348 +post9,984), 후보/박스 register는 약1.8KB 별도다. 실제 FPGA BRAM capacity 반올림은 Vivado report에서 확인한다. 모든 read latency는1이며 추가 RAM output pipeline은 controller 수정 없이 넣으면 안 된다.

## 회귀검증

프로젝트 루트에서 실행한다. 현재 실행 방식은 Verilator+C++이며 xsim/보드 실행은 아니다.

```bash
python3 rtl/tools/run_tests.py --trace --all-images
# 이미 최신 TB를 빌드한 경우 전체 실영상만:
env -u LD_LIBRARY_PATH .venv/bin/python rtl/tools/test_all_images.py --jobs 3 --trace
# 단일 이미지:
env -u LD_LIBRARY_PATH .venv/bin/python rtl/tools/test_image.py image/frame_20261021_091348_677.jpg --phase 17321 --trace
```

정수 기준 모델과 raw3456byte, bbox decode/NMS, 영상9216pixel을 비교한다. reset 복구·busy frame drop·parameter SRAM과 frame/copy SRAM의 내용도 검사한다. post 단위 TB는 세 clock 위상에서221184 RGB 픽셀을 검사한다. 새 C++ bus stimulus는 clock edge 전에 settle eval을 수행해 계층 입력 race를 방지한다.

`results/<image_stem>/`에 이미지별 PNG/JSON이 저장되고, 전체 요약은 `results/dualclock_image_summary.json`, 모음은 `results/dualclock_contact_sheet.png`다. default 대표 결과는 `results/rtl_overlay.png`다. waveform은 해당 case의 `build/tb_pipeline.vcd`에 최종 영상 구간만 기록한다. 예전 optional SV TB는 `.sv.singleclock_reference`로 보존했으며 실행 대상은 최신 C++ TB다.

실영상 검증은 JPG를 기존 LANCZOS 전처리로96×96 변환해 DVP에 넣는다. 원본 고해상도 센서의 nearest scaler 경로 전체 통합 검증은 별도다. 전체 데이터셋 INT8 AP/정확도도 미평가다.

## 합성 상태

원본 단일27MHz의 실제 DC 실행은 analyze 통과 후 NPU elaboration timeout1800초(exit124)으로 종료됐다. **전체 netlist·면적·타이밍 PASS가 아니다.** [합성 결과 기록](../design_compiler/synthesis_result.md)에 로그·환경·판정을 남겼다.

Vivado는 현재 환경에 없어 실행하지 않았다. 최신 `vivado_synth.tcl`은 두 clock 제약과 `constraints/dualclock_cdc.tcl`을 사용한다. 실제 부품에서 BRAM mapping, synchronizer·held-bundle CDC, timing/DRC를 확인해야 한다.
