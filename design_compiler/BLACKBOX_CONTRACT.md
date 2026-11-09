# SRAM black-box 합성용 view 계약

사용자의 SRAM 블랙박스 합성 요청을 위해 현재 dual-clock RTL의 메모리 경계만 명시적 instance로 치환한다. production `rtl/`와 `orig/`는 변경하지 않는다. 합성 언어는 Verilog-2005다.

## 구조

현재 top/capture를 그대로 복사하고, npu_core의6개 큰 SRAM과 bbox_overlay의2개 큰 SRAM을 `npu_sram_bb` instance로 옮긴다. Input frame만 pixel27MHz write/NPU100MHz read이고 나머지 read/write는100MHz다. 각 instance는 독립 read/write clock, write enable/address/data, read enable/address,1clock registered read data를 제공한다. 메모리 output register는 SRAM 경계 안에 포함되어 면적에서 제외된다.

제외되는 memory는 총220332bytes다. descriptor working registers, probability registers, 후보/kept/bbox bank, FSM/MAC/address generation/CDC/render 로직은 그대로 남긴다.

| instance | depth×width |
|---|---|
| u_npu/u_mem_input_frame |9216×8|
| u_npu/u_mem_activation |108288×8|
| u_npu/u_mem_weight |89820×8|
| u_npu/u_mem_bias |274×32|
| u_npu/u_mem_multiplier |274×32|
| u_npu/u_mem_descriptor |208×32|
| u_post/u_mem_lut |1536×16|
| u_post/u_mem_result |3456×16|

## 합성과 검증의 분리

DC에는 empty-body `npu_sram_bb.v`를 읽고8개 instance 및 reference를 dont_touch한다. 이 memory의 면적·setup/hold/clock-to-Q 모델은 없으므로 결과는 **연산/제어 표준셀 면적·타이밍**만 의미한다.

검증에는 같은 interface의 `verify/npu_sram_bb.v`를 대신 읽는다. 원래와 같은 registered read latency1, read enable hold, array no-reset을 보존한다. write/read collision에 의존하지 않는 기존 ownership 계약을 유지한다. Lint 후 기존 실제 image TB를 복사해 wrapper memory array 경로만 바꿔 full3456 logits/최종 bbox/RGB 검사와 input/parameter SRAM 전수 검사를 실행한다.

## 생성·감사

`python3 design_compiler/scripts/prepare_blackbox_view.py`가 원본의 정확한 memory-only 구간을 확인한 뒤 치환한다. 예상 선언/구간이 맞지 않으면 중단하며 원본 SHA256와 생성 SHA256, memory 목록을 manifest에 남긴다. Array에 인접한 control 로직은 원본과 동일하다. Black-box 합성용 source와 behavioral 검증용 source를 동시에 읽지 않는다.

## 실행 검증 결과

- strict Verilog-2005 lint(`-Wall -Wno-UNUSED`) PASS. 검증용 memory model 파일은 module명과 동일한 `verify/npu_sram_bb.v`다.
- Verilator/C++ 대표 실영상 `frame_20261021_085042_877.jpg` 실행 PASS:3456개 logits byte 전부 기존 정수 golden과 일치, NPU/post40,984,186cycles.
- 기존 TB의 weight/bias/multiplier/descriptor SRAM 전수 비교, frame SRAM/copy SRAM 전수 비교, reset 복구, busy frame ownership, CDC와 sync/bypass/config 검사를 그대로 유지했다. C++의 wrapper memory hierarchy 경로만 바꿨다.
- 별도 fixed-point decoder와 bbox1개 `(29,37)-(48,86)`,score26158 비교 PASS. 최종9216개 RGB pixel과 빨간 테두리136pixel 비교 PASS.
- 결과: `blackbox_rtl/verify/case/simulation.log`, `blackbox_rtl/verify/verification.json`, `blackbox_rtl/verify/case/rtl/results/`.
- `blackbox_rtl/verify/memory_extraction.diff`에 원본과 view 차이를 보존했다. NPU arithmetic/FSM 및 RAM 부분 이후 전체 post 로직이 원본과 같음을 assert했다. source SHA256도 검증 후 동일하다. 최신 timing 분할 view의 근거는 `verify/verification.json`의 source manifest hash와 최종 실행의 `validation/blackbox_view_verification.json`에 있다.

다음 명령으로 검증 모델을 합성용 empty model 대신 읽어 빌드했다. 같은 module명 두 파일을 동시에 넣지 않는다.

```bash
verilator --cc --exe --build --trace --language 1364-2005 -Wall -Wno-UNUSED \
  --top-module person_detector_top --Mdir design_compiler/blackbox_rtl/verify/build \
  design_compiler/blackbox_rtl/verify/npu_sram_bb.v \
  design_compiler/blackbox_rtl/dvp_capture.v design_compiler/blackbox_rtl/npu_core.v \
  design_compiler/blackbox_rtl/bbox_overlay.v design_compiler/blackbox_rtl/person_detector_top.v \
  "$PWD/design_compiler/blackbox_rtl/verify/test_pipeline_blackbox_view.cpp" \
  -CFLAGS '-std=c++11 -O2' -o test
```

실행은 `blackbox_rtl/verify/case/`에서 빌드된 `test --trace`를 호출한다. copy한 case/rtl/generated 자료만 읽으며 생산 parameter 파일을 쓰지 않는다. 이 검증은 SRAM 경계 치환의 동작 일치이며 실제 empty memory를 넣은 gate-level 기능 검증이나 SRAM 물리 타이밍 검증은 아니다.
