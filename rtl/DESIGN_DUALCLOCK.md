# 두 클록 구현 계약과 구조

변경 기준은 `orig/rtl/`의 단일27MHz 버전이다. 모델 parameter와 양자화/descriptor 내용은 변경하지 않는다.

## 인터페이스와 소유권 계약

- top은 `clk_pixel`27MHz, `clk_npu`100MHz, 공통 active-low `rst_n`을 받는다. 두 클록은 `set_clock_groups -asynchronous`의 서로 다른 group으로 선언한다. 도메인 간 setup/hold는 제외하고 각 domain 내부 및 synchronizer stage1→stage2는 정상 timing 대상이다.
- reset은 두 도메인에 함께 비동기 assert하고 각 도메인의2FF로 동기 release한다. 한 도메인만 독립 reset하는 기능은 제공하지 않는다.
- DVP, capture, 영상 overlay와 `o_ready/o_busy/o_frame_error`는 pixel 도메인이다.
- `i_load_*`, `i_cfg_*`, `o_cfg_rdata`, raw logits, bbox debug stream 및 두 done 출력은 NPU 도메인이다. 외부 I2C/CPU wrapper는100MHz에 맞추거나 별도 CDC를 구현한다.
- frame capture 완료 pulse를 pixel request toggle로 바꾸고100MHz2FF+edge detection으로 NPU start를 생성한다.
- frame SRAM9216×8은 pixel write / NPU read의 dual-clock RAM이다. NPU가 input을 기존 activation SRAM에 복사한 뒤 원래 그래프를 실행한다. activation SRAM에는 여전히100MHz read/write만 존재한다.
- capture는 요청 후 `WAIT`에서 입력 SRAM write를 멈춘다. post boxes_done을100MHz toggle→pixel2FF로 전달받을 때까지 write 소유권을 반환하지 않는다. 개별 INT8 byte에 synchronizer를 넣지 않는다.
- post 연산과 결과 메모리는100MHz다. 완료된 inactive bbox bank/count를 고정하고 request toggle을 pixel2FF로 보낸다. pixel은 다음 VSYNC에서만 bank/count를 채택한다. ack toggle이 NPU2FF로 돌아오기 전까지 해당 bank/payload를 바꾸지 않는다.
- overlay enable은 level2FF CDC를 거치며 나머지 실행 설정은 NPU 영역에서 snapshot한다.
- top 직속 instance는 기존3개를 유지한다. CDC 제어 register는 top과 post 내부에 있다.

## 타이밍·메모리

입력 frame copy는18432 NPU clocks다. 검증된 NPU start→done40965537clocks,100MHz 약409.65537ms. 논리 SRAM220332bytes(기존211116+입력9216)이며 작은 후보·박스 register가 별도다. 실제 수치는 완료된 회귀검증 JSON을 기준으로 한다.

## 검증 목표

서로 위상이 다른 정확한27/100MHz event clock으로3456개 NPU byte와9216개 output pixel을 비교한다. frame busy drop, reset 후 handshake 재초기화, request/ack 지연, frame 중간 박스 교체 금지, threshold/NMS/빈 결과/top50을 검사한다. `image/` 전체를 frozen model로 실행하고 이미지별 output PNG와 JSON을 저장한다.

2FF/toggle 검증은 디지털 프로토콜 검증이며 아날로그 metastability를 시뮬레이션하지 않는다. implementation에서는 synchronizer 배치와 bundled bbox bus의 source→destination 지연 제약을 적용하고 CDC/timing 보고서로 확인해야 한다.

최종 검증: image/12개 전부 PASS,41472 raw bytes 및110592 RGB pixels 비교. 상세는 [DUALCLOCK_VALIDATION.md](results/DUALCLOCK_VALIDATION.md)에 있다.
