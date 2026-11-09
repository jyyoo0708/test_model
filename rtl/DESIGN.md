# 구현 계약 — pixel27MHz / NPU100MHz

이번 사용자의 RTL 구현·합성·CDC 요청이 기존 AGENTS.md의 RTL 제외 지시를 대체한다. 합성 언어는 Verilog-2005이며 TB/모델 변환은 C++/Python이다. 원본 전체는 `../orig/`에 보존한다.

- top 직속 모듈은 `dvp_capture`, `npu_core`, `bbox_overlay` 세 개다.
- `clk_pixel`27MHz에서 DVP/capture/render, `clk_npu`100MHz에서 NPU/copy/post/model preload/config/debug를 실행한다.
- 공통 active-low reset의 assert는 비동기, release는 domain별2FF 동기화다. SRAM array는 지우지 않는다.
- DVP는 high VSYNC envelope와 high line-valid HSYNC, 매 clock 한 pixel이다. 첫 pixel 전·줄 사이·마지막 줄 후 최소 한 pixel blank가 필요하다.
- sensor default96×96이며 gray8/RGB888/병렬 RGB565를 parameter로 선택한다. nearest 축소와gray114 중앙padding 후 입력q=(gray*127+127)/255를 만든다.
- pixel 입력 frame SRAM을9216×8 dual-clock RAM으로 분리하고, 완료 요청 toggle→NPU2FF로 소유권을 넘긴다. NPU는18432clocks로 원래 activation 주소0~9215에 복사한다.
- capture는 완료된 frame을 연산 중 덮어쓰지 않는다. post done을toggle→pixel2FF로 전달받은 뒤 padding 준비를 재개한다. 연산 중 frame은 건너뛴다.
- 전체 학습 그래프와 INT8 weight/activation, INT32 bias/누산, 채널별 정수 multiplier·shift24 재양자화를 유지한다. 기존 parameter5개hex의 bit열은 바뀌지 않는다.
- raw 출력6×24×24 INT8은100MHz stream이다. post는peak3/top50/threshold/최소크기/NMS를 순차 실행한다.
- post는 완료된 inactive bbox bank/count를 고정하고 pixel request/ack handshake를 수행한다. pixel은 동기화된 요청 이후 VSYNC rising에서만 새 bank를 게시한다.
- RGB888과sync는 한 pixel clock register를 거치고 빨간 bbox 테두리를 그린다. 박스는 후속 bypass frame에 적용된다.
- `o_box_*`/`o_boxes_done`/`o_npu_done`은100MHz, `o_ready/o_busy/o_frame_error`는pixel domain이다. done은 서로 다른 처리 단계의 한clock pulse다.
- input frame read와 모든 SRAM read latency는1clock이다. 논리 SRAM220332bytes이며 후보/박스의 작은 register 저장소가 별도다.

구조/CDC 상세는 [DESIGN_DUALCLOCK.md](DESIGN_DUALCLOCK.md), 코어는 [DESIGN_NPU.md](DESIGN_NPU.md), 후처리는 [DESIGN_POST.md](DESIGN_POST.md), 주소/블록/Vivado 설정은 [VIVADO_HANDOFF.md](VIVADO_HANDOFF.md)에 있다.

검증 계약: 서로 다른 위상의 정확한27/100MHz clock으로3456개 INT8 byte, bbox 및9216개 RGB pixel을 기준과 비교한다. reset/프레임 소유권/busy drop/parameter preload/empty result/원자적 박스 게시를 검사하고 image/ 전체 결과를 보존한다. 디지털 simulation은 analog metastability나 실제 FPGA timing 통과를 증명하지 않는다.
