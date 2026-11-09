# 두 클록 RTL 최종 검증 결과

pixel27MHz / NPU·후처리100MHz의 비동기 시뮬레이션에서 image/의12개 이미지 모두 통과했다. 후보 순차 scan/NMS5단계 분할 후 다시 실행한 결과다. Frozen INT8 parameter5개 파일은 orig/와 byte 단위로 동일하며 수정 전후12개 이미지의 bbox 좌표·점수·순서가 일치했다.

| 항목 | 결과 |
|---|---|
| 전체 실영상 |12/12 PASS, 실패0|
| NPU raw INT8 비교 |41,472bytes 전부 일치|
| 실영상 overlay RGB 비교 |110,592pixels 전부 일치|
| post 단위시험 |3위상,221,184pixels PASS|
| NPU 단위시험 |3456/3456bytes,40,965,537cycles PASS|
| clock/reset/소유권 |비동기 위상, 동시 edge, copy/compute reset abort, busy frame drop, bank request/ack PASS|
| parameter/input 검사 |weight/bias/multiplier/descriptor SRAM과 frame/copy SRAM 전수 비교 PASS|
| RTL lint |Verilog-2005 -Wall, UNUSED만 제외, correctness 경고 없음|
| 논리 SRAM |220,332bytes, 후보/박스 register 약1.8KB 별도|
| Vivado/물리 CDC·STA |미실행|
| 최신 ASIC DC 합성 | 후보/NMS 분할 및 최적화 후 NPU setup+4.81ns,pixel+29.45ns,setup/hold 위반0; SRAM 제외 |
| 원본 ASIC DC 합성 |analyze PASS, NPU elaboration timeout1800s(exit124), mapped area/timing 미생성|

이미지별 결과는 다음과 같다. bbox는96×96 inclusive 좌표이고 점수는Q0.16/65536이다. 검출0개도 정수 기준 모델과 일치한 결과이며 실제 dataset AP를 평가한 것은 아니다.

| 이미지 | bbox 수 | 박스(x1,y1,x2,y2; score) | NPU+post cycles |
|---|---:|---|---:|
| frame_20261021_085042_877.jpg | 1 | (29,37,48,86; 0.3991) | 40984186 |
| frame_20261021_085653_208.jpg | 1 | (41,25,60,72; 0.3541) | 40984186 |
| frame_20261021_091348_677.jpg | 1 | (33,31,53,82; 0.5015) | 40984186 |
| frame_20261021_113200_030.jpg | 0 | 없음 | 40984081 |
| frame_20261021_123336_503.jpg | 2 | (45,7,57,44; 0.6234) / (56,7,75,56; 0.5146) | 40984296 |
| frame_20261021_124417_138.jpg | 2 | (26,13,42,49; 0.3311) / (42,22,65,69; 0.3255) | 40984296 |
| frame_20261021_125545_552.jpg | 1 | (51,31,71,81; 0.5346) | 40984186 |
| frame_20261023_113720_582.jpg | 0 | 없음 | 40984081 |
| frame_20261023_134306_888.jpg | 0 | 없음 | 40984081 |
| frame_20261026_100635_512.jpg | 1 | (55,45,69,79; 0.3564) | 40984186 |
| test1.jfif | 1 | (57,46,75,85; 0.3620) | 40984186 |
| test2.jfif | 1 | (38,21,55,71; 0.4785) | 40984186 |

## 산출물과 재현

- 이미지 모음: [dualclock_contact_sheet.png](dualclock_contact_sheet.png)
- 전체 수치: [dualclock_image_summary.json](dualclock_image_summary.json)
- 최종 source/parameter hash: [dualclock_source_manifest.json](dualclock_source_manifest.json)
- 최신 timing 분할 합성: [timing_slice_result.md](../../design_compiler/timing_slice_result.md)
- 원본 합성 판정: [synthesis_result.md](../../design_compiler/synthesis_result.md)
- 이미지별 PNG/PPM/JSON: 각 image_stem 디렉터리
- 원본 전체 백업: ../../orig/

재실행: 프로젝트 루트에서 `python3 rtl/tools/run_tests.py --trace --all-images`.

27MHz/100MHz는 RTL 시뮬레이션에 실제로 구동한 주파수다. NPU의40965537cycles는100MHz409.65537ms, 처리율 상한2.441076FPS다. 대표 NPU+post 구간은40984186cycles(409.84186ms,2.439966FPS 상한)이며 센서 캡처·padding 준비·다음VSYNC 대기는 추가다. 실제 FPGA의100MHz 달성 여부는 별도 Vivado implementation과 CDC/STA 보고서가 필요하다.

초기 위상 시험의 weight[0] 누락은 clock edge 전에 새 C++ TB 입력을 settle하지 않은 문제였다. driver를 수정하고 SRAM 전수 검사를 추가한 뒤12개 전부 다시 실행한 최종 결과다. RTL datapath를 바꾸거나 parameter를 재calibration해 맞춘 것이 아니다.
