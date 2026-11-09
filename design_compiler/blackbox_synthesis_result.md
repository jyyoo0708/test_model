# SRAM 블랙박스 ASIC 합성 결과 — pixel27MHz / NPU100MHz

**표준셀 매핑과 netlist/area/timing 리포트 생성은 완료했다. SRAM 내부 면적과 timing arc는 제외했다.**

**현재 setup 목표는 충족했다(아래 가정 범위).**

## 합성 결과

| 항목 | 값 |
|---|---|
| Top | `person_detector_top`, 현재 두 클록 RTL의 메모리 경계 분리 view |
| 단계 | analyze / elaborate / link / compile / report 완료, 종료 코드0 |
| **표준셀 합산 면적** | **86,321.545900µm²** (약0.086322mm²) |
| 독립 netlist 면적 집계 | 86,321.547000µm², DC와 일치 |
| 표준셀 수 | 82,828개 (계층을 재귀적으로 계산) |
| SRAM |8개 instance 유지,220332bytes의 내부 회로 제외 |
| 작은 후보/박스·제어 register | 표준셀 면적에 포함 |
| 포함되지 않은 기능 | 외부 I2C slave·보드 PHY·PLL/MMCM은 원래 top에 없으며 추가하지 않음 |
| 배치/배선/CTS | 미실행, die/core 면적 또는 routed timing이 아님 |

## Timing

| 경로 그룹 | 최악 setup slack(ns) | 판정 |
|---|---:|---|
| 100MHz 레지스터/I/O | +4.81 | 충족 |
| 27MHz 레지스터/I/O | +29.45 | 충족 |
| SRAM read-output→NPU 로직 (9ns 가정) | +3.94 | 충족 |
| NPU→SRAM 입력 경계 (주소/en/data,9ns 가정) | +3.48 | 충족 |
| pixel→input-frame write (36ns 가정) | +35.75 | 충족 |

최악 보고 hold slack: **+0.00ns**. 현재 보고된 hold 경로는 위반이 없다.

전체 위반은 `constraints.rpt`, 그룹별 상세 경로는 `timing_*.rpt`, 전체 QoR은 `qor.rpt`에 보존했다. Setup slack이 양수여도 hold/전기적 제약/미제약 경로가 남으면 전체 closure로 표시하지 않는다.

## 비동기 clock 관계와 별도 payload 검사

`clk_pixel`과 `clk_npu`는 `set_clock_groups -name pixel_npu_async -asynchronous`의 서로 다른 group이다. 일반 합성·SDC·DDC와 setup/hold 리포트에서 두 clock 사이 경로는 제외했다. 각 domain 내부 및 synchronizer stage1→stage2는 해당 clock 주기로 분석한다.

고정 bbox bundle의20ns는 주파수나 일반 STA required time이 아니다. 최종 구현 artifact를 저장한 뒤 메모리상의 별도 진단 view에서만 clock grouping을 잠시 제거해 settling budget을 확인했다. 별도 진단 최악 slack은+18.23ns다. 이 값은 위 setup 목표 판정에 포함하지 않는다.

[clock groups](results/blackbox_dualclock_20261106_131129/clock_groups.rpt), [NPU→pixel No paths](results/blackbox_dualclock_20261106_131129/timing_async_npu_to_pixel.rpt), [pixel→NPU No paths](results/blackbox_dualclock_20261106_131129/timing_async_pixel_to_npu.rpt), [별도 settling audit](results/blackbox_dualclock_20261106_131129/bbox_settling_audit.rpt).

## 사용 조건과 해석 범위

- 도구: Design Compiler I-2013.12-SP4, 실행 명령 `compile`. 기본 매핑 최적화를 수행했다.
- 라이선스: 사용법의 `27000@uxftools`에 연결했다. 공유 서버를 새로 시작하거나 종료하지 않았다.
- 라이브러리: Samsung28nm LPP,9-track,LVT, SS0.950V,-25°C.
- DB: `/tools/lib/ln28lpp/primitive/ln28lpp_sc_9t_base_lvt_c130_FE-Common_sec190802_0202/LIBERTY/sc9_cmos28lpp_base_lvt_c34_ss_nominal_max_0p950v_m25c_sadhm.db`.
- `clk_npu`10ns, `clk_pixel`37.037037ns. setup uncertainty0.1ns / hold0.05ns. I/O max delay1ns/min0ns, input transition0.2ns, output load0.01pF, clock transition0.1ns는 임시 예산이다.
- SRAM에는 면적/clock-to-Q/setup/hold/입력 cap/출력 slew의 실제 macro 모델이 없다. 주소·enable·write-data/read-data 연결 로직이 timing에서 빠지지 않도록 NPU9ns, pixel write36ns의 명시적 max-delay를 부여했다. 약1ns의 macro 여유를 가정한 외부 로직 예산이며 실제 SRAM 포함 timing 증거가 아니다.
- 두 clock group 간 setup/hold는 asynchronous 예외로 제외했다. 같은 domain의stage1→stage2는 정상 timing 대상이다. Bbox20ns는 별도 진단 view에서만 사용했다. 공통 외부 reset은 false path이며 reset release와 보드 reset 조건은 별도 검토 대상이다.
- FPGA BRAM의 단위/주파수/면적을 의미하지 않는다. 실제 ASIC SRAM macro를 연결하면 SRAM 면적 및 경계 timing을 다시 계산해야 한다.
- SRAM read-output register는 memory wrapper 내부 경계에 포함해 제외했다. 후보/kept/bbox bank와 descriptor working registers, MAC/주소 발생/FSM/CDC/scaler/render는 합성 로직으로 유지했다.
- 합성 중 quantized hex 값을 상수로 박아 넣지 않았다. preload SRAM을 외부 black box로 모델링한 실행 엔진의 면적이다.

## 구조·기능 감사

- Netlist SHA256: `a889bbb1abd0889b64f1a388533e936fd2a27abf152b6b50367c8ed6e8a5e32a`.
- 8개 SRAM이 최종 netlist에도 존재하며, empty-body 메모리 내부에 표준셀이 없음을 확인했다. 서로 같은274×32 규격인 bias/multiplier가 reference 하나를 공유해 고유 black-box design은7개다.
- 계층별 netlist를 재귀적으로 펼쳐 표준셀별 수량×동일 Liberty의 cell area를 합산했고 `area.rpt`와 대조했다. 면적 단위는 대응 LEF의 NAND2 cell 크기0.52×0.9µm와 Liberty area0.468의 일치로 확인했다.
- 이번 실행의 `rtl/manifest.json`이 해당 소스 snapshot을 식별한다. `orig/` 백업은 변경하지 않았다. 추출 view에서 NPU FSM/산술과 post의 메모리 이외 로직이 유지됐는지 생성기와 diff로 확인했다.
- 별도 behavioral SRAM 모델을 연결한 추출 view의 behavioral 검증 결과는 `rtl/verify/verification.json`에서 해당 source 기준인지 확인한다. 이번 소스의 검증은 `validation/blackbox_view_verification.json`에 manifest hash와 함께 보존했다. logits3456/3456byte, bbox(29,37)-(48,86),score26158 및 RGB9216pixel이 일치했다. parameter/frame/copy SRAM 전수 검사와 reset/busy/CDC 검사도 통과했다.
- 이 검증은 합성 전 behavioral view 검증이다. 실제 SRAM macro와 표준셀 gate netlist의 기능/등가성 검증을 완료했다는 의미는 아니다.

## 경고와 남은 확인

100MHz 최악 보고 경로: `u_post/current_x2_q_reg_1_` → `u_post/box_sensor_x2_q_reg_85__14_`. 경로 세부 delay는 `timing_clk_npu.rpt`에 보존했다.

QoR 기준 setup 위반0개, hold 위반0개다. 전기적 제약 위반은 `constraints.rpt`와 `qor.rpt`를 확인한다. SRAM 경계 hold는 macro 모델 부재의 영향을 받으므로 실제 macro 모델로 재검토해야 한다.

의도적인 black box 경고 및 상수/미연결 port, feedthrough, 여러 output이 같은 net에 연결되는 경고는 원문에 남겼다. 이를 모두 무시하고 무경고 설계라고 보고하지 않는다. Mapping 전 일부 DW 내부 endpoint 경고는 최종 `check_timing_post.rpt`와 구분한다.

최종 구조 경고는 `check_design_post.rpt`에 보존했다. 미구동·미연결·상수화된 산술 bit 경고는 실제 load와 함께 검토해야 한다. gate-level 등가성 검증 없이 경고를 해결됐다고 표시하지 않는다.

SRAM 입력 cap을 제공하지 않아 SRAM 연결 net103개에서 min-capacitance 위반이 보고됐다. `constraints.rpt`의0.00 표시는 기본 출력 정밀도로 반올림된 값이다. Setup/hold 위반이0이어도 이 경고를 포함한 전체 ASIC signoff 완료로 해석하지 않는다. 실제 SRAM Liberty/입력 부하와 배치·배선 조건을 적용해 재검증해야 한다.

최종 check_timing 원문:

```text
Information: Checking generated_clocks...

Information: Checking loops...

Information: Checking no_input_delay...

Information: Checking unconstrained_endpoints...

Information: Checking pulse_clock_cell_type...

Information: Checking no_driving_cell...

Information: Checking partial_input_delay...
1
```

## 파일과 재실행

최종 실행 폴더: `results/blackbox_dualclock_20261106_131129/`.

- [area.rpt](results/blackbox_dualclock_20261106_131129/area.rpt)
- [timing_setup.rpt](results/blackbox_dualclock_20261106_131129/timing_setup.rpt)
- [timing_hold.rpt](results/blackbox_dualclock_20261106_131129/timing_hold.rpt)
- [constraints.rpt](results/blackbox_dualclock_20261106_131129/constraints.rpt)
- [qor.rpt](results/blackbox_dualclock_20261106_131129/qor.rpt)
- [netlist](results/blackbox_dualclock_20261106_131129/person_detector_top_syn.v), [DDC](results/blackbox_dualclock_20261106_131129/person_detector_top.ddc), [SDC](results/blackbox_dualclock_20261106_131129/person_detector_top.sdc)
- [독립 감사 JSON](results/blackbox_dualclock_20261106_131129/audited_results.json), [실행 조건](results/blackbox_dualclock_20261106_131129/run_configuration.json), [console](results/blackbox_dualclock_20261106_131129/console.log)
- 추출/검증 계약: [BLACKBOX_CONTRACT.md](BLACKBOX_CONTRACT.md)

```bash
python3 design_compiler/scripts/prepare_blackbox_view.py
bash design_compiler/scripts/synth_blackbox.sh
python3 design_compiler/scripts/audit_blackbox_result.py
python3 design_compiler/scripts/write_blackbox_report.py
```

각 실행은 새로운 timestamp 폴더에 source/Tcl snapshot을 보존한다. 현재 경로는 `latest_blackbox_run.txt`다. 일반 `compile` 시도는2400초에 timeout되어 최종 리포트를 얻지 못했다. DC2020은 기존 라이선스 daemon과 호환되지 않아 실행하지 못했고 서버 설정은 변경하지 않았다. 초기 결과는 DC2013의 `compile -only_design_rule`였고 RTL 분할 후 일반 `compile`도 완료했다. 이번 실행은 asynchronous clock group을 적용한 최신 일반 `compile`이다. 최초 진단 시도의 CDC pin 선택 오류는 수정된 재실행으로 대체했으며 `NOT_FINAL.txt` 표시가 있는 결과를 최종 timing 근거로 쓰지 않는다.

실행 상태 원문:

```text
Run started: 2026-11-06T13:11:29+09:00
Run ended: 2026-11-06T13:18:21+09:00
Exit code: 0
```
