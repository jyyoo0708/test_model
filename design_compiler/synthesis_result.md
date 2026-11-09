# 원본 단일27MHz RTL의 Design Compiler 실행 결과

최신 두 클록 SRAM 블랙박스 합성은 별도로 완료했다. [최신 area/timing 결과](blackbox_synthesis_result.md)를 참고한다. 아래는 원본 실행 기록이다.

## 판정

**전체 합성 미완료: RTL analyze는 통과했으나 NPU elaboration이1800초 실행 제한 안에 완료되지 않아 종료 코드124로 종료됐다.** 따라서 매핑된 전체 netlist/DDC, area/QoR/setup/hold 리포트가 생성되지 않았고, 합성 또는27MHz 타이밍 PASS로 판단할 수 없다. 문법 오류로 실패했다고 단정하지도 않는다.

| 단계 | 결과 | 근거 |
|---|---|---|
| 라이선스 연결 | PASS | `27000@uxftools` 서버와 `snpslmd` UP, 실제 DC 초기화·analyze 수행 |
| 원본4개 RTL analyze | PASS | console의 `Presto compilation completed successfully` |
| top/capture elaboration | 일부 진행 | top 및 capture 처리 후 `Building the design 'npu_core'`까지 도달 |
| NPU 및 전체 elaboration | TIMEOUT | 이후 NPU elaboration 완료 없이1800초에 종료 |
| 전체 link/compile | NOT_RUN | 앞 단계가 반환하지 않아 도달하지 않음 |
| 최종 area/timing/QoR | NOT_GENERATED | mapped design이 생성되지 않음 |
| FPGA/Vivado 합성 | NOT_RUN | 이번 실행은 Synopsys ASIC 표준셀 합성 시도 |

## 실행 환경과 정확한 대상

- 도구: `/tools/synopsys/syn/I-2013.12-SP4/amd64/syn/bin/dc_shell`, **I-2013.12-SP4 RHEL64**.
- 실행 시각: **2026-11-05 17:57:46 ~18:27:46 KST**, wall time30:00.09.
- top: `person_detector_top`; 원본 단일 `clk`, sensor96×96, gray8 기본 parameter.
- 원본 snapshot: `rtl_snapshot/`의4개 `.v`. 파일별 SHA256은 `source_manifest.json`에 있다.
- 표준셀 DB: `/tools/lib/ln28lpp/primitive/ln28lpp_sc_9t_base_lvt_c130_FE-Common_sec190802_0202/LIBERTY/sc9_cmos28lpp_base_lvt_c34_ss_nominal_max_0p950v_m25c_sadhm.db`.
- Samsung28nm LPP,9-track,LVT, SS0.950V,-25°C 코너 DB를 기존 설치 합성 예제와 동일하게 선택했다. 별도 SRAM macro DB는 제공/연결하지 않았다.
- 합성 Tcl의 요청 주기37.037037ns(27MHz), setup uncertainty0.1ns/hold0.05ns, 임시 I/O max delay1ns/min0ns, 입력 transition0.2ns, 출력 load0.01(library 단위), reset false path 조건을 준비했다. **elaboration 이후에 설정하는 명령이므로 이번 실행에서는 이 제약 적용 단계에도 도달하지 않았다.**
- 원본 전체 behavioral RAM을 포함했으며 메모리를 black box로 빼거나 작은 크기로 줄이지 않았다.
- 주소공간 상한32GiB, 실행 제한1800초. 실제 최대RSS1,816,244KiB(약1.73GiB)로 memory limit에 걸린 기록은 없다. userCPU1794.20초,CPU약99%다.

## 라이선스 사용 방식

`사용법.txt`를 읽고 해당 문서의 `27000@uxftools`를 이 프로세스의 `SNPSLMD_LICENSE_FILE`, `LM_LICENSE_FILE`에 설정했다. 기존 라이선스 서버가 UP이므로 `lmgrd`를 새로 시작하지 않았고 공유 서버를 종료하는 `lmdown`도 실행하지 않았다. 샌드박스 내부의 첫 `lmstat`은 네트워크 권한 오류였으며 승인된 샌드박스 밖 실행으로 정상 접속했다. 실제 합성도 같은 허용 환경에서 실행했다. DC 종료로 해당 실행의 checkout이 반환됐다.

## 관찰된 경고와 실패 원인 해석

capture의 signed/unsigned 변환 및 도달 불가능 default branch, NPU의 signed/unsigned 대입·part-select 경고가 기록됐다. 확인된 최종 종료 원인은 **외부 timeout**이다. 라이선스 접속 실패, 메모리 부족 또는 RTL syntax error가 최종 원인이라는 근거는 없다.

원본의 큰 SRAM 배열은 합계1,688,928bits(211,116bytes)다. `ram_style="block"`은 FPGA용 속성이며 DC와 표준셀 DB만으로 Vivado BRAM을 생성하지 않는다. 실제 ASIC SRAM wrapper/DB가 없으면 큰 array를 표준셀로 처리하는 부담이 생긴다. 이 점은 장시간 elaboration의 유력한 구조적 요인이지만, 이번 로그만으로 내부 알고리즘의 정확한 병목을 확정하지는 않는다.

전체 ASIC PPA를 얻으려면 타깃 공정의 SRAM macro와 wrapper를 연결해 재합성하는 방법을 우선 검토한다. 메모리를 black box로 제외한 논리 합성은 별도 진단으로 가능하나 메모리 면적·타이밍이 빠진 값이며 전체 칩 PPA라고 표시하면 안 된다. 실행 시간 상한을 늘리는 방식도 가능하지만 완주 시간과 자원 사용은 확인되지 않았다.

## 생성 파일

| 파일/경로 | 내용 |
|---|---|
| `results/full_27mhz/console.log` | 실제 DC 전체 console 및 timeout/resource 사용 기록 |
| `results/full_27mhz/run_status.txt` | 시작·종료 시각과 exit124 |
| `results/full_27mhz/command.log`, `filenames.log` | DC 명령·파일 기록 |
| `results/full_27mhz/person_detector_top.svf` | 중단 전 생성된 변환 정보. 완료된 합성/등가성 증거가 아님 |
| `results/full_27mhz/work/` | analyze 중간 파일 |
| `source_manifest.json`, `rtl_snapshot/` | 실제 실행한 원본과 hash |
| `memory_inventory.json` | 원본 SRAM 용량과 macro 미연결 상태 |
| `scripts/synth_27mhz.sh`, `scripts/run_27mhz.tcl` | 동일 snapshot 대상 재실행 절차 |

재실행은 프로젝트 루트에서 `bash design_compiler/scripts/synth_27mhz.sh`다. 같은 결과 경로를 갱신하므로 필요하면 기존 리포트를 별도 보관하고 실행한다.

## 이후 두 클록 변경과의 구분

이 보고서는 **원본 단일27MHz snapshot에만** 해당한다. 프로젝트 전체 원본은 `../orig/`에 별도 백업했다. 이후 작업 중인 `clk_pixel`27MHz/`clk_npu`100MHz CDC RTL의 합성 성공이나100MHz timing을 이 기록으로 주장하지 않는다. 두 클록 변경은 별도의 기능 시뮬레이션·정수 출력 비교·이미지 검증으로 평가한다.
