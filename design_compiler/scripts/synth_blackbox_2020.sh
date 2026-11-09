#!/usr/bin/env bash
set -euo pipefail
dc_bb_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NPU_DC_ROOT=$(cd -- "$dc_bb_script_dir/.." && pwd)
export SNPSLMD_LICENSE_FILE=27000@uxftools
export LM_LICENSE_FILE=27000@uxftools
dc_bb_stamp=$(date +%Y%m%d_%H%M%S)
export NPU_DC_OUT="$NPU_DC_ROOT/results/blackbox_dualclock_2020_$dc_bb_stamp"
mkdir -p "$NPU_DC_OUT/rtl"
cp -a "$NPU_DC_ROOT/blackbox_rtl/." "$NPU_DC_OUT/rtl/"
cp "$dc_bb_script_dir/run_blackbox_2020.tcl" "$NPU_DC_OUT/run_synthesis.tcl"
printf '%s\n' "$NPU_DC_OUT" > "$NPU_DC_ROOT/latest_blackbox_run.txt"
cd -- "$NPU_DC_OUT"
printf 'Run started: %s\n' "$(date -Is)" > run_status.txt
set +e
(
    ulimit -v 33554432
    /usr/bin/time -v timeout --signal=TERM --kill-after=30s 2400s \
        /tools/synopsys_221202/syn/R-2020.09-SP5/bin/dc_shell \
        -f "$NPU_DC_OUT/run_synthesis.tcl"
) > console.log 2>&1
dc_bb_status=$?
printf 'Run ended: %s\nExit code: %s\n' "$(date -Is)" "$dc_bb_status" >> run_status.txt
tail -n 45 console.log
exit "$dc_bb_status"
