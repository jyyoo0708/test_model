#!/usr/bin/env bash
set -euo pipefail
dc_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NPU_DC_ROOT=$(cd -- "$dc_script_dir/.." && pwd)
export SNPSLMD_LICENSE_FILE=27000@uxftools
export LM_LICENSE_FILE=27000@uxftools
export NPU_DC_OUT="$NPU_DC_ROOT/results/full_27mhz"
mkdir -p "$NPU_DC_OUT"
cd -- "$NPU_DC_OUT"
printf 'Run started: %s\n' "$(date -Is)" > run_status.txt
set +e
# Bound this all-standard-cell attempt; memory macros are not supplied.
# The address space bound protects the shared machine during array expansion.
(
    ulimit -v 33554432
    /usr/bin/time -v timeout --signal=TERM --kill-after=30s 1800s \
        /tools/synopsys/syn/I-2013.12-SP4/amd64/syn/bin/dc_shell \
        -f "$dc_script_dir/run_27mhz.tcl"
) > console.log 2>&1
dc_run_status=$?
printf 'Run ended: %s\nExit code: %s\n' "$(date -Is)" "$dc_run_status" >> run_status.txt
tail -n 35 console.log
exit "$dc_run_status"
